//! # tool_executor
//!
//! ## 职责
//! 批量执行 LLM 请求的工具调用，支持 Shared/Exclusive 并发调度，
//! 集成 Hook + Permission 权限管线，实时发射 `AgentEvent`。
//!
//! ## 设计
//! - **Partition + JoinSet** — 按 `ConcurrencyMode` 分批，Shared 并发、Exclusive 串行
//! - **权限管线** — Hook(PreToolUse) → Tool::check_permissions → Ruleset → 决策
//! - **取消传播** — 共享 `CancellationToken`，取消后未执行的工具产出 "cancelled" 假结果
//! - **结果保序** — 工具结果按原始 tool call 顺序返回
//!
//! ## 调用者
//! - `katu-agent::runner` (future) — Agent loop 核心循环

use std::sync::Arc;

use chrono::Utc;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use katu_core::agent_event::AgentEvent;
use katu_core::hook::{AggregatedHookOutput, HookInput, HookRegistry};
use katu_core::message::{AssistantBlock, AssistantMessage, ContentBlock, ToolResultMessage};
use katu_core::permission::{PermissionResult, Ruleset};
use katu_core::tool::{ConcurrencyMode, ToolCallContext, ToolOutput};
use katu_core::types::{MessageId, ToolCallId};
use katu_core::{CancellationToken, Tool};

// ===========================================================================
// Public Types
// ===========================================================================

/// 工具执行器配置。
#[derive(Debug, Clone)]
pub struct ToolExecutorConfig {
    /// 最大并发执行数（Shared batch 内的并行上限）。
    pub max_concurrency: usize,
    /// 传入 `ToolCallContext.extra` 的额外上下文。
    pub extra: serde_json::Value,
}

impl Default for ToolExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 10,
            extra: serde_json::Value::Object(serde_json::Map::new()),
        }
    }
}

/// 工具批次执行结果。
#[derive(Debug)]
pub struct ToolBatchResult {
    /// 所有工具的执行结果（按原始 tool call 顺序）。
    pub tool_results: Vec<ToolResultMessage>,
    /// 是否被中断（取消或 Hook 阻止继续）。
    pub interrupted: bool,
}

/// 工具执行错误。
#[derive(Debug, thiserror::Error)]
pub enum ToolExecutorError {
    /// 事件 channel 已关闭。
    #[error("event channel closed")]
    ChannelClosed,
}

/// 模块级 Result 别名。
pub type Result<T, E = ToolExecutorError> = std::result::Result<T, E>;

// ===========================================================================
// ToolExecutor
// ===========================================================================

/// 工具执行器 — 批量执行 assistant message 中的工具调用。
pub struct ToolExecutor;

impl ToolExecutor {
    /// 执行 assistant message 中的所有工具调用。
    ///
    /// # 流程
    /// 1. 提取所有 `ToolCall` blocks
    /// 2. 按 `ConcurrencyMode` 分批 (Partition)
    /// 3. Shared batch → 并发执行（受 `max_concurrency` 限制）
    /// 4. Exclusive batch → 串行执行
    /// 5. 每个工具经过: Hook(PreToolUse) → validate → execute → Hook(PostToolUse)
    pub async fn execute_batch(
        assistant_message: &AssistantMessage,
        tools: &[Arc<dyn Tool>],
        hooks: &HookRegistry,
        ruleset: &Ruleset,
        event_tx: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        config: &ToolExecutorConfig,
    ) -> Result<ToolBatchResult> {
        let tool_calls = Self::extract_tool_calls(assistant_message);
        if tool_calls.is_empty() {
            return Ok(ToolBatchResult {
                tool_results: Vec::new(),
                interrupted: false,
            });
        }

        let batches = Self::partition_by_concurrency(&tool_calls, tools);
        let mut results: Vec<Option<ToolResultMessage>> = vec![None; tool_calls.len()];
        let mut interrupted = false;

        for batch in &batches {
            if cancel.is_cancelled() {
                interrupted = true;
                break;
            }

            if batch.is_concurrent {
                Self::execute_concurrent_batch(
                    &batch.entries, tools, hooks, ruleset,
                    event_tx, cancel, config, &mut results,
                ).await?;
            } else {
                for entry in &batch.entries {
                    if cancel.is_cancelled() {
                        interrupted = true;
                        break;
                    }
                    let result = Self::execute_single_tool(
                        entry, tools, hooks, ruleset, event_tx, cancel, config,
                    ).await?;
                    results[entry.original_index] = Some(result);
                }
            }
        }

        // 未执行的工具产出 "cancelled" 假结果
        for (i, slot) in results.iter_mut().enumerate() {
            if slot.is_none() {
                let tc = &tool_calls[i];
                Self::send(event_tx, AgentEvent::ToolFailed {
                    call_id: tc.call_id.clone(),
                    tool_name: tc.name.clone(),
                    error: "Tool execution was cancelled".into(),
                    is_retryable: false,
                })?;
                *slot = Some(Self::make_cancelled_result(&tc.call_id, &tc.name));
                interrupted = true;
            }
        }

        Ok(ToolBatchResult {
            tool_results: results.into_iter().flatten().collect(),
            interrupted,
        })
    }
}

// ===========================================================================
// Internal Types
// ===========================================================================

struct ExtractedToolCall {
    call_id: ToolCallId,
    name: String,
    arguments: serde_json::Value,
}

struct BatchEntry {
    original_index: usize,
    call_id: ToolCallId,
    name: String,
    arguments: serde_json::Value,
}

struct ToolBatch {
    is_concurrent: bool,
    entries: Vec<BatchEntry>,
}

enum PreToolResult {
    Denied { message: String },
    Proceed { effective_args: serde_json::Value },
}

// ===========================================================================
// Private Implementation
// ===========================================================================

impl ToolExecutor {
    // ─── Extraction & Partitioning ───────────────────────────────────────────

    fn extract_tool_calls(msg: &AssistantMessage) -> Vec<ExtractedToolCall> {
        msg.content
            .iter()
            .filter_map(|block| match block {
                AssistantBlock::ToolCall { id, name, arguments } => Some(ExtractedToolCall {
                    call_id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                }),
                _ => None,
            })
            .collect()
    }

    fn partition_by_concurrency(
        tool_calls: &[ExtractedToolCall],
        tools: &[Arc<dyn Tool>],
    ) -> Vec<ToolBatch> {
        let mut batches: Vec<ToolBatch> = Vec::new();

        for (i, tc) in tool_calls.iter().enumerate() {
            let mode = Self::find_tool(tools, &tc.name)
                .map(|t| t.concurrency_mode())
                .unwrap_or(ConcurrencyMode::Shared);

            let entry = BatchEntry {
                original_index: i,
                call_id: tc.call_id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
            };

            let is_concurrent = mode == ConcurrencyMode::Shared;

            if let Some(last) = batches.last_mut() {
                if last.is_concurrent && is_concurrent {
                    last.entries.push(entry);
                    continue;
                }
            }

            batches.push(ToolBatch {
                is_concurrent,
                entries: vec![entry],
            });
        }

        batches
    }

    fn find_tool<'a>(tools: &'a [Arc<dyn Tool>], name: &str) -> Option<&'a Arc<dyn Tool>> {
        tools.iter().find(|t| t.definition().name == name)
    }

    // ─── Batch Execution ─────────────────────────────────────────────────────

    async fn execute_concurrent_batch(
        entries: &[BatchEntry],
        tools: &[Arc<dyn Tool>],
        hooks: &HookRegistry,
        ruleset: &Ruleset,
        event_tx: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        config: &ToolExecutorConfig,
        results: &mut [Option<ToolResultMessage>],
    ) -> Result<()> {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(config.max_concurrency));
        let mut handles = Vec::with_capacity(entries.len());

        for entry in entries {
            let sem = semaphore.clone();
            let call_id = entry.call_id.clone();
            let name = entry.name.clone();
            let arguments = entry.arguments.clone();
            let original_index = entry.original_index;
            let tool = Self::find_tool(tools, &name).cloned();
            let cancel = cancel.clone();
            let extra = config.extra.clone();
            let event_tx = event_tx.clone();

            // Hook + permission 检查在 spawn 前执行（HookRegistry 不是 Send）
            let pre_result = Self::run_pre_tool_pipeline(
                &call_id, &name, &arguments, tool.as_ref(), hooks, ruleset,
            ).await;

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.expect("semaphore closed");

                if cancel.is_cancelled() {
                    return (original_index, ToolExecutor::make_cancelled_result(&call_id, &name));
                }

                let result = match pre_result {
                    PreToolResult::Denied { message } => {
                        let _ = event_tx.send(AgentEvent::ToolFailed {
                            call_id: call_id.clone(),
                            tool_name: name.clone(),
                            error: message.clone(),
                            is_retryable: false,
                        });
                        ToolExecutor::make_denied_result(&call_id, &name, &message)
                    }
                    PreToolResult::Proceed { effective_args } => {
                        ToolExecutor::execute_tool_inner(
                            &call_id, &name, effective_args,
                            tool.as_ref(), &cancel, &extra, &event_tx,
                        ).await
                    }
                };

                (original_index, result)
            });

            handles.push(handle);
        }

        for handle in handles {
            match handle.await {
                Ok((idx, result)) => {
                    results[idx] = Some(result);
                }
                Err(e) => {
                    warn!("tool_executor: task panicked: {e}");
                }
            }
        }

        Ok(())
    }

    async fn execute_single_tool(
        entry: &BatchEntry,
        tools: &[Arc<dyn Tool>],
        hooks: &HookRegistry,
        ruleset: &Ruleset,
        event_tx: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        config: &ToolExecutorConfig,
    ) -> Result<ToolResultMessage> {
        let tool = Self::find_tool(tools, &entry.name).cloned();

        let pre_result = Self::run_pre_tool_pipeline(
            &entry.call_id, &entry.name, &entry.arguments,
            tool.as_ref(), hooks, ruleset,
        ).await;

        let result = match pre_result {
            PreToolResult::Denied { message } => {
                Self::send(event_tx, AgentEvent::ToolFailed {
                    call_id: entry.call_id.clone(),
                    tool_name: entry.name.clone(),
                    error: message.clone(),
                    is_retryable: false,
                })?;
                Self::make_denied_result(&entry.call_id, &entry.name, &message)
            }
            PreToolResult::Proceed { effective_args } => {
                Self::execute_tool_inner(
                    &entry.call_id, &entry.name, effective_args,
                    tool.as_ref(), cancel, &config.extra, event_tx,
                ).await
            }
        };

        Ok(result)
    }

    // ─── Pre-Tool Pipeline ───────────────────────────────────────────────────

    async fn run_pre_tool_pipeline(
        call_id: &ToolCallId,
        tool_name: &str,
        arguments: &serde_json::Value,
        tool: Option<&Arc<dyn Tool>>,
        hooks: &HookRegistry,
        ruleset: &Ruleset,
    ) -> PreToolResult {
        let mut effective_args = arguments.clone();

        // 1. Hook(PreToolUse)
        let hook_input = HookInput::PreToolUse {
            tool_name: tool_name.to_string(),
            tool_input: arguments.clone(),
            call_id: call_id.clone(),
        };

        let matching = hooks.matching(&hook_input);
        if !matching.is_empty() {
            let mut aggregated = AggregatedHookOutput::default();
            for registered in &matching {
                let output = registered.hook.on_event(&hook_input).await;
                aggregated.merge(output, registered.hook.name());
            }

            if aggregated.is_denied() {
                let reason = aggregated
                    .blocking_errors.first().cloned()
                    .unwrap_or_else(|| "Denied by hook".into());
                return PreToolResult::Denied { message: reason };
            }

            if let Some(updated) = aggregated.updated_input {
                effective_args = updated;
            }
        }

        // 2. Tool::check_permissions()
        if let Some(tool) = tool {
            let ctx = ToolCallContext::new(call_id.clone());
            let perm = tool.check_permissions(&effective_args, &ctx);
            match perm {
                PermissionResult::Deny { message } => {
                    return PreToolResult::Denied { message };
                }
                PermissionResult::Ask { message } => {
                    debug!("tool_executor: tool {} requests user confirmation: {}", tool_name, message);
                    return PreToolResult::Denied {
                        message: format!("User confirmation required: {message}"),
                    };
                }
                PermissionResult::Allow | PermissionResult::Passthrough => {}
            }
        }

        // 3. Ruleset 规则引擎
        let permission_key = tool
            .map(|t| t.permission_key().to_string())
            .unwrap_or_else(|| tool_name.to_string());
        let content = effective_args.to_string();

        if let Some(behavior) = ruleset.evaluate(&permission_key, &content) {
            if behavior.is_deny() {
                return PreToolResult::Denied {
                    message: format!("Denied by permission rule for '{permission_key}'"),
                };
            }
        }

        PreToolResult::Proceed { effective_args }
    }

    // ─── Tool Execution Core ─────────────────────────────────────────────────

    async fn execute_tool_inner(
        call_id: &ToolCallId,
        tool_name: &str,
        arguments: serde_json::Value,
        tool: Option<&Arc<dyn Tool>>,
        cancel: &CancellationToken,
        extra: &serde_json::Value,
        event_tx: &mpsc::UnboundedSender<AgentEvent>,
    ) -> ToolResultMessage {
        let Some(tool) = tool else {
            let error_msg = format!("Tool not found: {tool_name}");
            warn!("tool_executor: {error_msg}");
            let _ = event_tx.send(AgentEvent::ToolFailed {
                call_id: call_id.clone(),
                tool_name: tool_name.to_string(),
                error: error_msg.clone(),
                is_retryable: false,
            });
            return Self::make_error_result(call_id, tool_name, &error_msg);
        };

        let _ = event_tx.send(AgentEvent::ToolCalled {
            call_id: call_id.clone(),
            tool_name: tool_name.to_string(),
            arguments: arguments.clone(),
        });

        let ctx = ToolCallContext::new(call_id.clone())
            .with_cancellation(cancel.clone())
            .with_extra(extra.clone());

        // Validate
        if let Err(e) = tool.validate(&arguments, &ctx).await {
            let error_msg = format!("Validation failed: {e}");
            let _ = event_tx.send(AgentEvent::ToolFailed {
                call_id: call_id.clone(),
                tool_name: tool_name.to_string(),
                error: error_msg.clone(),
                is_retryable: false,
            });
            return Self::make_error_result(call_id, tool_name, &error_msg);
        }

        // Execute
        match tool.execute(arguments, &ctx).await {
            Ok(output) => {
                if output.is_error {
                    let _ = event_tx.send(AgentEvent::ToolFailed {
                        call_id: call_id.clone(),
                        tool_name: tool_name.to_string(),
                        error: output.content.clone(),
                        is_retryable: false,
                    });
                } else {
                    let _ = event_tx.send(AgentEvent::ToolSucceeded {
                        call_id: call_id.clone(),
                        tool_name: tool_name.to_string(),
                        output: output.clone(),
                    });
                }
                Self::make_result(call_id, tool_name, &output)
            }
            Err(e) => {
                let error_msg = e.to_string();
                let is_retryable = e.retryable();
                let _ = event_tx.send(AgentEvent::ToolFailed {
                    call_id: call_id.clone(),
                    tool_name: tool_name.to_string(),
                    error: error_msg.clone(),
                    is_retryable,
                });
                Self::make_error_result(call_id, tool_name, &error_msg)
            }
        }
    }

    // ─── Result Constructors ─────────────────────────────────────────────────

    fn make_result(call_id: &ToolCallId, tool_name: &str, output: &ToolOutput) -> ToolResultMessage {
        ToolResultMessage {
            id: MessageId::new(),
            tool_call_id: call_id.clone(),
            tool_name: tool_name.to_string(),
            content: vec![ContentBlock::Text { text: output.content.clone() }],
            is_error: output.is_error,
            timestamp: Utc::now(),
        }
    }

    fn make_error_result(call_id: &ToolCallId, tool_name: &str, error: &str) -> ToolResultMessage {
        ToolResultMessage {
            id: MessageId::new(),
            tool_call_id: call_id.clone(),
            tool_name: tool_name.to_string(),
            content: vec![ContentBlock::Text { text: error.to_string() }],
            is_error: true,
            timestamp: Utc::now(),
        }
    }

    fn make_cancelled_result(call_id: &ToolCallId, tool_name: &str) -> ToolResultMessage {
        Self::make_error_result(call_id, tool_name, "Tool execution was cancelled")
    }

    fn make_denied_result(call_id: &ToolCallId, tool_name: &str, message: &str) -> ToolResultMessage {
        Self::make_error_result(call_id, tool_name, &format!("Permission denied: {message}"))
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    fn send(
        tx: &mpsc::UnboundedSender<AgentEvent>,
        event: AgentEvent,
    ) -> Result<()> {
        tx.send(event).map_err(|_| ToolExecutorError::ChannelClosed)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use katu_core::hook::{Hook, HookEvent, HookOutput, HookSource};
    use katu_core::permission::PermissionRule;
    use katu_core::tool::ToolDefinition;
    use katu_core::types::FinishReason;

    // ── 测试工具 ──

    struct EchoTool;
    static ECHO_DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
        ToolDefinition::new("echo", "Echo input", serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }))
    });

    #[async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> &ToolDefinition { &ECHO_DEF }
        async fn execute(&self, args: serde_json::Value, _ctx: &ToolCallContext) -> katu_core::Result<ToolOutput> {
            Ok(ToolOutput::success(args["text"].as_str().unwrap_or("")))
        }
    }

    struct FailTool;
    static FAIL_DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
        ToolDefinition::no_params("fail", "Always fails")
    });

    #[async_trait]
    impl Tool for FailTool {
        fn definition(&self) -> &ToolDefinition { &FAIL_DEF }
        async fn execute(&self, _args: serde_json::Value, _ctx: &ToolCallContext) -> katu_core::Result<ToolOutput> {
            Ok(ToolOutput::error("something went wrong"))
        }
    }

    struct ExclusiveWriteTool;
    static WRITE_DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
        ToolDefinition::new("write_file", "Write file", serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" }, "content": { "type": "string" } },
            "required": ["path", "content"]
        }))
    });

    #[async_trait]
    impl Tool for ExclusiveWriteTool {
        fn definition(&self) -> &ToolDefinition { &WRITE_DEF }
        async fn execute(&self, args: serde_json::Value, _ctx: &ToolCallContext) -> katu_core::Result<ToolOutput> {
            Ok(ToolOutput::success(format!("Written to {}", args["path"].as_str().unwrap_or("?"))))
        }
        fn concurrency_mode(&self) -> ConcurrencyMode { ConcurrencyMode::Exclusive }
    }

    // ── 测试 Hook ──

    struct DenyHook;

    #[async_trait]
    impl Hook for DenyHook {
        fn name(&self) -> &str { "deny_hook" }
        fn events(&self) -> &[HookEvent] { &[HookEvent::PreToolUse] }
        async fn on_event(&self, _input: &katu_core::hook::HookInput) -> HookOutput {
            HookOutput::deny("blocked by test hook")
        }
    }

    // ── 辅助函数 ──

    fn make_assistant_msg(blocks: Vec<AssistantBlock>) -> AssistantMessage {
        AssistantMessage {
            id: MessageId::new(), content: blocks,
            model: "test".into(), provider: "test".into(),
            finish_reason: FinishReason::ToolCalls, usage: None, timestamp: Utc::now(),
        }
    }

    fn tc(id: &str, name: &str, args: serde_json::Value) -> AssistantBlock {
        AssistantBlock::ToolCall { id: ToolCallId::new(id), name: name.into(), arguments: args }
    }

    // ── 测试用例 ──

    #[tokio::test]
    async fn test_single_tool_execution() {
        let msg = make_assistant_msg(vec![tc("c1", "echo", serde_json::json!({"text": "hello"}))]);
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool)];
        let (tx, mut rx) = mpsc::unbounded_channel();

        let result = ToolExecutor::execute_batch(
            &msg, &tools, &HookRegistry::new(), &Ruleset::new(), &tx, &CancellationToken::new(), &ToolExecutorConfig::default(),
        ).await.unwrap();

        assert_eq!(result.tool_results.len(), 1);
        assert!(!result.interrupted);
        assert!(!result.tool_results[0].is_error);

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() { events.push(e); }
        assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolCalled { tool_name, .. } if tool_name == "echo")));
        assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolSucceeded { .. })));
    }

    #[tokio::test]
    async fn test_tool_not_found() {
        let msg = make_assistant_msg(vec![tc("c1", "nonexistent", serde_json::json!({}))]);
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool)];
        let (tx, _rx) = mpsc::unbounded_channel();

        let result = ToolExecutor::execute_batch(
            &msg, &tools, &HookRegistry::new(), &Ruleset::new(), &tx, &CancellationToken::new(), &ToolExecutorConfig::default(),
        ).await.unwrap();

        assert_eq!(result.tool_results.len(), 1);
        assert!(result.tool_results[0].is_error);
    }

    #[tokio::test]
    async fn test_tool_error_result() {
        let msg = make_assistant_msg(vec![tc("c1", "fail", serde_json::json!({}))]);
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(FailTool)];
        let (tx, _rx) = mpsc::unbounded_channel();

        let result = ToolExecutor::execute_batch(
            &msg, &tools, &HookRegistry::new(), &Ruleset::new(), &tx, &CancellationToken::new(), &ToolExecutorConfig::default(),
        ).await.unwrap();

        assert!(result.tool_results[0].is_error);
    }

    #[tokio::test]
    async fn test_multiple_shared_concurrent() {
        let msg = make_assistant_msg(vec![
            tc("c1", "echo", serde_json::json!({"text": "a"})),
            tc("c2", "echo", serde_json::json!({"text": "b"})),
            tc("c3", "echo", serde_json::json!({"text": "c"})),
        ]);
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool)];
        let (tx, _rx) = mpsc::unbounded_channel();

        let result = ToolExecutor::execute_batch(
            &msg, &tools, &HookRegistry::new(), &Ruleset::new(), &tx, &CancellationToken::new(), &ToolExecutorConfig::default(),
        ).await.unwrap();

        assert_eq!(result.tool_results.len(), 3);
        assert!(!result.interrupted);
        assert_eq!(result.tool_results[0].tool_call_id, ToolCallId::new("c1"));
        assert_eq!(result.tool_results[1].tool_call_id, ToolCallId::new("c2"));
        assert_eq!(result.tool_results[2].tool_call_id, ToolCallId::new("c3"));
    }

    #[tokio::test]
    async fn test_exclusive_serial() {
        let msg = make_assistant_msg(vec![
            tc("c1", "echo", serde_json::json!({"text": "a"})),
            tc("c2", "write_file", serde_json::json!({"path": "a.rs", "content": "fn main(){}"})),
            tc("c3", "echo", serde_json::json!({"text": "b"})),
        ]);
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool), Arc::new(ExclusiveWriteTool)];
        let (tx, _rx) = mpsc::unbounded_channel();

        let result = ToolExecutor::execute_batch(
            &msg, &tools, &HookRegistry::new(), &Ruleset::new(), &tx, &CancellationToken::new(), &ToolExecutorConfig::default(),
        ).await.unwrap();

        assert_eq!(result.tool_results.len(), 3);
        assert!(!result.interrupted);
        assert_eq!(result.tool_results[0].tool_call_id, ToolCallId::new("c1"));
        assert_eq!(result.tool_results[1].tool_call_id, ToolCallId::new("c2"));
        assert_eq!(result.tool_results[2].tool_call_id, ToolCallId::new("c3"));
    }

    #[tokio::test]
    async fn test_cancellation_skips_remaining() {
        let msg = make_assistant_msg(vec![
            tc("c1", "echo", serde_json::json!({"text": "a"})),
            tc("c2", "echo", serde_json::json!({"text": "b"})),
        ]);
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool)];
        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = ToolExecutor::execute_batch(
            &msg, &tools, &HookRegistry::new(), &Ruleset::new(), &tx, &cancel, &ToolExecutorConfig::default(),
        ).await.unwrap();

        assert!(result.interrupted);
        assert!(result.tool_results.iter().all(|r| r.is_error));
    }

    #[tokio::test]
    async fn test_permission_deny_by_ruleset() {
        let msg = make_assistant_msg(vec![tc("c1", "echo", serde_json::json!({"text": "hello"}))]);
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool)];
        let mut ruleset = Ruleset::new();
        ruleset.add(PermissionRule::deny(katu_core::permission::RuleSource::Policy, "echo", "*"));
        let (tx, _rx) = mpsc::unbounded_channel();

        let result = ToolExecutor::execute_batch(
            &msg, &tools, &HookRegistry::new(), &ruleset, &tx, &CancellationToken::new(), &ToolExecutorConfig::default(),
        ).await.unwrap();

        assert!(result.tool_results[0].is_error);
    }

    #[tokio::test]
    async fn test_hook_deny() {
        let msg = make_assistant_msg(vec![tc("c1", "echo", serde_json::json!({"text": "hello"}))]);
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool)];
        let mut hooks = HookRegistry::new();
        hooks.register(Arc::new(DenyHook), HookSource::Programmatic, 0);
        let (tx, _rx) = mpsc::unbounded_channel();

        let result = ToolExecutor::execute_batch(
            &msg, &tools, &hooks, &Ruleset::new(), &tx, &CancellationToken::new(), &ToolExecutorConfig::default(),
        ).await.unwrap();

        assert!(result.tool_results[0].is_error);
    }

    #[tokio::test]
    async fn test_empty_tool_calls() {
        let msg = make_assistant_msg(vec![AssistantBlock::Text { text: "No tools".into() }]);
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool)];
        let (tx, _rx) = mpsc::unbounded_channel();

        let result = ToolExecutor::execute_batch(
            &msg, &tools, &HookRegistry::new(), &Ruleset::new(), &tx, &CancellationToken::new(), &ToolExecutorConfig::default(),
        ).await.unwrap();

        assert!(result.tool_results.is_empty());
        assert!(!result.interrupted);
    }
}
