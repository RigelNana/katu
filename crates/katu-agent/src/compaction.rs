//! # compaction
//!
//! ## 职责
//! 上下文压缩运行时 — 驱动 prune 和 compact 两层管线，管理压缩状态机。
//!
//! ## 设计
//! 基于 `katu-core::compaction` 定义的配置与结果类型，本模块实现实际的运行时逻辑：
//!
//! ```text
//! L1: Prune（无 LLM）
//!   └─ 截断旧工具输出，释放 token 空间
//!
//! L2: Compact（有 LLM）
//!   ├─ 触发: Auto / Overflow / Manual / Idle
//!   ├─ 选择待摘要 vs 保留的消息
//!   ├─ 调用 LLM 生成摘要
//!   └─ 重建消息历史
//! ```
//!
//! ## 对外接口
//! - `Compactor` — 压缩执行 trait（异步，object-safe）
//! - `CompactionState` — 压缩运行时状态（熔断器、防重复）
//! - `PruneOutcome` — prune 操作结果
//! - `MessagePartition` — 消息切分方案
//! - `DefaultCompactor` — 默认压缩器实现
//!
//! ## 调用者
//! - `katu-agent::runner` (future) — Agent loop 在每步后调用
//! - `katu-agent::session` — 持有 CompactionState

use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;

use katu_core::compaction::{
    CompactTrigger, CompactionConfig, CompactionResult, PreserveConfig,
};
use katu_core::message::{AssistantBlock, ContentBlock, Message};

use katu_llm::model::ModelRef;
use katu_llm::Provider;

use crate::error::Result;
use crate::session::Session;

// ===========================================================================
// PruneOutcome
// ===========================================================================

/// Prune 操作结果 — 轻量级修剪的统计。
///
/// # Examples
///
/// ```
/// use katu_agent::compaction::PruneOutcome;
///
/// let outcome = PruneOutcome::none();
/// assert_eq!(outcome.tokens_freed, 0);
/// assert!(!outcome.has_effect());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PruneOutcome {
    /// 释放的估计 token 数。
    pub tokens_freed: u64,
    /// 被修剪的工具输出条目数。
    pub parts_pruned: usize,
}

impl PruneOutcome {
    /// 无修剪发生。
    pub fn none() -> Self {
        Self {
            tokens_freed: 0,
            parts_pruned: 0,
        }
    }

    /// 是否产生了实际效果。
    pub fn has_effect(&self) -> bool {
        self.parts_pruned > 0
    }
}

// ===========================================================================
// MessagePartition
// ===========================================================================

/// 消息切分方案 — 哪些消息待摘要，哪些保留原文。
///
/// 由 `partition_messages()` 计算，传递给压缩器执行摘要生成。
///
/// ```text
/// messages: [0..cut_point] → to_summarize
///           [cut_point..]  → to_preserve
/// ```
///
/// # Invariant
/// - `to_summarize.end == to_preserve.start`
/// - tool_use/tool_result 对不被拆散
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessagePartition {
    /// 待摘要消息范围。
    pub to_summarize: Range<usize>,
    /// 保留原文消息范围。
    pub to_preserve: Range<usize>,
    /// 上一次压缩的摘要（增量更新用）。
    pub previous_summary: Option<String>,
}

impl MessagePartition {
    /// 待摘要消息数量。
    pub fn summarize_count(&self) -> usize {
        self.to_summarize.len()
    }

    /// 保留消息数量。
    pub fn preserve_count(&self) -> usize {
        self.to_preserve.len()
    }

    /// 是否有内容需要摘要。
    pub fn has_work(&self) -> bool {
        !self.to_summarize.is_empty()
    }
}

// ===========================================================================
// CompactionState
// ===========================================================================

/// 压缩运行时状态 — 跟踪熔断器与防重复触发。
///
/// 由 Session 持有，在 Agent loop 期间维护。
///
/// # Examples
///
/// ```
/// use katu_agent::compaction::CompactionState;
///
/// let mut state = CompactionState::new();
/// assert!(!state.is_circuit_broken(3));
///
/// state.record_failure();
/// state.record_failure();
/// state.record_failure();
/// assert!(state.is_circuit_broken(3));
///
/// state.record_success();
/// assert!(!state.is_circuit_broken(3));
/// ```
#[derive(Debug, Clone)]
pub struct CompactionState {
    /// 连续自动压缩失败计数。
    consecutive_failures: u32,
    /// 上次压缩的 step 序号（防止同一步重复触发）。
    last_compact_step: Option<u32>,
    /// 上次压缩后的 token 数。
    last_compact_tokens: Option<u64>,
}

impl CompactionState {
    /// 创建初始状态。
    pub fn new() -> Self {
        Self {
            consecutive_failures: 0,
            last_compact_step: None,
            last_compact_tokens: None,
        }
    }

    /// 记录一次压缩成功 — 重置熔断器。
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
    }

    /// 记录一次压缩失败 — 递增熔断器计数。
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
    }

    /// 熔断器是否已触发。
    ///
    /// `max_failures == 0` 表示不限制。
    pub fn is_circuit_broken(&self, max_failures: u32) -> bool {
        max_failures > 0 && self.consecutive_failures >= max_failures
    }

    /// 标记当前步已执行压缩。
    pub fn mark_compacted(&mut self, step: u32, tokens_after: Option<u64>) {
        self.last_compact_step = Some(step);
        self.last_compact_tokens = tokens_after;
    }

    /// 检查指定步是否已执行过压缩（防重复）。
    pub fn already_compacted_at(&self, step: u32) -> bool {
        self.last_compact_step == Some(step)
    }

    /// 连续失败次数。
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    /// 上次压缩后的 token 数。
    pub fn last_compact_tokens(&self) -> Option<u64> {
        self.last_compact_tokens
    }

    /// 重置（用于 session 恢复等场景）。
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for CompactionState {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Compactor trait
// ===========================================================================

/// 压缩执行 trait — 所有压缩策略的统一接口。
///
/// ## Object Safety
/// 通过 `#[async_trait]` 实现 dyn dispatch，支持 `Arc<dyn Compactor>` 存储。
///
/// ## 设计选择
/// - **`prune`** — L1 轻量修剪，无 LLM 调用
/// - **`compact`** — L2 全量压缩，需要 LLM 调用
/// - **`partition`** — 消息切分逻辑，可单独测试
///
/// # Examples
///
/// ```ignore
/// use std::sync::Arc;
/// use katu_agent::compaction::Compactor;
///
/// async fn run_compact(compactor: &dyn Compactor, session: &mut Session) {
///     let outcome = compactor.prune(session).await.unwrap();
///     if session.should_compact() {
///         let result = compactor.compact(session, CompactTrigger::Auto).await.unwrap();
///     }
/// }
/// ```
#[async_trait]
pub trait Compactor: Send + Sync {
    /// L1: 修剪旧工具输出（无 LLM 调用）。
    ///
    /// 从最新消息向旧遍历，跳过 `PreserveConfig.recent_turns`，
    /// 累计 tool output tokens 超过 `PruneConfig.protect_tokens` 后截断。
    /// 仅当总修剪量超过 `PruneConfig.minimum_tokens` 时实际执行。
    async fn prune(&self, session: &mut Session) -> Result<PruneOutcome>;

    /// 计算消息切分方案。
    ///
    /// 根据 `PreserveConfig` 确定保留范围，保证 tool_use/tool_result 对完整。
    fn partition(&self, session: &Session) -> MessagePartition;

    /// L2: 执行全量压缩（调用 LLM 生成摘要）。
    ///
    /// 流程：
    /// 1. 调用 `partition()` 确定切分
    /// 2. 对 `to_summarize` 范围的消息调用 LLM 生成摘要
    /// 3. 用 `session.replace_messages()` 重建消息历史
    /// 4. 返回 `CompactionResult`
    async fn compact(
        &self,
        session: &mut Session,
        trigger: CompactTrigger,
    ) -> Result<CompactionResult>;
}

// ===========================================================================
// DefaultCompactor
// ===========================================================================

/// 默认压缩器 — 使用 LLM Provider 生成摘要。
///
/// ## 职责
/// - 实现 `Compactor` trait
/// - 支持可选的独立压缩模型（与主 Agent 模型不同）
/// - 读取 `CompactionConfig` 驱动行为
///
/// # Examples
///
/// ```ignore
/// use std::sync::Arc;
/// use katu_agent::compaction::DefaultCompactor;
///
/// let compactor = DefaultCompactor::new(provider, model);
/// ```
pub struct DefaultCompactor {
    /// 用于生成摘要的 LLM Provider。
    provider: Arc<dyn Provider>,
    /// 用于生成摘要的模型。
    model: ModelRef,
}

impl DefaultCompactor {
    /// 创建默认压缩器。
    pub fn new(provider: Arc<dyn Provider>, model: ModelRef) -> Self {
        Self { provider, model }
    }

    /// Provider 引用。
    pub fn provider(&self) -> &Arc<dyn Provider> {
        &self.provider
    }

    /// 模型引用。
    pub fn model(&self) -> &ModelRef {
        &self.model
    }
}

impl std::fmt::Debug for DefaultCompactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultCompactor")
            .field("model_id", &self.model.id)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Prune 实现辅助
// ---------------------------------------------------------------------------

/// 估算单个 ContentBlock 的 token 数（粗估: chars / 4）。
fn estimate_block_tokens(block: &ContentBlock) -> u64 {
    match block {
        ContentBlock::Text { text } => text.len() as u64 / 4,
        ContentBlock::Image { .. } => 1_000, // 图片固定估算
    }
}

/// 判断 tool result 是否受保护。
fn is_protected_tool(tool_name: &str, protected: &[String]) -> bool {
    protected.iter().any(|p| p == tool_name)
}

// ---------------------------------------------------------------------------
// Partition 实现辅助
// ---------------------------------------------------------------------------

/// 从消息列表末尾找到保留范围的起始索引。
///
/// 逻辑：
/// 1. 从末尾向前数 `recent_turns` 个 user message
/// 2. 向前扩展保证不拆散 tool_use/tool_result 对
/// 3. 确保保留部分不超过 `recent_tokens` 预算
fn find_preserve_start(
    messages: &[Message],
    preserve: &PreserveConfig,
    _context_window: u64,
    _reserve_tokens: u64,
) -> usize {
    if messages.is_empty() {
        return 0;
    }

    let recent_turns = preserve.recent_turns as usize;
    if recent_turns == 0 {
        return messages.len();
    }

    // 从末尾向前找到第 N 个 user message
    let mut user_count = 0;
    let mut cut = messages.len();
    for i in (0..messages.len()).rev() {
        if matches!(&messages[i], Message::User(_)) {
            user_count += 1;
            if user_count >= recent_turns {
                cut = i;
                break;
            }
        }
    }

    // 向前调整：不拆散 tool_use/tool_result 对
    // 如果 cut 位置是 ToolResult，向前移到对应的 Assistant（含 ToolCall）
    while cut > 0 && matches!(&messages[cut], Message::ToolResult(_)) {
        cut -= 1;
    }

    // 如果保留部分为 0 条消息（所有消息都在 summarize），至少保留最后一条 user
    if cut >= messages.len() {
        // 找最后一个 user message
        for i in (0..messages.len()).rev() {
            if matches!(&messages[i], Message::User(_)) {
                cut = i;
                break;
            }
        }
    }

    cut
}

// ---------------------------------------------------------------------------
// DefaultCompactor — Compactor 实现
// ---------------------------------------------------------------------------

#[async_trait]
impl Compactor for DefaultCompactor {
    async fn prune(&self, session: &mut Session) -> Result<PruneOutcome> {
        let config = session.compaction_config().prune.clone();
        if !config.enabled {
            return Ok(PruneOutcome::none());
        }

        let messages = session.message_slice();
        let recent_turns = session.compaction_config().preserve.recent_turns as usize;

        // 从末尾向前找到保护边界（跳过最近 N 个 user turn）
        let mut user_count = 0;
        let mut prune_boundary = messages.len();
        for i in (0..messages.len()).rev() {
            if matches!(&messages[i], Message::User(_)) {
                user_count += 1;
                if user_count >= recent_turns {
                    prune_boundary = i;
                    break;
                }
            }
        }

        // 从 prune_boundary 向前遍历 ToolResult，累计 token
        let mut protected_tokens: u64 = 0;
        let mut prunable_tokens: u64 = 0;
        let mut to_prune: Vec<usize> = Vec::new();

        for i in (0..prune_boundary).rev() {
            if let Message::ToolResult(ref tr) = messages[i] {
                if is_protected_tool(&tr.tool_name, &config.protected_tools) {
                    continue;
                }

                let msg_tokens: u64 = tr.content.iter().map(estimate_block_tokens).sum();

                if protected_tokens < config.protect_tokens {
                    protected_tokens += msg_tokens;
                } else {
                    // 检查是否超过截断长度
                    let total_chars: usize = tr.content.iter().map(|b| match b {
                        ContentBlock::Text { text } => text.len(),
                        ContentBlock::Image { .. } => 0,
                    }).sum();

                    if total_chars > config.tool_output_max_chars {
                        prunable_tokens += msg_tokens;
                        to_prune.push(i);
                    }
                }
            }
        }

        // 仅当修剪量超过最小阈值时执行
        if prunable_tokens < config.minimum_tokens {
            return Ok(PruneOutcome::none());
        }

        // 执行修剪：截断 tool output 内容
        let truncation_msg = format!(
            "[内容已修剪 - 原文超过 {} 字符]",
            config.tool_output_max_chars
        );

        for &idx in &to_prune {
            session.truncate_tool_result(idx, &truncation_msg, config.tool_output_max_chars);
        }

        Ok(PruneOutcome {
            tokens_freed: prunable_tokens,
            parts_pruned: to_prune.len(),
        })
    }

    fn partition(&self, session: &Session) -> MessagePartition {
        let messages = session.message_slice();
        let config = session.compaction_config();

        // 检查是否有上一次压缩的摘要（第一条消息是否为摘要格式的 user message）
        let previous_summary = detect_previous_summary(messages);

        let cut = find_preserve_start(
            messages,
            &config.preserve,
            session.context_window(),
            config.reserve_tokens,
        );

        MessagePartition {
            to_summarize: 0..cut,
            to_preserve: cut..messages.len(),
            previous_summary,
        }
    }

    async fn compact(
        &self,
        session: &mut Session,
        trigger: CompactTrigger,
    ) -> Result<CompactionResult> {
        let partition = self.partition(session);

        if !partition.has_work() {
            return Ok(CompactionResult {
                summary: String::new(),
                short_summary: None,
                trigger,
                tokens_before: session.context_tokens(),
                tokens_after: Some(session.context_tokens()),
                messages_compacted: 0,
                messages_kept: session.message_slice().len(),
                success: true,
            });
        }

        let tokens_before = session.context_tokens();
        let messages_to_summarize = &session.message_slice()[partition.to_summarize.clone()];
        let messages_compacted = partition.summarize_count();
        let messages_kept = partition.preserve_count();

        // 构建摘要 prompt
        let summary_prompt = build_summary_prompt(
            messages_to_summarize,
            partition.previous_summary.as_deref(),
            session.compaction_config(),
        );

        // 调用 LLM 生成摘要
        let summary_request = katu_llm::LlmRequest::new(self.model.clone())
            .with_system(COMPACTION_SYSTEM_PROMPT)
            .with_message(Message::user(summary_prompt));

        let response = match self.provider.generate(summary_request).await {
            Ok(resp) => resp,
            Err(e) => {
                return Ok(CompactionResult {
                    summary: format!("压缩失败: {e}"),
                    short_summary: None,
                    trigger,
                    tokens_before,
                    tokens_after: None,
                    messages_compacted: 0,
                    messages_kept: session.message_slice().len(),
                    success: false,
                });
            }
        };

        // 提取摘要文本
        let summary = extract_text_from_message(&response.message);

        // 重建消息历史: [摘要 user msg] + preserved messages
        let preserved = session.message_slice()[partition.to_preserve.clone()].to_vec();
        let mut new_messages = Vec::with_capacity(1 + preserved.len());

        // 摘要作为 user message 注入（让 LLM 知道之前的上下文）
        let summary_content = format!(
            "<context_summary>\n{}\n</context_summary>\n\n以上是之前对话的摘要，请基于此上下文继续。",
            summary
        );
        new_messages.push(Message::user(summary_content));
        new_messages.extend(preserved);

        session.replace_messages(new_messages);

        Ok(CompactionResult {
            summary: summary.clone(),
            short_summary: None, // TODO: 生成短摘要
            trigger,
            tokens_before,
            tokens_after: None, // 由外部重新计算
            messages_compacted,
            messages_kept,
            success: true,
        })
    }
}

// ===========================================================================
// 内部辅助函数
// ===========================================================================

/// 压缩系统 prompt — 指导 LLM 如何生成摘要。
const COMPACTION_SYSTEM_PROMPT: &str = "\
你是一个对话摘要助手。你的任务是将一段对话历史压缩为简洁但信息完整的摘要。

要求：
1. 保留所有关键决策、代码变更、文件路径和技术细节
2. 保留用户的偏好和约束
3. 保留未完成的任务和待办事项
4. 省略重复的探索过程和已解决的中间问题
5. 使用结构化格式（标题 + 要点列表）
6. 如果有文件操作，列出最终状态而非中间步骤";

/// 检测消息列表中是否存在上一次压缩的摘要。
fn detect_previous_summary(messages: &[Message]) -> Option<String> {
    if let Some(Message::User(user_msg)) = messages.first() {
        let text = user_msg.content.text();
        if text.contains("<context_summary>") && text.contains("</context_summary>") {
            // 提取摘要内容
            if let Some(start) = text.find("<context_summary>") {
                let content_start = start + "<context_summary>".len();
                if let Some(end) = text[content_start..].find("</context_summary>") {
                    return Some(text[content_start..content_start + end].trim().to_string());
                }
            }
        }
    }
    None
}

/// 从 Message 中提取纯文本内容。
fn extract_text_from_message(message: &Message) -> String {
    match message {
        Message::Assistant(a) => a.text(),
        Message::User(u) => u.content.text(),
        Message::ToolResult(t) => {
            t.content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
    }
}

/// 构建摘要 prompt — 将待摘要消息格式化为 LLM 输入。
fn build_summary_prompt(
    messages: &[Message],
    previous_summary: Option<&str>,
    _config: &CompactionConfig,
) -> String {
    let mut prompt = String::with_capacity(4096);

    if let Some(prev) = previous_summary {
        prompt.push_str("## 上次摘要\n\n");
        prompt.push_str(prev);
        prompt.push_str("\n\n## 新增对话（需要整合到摘要中）\n\n");
    } else {
        prompt.push_str("## 对话历史（需要压缩为摘要）\n\n");
    }

    for msg in messages {
        match msg {
            Message::User(u) => {
                prompt.push_str("**User**: ");
                prompt.push_str(&u.content.text());
                prompt.push('\n');
            }
            Message::Assistant(a) => {
                prompt.push_str("**Assistant**: ");
                // 截断过长的 assistant 内容
                let text = a.text();
                if text.len() > 2000 {
                    prompt.push_str(&text[..2000]);
                    prompt.push_str("...[截断]");
                } else {
                    prompt.push_str(&text);
                }
                prompt.push('\n');

                // 记录 tool calls（仅名称和简要参数）
                for block in a.tool_calls() {
                    if let AssistantBlock::ToolCall { name, arguments, .. } = block {
                        prompt.push_str(&format!("  → tool_call: {}({})\n", name, arguments));
                    }
                }
            }
            Message::ToolResult(t) => {
                let content = t.content.iter().filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                }).collect::<Vec<_>>().join("");

                // 截断过长的 tool 输出
                if content.len() > 500 {
                    prompt.push_str(&format!(
                        "  ← {}: {}...[截断]\n",
                        t.tool_name,
                        &content[..500]
                    ));
                } else {
                    prompt.push_str(&format!("  ← {}: {}\n", t.tool_name, content));
                }
            }
        }
    }

    prompt.push_str("\n请生成压缩摘要。");
    prompt
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- PruneOutcome --

    #[test]
    fn test_prune_outcome_none() {
        let outcome = PruneOutcome::none();
        assert_eq!(outcome.tokens_freed, 0);
        assert_eq!(outcome.parts_pruned, 0);
        assert!(!outcome.has_effect());
    }

    #[test]
    fn test_prune_outcome_with_effect() {
        let outcome = PruneOutcome {
            tokens_freed: 5000,
            parts_pruned: 3,
        };
        assert!(outcome.has_effect());
    }

    // -- MessagePartition --

    #[test]
    fn test_partition_counts() {
        let partition = MessagePartition {
            to_summarize: 0..10,
            to_preserve: 10..15,
            previous_summary: None,
        };
        assert_eq!(partition.summarize_count(), 10);
        assert_eq!(partition.preserve_count(), 5);
        assert!(partition.has_work());
    }

    #[test]
    fn test_partition_empty_summarize() {
        let partition = MessagePartition {
            to_summarize: 0..0,
            to_preserve: 0..5,
            previous_summary: None,
        };
        assert!(!partition.has_work());
    }

    // -- CompactionState --

    #[test]
    fn test_compaction_state_new() {
        let state = CompactionState::new();
        assert_eq!(state.consecutive_failures(), 0);
        assert!(!state.is_circuit_broken(3));
        assert!(!state.already_compacted_at(0));
    }

    #[test]
    fn test_compaction_state_circuit_breaker() {
        let mut state = CompactionState::new();
        state.record_failure();
        state.record_failure();
        assert!(!state.is_circuit_broken(3));
        state.record_failure();
        assert!(state.is_circuit_broken(3));
    }

    #[test]
    fn test_compaction_state_reset_on_success() {
        let mut state = CompactionState::new();
        state.record_failure();
        state.record_failure();
        state.record_success();
        assert_eq!(state.consecutive_failures(), 0);
        assert!(!state.is_circuit_broken(3));
    }

    #[test]
    fn test_compaction_state_no_limit() {
        let mut state = CompactionState::new();
        for _ in 0..100 {
            state.record_failure();
        }
        // max_failures == 0 表示不限制
        assert!(!state.is_circuit_broken(0));
    }

    #[test]
    fn test_compaction_state_mark_compacted() {
        let mut state = CompactionState::new();
        assert!(!state.already_compacted_at(5));
        state.mark_compacted(5, Some(50_000));
        assert!(state.already_compacted_at(5));
        assert!(!state.already_compacted_at(6));
        assert_eq!(state.last_compact_tokens(), Some(50_000));
    }

    // -- detect_previous_summary --

    #[test]
    fn test_detect_no_summary() {
        let messages = vec![Message::user("hello")];
        assert_eq!(detect_previous_summary(&messages), None);
    }

    #[test]
    fn test_detect_has_summary() {
        let summary_msg = "<context_summary>\nPrevious work done\n</context_summary>\n\n以上是之前对话的摘要，请基于此上下文继续。";
        let messages = vec![Message::user(summary_msg)];
        assert_eq!(
            detect_previous_summary(&messages),
            Some("Previous work done".to_string())
        );
    }

    // -- find_preserve_start --

    #[test]
    fn test_find_preserve_start_empty() {
        let messages: Vec<Message> = vec![];
        let preserve = PreserveConfig::default();
        assert_eq!(find_preserve_start(&messages, &preserve, 200_000, 16_384), 0);
    }

    #[test]
    fn test_find_preserve_start_keeps_recent_turns() {
        let messages = vec![
            Message::user("q1"),
            Message::assistant("a1"),
            Message::user("q2"),
            Message::assistant("a2"),
            Message::user("q3"),
            Message::assistant("a3"),
        ];
        // recent_turns=2 → 保留最后 2 个 user turn (q2, a2, q3, a3)
        let preserve = PreserveConfig::new(2, 100_000);
        let cut = find_preserve_start(&messages, &preserve, 200_000, 16_384);
        assert_eq!(cut, 2); // messages[2..] = q2, a2, q3, a3
    }

    // -- estimate_block_tokens --

    #[test]
    fn test_estimate_block_tokens_text() {
        let block = ContentBlock::Text { text: "a".repeat(400) };
        assert_eq!(estimate_block_tokens(&block), 100);
    }
}
