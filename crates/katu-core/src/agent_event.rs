//! # katu_core::agent_event
//!
//! ## 职责
//! 定义 Agent 语义层事件 — Agent loop 在执行过程中产出的已发生事件。
//!
//! ## 设计
//! `AgentEvent` 是 **StreamEvent 的上层抽象**：
//! - `StreamEvent`（event.rs）= LLM provider 层的原始 SSE 事件
//! - `AgentEvent`（本模块）= Agent loop 语义层的已发生事实
//!
//! Agent loop 消费 `StreamEvent` 流，经过工具执行、状态管理等逻辑后，
//! 产出 `AgentEvent` 供 UI/日志/持久化层订阅。
//!
//! ## 事件分类（28 种）
//! 按 namespace 分 8 组：
//! - **Agent** — 整个 Agent 运行的开始/结束
//! - **Step** — 单次 LLM 推理步骤的生命周期
//! - **Text** — 文本流式输出
//! - **Reasoning** — 思维链流式输出
//! - **Tool** — 工具从参数生成到执行完成的全生命周期
//! - **Retry** — API 调用重试
//! - **Compaction** — 上下文压缩（含修剪和 token 预算变更）
//! - **State** — 配置/状态变更
//!
//! ## 与 StreamEvent 的关系
//! ```text
//! LLM Provider ──SSE──► StreamEvent ──Agent Loop──► AgentEvent ──► UI / 日志
//! ```
//!
//! ## 调用者
//! - `katu-agent` (future) — Agent loop 产出 AgentEvent
//! - UI 层 — 实时渲染 Agent 执行状态
//! - 持久化层 — 记录执行历史
//! - 遥测层 — 统计和监控

use serde::{Deserialize, Serialize};

use crate::compaction::{CompactTrigger, CompactionResult, CompactionStrategy, TokenBudgetState};
use crate::tool::ToolOutput;
use crate::types::{AgentId, FinishReason, ModelId, SessionId, ToolCallId};
use crate::usage::Usage;

// ===========================================================================
// AgentEvent
// ===========================================================================

/// Agent 语义层事件 — Agent loop 在执行过程中产出的已发生事实。
///
/// 事件是 **不可变的已发生事实**，消费者只能观察，不能拦截或修改。
/// 如需拦截/修改 Agent 行为，请使用 Hook 系统（future）。
///
/// # 事件分层
///
/// ```text
/// AgentStarted
/// ├── StepStarted
/// │   ├── TextStarted → TextDelta* → TextEnded
/// │   ├── ReasoningStarted → ReasoningDelta* → ReasoningEnded
/// │   ├── ToolInputStarted → ToolInputDelta* → ToolInputEnded
/// │   │   └── ToolCalled → ToolProgress* → ToolSucceeded / ToolFailed
/// │   └── StepEnded / StepFailed
/// ├── Retried
/// ├── PruneCompleted
/// ├── CompactionStarted → CompactionDelta* → CompactionEnded
/// ├── TokenBudgetChanged
/// ├── ModelSwitched / AgentSwitched / UserPrompted
/// └── AgentEnded
/// ```
///
/// # Serde 格式
///
/// 所有事件序列化为 `{"type": "snake_case_variant", ...fields}` 格式，
/// 与 `StreamEvent` 保持风格一致。
///
/// # Examples
///
/// ```
/// use katu_core::agent_event::AgentEvent;
/// use katu_core::{SessionId, ModelId};
///
/// let event = AgentEvent::AgentStarted {
///     session_id: SessionId::new(),
///     agent_name: "coder".into(),
///     model_id: ModelId::new("gpt-4o"),
/// };
/// let json = serde_json::to_string(&event).unwrap();
/// assert!(json.contains(r#""type":"agent_started""#));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    // ── 1. Agent 生命周期 (2) ────────────────────────────────

    /// Agent loop 开始执行。
    AgentStarted {
        session_id: SessionId,
        agent_name: String,
        model_id: ModelId,
    },

    /// Agent loop 执行结束。
    AgentEnded {
        session_id: SessionId,
        finish_reason: AgentFinishReason,
        /// 整个 Agent 运行的累计 token 用量。
        #[serde(skip_serializing_if = "Option::is_none")]
        total_usage: Option<Usage>,
        /// 总执行步数。
        steps: u32,
    },

    // ── 2. Step 生命周期 (3) ─────────────────────────────────

    /// 一个推理步骤开始（一次 LLM API 调用）。
    StepStarted {
        step_index: u32,
        model_id: ModelId,
        agent_name: String,
    },

    /// 一个推理步骤正常完成。
    StepEnded {
        step_index: u32,
        finish_reason: FinishReason,
        /// 本步骤的 token 用量。
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },

    /// 一个推理步骤失败（API 错误、网络错误等）。
    StepFailed {
        step_index: u32,
        error: String,
    },

    // ── 3. Text 流式 (3) ────────────────────────────────────

    /// 文本内容块开始。
    TextStarted {
        content_index: usize,
    },

    /// 文本增量。
    TextDelta {
        content_index: usize,
        delta: String,
    },

    /// 文本内容块完成，携带完整文本。
    TextEnded {
        content_index: usize,
        text: String,
    },

    // ── 4. Reasoning 流式 (3) ───────────────────────────────

    /// 思维链内容块开始。
    ReasoningStarted {
        content_index: usize,
    },

    /// 思维链增量。
    ReasoningDelta {
        content_index: usize,
        delta: String,
    },

    /// 思维链内容块完成，携带完整文本。
    ReasoningEnded {
        content_index: usize,
        text: String,
    },

    // ── 5. Tool 生命周期 (7) ────────────────────────────────

    /// LLM 开始生成工具调用参数。
    ToolInputStarted {
        call_id: ToolCallId,
        tool_name: String,
    },

    /// 工具参数 JSON 增量。
    ToolInputDelta {
        call_id: ToolCallId,
        delta: String,
    },

    /// 工具参数生成完毕，携带完整参数。
    ToolInputEnded {
        call_id: ToolCallId,
        arguments: serde_json::Value,
    },

    /// 工具开始执行（参数已解析、权限已通过）。
    ToolCalled {
        call_id: ToolCallId,
        tool_name: String,
        arguments: serde_json::Value,
    },

    /// 工具执行中间进度。
    ToolProgress {
        call_id: ToolCallId,
        /// 进度消息（如 "正在读取文件..."）。
        message: String,
        /// 可选的结构化进度数据。
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<serde_json::Value>,
    },

    /// 工具执行成功。
    ToolSucceeded {
        call_id: ToolCallId,
        tool_name: String,
        output: ToolOutput,
    },

    /// 工具执行失败。
    ToolFailed {
        call_id: ToolCallId,
        tool_name: String,
        error: String,
        /// 是否可重试。
        #[serde(default)]
        is_retryable: bool,
    },

    // ── 6. Retry (1) ────────────────────────────────────────

    /// API 调用重试。
    Retried {
        /// 当前重试次数（从 1 开始）。
        attempt: u32,
        /// 触发重试的错误。
        error: String,
        /// 重试间隔（毫秒）。
        delay_ms: u64,
    },

    // ── 7. Compaction (5) ───────────────────────────────────

    /// 旧工具输出修剪完成。
    ///
    /// Prune 是轻量级的上下文优化，独立于全量压缩。
    /// 截断旧的、体积大的工具输出，释放 token 空间。
    PruneCompleted {
        /// 修剪释放的估计 token 数。
        tokens_freed: u64,
        /// 被修剪的工具输出条数。
        parts_pruned: usize,
    },

    /// 上下文压缩开始。
    CompactionStarted {
        /// 触发原因。
        trigger: CompactTrigger,
        /// 使用的压缩策略。
        strategy: CompactionStrategy,
        /// 压缩前的 prompt token 数。
        tokens_before: u64,
    },

    /// 压缩摘要增量。
    CompactionDelta {
        delta: String,
    },

    /// 上下文压缩完成。
    CompactionEnded {
        /// 压缩完整结果。
        result: CompactionResult,
    },

    /// Token 用量预算状态变更。
    ///
    /// 当 token 用量跨越阈值边界时产出，用于 UI 进度条渲染。
    TokenBudgetChanged {
        /// 当前已使用 token 数。
        used_tokens: u64,
        /// 模型 context window 大小。
        context_window: u64,
        /// 新状态。
        state: TokenBudgetState,
    },

    // ── 8. State 变更 (3) ───────────────────────────────────

    /// 模型已切换。
    ModelSwitched {
        #[serde(skip_serializing_if = "Option::is_none")]
        from: Option<ModelId>,
        to: ModelId,
    },

    /// Agent 已切换（进入或退出子 Agent）。
    AgentSwitched {
        #[serde(skip_serializing_if = "Option::is_none")]
        from_agent: Option<AgentId>,
        to_agent: AgentId,
        agent_name: String,
    },

    /// 用户提交了新 prompt。
    UserPrompted {
        /// 用户输入内容的摘要（可能截断，避免事件过大）。
        content_preview: String,
    },
}

// ===========================================================================
// 辅助枚举
// ===========================================================================

/// Agent 运行结束原因。
///
/// 区别于 `FinishReason`（LLM 单步停止原因），`AgentFinishReason`
/// 描述整个 Agent loop 的终止原因。
///
/// # Examples
///
/// ```
/// use katu_core::agent_event::AgentFinishReason;
///
/// let reason = AgentFinishReason::Completed;
/// let json = serde_json::to_string(&reason).unwrap();
/// assert_eq!(json, r#""completed""#);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentFinishReason {
    /// 正常完成 — LLM 在最终步骤返回 end_turn 且无待处理的 tool call。
    Completed,
    /// 用户中断。
    UserAbort,
    /// 达到最大步数限制。
    MaxSteps,
    /// 达到 token 预算上限。
    TokenBudget,
    /// 不可恢复错误导致终止。
    Error,
    /// 超时。
    Timeout,
}

impl std::fmt::Display for AgentFinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Completed => write!(f, "completed"),
            Self::UserAbort => write!(f, "user_abort"),
            Self::MaxSteps => write!(f, "max_steps"),
            Self::TokenBudget => write!(f, "token_budget"),
            Self::Error => write!(f, "error"),
            Self::Timeout => write!(f, "timeout"),
        }
    }
}

// ===========================================================================
// AgentEvent — helper methods
// ===========================================================================

impl AgentEvent {
    /// 返回此事件的类型标签。
    pub fn kind(&self) -> AgentEventKind {
        match self {
            Self::AgentStarted { .. } => AgentEventKind::AgentStarted,
            Self::AgentEnded { .. } => AgentEventKind::AgentEnded,
            Self::StepStarted { .. } => AgentEventKind::StepStarted,
            Self::StepEnded { .. } => AgentEventKind::StepEnded,
            Self::StepFailed { .. } => AgentEventKind::StepFailed,
            Self::TextStarted { .. } => AgentEventKind::TextStarted,
            Self::TextDelta { .. } => AgentEventKind::TextDelta,
            Self::TextEnded { .. } => AgentEventKind::TextEnded,
            Self::ReasoningStarted { .. } => AgentEventKind::ReasoningStarted,
            Self::ReasoningDelta { .. } => AgentEventKind::ReasoningDelta,
            Self::ReasoningEnded { .. } => AgentEventKind::ReasoningEnded,
            Self::ToolInputStarted { .. } => AgentEventKind::ToolInputStarted,
            Self::ToolInputDelta { .. } => AgentEventKind::ToolInputDelta,
            Self::ToolInputEnded { .. } => AgentEventKind::ToolInputEnded,
            Self::ToolCalled { .. } => AgentEventKind::ToolCalled,
            Self::ToolProgress { .. } => AgentEventKind::ToolProgress,
            Self::ToolSucceeded { .. } => AgentEventKind::ToolSucceeded,
            Self::ToolFailed { .. } => AgentEventKind::ToolFailed,
            Self::Retried { .. } => AgentEventKind::Retried,
            Self::PruneCompleted { .. } => AgentEventKind::PruneCompleted,
            Self::CompactionStarted { .. } => AgentEventKind::CompactionStarted,
            Self::CompactionDelta { .. } => AgentEventKind::CompactionDelta,
            Self::CompactionEnded { .. } => AgentEventKind::CompactionEnded,
            Self::TokenBudgetChanged { .. } => AgentEventKind::TokenBudgetChanged,
            Self::ModelSwitched { .. } => AgentEventKind::ModelSwitched,
            Self::AgentSwitched { .. } => AgentEventKind::AgentSwitched,
            Self::UserPrompted { .. } => AgentEventKind::UserPrompted,
        }
    }

    /// 是否为终态事件（AgentEnded）。
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::AgentEnded { .. })
    }

    /// 是否为文本增量事件。
    pub fn is_text_delta(&self) -> bool {
        matches!(self, Self::TextDelta { .. })
    }

    /// 提取文本增量内容，非 TextDelta 事件返回 None。
    pub fn as_text_delta(&self) -> Option<&str> {
        match self {
            Self::TextDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        }
    }

    /// 是否为工具生命周期事件。
    pub fn is_tool_event(&self) -> bool {
        matches!(
            self,
            Self::ToolInputStarted { .. }
                | Self::ToolInputDelta { .. }
                | Self::ToolInputEnded { .. }
                | Self::ToolCalled { .. }
                | Self::ToolProgress { .. }
                | Self::ToolSucceeded { .. }
                | Self::ToolFailed { .. }
        )
    }

    /// 提取关联的 ToolCallId（如果是工具事件）。
    pub fn tool_call_id(&self) -> Option<&ToolCallId> {
        match self {
            Self::ToolInputStarted { call_id, .. }
            | Self::ToolInputDelta { call_id, .. }
            | Self::ToolInputEnded { call_id, .. }
            | Self::ToolCalled { call_id, .. }
            | Self::ToolProgress { call_id, .. }
            | Self::ToolSucceeded { call_id, .. }
            | Self::ToolFailed { call_id, .. } => Some(call_id),
            _ => None,
        }
    }

    /// 是否为 delta 类事件（高频、增量）。
    pub fn is_delta(&self) -> bool {
        matches!(
            self,
            Self::TextDelta { .. }
                | Self::ReasoningDelta { .. }
                | Self::ToolInputDelta { .. }
                | Self::CompactionDelta { .. }
        )
    }
}

// ===========================================================================
// AgentEventKind — 判别标签
// ===========================================================================

/// AgentEvent 的类型判别标签（无数据），用于过滤和分类。
///
/// 与 `AgentEvent` 一一对应，可用于：
/// - 事件过滤器（只订阅关心的事件类型）
/// - 统计计数
/// - Hook 系统的事件匹配
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentEventKind {
    // Agent
    AgentStarted,
    AgentEnded,
    // Step
    StepStarted,
    StepEnded,
    StepFailed,
    // Text
    TextStarted,
    TextDelta,
    TextEnded,
    // Reasoning
    ReasoningStarted,
    ReasoningDelta,
    ReasoningEnded,
    // Tool
    ToolInputStarted,
    ToolInputDelta,
    ToolInputEnded,
    ToolCalled,
    ToolProgress,
    ToolSucceeded,
    ToolFailed,
    // Retry
    Retried,
    // Compaction
    PruneCompleted,
    CompactionStarted,
    CompactionDelta,
    CompactionEnded,
    TokenBudgetChanged,
    // State
    ModelSwitched,
    AgentSwitched,
    UserPrompted,
}

impl std::fmt::Display for AgentEventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::AgentStarted => "agent_started",
            Self::AgentEnded => "agent_ended",
            Self::StepStarted => "step_started",
            Self::StepEnded => "step_ended",
            Self::StepFailed => "step_failed",
            Self::TextStarted => "text_started",
            Self::TextDelta => "text_delta",
            Self::TextEnded => "text_ended",
            Self::ReasoningStarted => "reasoning_started",
            Self::ReasoningDelta => "reasoning_delta",
            Self::ReasoningEnded => "reasoning_ended",
            Self::ToolInputStarted => "tool_input_started",
            Self::ToolInputDelta => "tool_input_delta",
            Self::ToolInputEnded => "tool_input_ended",
            Self::ToolCalled => "tool_called",
            Self::ToolProgress => "tool_progress",
            Self::ToolSucceeded => "tool_succeeded",
            Self::ToolFailed => "tool_failed",
            Self::Retried => "retried",
            Self::PruneCompleted => "prune_completed",
            Self::CompactionStarted => "compaction_started",
            Self::CompactionDelta => "compaction_delta",
            Self::CompactionEnded => "compaction_ended",
            Self::TokenBudgetChanged => "token_budget_changed",
            Self::ModelSwitched => "model_switched",
            Self::AgentSwitched => "agent_switched",
            Self::UserPrompted => "user_prompted",
        };
        write!(f, "{s}")
    }
}


// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- AgentFinishReason --

    #[test]
    fn test_agent_finish_reason_serde_roundtrip() {
        let reasons = [
            AgentFinishReason::Completed,
            AgentFinishReason::UserAbort,
            AgentFinishReason::MaxSteps,
            AgentFinishReason::TokenBudget,
            AgentFinishReason::Error,
            AgentFinishReason::Timeout,
        ];
        for reason in &reasons {
            let json = serde_json::to_string(reason).unwrap();
            let restored: AgentFinishReason = serde_json::from_str(&json).unwrap();
            assert_eq!(reason, &restored);
        }
    }

    #[test]
    fn test_agent_finish_reason_display() {
        assert_eq!(AgentFinishReason::Completed.to_string(), "completed");
        assert_eq!(AgentFinishReason::UserAbort.to_string(), "user_abort");
        assert_eq!(AgentFinishReason::MaxSteps.to_string(), "max_steps");
    }

    // -- CompactTrigger (from compaction module) --

    #[test]
    fn test_compact_trigger_serde() {
        for trigger in [
            CompactTrigger::Auto,
            CompactTrigger::Manual,
            CompactTrigger::Overflow,
            CompactTrigger::Idle,
        ] {
            let json = serde_json::to_string(&trigger).unwrap();
            let restored: CompactTrigger = serde_json::from_str(&json).unwrap();
            assert_eq!(trigger, restored);
        }
    }

    // -- AgentEvent serde --

    #[test]
    fn test_agent_started_serde() {
        let event = AgentEvent::AgentStarted {
            session_id: SessionId::new(),
            agent_name: "coder".into(),
            model_id: ModelId::new("gpt-4o"),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"agent_started""#));
        assert!(json.contains(r#""agent_name":"coder""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_agent_ended_serde() {
        let event = AgentEvent::AgentEnded {
            session_id: SessionId::new(),
            finish_reason: AgentFinishReason::Completed,
            total_usage: None,
            steps: 3,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"agent_ended""#));
        assert!(!json.contains("total_usage"));
        assert!(json.contains(r#""steps":3"#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_agent_ended_with_usage() {
        let event = AgentEvent::AgentEnded {
            session_id: SessionId::new(),
            finish_reason: AgentFinishReason::MaxSteps,
            total_usage: Some(Usage {
                input_tokens: 1000,
                output_tokens: 500,
                total_tokens: 1500,
                ..Default::default()
            }),
            steps: 10,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("total_usage"));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    // -- Step events --

    #[test]
    fn test_step_started_serde() {
        let event = AgentEvent::StepStarted {
            step_index: 0,
            model_id: ModelId::new("claude-sonnet-4-20250514"),
            agent_name: "default".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"step_started""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_step_ended_serde() {
        let event = AgentEvent::StepEnded {
            step_index: 0,
            finish_reason: FinishReason::Stop,
            usage: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"step_ended""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_step_failed_serde() {
        let event = AgentEvent::StepFailed {
            step_index: 2,
            error: "rate limit exceeded".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"step_failed""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    // -- Text events --

    #[test]
    fn test_text_lifecycle_serde() {
        let started = AgentEvent::TextStarted { content_index: 0 };
        let delta = AgentEvent::TextDelta {
            content_index: 0,
            delta: "Hello ".into(),
        };
        let ended = AgentEvent::TextEnded {
            content_index: 0,
            text: "Hello world".into(),
        };
        for event in [&started, &delta, &ended] {
            let json = serde_json::to_string(event).unwrap();
            let restored: AgentEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(event, &restored);
        }
    }

    // -- Reasoning events --

    #[test]
    fn test_reasoning_lifecycle_serde() {
        let started = AgentEvent::ReasoningStarted { content_index: 1 };
        let delta = AgentEvent::ReasoningDelta {
            content_index: 1,
            delta: "Let me think...".into(),
        };
        let ended = AgentEvent::ReasoningEnded {
            content_index: 1,
            text: "Let me think about this carefully.".into(),
        };
        for event in [&started, &delta, &ended] {
            let json = serde_json::to_string(event).unwrap();
            let restored: AgentEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(event, &restored);
        }
    }

    // -- Tool events --

    #[test]
    fn test_tool_input_lifecycle_serde() {
        let call_id = ToolCallId::new("call_abc123");

        let started = AgentEvent::ToolInputStarted {
            call_id: call_id.clone(),
            tool_name: "read_file".into(),
        };
        let delta = AgentEvent::ToolInputDelta {
            call_id: call_id.clone(),
            delta: r#"{"path": "src/"#.into(),
        };
        let ended = AgentEvent::ToolInputEnded {
            call_id: call_id.clone(),
            arguments: json!({"path": "src/main.rs"}),
        };

        for event in [&started, &delta, &ended] {
            let json = serde_json::to_string(event).unwrap();
            let restored: AgentEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(event, &restored);
        }
    }

    #[test]
    fn test_tool_called_serde() {
        let event = AgentEvent::ToolCalled {
            call_id: ToolCallId::new("call_1"),
            tool_name: "bash".into(),
            arguments: json!({"command": "ls -la"}),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"tool_called""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_tool_progress_serde() {
        let event = AgentEvent::ToolProgress {
            call_id: ToolCallId::new("call_1"),
            message: "Reading file...".into(),
            data: Some(json!({"bytes_read": 1024})),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"tool_progress""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_tool_progress_no_data_serde() {
        let event = AgentEvent::ToolProgress {
            call_id: ToolCallId::new("call_1"),
            message: "Working...".into(),
            data: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains(r#""data""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_tool_succeeded_serde() {
        let event = AgentEvent::ToolSucceeded {
            call_id: ToolCallId::new("call_1"),
            tool_name: "read_file".into(),
            output: ToolOutput::success("file contents here"),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"tool_succeeded""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_tool_failed_serde() {
        let event = AgentEvent::ToolFailed {
            call_id: ToolCallId::new("call_2"),
            tool_name: "write_file".into(),
            error: "permission denied".into(),
            is_retryable: false,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"tool_failed""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_tool_failed_retryable_default() {
        let json = r#"{
            "type":"tool_failed",
            "call_id":"call_x",
            "tool_name":"bash",
            "error":"timeout"
        }"#;
        let event: AgentEvent = serde_json::from_str(json).unwrap();
        if let AgentEvent::ToolFailed { is_retryable, .. } = event {
            assert!(!is_retryable);
        } else {
            panic!("expected ToolFailed");
        }
    }

    // -- Retry --

    #[test]
    fn test_retried_serde() {
        let event = AgentEvent::Retried {
            attempt: 2,
            error: "rate limit".into(),
            delay_ms: 5000,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"retried""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    // -- Compaction --

    #[test]
    fn test_prune_completed_serde() {
        let event = AgentEvent::PruneCompleted {
            tokens_freed: 25_000,
            parts_pruned: 12,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"prune_completed""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_compaction_lifecycle_serde() {
        let started = AgentEvent::CompactionStarted {
            trigger: CompactTrigger::Auto,
            strategy: CompactionStrategy::Summarize,
            tokens_before: 150_000,
        };
        let delta = AgentEvent::CompactionDelta {
            delta: "Summary: ...".into(),
        };
        let ended = AgentEvent::CompactionEnded {
            result: CompactionResult {
                summary: "The user asked about Rust error handling...".into(),
                short_summary: Some("Rust error handling".into()),
                trigger: CompactTrigger::Auto,
                tokens_before: 150_000,
                tokens_after: Some(5_000),
                messages_compacted: 40,
                messages_kept: 8,
                success: true,
            },
        };
        for event in [&started, &delta, &ended] {
            let json = serde_json::to_string(event).unwrap();
            let restored: AgentEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(event, &restored);
        }
    }

    #[test]
    fn test_token_budget_changed_serde() {
        let event = AgentEvent::TokenBudgetChanged {
            used_tokens: 170_000,
            context_window: 200_000,
            state: TokenBudgetState::Warning {
                percent_remaining: 0.15,
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"token_budget_changed""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    // -- State --

    #[test]
    fn test_model_switched_serde() {
        let event = AgentEvent::ModelSwitched {
            from: Some(ModelId::new("gpt-4o")),
            to: ModelId::new("claude-sonnet-4-20250514"),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"model_switched""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_model_switched_no_from_serde() {
        let event = AgentEvent::ModelSwitched {
            from: None,
            to: ModelId::new("gpt-4o"),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains(r#""from""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_agent_switched_serde() {
        let event = AgentEvent::AgentSwitched {
            from_agent: None,
            to_agent: AgentId::new(),
            agent_name: "code-reviewer".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"agent_switched""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_user_prompted_serde() {
        let event = AgentEvent::UserPrompted {
            content_preview: "Fix the bug in main.rs".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"user_prompted""#));
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    // -- AgentEventKind --

    #[test]
    fn test_agent_event_kind_display() {
        assert_eq!(AgentEventKind::AgentStarted.to_string(), "agent_started");
        assert_eq!(AgentEventKind::ToolCalled.to_string(), "tool_called");
        assert_eq!(
            AgentEventKind::CompactionEnded.to_string(),
            "compaction_ended"
        );
    }


    // -- Helper methods --

    #[test]
    fn test_kind_method() {
        let event = AgentEvent::TextDelta {
            content_index: 0,
            delta: "hi".into(),
        };
        assert_eq!(event.kind(), AgentEventKind::TextDelta);

        let event = AgentEvent::ToolCalled {
            call_id: ToolCallId::new("c"),
            tool_name: "t".into(),
            arguments: json!({}),
        };
        assert_eq!(event.kind(), AgentEventKind::ToolCalled);
    }

    #[test]
    fn test_is_terminal() {
        let ended = AgentEvent::AgentEnded {
            session_id: SessionId::new(),
            finish_reason: AgentFinishReason::Completed,
            total_usage: None,
            steps: 1,
        };
        assert!(ended.is_terminal());

        let started = AgentEvent::AgentStarted {
            session_id: SessionId::new(),
            agent_name: "a".into(),
            model_id: ModelId::new("m"),
        };
        assert!(!started.is_terminal());
    }

    #[test]
    fn test_is_text_delta() {
        let delta = AgentEvent::TextDelta {
            content_index: 0,
            delta: "hello".into(),
        };
        assert!(delta.is_text_delta());
        assert_eq!(delta.as_text_delta(), Some("hello"));

        let other = AgentEvent::ReasoningDelta {
            content_index: 0,
            delta: "think".into(),
        };
        assert!(!other.is_text_delta());
        assert_eq!(other.as_text_delta(), None);
    }

    #[test]
    fn test_is_tool_event() {
        let tool = AgentEvent::ToolCalled {
            call_id: ToolCallId::new("c"),
            tool_name: "t".into(),
            arguments: json!({}),
        };
        assert!(tool.is_tool_event());
        assert_eq!(tool.tool_call_id(), Some(&ToolCallId::new("c")));

        let text = AgentEvent::TextDelta {
            content_index: 0,
            delta: "hi".into(),
        };
        assert!(!text.is_tool_event());
        assert_eq!(text.tool_call_id(), None);
    }

    #[test]
    fn test_is_delta() {
        assert!(AgentEvent::TextDelta {
            content_index: 0,
            delta: "a".into()
        }
        .is_delta());
        assert!(AgentEvent::ReasoningDelta {
            content_index: 0,
            delta: "b".into()
        }
        .is_delta());
        assert!(AgentEvent::ToolInputDelta {
            call_id: ToolCallId::new("c"),
            delta: "d".into()
        }
        .is_delta());
        assert!(AgentEvent::CompactionDelta {
            delta: "e".into()
        }
        .is_delta());

        assert!(!AgentEvent::TextStarted { content_index: 0 }.is_delta());
        assert!(!AgentEvent::ToolCalled {
            call_id: ToolCallId::new("c"),
            tool_name: "t".into(),
            arguments: json!({}),
        }
        .is_delta());
    }

    #[test]
    fn test_tool_call_id_all_tool_events() {
        let id = ToolCallId::new("test_id");
        let events = vec![
            AgentEvent::ToolInputStarted {
                call_id: id.clone(),
                tool_name: "t".into(),
            },
            AgentEvent::ToolInputDelta {
                call_id: id.clone(),
                delta: "d".into(),
            },
            AgentEvent::ToolInputEnded {
                call_id: id.clone(),
                arguments: json!({}),
            },
            AgentEvent::ToolCalled {
                call_id: id.clone(),
                tool_name: "t".into(),
                arguments: json!({}),
            },
            AgentEvent::ToolProgress {
                call_id: id.clone(),
                message: "m".into(),
                data: None,
            },
            AgentEvent::ToolSucceeded {
                call_id: id.clone(),
                tool_name: "t".into(),
                output: ToolOutput::success("ok"),
            },
            AgentEvent::ToolFailed {
                call_id: id.clone(),
                tool_name: "t".into(),
                error: "e".into(),
                is_retryable: false,
            },
        ];
        for event in &events {
            assert_eq!(
                event.tool_call_id(),
                Some(&id),
                "tool_call_id() failed for {:?}",
                event.kind()
            );
        }
    }
}
