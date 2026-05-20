//! # session
//!
//! ## 职责
//! 运行时会话状态 — 持有一次 Agent 对话的完整内存上下文。
//!
//! ## 设计
//! `Session` 是 Agent loop 的核心数据载体：
//! - **消息历史** — `Vec<Message>` 累积对话上下文
//! - **运行状态** — `SessionStatus` 跟踪 idle/running/cancelled
//! - **用量累计** — 逐步累加 `Usage`，提供费用追踪
//! - **步数看护** — `step_count` + `max_steps` 防止无限循环
//! - **上下文窗口** — token 计数 + 预算状态，驱动压缩决策
//!
//! ## 不包含
//! - 持久化（属于 `katu-app`）
//! - 工具注册表 / Hook 注册表（由 Runner 注入）
//! - 系统 prompt 组装逻辑（属于 `prompt` 模块）
//!
//! ## 调用者
//! - `katu-agent::runner` (future) — Agent loop 核心循环
//! - `katu-agent::stream_consumer` — 消费结果写入 session
//! - `katu-agent::tool_executor` — 工具结果写入 session

use chrono::{DateTime, Utc};

use katu_core::agent::AgentDefinition;
use katu_core::compaction::{CompactionConfig, TokenBudgetState};
use katu_core::message::{AssistantMessage, Message, ToolResultMessage, UserContent};
use katu_core::types::{ModelId, Role, SessionId};
use katu_core::usage::Usage;
use katu_core::CancellationToken;

// ===========================================================================
// SessionStatus
// ===========================================================================

/// 会话运行状态。
///
/// 状态转换：
/// ```text
/// Idle ──prompt()──► Running ──end_turn()──► Idle
///   │                   │
///   │                   ├──cancel()──► Cancelled
///   │                   │
///   └───────────────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    /// 空闲 — 等待用户输入。
    Idle,
    /// 运行中 — Agent loop 正在执行 LLM 调用或工具。
    Running,
    /// 已取消 — 用户或系统中断了当前执行。
    Cancelled,
}

impl SessionStatus {
    /// 是否处于运行中状态。
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }

    /// 是否已取消。
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }

    /// 是否空闲。
    pub fn is_idle(&self) -> bool {
        matches!(self, Self::Idle)
    }
}

// ===========================================================================
// Session
// ===========================================================================

/// 运行时会话 — 一次 Agent 对话的完整内存状态。
///
/// 由 `Session::new()` 创建，通过方法驱动消息追加、状态转换和用量累计。
///
/// # Examples
///
/// ```
/// use katu_agent::session::Session;
/// use katu_core::{AgentDefinition, AgentRole, ModelId};
///
/// let agent = AgentDefinition::new("build", AgentRole::Primary)
///     .with_max_steps(50);
///
/// let session = Session::new(agent, ModelId::new("gpt-4o"));
///
/// assert!(session.status().is_idle());
/// assert_eq!(session.step_count(), 0);
/// assert!(session.messages().next().is_none());
/// ```
pub struct Session {
    /// 会话唯一标识。
    id: SessionId,

    /// Agent 静态配置。
    agent: AgentDefinition,

    /// 当前使用的模型（运行时可切换）。
    model_id: ModelId,

    /// 对话历史。
    messages: Vec<Message>,

    /// 运行状态。
    status: SessionStatus,

    /// 协作式取消令牌。
    cancel_token: CancellationToken,

    /// 当前步数（一次 LLM 调用 + 工具执行 = 一步）。
    step_count: u32,

    /// 累计 token 用量。
    total_usage: Usage,

    /// 当前上下文 token 估计（外部设置）。
    context_tokens: u64,

    /// 模型 context window 大小。
    context_window: u64,

    /// 上下文压缩配置。
    compaction_config: CompactionConfig,

    /// 创建时间。
    created_at: DateTime<Utc>,

    /// 最后更新时间。
    updated_at: DateTime<Utc>,
}

impl Session {
    /// 创建新会话。
    ///
    /// # Arguments
    /// - `agent` — Agent 静态配置
    /// - `model_id` — 初始模型标识
    pub fn new(agent: AgentDefinition, model_id: ModelId) -> Self {
        let now = Utc::now();
        Self {
            id: SessionId::new(),
            agent,
            model_id,
            messages: Vec::new(),
            status: SessionStatus::Idle,
            cancel_token: CancellationToken::new(),
            step_count: 0,
            total_usage: Usage::default(),
            context_tokens: 0,
            context_window: 0,
            compaction_config: CompactionConfig::default(),
            created_at: now,
            updated_at: now,
        }
    }

    /// 设置 context window 大小（模型最大 token）。
    pub fn with_context_window(mut self, context_window: u64) -> Self {
        self.context_window = context_window;
        self
    }

    /// 设置上下文压缩配置。
    pub fn with_compaction_config(mut self, config: CompactionConfig) -> Self {
        self.compaction_config = config;
        self
    }

    /// 注入已有的取消令牌。
    pub fn with_cancel_token(mut self, token: CancellationToken) -> Self {
        self.cancel_token = token;
        self
    }
}

// ---------------------------------------------------------------------------
// Identity & Configuration
// ---------------------------------------------------------------------------

impl Session {
    /// 会话 ID。
    pub fn id(&self) -> &SessionId {
        &self.id
    }

    /// Agent 静态配置。
    pub fn agent(&self) -> &AgentDefinition {
        &self.agent
    }

    /// 当前使用的模型。
    pub fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// 切换模型。
    pub fn set_model_id(&mut self, model_id: ModelId) {
        self.model_id = model_id;
        self.touch();
    }
}

// ---------------------------------------------------------------------------
// Message History
// ---------------------------------------------------------------------------

impl Session {
    /// 消息迭代器。
    pub fn messages(&self) -> impl Iterator<Item = &Message> {
        self.messages.iter()
    }

    /// 消息总数。
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// 最后一条消息的角色。
    pub fn last_role(&self) -> Option<Role> {
        self.messages.last().map(|m| m.role())
    }

    /// 最后一条 Assistant 消息。
    pub fn last_assistant(&self) -> Option<&AssistantMessage> {
        self.messages.iter().rev().find_map(|m| match m {
            Message::Assistant(a) => Some(a),
            _ => None,
        })
    }

    /// 最后一条 Assistant 消息是否包含工具调用。
    pub fn has_pending_tool_calls(&self) -> bool {
        self.last_assistant()
            .map(|a| a.has_tool_calls())
            .unwrap_or(false)
    }

    /// 追加用户消息。
    pub fn push_user(&mut self, content: impl Into<UserContent>) {
        self.messages.push(Message::user(content));
        self.touch();
    }

    /// 追加 Assistant 消息并累加用量。
    pub fn push_assistant(&mut self, message: AssistantMessage) {
        if let Some(usage) = &message.usage {
            self.accumulate_usage(usage);
        }
        self.messages.push(Message::Assistant(message));
        self.touch();
    }

    /// 批量追加工具结果消息。
    pub fn push_tool_results(&mut self, results: Vec<ToolResultMessage>) {
        for result in results {
            self.messages.push(Message::ToolResult(result));
        }
        self.touch();
    }

    /// 追加任意消息（用于恢复等场景）。
    pub fn push_message(&mut self, message: Message) {
        if let Message::Assistant(ref a) = message {
            if let Some(usage) = &a.usage {
                self.accumulate_usage(usage);
            }
        }
        self.messages.push(message);
        self.touch();
    }

    /// 替换全部消息（用于压缩后重建上下文）。
    pub fn replace_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
        self.touch();
    }

    /// 消息切片引用。
    pub fn message_slice(&self) -> &[Message] {
        &self.messages
    }
}

// ---------------------------------------------------------------------------
// Running State
// ---------------------------------------------------------------------------

impl Session {
    /// 当前运行状态。
    pub fn status(&self) -> SessionStatus {
        self.status
    }

    /// 取消令牌引用（共享给 StreamConsumer / ToolExecutor）。
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// 标记进入运行状态。
    ///
    /// # Panics
    /// 如果当前不是 Idle 状态则 panic（调用方应先检查）。
    pub fn begin_run(&mut self) {
        assert!(
            self.status.is_idle(),
            "cannot begin run: session is {:?}",
            self.status
        );
        self.status = SessionStatus::Running;
        // 重置取消令牌（前一次可能已取消）
        self.cancel_token = CancellationToken::new();
        self.touch();
    }

    /// 标记一轮结束，回到空闲状态。
    pub fn end_run(&mut self) {
        self.status = SessionStatus::Idle;
        self.touch();
    }

    /// 取消当前执行。
    pub fn cancel(&mut self) {
        if self.status.is_running() {
            self.cancel_token.cancel();
            self.status = SessionStatus::Cancelled;
            self.touch();
        }
    }

    /// 从取消状态恢复为空闲（准备下一轮）。
    pub fn reset_after_cancel(&mut self) {
        if self.status.is_cancelled() {
            self.status = SessionStatus::Idle;
            self.cancel_token = CancellationToken::new();
            self.touch();
        }
    }
}

// ---------------------------------------------------------------------------
// Step Tracking
// ---------------------------------------------------------------------------

impl Session {
    /// 当前步数。
    pub fn step_count(&self) -> u32 {
        self.step_count
    }

    /// 递增步数，返回新步数。
    pub fn increment_step(&mut self) -> u32 {
        self.step_count += 1;
        self.step_count
    }

    /// 最大步数限制（来自 AgentDefinition 或默认值）。
    pub fn max_steps(&self) -> u32 {
        self.agent.max_steps.unwrap_or(DEFAULT_MAX_STEPS)
    }

    /// 是否已达到步数上限。
    pub fn is_over_step_limit(&self) -> bool {
        self.step_count >= self.max_steps()
    }
}

/// 默认最大步数。
const DEFAULT_MAX_STEPS: u32 = 100;

// ---------------------------------------------------------------------------
// Usage Tracking
// ---------------------------------------------------------------------------

impl Session {
    /// 累计 token 用量。
    pub fn total_usage(&self) -> &Usage {
        &self.total_usage
    }

    /// 累加一次 LLM 调用的用量。
    fn accumulate_usage(&mut self, usage: &Usage) {
        self.total_usage.input_tokens += usage.input_tokens;
        self.total_usage.output_tokens += usage.output_tokens;
        self.total_usage.cache_read_tokens += usage.cache_read_tokens;
        self.total_usage.cache_write_tokens += usage.cache_write_tokens;
        self.total_usage.total_tokens += usage.total_tokens;
        if let Some(r) = usage.reasoning_tokens {
            *self.total_usage.reasoning_tokens.get_or_insert(0) += r;
        }
        if let Some(cost) = &usage.cost {
            let total_cost = self.total_usage.cost.get_or_insert(katu_core::usage::Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            });
            total_cost.input += cost.input;
            total_cost.output += cost.output;
            total_cost.cache_read += cost.cache_read;
            total_cost.cache_write += cost.cache_write;
            total_cost.total += cost.total;
        }
    }
}

// ---------------------------------------------------------------------------
// Context Window
// ---------------------------------------------------------------------------

impl Session {
    /// 当前上下文 token 估计。
    pub fn context_tokens(&self) -> u64 {
        self.context_tokens
    }

    /// 模型 context window 大小。
    pub fn context_window(&self) -> u64 {
        self.context_window
    }

    /// 压缩配置。
    pub fn compaction_config(&self) -> &CompactionConfig {
        &self.compaction_config
    }

    /// 更新上下文 token 计数（由外部 token 计数器设置）。
    pub fn set_context_tokens(&mut self, tokens: u64) {
        self.context_tokens = tokens;
    }

    /// 计算当前 token 预算状态。
    pub fn budget_state(&self) -> TokenBudgetState {
        TokenBudgetState::from_usage(
            self.context_tokens,
            self.context_window,
            self.compaction_config.reserve_tokens as u64,
        )
    }

    /// 是否应触发自动压缩。
    pub fn should_compact(&self) -> bool {
        self.compaction_config.auto_enabled && self.budget_state().should_auto_compact()
    }
}

// ---------------------------------------------------------------------------
// Timestamps
// ---------------------------------------------------------------------------

impl Session {
    /// 创建时间。
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// 最后更新时间。
    pub fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }

    /// 更新时间戳。
    fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use katu_core::agent::AgentRole;
    use katu_core::message::AssistantBlock;
    use katu_core::types::{FinishReason, MessageId, ToolCallId};
    use katu_core::usage::Cost;

    fn test_agent() -> AgentDefinition {
        AgentDefinition::new("test", AgentRole::Primary)
            .with_max_steps(10)
    }

    fn test_session() -> Session {
        Session::new(test_agent(), ModelId::new("gpt-4o"))
            .with_context_window(200_000)
    }

    fn make_assistant(text: &str, usage: Option<Usage>) -> AssistantMessage {
        AssistantMessage {
            id: MessageId::new(),
            content: vec![AssistantBlock::Text { text: text.into() }],
            model: "gpt-4o".into(),
            provider: "openai".into(),
            finish_reason: FinishReason::Stop,
            usage,
            timestamp: Utc::now(),
        }
    }

    fn make_assistant_with_tool_call() -> AssistantMessage {
        AssistantMessage {
            id: MessageId::new(),
            content: vec![
                AssistantBlock::Text { text: "reading file".into() },
                AssistantBlock::ToolCall {
                    id: ToolCallId::new("call_1"),
                    name: "read_file".into(),
                    arguments: serde_json::json!({"path": "src/main.rs"}),
                },
            ],
            model: "gpt-4o".into(),
            provider: "openai".into(),
            finish_reason: FinishReason::ToolCalls,
            usage: Some(Usage {
                input_tokens: 100,
                output_tokens: 50,
                total_tokens: 150,
                ..Default::default()
            }),
            timestamp: Utc::now(),
        }
    }

    // -- 创建 --

    #[test]
    fn test_new_session() {
        let session = test_session();
        assert!(session.status().is_idle());
        assert_eq!(session.step_count(), 0);
        assert_eq!(session.message_count(), 0);
        assert!(session.messages().next().is_none());
        assert_eq!(session.model_id().as_str(), "gpt-4o");
        assert_eq!(session.agent().name.as_str(), "test");
        assert_eq!(session.context_window(), 200_000);
    }

    #[test]
    fn test_session_id_unique() {
        let s1 = test_session();
        let s2 = test_session();
        assert_ne!(s1.id(), s2.id());
    }

    // -- 消息历史 --

    #[test]
    fn test_push_user() {
        let mut session = test_session();
        session.push_user("hello");
        assert_eq!(session.message_count(), 1);
        assert_eq!(session.last_role(), Some(Role::User));
    }

    #[test]
    fn test_push_assistant() {
        let mut session = test_session();
        session.push_user("hi");
        session.push_assistant(make_assistant("hello!", None));
        assert_eq!(session.message_count(), 2);
        assert_eq!(session.last_role(), Some(Role::Assistant));
    }

    #[test]
    fn test_push_tool_results() {
        let mut session = test_session();
        session.push_user("read file");
        session.push_assistant(make_assistant_with_tool_call());

        let results = vec![ToolResultMessage {
            id: MessageId::new(),
            tool_call_id: ToolCallId::new("call_1"),
            tool_name: "read_file".into(),
            content: vec![katu_core::message::ContentBlock::Text {
                text: "file contents".into(),
            }],
            is_error: false,
            timestamp: Utc::now(),
        }];
        session.push_tool_results(results);
        assert_eq!(session.message_count(), 3);
        assert_eq!(session.last_role(), Some(Role::Tool));
    }

    #[test]
    fn test_has_pending_tool_calls() {
        let mut session = test_session();
        assert!(!session.has_pending_tool_calls());

        session.push_assistant(make_assistant_with_tool_call());
        assert!(session.has_pending_tool_calls());

        session.push_assistant(make_assistant("done", None));
        assert!(!session.has_pending_tool_calls());
    }

    #[test]
    fn test_last_assistant() {
        let mut session = test_session();
        assert!(session.last_assistant().is_none());

        session.push_assistant(make_assistant("first", None));
        session.push_user("next");
        session.push_assistant(make_assistant("second", None));

        let last = session.last_assistant().unwrap();
        assert_eq!(last.text(), "second");
    }

    #[test]
    fn test_replace_messages() {
        let mut session = test_session();
        session.push_user("one");
        session.push_user("two");
        assert_eq!(session.message_count(), 2);

        session.replace_messages(vec![Message::user("compacted")]);
        assert_eq!(session.message_count(), 1);
    }

    #[test]
    fn test_message_slice() {
        let mut session = test_session();
        session.push_user("a");
        session.push_user("b");
        let slice = session.message_slice();
        assert_eq!(slice.len(), 2);
    }

    // -- 运行状态 --

    #[test]
    fn test_status_lifecycle() {
        let mut session = test_session();

        assert!(session.status().is_idle());
        session.begin_run();
        assert!(session.status().is_running());
        session.end_run();
        assert!(session.status().is_idle());
    }

    #[test]
    fn test_cancel() {
        let mut session = test_session();
        session.begin_run();

        let token = session.cancel_token().clone();
        assert!(!token.is_cancelled());

        session.cancel();
        assert!(session.status().is_cancelled());
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancel_idle_is_noop() {
        let mut session = test_session();
        session.cancel();
        assert!(session.status().is_idle());
    }

    #[test]
    fn test_reset_after_cancel() {
        let mut session = test_session();
        session.begin_run();
        session.cancel();
        assert!(session.status().is_cancelled());

        session.reset_after_cancel();
        assert!(session.status().is_idle());
        assert!(!session.cancel_token().is_cancelled());
    }

    #[test]
    fn test_begin_run_resets_cancel_token() {
        let mut session = test_session();
        session.begin_run();
        session.cancel();
        session.reset_after_cancel();

        session.begin_run();
        assert!(!session.cancel_token().is_cancelled());
    }

    #[test]
    #[should_panic(expected = "cannot begin run")]
    fn test_begin_run_while_running_panics() {
        let mut session = test_session();
        session.begin_run();
        session.begin_run();
    }

    // -- 步数追踪 --

    #[test]
    fn test_step_tracking() {
        let mut session = test_session();
        assert_eq!(session.step_count(), 0);
        assert_eq!(session.max_steps(), 10);
        assert!(!session.is_over_step_limit());

        for i in 1..=10 {
            let step = session.increment_step();
            assert_eq!(step, i);
        }
        assert!(session.is_over_step_limit());
    }

    #[test]
    fn test_default_max_steps() {
        let agent = AgentDefinition::new("no_limit", AgentRole::Primary);
        let session = Session::new(agent, ModelId::new("gpt-4o"));
        assert_eq!(session.max_steps(), DEFAULT_MAX_STEPS);
    }

    // -- 用量追踪 --

    #[test]
    fn test_usage_accumulation() {
        let mut session = test_session();

        let usage1 = Usage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: 20,
            cache_write_tokens: 10,
            reasoning_tokens: Some(5),
            total_tokens: 150,
            cost: None,
        };
        session.push_assistant(make_assistant("a", Some(usage1)));

        let usage2 = Usage {
            input_tokens: 200,
            output_tokens: 80,
            cache_read_tokens: 30,
            cache_write_tokens: 0,
            reasoning_tokens: Some(10),
            total_tokens: 280,
            cost: None,
        };
        session.push_assistant(make_assistant("b", Some(usage2)));

        let total = session.total_usage();
        assert_eq!(total.input_tokens, 300);
        assert_eq!(total.output_tokens, 130);
        assert_eq!(total.cache_read_tokens, 50);
        assert_eq!(total.cache_write_tokens, 10);
        assert_eq!(total.reasoning_tokens, Some(15));
        assert_eq!(total.total_tokens, 430);
    }

    #[test]
    fn test_usage_accumulation_with_cost() {
        let mut session = test_session();

        let usage = Usage {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            cost: Some(Cost {
                input: 0.01,
                output: 0.03,
                cache_read: 0.001,
                cache_write: 0.002,
                total: 0.043,
            }),
            ..Default::default()
        };
        session.push_assistant(make_assistant("a", Some(usage)));

        let total_cost = session.total_usage().cost.as_ref().unwrap();
        assert!((total_cost.total - 0.043).abs() < f64::EPSILON);
    }

    #[test]
    fn test_push_message_accumulates_assistant_usage() {
        let mut session = test_session();
        let msg = Message::Assistant(make_assistant("x", Some(Usage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        })));
        session.push_message(msg);
        assert_eq!(session.total_usage().input_tokens, 10);
    }

    #[test]
    fn test_push_message_no_usage_for_user() {
        let mut session = test_session();
        session.push_message(Message::user("hello"));
        assert_eq!(session.total_usage().input_tokens, 0);
    }

    // -- 上下文窗口 --

    #[test]
    fn test_context_window() {
        let mut session = test_session();
        session.set_context_tokens(150_000);
        assert_eq!(session.context_tokens(), 150_000);
    }

    #[test]
    fn test_budget_state_normal() {
        let mut session = test_session();
        session.set_context_tokens(50_000);
        let state = session.budget_state();
        assert!(matches!(state, TokenBudgetState::Normal { .. }));
    }

    #[test]
    fn test_should_compact() {
        let mut session = Session::new(test_agent(), ModelId::new("gpt-4o"))
            .with_context_window(200_000)
            .with_compaction_config(CompactionConfig::default());

        // 正常范围 — 不需要压缩
        session.set_context_tokens(50_000);
        assert!(!session.should_compact());

        // 超过阈值 — 需要压缩
        session.set_context_tokens(195_000);
        assert!(session.should_compact());
    }

    #[test]
    fn test_should_compact_disabled() {
        let config = CompactionConfig::default().with_auto_enabled(false);
        let mut session = Session::new(test_agent(), ModelId::new("gpt-4o"))
            .with_context_window(200_000)
            .with_compaction_config(config);

        session.set_context_tokens(195_000);
        assert!(!session.should_compact());
    }

    // -- 模型切换 --

    #[test]
    fn test_set_model_id() {
        let mut session = test_session();
        assert_eq!(session.model_id().as_str(), "gpt-4o");
        session.set_model_id(ModelId::new("claude-sonnet-4-20250514"));
        assert_eq!(session.model_id().as_str(), "claude-sonnet-4-20250514");
    }

    // -- 时间戳 --

    #[test]
    fn test_timestamps() {
        let before = Utc::now();
        let session = test_session();
        let after = Utc::now();

        assert!(session.created_at() >= before);
        assert!(session.created_at() <= after);
        assert!(session.updated_at() >= before);
    }

    #[test]
    fn test_touch_updates_timestamp() {
        let mut session = test_session();
        let initial = session.updated_at();

        // 简短延迟确保时间戳变化
        std::thread::sleep(std::time::Duration::from_millis(2));
        session.push_user("trigger touch");

        assert!(session.updated_at() > initial);
    }
}
