//! # katu_core::hook
//!
//! ## 职责
//! 定义 Hook 系统的类型与执行契约 — Agent 生命周期的可拦截节点。
//!
//! ## 设计
//! Hook 系统与 `AgentEvent` 互补：
//! - `AgentEvent`（agent_event.rs）= 不可变的**已发生事实**，消费者只能观察
//! - `Hook`（本模块）= 可拦截的**执行节点**，Hook 可以拦截、修改、阻止
//!
//! ```text
//! Agent Loop ──节点──► HookRegistry.run() ──决策──► 继续/修改/阻止
//!                                                        │
//!                                                        ▼
//!                                              AgentEvent（记录事实）
//! ```
//!
//! ## 对外接口
//! - `HookEvent` — 可拦截的生命周期节点（10 种）
//! - `HookInput` — 各事件的类型化输入
//! - `HookOutput` — Hook 的决策结果
//! - `HookPermission` — 权限决策（allow / deny / ask）
//! - `Hook` — 执行 trait（async, object-safe）
//! - `HookSource` — Hook 来源标识
//! - `HookRegistry` — 注册与匹配
//! - `AggregatedHookOutput` — 多 Hook 结果聚合
//!
//! ## 调用者
//! - `katu-agent` (future) — Agent loop 在各生命周期节点调用 Hook
//! - 应用层 — 注册自定义 Hook 实现

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::tool::ToolOutput;
use crate::types::{SessionId, ToolCallId};

// ===========================================================================
// HookEvent
// ===========================================================================

/// 可拦截的 Agent 生命周期节点。
///
/// 与 `AgentEvent`（28 种只读观察事件）互补，`HookEvent` 只覆盖
/// **需要干预能力**的节点 — 拦截、修改输入/输出、权限决策。
///
/// # Examples
///
/// ```
/// use katu_core::hook::HookEvent;
///
/// let event = HookEvent::PreToolUse;
/// assert!(event.is_tool_event());
/// assert!(!HookEvent::SessionStart.is_tool_event());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEvent {
    // ── Tool 生命周期 ─────────────────────────────────────

    /// 工具执行前 — 可以 allow/deny/ask、修改 input。
    PreToolUse,
    /// 工具执行成功后 — 可以注入上下文、修改输出。
    PostToolUse,
    /// 工具执行失败后 — 可以注入诊断上下文。
    PostToolFailure,

    // ── 用户交互 ──────────────────────────────────────────

    /// 用户提交 prompt 前 — 可以注入上下文或拦截。
    UserPromptSubmit,

    // ── Session 生命周期 ──────────────────────────────────

    /// Session 开始。
    SessionStart,
    /// Session 结束。
    SessionEnd,

    // ── Agent 生命周期 ────────────────────────────────────

    /// Agent loop 单步结束判定 — 可以阻止停止、要求继续。
    Stop,
    /// SubAgent 启动前。
    SubAgentStart,

    // ── Compaction ────────────────────────────────────────

    /// 上下文压缩前。
    PreCompact,
    /// 上下文压缩后。
    PostCompact,
}

/// 所有 Hook 事件的完整列表，按定义顺序。
pub const ALL_HOOK_EVENTS: &[HookEvent] = &[
    HookEvent::PreToolUse,
    HookEvent::PostToolUse,
    HookEvent::PostToolFailure,
    HookEvent::UserPromptSubmit,
    HookEvent::SessionStart,
    HookEvent::SessionEnd,
    HookEvent::Stop,
    HookEvent::SubAgentStart,
    HookEvent::PreCompact,
    HookEvent::PostCompact,
];

impl HookEvent {
    /// 是否为 Tool 相关事件。
    pub fn is_tool_event(&self) -> bool {
        matches!(
            self,
            Self::PreToolUse | Self::PostToolUse | Self::PostToolFailure
        )
    }
}

impl std::fmt::Display for HookEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::PreToolUse => "pre_tool_use",
            Self::PostToolUse => "post_tool_use",
            Self::PostToolFailure => "post_tool_failure",
            Self::UserPromptSubmit => "user_prompt_submit",
            Self::SessionStart => "session_start",
            Self::SessionEnd => "session_end",
            Self::Stop => "stop",
            Self::SubAgentStart => "sub_agent_start",
            Self::PreCompact => "pre_compact",
            Self::PostCompact => "post_compact",
        };
        f.write_str(s)
    }
}

// ===========================================================================
// HookInput
// ===========================================================================

/// Hook 输入 — 每个 HookEvent 携带的上下文数据。
///
/// 使用 enum 确保类型安全，Hook 实现方通过 match 获取精确类型。
/// 所有变体均可序列化，支持跨进程 Hook（如 shell command hook）。
///
/// # Examples
///
/// ```
/// use katu_core::hook::{HookEvent, HookInput};
/// use katu_core::ToolCallId;
/// use serde_json::json;
///
/// let input = HookInput::PreToolUse {
///     tool_name: "bash".into(),
///     tool_input: json!({"command": "ls -la"}),
///     call_id: ToolCallId::new("call_1"),
/// };
/// assert_eq!(input.event(), HookEvent::PreToolUse);
/// assert_eq!(input.tool_name(), Some("bash"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "hook_event", rename_all = "snake_case")]
pub enum HookInput {
    /// 工具执行前。
    PreToolUse {
        tool_name: String,
        tool_input: serde_json::Value,
        call_id: ToolCallId,
    },

    /// 工具执行成功后。
    PostToolUse {
        tool_name: String,
        tool_input: serde_json::Value,
        tool_output: ToolOutput,
        call_id: ToolCallId,
    },

    /// 工具执行失败后。
    PostToolFailure {
        tool_name: String,
        tool_input: serde_json::Value,
        error: String,
        call_id: ToolCallId,
    },

    /// 用户提交 prompt 前。
    UserPromptSubmit {
        prompt: String,
    },

    /// Session 开始。
    SessionStart {
        session_id: SessionId,
    },

    /// Session 结束。
    SessionEnd {
        session_id: SessionId,
        reason: String,
    },

    /// Agent loop 单步结束判定。
    Stop {
        finish_reason: String,
    },

    /// SubAgent 启动前。
    SubAgentStart {
        agent_name: String,
    },

    /// 上下文压缩前。
    PreCompact {
        trigger: String,
        tokens_before: u64,
    },

    /// 上下文压缩后。
    PostCompact {
        trigger: String,
        tokens_after: u64,
    },
}

impl HookInput {
    /// 返回此输入对应的事件类型。
    pub fn event(&self) -> HookEvent {
        match self {
            Self::PreToolUse { .. } => HookEvent::PreToolUse,
            Self::PostToolUse { .. } => HookEvent::PostToolUse,
            Self::PostToolFailure { .. } => HookEvent::PostToolFailure,
            Self::UserPromptSubmit { .. } => HookEvent::UserPromptSubmit,
            Self::SessionStart { .. } => HookEvent::SessionStart,
            Self::SessionEnd { .. } => HookEvent::SessionEnd,
            Self::Stop { .. } => HookEvent::Stop,
            Self::SubAgentStart { .. } => HookEvent::SubAgentStart,
            Self::PreCompact { .. } => HookEvent::PreCompact,
            Self::PostCompact { .. } => HookEvent::PostCompact,
        }
    }

    /// 如果是 Tool 相关事件，返回工具名。
    pub fn tool_name(&self) -> Option<&str> {
        match self {
            Self::PreToolUse { tool_name, .. }
            | Self::PostToolUse { tool_name, .. }
            | Self::PostToolFailure { tool_name, .. } => Some(tool_name.as_str()),
            _ => None,
        }
    }

    /// 如果是 Tool 相关事件，返回 call_id。
    pub fn call_id(&self) -> Option<&ToolCallId> {
        match self {
            Self::PreToolUse { call_id, .. }
            | Self::PostToolUse { call_id, .. }
            | Self::PostToolFailure { call_id, .. } => Some(call_id),
            _ => None,
        }
    }
}

// ===========================================================================
// HookPermission
// ===========================================================================

/// Hook 的权限决策 — 仅对 `PreToolUse` 事件有意义。
///
/// 多个 Hook 的权限按严格度聚合：`Deny > Ask > Allow`。
///
/// # 重要
/// Hook 返回 `Allow` **不能**绕过 settings 中的 deny 规则。
/// Agent loop 应在 Hook 决策之后再检查规则级权限。
///
/// # Examples
///
/// ```
/// use katu_core::hook::HookPermission;
///
/// let deny = HookPermission::Deny { reason: Some("unsafe command".into()) };
/// let allow = HookPermission::Allow;
/// assert!(deny.is_deny());
/// assert!(!allow.is_deny());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HookPermission {
    /// 允许执行（可被 settings deny 规则覆盖）。
    Allow,

    /// 拒绝执行。
    Deny {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },

    /// 需要用户确认。
    Ask {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
}

impl HookPermission {
    /// 创建无原因的 Deny。
    pub fn deny() -> Self {
        Self::Deny { reason: None }
    }

    /// 创建带原因的 Deny。
    pub fn deny_with_reason(reason: impl Into<String>) -> Self {
        Self::Deny {
            reason: Some(reason.into()),
        }
    }

    /// 创建无消息的 Ask。
    pub fn ask() -> Self {
        Self::Ask { message: None }
    }

    /// 创建带消息的 Ask。
    pub fn ask_with_message(message: impl Into<String>) -> Self {
        Self::Ask {
            message: Some(message.into()),
        }
    }

    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }

    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Deny { .. })
    }

    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask { .. })
    }

    /// 返回严格度数值 — 用于聚合时比较优先级。
    ///
    /// Deny(2) > Ask(1) > Allow(0)。
    fn strictness(&self) -> u8 {
        match self {
            Self::Allow => 0,
            Self::Ask { .. } => 1,
            Self::Deny { .. } => 2,
        }
    }
}

// ===========================================================================
// HookOutput
// ===========================================================================

/// Hook 的执行结果 — 告诉 Agent loop 如何继续。
///
/// ## 设计要点
/// - `Default` = 无操作（passthrough），不影响正常流程
/// - 多个字段独立，可同时设置（如 allow + 注入 context）
/// - 权限聚合优先级由 `AggregatedHookOutput` 处理
///
/// # Examples
///
/// ```
/// use katu_core::hook::HookOutput;
///
/// // passthrough — 什么都不做
/// let out = HookOutput::passthrough();
/// assert!(!out.has_decision());
///
/// // deny + context
/// let out = HookOutput::deny("dangerous")
///     .with_context("This command modifies system files");
/// assert!(out.has_decision());
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HookOutput {
    /// 权限决策 — 仅对 PreToolUse 有意义。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission: Option<HookPermission>,

    /// 修改后的工具输入 — PreToolUse 时替换原始 input。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_input: Option<serde_json::Value>,

    /// 修改后的工具输出 — PostToolUse 时替换原始 output。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_output: Option<ToolOutput>,

    /// 注入给 LLM 的额外上下文。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_context: Vec<String>,

    /// 是否阻止 Agent 继续执行。
    #[serde(default)]
    pub prevent_continuation: bool,

    /// 阻止原因（`prevent_continuation = true` 时展示给用户）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,

    /// 阻塞性错误 — 反馈给 model 的错误消息。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocking_error: Option<String>,

    /// 系统消息 — 展示给用户的提示/警告。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_message: Option<String>,
}

impl HookOutput {
    /// 无操作 — 不影响正常流程。
    pub fn passthrough() -> Self {
        Self::default()
    }

    /// 允许执行。
    pub fn allow() -> Self {
        Self {
            permission: Some(HookPermission::Allow),
            ..Default::default()
        }
    }

    /// 拒绝执行。
    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            permission: Some(HookPermission::deny_with_reason(reason)),
            ..Default::default()
        }
    }

    /// 需要用户确认。
    pub fn ask(message: impl Into<String>) -> Self {
        Self {
            permission: Some(HookPermission::ask_with_message(message)),
            ..Default::default()
        }
    }

    /// 设置修改后的工具输入（builder 模式）。
    pub fn with_updated_input(mut self, input: serde_json::Value) -> Self {
        self.updated_input = Some(input);
        self
    }

    /// 设置修改后的工具输出（builder 模式）。
    pub fn with_updated_output(mut self, output: ToolOutput) -> Self {
        self.updated_output = Some(output);
        self
    }

    /// 追加额外上下文（builder 模式）。
    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.additional_context.push(ctx.into());
        self
    }

    /// 阻止 Agent 继续执行（builder 模式）。
    pub fn with_stop(mut self, reason: impl Into<String>) -> Self {
        self.prevent_continuation = true;
        self.stop_reason = Some(reason.into());
        self
    }

    /// 设置阻塞性错误（builder 模式）。
    pub fn with_blocking_error(mut self, error: impl Into<String>) -> Self {
        self.blocking_error = Some(error.into());
        self
    }

    /// 设置系统消息（builder 模式）。
    pub fn with_system_message(mut self, message: impl Into<String>) -> Self {
        self.system_message = Some(message.into());
        self
    }

    /// 是否做出了实质性决策（非 passthrough）。
    pub fn has_decision(&self) -> bool {
        self.permission.is_some()
            || self.updated_input.is_some()
            || self.updated_output.is_some()
            || !self.additional_context.is_empty()
            || self.prevent_continuation
            || self.blocking_error.is_some()
    }
}

// ===========================================================================
// Hook trait
// ===========================================================================

/// Hook 执行 trait — 所有可注册到 Agent loop 的 Hook 必须实现。
///
/// ## 设计选择
/// - **`on_event`** — 处理所有事件类型，Hook 自行 match 感兴趣的事件
/// - **`HookOutput`** — 默认 passthrough，显式 opt-in 干预
/// - **async** — 支持异步操作（网络请求、LLM 查询等）
/// - **`&self`** — 无状态偏好；需要状态的 Hook 使用内部可变性
///
/// ## Object Safety
/// 通过 `#[async_trait]` 实现 dyn dispatch，支持 `Arc<dyn Hook>` 存储。
///
/// # Examples
///
/// ```
/// use async_trait::async_trait;
/// use katu_core::hook::*;
///
/// struct DangerousCommandBlocker;
///
/// #[async_trait]
/// impl Hook for DangerousCommandBlocker {
///     fn name(&self) -> &str { "dangerous_command_blocker" }
///
///     fn events(&self) -> &[HookEvent] {
///         &[HookEvent::PreToolUse]
///     }
///
///     fn matcher(&self) -> Option<&str> {
///         Some("bash")
///     }
///
///     async fn on_event(&self, input: &HookInput) -> HookOutput {
///         if let HookInput::PreToolUse { tool_input, .. } = input {
///             let cmd = tool_input["command"].as_str().unwrap_or("");
///             if cmd.contains("rm -rf /") {
///                 return HookOutput::deny("Refusing to delete root filesystem");
///             }
///         }
///         HookOutput::passthrough()
///     }
/// }
/// ```
#[async_trait]
pub trait Hook: Send + Sync {
    /// Hook 名称 — 用于日志、诊断和去重。
    fn name(&self) -> &str;

    /// 声明此 Hook 关注的事件列表。
    ///
    /// `HookRegistry` 据此过滤，只对匹配的事件调用 `on_event`。
    /// 返回空切片表示关注所有事件。
    fn events(&self) -> &[HookEvent] {
        &[]
    }

    /// Matcher 模式 — 进一步过滤匹配条件。
    ///
    /// 对 Tool 相关事件匹配 `tool_name`，支持：
    /// - 精确匹配：`"bash"`
    /// - 管道分隔多选：`"bash|write_file"`
    /// - 通配符：`"read_*"`
    ///
    /// `None` 表示不过滤（匹配所有）。
    fn matcher(&self) -> Option<&str> {
        None
    }

    /// 执行 Hook 逻辑。
    ///
    /// 返回 `HookOutput::passthrough()` 表示不干预。
    async fn on_event(&self, input: &HookInput) -> HookOutput;
}

// ===========================================================================
// HookSource
// ===========================================================================

/// Hook 来源 — 用于冲突解决、日志追踪和安全策略。
///
/// # Examples
///
/// ```
/// use katu_core::hook::HookSource;
///
/// let src = HookSource::Plugin { name: "linter".into() };
/// assert!(matches!(src, HookSource::Plugin { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HookSource {
    /// 用户全局配置文件。
    Settings,
    /// 项目配置文件。
    Project,
    /// Plugin 注册。
    Plugin { name: String },
    /// SDK / 程序化注册。
    Programmatic,
    /// Session 级别临时注册。
    Session,
}

// ===========================================================================
// HookRegistry
// ===========================================================================

/// Hook 注册中心 — 管理已注册的 Hook 并按事件/matcher 分发。
///
/// ## 生命周期
/// - Agent 启动时构建（从配置 + 程序化注册）
/// - Agent loop 在各生命周期节点调用匹配的 Hook
/// - Session 结束时销毁
///
/// ## 线程安全
/// 使用 `Arc<dyn Hook>` 存储，支持跨 await 共享。
/// `HookRegistry` 本身在构建后作为 `Arc<HookRegistry>` 或引用传入 agent loop。
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use async_trait::async_trait;
/// use katu_core::hook::*;
///
/// struct MyHook;
///
/// #[async_trait]
/// impl Hook for MyHook {
///     fn name(&self) -> &str { "my_hook" }
///     fn events(&self) -> &[HookEvent] { &[HookEvent::PreToolUse] }
///     async fn on_event(&self, _input: &HookInput) -> HookOutput {
///         HookOutput::passthrough()
///     }
/// }
///
/// let mut registry = HookRegistry::new();
/// registry.register(Arc::new(MyHook), HookSource::Programmatic, 0);
/// assert_eq!(registry.len(), 1);
/// ```
pub struct HookRegistry {
    hooks: Vec<RegisteredHook>,
}

/// 已注册的 Hook 条目 — 包含 Hook 实例、来源和优先级。
pub struct RegisteredHook {
    /// Hook 实例。
    pub hook: Arc<dyn Hook>,
    /// 来源标识。
    pub source: HookSource,
    /// 优先级（数值越小越先执行）。
    pub priority: i32,
}

impl HookRegistry {
    /// 创建空的注册中心。
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// 注册一个 Hook。
    ///
    /// Hook 按 `priority` 升序排列（数值越小越先执行）。
    pub fn register(
        &mut self,
        hook: Arc<dyn Hook>,
        source: HookSource,
        priority: i32,
    ) {
        self.hooks.push(RegisteredHook {
            hook,
            source,
            priority,
        });
        self.hooks.sort_by_key(|h| h.priority);
    }

    /// 移除指定名称的所有 Hook。
    pub fn remove(&mut self, name: &str) {
        self.hooks.retain(|h| h.hook.name() != name);
    }

    /// 已注册的 Hook 数量。
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// 获取匹配指定输入的所有 Hook（按 priority 排序）。
    ///
    /// 匹配逻辑：
    /// 1. 事件匹配 — `hook.events()` 为空（匹配所有）或包含 `input.event()`
    /// 2. Matcher 匹配 — `hook.matcher()` 为 None（匹配所有）或模式匹配 tool_name
    pub fn matching(&self, input: &HookInput) -> Vec<&RegisteredHook> {
        let event = input.event();
        let tool_name = input.tool_name();

        self.hooks
            .iter()
            .filter(|h| {
                let events = h.hook.events();
                let event_match = events.is_empty() || events.contains(&event);
                if !event_match {
                    return false;
                }

                match (h.hook.matcher(), tool_name) {
                    (Some(pattern), Some(name)) => matches_pattern(name, pattern),
                    (Some(_), None) => false,
                    (None, _) => true,
                }
            })
            .collect()
    }

    /// 检查是否有任何 Hook 注册在指定事件上。
    ///
    /// 轻量级检查，不做 matcher 匹配，用于热路径的快速跳过。
    pub fn has_hooks_for(&self, event: HookEvent) -> bool {
        self.hooks.iter().any(|h| {
            let events = h.hook.events();
            events.is_empty() || events.contains(&event)
        })
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// AggregatedHookOutput
// ===========================================================================

/// 多个 Hook 结果的聚合 — 并行执行后合并。
///
/// ## 聚合规则
/// - **权限**：`Deny > Ask > Allow`（最严格的获胜）
/// - **updated_input / updated_output**：最后一个设置者获胜
/// - **additional_context / blocking_errors / system_messages**：合并
/// - **prevent_continuation**：任一 Hook 设置即生效
///
/// # Examples
///
/// ```
/// use katu_core::hook::{AggregatedHookOutput, HookOutput, HookPermission};
///
/// let mut agg = AggregatedHookOutput::default();
///
/// // Hook A: allow
/// agg.merge(HookOutput::allow(), "hook_a");
/// assert_eq!(agg.permission, Some(HookPermission::Allow));
///
/// // Hook B: deny（更严格，覆盖 allow）
/// agg.merge(HookOutput::deny("unsafe"), "hook_b");
/// assert!(agg.permission.as_ref().unwrap().is_deny());
/// ```
#[derive(Debug, Clone, Default)]
pub struct AggregatedHookOutput {
    /// 聚合后的权限决策（最严格的获胜）。
    pub permission: Option<HookPermission>,

    /// 最终的 updated_input（最后一个设置者获胜）。
    pub updated_input: Option<serde_json::Value>,

    /// 最终的 updated_output。
    pub updated_output: Option<ToolOutput>,

    /// 所有 Hook 注入的上下文（合并）。
    pub additional_context: Vec<String>,

    /// 任一 Hook 阻止继续。
    pub prevent_continuation: bool,

    /// 第一个 stop_reason。
    pub stop_reason: Option<String>,

    /// 所有 blocking_error（合并，带 hook 名前缀）。
    pub blocking_errors: Vec<String>,

    /// 所有 system_message（合并）。
    pub system_messages: Vec<String>,
}

impl AggregatedHookOutput {
    /// 合并单个 Hook 的输出。
    pub fn merge(&mut self, output: HookOutput, hook_name: &str) {
        // 权限聚合：Deny > Ask > Allow
        if let Some(ref new_perm) = output.permission {
            match &self.permission {
                Some(existing) if existing.strictness() >= new_perm.strictness() => {
                    // 已有更严格或同级的决策，保持不变
                }
                _ => {
                    self.permission = output.permission.clone();
                }
            }
        }

        if output.updated_input.is_some() {
            self.updated_input = output.updated_input;
        }
        if output.updated_output.is_some() {
            self.updated_output = output.updated_output;
        }
        self.additional_context.extend(output.additional_context);

        if output.prevent_continuation {
            self.prevent_continuation = true;
            if self.stop_reason.is_none() {
                self.stop_reason = output.stop_reason;
            }
        }

        if let Some(err) = output.blocking_error {
            self.blocking_errors.push(format!("[{hook_name}] {err}"));
        }
        if let Some(msg) = output.system_message {
            self.system_messages.push(msg);
        }
    }

    /// 是否有任何 Hook 做出了实质性决策。
    pub fn has_decision(&self) -> bool {
        self.permission.is_some()
            || self.updated_input.is_some()
            || self.updated_output.is_some()
            || !self.additional_context.is_empty()
            || self.prevent_continuation
            || !self.blocking_errors.is_empty()
    }

    /// 是否有阻塞性错误。
    pub fn has_blocking_errors(&self) -> bool {
        !self.blocking_errors.is_empty()
    }

    /// 是否被拒绝。
    pub fn is_denied(&self) -> bool {
        matches!(&self.permission, Some(p) if p.is_deny())
    }
}

// ===========================================================================
// 工具函数
// ===========================================================================

/// 模式匹配 — 判断 `value` 是否匹配 `pattern`。
///
/// 支持三种语法：
/// - 精确匹配：`"bash"` 匹配 `"bash"`
/// - 管道分隔多选：`"bash|write_file"` 匹配 `"bash"` 或 `"write_file"`
/// - 通配符 `*`：`"read_*"` 匹配 `"read_file"`, `"read_dir"` 等
///
/// # Examples
///
/// ```
/// use katu_core::hook::matches_pattern;
///
/// assert!(matches_pattern("bash", "bash"));
/// assert!(matches_pattern("bash", "bash|write_file"));
/// assert!(matches_pattern("read_file", "read_*"));
/// assert!(!matches_pattern("write_file", "read_*"));
/// ```
pub fn matches_pattern(value: &str, pattern: &str) -> bool {
    if pattern.contains('|') {
        return pattern.split('|').any(|p| matches_single_pattern(value, p.trim()));
    }
    matches_single_pattern(value, pattern)
}

/// 单个模式匹配（支持 `*` 通配符）。
fn matches_single_pattern(value: &str, pattern: &str) -> bool {
    if !pattern.contains('*') {
        return value == pattern;
    }

    let parts: Vec<&str> = pattern.split('*').collect();

    // 单个 `*` → 匹配所有
    if parts.len() == 2 && parts[0].is_empty() && parts[1].is_empty() {
        return true;
    }

    // 前缀匹配：`read_*`
    if parts.len() == 2 && parts[1].is_empty() {
        return value.starts_with(parts[0]);
    }

    // 后缀匹配：`*_file`
    if parts.len() == 2 && parts[0].is_empty() {
        return value.ends_with(parts[1]);
    }

    // 前后匹配：`pre_*_use`
    if parts.len() == 2 {
        return value.starts_with(parts[0])
            && value.ends_with(parts[1])
            && value.len() >= parts[0].len() + parts[1].len();
    }

    // 多段通配符：逐段贪心匹配
    let mut remaining = value;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            if !remaining.starts_with(part) {
                return false;
            }
            remaining = &remaining[part.len()..];
        } else if let Some(pos) = remaining.find(part) {
            remaining = &remaining[pos + part.len()..];
        } else {
            return false;
        }
    }
    true
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- HookEvent --

    #[test]
    fn test_hook_event_is_tool_event() {
        assert!(HookEvent::PreToolUse.is_tool_event());
        assert!(HookEvent::PostToolUse.is_tool_event());
        assert!(HookEvent::PostToolFailure.is_tool_event());
        assert!(!HookEvent::SessionStart.is_tool_event());
        assert!(!HookEvent::Stop.is_tool_event());
    }

    #[test]
    fn test_hook_event_display() {
        assert_eq!(HookEvent::PreToolUse.to_string(), "pre_tool_use");
        assert_eq!(HookEvent::PostToolUse.to_string(), "post_tool_use");
        assert_eq!(HookEvent::SessionStart.to_string(), "session_start");
    }

    #[test]
    fn test_hook_event_serde_roundtrip() {
        for event in ALL_HOOK_EVENTS {
            let json_str = serde_json::to_string(event).unwrap();
            let restored: HookEvent = serde_json::from_str(&json_str).unwrap();
            assert_eq!(*event, restored);
        }
    }

    #[test]
    fn test_all_hook_events_count() {
        assert_eq!(ALL_HOOK_EVENTS.len(), 10);
    }

    // -- HookInput --

    #[test]
    fn test_hook_input_event() {
        let input = HookInput::PreToolUse {
            tool_name: "bash".into(),
            tool_input: json!({}),
            call_id: ToolCallId::new("c1"),
        };
        assert_eq!(input.event(), HookEvent::PreToolUse);
    }

    #[test]
    fn test_hook_input_tool_name() {
        let tool_input = HookInput::PreToolUse {
            tool_name: "bash".into(),
            tool_input: json!({}),
            call_id: ToolCallId::new("c1"),
        };
        assert_eq!(tool_input.tool_name(), Some("bash"));

        let non_tool = HookInput::SessionStart {
            session_id: SessionId::new(),
        };
        assert_eq!(non_tool.tool_name(), None);
    }

    #[test]
    fn test_hook_input_call_id() {
        let input = HookInput::PostToolFailure {
            tool_name: "bash".into(),
            tool_input: json!({}),
            error: "exit code 1".into(),
            call_id: ToolCallId::new("c2"),
        };
        assert_eq!(input.call_id().unwrap().as_str(), "c2");

        let non_tool = HookInput::Stop {
            finish_reason: "completed".into(),
        };
        assert!(non_tool.call_id().is_none());
    }

    #[test]
    fn test_hook_input_serde_roundtrip() {
        let input = HookInput::PreToolUse {
            tool_name: "read_file".into(),
            tool_input: json!({"path": "/tmp/test.txt"}),
            call_id: ToolCallId::new("call_42"),
        };
        let json_str = serde_json::to_string(&input).unwrap();
        assert!(json_str.contains("pre_tool_use"));
        let restored: HookInput = serde_json::from_str(&json_str).unwrap();
        assert_eq!(restored.event(), HookEvent::PreToolUse);
        assert_eq!(restored.tool_name(), Some("read_file"));
    }

    // -- HookPermission --

    #[test]
    fn test_hook_permission_variants() {
        assert!(HookPermission::Allow.is_allow());
        assert!(HookPermission::deny().is_deny());
        assert!(HookPermission::ask().is_ask());
    }

    #[test]
    fn test_hook_permission_with_reason() {
        let deny = HookPermission::deny_with_reason("unsafe");
        match deny {
            HookPermission::Deny { reason } => assert_eq!(reason, Some("unsafe".into())),
            _ => panic!("expected Deny"),
        }
    }

    #[test]
    fn test_hook_permission_strictness() {
        assert!(HookPermission::Allow.strictness() < HookPermission::ask().strictness());
        assert!(HookPermission::ask().strictness() < HookPermission::deny().strictness());
    }

    #[test]
    fn test_hook_permission_serde_roundtrip() {
        for perm in [
            HookPermission::Allow,
            HookPermission::deny(),
            HookPermission::deny_with_reason("test"),
            HookPermission::ask(),
            HookPermission::ask_with_message("confirm?"),
        ] {
            let json_str = serde_json::to_string(&perm).unwrap();
            let restored: HookPermission = serde_json::from_str(&json_str).unwrap();
            assert_eq!(perm, restored);
        }
    }

    // -- HookOutput --

    #[test]
    fn test_hook_output_passthrough() {
        let out = HookOutput::passthrough();
        assert!(!out.has_decision());
        assert!(out.permission.is_none());
        assert!(out.additional_context.is_empty());
    }

    #[test]
    fn test_hook_output_allow() {
        let out = HookOutput::allow();
        assert!(out.has_decision());
        assert!(out.permission.as_ref().unwrap().is_allow());
    }

    #[test]
    fn test_hook_output_deny() {
        let out = HookOutput::deny("bad command");
        assert!(out.has_decision());
        assert!(out.permission.as_ref().unwrap().is_deny());
    }

    #[test]
    fn test_hook_output_ask() {
        let out = HookOutput::ask("are you sure?");
        assert!(out.has_decision());
        assert!(out.permission.as_ref().unwrap().is_ask());
    }

    #[test]
    fn test_hook_output_builder() {
        let out = HookOutput::allow()
            .with_updated_input(json!({"command": "ls"}))
            .with_context("working directory: /tmp")
            .with_system_message("Input sanitized");

        assert!(out.permission.as_ref().unwrap().is_allow());
        assert_eq!(out.updated_input.as_ref().unwrap()["command"], "ls");
        assert_eq!(out.additional_context.len(), 1);
        assert_eq!(out.system_message, Some("Input sanitized".into()));
    }

    #[test]
    fn test_hook_output_with_stop() {
        let out = HookOutput::passthrough().with_stop("loop detected");
        assert!(out.prevent_continuation);
        assert_eq!(out.stop_reason, Some("loop detected".into()));
        assert!(out.has_decision());
    }

    #[test]
    fn test_hook_output_with_blocking_error() {
        let out = HookOutput::passthrough().with_blocking_error("lint failed");
        assert!(out.has_decision());
        assert_eq!(out.blocking_error, Some("lint failed".into()));
    }

    #[test]
    fn test_hook_output_serde_roundtrip() {
        let out = HookOutput::deny("test")
            .with_context("ctx1")
            .with_system_message("msg1");
        let json_str = serde_json::to_string(&out).unwrap();
        let restored: HookOutput = serde_json::from_str(&json_str).unwrap();
        assert_eq!(restored.additional_context, vec!["ctx1"]);
        assert_eq!(restored.system_message, Some("msg1".into()));
    }

    // -- AggregatedHookOutput --

    #[test]
    fn test_aggregated_merge_permission_deny_wins() {
        let mut agg = AggregatedHookOutput::default();

        agg.merge(HookOutput::allow(), "hook_a");
        assert!(agg.permission.as_ref().unwrap().is_allow());

        agg.merge(HookOutput::deny("nope"), "hook_b");
        assert!(agg.permission.as_ref().unwrap().is_deny());

        // allow after deny — deny 仍然获胜
        agg.merge(HookOutput::allow(), "hook_c");
        assert!(agg.permission.as_ref().unwrap().is_deny());
    }

    #[test]
    fn test_aggregated_merge_permission_ask_beats_allow() {
        let mut agg = AggregatedHookOutput::default();

        agg.merge(HookOutput::allow(), "hook_a");
        agg.merge(HookOutput::ask("confirm?"), "hook_b");
        assert!(agg.permission.as_ref().unwrap().is_ask());

        // allow after ask — ask 仍然获胜
        agg.merge(HookOutput::allow(), "hook_c");
        assert!(agg.permission.as_ref().unwrap().is_ask());
    }

    #[test]
    fn test_aggregated_merge_context() {
        let mut agg = AggregatedHookOutput::default();

        agg.merge(
            HookOutput::passthrough().with_context("ctx1"),
            "hook_a",
        );
        agg.merge(
            HookOutput::passthrough().with_context("ctx2"),
            "hook_b",
        );
        assert_eq!(agg.additional_context, vec!["ctx1", "ctx2"]);
    }

    #[test]
    fn test_aggregated_merge_blocking_errors() {
        let mut agg = AggregatedHookOutput::default();

        agg.merge(
            HookOutput::passthrough().with_blocking_error("err1"),
            "linter",
        );
        agg.merge(
            HookOutput::passthrough().with_blocking_error("err2"),
            "validator",
        );
        assert_eq!(agg.blocking_errors.len(), 2);
        assert!(agg.blocking_errors[0].contains("[linter]"));
        assert!(agg.blocking_errors[1].contains("[validator]"));
        assert!(agg.has_blocking_errors());
    }

    #[test]
    fn test_aggregated_merge_stop() {
        let mut agg = AggregatedHookOutput::default();

        agg.merge(HookOutput::passthrough(), "hook_a");
        assert!(!agg.prevent_continuation);

        agg.merge(
            HookOutput::passthrough().with_stop("first reason"),
            "hook_b",
        );
        assert!(agg.prevent_continuation);
        assert_eq!(agg.stop_reason, Some("first reason".into()));

        // 第二个 stop — prevent_continuation 已为 true，stop_reason 保持第一个
        agg.merge(
            HookOutput::passthrough().with_stop("second reason"),
            "hook_c",
        );
        assert_eq!(agg.stop_reason, Some("first reason".into()));
    }

    #[test]
    fn test_aggregated_merge_updated_input_last_wins() {
        let mut agg = AggregatedHookOutput::default();

        agg.merge(
            HookOutput::allow().with_updated_input(json!({"a": 1})),
            "hook_a",
        );
        agg.merge(
            HookOutput::allow().with_updated_input(json!({"b": 2})),
            "hook_b",
        );
        assert_eq!(agg.updated_input, Some(json!({"b": 2})));
    }

    #[test]
    fn test_aggregated_has_decision() {
        let agg = AggregatedHookOutput::default();
        assert!(!agg.has_decision());

        let mut agg2 = AggregatedHookOutput::default();
        agg2.merge(HookOutput::allow(), "h");
        assert!(agg2.has_decision());
    }

    #[test]
    fn test_aggregated_is_denied() {
        let mut agg = AggregatedHookOutput::default();
        assert!(!agg.is_denied());

        agg.merge(HookOutput::deny("no"), "h");
        assert!(agg.is_denied());
    }

    // -- matches_pattern --

    #[test]
    fn test_matches_pattern_exact() {
        assert!(matches_pattern("bash", "bash"));
        assert!(!matches_pattern("bash", "write_file"));
    }

    #[test]
    fn test_matches_pattern_pipe_separated() {
        assert!(matches_pattern("bash", "bash|write_file"));
        assert!(matches_pattern("write_file", "bash|write_file"));
        assert!(!matches_pattern("read_file", "bash|write_file"));
    }

    #[test]
    fn test_matches_pattern_wildcard_star() {
        assert!(matches_pattern("read_file", "read_*"));
        assert!(matches_pattern("read_dir", "read_*"));
        assert!(!matches_pattern("write_file", "read_*"));
    }

    #[test]
    fn test_matches_pattern_wildcard_suffix() {
        assert!(matches_pattern("read_file", "*_file"));
        assert!(matches_pattern("write_file", "*_file"));
        assert!(!matches_pattern("read_dir", "*_file"));
    }

    #[test]
    fn test_matches_pattern_wildcard_middle() {
        assert!(matches_pattern("pre_tool_use", "pre_*_use"));
        assert!(matches_pattern("pre_compact_use", "pre_*_use"));
        assert!(!matches_pattern("pre_tool_fail", "pre_*_use"));
    }

    #[test]
    fn test_matches_pattern_star_matches_all() {
        assert!(matches_pattern("anything", "*"));
        assert!(matches_pattern("", "*"));
    }

    #[test]
    fn test_matches_pattern_pipe_with_wildcard() {
        assert!(matches_pattern("read_file", "bash|read_*"));
        assert!(matches_pattern("bash", "bash|read_*"));
        assert!(!matches_pattern("write_file", "bash|read_*"));
    }

    // -- HookRegistry --

    struct PassthroughHook {
        hook_name: String,
        hook_events: Vec<HookEvent>,
        hook_matcher: Option<String>,
    }

    impl PassthroughHook {
        fn new(name: &str) -> Self {
            Self {
                hook_name: name.into(),
                hook_events: vec![],
                hook_matcher: None,
            }
        }

        fn with_events(mut self, events: Vec<HookEvent>) -> Self {
            self.hook_events = events;
            self
        }

        fn with_matcher(mut self, matcher: &str) -> Self {
            self.hook_matcher = Some(matcher.into());
            self
        }
    }

    #[async_trait]
    impl Hook for PassthroughHook {
        fn name(&self) -> &str {
            &self.hook_name
        }

        fn events(&self) -> &[HookEvent] {
            &self.hook_events
        }

        fn matcher(&self) -> Option<&str> {
            self.hook_matcher.as_deref()
        }

        async fn on_event(&self, _input: &HookInput) -> HookOutput {
            HookOutput::passthrough()
        }
    }

    #[test]
    fn test_registry_new_empty() {
        let reg = HookRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_register_and_len() {
        let mut reg = HookRegistry::new();
        reg.register(
            Arc::new(PassthroughHook::new("a")),
            HookSource::Programmatic,
            0,
        );
        reg.register(
            Arc::new(PassthroughHook::new("b")),
            HookSource::Programmatic,
            0,
        );
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = HookRegistry::new();
        reg.register(
            Arc::new(PassthroughHook::new("a")),
            HookSource::Programmatic,
            0,
        );
        reg.register(
            Arc::new(PassthroughHook::new("b")),
            HookSource::Programmatic,
            0,
        );
        reg.remove("a");
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.hooks[0].hook.name(), "b");
    }

    #[test]
    fn test_registry_matching_by_event() {
        let mut reg = HookRegistry::new();
        reg.register(
            Arc::new(PassthroughHook::new("pre_only").with_events(vec![HookEvent::PreToolUse])),
            HookSource::Programmatic,
            0,
        );
        reg.register(
            Arc::new(PassthroughHook::new("post_only").with_events(vec![HookEvent::PostToolUse])),
            HookSource::Programmatic,
            0,
        );
        reg.register(
            Arc::new(PassthroughHook::new("all_events")),
            HookSource::Programmatic,
            0,
        );

        let input = HookInput::PreToolUse {
            tool_name: "bash".into(),
            tool_input: json!({}),
            call_id: ToolCallId::new("c1"),
        };
        let matched = reg.matching(&input);
        assert_eq!(matched.len(), 2);

        let names: Vec<&str> = matched.iter().map(|h| h.hook.name()).collect();
        assert!(names.contains(&"pre_only"));
        assert!(names.contains(&"all_events"));
        assert!(!names.contains(&"post_only"));
    }

    #[test]
    fn test_registry_matching_by_matcher() {
        let mut reg = HookRegistry::new();
        reg.register(
            Arc::new(
                PassthroughHook::new("bash_only")
                    .with_events(vec![HookEvent::PreToolUse])
                    .with_matcher("bash"),
            ),
            HookSource::Programmatic,
            0,
        );
        reg.register(
            Arc::new(
                PassthroughHook::new("write_family")
                    .with_events(vec![HookEvent::PreToolUse])
                    .with_matcher("write_*"),
            ),
            HookSource::Programmatic,
            0,
        );

        // bash → 匹配 bash_only
        let input_bash = HookInput::PreToolUse {
            tool_name: "bash".into(),
            tool_input: json!({}),
            call_id: ToolCallId::new("c1"),
        };
        let matched = reg.matching(&input_bash);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].hook.name(), "bash_only");

        // write_file → 匹配 write_family
        let input_write = HookInput::PreToolUse {
            tool_name: "write_file".into(),
            tool_input: json!({}),
            call_id: ToolCallId::new("c2"),
        };
        let matched = reg.matching(&input_write);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].hook.name(), "write_family");

        // read_file → 无匹配
        let input_read = HookInput::PreToolUse {
            tool_name: "read_file".into(),
            tool_input: json!({}),
            call_id: ToolCallId::new("c3"),
        };
        let matched = reg.matching(&input_read);
        assert!(matched.is_empty());
    }

    #[test]
    fn test_registry_matching_non_tool_event_with_matcher() {
        let mut reg = HookRegistry::new();
        // 有 matcher 但事件不是 tool 事件 → 不匹配
        reg.register(
            Arc::new(
                PassthroughHook::new("h")
                    .with_events(vec![HookEvent::SessionStart])
                    .with_matcher("bash"),
            ),
            HookSource::Programmatic,
            0,
        );

        let input = HookInput::SessionStart {
            session_id: SessionId::new(),
        };
        let matched = reg.matching(&input);
        assert!(matched.is_empty());
    }

    #[test]
    fn test_registry_priority_order() {
        let mut reg = HookRegistry::new();
        reg.register(
            Arc::new(PassthroughHook::new("low")),
            HookSource::Programmatic,
            10,
        );
        reg.register(
            Arc::new(PassthroughHook::new("high")),
            HookSource::Programmatic,
            -10,
        );
        reg.register(
            Arc::new(PassthroughHook::new("mid")),
            HookSource::Programmatic,
            0,
        );

        let input = HookInput::SessionStart {
            session_id: SessionId::new(),
        };
        let matched = reg.matching(&input);
        assert_eq!(matched[0].hook.name(), "high");
        assert_eq!(matched[1].hook.name(), "mid");
        assert_eq!(matched[2].hook.name(), "low");
    }

    #[test]
    fn test_registry_has_hooks_for() {
        let mut reg = HookRegistry::new();
        reg.register(
            Arc::new(PassthroughHook::new("pre_only").with_events(vec![HookEvent::PreToolUse])),
            HookSource::Programmatic,
            0,
        );

        assert!(reg.has_hooks_for(HookEvent::PreToolUse));
        assert!(!reg.has_hooks_for(HookEvent::PostToolUse));
    }

    #[test]
    fn test_registry_has_hooks_for_all_events() {
        let mut reg = HookRegistry::new();
        // events() = [] 表示关注所有事件
        reg.register(
            Arc::new(PassthroughHook::new("global")),
            HookSource::Programmatic,
            0,
        );

        for event in ALL_HOOK_EVENTS {
            assert!(reg.has_hooks_for(*event));
        }
    }

    // -- HookSource --

    #[test]
    fn test_hook_source_serde_roundtrip() {
        for source in [
            HookSource::Settings,
            HookSource::Project,
            HookSource::Plugin {
                name: "linter".into(),
            },
            HookSource::Programmatic,
            HookSource::Session,
        ] {
            let json_str = serde_json::to_string(&source).unwrap();
            let restored: HookSource = serde_json::from_str(&json_str).unwrap();
            assert_eq!(source, restored);
        }
    }

    // -- Hook trait async --

    #[tokio::test]
    async fn test_hook_trait_async_execution() {
        struct DenyBashHook;

        #[async_trait]
        impl Hook for DenyBashHook {
            fn name(&self) -> &str {
                "deny_bash"
            }

            fn events(&self) -> &[HookEvent] {
                &[HookEvent::PreToolUse]
            }

            fn matcher(&self) -> Option<&str> {
                Some("bash")
            }

            async fn on_event(&self, input: &HookInput) -> HookOutput {
                if let HookInput::PreToolUse { tool_input, .. } = input {
                    let cmd = tool_input["command"].as_str().unwrap_or("");
                    if cmd.contains("rm -rf") {
                        return HookOutput::deny("dangerous command");
                    }
                }
                HookOutput::passthrough()
            }
        }

        let hook: Arc<dyn Hook> = Arc::new(DenyBashHook);

        // 安全命令 → passthrough
        let safe_input = HookInput::PreToolUse {
            tool_name: "bash".into(),
            tool_input: json!({"command": "ls -la"}),
            call_id: ToolCallId::new("c1"),
        };
        let output = hook.on_event(&safe_input).await;
        assert!(!output.has_decision());

        // 危险命令 → deny
        let dangerous_input = HookInput::PreToolUse {
            tool_name: "bash".into(),
            tool_input: json!({"command": "rm -rf /"}),
            call_id: ToolCallId::new("c2"),
        };
        let output = hook.on_event(&dangerous_input).await;
        assert!(output.permission.as_ref().unwrap().is_deny());
    }

    #[tokio::test]
    async fn test_hook_trait_dyn_dispatch() {
        let hook: Arc<dyn Hook> = Arc::new(PassthroughHook::new("test"));
        assert_eq!(hook.name(), "test");

        let input = HookInput::SessionStart {
            session_id: SessionId::new(),
        };
        let output = hook.on_event(&input).await;
        assert!(!output.has_decision());
    }
}
