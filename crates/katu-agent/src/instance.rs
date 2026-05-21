//! # instance
//!
//! ## 职责
//! 定义 Agent 运行实例 (`AgentInstance`) — 一次 Agent 执行的完整运行时上下文，
//! 以及其构建器 (`InstanceBuilder`) 和运行级配置 (`RunConfig`)。
//!
//! ## 设计
//! `AgentInstance` 与 `Session` 的职责边界：
//! - **Session** = 持久化会话状态（消息历史、用量累计、状态机）
//! - **AgentInstance** = 一次 run 的临时运行时上下文（已解析的模型、工具、prompt、配置）
//!
//! 每次 `AgentRunner::run()` (future) 接收一个 `AgentInstance`，
//! run 结束后 instance 销毁，Session 中的消息和用量保留。
//!
//! ```text
//! InstanceBuilder
//!   ├── AgentDefinition  → 提供 name, role, tool_filter, system_prompt
//!   ├── ModelRef          → 已解析的模型引用
//!   ├── Vec<Arc<dyn Tool>>→ 经 ToolFilter 过滤后的可用工具
//!   ├── PromptBuilder     → 组装 system prompt
//!   └── Session           → 会话状态
//!   │
//!   ▼
//! AgentInstance ──→ AgentRunner::run() (future)
//! ```
//!
//! ## 调用者
//! - `katu-agent::runner` (future) — Agent loop 核心循环
//! - 应用层 — 通过 `InstanceBuilder` 构建实例

use std::sync::Arc;

use tokio::sync::mpsc;

use katu_core::agent::AgentDefinition;
use katu_core::agent_event::AgentEvent;
use katu_core::hook::HookRegistry;
use katu_core::permission::Ruleset;
use katu_core::tool::ToolDefinition;
use katu_core::types::AgentId;
use katu_core::Tool;

use katu_llm::model::ModelRef;
use katu_llm::Provider;

use crate::error::{AgentError, Result};
use crate::prompt::{EnvironmentInfo, PromptBuilder};
use crate::retry::RetryConfig;
use crate::session::Session;
use crate::tool_executor::ToolExecutorConfig;

// ===========================================================================
// RunConfig
// ===========================================================================

/// 运行级配置 — 控制单次 Agent 执行的行为参数。
///
/// 包含重试策略、工具执行配置、压缩控制等运行时行为选项。
/// 与 `AgentDefinition`（静态配置）互补：
/// - `AgentDefinition` = "Agent 是什么"
/// - `RunConfig` = "这次执行怎么运行"
///
/// # Examples
///
/// ```
/// use katu_agent::instance::RunConfig;
/// use katu_agent::retry::RetryConfig;
///
/// // 默认配置
/// let config = RunConfig::default();
/// assert!(config.auto_compact());
///
/// // 自定义
/// let config = RunConfig::new()
///     .with_max_steps(20)
///     .with_auto_compact(false)
///     .with_retry(RetryConfig::disabled());
/// ```
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// 最大步数覆盖（优先于 `AgentDefinition.max_steps`）。
    ///
    /// `None` 时使用 `AgentDefinition.max_steps` 或系统默认值。
    max_steps: Option<u32>,

    /// 是否启用自动上下文压缩。
    auto_compact: bool,

    /// 重试策略。
    retry: RetryConfig,

    /// 工具执行器配置。
    tool_executor: ToolExecutorConfig,
}

impl RunConfig {
    /// 创建默认运行配置。
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置最大步数覆盖。
    pub fn with_max_steps(mut self, steps: u32) -> Self {
        self.max_steps = Some(steps);
        self
    }

    /// 设置是否启用自动压缩。
    pub fn with_auto_compact(mut self, enabled: bool) -> Self {
        self.auto_compact = enabled;
        self
    }

    /// 设置重试策略。
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    /// 设置工具执行器配置。
    pub fn with_tool_executor(mut self, config: ToolExecutorConfig) -> Self {
        self.tool_executor = config;
        self
    }
}

// ---------------------------------------------------------------------------
// 读取方法
// ---------------------------------------------------------------------------

impl RunConfig {
    /// 最大步数覆盖值。
    pub fn max_steps_override(&self) -> Option<u32> {
        self.max_steps
    }

    /// 是否启用自动压缩。
    pub fn auto_compact(&self) -> bool {
        self.auto_compact
    }

    /// 重试策略引用。
    pub fn retry(&self) -> &RetryConfig {
        &self.retry
    }

    /// 工具执行器配置引用。
    pub fn tool_executor(&self) -> &ToolExecutorConfig {
        &self.tool_executor
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_steps: None,
            auto_compact: true,
            retry: RetryConfig::default(),
            tool_executor: ToolExecutorConfig::default(),
        }
    }
}

// ===========================================================================
// AgentInstance
// ===========================================================================

/// Agent 运行实例 — 一次完整 Agent 执行的运行时上下文。
///
/// 由 `InstanceBuilder::build()` 创建，持有运行所需的全部已解析依赖：
/// - 已解析的模型引用（`ModelRef`）
/// - 经 `ToolFilter` 过滤的可用工具
/// - LLM Provider 引用
/// - 已组装的 system prompt
/// - 事件发射 channel
/// - 会话状态
///
/// ## 生命周期
/// ```text
/// InstanceBuilder::build() → AgentInstance → AgentRunner::run() → 销毁
///                                  ↑                    │
///                                  │                    ▼
///                               Session ←────── 消息/用量保留
/// ```
///
/// # Examples
///
/// ```ignore
/// use katu_agent::instance::InstanceBuilder;
///
/// let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
///
/// let instance = InstanceBuilder::new(agent_def, model_ref, provider)
///     .with_tools(tools)
///     .with_event_sender(event_tx)
///     .build()?;
/// ```
pub struct AgentInstance {
    /// Agent 实例唯一标识（每次 run 不同）。
    agent_id: AgentId,

    /// 会话状态（消息历史、运行状态、用量）。
    session: Session,

    /// 已解析的模型引用。
    model: ModelRef,

    /// LLM Provider。
    provider: Arc<dyn Provider>,

    /// 可用工具集（经 ToolFilter 过滤后）。
    tools: Vec<Arc<dyn Tool>>,

    /// 工具定义列表（发送给 LLM 的 schema）。
    tool_definitions: Vec<ToolDefinition>,

    /// 系统 prompt 构建器。
    prompt_builder: PromptBuilder,

    /// Hook 注册表。
    hooks: Arc<HookRegistry>,

    /// 权限规则集。
    ruleset: Ruleset,

    /// 事件发射 channel。
    event_tx: mpsc::UnboundedSender<AgentEvent>,

    /// 父 Agent ID（SubAgent 场景下指向调用者）。
    parent_agent_id: Option<AgentId>,

    /// 运行级配置。
    config: RunConfig,

    /// 运行时环境信息（注入到 system prompt）。
    environment: EnvironmentInfo,
}

// 手动实现 Debug — Arc<dyn Provider> 和 Arc<dyn Tool> 不满足 Debug bound。
impl std::fmt::Debug for AgentInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentInstance")
            .field("agent_id", &self.agent_id)
            .field("agent_name", &self.session.agent().name)
            .field("model_id", &self.model.id)
            .field("tool_count", &self.tools.len())
            .field("parent_agent_id", &self.parent_agent_id)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Identity & Configuration
// ---------------------------------------------------------------------------

impl AgentInstance {
    /// Agent 实例 ID。
    pub fn agent_id(&self) -> &AgentId {
        &self.agent_id
    }

    /// Agent 定义（来自 Session）。
    pub fn agent(&self) -> &AgentDefinition {
        self.session.agent()
    }

    /// 已解析的模型引用。
    pub fn model(&self) -> &ModelRef {
        &self.model
    }

    /// LLM Provider。
    pub fn provider(&self) -> &Arc<dyn Provider> {
        &self.provider
    }

    /// 父 Agent ID（SubAgent 场景）。
    pub fn parent_agent_id(&self) -> Option<&AgentId> {
        self.parent_agent_id.as_ref()
    }

    /// 是否为 SubAgent 实例。
    pub fn is_sub_agent(&self) -> bool {
        self.parent_agent_id.is_some()
    }

    /// 运行级配置。
    pub fn config(&self) -> &RunConfig {
        &self.config
    }

    /// 环境信息引用。
    pub fn environment(&self) -> &EnvironmentInfo {
        &self.environment
    }
}

// ---------------------------------------------------------------------------
// Session Access
// ---------------------------------------------------------------------------

impl AgentInstance {
    /// 会话引用（只读）。
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// 会话可变引用（Runner loop 内修改消息/状态）。
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

impl AgentInstance {
    /// 可用工具列表引用。
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    /// 工具定义列表（发送给 LLM 的 schema）。
    pub fn tool_definitions(&self) -> &[ToolDefinition] {
        &self.tool_definitions
    }

    /// 查找工具 by name。
    pub fn find_tool(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.definition().name == name)
    }

    /// 工具数量。
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

impl AgentInstance {
    /// Prompt 构建器引用。
    pub fn prompt_builder(&self) -> &PromptBuilder {
        &self.prompt_builder
    }

    /// Prompt 构建器可变引用（动态注册 provider）。
    pub fn prompt_builder_mut(&mut self) -> &mut PromptBuilder {
        &mut self.prompt_builder
    }
}

// ---------------------------------------------------------------------------
// Hooks & Permissions
// ---------------------------------------------------------------------------

impl AgentInstance {
    /// Hook 注册表引用。
    pub fn hooks(&self) -> &HookRegistry {
        &self.hooks
    }

    /// 权限规则集引用。
    pub fn ruleset(&self) -> &Ruleset {
        &self.ruleset
    }
}

// ---------------------------------------------------------------------------
// Event Emitting
// ---------------------------------------------------------------------------

impl AgentInstance {
    /// 事件发射 channel 引用。
    pub fn event_sender(&self) -> &mpsc::UnboundedSender<AgentEvent> {
        &self.event_tx
    }
}

// ===========================================================================
// InstanceBuilder
// ===========================================================================

/// `AgentInstance` 构建器 — Builder 模式组装 Agent 运行实例。
///
/// ## 必填字段（通过 `new()` 提供）
/// - `AgentDefinition` — Agent 静态配置
/// - `ModelRef` — 已解析的模型引用
/// - `Provider` — LLM 调用实现
///
/// ## 必填字段（通过 `with_*()` 提供）
/// - `event_sender` — 事件发射 channel（Runner 运行前必须设置）
///
/// ## 可选字段
/// - `tools` — 原始工具列表（会经 ToolFilter 过滤）
/// - `hooks` — Hook 注册表
/// - `ruleset` — 权限规则集
/// - `prompt_builder` — 自定义 prompt 构建器
/// - `parent_agent_id` — SubAgent 的调用者
/// - `config` — 运行级配置
/// - `messages` — 预加载的消息历史
///
/// # Examples
///
/// ```ignore
/// use katu_agent::instance::InstanceBuilder;
///
/// let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
///
/// let instance = InstanceBuilder::new(agent_def, model_ref, provider)
///     .with_tools(all_tools)
///     .with_hooks(hook_registry)
///     .with_event_sender(tx)
///     .build()?;
/// ```
pub struct InstanceBuilder {
    agent: AgentDefinition,
    model: ModelRef,
    provider: Arc<dyn Provider>,
    tools: Vec<Arc<dyn Tool>>,
    hooks: Option<Arc<HookRegistry>>,
    ruleset: Option<Ruleset>,
    prompt_builder: Option<PromptBuilder>,
    event_tx: Option<mpsc::UnboundedSender<AgentEvent>>,
    parent_agent_id: Option<AgentId>,
    config: RunConfig,
    messages: Vec<katu_core::Message>,
    context_window: Option<u64>,
    environment: Option<EnvironmentInfo>,
}

impl InstanceBuilder {
    /// 创建构建器 — 必填字段。
    ///
    /// # Arguments
    /// - `agent` — Agent 静态配置
    /// - `model` — 已解析的模型引用（含连接信息、能力、限制）
    /// - `provider` — LLM Provider 实现
    pub fn new(
        agent: AgentDefinition,
        model: ModelRef,
        provider: Arc<dyn Provider>,
    ) -> Self {
        Self {
            agent,
            model,
            provider,
            tools: Vec::new(),
            hooks: None,
            ruleset: None,
            prompt_builder: None,
            event_tx: None,
            parent_agent_id: None,
            config: RunConfig::default(),
            messages: Vec::new(),
            context_window: None,
            environment: None,
        }
    }

    /// 设置可用工具列表（构建时会经 `ToolFilter` 过滤）。
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    /// 设置 Hook 注册表。
    pub fn with_hooks(mut self, hooks: Arc<HookRegistry>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// 设置权限规则集。
    pub fn with_ruleset(mut self, ruleset: Ruleset) -> Self {
        self.ruleset = Some(ruleset);
        self
    }

    /// 设置自定义 prompt 构建器（默认使用 `PromptBuilder::with_defaults()`）。
    pub fn with_prompt_builder(mut self, builder: PromptBuilder) -> Self {
        self.prompt_builder = Some(builder);
        self
    }

    /// 设置事件发射 channel（必须在 `build()` 前调用）。
    pub fn with_event_sender(mut self, tx: mpsc::UnboundedSender<AgentEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// 设置父 Agent ID（SubAgent 场景）。
    pub fn with_parent(mut self, parent_id: AgentId) -> Self {
        self.parent_agent_id = Some(parent_id);
        self
    }

    /// 设置运行级配置。
    pub fn with_config(mut self, config: RunConfig) -> Self {
        self.config = config;
        self
    }

    /// 设置环境信息（默认自动检测当前环境）。
    pub fn with_environment(mut self, env: EnvironmentInfo) -> Self {
        self.environment = Some(env);
        self
    }

    /// 预加载消息历史（用于恢复/续接场景）。
    pub fn with_messages(mut self, messages: Vec<katu_core::Message>) -> Self {
        self.messages = messages;
        self
    }

    /// 覆盖 context window 大小（默认从 `ModelRef.limits.context_window` 取）。
    pub fn with_context_window(mut self, window: u64) -> Self {
        self.context_window = Some(window);
        self
    }

    /// 构建 `AgentInstance`。
    ///
    /// ## 构建流程
    /// 1. 验证必填字段（event_sender）
    /// 2. 创建 Session（或恢复已有消息）
    /// 3. 按 ToolFilter 过滤工具
    /// 4. 提取工具定义列表
    /// 5. 初始化 PromptBuilder
    ///
    /// ## Errors
    /// - `AgentError::Build` — 缺少 event_sender
    pub fn build(self) -> Result<AgentInstance> {
        // 1. 验证必填字段
        let event_tx = self.event_tx.ok_or_else(|| {
            AgentError::build("event_sender is required: call with_event_sender() before build()")
        })?;

        // 2. 创建 Session
        let context_window = self
            .context_window
            .unwrap_or(self.model.limits.context_window as u64);

        let mut session = Session::new(
            self.agent.clone(),
            self.model.id.clone(),
        )
        .with_context_window(context_window);

        // 恢复消息历史
        if !self.messages.is_empty() {
            session.replace_messages(self.messages);
        }

        // 3. 按 ToolFilter 过滤工具
        let tool_filter = &self.agent.tool_filter;
        let tools: Vec<Arc<dyn Tool>> = self
            .tools
            .into_iter()
            .filter(|t| tool_filter.is_allowed(&t.definition().name))
            .collect();

        // 4. 提取工具定义列表
        let tool_definitions: Vec<ToolDefinition> = tools
            .iter()
            .map(|t| t.definition().clone())
            .collect();

        // 5. 初始化 PromptBuilder
        let prompt_builder = self
            .prompt_builder
            .unwrap_or_else(PromptBuilder::with_defaults);

        // 6. 默认 hooks / ruleset
        let hooks = self.hooks.unwrap_or_else(|| Arc::new(HookRegistry::new()));
        let ruleset = self.ruleset.unwrap_or_default();

        // 7. 环境信息
        let environment = self.environment.unwrap_or_else(|| {
            EnvironmentInfo::new(
                std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
                std::env::consts::OS,
            )
        });

        Ok(AgentInstance {
            agent_id: AgentId::new(),
            session,
            model: self.model,
            provider: self.provider,
            tools,
            tool_definitions,
            prompt_builder,
            hooks,
            ruleset,
            event_tx,
            parent_agent_id: self.parent_agent_id,
            config: self.config,
            environment,
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use katu_core::{AgentRole, ModelId, ProviderId, RouteId};
    use katu_llm::model::ModelLimits;
    use std::pin::Pin;
    use std::future::Future;

    // ─── Mock Provider ──────────────────────────────────────────────────────

    struct MockProvider;

    impl Provider for MockProvider {
        fn stream(
            &self,
            _request: katu_llm::LlmRequest,
        ) -> Pin<Box<dyn Future<Output = katu_core::Result<katu_llm::EventStream>> + Send + '_>>
        {
            Box::pin(async { Err(katu_core::Error::Cancelled) })
        }

        fn generate(
            &self,
            _request: katu_llm::LlmRequest,
        ) -> Pin<Box<dyn Future<Output = katu_core::Result<katu_llm::LlmResponse>> + Send + '_>>
        {
            Box::pin(async { Err(katu_core::Error::Cancelled) })
        }
    }

    fn sample_agent() -> AgentDefinition {
        AgentDefinition::new("test", AgentRole::Primary)
            .with_description("Test agent")
            .with_max_steps(10)
    }

    fn sample_model() -> ModelRef {
        ModelRef::new(
            ModelId::new("gpt-4o"),
            ProviderId::new("openai"),
            RouteId::new("openai-chat"),
            "https://api.openai.com/v1",
            ModelLimits {
                context_window: 128_000,
                max_output_tokens: 4096,
            },
        )
    }

    fn sample_provider() -> Arc<dyn Provider> {
        Arc::new(MockProvider)
    }

    // ─── RunConfig Tests ────────────────────────────────────────────────────

    #[test]
    fn test_run_config_default() {
        let config = RunConfig::default();
        assert!(config.auto_compact());
        assert!(config.max_steps_override().is_none());
        assert!(config.retry().is_enabled());
    }

    #[test]
    fn test_run_config_builder() {
        let config = RunConfig::new()
            .with_max_steps(20)
            .with_auto_compact(false)
            .with_retry(RetryConfig::disabled());

        assert_eq!(config.max_steps_override(), Some(20));
        assert!(!config.auto_compact());
        assert!(!config.retry().is_enabled());
    }

    // ─── InstanceBuilder Tests ──────────────────────────────────────────────

    #[test]
    fn test_build_missing_event_sender() {
        let result = InstanceBuilder::new(
            sample_agent(),
            sample_model(),
            sample_provider(),
        )
        .build();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, AgentError::Build { .. }));
        assert!(err.to_string().contains("event_sender"));
    }

    #[test]
    fn test_build_minimal() {
        let (tx, _rx) = mpsc::unbounded_channel();

        let instance = InstanceBuilder::new(
            sample_agent(),
            sample_model(),
            sample_provider(),
        )
        .with_event_sender(tx)
        .build()
        .unwrap();

        assert_eq!(instance.agent().name.as_str(), "test");
        assert_eq!(instance.model().id.as_str(), "gpt-4o");
        assert!(instance.tools().is_empty());
        assert!(instance.tool_definitions().is_empty());
        assert!(!instance.is_sub_agent());
        assert!(instance.session().status().is_idle());
    }

    #[test]
    fn test_build_with_context_window() {
        let (tx, _rx) = mpsc::unbounded_channel();

        let instance = InstanceBuilder::new(
            sample_agent(),
            sample_model(),
            sample_provider(),
        )
        .with_event_sender(tx)
        .with_context_window(200_000)
        .build()
        .unwrap();

        assert_eq!(instance.session().context_window(), 200_000);
    }

    #[test]
    fn test_build_context_window_from_model() {
        let (tx, _rx) = mpsc::unbounded_channel();

        let instance = InstanceBuilder::new(
            sample_agent(),
            sample_model(), // context_window = 128_000
            sample_provider(),
        )
        .with_event_sender(tx)
        .build()
        .unwrap();

        assert_eq!(instance.session().context_window(), 128_000);
    }

    #[test]
    fn test_build_with_parent() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let parent_id = AgentId::new();

        let instance = InstanceBuilder::new(
            sample_agent(),
            sample_model(),
            sample_provider(),
        )
        .with_event_sender(tx)
        .with_parent(parent_id.clone())
        .build()
        .unwrap();

        assert!(instance.is_sub_agent());
        assert_eq!(instance.parent_agent_id(), Some(&parent_id));
    }

    #[test]
    fn test_build_with_config() {
        let (tx, _rx) = mpsc::unbounded_channel();

        let config = RunConfig::new()
            .with_max_steps(5)
            .with_auto_compact(false);

        let instance = InstanceBuilder::new(
            sample_agent(),
            sample_model(),
            sample_provider(),
        )
        .with_event_sender(tx)
        .with_config(config)
        .build()
        .unwrap();

        assert_eq!(instance.config().max_steps_override(), Some(5));
        assert!(!instance.config().auto_compact());
    }

    #[test]
    fn test_build_with_messages() {
        let (tx, _rx) = mpsc::unbounded_channel();

        let messages = vec![
            katu_core::Message::user("Hello"),
            katu_core::Message::assistant("Hi there!"),
        ];

        let instance = InstanceBuilder::new(
            sample_agent(),
            sample_model(),
            sample_provider(),
        )
        .with_event_sender(tx)
        .with_messages(messages)
        .build()
        .unwrap();

        assert_eq!(instance.session().message_count(), 2);
    }

    #[test]
    fn test_tool_filtering() {
        use async_trait::async_trait;
        use katu_core::{ToolCallContext, ToolDefinition, ToolOutput};

        // 创建两个工具
        struct TestTool {
            def: ToolDefinition,
        }

        #[async_trait]
        impl Tool for TestTool {
            fn definition(&self) -> &ToolDefinition {
                &self.def
            }
            async fn execute(
                &self,
                _args: serde_json::Value,
                _ctx: &ToolCallContext,
            ) -> katu_core::Result<ToolOutput> {
                Ok(ToolOutput::success("ok"))
            }
        }

        let tool_read = Arc::new(TestTool {
            def: ToolDefinition::no_params("read_file", "Read a file"),
        }) as Arc<dyn Tool>;

        let tool_bash = Arc::new(TestTool {
            def: ToolDefinition::no_params("bash", "Execute shell command"),
        }) as Arc<dyn Tool>;

        // Agent 只允许 read_file
        let agent = AgentDefinition::new("reader", AgentRole::SubAgent)
            .with_tool_filter(katu_core::ToolFilter::allow_list(["read_file"]));

        let (tx, _rx) = mpsc::unbounded_channel();

        let instance = InstanceBuilder::new(
            agent,
            sample_model(),
            sample_provider(),
        )
        .with_tools(vec![tool_read, tool_bash])
        .with_event_sender(tx)
        .build()
        .unwrap();

        // 只有 read_file 通过过滤
        assert_eq!(instance.tool_count(), 1);
        assert!(instance.find_tool("read_file").is_some());
        assert!(instance.find_tool("bash").is_none());
        assert_eq!(instance.tool_definitions().len(), 1);
        assert_eq!(instance.tool_definitions()[0].name, "read_file");
    }

    #[test]
    fn test_agent_id_uniqueness() {
        let (tx1, _rx1) = mpsc::unbounded_channel();
        let (tx2, _rx2) = mpsc::unbounded_channel();

        let inst1 = InstanceBuilder::new(
            sample_agent(),
            sample_model(),
            sample_provider(),
        )
        .with_event_sender(tx1)
        .build()
        .unwrap();

        let inst2 = InstanceBuilder::new(
            sample_agent(),
            sample_model(),
            sample_provider(),
        )
        .with_event_sender(tx2)
        .build()
        .unwrap();

        assert_ne!(inst1.agent_id(), inst2.agent_id());
    }
}
