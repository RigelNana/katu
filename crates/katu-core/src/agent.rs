//! # katu_core::agent
//!
//! ## 职责
//! 定义 Agent 的静态配置数据模型 — 描述 Agent "是什么"而非"如何运行"。
//!
//! ## 设计原则
//! - **纯数据层** — 只有配置结构，不包含运行时行为（循环、状态机属于 `katu-agent`）
//! - **Builder 模式** — 必填字段通过构造函数，可选字段通过 `with_*` 链式调用
//! - **Serde 友好** — 可从配置文件加载 / 序列化持久化
//!
//! ## 对外接口
//! - `AgentDefinition` — Agent 的完整静态配置
//! - `AgentName` — Agent 名称（newtype）
//! - `AgentRole` — Agent 角色（Primary / SubAgent / Internal）
//! - `AgentModelRef` — 模型引用（继承 / 按 ID / 按别名）
//! - `ToolFilter` — 工具过滤规则
//!
//! ## 调用者
//! - `katu-agent` (future) — AgentRunner 根据 AgentDefinition 驱动循环
//! - `katu-llm` (future) — 解析 AgentModelRef 为完整 ModelRef

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::generation::GenerationOptions;
use crate::tool::ToolChoice;
use crate::types::{ModelId, ProviderId};

// ===========================================================================
// AgentName
// ===========================================================================

/// Agent 名称 — 唯一标识符。
///
/// 命名约定：`snake_case`，如 `"build"`, `"explore"`, `"title"`。
/// 用于注册表查找和 subagent 调度。
///
/// # Examples
///
/// ```
/// use katu_core::AgentName;
///
/// let name = AgentName::new("explore");
/// assert_eq!(name.as_str(), "explore");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentName(String);

impl AgentName {
    /// 从字符串创建 AgentName。
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// 获取名称字符串引用。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AgentName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// ===========================================================================
// AgentRole
// ===========================================================================

/// Agent 角色 — 决定 Agent 在系统中的调度位置。
///
/// # Examples
///
/// ```
/// use katu_core::AgentRole;
///
/// let role = AgentRole::Primary;
/// assert!(role.is_primary());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    /// 主 Agent — 直接面向用户，接收用户输入。
    Primary,
    /// 子 Agent — 由其他 Agent 通过 tool_call 调度，结果返回给调用者。
    SubAgent,
    /// 内部 Agent — 系统用途（compaction、title 生成等），不直接与用户交互。
    Internal,
}

impl AgentRole {
    /// 是否为主 Agent。
    pub fn is_primary(&self) -> bool {
        matches!(self, Self::Primary)
    }

    /// 是否为子 Agent。
    pub fn is_sub_agent(&self) -> bool {
        matches!(self, Self::SubAgent)
    }

    /// 是否为内部 Agent。
    pub fn is_internal(&self) -> bool {
        matches!(self, Self::Internal)
    }
}

// ===========================================================================
// AgentModelRef
// ===========================================================================

/// 模型引用 — 轻量级标识，运行时解析为完整 ModelRef。
///
/// Agent 定义中不直接持有 API key、base_url 等敏感信息，
/// 而是通过引用方式在运行时由 ModelResolver 解析。
///
/// # Examples
///
/// ```
/// use katu_core::{AgentModelRef, ModelId, ProviderId};
///
/// // 继承调用者的模型
/// let inherit = AgentModelRef::Inherit;
///
/// // 按 ID 指定
/// let specific = AgentModelRef::by_id(
///     ModelId::new("gpt-4o"),
///     ProviderId::new("openai"),
/// );
///
/// // 按别名（如 "fast", "strong", "cheap"）
/// let alias = AgentModelRef::by_alias("fast");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentModelRef {
    /// 继承父 Agent / 调用者的模型配置。
    Inherit,
    /// 按 ID 精确指定模型和 provider。
    ById {
        model_id: ModelId,
        provider_id: ProviderId,
    },
    /// 按别名引用（运行时由配置映射到具体模型）。
    ByAlias {
        alias: String,
    },
}

impl AgentModelRef {
    /// 创建按 ID 指定的模型引用。
    pub fn by_id(model_id: ModelId, provider_id: ProviderId) -> Self {
        Self::ById {
            model_id,
            provider_id,
        }
    }

    /// 创建按别名指定的模型引用。
    pub fn by_alias(alias: impl Into<String>) -> Self {
        Self::ByAlias {
            alias: alias.into(),
        }
    }
}

// ===========================================================================
// ToolFilter
// ===========================================================================

/// 工具过滤规则 — 决定 Agent 可使用哪些工具。
///
/// 运行时 AgentRunner 根据此规则过滤 ToolRegistry 中的可用工具。
///
/// # Examples
///
/// ```
/// use katu_core::ToolFilter;
///
/// // 允许所有
/// let all = ToolFilter::AllowAll;
/// assert!(all.is_allowed("read_file"));
///
/// // 白名单
/// let allow = ToolFilter::allow_list(["read_file", "grep"]);
/// assert!(allow.is_allowed("read_file"));
/// assert!(!allow.is_allowed("bash"));
///
/// // 黑名单
/// let deny = ToolFilter::deny_list(["bash"]);
/// assert!(deny.is_allowed("read_file"));
/// assert!(!deny.is_allowed("bash"));
///
/// // 无工具
/// let none = ToolFilter::None;
/// assert!(!none.is_allowed("read_file"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolFilter {
    /// 允许所有已注册工具。
    #[default]
    AllowAll,
    /// 只允许列出的工具（白名单）。
    AllowList { tools: Vec<String> },
    /// 禁止列出的工具，其余允许（黑名单）。
    DenyList { tools: Vec<String> },
    /// 无工具 — 纯对话模式。
    None,
}

impl ToolFilter {
    /// 创建白名单过滤器。
    pub fn allow_list(tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::AllowList {
            tools: tools.into_iter().map(Into::into).collect(),
        }
    }

    /// 创建黑名单过滤器。
    pub fn deny_list(tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::DenyList {
            tools: tools.into_iter().map(Into::into).collect(),
        }
    }

    /// 判断指定工具名是否被允许。
    pub fn is_allowed(&self, tool_name: &str) -> bool {
        match self {
            Self::AllowAll => true,
            Self::AllowList { tools } => tools.iter().any(|t| t == tool_name),
            Self::DenyList { tools } => !tools.iter().any(|t| t == tool_name),
            Self::None => false,
        }
    }
}

// ===========================================================================
// AgentDefinition
// ===========================================================================

/// Agent 定义 — 描述 Agent 的完整静态配置。
///
/// 这是纯数据结构，不包含运行时行为。`katu-agent` 中的 AgentRunner
/// 根据此定义来驱动实际的 LLM 循环。
///
/// ## Builder 模式
/// 必填字段通过 `AgentDefinition::new()` 提供，可选字段通过 `with_*` 链式设置。
///
/// # Examples
///
/// ```
/// use katu_core::{AgentDefinition, AgentName, AgentRole, AgentModelRef, ToolFilter};
///
/// // 主编码 Agent
/// let build = AgentDefinition::new("build", AgentRole::Primary)
///     .with_description("Default coding agent")
///     .with_system_prompt("You are a coding assistant.")
///     .with_max_steps(50);
///
/// // 只读搜索 subagent
/// let explore = AgentDefinition::new("explore", AgentRole::SubAgent)
///     .with_description("Fast read-only search agent")
///     .with_system_prompt("You are a file search specialist.")
///     .with_model(AgentModelRef::by_alias("fast"))
///     .with_tool_filter(ToolFilter::allow_list(["read_file", "grep", "glob"]))
///     .with_max_steps(10);
///
/// // 内部 title 生成
/// let title = AgentDefinition::new("title", AgentRole::Internal)
///     .with_description("Generates conversation titles")
///     .with_system_prompt("Generate a short title for this conversation.")
///     .with_model(AgentModelRef::by_alias("cheap"))
///     .with_tool_filter(ToolFilter::None)
///     .with_max_steps(1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// 唯一标识名（snake_case）。
    pub name: AgentName,

    /// Agent 角色。
    pub role: AgentRole,

    /// 人类可读描述。
    ///
    /// 两个用途：
    /// 1. 主 Agent 选择 subagent 时，LLM 据此判断何时调用
    /// 2. 配置文件中的说明文本
    #[serde(default)]
    pub description: String,

    /// System prompt 片段（按顺序拼接）。
    ///
    /// 使用 Vec 支持组合式 prompt 构建（基础指令 + 项目规则 + 工具说明）。
    /// 运行时拼接为单个 system prompt 字符串发送给 LLM。
    #[serde(default)]
    pub system_prompt: Vec<String>,

    /// 模型引用 — None 表示继承调用者的模型配置。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<AgentModelRef>,

    /// 工具过滤规则。
    #[serde(default)]
    pub tool_filter: ToolFilter,

    /// 默认工具选择策略。
    ///
    /// None 表示使用 ToolChoice::Auto（模型自行决定）。
    /// 常用于 Internal Agent 强制 ToolChoice::None 以禁用工具。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Agent 级生成参数覆盖。
    ///
    /// AgentRunner 构建 LlmRequest 时合并优先级：
    /// Request > Agent > Model > Provider 默认。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationOptions>,

    /// 最大循环步数（一次 LLM 调用 → tool_call → result 为一步）。
    ///
    /// None 表示由 AgentRunner 全局配置决定。
    /// 防止无限循环的安全措施。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<u32>,

    /// 可调度的子 Agent 名称列表。
    ///
    /// 运行时由 AgentRegistry 解析为具体的 AgentDefinition。
    /// 空列表表示不可调度子 Agent。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sub_agents: Vec<AgentName>,

    /// 结构化输出 JSON Schema。
    ///
    /// 用于 SubAgent 返回结构化数据时约束 LLM 输出格式。
    /// None 表示自由文本输出。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,

    /// Provider 特有选项透传。
    ///
    /// 由 Provider adapter 直接消费，katu 框架层不解析。
    /// 例如：OpenAI 的 `service_tier`、Anthropic 的 `metadata` 等。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<serde_json::Value>,
}

impl AgentDefinition {
    /// 创建 Agent 定义 — 只需必填的 name 和 role。
    pub fn new(name: impl Into<String>, role: AgentRole) -> Self {
        Self {
            name: AgentName::new(name),
            role,
            description: String::new(),
            system_prompt: Vec::new(),
            model: None,
            tool_filter: ToolFilter::default(),
            tool_choice: None,
            generation: None,
            max_steps: None,
            sub_agents: Vec::new(),
            output_schema: None,
            provider_options: None,
        }
    }

    /// 设置描述。
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// 设置 system prompt（单段，替换已有内容）。
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = vec![prompt.into()];
        self
    }

    /// 追加 system prompt 片段。
    pub fn append_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt.push(prompt.into());
        self
    }

    /// 批量设置 system prompt 片段。
    pub fn with_system_prompts(mut self, prompts: Vec<String>) -> Self {
        self.system_prompt = prompts;
        self
    }

    /// 设置模型引用。
    pub fn with_model(mut self, model: AgentModelRef) -> Self {
        self.model = Some(model);
        self
    }

    /// 设置工具过滤规则。
    pub fn with_tool_filter(mut self, filter: ToolFilter) -> Self {
        self.tool_filter = filter;
        self
    }

    /// 设置默认工具选择策略。
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// 设置生成参数覆盖。
    pub fn with_generation(mut self, generation: GenerationOptions) -> Self {
        self.generation = Some(generation);
        self
    }

    /// 设置最大循环步数。
    pub fn with_max_steps(mut self, steps: u32) -> Self {
        self.max_steps = Some(steps);
        self
    }

    /// 设置可调度的子 Agent 列表。
    pub fn with_sub_agents(mut self, agents: Vec<AgentName>) -> Self {
        self.sub_agents = agents;
        self
    }

    /// 添加一个子 Agent。
    pub fn add_sub_agent(mut self, agent: impl Into<String>) -> Self {
        self.sub_agents.push(AgentName::new(agent));
        self
    }

    /// 设置结构化输出 schema。
    pub fn with_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// 设置 provider 透传选项。
    pub fn with_provider_options(mut self, options: serde_json::Value) -> Self {
        self.provider_options = Some(options);
        self
    }

    /// 获取拼接后的完整 system prompt。
    ///
    /// 多段之间以双换行连接。
    pub fn joined_system_prompt(&self) -> String {
        self.system_prompt.join("\n\n")
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- AgentName --

    #[test]
    fn test_agent_name_new() {
        let name = AgentName::new("explore");
        assert_eq!(name.as_str(), "explore");
        assert_eq!(name.to_string(), "explore");
    }

    #[test]
    fn test_agent_name_serde_roundtrip() {
        let name = AgentName::new("build");
        let json = serde_json::to_string(&name).unwrap();
        let restored: AgentName = serde_json::from_str(&json).unwrap();
        assert_eq!(name, restored);
    }

    // -- AgentRole --

    #[test]
    fn test_agent_role_predicates() {
        assert!(AgentRole::Primary.is_primary());
        assert!(!AgentRole::Primary.is_sub_agent());
        assert!(!AgentRole::Primary.is_internal());

        assert!(!AgentRole::SubAgent.is_primary());
        assert!(AgentRole::SubAgent.is_sub_agent());
        assert!(!AgentRole::SubAgent.is_internal());

        assert!(!AgentRole::Internal.is_primary());
        assert!(!AgentRole::Internal.is_sub_agent());
        assert!(AgentRole::Internal.is_internal());
    }

    #[test]
    fn test_agent_role_serde() {
        let json = serde_json::to_string(&AgentRole::SubAgent).unwrap();
        assert_eq!(json, r#""sub_agent""#);
        let restored: AgentRole = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, AgentRole::SubAgent);
    }

    // -- AgentModelRef --

    #[test]
    fn test_agent_model_ref_inherit() {
        let r = AgentModelRef::Inherit;
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""type":"inherit""#));
    }

    #[test]
    fn test_agent_model_ref_by_id() {
        let r = AgentModelRef::by_id(
            ModelId::new("gpt-4o"),
            ProviderId::new("openai"),
        );
        if let AgentModelRef::ById { model_id, provider_id } = &r {
            assert_eq!(model_id.as_str(), "gpt-4o");
            assert_eq!(provider_id.as_str(), "openai");
        } else {
            panic!("expected ById");
        }
    }

    #[test]
    fn test_agent_model_ref_by_alias() {
        let r = AgentModelRef::by_alias("fast");
        if let AgentModelRef::ByAlias { alias } = &r {
            assert_eq!(alias, "fast");
        } else {
            panic!("expected ByAlias");
        }
    }

    #[test]
    fn test_agent_model_ref_serde_roundtrip() {
        let refs = vec![
            AgentModelRef::Inherit,
            AgentModelRef::by_id(ModelId::new("claude-sonnet-4-20250514"), ProviderId::new("anthropic")),
            AgentModelRef::by_alias("cheap"),
        ];
        for r in refs {
            let json = serde_json::to_string(&r).unwrap();
            let restored: AgentModelRef = serde_json::from_str(&json).unwrap();
            assert_eq!(r, restored);
        }
    }

    // -- ToolFilter --

    #[test]
    fn test_tool_filter_allow_all() {
        let f = ToolFilter::AllowAll;
        assert!(f.is_allowed("anything"));
    }

    #[test]
    fn test_tool_filter_allow_list() {
        let f = ToolFilter::allow_list(["read_file", "grep"]);
        assert!(f.is_allowed("read_file"));
        assert!(f.is_allowed("grep"));
        assert!(!f.is_allowed("bash"));
    }

    #[test]
    fn test_tool_filter_deny_list() {
        let f = ToolFilter::deny_list(["bash", "write_file"]);
        assert!(f.is_allowed("read_file"));
        assert!(!f.is_allowed("bash"));
        assert!(!f.is_allowed("write_file"));
    }

    #[test]
    fn test_tool_filter_none() {
        let f = ToolFilter::None;
        assert!(!f.is_allowed("anything"));
    }

    #[test]
    fn test_tool_filter_default_is_allow_all() {
        assert_eq!(ToolFilter::default(), ToolFilter::AllowAll);
    }

    #[test]
    fn test_tool_filter_serde_roundtrip() {
        let filters = vec![
            ToolFilter::AllowAll,
            ToolFilter::allow_list(["read_file"]),
            ToolFilter::deny_list(["bash"]),
            ToolFilter::None,
        ];
        for f in filters {
            let json = serde_json::to_string(&f).unwrap();
            let restored: ToolFilter = serde_json::from_str(&json).unwrap();
            assert_eq!(f, restored);
        }
    }

    // -- AgentDefinition --

    #[test]
    fn test_agent_definition_minimal() {
        let agent = AgentDefinition::new("test", AgentRole::Primary);
        assert_eq!(agent.name.as_str(), "test");
        assert_eq!(agent.role, AgentRole::Primary);
        assert!(agent.description.is_empty());
        assert!(agent.system_prompt.is_empty());
        assert!(agent.model.is_none());
        assert_eq!(agent.tool_filter, ToolFilter::AllowAll);
        assert!(agent.tool_choice.is_none());
        assert!(agent.generation.is_none());
        assert!(agent.max_steps.is_none());
        assert!(agent.sub_agents.is_empty());
        assert!(agent.output_schema.is_none());
        assert!(agent.provider_options.is_none());
    }

    #[test]
    fn test_agent_definition_builder() {
        use crate::tool::ToolChoice;
        use crate::generation::GenerationOptions;

        let agent = AgentDefinition::new("explore", AgentRole::SubAgent)
            .with_description("Search agent")
            .with_system_prompt("You are a search specialist.")
            .append_system_prompt("Be thorough.")
            .with_model(AgentModelRef::by_alias("fast"))
            .with_tool_filter(ToolFilter::allow_list(["read_file", "grep"]))
            .with_tool_choice(ToolChoice::Auto)
            .with_generation(GenerationOptions::new().with_temperature(0.3))
            .with_max_steps(10)
            .add_sub_agent("deep_search");

        assert_eq!(agent.name.as_str(), "explore");
        assert_eq!(agent.role, AgentRole::SubAgent);
        assert_eq!(agent.description, "Search agent");
        assert_eq!(agent.system_prompt.len(), 2);
        assert_eq!(agent.joined_system_prompt(), "You are a search specialist.\n\nBe thorough.");
        assert!(agent.model.is_some());
        assert!(agent.tool_filter.is_allowed("read_file"));
        assert!(!agent.tool_filter.is_allowed("bash"));
        assert_eq!(agent.tool_choice, Some(ToolChoice::Auto));
        assert_eq!(agent.generation.as_ref().unwrap().temperature, Some(0.3));
        assert_eq!(agent.max_steps, Some(10));
        assert_eq!(agent.sub_agents.len(), 1);
        assert_eq!(agent.sub_agents[0].as_str(), "deep_search");
    }

    #[test]
    fn test_agent_definition_serde_roundtrip() {
        use crate::tool::ToolChoice;
        use crate::generation::GenerationOptions;

        let agent = AgentDefinition::new("build", AgentRole::Primary)
            .with_description("Coding agent")
            .with_system_prompt("Help with code.")
            .with_model(AgentModelRef::by_id(
                ModelId::new("gpt-4o"),
                ProviderId::new("openai"),
            ))
            .with_tool_filter(ToolFilter::deny_list(["dangerous_tool"]))
            .with_tool_choice(ToolChoice::Required)
            .with_generation(GenerationOptions::new().with_temperature(0.5).with_max_tokens(4096))
            .with_max_steps(50)
            .add_sub_agent("explore")
            .add_sub_agent("title")
            .with_output_schema(serde_json::json!({"type": "object"}))
            .with_provider_options(serde_json::json!({"service_tier": "default"}));

        let json = serde_json::to_string_pretty(&agent).unwrap();
        let restored: AgentDefinition = serde_json::from_str(&json).unwrap();

        assert_eq!(agent.name, restored.name);
        assert_eq!(agent.role, restored.role);
        assert_eq!(agent.description, restored.description);
        assert_eq!(agent.system_prompt, restored.system_prompt);
        assert_eq!(agent.model, restored.model);
        assert_eq!(agent.tool_filter, restored.tool_filter);
        assert_eq!(agent.tool_choice, restored.tool_choice);
        assert_eq!(agent.generation, restored.generation);
        assert_eq!(agent.max_steps, restored.max_steps);
        assert_eq!(agent.sub_agents, restored.sub_agents);
        assert_eq!(agent.output_schema, restored.output_schema);
        assert_eq!(agent.provider_options, restored.provider_options);
    }

    #[test]
    fn test_agent_definition_joined_prompt_empty() {
        let agent = AgentDefinition::new("empty", AgentRole::Internal);
        assert_eq!(agent.joined_system_prompt(), "");
    }

    #[test]
    fn test_agent_definition_joined_prompt_single() {
        let agent = AgentDefinition::new("t", AgentRole::Internal)
            .with_system_prompt("Hello");
        assert_eq!(agent.joined_system_prompt(), "Hello");
    }
}
