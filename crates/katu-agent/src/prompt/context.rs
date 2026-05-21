//! # prompt::context
//!
//! ## 职责
//! 定义 Prompt 构建时的输入上下文。
//!
//! ## 对外接口
//! - `EnvironmentInfo` — 运行时环境快照
//! - `PromptContext` — 传递给段生成器的上下文
//!
//! ## 调用者
//! - `prompt::builder` — PromptBuilder 消费 PromptContext
//! - `prompt::builtin` — 内置段生成器读取上下文

use std::path::PathBuf;

use chrono::Utc;

use katu_core::agent::AgentDefinition;
use katu_core::tool::ToolDefinition;
use katu_core::types::{ModelId, ProviderId};

// ===========================================================================
// EnvironmentInfo
// ===========================================================================

/// 运行时环境信息 — 注入到 system prompt。
///
/// 在会话创建时收集，同一会话内保持不变。
/// 参考 OpenCode 的 `<env>` 标签和 Claude Code 的 `computeSimpleEnvInfo`。
///
/// # Examples
///
/// ```
/// use katu_agent::prompt::EnvironmentInfo;
///
/// let env = EnvironmentInfo::new("/home/user/project", "linux")
///     .with_git_repo(true)
///     .with_shell("zsh");
///
/// assert!(env.is_git_repo);
/// assert_eq!(env.shell.as_deref(), Some("zsh"));
/// ```
#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    /// 工作目录。
    pub cwd: PathBuf,
    /// 项目根目录（workspace root）。
    pub workspace_root: Option<PathBuf>,
    /// 操作系统（linux / macos / windows）。
    pub platform: String,
    /// 是否为 git 仓库。
    pub is_git_repo: bool,
    /// 当前日期（YYYY-MM-DD）。
    pub date: String,
    /// Shell 类型（bash / zsh / fish...）。
    pub shell: Option<String>,
}

impl EnvironmentInfo {
    /// 创建环境信息 — 必填 cwd 和 platform，日期自动填充为当前日期。
    pub fn new(cwd: impl Into<PathBuf>, platform: impl Into<String>) -> Self {
        Self {
            cwd: cwd.into(),
            workspace_root: None,
            platform: platform.into(),
            is_git_repo: false,
            date: Utc::now().format("%Y-%m-%d").to_string(),
            shell: None,
        }
    }

    /// 设置 workspace 根目录。
    pub fn with_workspace_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(root.into());
        self
    }

    /// 设置 git 仓库标记。
    pub fn with_git_repo(mut self, is_git: bool) -> Self {
        self.is_git_repo = is_git;
        self
    }

    /// 设置日期（覆盖默认的当前日期）。
    pub fn with_date(mut self, date: impl Into<String>) -> Self {
        self.date = date.into();
        self
    }

    /// 设置 shell 类型。
    pub fn with_shell(mut self, shell: impl Into<String>) -> Self {
        self.shell = Some(shell.into());
        self
    }
}

// ===========================================================================
// PromptContext
// ===========================================================================

/// Prompt 构建上下文 — 传递给段生成器的所有运行时信息。
///
/// 由 AgentRunner 在每轮 LLM 调用前构建，提供给 `PromptSectionProvider`。
/// 使用 Builder 模式：必填字段通过 `new()`，可选字段通过 `with_*()` 设置。
///
/// # Examples
///
/// ```
/// use katu_agent::prompt::{PromptContext, EnvironmentInfo};
/// use katu_core::{AgentDefinition, AgentRole, ModelId, ProviderId, ToolDefinition};
///
/// let agent = AgentDefinition::new("build", AgentRole::Primary);
/// let model_id = ModelId::new("gpt-4o");
/// let provider_id = ProviderId::new("openai");
/// let env = EnvironmentInfo::new("/tmp", "linux");
/// let tools = vec![ToolDefinition::no_params("bash", "Run shell command")];
///
/// let ctx = PromptContext::new(&agent, &model_id, &provider_id, &env)
///     .with_tools(&tools)
///     .with_step_count(3);
///
/// assert_eq!(ctx.step_count, 3);
/// assert_eq!(ctx.tools.len(), 1);
/// ```
pub struct PromptContext<'a> {
    /// Agent 静态配置。
    pub agent: &'a AgentDefinition,
    /// 当前模型标识。
    pub model_id: &'a ModelId,
    /// Provider 标识（用于 provider 特定 prompt 选择）。
    pub provider_id: &'a ProviderId,
    /// 当前可用工具定义。
    pub tools: &'a [ToolDefinition],
    /// 环境信息。
    pub environment: &'a EnvironmentInfo,
    /// 用户自定义指令（来自项目规则文件等）。
    pub user_instructions: &'a [String],
    /// 当前会话步数。
    pub step_count: u32,
    /// 当前消息数。
    pub message_count: usize,
}

impl<'a> PromptContext<'a> {
    /// 创建上下文 — 必填字段。
    pub fn new(
        agent: &'a AgentDefinition,
        model_id: &'a ModelId,
        provider_id: &'a ProviderId,
        environment: &'a EnvironmentInfo,
    ) -> Self {
        Self {
            agent,
            model_id,
            provider_id,
            tools: &[],
            environment,
            user_instructions: &[],
            step_count: 0,
            message_count: 0,
        }
    }

    /// 设置当前可用工具。
    pub fn with_tools(mut self, tools: &'a [ToolDefinition]) -> Self {
        self.tools = tools;
        self
    }

    /// 设置用户自定义指令。
    pub fn with_user_instructions(mut self, instructions: &'a [String]) -> Self {
        self.user_instructions = instructions;
        self
    }

    /// 设置当前步数。
    pub fn with_step_count(mut self, count: u32) -> Self {
        self.step_count = count;
        self
    }

    /// 设置当前消息数。
    pub fn with_message_count(mut self, count: usize) -> Self {
        self.message_count = count;
        self
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use katu_core::AgentRole;

    #[test]
    fn test_environment_info_new() {
        let env = EnvironmentInfo::new("/tmp", "linux");
        assert_eq!(env.cwd, PathBuf::from("/tmp"));
        assert_eq!(env.platform, "linux");
        assert!(!env.is_git_repo);
        assert!(env.workspace_root.is_none());
        assert!(env.shell.is_none());
        assert!(!env.date.is_empty());
    }

    #[test]
    fn test_environment_info_builder() {
        let env = EnvironmentInfo::new("/home/user/project", "macos")
            .with_workspace_root("/home/user/project")
            .with_git_repo(true)
            .with_date("2025-01-01")
            .with_shell("zsh");

        assert_eq!(env.cwd, PathBuf::from("/home/user/project"));
        assert_eq!(
            env.workspace_root,
            Some(PathBuf::from("/home/user/project"))
        );
        assert!(env.is_git_repo);
        assert_eq!(env.date, "2025-01-01");
        assert_eq!(env.shell.as_deref(), Some("zsh"));
    }

    #[test]
    fn test_prompt_context_new() {
        let agent = AgentDefinition::new("test", AgentRole::Primary);
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = EnvironmentInfo::new("/tmp", "linux");

        let ctx = PromptContext::new(&agent, &model_id, &provider_id, &env);

        assert_eq!(ctx.agent.name.as_str(), "test");
        assert!(ctx.tools.is_empty());
        assert!(ctx.user_instructions.is_empty());
        assert_eq!(ctx.step_count, 0);
        assert_eq!(ctx.message_count, 0);
    }

    #[test]
    fn test_prompt_context_builder() {
        let agent = AgentDefinition::new("build", AgentRole::Primary);
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = EnvironmentInfo::new("/tmp", "linux");
        let tools = vec![ToolDefinition::no_params("bash", "Run shell command")];
        let instructions = vec!["Rule 1".to_string()];

        let ctx = PromptContext::new(&agent, &model_id, &provider_id, &env)
            .with_tools(&tools)
            .with_user_instructions(&instructions)
            .with_step_count(5)
            .with_message_count(10);

        assert_eq!(ctx.tools.len(), 1);
        assert_eq!(ctx.user_instructions.len(), 1);
        assert_eq!(ctx.step_count, 5);
        assert_eq!(ctx.message_count, 10);
    }
}
