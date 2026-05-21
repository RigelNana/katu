//! # tool
//!
//! ## 职责
//! `BashTool` — Shell 命令执行工具的 `impl Tool`。
//!
//! ## 权限集成
//! ```text
//! check_permissions()
//!   ├─ Layer 1: HardDenyChecker → Deny（不可覆盖）
//!   ├─ Layer 2: ReadOnlyChecker → Allow（可配置关闭）
//!   └─ Layer 3: → Passthrough（交给框架 Ruleset）
//!
//! permission_request()
//!   └─ CommandDescriptor → PermissionRequest（细粒度 key + pattern）
//! ```

use async_trait::async_trait;
use serde_json::json;

use katu_core::permission::{PermissionRequest, PermissionResult};
use katu_core::tool::{
    ConcurrencyMode, ToolCallContext, ToolDefinition, ToolOutput,
};
use katu_core::{Result, Tool};

use super::command::CommandDescriptor;
use super::config::BashToolConfig;
use super::executor::ShellExecutor;
use super::policy::{HardDenyChecker, ReadOnlyChecker};

// ===========================================================================
// ToolDefinition (static)
// ===========================================================================

static BASH_TOOL_DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
    ToolDefinition::new(
        "bash",
        "Run a shell command in a non-interactive bash session. \
         Use this to execute system commands, run scripts, manage files, \
         and interact with development tools. Commands run in a \
         non-interactive environment with pagers disabled.",
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute. \
                        Avoid interactive commands that require user input. \
                        For long-running commands, consider adding timeouts."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120, max: 3600). \
                        The command will be killed if it exceeds this duration."
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command. \
                        Defaults to the project root."
                }
            },
            "required": ["command"]
        }),
    )
});

// ===========================================================================
// BashTool
// ===========================================================================

/// Shell 命令执行工具。
///
/// 实现 `Tool` trait，支持三层权限检查、细粒度权限请求、退出码语义化等。
///
/// # 构造
///
/// ```
/// use katu_tool::shell::{BashTool, BashToolConfig};
///
/// // 默认配置
/// let tool = BashTool::new();
///
/// // 自定义配置
/// let tool = BashTool::with_config(
///     BashToolConfig::default()
///         .with_allow_readonly(false)
///         .with_permission_prefix("shell")
/// );
/// ```
///
/// # 权限
///
/// `check_permissions()` 执行三层检查：
/// 1. **硬拦截** — 绝对禁止的命令模式（如 `rm -rf /`），返回 `Deny`
/// 2. **只读放行** — 纯只读命令（如 `ls`, `grep`），返回 `Allow`
/// 3. **Passthrough** — 其他命令交给框架 Ruleset 决策
///
/// `permission_request()` 构造细粒度权限请求：
/// - permission key: `"bash:git"`, `"bash:npm"` 等
/// - pattern: 完整命令文本
/// - always_allow_patterns: 宽泛模式（如 `"git push *"`）
#[derive(Debug)]
pub struct BashTool {
    config: BashToolConfig,
    hard_deny: HardDenyChecker,
    readonly: ReadOnlyChecker,
    executor: ShellExecutor,
}

impl BashTool {
    /// 使用默认配置创建。
    pub fn new() -> Self {
        Self::with_config(BashToolConfig::default())
    }

    /// 使用自定义配置创建。
    pub fn with_config(config: BashToolConfig) -> Self {
        let hard_deny = if config.disable_builtin_hard_deny {
            let mut checker = HardDenyChecker::custom_only(Vec::new());
            for rule in &config.extra_hard_deny {
                checker.add_rule(rule.clone());
            }
            checker
        } else {
            let mut checker = HardDenyChecker::default();
            for rule in &config.extra_hard_deny {
                checker.add_rule(rule.clone());
            }
            checker
        };

        let readonly = ReadOnlyChecker::new(config.extra_readonly.clone());

        let executor = ShellExecutor::new()
            .with_output_limits(
                config.output_head_bytes.unwrap_or(32 * 1024),
                config.output_tail_bytes.unwrap_or(32 * 1024),
            );

        Self {
            config,
            hard_deny,
            readonly,
            executor,
        }
    }

    /// 获取配置的引用。
    pub fn config(&self) -> &BashToolConfig {
        &self.config
    }

    /// 构建权限 key — 使用配置的前缀。
    fn build_permission_key(&self, descriptor: &CommandDescriptor) -> String {
        let primary = descriptor
            .commands()
            .iter()
            .max_by_key(|c| c.risk())
            .map(|c| c.name());

        match primary {
            Some(name) if !name.is_empty() => {
                format!("{}:{name}", self.config.permission_prefix)
            }
            _ => self.config.permission_prefix.clone(),
        }
    }
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for BashTool {
    fn definition(&self) -> &ToolDefinition {
        &BASH_TOOL_DEF
    }

    fn concurrency_mode(&self) -> ConcurrencyMode {
        ConcurrencyMode::Exclusive
    }

    fn permission_key(&self) -> &str {
        &self.config.permission_prefix
    }

    fn check_permissions(
        &self,
        args: &serde_json::Value,
        _ctx: &ToolCallContext,
    ) -> PermissionResult {
        let command = match args["command"].as_str() {
            Some(cmd) if !cmd.trim().is_empty() => cmd,
            _ => return PermissionResult::Passthrough,
        };

        let descriptor = CommandDescriptor::parse(command);

        // Layer 1: 硬拦截 — 不可覆盖
        if let Some(reason) = self.hard_deny.check(&descriptor) {
            return PermissionResult::Deny {
                message: reason,
            };
        }

        // Layer 2: 只读放行
        if self.config.allow_readonly && self.readonly.is_readonly(&descriptor) {
            return PermissionResult::Allow;
        }

        // Layer 3: 交给框架
        PermissionResult::Passthrough
    }

    fn permission_request(
        &self,
        args: &serde_json::Value,
        _ctx: &ToolCallContext,
    ) -> Option<PermissionRequest> {
        let command = args["command"].as_str()?;
        if command.trim().is_empty() {
            return None;
        }

        let descriptor = CommandDescriptor::parse(command);
        let key = self.build_permission_key(&descriptor);

        Some(
            PermissionRequest::new(&key, descriptor.content_for_matching())
                .with_tool_name("bash")
                .with_always_allow(vec![descriptor.always_allow_pattern()])
                .with_metadata(json!({
                    "risk_level": descriptor.risk().as_str(),
                    "sub_commands": descriptor.command_names(),
                })),
        )
    }

    async fn validate(
        &self,
        args: &serde_json::Value,
        _ctx: &ToolCallContext,
    ) -> Result<()> {
        let command = args["command"]
            .as_str()
            .unwrap_or("");

        if command.trim().is_empty() {
            return Err(katu_core::Error::tool(
                "bash",
                _ctx.call_id.clone(),
                "command must not be empty",
            ));
        }

        // 超时范围检查
        if let Some(timeout) = args["timeout"].as_u64() {
            if timeout == 0 || timeout > self.config.max_timeout_secs {
                return Err(katu_core::Error::tool(
                    "bash",
                    _ctx.call_id.clone(),
                    format!(
                        "timeout must be between 1 and {} seconds, got {timeout}",
                        self.config.max_timeout_secs
                    ),
                ));
            }
        }

        Ok(())
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: &ToolCallContext,
    ) -> Result<ToolOutput> {
        let command = args["command"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let timeout_secs = args["timeout"]
            .as_u64()
            .unwrap_or(self.config.default_timeout_secs)
            .min(self.config.max_timeout_secs);

        let cwd = args["cwd"]
            .as_str()
            .unwrap_or(".")
            .to_string();

        // 解析 cwd 路径
        let cwd_path = std::path::Path::new(&cwd);
        let cwd_path = if cwd_path.is_relative() {
            std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join(cwd_path)
        } else {
            cwd_path.to_path_buf()
        };

        // 检查 cwd 是否存在
        if !cwd_path.is_dir() {
            return Err(katu_core::Error::tool(
                "bash",
                _ctx.call_id.clone(),
                format!("Working directory does not exist: {}", cwd_path.display()),
            ));
        }

        // 直接使用 ToolCallContext 的 CancellationToken（child token 实现层级取消）
        let cancel = _ctx.cancellation.child_token();

        let result = self.executor.run(
            &command,
            &cwd_path,
            std::time::Duration::from_secs(timeout_secs),
            cancel,
        ).await;

        let title = format!("bash: {}", truncate_for_title(&command, 60));

        if result.timed_out {
            return Err(katu_core::Error::tool(
                "bash",
                _ctx.call_id.clone(),
                format!(
                    "Command timed out after {timeout_secs} seconds\n\n{}",
                    result.stdout()
                ),
            ));
        }

        if result.cancelled {
            return Err(katu_core::Error::tool(
                "bash",
                _ctx.call_id.clone(),
                "Command was cancelled",
            ));
        }

        let exit_code = result.exit_code.unwrap_or(-1);
        let output_text = result.stdout().to_string();
        let output_text = if output_text.is_empty() {
            "(no output)".to_string()
        } else {
            output_text
        };

        let is_error = exit_code != 0;
        let content = if is_error {
            format!("{output_text}\n\nCommand exited with code {exit_code}")
        } else {
            output_text
        };

        let output = if is_error {
            ToolOutput::error(content)
        } else {
            ToolOutput::success(content)
        };

        Ok(output
            .with_title(title)
            .with_metadata(json!({
                "command": command,
                "exit_code": exit_code,
                "timed_out": result.timed_out,
                "total_bytes": result.output.total_bytes(),
                "total_lines": result.output.total_lines(),
                "truncated": result.output.is_truncated(),
            })))
    }
}

/// 截断命令文本用于 title 显示。
fn truncate_for_title(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use katu_core::types::ToolCallId;

    fn ctx() -> ToolCallContext {
        ToolCallContext::new(ToolCallId::new("test"))
    }

    // ── check_permissions ──

    #[test]
    fn test_hard_deny_rm_rf_root() {
        let tool = BashTool::new();
        let args = json!({"command": "rm -rf /"});
        let result = tool.check_permissions(&args, &ctx());
        assert!(result.is_deny());
    }

    #[test]
    fn test_hard_deny_in_chain() {
        let tool = BashTool::new();
        let args = json!({"command": "echo hi && rm -rf /"});
        let result = tool.check_permissions(&args, &ctx());
        assert!(result.is_deny());
    }

    #[test]
    fn test_readonly_allow() {
        let tool = BashTool::new();
        let args = json!({"command": "ls -la"});
        let result = tool.check_permissions(&args, &ctx());
        assert!(result.is_allow());
    }

    #[test]
    fn test_readonly_pipe_allow() {
        let tool = BashTool::new();
        let args = json!({"command": "cat foo.txt | grep bar | wc -l"});
        let result = tool.check_permissions(&args, &ctx());
        assert!(result.is_allow());
    }

    #[test]
    fn test_readonly_disabled() {
        let tool = BashTool::with_config(
            BashToolConfig::default().with_allow_readonly(false),
        );
        let args = json!({"command": "ls -la"});
        let result = tool.check_permissions(&args, &ctx());
        assert!(result.is_passthrough());
    }

    #[test]
    fn test_write_command_passthrough() {
        let tool = BashTool::new();
        let args = json!({"command": "cargo build"});
        let result = tool.check_permissions(&args, &ctx());
        assert!(result.is_passthrough());
    }

    #[test]
    fn test_unknown_command_passthrough() {
        let tool = BashTool::new();
        let args = json!({"command": "my_custom_script --foo"});
        let result = tool.check_permissions(&args, &ctx());
        assert!(result.is_passthrough());
    }

    #[test]
    fn test_empty_command_passthrough() {
        let tool = BashTool::new();
        let args = json!({"command": ""});
        let result = tool.check_permissions(&args, &ctx());
        assert!(result.is_passthrough());
    }

    // ── permission_request ──

    #[test]
    fn test_permission_request_git() {
        let tool = BashTool::new();
        let args = json!({"command": "git push origin main"});
        let req = tool.permission_request(&args, &ctx()).unwrap();
        assert_eq!(req.permission, "bash:git");
        assert_eq!(req.patterns, vec!["git push origin main"]);
        assert_eq!(req.always_allow_patterns, vec!["git push *"]);
    }

    #[test]
    fn test_permission_request_custom_prefix() {
        let tool = BashTool::with_config(
            BashToolConfig::default().with_permission_prefix("shell"),
        );
        let args = json!({"command": "npm run build"});
        let req = tool.permission_request(&args, &ctx()).unwrap();
        assert_eq!(req.permission, "shell:npm");
    }

    #[test]
    fn test_permission_request_empty() {
        let tool = BashTool::new();
        let args = json!({"command": ""});
        assert!(tool.permission_request(&args, &ctx()).is_none());
    }

    #[test]
    fn test_permission_request_metadata() {
        let tool = BashTool::new();
        let args = json!({"command": "curl https://example.com"});
        let req = tool.permission_request(&args, &ctx()).unwrap();
        assert_eq!(req.metadata["risk_level"], "network_egress");
    }

    // ── validate ──

    #[tokio::test]
    async fn test_validate_empty_command() {
        let tool = BashTool::new();
        let args = json!({"command": ""});
        assert!(tool.validate(&args, &ctx()).await.is_err());
    }

    #[tokio::test]
    async fn test_validate_timeout_zero() {
        let tool = BashTool::new();
        let args = json!({"command": "ls", "timeout": 0});
        assert!(tool.validate(&args, &ctx()).await.is_err());
    }

    #[tokio::test]
    async fn test_validate_timeout_too_large() {
        let tool = BashTool::new();
        let args = json!({"command": "ls", "timeout": 99999});
        assert!(tool.validate(&args, &ctx()).await.is_err());
    }

    #[tokio::test]
    async fn test_validate_ok() {
        let tool = BashTool::new();
        let args = json!({"command": "ls -la", "timeout": 30});
        assert!(tool.validate(&args, &ctx()).await.is_ok());
    }

    // ── execute ──

    #[tokio::test]
    async fn test_execute_echo() {
        let tool = BashTool::new();
        let args = json!({"command": "echo hello"});
        let result = tool.execute(args, &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("hello"));
    }

    #[tokio::test]
    async fn test_execute_exit_code() {
        let tool = BashTool::new();
        let args = json!({"command": "exit 1"});
        let result = tool.execute(args, &ctx()).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("exited with code 1"));
    }

    #[tokio::test]
    async fn test_execute_cwd() {
        let tool = BashTool::new();
        let args = json!({"command": "pwd", "cwd": "/tmp"});
        let result = tool.execute(args, &ctx()).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("/tmp") || result.content.contains("/private/tmp"));
    }

    #[tokio::test]
    async fn test_execute_nonexistent_cwd() {
        let tool = BashTool::new();
        let args = json!({"command": "echo hi", "cwd": "/nonexistent_dir_xyz"});
        let result = tool.execute(args, &ctx()).await;
        assert!(result.is_err());
    }

    // ── concurrency_mode ──

    #[test]
    fn test_exclusive_mode() {
        let tool = BashTool::new();
        assert_eq!(tool.concurrency_mode(), ConcurrencyMode::Exclusive);
    }

    // ── definition ──

    #[test]
    fn test_definition() {
        let tool = BashTool::new();
        assert_eq!(tool.definition().name, "bash");
        assert!(tool.definition().parameters["properties"]["command"].is_object());
    }
}
