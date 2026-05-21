//! # config
//!
//! ## 职责
//! Bash 工具配置 — 库使用者可据此控制权限策略与执行行为。

use super::policy::HardDenyRule;

// ===========================================================================
// BashToolConfig
// ===========================================================================

/// Bash 工具配置。
///
/// 库使用者通过此结构控制 BashTool 的权限策略和执行行为。
/// 使用 builder 模式构造。
///
/// # Examples
///
/// ```
/// use katu_tool::shell::BashToolConfig;
///
/// let config = BashToolConfig::default();
/// assert!(config.allow_readonly);
/// assert!(!config.disable_builtin_hard_deny);
///
/// let config = BashToolConfig::default()
///     .with_allow_readonly(false)
///     .with_permission_prefix("shell");
/// assert!(!config.allow_readonly);
/// assert_eq!(config.permission_prefix, "shell");
/// ```
#[derive(Debug, Clone)]
pub struct BashToolConfig {
    /// 是否自动放行只读命令（默认 `true`）。
    ///
    /// 当 `true` 时，只包含只读命令的输入会在 `check_permissions()` 中
    /// 直接返回 `Allow`，不经过 Ruleset。
    pub allow_readonly: bool,

    /// 额外的硬拦截规则（与内置规则合并）。
    pub extra_hard_deny: Vec<HardDenyRule>,

    /// 额外的只读命令白名单（追加到内置列表）。
    pub extra_readonly: Vec<String>,

    /// 禁用内置硬拦截规则（仅在沙箱环境中使用）。
    ///
    /// 当 `true` 时，只有 `extra_hard_deny` 中的规则生效。
    pub disable_builtin_hard_deny: bool,

    /// 权限 key 前缀（默认 `"bash"`）。
    ///
    /// 影响 `permission_key()` 和 `permission_request()` 返回的 key。
    /// 例如设为 `"shell"` 时，key 变为 `"shell:git"` 而非 `"bash:git"`。
    pub permission_prefix: String,

    /// 默认超时（秒）。
    pub default_timeout_secs: u64,

    /// 最大超时（秒）。
    pub max_timeout_secs: u64,

    /// 输出 head 最大字节数（默认 32KB）。
    pub output_head_bytes: Option<usize>,

    /// 输出 tail 最大字节数（默认 32KB）。
    pub output_tail_bytes: Option<usize>,
}

impl Default for BashToolConfig {
    fn default() -> Self {
        Self {
            allow_readonly: true,
            extra_hard_deny: Vec::new(),
            extra_readonly: Vec::new(),
            disable_builtin_hard_deny: false,
            permission_prefix: "bash".into(),
            default_timeout_secs: 120,
            max_timeout_secs: 3600,
            output_head_bytes: None,
            output_tail_bytes: None,
        }
    }
}

impl BashToolConfig {
    /// 设置是否自动放行只读命令。
    pub fn with_allow_readonly(mut self, allow: bool) -> Self {
        self.allow_readonly = allow;
        self
    }

    /// 添加额外的硬拦截规则。
    pub fn with_extra_hard_deny(mut self, rules: Vec<HardDenyRule>) -> Self {
        self.extra_hard_deny = rules;
        self
    }

    /// 添加额外的只读命令。
    pub fn with_extra_readonly(mut self, commands: Vec<String>) -> Self {
        self.extra_readonly = commands;
        self
    }

    /// 禁用内置硬拦截规则。
    pub fn with_disable_builtin_hard_deny(mut self, disable: bool) -> Self {
        self.disable_builtin_hard_deny = disable;
        self
    }

    /// 设置权限 key 前缀。
    pub fn with_permission_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.permission_prefix = prefix.into();
        self
    }

    /// 设置默认超时。
    pub fn with_default_timeout_secs(mut self, secs: u64) -> Self {
        self.default_timeout_secs = secs;
        self
    }

    /// 设置最大超时。
    pub fn with_max_timeout_secs(mut self, secs: u64) -> Self {
        self.max_timeout_secs = secs;
        self
    }

    /// 设置输出截断限制。
    pub fn with_output_limits(mut self, head_bytes: usize, tail_bytes: usize) -> Self {
        self.output_head_bytes = Some(head_bytes);
        self.output_tail_bytes = Some(tail_bytes);
        self
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BashToolConfig::default();
        assert!(config.allow_readonly);
        assert!(!config.disable_builtin_hard_deny);
        assert_eq!(config.permission_prefix, "bash");
        assert_eq!(config.default_timeout_secs, 120);
        assert_eq!(config.max_timeout_secs, 3600);
        assert!(config.extra_hard_deny.is_empty());
        assert!(config.extra_readonly.is_empty());
    }

    #[test]
    fn test_builder() {
        let config = BashToolConfig::default()
            .with_allow_readonly(false)
            .with_permission_prefix("shell")
            .with_default_timeout_secs(60)
            .with_max_timeout_secs(600);

        assert!(!config.allow_readonly);
        assert_eq!(config.permission_prefix, "shell");
        assert_eq!(config.default_timeout_secs, 60);
        assert_eq!(config.max_timeout_secs, 600);
    }

    #[test]
    fn test_extra_rules() {
        let config = BashToolConfig::default()
            .with_extra_hard_deny(vec![
                HardDenyRule::new("docker", "docker rm*", "no"),
            ])
            .with_extra_readonly(vec!["my_tool".into()]);

        assert_eq!(config.extra_hard_deny.len(), 1);
        assert_eq!(config.extra_readonly, vec!["my_tool"]);
    }
}
