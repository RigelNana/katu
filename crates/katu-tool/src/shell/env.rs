//! # env
//!
//! ## 职责
//! 非交互式 Shell 环境变量 — 禁用 pager、编辑器提示、进度条等。
//!
//! ## 参考
//! - oh-my-pi `non-interactive-env.ts`
//! - claude-code `subprocessEnv.ts`

use std::collections::HashMap;

// ===========================================================================
// NonInteractiveEnv
// ===========================================================================

/// 非交互式环境变量集合。
///
/// 注入到子进程中，确保命令不会阻塞在 pager、编辑器或交互式提示上。
///
/// # Examples
///
/// ```
/// use katu_tool::shell::env::NonInteractiveEnv;
///
/// let env = NonInteractiveEnv::default();
/// assert_eq!(env.vars().get("PAGER"), Some(&"cat".to_string()));
/// assert_eq!(env.vars().get("TERM"), Some(&"dumb".to_string()));
///
/// let custom = NonInteractiveEnv::default()
///     .with_var("MY_VAR", "value")
///     .without_var("CI");
/// assert_eq!(custom.vars().get("MY_VAR"), Some(&"value".to_string()));
/// assert!(!custom.vars().contains_key("CI"));
/// ```
#[derive(Debug, Clone)]
pub struct NonInteractiveEnv {
    vars: HashMap<String, String>,
}

impl NonInteractiveEnv {
    /// 创建空环境。
    pub fn empty() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    /// 返回环境变量的引用。
    pub fn vars(&self) -> &HashMap<String, String> {
        &self.vars
    }

    /// 消费并返回环境变量。
    pub fn into_vars(self) -> HashMap<String, String> {
        self.vars
    }

    /// 添加一个环境变量。
    pub fn with_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.vars.insert(key.into(), value.into());
        self
    }

    /// 移除一个环境变量。
    pub fn without_var(mut self, key: &str) -> Self {
        self.vars.remove(key);
        self
    }

    /// 将 vars 合并到一个已有的环境变量 map（覆盖同名 key）。
    pub fn apply_to(&self, target: &mut HashMap<String, String>) {
        for (k, v) in &self.vars {
            target.insert(k.clone(), v.clone());
        }
    }

    /// 返回适用于 `std::process::Command::envs()` 的迭代器。
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.vars.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

impl Default for NonInteractiveEnv {
    /// 内置默认非交互式环境变量。
    ///
    /// 涵盖：
    /// - Pager 禁用（git, man, psql, bat, delta, gh 等）
    /// - 终端特性降级（TERM=dumb, NO_COLOR=1）
    /// - 编辑器/凭证提示禁用（GIT_EDITOR, VISUAL, EDITOR, SSH_ASKPASS）
    /// - 包管理器静默模式（npm, pnpm, yarn, cargo, pip 等）
    /// - CI 标记（CI=1 — 许多工具据此跳过交互式行为）
    fn default() -> Self {
        let vars: Vec<(&str, &str)> = vec![
            // ── Pager 禁用 ──
            ("PAGER", "cat"),
            ("GIT_PAGER", "cat"),
            ("MANPAGER", "cat"),
            ("SYSTEMD_PAGER", "cat"),
            ("BAT_PAGER", "cat"),
            ("DELTA_PAGER", "cat"),
            ("GH_PAGER", "cat"),
            ("GLAB_PAGER", "cat"),
            ("PSQL_PAGER", "cat"),
            ("MYSQL_PAGER", "cat"),
            ("AWS_PAGER", ""),
            ("HOMEBREW_PAGER", "cat"),
            ("LESS", "FRX"),
            // ── 终端特性 ──
            ("TERM", "dumb"),
            ("GPG_TTY", "not a tty"),
            ("NO_COLOR", "1"),
            ("PYTHONUNBUFFERED", "1"),
            // ── 编辑器 & 凭证提示 ──
            ("GIT_EDITOR", "true"),
            ("VISUAL", "true"),
            ("EDITOR", "true"),
            ("GIT_TERMINAL_PROMPT", "0"),
            ("SSH_ASKPASS", "/usr/bin/false"),
            // ── CI 标记 ──
            ("CI", "1"),
            // ── npm ──
            ("npm_config_yes", "true"),
            ("npm_config_update_notifier", "false"),
            ("npm_config_fund", "false"),
            ("npm_config_audit", "false"),
            ("npm_config_progress", "false"),
            // ── pnpm ──
            ("PNPM_DISABLE_SELF_UPDATE_CHECK", "true"),
            ("PNPM_UPDATE_NOTIFIER", "false"),
            // ── yarn ──
            ("YARN_ENABLE_TELEMETRY", "0"),
            ("YARN_ENABLE_PROGRESS_BARS", "0"),
            // ── Cargo ──
            ("CARGO_TERM_PROGRESS_WHEN", "never"),
            // ── 系统包管理 ──
            ("DEBIAN_FRONTEND", "noninteractive"),
            // ── pip ──
            ("PIP_NO_INPUT", "1"),
            ("PIP_DISABLE_PIP_VERSION_CHECK", "1"),
            // ── Terraform ──
            ("TF_INPUT", "0"),
            ("TF_IN_AUTOMATION", "1"),
            // ── GitHub CLI ──
            ("GH_PROMPT_DISABLED", "1"),
            // ── Composer (PHP) ──
            ("COMPOSER_NO_INTERACTION", "1"),
            // ── Google Cloud SDK ──
            ("CLOUDSDK_CORE_DISABLE_PROMPTS", "1"),
            // ── katu 标记 — 让脚本可以检测自身是否被 katu 调用 ──
            ("KATU", "1"),
        ];

        Self {
            vars: vars
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_has_pager() {
        let env = NonInteractiveEnv::default();
        assert_eq!(env.vars().get("PAGER"), Some(&"cat".to_string()));
        assert_eq!(env.vars().get("GIT_PAGER"), Some(&"cat".to_string()));
    }

    #[test]
    fn test_default_has_term_dumb() {
        let env = NonInteractiveEnv::default();
        assert_eq!(env.vars().get("TERM"), Some(&"dumb".to_string()));
    }

    #[test]
    fn test_default_has_ci() {
        let env = NonInteractiveEnv::default();
        assert_eq!(env.vars().get("CI"), Some(&"1".to_string()));
    }

    #[test]
    fn test_default_has_katu_marker() {
        let env = NonInteractiveEnv::default();
        assert_eq!(env.vars().get("KATU"), Some(&"1".to_string()));
    }

    #[test]
    fn test_with_var() {
        let env = NonInteractiveEnv::default().with_var("FOO", "bar");
        assert_eq!(env.vars().get("FOO"), Some(&"bar".to_string()));
    }

    #[test]
    fn test_without_var() {
        let env = NonInteractiveEnv::default().without_var("CI");
        assert!(!env.vars().contains_key("CI"));
    }

    #[test]
    fn test_apply_to() {
        let env = NonInteractiveEnv::empty().with_var("A", "1").with_var("B", "2");
        let mut target = HashMap::new();
        target.insert("B".to_string(), "old".to_string());
        target.insert("C".to_string(), "3".to_string());
        env.apply_to(&mut target);
        assert_eq!(target.get("A"), Some(&"1".to_string()));
        assert_eq!(target.get("B"), Some(&"2".to_string()));
        assert_eq!(target.get("C"), Some(&"3".to_string()));
    }

    #[test]
    fn test_empty() {
        let env = NonInteractiveEnv::empty();
        assert!(env.vars().is_empty());
    }

    #[test]
    fn test_into_vars() {
        let env = NonInteractiveEnv::empty().with_var("X", "1");
        let map = env.into_vars();
        assert_eq!(map.get("X"), Some(&"1".to_string()));
    }
}
