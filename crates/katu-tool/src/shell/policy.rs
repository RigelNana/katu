//! # policy
//!
//! ## 职责
//! Shell 命令安全策略 — 硬拦截规则与只读命令白名单。
//!
//! ## 设计
//! - **HardDenyChecker** — 绝对禁止的命令模式，不可被任何规则/用户覆盖
//! - **ReadOnlyChecker** — 安全的只读命令，可配置自动放行
//!
//! ## 参考
//! - Claude-Code `readOnlyValidation.ts`（~1991 行只读白名单 + flag 验证）
//! - Oh-My-Pi `non-interactive-env.ts`（环境变量安全策略）

use super::command::{CommandDescriptor, CommandEntry, RiskLevel};

// ===========================================================================
// HardDenyRule
// ===========================================================================

/// 硬拦截规则 — 匹配到即无条件 Deny。
///
/// # Examples
///
/// ```
/// use katu_tool::shell::policy::HardDenyRule;
///
/// let rule = HardDenyRule::new("rm", "rm -rf /", "禁止递归删除根目录");
/// assert!(rule.matches_entry_text("rm -rf /"));
/// assert!(!rule.matches_entry_text("rm foo.txt"));
/// ```
#[derive(Debug, Clone)]
pub struct HardDenyRule {
    /// 命令名。
    pub command: String,
    /// 匹配模式（简单的 contains / starts_with / 通配符）。
    pub pattern: String,
    /// 拒绝原因。
    pub reason: String,
}

impl HardDenyRule {
    /// 创建新规则。
    pub fn new(
        command: impl Into<String>,
        pattern: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            command: command.into(),
            pattern: pattern.into(),
            reason: reason.into(),
        }
    }

    /// 检查命令文本是否匹配此规则。
    pub fn matches_entry_text(&self, text: &str) -> bool {
        let normalized = normalize_whitespace(text);

        if self.pattern.ends_with('*') {
            let prefix = &self.pattern[..self.pattern.len() - 1];
            normalized.starts_with(prefix)
        } else if self.pattern.starts_with('*') {
            let suffix = &self.pattern[1..];
            normalized.ends_with(suffix)
        } else if self.pattern.contains('*') {
            // 简单的单星通配符
            let parts: Vec<&str> = self.pattern.split('*').collect();
            if parts.len() == 2 {
                normalized.starts_with(parts[0]) && normalized.ends_with(parts[1])
            } else {
                normalized.contains(&self.pattern)
            }
        } else {
            normalized == self.pattern || normalized.starts_with(&format!("{} ", self.pattern))
        }
    }

    /// 检查 CommandEntry 是否匹配。
    ///
    /// 命令名匹配支持前缀（如 `"mkfs"` 匹配 `"mkfs.ext4"`）。
    fn matches_entry(&self, entry: &CommandEntry) -> bool {
        let name_matches = entry.name() == self.command
            || entry.name().starts_with(&format!("{}.", self.command));
        name_matches && self.matches_entry_text(entry.full())
    }
}

// ===========================================================================
// HardDenyChecker
// ===========================================================================

/// 硬拦截检查器 — 不可被任何规则或用户覆盖的安全底线。
///
/// # Examples
///
/// ```
/// use katu_tool::shell::{CommandDescriptor, policy::HardDenyChecker};
///
/// let checker = HardDenyChecker::default();
/// let desc = CommandDescriptor::parse("rm -rf /");
/// assert!(checker.check(&desc).is_some());
///
/// let desc = CommandDescriptor::parse("ls -la");
/// assert!(checker.check(&desc).is_none());
/// ```
#[derive(Debug, Clone)]
pub struct HardDenyChecker {
    rules: Vec<HardDenyRule>,
}

impl HardDenyChecker {
    /// 创建带自定义规则的检查器。
    pub fn new(rules: Vec<HardDenyRule>) -> Self {
        Self { rules }
    }

    /// 创建仅使用自定义规则（不含内置规则）的检查器。
    pub fn custom_only(rules: Vec<HardDenyRule>) -> Self {
        Self { rules }
    }

    /// 检查命令是否匹配硬拦截规则。
    ///
    /// 返回 `Some(reason)` 表示命中硬拦截，`None` 表示通过。
    pub fn check(&self, descriptor: &CommandDescriptor) -> Option<String> {
        for entry in descriptor.commands() {
            for rule in &self.rules {
                if rule.matches_entry(entry) {
                    return Some(rule.reason.clone());
                }
            }
        }
        None
    }

    /// 添加规则。
    pub fn add_rule(&mut self, rule: HardDenyRule) {
        self.rules.push(rule);
    }

    /// 规则数量。
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

impl Default for HardDenyChecker {
    /// 内置默认硬拦截规则。
    fn default() -> Self {
        Self::new(builtin_hard_deny_rules())
    }
}

/// 内置硬拦截规则列表。
fn builtin_hard_deny_rules() -> Vec<HardDenyRule> {
    vec![
        // ── 文件系统破坏 ──
        HardDenyRule::new("rm", "rm -rf /", "禁止递归删除根目录"),
        HardDenyRule::new("rm", "rm -rf /*", "禁止递归删除根目录子项"),
        HardDenyRule::new("rm", "rm -rf ~", "禁止递归删除主目录"),
        HardDenyRule::new("rm", "rm -rf ~/", "禁止递归删除主目录"),
        HardDenyRule::new("rm", "rm -rf $HOME", "禁止递归删除主目录"),
        HardDenyRule::new("rm", "rm -rf --no-preserve-root /", "禁止递归删除根目录"),
        HardDenyRule::new("rm", "rm -fr /", "禁止递归删除根目录"),
        HardDenyRule::new("rm", "rm -fr /*", "禁止递归删除根目录子项"),
        HardDenyRule::new("chmod", "chmod -R 777 /", "禁止递归修改根目录权限"),
        HardDenyRule::new("chmod", "chmod -R 777 /*", "禁止递归修改根目录子项权限"),
        HardDenyRule::new("chown", "chown -R *:* /", "禁止递归修改根目录所有者"),
        HardDenyRule::new("dd", "dd *of=/dev/*", "禁止直接写入块设备"),
        HardDenyRule::new("mkfs", "mkfs*", "禁止格式化文件系统"),
        HardDenyRule::new("shred", "shred*", "禁止安全擦除"),
        HardDenyRule::new("wipefs", "wipefs*", "禁止擦除文件系统签名"),
        // ── 进程/系统 ──
        HardDenyRule::new("kill", "kill -9 1", "禁止杀死 init 进程"),
        HardDenyRule::new("reboot", "reboot*", "禁止重启"),
        HardDenyRule::new("shutdown", "shutdown*", "禁止关机"),
        HardDenyRule::new("halt", "halt*", "禁止停机"),
        HardDenyRule::new("poweroff", "poweroff*", "禁止关机"),
        HardDenyRule::new("init", "init 0", "禁止关机"),
        HardDenyRule::new("init", "init 6", "禁止重启"),
        // ── Fork 炸弹 ──
        HardDenyRule::new("bash", "bash -c *:(){ :|:& };:*", "禁止 fork 炸弹"),
    ]
}

// ===========================================================================
// ReadOnlyChecker
// ===========================================================================

/// 只读命令检查器 — 安全命令的自动放行。
///
/// 当命令的所有子命令都被判定为只读（`RiskLevel::ReadOnly`），
/// 且不包含需要额外检查 flag 的命令时，返回 `true`。
///
/// # Examples
///
/// ```
/// use katu_tool::shell::{CommandDescriptor, policy::ReadOnlyChecker};
///
/// let checker = ReadOnlyChecker::default();
///
/// assert!(checker.is_readonly(&CommandDescriptor::parse("ls -la | grep foo")));
/// assert!(!checker.is_readonly(&CommandDescriptor::parse("rm foo.txt")));
/// ```
#[derive(Debug, Clone)]
pub struct ReadOnlyChecker {
    /// 需要额外检查 flag 的命令及其禁止 flag 列表。
    flag_deny_rules: Vec<FlagDenyRule>,
    /// 额外的只读命令（追加到内置列表）。
    extra_readonly: Vec<String>,
}

/// Flag 检查规则 — 某些命令只有不带特定 flag 时才是只读的。
#[derive(Debug, Clone)]
struct FlagDenyRule {
    /// 命令名。
    command: String,
    /// 如果出现这些 flag，则不是只读的。
    deny_flags: Vec<String>,
}

impl ReadOnlyChecker {
    /// 创建检查器。
    pub fn new(extra_readonly: Vec<String>) -> Self {
        Self {
            flag_deny_rules: builtin_flag_deny_rules(),
            extra_readonly,
        }
    }

    /// 检查命令是否全部只读。
    pub fn is_readonly(&self, descriptor: &CommandDescriptor) -> bool {
        if descriptor.is_empty() {
            return false;
        }

        for entry in descriptor.commands() {
            // 首先检查 RiskLevel
            if entry.risk() != RiskLevel::ReadOnly {
                // 额外只读列表
                if !self.extra_readonly.iter().any(|s| s == entry.name()) {
                    return false;
                }
            }

            // 然后检查 flag deny 规则
            if self.has_denied_flag(entry) {
                return false;
            }
        }

        true
    }

    /// 检查命令是否包含被禁止的 flag。
    fn has_denied_flag(&self, entry: &CommandEntry) -> bool {
        for rule in &self.flag_deny_rules {
            if entry.name() == rule.command {
                // 空 deny_flags 表示该命令始终不被视为只读（如 xargs）
                if rule.deny_flags.is_empty() {
                    return true;
                }
                let full = entry.full();
                for flag in &rule.deny_flags {
                    // 检查 flag 是否出现在命令文本中
                    if full.split_whitespace().any(|t| t == flag.as_str()) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl Default for ReadOnlyChecker {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

/// 内置 flag deny 规则。
///
/// 参考 claude-code `readOnlyValidation.ts` 中的 COMMAND_ALLOWLIST。
/// 这些命令通常是只读的，但特定 flag 会让它们执行任意命令或写入文件。
fn builtin_flag_deny_rules() -> Vec<FlagDenyRule> {
    vec![
        // sed -i/--in-place 是写入的
        FlagDenyRule {
            command: "sed".into(),
            deny_flags: vec!["-i".into(), "--in-place".into()],
        },
        // find -exec/-execdir/-delete/-ok 可以执行/删除
        FlagDenyRule {
            command: "find".into(),
            deny_flags: vec![
                "-exec".into(),
                "-execdir".into(),
                "-delete".into(),
                "-ok".into(),
                "-okdir".into(),
                "-fls".into(),
                "-fprint".into(),
                "-fprint0".into(),
                "-fprintf".into(),
            ],
        },
        // fd -x/-X/--exec/--exec-batch 执行任意命令
        FlagDenyRule {
            command: "fd".into(),
            deny_flags: vec![
                "-x".into(),
                "-X".into(),
                "--exec".into(),
                "--exec-batch".into(),
                "-l".into(),        // --list-details 内部调用 ls
                "--list-details".into(),
            ],
        },
        // fdfind 是 fd 的 Debian 别名
        FlagDenyRule {
            command: "fdfind".into(),
            deny_flags: vec![
                "-x".into(),
                "-X".into(),
                "--exec".into(),
                "--exec-batch".into(),
                "-l".into(),
                "--list-details".into(),
            ],
        },
        // xargs 默认执行子命令 — 除非管道后明确是只读命令，保守拒绝
        // 用空 deny_flags + 特殊处理：在 has_denied_flag 中 xargs 始终返回 true
        FlagDenyRule {
            command: "xargs".into(),
            deny_flags: vec![],
        },
        // awk 带 system()/getline 可执行命令，但 flag 无法判断，暂不限制
        // sort -o 会写入文件
        FlagDenyRule {
            command: "sort".into(),
            deny_flags: vec!["-o".into(), "--output".into()],
        },
        // tee 始终写入文件，但 tee 不在只读列表中所以不需要这里
        // grep --include/--exclude 只是过滤，仍然只读 — 无需限制
    ]
}

// ===========================================================================
// 辅助函数
// ===========================================================================

/// 规范化空白 — 连续空白压缩为单个空格。
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── HardDenyChecker ──

    #[test]
    fn test_hard_deny_rm_rf_root() {
        let checker = HardDenyChecker::default();
        let desc = CommandDescriptor::parse("rm -rf /");
        assert!(checker.check(&desc).is_some());
    }

    #[test]
    fn test_hard_deny_rm_rf_root_star() {
        let checker = HardDenyChecker::default();
        let desc = CommandDescriptor::parse("rm -rf /*");
        assert!(checker.check(&desc).is_some());
    }

    #[test]
    fn test_hard_deny_rm_rf_home() {
        let checker = HardDenyChecker::default();
        let desc = CommandDescriptor::parse("rm -rf ~");
        assert!(checker.check(&desc).is_some());
    }

    #[test]
    fn test_hard_deny_mkfs() {
        let checker = HardDenyChecker::default();
        let desc = CommandDescriptor::parse("mkfs.ext4 /dev/sda1");
        assert!(checker.check(&desc).is_some());
    }

    #[test]
    fn test_hard_deny_reboot() {
        let checker = HardDenyChecker::default();
        let desc = CommandDescriptor::parse("reboot");
        assert!(checker.check(&desc).is_some());
    }

    #[test]
    fn test_hard_deny_allows_normal_rm() {
        let checker = HardDenyChecker::default();
        let desc = CommandDescriptor::parse("rm foo.txt");
        assert!(checker.check(&desc).is_none());
    }

    #[test]
    fn test_hard_deny_allows_rm_rf_project_dir() {
        let checker = HardDenyChecker::default();
        let desc = CommandDescriptor::parse("rm -rf ./build");
        assert!(checker.check(&desc).is_none());
    }

    #[test]
    fn test_hard_deny_sudo_rm_rf_root() {
        let checker = HardDenyChecker::default();
        // sudo 被 CommandEntry 剥离，所以 name = "rm"
        let desc = CommandDescriptor::parse("sudo rm -rf /");
        assert!(checker.check(&desc).is_some());
    }

    #[test]
    fn test_hard_deny_in_chain() {
        let checker = HardDenyChecker::default();
        let desc = CommandDescriptor::parse("echo hello && rm -rf /");
        assert!(checker.check(&desc).is_some());
    }

    #[test]
    fn test_hard_deny_custom_rule() {
        let mut checker = HardDenyChecker::default();
        checker.add_rule(HardDenyRule::new("docker", "docker rm*", "禁止删除容器"));
        let desc = CommandDescriptor::parse("docker rm my_container");
        assert!(checker.check(&desc).is_some());
    }

    // ── ReadOnlyChecker ──

    #[test]
    fn test_readonly_ls_grep() {
        let checker = ReadOnlyChecker::default();
        assert!(checker.is_readonly(&CommandDescriptor::parse("ls -la | grep foo")));
    }

    #[test]
    fn test_readonly_cat() {
        let checker = ReadOnlyChecker::default();
        assert!(checker.is_readonly(&CommandDescriptor::parse("cat /etc/hosts")));
    }

    #[test]
    fn test_readonly_git_log() {
        let checker = ReadOnlyChecker::default();
        assert!(checker.is_readonly(&CommandDescriptor::parse("git log --oneline -10")));
    }

    #[test]
    fn test_not_readonly_rm() {
        let checker = ReadOnlyChecker::default();
        assert!(!checker.is_readonly(&CommandDescriptor::parse("rm foo.txt")));
    }

    #[test]
    fn test_not_readonly_sed_inplace() {
        let checker = ReadOnlyChecker::default();
        assert!(!checker.is_readonly(&CommandDescriptor::parse("sed -i 's/foo/bar/g' file.txt")));
    }

    #[test]
    fn test_readonly_sed_without_inplace() {
        let checker = ReadOnlyChecker::default();
        // sed 不在内置只读列表中 (classify_risk 返回 ProjectWrite)
        // 所以需要看命令本身的 risk level
        let desc = CommandDescriptor::parse("sed 's/foo/bar/g' file.txt");
        // sed 被分类为 ProjectWrite，不是 ReadOnly
        assert!(!checker.is_readonly(&desc));
    }

    #[test]
    fn test_not_readonly_mixed() {
        let checker = ReadOnlyChecker::default();
        assert!(!checker.is_readonly(&CommandDescriptor::parse("ls -la && rm foo")));
    }

    #[test]
    fn test_not_readonly_empty() {
        let checker = ReadOnlyChecker::default();
        assert!(!checker.is_readonly(&CommandDescriptor::parse("")));
    }

    #[test]
    fn test_readonly_extra() {
        let checker = ReadOnlyChecker::new(vec!["my_read_tool".into()]);
        assert!(checker.is_readonly(&CommandDescriptor::parse("my_read_tool --verbose")));
    }

    #[test]
    fn test_not_readonly_find_exec() {
        let checker = ReadOnlyChecker::default();
        assert!(!checker.is_readonly(&CommandDescriptor::parse("find . -name '*.rs' -exec rm {} +")));
    }

    #[test]
    fn test_readonly_find_without_exec() {
        let checker = ReadOnlyChecker::default();
        assert!(checker.is_readonly(&CommandDescriptor::parse("find . -name '*.rs' -type f")));
    }

    #[test]
    fn test_not_readonly_find_delete() {
        let checker = ReadOnlyChecker::default();
        assert!(!checker.is_readonly(&CommandDescriptor::parse("find /tmp -name '*.log' -delete")));
    }

    #[test]
    fn test_not_readonly_fd_exec() {
        let checker = ReadOnlyChecker::default();
        assert!(!checker.is_readonly(&CommandDescriptor::parse("fd -x rm")));
    }

    #[test]
    fn test_readonly_fd_without_exec() {
        let checker = ReadOnlyChecker::default();
        assert!(checker.is_readonly(&CommandDescriptor::parse("fd --extension rs")));
    }

    #[test]
    fn test_not_readonly_xargs() {
        let checker = ReadOnlyChecker::default();
        // xargs 始终非只读（它执行的命令未知）
        assert!(!checker.is_readonly(&CommandDescriptor::parse("ls | xargs grep foo")));
    }

    #[test]
    fn test_not_readonly_sort_output() {
        let checker = ReadOnlyChecker::default();
        assert!(!checker.is_readonly(&CommandDescriptor::parse("sort -o output.txt input.txt")));
    }

    #[test]
    fn test_readonly_sort_without_output() {
        let checker = ReadOnlyChecker::default();
        assert!(checker.is_readonly(&CommandDescriptor::parse("sort -n -r input.txt")));
    }

    // ── HardDenyRule matching ──

    #[test]
    fn test_rule_exact_match() {
        let rule = HardDenyRule::new("kill", "kill -9 1", "no");
        assert!(rule.matches_entry_text("kill -9 1"));
        assert!(!rule.matches_entry_text("kill -9 42"));
    }

    #[test]
    fn test_rule_prefix_wildcard() {
        let rule = HardDenyRule::new("mkfs", "mkfs*", "no");
        assert!(rule.matches_entry_text("mkfs.ext4 /dev/sda1"));
        assert!(rule.matches_entry_text("mkfs"));
    }

    #[test]
    fn test_rule_whitespace_normalization() {
        let rule = HardDenyRule::new("rm", "rm -rf /", "no");
        assert!(rule.matches_entry_text("rm  -rf  /"));
    }
}
