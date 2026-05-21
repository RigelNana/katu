//! # command
//!
//! ## 职责
//! 从原始 shell 命令字符串中提取结构化信息，用于权限检查。
//!
//! ## 设计
//! - **轻量级** — 不做完整的 bash AST 解析，只做 token 级拆分
//! - **保守分类** — 无法识别的命令标记为 `Unknown`（保守处理）
//! - **纯函数** — 无副作用，解析结果可缓存

use std::fmt;

use serde::{Deserialize, Serialize};

// ===========================================================================
// RiskLevel
// ===========================================================================

/// 命令风险等级 — 用于权限决策的分类标签。
///
/// 从低到高排列，`PartialOrd` / `Ord` 按此排序。
///
/// # Examples
///
/// ```
/// use katu_tool::shell::RiskLevel;
///
/// assert!(RiskLevel::ReadOnly < RiskLevel::Destructive);
/// assert_eq!(RiskLevel::Unknown.as_str(), "unknown");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    /// 只读操作 — ls, cat, grep, find, echo, pwd 等。
    ReadOnly,
    /// 项目内写入 — cargo build, npm install, git commit, touch, mkdir 等。
    ProjectWrite,
    /// 系统级操作 — apt, brew, pip install -g, systemctl 等。
    SystemWrite,
    /// 高危操作 — rm -rf, chmod -R, dd, mkfs 等。
    Destructive,
    /// 网络外传 — curl -X POST, ssh, scp 等。
    NetworkEgress,
    /// 未知命令 — 无法分类，保守处理。
    Unknown,
}

impl RiskLevel {
    /// 转为字符串标签。
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::ProjectWrite => "project_write",
            Self::SystemWrite => "system_write",
            Self::Destructive => "destructive",
            Self::NetworkEgress => "network_egress",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ===========================================================================
// CommandEntry
// ===========================================================================

/// 单个子命令的信息。
///
/// 从管道/链式命令中拆分出的一个命令。
///
/// # Examples
///
/// ```
/// use katu_tool::shell::CommandEntry;
///
/// let entry = CommandEntry::new("git push origin main");
/// assert_eq!(entry.name(), "git");
/// assert_eq!(entry.prefix(), "git push");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandEntry {
    /// 命令名（第一个 token）。
    name: String,
    /// 完整的子命令文本（已 trim）。
    full: String,
    /// 带子命令的前缀（前两个 token）。
    prefix: String,
    /// 该子命令的风险等级。
    risk: RiskLevel,
}

impl CommandEntry {
    /// 从子命令文本构造。
    pub fn new(text: &str) -> Self {
        let trimmed = text.trim();
        let tokens = Self::extract_tokens(trimmed);

        let name = tokens.first().map(|s| s.as_str()).unwrap_or("").to_string();
        let prefix = if tokens.len() >= 2 {
            format!("{} {}", tokens[0], tokens[1])
        } else {
            name.clone()
        };
        // full 使用去除前缀包装后的 token 重组，
        // 使 HardDenyRule 能正确匹配（不含 sudo/env 等）。
        let full = tokens.join(" ");
        let risk = classify_risk(&name, &prefix, &full);

        Self {
            name,
            full,
            prefix,
            risk,
        }
    }

    /// 命令名。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 完整子命令文本。
    pub fn full(&self) -> &str {
        &self.full
    }

    /// 带子命令的前缀（前两个 token）。
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// 风险等级。
    pub fn risk(&self) -> RiskLevel {
        self.risk
    }

    /// 提取 token 列表 — 跳过前导环境变量赋值和 sudo 等包装。
    fn extract_tokens(text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = Vec::new();
        let mut skip_wrappers = true;

        for token in text.split_whitespace() {
            if skip_wrappers {
                // 跳过 VAR=val 形式的环境变量赋值
                if token.contains('=') && !token.starts_with('-') {
                    let key = token.split('=').next().unwrap_or("");
                    if key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                        continue;
                    }
                }
                // 跳过 sudo, env, nice, nohup, time 等前缀包装
                if matches!(
                    token,
                    "sudo" | "env" | "nice" | "nohup" | "time" | "command" | "builtin" | "exec"
                ) {
                    continue;
                }
                skip_wrappers = false;
            }
            tokens.push(token.to_string());
        }

        tokens
    }
}

// ===========================================================================
// CommandDescriptor
// ===========================================================================

/// 命令描述符 — 从原始命令字符串中提取的结构化信息。
///
/// 这是权限检查的输入，不涉及执行。支持管道 (`|`) 和链式
/// 命令 (`&&`, `||`, `;`)。
///
/// # Examples
///
/// ```
/// use katu_tool::shell::{CommandDescriptor, RiskLevel};
///
/// let desc = CommandDescriptor::parse("ls -la | grep foo");
/// assert_eq!(desc.commands().len(), 2);
/// assert_eq!(desc.risk(), RiskLevel::ReadOnly);
/// assert_eq!(desc.permission_key(), "bash:grep");
///
/// let desc = CommandDescriptor::parse("git push origin main");
/// assert_eq!(desc.permission_key(), "bash:git");
/// assert_eq!(desc.always_allow_pattern(), "git push *");
/// ```
#[derive(Debug, Clone)]
pub struct CommandDescriptor {
    /// 原始命令文本。
    raw: String,
    /// 拆分出的子命令列表。
    commands: Vec<CommandEntry>,
    /// 综合风险等级（所有子命令中的最高等级）。
    risk: RiskLevel,
}

impl CommandDescriptor {
    /// 从原始命令字符串解析。
    ///
    /// 按 `&&`, `||`, `;`, `|` 拆分子命令，对每个子命令提取名称、前缀和风险等级。
    pub fn parse(raw: &str) -> Self {
        let parts = split_command_chain(raw);
        let commands: Vec<CommandEntry> = parts
            .iter()
            .filter(|s| !s.trim().is_empty())
            .map(|s| CommandEntry::new(s))
            .collect();

        let risk = commands
            .iter()
            .map(|c| c.risk)
            .max()
            .unwrap_or(RiskLevel::Unknown);

        Self {
            raw: raw.to_string(),
            commands,
            risk,
        }
    }

    /// 原始命令文本。
    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// 子命令列表。
    pub fn commands(&self) -> &[CommandEntry] {
        &self.commands
    }

    /// 综合风险等级。
    pub fn risk(&self) -> RiskLevel {
        self.risk
    }

    /// 所有子命令的名称列表。
    pub fn command_names(&self) -> Vec<&str> {
        self.commands.iter().map(|c| c.name()).collect()
    }

    /// 生成权限 key。
    ///
    /// 规则：
    /// - 单命令 → `"bash:git"`
    /// - 管道/链中有多个命令 → 取最高风险命令 → `"bash:rm"`
    /// - 空命令 → `"bash"`
    pub fn permission_key(&self) -> String {
        let primary = self
            .commands
            .iter()
            .max_by_key(|c| c.risk)
            .map(|c| c.name());

        match primary {
            Some(name) if !name.is_empty() => format!("bash:{name}"),
            _ => "bash".to_string(),
        }
    }

    /// 用于 Ruleset 匹配的 content 字符串。
    pub fn content_for_matching(&self) -> &str {
        &self.raw
    }

    /// 生成 "always allow" 时应持久化的宽泛模式。
    ///
    /// 取主命令的前缀 + `*`：
    /// - `"git push origin main"` → `"git push *"`
    /// - `"npm run build"` → `"npm run *"`
    /// - `"ls -la"` → `"ls *"`
    /// - `"echo hello"` → `"echo *"`
    pub fn always_allow_pattern(&self) -> String {
        let primary = self
            .commands
            .iter()
            .max_by_key(|c| c.risk);

        match primary {
            Some(entry) if entry.prefix() != entry.name() => {
                format!("{} *", entry.prefix())
            }
            Some(entry) if !entry.name().is_empty() => {
                format!("{} *", entry.name())
            }
            _ => "*".to_string(),
        }
    }

    /// 是否为空命令。
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// 是否只包含只读命令。
    pub fn is_all_readonly(&self) -> bool {
        !self.commands.is_empty()
            && self.commands.iter().all(|c| c.risk == RiskLevel::ReadOnly)
    }
}

// ===========================================================================
// 内部辅助函数
// ===========================================================================

/// 按 `&&`, `||`, `;`, `|` 拆分命令链。
///
/// 简单的字符级拆分，不处理引号内的分隔符（Phase 2 改进）。
fn split_command_chain(raw: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut chars = raw.chars().peekable();
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    while let Some(c) = chars.next() {
        match c {
            '\'' if !in_double_quote => {
                in_single_quote = !in_single_quote;
                current.push(c);
            }
            '"' if !in_single_quote => {
                in_double_quote = !in_double_quote;
                current.push(c);
            }
            _ if in_single_quote || in_double_quote => {
                current.push(c);
            }
            '&' if chars.peek() == Some(&'&') => {
                chars.next(); // 消费第二个 &
                parts.push(std::mem::take(&mut current));
            }
            '|' if chars.peek() == Some(&'|') => {
                chars.next(); // 消费第二个 |
                parts.push(std::mem::take(&mut current));
            }
            '|' => {
                parts.push(std::mem::take(&mut current));
            }
            ';' => {
                parts.push(std::mem::take(&mut current));
            }
            _ => {
                current.push(c);
            }
        }
    }

    if !current.trim().is_empty() {
        parts.push(current);
    }

    parts
}

/// 根据命令名、前缀和全文分类风险等级。
fn classify_risk(name: &str, prefix: &str, _full: &str) -> RiskLevel {
    // 只读命令
    if is_readonly_command(name) {
        return RiskLevel::ReadOnly;
    }

    // Git 子命令细分
    if name == "git" {
        return classify_git_risk(prefix);
    }

    // 高危命令
    if matches!(name, "rm" | "rmdir" | "mkfs" | "dd" | "shred" | "wipefs") {
        return RiskLevel::Destructive;
    }
    if matches!(name, "chmod" | "chown" | "chgrp") {
        return RiskLevel::Destructive;
    }

    // 系统级
    if matches!(
        name,
        "apt" | "apt-get" | "yum" | "dnf" | "pacman" | "brew"
            | "snap" | "flatpak" | "systemctl" | "service"
            | "mount" | "umount" | "fdisk" | "parted"
            | "useradd" | "userdel" | "usermod" | "groupadd"
            | "iptables" | "ufw" | "firewall-cmd"
    ) {
        return RiskLevel::SystemWrite;
    }

    // 网络外传
    if matches!(name, "ssh" | "scp" | "rsync" | "ftp" | "sftp" | "nc" | "ncat" | "socat") {
        return RiskLevel::NetworkEgress;
    }
    if matches!(name, "curl" | "wget" | "http" | "httpie") {
        return RiskLevel::NetworkEgress;
    }

    // 项目级写入
    if matches!(
        name,
        "cargo" | "npm" | "npx" | "yarn" | "pnpm" | "bun"
            | "pip" | "pip3" | "pipenv" | "poetry" | "uv"
            | "go" | "mvn" | "gradle"
            | "make" | "cmake" | "ninja"
            | "touch" | "mkdir" | "cp" | "mv" | "ln"
            | "tee" | "patch" | "sed" | "install"
    ) {
        return RiskLevel::ProjectWrite;
    }

    RiskLevel::Unknown
}

/// 只读命令白名单。
fn is_readonly_command(name: &str) -> bool {
    matches!(
        name,
        "ls" | "cat" | "head" | "tail" | "less" | "more"
            | "wc" | "du" | "df" | "file" | "stat"
            | "find" | "fd" | "fdfind" | "tree" | "realpath" | "readlink"
            | "basename" | "dirname" | "test" | "["
            | "grep" | "rg" | "ag" | "ack"
            | "awk" | "sort" | "uniq" | "cut" | "tr"
            | "diff" | "comm" | "join" | "paste"
            | "jq" | "yq" | "xq"
            | "echo" | "printf" | "pwd" | "env" | "printenv"
            | "whoami" | "id" | "hostname" | "uname" | "date"
            | "which" | "whereis" | "type" | "command"
            | "ps" | "free" | "uptime" | "lsof"
            | "true" | "false" | "yes" | "seq" | "expr"
            | "md5sum" | "sha256sum" | "sha1sum" | "b2sum"
            | "xxd" | "od" | "hexdump"
    )
}

/// Git 子命令风险细分。
fn classify_git_risk(prefix: &str) -> RiskLevel {
    let sub = prefix
        .strip_prefix("git ")
        .unwrap_or("")
        .trim();

    match sub {
        // 只读
        "log" | "status" | "diff" | "show" | "branch" | "tag"
        | "remote" | "rev-parse" | "ls-files" | "ls-tree"
        | "blame" | "shortlog" | "describe" | "cat-file"
        | "for-each-ref" | "reflog" | "stash list" => RiskLevel::ReadOnly,

        // 网络
        "push" | "fetch" | "pull" | "clone" => RiskLevel::NetworkEgress,

        // 危险 — 历史重写
        "reset --hard" | "clean" | "filter-branch" | "rebase" => RiskLevel::Destructive,

        // 其他 git 命令 — 项目写入
        _ => RiskLevel::ProjectWrite,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── CommandEntry ──

    #[test]
    fn test_entry_simple() {
        let entry = CommandEntry::new("ls -la");
        assert_eq!(entry.name(), "ls");
        assert_eq!(entry.prefix(), "ls -la");
        assert_eq!(entry.risk(), RiskLevel::ReadOnly);
    }

    #[test]
    fn test_entry_with_subcommand() {
        let entry = CommandEntry::new("git push origin main");
        assert_eq!(entry.name(), "git");
        assert_eq!(entry.prefix(), "git push");
        assert_eq!(entry.risk(), RiskLevel::NetworkEgress);
    }

    #[test]
    fn test_entry_strips_env_var() {
        let entry = CommandEntry::new("FOO=bar cargo build");
        assert_eq!(entry.name(), "cargo");
        assert_eq!(entry.prefix(), "cargo build");
    }

    #[test]
    fn test_entry_strips_sudo() {
        let entry = CommandEntry::new("sudo rm -rf /tmp/cache");
        assert_eq!(entry.name(), "rm");
        assert_eq!(entry.risk(), RiskLevel::Destructive);
    }

    #[test]
    fn test_entry_strips_env_prefix() {
        let entry = CommandEntry::new("env FOO=1 BAR=2 npm run build");
        assert_eq!(entry.name(), "npm");
        assert_eq!(entry.prefix(), "npm run");
    }

    // ── CommandDescriptor ──

    #[test]
    fn test_parse_single() {
        let desc = CommandDescriptor::parse("cargo test --release");
        assert_eq!(desc.commands().len(), 1);
        assert_eq!(desc.risk(), RiskLevel::ProjectWrite);
        assert_eq!(desc.permission_key(), "bash:cargo");
    }

    #[test]
    fn test_parse_pipe() {
        let desc = CommandDescriptor::parse("ls -la | grep foo");
        assert_eq!(desc.commands().len(), 2);
        assert_eq!(desc.command_names(), vec!["ls", "grep"]);
        assert!(desc.is_all_readonly());
    }

    #[test]
    fn test_parse_chain_and() {
        let desc = CommandDescriptor::parse("mkdir -p out && cargo build");
        assert_eq!(desc.commands().len(), 2);
        assert_eq!(desc.command_names(), vec!["mkdir", "cargo"]);
        assert_eq!(desc.risk(), RiskLevel::ProjectWrite);
    }

    #[test]
    fn test_parse_chain_semicolon() {
        let desc = CommandDescriptor::parse("echo hello; rm -rf /tmp/foo");
        assert_eq!(desc.commands().len(), 2);
        assert_eq!(desc.risk(), RiskLevel::Destructive);
        // permission_key 取最高风险命令
        assert_eq!(desc.permission_key(), "bash:rm");
    }

    #[test]
    fn test_parse_quoted_pipe() {
        // 引号内的 | 不应拆分
        let desc = CommandDescriptor::parse("echo 'hello | world'");
        assert_eq!(desc.commands().len(), 1);
        assert_eq!(desc.command_names(), vec!["echo"]);
    }

    #[test]
    fn test_parse_empty() {
        let desc = CommandDescriptor::parse("");
        assert!(desc.is_empty());
        assert_eq!(desc.permission_key(), "bash");
    }

    // ── permission_key ──

    #[test]
    fn test_permission_key_git() {
        let desc = CommandDescriptor::parse("git push origin main");
        assert_eq!(desc.permission_key(), "bash:git");
    }

    #[test]
    fn test_permission_key_mixed() {
        // rm 是 Destructive, ls 是 ReadOnly → 取 rm
        let desc = CommandDescriptor::parse("ls && rm foo");
        assert_eq!(desc.permission_key(), "bash:rm");
    }

    // ── always_allow_pattern ──

    #[test]
    fn test_always_allow_subcommand() {
        let desc = CommandDescriptor::parse("git push origin main");
        assert_eq!(desc.always_allow_pattern(), "git push *");
    }

    #[test]
    fn test_always_allow_simple() {
        let desc = CommandDescriptor::parse("ls -la");
        assert_eq!(desc.always_allow_pattern(), "ls -la *");
    }

    #[test]
    fn test_always_allow_npm_run() {
        let desc = CommandDescriptor::parse("npm run build");
        assert_eq!(desc.always_allow_pattern(), "npm run *");
    }

    // ── RiskLevel ──

    #[test]
    fn test_risk_ordering() {
        assert!(RiskLevel::ReadOnly < RiskLevel::ProjectWrite);
        assert!(RiskLevel::ProjectWrite < RiskLevel::SystemWrite);
        assert!(RiskLevel::SystemWrite < RiskLevel::Destructive);
        assert!(RiskLevel::Destructive < RiskLevel::NetworkEgress);
        assert!(RiskLevel::NetworkEgress < RiskLevel::Unknown);
    }

    #[test]
    fn test_git_readonly() {
        let desc = CommandDescriptor::parse("git log --oneline -10");
        assert_eq!(desc.risk(), RiskLevel::ReadOnly);
    }

    #[test]
    fn test_git_push_network() {
        let desc = CommandDescriptor::parse("git push");
        assert_eq!(desc.risk(), RiskLevel::NetworkEgress);
    }

    #[test]
    fn test_unknown_command() {
        let desc = CommandDescriptor::parse("my_custom_script --verbose");
        assert_eq!(desc.risk(), RiskLevel::Unknown);
    }

    // ── split_command_chain ──

    #[test]
    fn test_split_or() {
        let parts = split_command_chain("cmd1 || cmd2");
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn test_split_complex() {
        let parts = split_command_chain("a && b | c; d || e");
        assert_eq!(parts.len(), 5);
    }
}
