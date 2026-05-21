//! # shell
//!
//! ## 职责
//! Shell 命令执行工具 — 在子进程中运行 bash/zsh/sh 命令。
//!
//! ## 模块结构
//! - `command` — 命令解析与描述（`CommandDescriptor`, `RiskLevel`）
//! - `policy` — 安全策略（硬拦截规则 + 只读命令白名单）
//! - `config` — 工具配置（`BashToolConfig`）
//! - `tool` — `BashTool` 实现（`impl Tool`）
//!
//! ## 权限集成
//! ```text
//! check_permissions()
//!   ├─ Layer 1: 硬拦截 (policy::HardDenyChecker) → Deny
//!   ├─ Layer 2: 只读放行 (policy::ReadOnlyChecker) → Allow
//!   └─ Layer 3: Passthrough → 交给框架 Ruleset
//!
//! permission_request()
//!   └─ CommandDescriptor::parse() → PermissionRequest {
//!        permission: "bash:git",
//!        patterns: ["git push origin main"],
//!        always_allow_patterns: ["git push *"]
//!      }
//! ```

mod command;
mod config;
pub mod env;
mod executor;
mod output;
pub mod policy;
mod tool;

pub use command::{CommandDescriptor, CommandEntry, RiskLevel};
pub use config::BashToolConfig;
pub use executor::{OutputCallback, ShellExecutor, ShellResult};
pub use output::OutputCollector;
pub use policy::{HardDenyChecker, ReadOnlyChecker};
pub use tool::BashTool;
