//! # katu-core
//!
//! ## 职责
//! 定义 Katu AI Agent 框架的核心 trait 与类型。
//! 零外部 katu 依赖，是整个系统的"接口契约"层。
//!
//! ## 对外接口
//! - `types` — newtype ID 与基础枚举 (SessionId, MessageId, Role, FinishReason...)
//! - `error` — 全局错误类型 (Error, ProviderErrorKind, AuthErrorKind, Result)
//! - `usage` — token 用量与费用类型 (Usage, Cost)
//! - `message` — 对话消息类型 (Message, ContentBlock, AssistantBlock...)
//! - `generation` — LLM 生成参数 (GenerationOptions)
//! - `event` — LLM 流式事件 (StreamEvent, ToolResultValue)
//! - `agent_event` — Agent 语义层事件 (AgentEvent, AgentEventKind, AgentFinishReason...)
//! - `tool` — 工具类型与执行契约 (Tool, ToolDefinition, ToolOutput, ToolCallContext...)
//! - `agent` — Agent 定义数据模型 (AgentDefinition, AgentRole, ToolFilter...)
//! - `hook` — Hook 系统类型与执行契约 (Hook, HookEvent, HookInput, HookOutput, HookRegistry...)
//! - `permission` — 权限系统类型与规则引擎 (PermissionRule, Ruleset, PermissionDecision, PermissionMode...)

pub mod agent;
pub mod agent_event;
pub mod compaction;
pub mod error;
pub mod event;
pub mod generation;
pub mod hook;
pub mod message;
pub mod permission;
pub mod tool;
pub mod types;
pub mod usage;

// re-export 常用类型到 crate 根
pub use agent::{AgentDefinition, AgentModelRef, AgentName, AgentRole, ToolFilter};
pub use agent_event::{AgentEvent, AgentEventKind, AgentFinishReason};
pub use compaction::{CompactTrigger, CompactionConfig, CompactionResult, TokenBudgetState};
pub use error::{Error, Result};
pub use generation::GenerationOptions;
pub use event::{StreamEvent, ToolResultValue};
pub use message::{AssistantBlock, AssistantMessage, ContentBlock, Message, ToolResultMessage, UserContent, UserMessage};
pub use tool::{
    CancellationToken, ConcurrencyMode, Tool, ToolCallContext, ToolChoice, ToolDefinition,
    ToolOutput,
};
pub use types::*;
pub use usage::{Cost, Usage};
