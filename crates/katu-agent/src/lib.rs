//! # katu-agent
//!
//! ## 职责
//! Agent loop 实现 — 消费 LLM 流式响应，驱动工具执行，产出 AgentEvent。
//!
//! ## 模块
//! - `error` — 统一错误类型 (`AgentError`)
//! - `retry` — 重试策略配置 (`RetryConfig`)
//! - `instance` — Agent 运行实例 (`AgentInstance`, `InstanceBuilder`, `RunConfig`)
//! - `prompt` — 模块化 prompt 构建系统
//! - `runner` — Agent loop 核心驱动器 (`Runner`, `RunOutcome`)
//! - `session` — 会话状态管理 (`Session`)
//! - `stream_consumer` — 将 `StreamEvent` 流累积为 `AssistantMessage` + 实时发射 `AgentEvent`
//! - `tool_executor` — 工具批量执行与权限管线
//!
//! ## 依赖
//! - `katu-core` — 类型定义 (StreamEvent, AgentEvent, Message, Tool...)
//! - `katu-llm` — LLM 抽象层 (Provider, ModelRef, LlmRequest)

pub mod compaction;
pub mod error;
pub(crate) mod event_sender;
pub mod instance;
pub mod prompt;
pub mod retry;
pub mod runner;
pub mod session;
pub mod stream_consumer;
pub mod tool_executor;
