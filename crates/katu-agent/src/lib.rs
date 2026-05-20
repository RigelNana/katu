//! # katu-agent
//!
//! ## 职责
//! Agent loop 实现 — 消费 LLM 流式响应，驱动工具执行，产出 AgentEvent。
//!
//! ## 模块
//! - `stream_consumer` — 将 `StreamEvent` 流累积为 `AssistantMessage` + 实时发射 `AgentEvent`
//!
//! ## 依赖
//! - `katu-core` — 类型定义 (StreamEvent, AgentEvent, Message, Tool...)

pub(crate) mod event_sender;
pub mod session;
pub mod stream_consumer;
pub mod tool_executor;
