//! # katu-provider-http
//!
//! HTTP/SSE 传输层 — 所有 HTTP-based LLM provider adapter 的共享基础设施。
//!
//! 提供：
//! - `HttpProviderClient` — 封装 reqwest，处理请求构建、发送、错误映射
//! - `SseStream` / `SseEvent` — SSE 事件流类型和创建工具
//!
//! 具体 provider（OpenAI、Anthropic、DeepSeek）依赖此 crate，
//! 只需实现 wire type 转换和 StreamEvent 解析。

pub mod client;
pub mod sse;

pub use client::HttpProviderClient;
pub use sse::{SseEvent, SseStream, from_response};
