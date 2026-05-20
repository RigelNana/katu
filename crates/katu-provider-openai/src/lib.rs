//! # katu-provider-openai
//!
//! OpenAI Chat Completions API 的 Provider 适配器。
//!
//! 将 `katu-llm` 的通用 `LlmRequest` 转换为 OpenAI wire format，
//! 发送 HTTP 请求，解析 SSE 流并转换为 `StreamEvent`。
//!
//! 兼容所有使用 OpenAI Chat Completions API 格式的服务商：
//! OpenAI、Azure OpenAI、DeepSeek、Together、OpenRouter 等。
//!
//! ## 模块
//! - `types` — OpenAI REST API wire types (request/response JSON)
//! - `convert` — LlmRequest ↔ OpenAI wire format 转换
//! - `stream` — SSE 流解析 → StreamEvent
//! - `provider` — `OpenAiProvider` impl `Provider` trait

pub mod convert;
pub mod provider;
pub mod stream;
pub mod types;

pub use provider::OpenAiProvider;
