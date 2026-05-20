//! # katu-llm
//!
//! ## 职责
//! 定义 LLM 抽象层的核心数据类型，包括模型引用、生成参数、
//! 模型能力描述、定价信息、HTTP 选项与缓存策略。
//!
//! ## 模块
//! - `model` — 模型相关类型 (ModelRef, ModelLimits, ModelPricing, ModelCapabilities, ...)
//! - `http` — HTTP 传输选项 (HttpOptions)
//! - `cache` — 缓存策略 (CachePolicy)
//! - `request` — 请求与响应 (LlmRequest, LlmResponse)
//! - `provider` — Provider trait (stream, generate)

pub mod cache;
pub mod http;
pub mod model;
pub mod provider;
pub mod request;

// re-export 常用类型到 crate 根
pub use cache::CachePolicy;
pub use katu_core::GenerationOptions;
pub use http::HttpOptions;
pub use model::{
    InputModality, ModelCapabilities, ModelLimits, ModelPricing, ModelRef, ReasoningEffort,
    ThinkingConfig, ThinkingMode,
};
pub use provider::{EventStream, Provider};
pub use request::{LlmRequest, LlmResponse};
