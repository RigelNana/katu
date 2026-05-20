//! # katu_llm::provider
//!
//! ## 职责
//! 定义 LLM Provider trait — 所有 provider 实现的统一接口。
//!
//! ## 对外接口
//! - `Provider` — 异步 trait，stream + generate

use std::future::Future;
use std::pin::Pin;

use futures_core::Stream;

use katu_core::{Error, StreamEvent};

use crate::request::{LlmRequest, LlmResponse};

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

/// LLM Provider 的统一异步接口。
///
/// 每个 provider 实现（Anthropic、OpenAI、DeepSeek 等）
/// 需实现此 trait，提供流式和非流式两种调用方式。
///
/// ## 流式 (stream)
/// 返回 `StreamEvent` 异步流，逐 token 推送给调用方。
///
/// ## 非流式 (generate)
/// 等待完整响应后返回 `LlmResponse`。默认实现通过收集流来完成。
///
/// # Examples
/// ```ignore
/// use katu_llm::{LlmRequest, provider::Provider};
///
/// async fn run(provider: &dyn Provider, request: LlmRequest) {
///     // 流式
///     let mut stream = provider.stream(request.clone()).await.unwrap();
///     // 非流式
///     let response = provider.generate(request).await.unwrap();
/// }
/// ```
pub trait Provider: Send + Sync {
    /// 流式调用 — 返回 `StreamEvent` 异步流。
    ///
    /// 调用方通过 `while let Some(event) = stream.next().await` 逐个消费事件。
    fn stream(
        &self,
        request: LlmRequest,
    ) -> Pin<Box<dyn Future<Output = Result<EventStream, Error>> + Send + '_>>;

    /// 非流式调用 — 等待完整响应。
    ///
    /// 默认实现通过收集 `stream()` 的所有事件来构建 `LlmResponse`。
    /// Provider 可覆写以使用原生非流式 API。
    fn generate(
        &self,
        request: LlmRequest,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, Error>> + Send + '_>>;
}

/// StreamEvent 的异步流类型别名。
pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>>;
