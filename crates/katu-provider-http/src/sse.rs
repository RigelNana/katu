//! SSE（Server-Sent Events）流工具。
//!
//! 将 `reqwest::Response` 的字节流转换为类型化的 SSE 事件流，
//! 所有 HTTP-based streaming provider（OpenAI、Anthropic 等）共享此层。

use std::pin::Pin;

use futures_core::Stream;

use katu_core::Error;
use katu_core::error::ProviderErrorKind;

// Re-export SSE Event 类型，provider 无需直接依赖 eventsource-stream
pub use eventsource_stream::Event as SseEvent;

/// SSE 事件流类型 — 每个 item 是一个 `SseEvent` 或错误。
///
/// Provider 从此流消费原始 SSE 事件，然后根据自身协议
/// 解析 `event.data` 为具体的 chunk 类型。
pub type SseStream = Pin<Box<dyn Stream<Item = Result<SseEvent, Error>> + Send>>;

/// 将 `reqwest::Response` 转换为 SSE 事件流。
///
/// 使用 `eventsource-stream` 解析 SSE 协议，将底层传输错误
/// 映射为 `katu_core::Error`。
///
/// # 用法
///
/// ```ignore
/// use katu_provider_http::sse;
///
/// let stream = sse::from_response(response);
/// tokio::pin!(stream);
/// while let Some(event) = stream.next().await {
///     let event = event?;
///     if event.data == "[DONE]" { break; }
///     let chunk: MyChunk = serde_json::from_str(&event.data)?;
/// }
/// ```
pub fn from_response(response: reqwest::Response) -> SseStream {
    use eventsource_stream::Eventsource;
    use tokio_stream::StreamExt;

    let sse = response.bytes_stream().eventsource();

    Box::pin(sse.map(|result| {
        result.map_err(|e| Error::Provider {
            kind: ProviderErrorKind::Transport {
                message: format!("SSE parse error: {e}"),
            },
            suggestion: None,
        })
    }))
}

/// 向后兼容别名。
#[deprecated(since = "0.2.0", note = "use `sse::from_response` instead")]
pub fn into_sse_stream(response: reqwest::Response) -> SseStream {
    from_response(response)
}
