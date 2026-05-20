//! OpenAI Chat Completions Provider 实现。

use std::future::Future;
use std::pin::Pin;

use katu_core::error::ProviderErrorKind;
use katu_core::{
    AssistantBlock, AssistantMessage, Error, FinishReason, Message, MessageId, ToolCallId,
};
use katu_llm::provider::{EventStream, Provider};
use katu_llm::request::{LlmRequest, LlmResponse};
use katu_provider_http::HttpProviderClient;

use crate::convert;
use crate::stream;
use crate::types::*;

// ===========================================================================
// OpenAiProvider
// ===========================================================================

/// OpenAI Chat Completions 的 endpoint path。
const ENDPOINT: &str = "/chat/completions";

/// OpenAI Chat Completions API 的 Provider 实现。
///
/// 支持所有兼容 OpenAI API 格式的服务商（OpenAI、Azure、DeepSeek、Together、
/// OpenRouter 等），通过 `ModelRef.base_url` 指定不同端点。
///
/// # Examples
/// ```ignore
/// use katu_provider_openai::OpenAiProvider;
///
/// let provider = OpenAiProvider::new();
/// // 使用 ModelRef 中的 base_url 和 api_key 进行请求
/// ```
pub struct OpenAiProvider {
    http: HttpProviderClient,
}

impl OpenAiProvider {
    /// 创建一个默认的 OpenAiProvider。
    pub fn new() -> Self {
        Self {
            http: HttpProviderClient::new(),
        }
    }

    /// 使用自定义 `HttpProviderClient` 创建。
    pub fn with_http(http: HttpProviderClient) -> Self {
        Self { http }
    }

    /// 非流式调用：发送请求并解析完整响应。
    async fn do_generate(&self, request: LlmRequest) -> Result<LlmResponse, Error> {
        let body = convert::build_request(&request, false);

        let resp: ChatCompletionResponse = self
            .http
            .post_json(&request, ENDPOINT, &body)
            .await?;

        // 转换为 LlmResponse
        let choice = resp.choices.first().ok_or_else(|| Error::Provider {
            kind: ProviderErrorKind::InvalidResponse {
                message: "empty choices array".into(),
            },
            suggestion: None,
        })?;

        let finish_reason = choice
            .finish_reason
            .as_deref()
            .map(convert::convert_finish_reason)
            .unwrap_or(FinishReason::Stop);

        let usage = resp
            .usage
            .as_ref()
            .map(convert::convert_usage)
            .unwrap_or_default();

        let mut content = Vec::new();
        if let Some(text) = &choice.message.content {
            if !text.is_empty() {
                content.push(AssistantBlock::Text { text: text.clone() });
            }
        }
        if let Some(tool_calls) = &choice.message.tool_calls {
            for tc in tool_calls {
                let arguments: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null);
                content.push(AssistantBlock::ToolCall {
                    id: ToolCallId::new(&tc.id),
                    name: tc.function.name.clone(),
                    arguments,
                });
            }
        }

        let message = Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            content,
            model: resp.model,
            provider: request.model.provider.as_str().to_owned(),
            finish_reason,
            usage: Some(usage.clone()),
            timestamp: chrono::Utc::now(),
        });

        Ok(LlmResponse {
            message,
            finish_reason,
            usage,
        })
    }

    /// 流式调用：发送请求并返回 SSE 事件流。
    async fn do_stream(&self, request: LlmRequest) -> Result<EventStream, Error> {
        let body = convert::build_request(&request, true);
        let sse = self.http.post_sse(&request, ENDPOINT, &body).await?;
        Ok(stream::create_event_stream(sse))
    }
}

impl Default for OpenAiProvider {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Provider trait implementation
// ===========================================================================

impl Provider for OpenAiProvider {
    fn stream(
        &self,
        request: LlmRequest,
    ) -> Pin<Box<dyn Future<Output = Result<EventStream, Error>> + Send + '_>> {
        Box::pin(self.do_stream(request))
    }

    fn generate(
        &self,
        request: LlmRequest,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, Error>> + Send + '_>> {
        Box::pin(self.do_generate(request))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use katu_core::{ModelId, ProviderId, RouteId};
    use katu_llm::model::{ModelLimits, ModelRef};

    #[test]
    fn test_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OpenAiProvider>();
    }

    #[test]
    fn test_default_construction() {
        let _provider = OpenAiProvider::new();
    }

    #[test]
    fn test_endpoint_constant() {
        assert_eq!(ENDPOINT, "/chat/completions");
    }

    #[test]
    fn test_build_request_body() {
        let model = ModelRef::new(
            ModelId::new("gpt-4o"),
            ProviderId::new("openai"),
            RouteId::new("openai-chat"),
            "https://api.openai.com/v1",
            ModelLimits {
                context_window: 128_000,
                max_output_tokens: 4096,
            },
        );

        let req = LlmRequest::new(model).with_message(Message::user("Hello"));
        let body = convert::build_request(&req, true);
        assert_eq!(body.model, "gpt-4o");
        assert_eq!(body.stream, Some(true));
    }
}
