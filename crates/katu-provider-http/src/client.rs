//! HTTP Provider 客户端 — 所有 HTTP-based LLM provider 的共享传输层。
//!
//! 封装了 `reqwest::Client`，提供：
//! - 从 `LlmRequest` 自动构建 headers（api_key、extra headers）和 URL
//! - HTTP 状态码 → `katu_core::Error` 的通用映射
//! - `post_json` / `post_sse` 高阶方法，provider 无需直接操作 reqwest

use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::Serialize;
use serde::de::DeserializeOwned;

use katu_core::Error;
use katu_core::error::{AuthErrorKind, ProviderErrorKind};
use katu_llm::request::LlmRequest;

use crate::sse::{self, SseStream};

// ===========================================================================
// HttpProviderClient
// ===========================================================================

/// 所有 HTTP-based provider 共享的传输层客户端。
///
/// 封装请求构建、发送、错误映射等样板逻辑，
/// 让具体 provider（OpenAI、Anthropic、DeepSeek 等）
/// 只关注 wire type 转换和 StreamEvent 解析。
///
/// # Examples
/// ```ignore
/// use katu_provider_http::HttpProviderClient;
///
/// let client = HttpProviderClient::new();
/// let resp: MyResponse = client.post_json(&request, "/chat/completions", &body).await?;
/// ```
pub struct HttpProviderClient {
    client: reqwest::Client,
}

impl HttpProviderClient {
    /// 使用默认超时创建客户端（连接 30s，总超时 300s）。
    pub fn new() -> Self {
        Self::with_timeouts(Duration::from_secs(300), Duration::from_secs(30))
    }

    /// 使用自定义超时创建客户端。
    pub fn with_timeouts(timeout: Duration, connect_timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .connect_timeout(connect_timeout)
            .build()
            .expect("failed to create HTTP client");
        Self { client }
    }

    /// 使用已有的 `reqwest::Client` 创建。
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// 获取底层 `reqwest::Client` 的引用（escape hatch）。
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    // -----------------------------------------------------------------------
    // 高阶请求方法
    // -----------------------------------------------------------------------

    /// POST JSON 请求，返回反序列化后的响应。
    ///
    /// 自动构建 headers/URL、检查 HTTP 状态、解析 JSON。
    /// 适用于非流式 API 调用。
    pub async fn post_json<B: Serialize, R: DeserializeOwned>(
        &self,
        req: &LlmRequest,
        path: &str,
        body: &B,
    ) -> Result<R, Error> {
        let response = self.send_post(req, path, body).await?;
        response.json::<R>().await.map_err(|e| Error::Provider {
            kind: ProviderErrorKind::InvalidResponse {
                message: format!("failed to parse response JSON: {e}"),
            },
            suggestion: None,
        })
    }

    /// POST JSON 请求，返回 SSE 事件流。
    ///
    /// 自动构建 headers/URL、检查 HTTP 状态、创建 SSE 流。
    /// 适用于流式 API 调用。
    pub async fn post_sse<B: Serialize>(
        &self,
        req: &LlmRequest,
        path: &str,
        body: &B,
    ) -> Result<SseStream, Error> {
        let response = self.send_post(req, path, body).await?;
        Ok(sse::from_response(response))
    }

    // -----------------------------------------------------------------------
    // 内部方法
    // -----------------------------------------------------------------------

    /// 发送 POST 请求并检查 HTTP 状态。
    async fn send_post<B: Serialize>(
        &self,
        req: &LlmRequest,
        path: &str,
        body: &B,
    ) -> Result<reqwest::Response, Error> {
        let url = Self::build_url(req, path);
        let headers = Self::build_headers(req);

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(body)
            .send()
            .await
            .map_err(|e| Error::Provider {
                kind: ProviderErrorKind::Transport {
                    message: format!("request failed: {e}"),
                },
                suggestion: None,
            })?;

        if !response.status().is_success() {
            return Err(Self::map_error_response(response).await);
        }

        Ok(response)
    }

    // -----------------------------------------------------------------------
    // 请求构建
    // -----------------------------------------------------------------------

    /// 从 `LlmRequest` 构建 HTTP headers。
    ///
    /// 合并 Content-Type、Authorization（Bearer api_key）、
    /// ModelRef 级别的 headers、Request 级别的 HTTP 覆写 headers。
    pub fn build_headers(req: &LlmRequest) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // API key → Bearer token
        if let Some(api_key) = &req.model.api_key {
            if let Ok(val) = HeaderValue::from_str(&format!("Bearer {api_key}")) {
                headers.insert(AUTHORIZATION, val);
            }
        }

        // ModelRef 级别的额外 headers
        if let Some(extra) = &req.model.headers {
            for (k, v) in extra {
                if let (Ok(name), Ok(val)) = (
                    reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                    HeaderValue::from_str(v),
                ) {
                    headers.insert(name, val);
                }
            }
        }

        // Request 级别的 HTTP 覆写 headers
        if let Some(http) = &req.http {
            if let Some(extra) = &http.headers {
                for (k, v) in extra {
                    if let (Ok(name), Ok(val)) = (
                        reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                        HeaderValue::from_str(v),
                    ) {
                        headers.insert(name, val);
                    }
                }
            }
        }

        headers
    }

    /// 从 `LlmRequest` 构建请求 URL。
    ///
    /// 格式：`{base_url}{path}?{query_params}`
    /// 合并 ModelRef 和 Request 级别的 query params。
    pub fn build_url(req: &LlmRequest, path: &str) -> String {
        let base = req.model.base_url.trim_end_matches('/');
        let mut url = format!("{base}{path}");

        let mut params: Vec<(&str, &str)> = Vec::new();
        if let Some(qp) = &req.model.query_params {
            for (k, v) in qp {
                params.push((k.as_str(), v.as_str()));
            }
        }
        if let Some(http) = &req.http {
            if let Some(qp) = &http.query_params {
                for (k, v) in qp {
                    params.push((k.as_str(), v.as_str()));
                }
            }
        }

        if !params.is_empty() {
            url.push('?');
            let qs: Vec<String> = params.iter().map(|(k, v)| format!("{k}={v}")).collect();
            url.push_str(&qs.join("&"));
        }

        url
    }

    // -----------------------------------------------------------------------
    // 错误映射
    // -----------------------------------------------------------------------

    /// 将 HTTP 错误响应映射为 `katu_core::Error`。
    ///
    /// 通用的 status code → `ProviderErrorKind` 映射，
    /// 适用于 OpenAI、Anthropic、DeepSeek 等大多数 provider。
    /// 错误消息通过 best-effort 解析 JSON body 提取。
    pub async fn map_error_response(response: reqwest::Response) -> Error {
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs);

        let body = response.text().await.unwrap_or_default();

        // Best-effort: 尝试提取嵌套的 error.message 或 message
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .or_else(|| v.get("message"))
                    .and_then(|m| m.as_str())
                    .map(String::from)
            })
            .unwrap_or_else(|| format!("HTTP {status}: {body}"));

        let kind = match status {
            401 => ProviderErrorKind::Authentication {
                message,
                kind: AuthErrorKind::Invalid,
            },
            403 => ProviderErrorKind::Authentication {
                message,
                kind: AuthErrorKind::InsufficientPermissions,
            },
            429 => ProviderErrorKind::RateLimit {
                message,
                retry_after,
            },
            400 | 422 => ProviderErrorKind::InvalidRequest { message },
            500..=599 => ProviderErrorKind::ServerError {
                message,
                status,
                retry_after,
            },
            _ => ProviderErrorKind::Unknown { message },
        };

        Error::Provider {
            kind,
            suggestion: None,
        }
    }
}

impl Default for HttpProviderClient {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use katu_core::{Message, ModelId, ProviderId, RouteId};
    use katu_llm::model::{ModelLimits, ModelRef};

    fn sample_request() -> LlmRequest {
        let model = ModelRef::new(
            ModelId::new("gpt-4o"),
            ProviderId::new("openai"),
            RouteId::new("openai-chat"),
            "https://api.openai.com/v1",
            ModelLimits {
                context_window: 128_000,
                max_output_tokens: 4096,
            },
        )
        .with_api_key("sk-test-key");

        LlmRequest::new(model).with_message(Message::user("Hello"))
    }

    #[test]
    fn test_build_url_with_path() {
        let req = sample_request();
        let url = HttpProviderClient::build_url(&req, "/chat/completions");
        assert_eq!(url, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_build_url_anthropic_path() {
        let model = ModelRef::new(
            ModelId::new("claude-sonnet-4-20250514"),
            ProviderId::new("anthropic"),
            RouteId::new("anthropic-messages"),
            "https://api.anthropic.com/v1",
            ModelLimits {
                context_window: 200_000,
                max_output_tokens: 8192,
            },
        );
        let req = LlmRequest::new(model);
        let url = HttpProviderClient::build_url(&req, "/messages");
        assert_eq!(url, "https://api.anthropic.com/v1/messages");
    }

    #[test]
    fn test_build_url_with_query_params() {
        let model = ModelRef::new(
            ModelId::new("gpt-4o"),
            ProviderId::new("azure"),
            RouteId::new("openai-chat"),
            "https://myinstance.openai.azure.com/openai/deployments/gpt-4o",
            ModelLimits {
                context_window: 128_000,
                max_output_tokens: 4096,
            },
        )
        .with_query_param("api-version", "2024-02-01");

        let req = LlmRequest::new(model);
        let url = HttpProviderClient::build_url(&req, "/chat/completions");
        assert!(url.contains("api-version=2024-02-01"));
    }

    #[test]
    fn test_build_headers_bearer_token() {
        let req = sample_request();
        let headers = HttpProviderClient::build_headers(&req);
        let auth = headers.get(AUTHORIZATION).unwrap().to_str().unwrap();
        assert_eq!(auth, "Bearer sk-test-key");
        assert_eq!(
            headers.get(CONTENT_TYPE).unwrap().to_str().unwrap(),
            "application/json"
        );
    }

    #[test]
    fn test_build_headers_no_api_key() {
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
        let req = LlmRequest::new(model);
        let headers = HttpProviderClient::build_headers(&req);
        assert!(headers.get(AUTHORIZATION).is_none());
    }

    #[test]
    fn test_build_headers_extra() {
        let model = ModelRef::new(
            ModelId::new("claude-sonnet-4-20250514"),
            ProviderId::new("anthropic"),
            RouteId::new("anthropic-messages"),
            "https://api.anthropic.com/v1",
            ModelLimits {
                context_window: 200_000,
                max_output_tokens: 8192,
            },
        )
        .with_header("x-api-key", "sk-ant-xxx")
        .with_header("anthropic-version", "2023-06-01");

        let req = LlmRequest::new(model);
        let headers = HttpProviderClient::build_headers(&req);
        assert_eq!(
            headers.get("x-api-key").unwrap().to_str().unwrap(),
            "sk-ant-xxx"
        );
        assert_eq!(
            headers.get("anthropic-version").unwrap().to_str().unwrap(),
            "2023-06-01"
        );
    }

    #[test]
    fn test_client_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HttpProviderClient>();
    }
}
