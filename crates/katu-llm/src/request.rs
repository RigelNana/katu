//! # katu_llm::request
//!
//! ## 职责
//! 定义 LLM 请求与响应的完整数据结构。
//!
//! ## 对外接口
//! - `LlmRequest` — 发给 Provider 的完整请求
//! - `LlmResponse` — Provider 返回的完整响应

use serde::{Deserialize, Serialize};

use katu_core::{FinishReason, Message, ToolChoice, ToolDefinition, Usage};

use crate::cache::CachePolicy;
use katu_core::GenerationOptions;
use crate::http::HttpOptions;
use crate::model::ModelRef;

// ---------------------------------------------------------------------------
// LlmRequest
// ---------------------------------------------------------------------------

/// 发给 LLM Provider 的完整请求。
///
/// 聚合了模型引用、消息历史、工具定义、生成参数等所有信息。
/// Provider 适配层从此结构中提取字段并转换为 provider 原生格式。
///
/// ## 参数解析优先级
/// ```text
/// LlmRequest.generation > ModelRef.generation > Route defaults
/// LlmRequest.http       > ModelRef.http       > Route defaults
/// ```
///
/// # Examples
/// ```
/// use katu_core::{ModelId, ProviderId, RouteId, Message, UserMessage, UserContent};
/// use katu_llm::request::LlmRequest;
/// use katu_llm::model::{ModelRef, ModelLimits};
/// use katu_llm::GenerationOptions;
///
/// let model = ModelRef::new(
///     ModelId::new("gpt-4o"),
///     ProviderId::new("openai"),
///     RouteId::new("openai-chat"),
///     "https://api.openai.com/v1",
///     ModelLimits { context_window: 128_000, max_output_tokens: 4096 },
/// );
///
/// let request = LlmRequest::new(model)
///     .with_system("You are a helpful assistant.")
///     .with_message(Message::user("Hello!"))
///     .with_generation(GenerationOptions::new().with_max_tokens(1024));
///
/// assert_eq!(request.system.as_deref(), Some("You are a helpful assistant."));
/// assert_eq!(request.messages.len(), 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    /// 目标模型引用
    pub model: ModelRef,
    /// System prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// 对话消息历史
    pub messages: Vec<Message>,
    /// 可用工具定义
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    /// 工具选择策略
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// 请求级生成参数（覆盖 ModelRef 默认值）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationOptions>,
    /// 请求级缓存策略
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CachePolicy>,
    /// 请求级 HTTP 覆写
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http: Option<HttpOptions>,
    /// Provider 特有的非标选项
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<serde_json::Value>,
    /// 请求元数据（追踪、审计用）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl LlmRequest {
    /// 创建一个最小化的 LlmRequest。
    pub fn new(model: ModelRef) -> Self {
        Self {
            model,
            system: None,
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            generation: None,
            cache: None,
            http: None,
            provider_options: None,
            metadata: None,
        }
    }

    /// 设置 system prompt。
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// 添加一条消息。
    pub fn with_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// 批量设置消息。
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// 设置工具定义。
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// 设置工具选择策略。
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// 设置请求级生成参数。
    pub fn with_generation(mut self, generation: GenerationOptions) -> Self {
        self.generation = Some(generation);
        self
    }

    /// 设置缓存策略。
    pub fn with_cache(mut self, cache: CachePolicy) -> Self {
        self.cache = Some(cache);
        self
    }

    /// 设置 HTTP 覆写。
    pub fn with_http(mut self, http: HttpOptions) -> Self {
        self.http = Some(http);
        self
    }

    /// 设置 provider 私有选项。
    pub fn with_provider_options(mut self, options: serde_json::Value) -> Self {
        self.provider_options = Some(options);
        self
    }

    /// 设置元数据。
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// 解析最终生效的 GenerationOptions。
    ///
    /// 合并 ModelRef.generation 和 Request.generation，
    /// Request 级优先。
    pub fn resolved_generation(&self) -> GenerationOptions {
        match (&self.model.generation, &self.generation) {
            (Some(model_gen), Some(req_gen)) => model_gen.merge(req_gen),
            (Some(model_gen), None) => model_gen.clone(),
            (None, Some(req_gen)) => req_gen.clone(),
            (None, None) => GenerationOptions::default(),
        }
    }

    /// 解析最终生效的 HttpOptions。
    ///
    /// 合并 ModelRef.http 和 Request.http，Request 级优先。
    pub fn resolved_http(&self) -> Option<HttpOptions> {
        match (&self.model.http, &self.http) {
            (Some(model_http), Some(req_http)) => Some(model_http.merge(req_http)),
            (Some(h), None) | (None, Some(h)) => Some(h.clone()),
            (None, None) => None,
        }
    }

    /// 解析最终生效的 CachePolicy。
    pub fn resolved_cache(&self) -> CachePolicy {
        self.cache
            .clone()
            .or_else(|| self.model.cache_policy.clone())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// LlmResponse
// ---------------------------------------------------------------------------

/// Provider 返回的完整响应。
///
/// 从流式事件收集而来，包含最终的 assistant 消息、
/// 停止原因和 token 用量统计。
///
/// # Examples
/// ```
/// use katu_core::{FinishReason, Usage, Message};
/// use katu_llm::request::LlmResponse;
///
/// let response = LlmResponse {
///     message: Message::assistant("Hello!"),
///     finish_reason: FinishReason::Stop,
///     usage: Usage::default(),
/// };
/// assert_eq!(response.finish_reason, FinishReason::Stop);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// 完整的 assistant 消息
    pub message: Message,
    /// 停止原因
    pub finish_reason: FinishReason,
    /// Token 用量统计
    pub usage: Usage,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use katu_core::{ModelId, ProviderId, RouteId};

    use crate::model::ModelLimits;

    fn sample_model() -> ModelRef {
        ModelRef::new(
            ModelId::new("gpt-4o"),
            ProviderId::new("openai"),
            RouteId::new("openai-chat"),
            "https://api.openai.com/v1",
            ModelLimits {
                context_window: 128_000,
                max_output_tokens: 4096,
            },
        )
    }

    #[test]
    fn test_new_is_minimal() {
        let req = LlmRequest::new(sample_model());
        assert!(req.system.is_none());
        assert!(req.messages.is_empty());
        assert!(req.tools.is_empty());
        assert!(req.tool_choice.is_none());
        assert!(req.generation.is_none());
    }

    #[test]
    fn test_builder_chain() {
        let req = LlmRequest::new(sample_model())
            .with_system("You are helpful.")
            .with_message(Message::user("Hi"))
            .with_generation(GenerationOptions::new().with_max_tokens(1024));

        assert_eq!(req.system.as_deref(), Some("You are helpful."));
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.generation.as_ref().unwrap().max_tokens, Some(1024));
    }

    #[test]
    fn test_resolved_generation_request_overrides_model() {
        let model = sample_model()
            .with_generation(GenerationOptions::new().with_max_tokens(2048).with_temperature(0.5));

        let req = LlmRequest::new(model)
            .with_generation(GenerationOptions::new().with_max_tokens(4096));

        let resolved = req.resolved_generation();
        assert_eq!(resolved.max_tokens, Some(4096)); // request wins
        assert_eq!(resolved.temperature, Some(0.5)); // model fallback
    }

    #[test]
    fn test_resolved_generation_model_only() {
        let model =
            sample_model().with_generation(GenerationOptions::new().with_temperature(0.7));

        let req = LlmRequest::new(model);
        let resolved = req.resolved_generation();
        assert_eq!(resolved.temperature, Some(0.7));
        assert_eq!(resolved.max_tokens, None);
    }

    #[test]
    fn test_resolved_generation_neither() {
        let req = LlmRequest::new(sample_model());
        let resolved = req.resolved_generation();
        assert_eq!(resolved, GenerationOptions::default());
    }

    #[test]
    fn test_resolved_cache_defaults_to_auto() {
        let req = LlmRequest::new(sample_model());
        assert_eq!(req.resolved_cache(), CachePolicy::Auto);
    }

    #[test]
    fn test_resolved_cache_request_overrides_model() {
        let model = sample_model().with_cache_policy(CachePolicy::Auto);
        let req = LlmRequest::new(model).with_cache(CachePolicy::None);
        assert_eq!(req.resolved_cache(), CachePolicy::None);
    }

    #[test]
    fn test_resolved_http_merge() {
        let model = sample_model().with_http(
            HttpOptions::new()
                .with_header("x-base", "1")
                .with_query_param("v", "1"),
        );

        let req = LlmRequest::new(model)
            .with_http(HttpOptions::new().with_header("x-req", "2"));

        let resolved = req.resolved_http().unwrap();
        let headers = resolved.headers.unwrap();
        assert_eq!(headers.get("x-base").unwrap(), "1");
        assert_eq!(headers.get("x-req").unwrap(), "2");
    }

    #[test]
    fn test_llm_response() {
        let response = LlmResponse {
            message: Message::assistant("Hello!"),
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }
}
