//! # katu_llm::model
//!
//! ## 职责
//! 定义模型引用 (`ModelRef`) 及其组成类型：能力描述、限制、定价、思考配置等。
//!
//! ## 对外接口
//! - `ModelRef` — 可执行的模型引用（身份 + 连接 + 能力 + 默认参数）
//! - `ModelLimits` — token 上限
//! - `ModelPricing` — 费率定义
//! - `ModelCapabilities` — 功能标志
//! - `InputModality` — 支持的输入模态
//! - `ThinkingMode` — 思考控制模式
//! - `ThinkingConfig` — 思考能力配置
//! - `ReasoningEffort` — 推理强度级别

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use katu_core::{ModelId, ProviderId, RouteId};

use crate::cache::CachePolicy;
use katu_core::GenerationOptions;
use crate::http::HttpOptions;

// ---------------------------------------------------------------------------
// InputModality
// ---------------------------------------------------------------------------

/// 模型支持的输入模态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputModality {
    /// 文本输入
    Text,
    /// 图像输入
    Image,
    /// 音频输入
    Audio,
    /// 视频输入
    Video,
}

// ---------------------------------------------------------------------------
// ReasoningEffort
// ---------------------------------------------------------------------------

/// 推理强度级别。
///
/// 对应 OpenAI `reasoning_effort` 和 Anthropic `thinking` 级别的统一抽象。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
    XHigh,
    Max
}

// ---------------------------------------------------------------------------
// ThinkingMode
// ---------------------------------------------------------------------------

/// 思考/推理的控制模式。
///
/// 不同 provider 使用不同机制控制推理行为：
/// - Anthropic: adaptive 或 budget（指定 token 预算）
/// - OpenAI: effort 级别（low/medium/high）
/// - 其他 provider: 可能不支持
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingMode {
    /// 自适应思考（Anthropic adaptive thinking）
    Adaptive,
    /// Budget 模式（指定 token 预算上限）
    Budget,
    /// Effort 级别模式（OpenAI reasoning_effort）
    Effort,
}

// ---------------------------------------------------------------------------
// ThinkingConfig
// ---------------------------------------------------------------------------

/// 模型的思考/推理能力配置。
///
/// 描述模型如何支持推理，以及推理的控制参数。
/// 仅在 `ModelCapabilities::thinking` 为 `Some` 时有意义。
///
/// # Examples
/// ```
/// use katu_llm::model::{ThinkingConfig, ThinkingMode, ReasoningEffort};
///
/// // Anthropic adaptive
/// let config = ThinkingConfig {
///     mode: ThinkingMode::Adaptive,
///     default_budget: None,
///     min_effort: None,
///     max_effort: None,
/// };
///
/// // OpenAI effort-based
/// let config = ThinkingConfig {
///     mode: ThinkingMode::Effort,
///     default_budget: None,
///     min_effort: Some(ReasoningEffort::Low),
///     max_effort: Some(ReasoningEffort::High),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// 思考控制模式
    pub mode: ThinkingMode,
    /// 默认思考 token 预算（仅 Budget 模式有意义）
    pub default_budget: Option<u32>,
    /// 支持的最低 effort 级别
    pub min_effort: Option<ReasoningEffort>,
    /// 支持的最高 effort 级别
    pub max_effort: Option<ReasoningEffort>,
}

// ---------------------------------------------------------------------------
// ModelCapabilities
// ---------------------------------------------------------------------------

/// 模型功能标志。
///
/// 描述模型支持哪些特性，供 Agent loop 和 Provider 适配层参考。
///
/// # Examples
/// ```
/// use katu_llm::model::{ModelCapabilities, InputModality};
///
/// let caps = ModelCapabilities {
///     input_modalities: vec![InputModality::Text, InputModality::Image],
///     tool_calls: true,
///     streaming_tool_input: true,
///     structured_output: true,
///     prompt_caching: true,
///     thinking: None,
/// };
/// assert!(caps.supports_modality(InputModality::Image));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// 支持的输入模态列表
    pub input_modalities: Vec<InputModality>,
    /// 是否支持工具调用
    pub tool_calls: bool,
    /// 是否支持流式工具参数输入
    pub streaming_tool_input: bool,
    /// 是否支持结构化输出（JSON mode / response_format）
    pub structured_output: bool,
    /// 是否支持 prompt caching
    pub prompt_caching: bool,
    /// 思考/推理能力配置，`None` 表示不支持
    pub thinking: Option<ThinkingConfig>,
}

impl ModelCapabilities {
    /// 检查是否支持指定输入模态。
    pub fn supports_modality(&self, modality: InputModality) -> bool {
        self.input_modalities.contains(&modality)
    }

    /// 检查是否支持推理/思考。
    pub fn supports_thinking(&self) -> bool {
        self.thinking.is_some()
    }
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            input_modalities: vec![InputModality::Text],
            tool_calls: true,
            streaming_tool_input: false,
            structured_output: false,
            prompt_caching: false,
            thinking: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ModelLimits
// ---------------------------------------------------------------------------

/// 模型 token 上限。
///
/// # Examples
/// ```
/// use katu_llm::ModelLimits;
///
/// let limits = ModelLimits {
///     context_window: 200_000,
///     max_output_tokens: 8192,
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLimits {
    /// 上下文窗口大小（input + output 总量上限）
    pub context_window: u32,
    /// 最大输出 token 数
    pub max_output_tokens: u32,
}

// ---------------------------------------------------------------------------
// ModelPricing
// ---------------------------------------------------------------------------

/// 模型费率定义（单位：美元 / 百万 token）。
///
/// 用于从 `Usage` 计算 `Cost`。
///
/// # Examples
/// ```
/// use katu_llm::ModelPricing;
///
/// let pricing = ModelPricing {
///     input: 3.0,        // $3 / M input tokens
///     output: 15.0,      // $15 / M output tokens
///     cache_read: 0.30,  // $0.30 / M cache read tokens
///     cache_write: 3.75, // $3.75 / M cache write tokens
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPricing {
    /// 输入 token 费率（$/M tokens）
    pub input: f64,
    /// 输出 token 费率（$/M tokens）
    pub output: f64,
    /// 缓存读取费率（$/M tokens）
    pub cache_read: f64,
    /// 缓存写入费率（$/M tokens）
    pub cache_write: f64,
}

// ---------------------------------------------------------------------------
// ModelRef
// ---------------------------------------------------------------------------

/// 可执行的模型引用。
///
/// `ModelRef` 是从"选择模型"到"发出请求"所需全部信息的聚合体：
/// - **身份**：provider/model/route 三元组 + 可选显示名
/// - **连接**：base_url, api_key, 额外 headers/query_params
/// - **能力**：token 限制、功能标志、输入模态、思考配置
/// - **默认参数**：生成选项、缓存策略（被 Request 级覆盖）
/// - **定价**：用于 Usage → Cost 计算
/// - **Provider 私有**：非标选项（如 Bedrock region, Vertex project_id）
///
/// # 参数合并链
/// ```text
/// LlmRequest.generation > Agent 配置 > ModelRef.generation > Route defaults
/// ```
///
/// # Examples
/// ```
/// use katu_core::{ModelId, ProviderId, RouteId};
/// use katu_llm::model::*;
/// use katu_llm::GenerationOptions;
///
/// let model = ModelRef::new(
///     ModelId::new("claude-sonnet-4-20250514"),
///     ProviderId::new("anthropic"),
///     RouteId::new("anthropic-messages"),
///     "https://api.anthropic.com/v1",
///     ModelLimits {
///         context_window: 200_000,
///         max_output_tokens: 8192,
///     },
/// )
/// .with_display_name("Claude Sonnet 4")
/// .with_api_key("sk-ant-xxx")
/// .with_capabilities(ModelCapabilities {
///     input_modalities: vec![InputModality::Text, InputModality::Image],
///     tool_calls: true,
///     streaming_tool_input: true,
///     structured_output: false,
///     prompt_caching: true,
///     thinking: Some(ThinkingConfig {
///         mode: ThinkingMode::Adaptive,
///         default_budget: None,
///         min_effort: None,
///         max_effort: None,
///     }),
/// })
/// .with_pricing(ModelPricing {
///     input: 3.0,
///     output: 15.0,
///     cache_read: 0.30,
///     cache_write: 3.75,
/// })
/// .with_generation(GenerationOptions::new().with_max_tokens(4096));
///
/// assert_eq!(model.id.as_str(), "claude-sonnet-4-20250514");
/// assert!(model.capabilities.supports_thinking());
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRef {
    // ─── 身份标识 ───

    /// 模型 ID（发送给 API 的 wire 值）
    pub id: ModelId,
    /// Provider 标识
    pub provider: ProviderId,
    /// 路由标识（决定使用哪个 Protocol 转换器）
    pub route: RouteId,
    /// 人类可读名称
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,

    // ─── 连接信息 ───

    /// API Base URL
    pub base_url: String,
    /// API Key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// 额外的固定请求头
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// URL 查询参数（如 Azure api-version）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_params: Option<HashMap<String, String>>,

    // ─── 能力与限制 ───

    /// Token 上限
    pub limits: ModelLimits,
    /// 功能标志
    pub capabilities: ModelCapabilities,

    // ─── 默认参数 ───

    /// 模型级默认生成参数（被 Request 级覆盖）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationOptions>,
    /// 默认缓存策略
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_policy: Option<CachePolicy>,

    // ─── 定价 ───

    /// 费率（用于 Usage → Cost 计算）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<ModelPricing>,

    // ─── Provider 私有 ───

    /// Provider 特有的非标选项
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<serde_json::Value>,
    /// HTTP 传输层覆写
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http: Option<HttpOptions>,
}

impl ModelRef {
    /// 创建一个 ModelRef，仅包含必需字段。
    pub fn new(
        id: ModelId,
        provider: ProviderId,
        route: RouteId,
        base_url: impl Into<String>,
        limits: ModelLimits,
    ) -> Self {
        Self {
            id,
            provider,
            route,
            display_name: None,
            base_url: base_url.into(),
            api_key: None,
            headers: None,
            query_params: None,
            limits,
            capabilities: ModelCapabilities::default(),
            generation: None,
            cache_policy: None,
            pricing: None,
            provider_options: None,
            http: None,
        }
    }

    /// 设置显示名称。
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// 设置 API key。
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// 添加一个额外请求头。
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }

    /// 添加一个查询参数。
    pub fn with_query_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }

    /// 设置模型能力。
    pub fn with_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// 设置默认生成参数。
    pub fn with_generation(mut self, generation: GenerationOptions) -> Self {
        self.generation = Some(generation);
        self
    }

    /// 设置缓存策略。
    pub fn with_cache_policy(mut self, policy: CachePolicy) -> Self {
        self.cache_policy = Some(policy);
        self
    }

    /// 设置定价。
    pub fn with_pricing(mut self, pricing: ModelPricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    /// 设置 provider 私有选项。
    pub fn with_provider_options(mut self, options: serde_json::Value) -> Self {
        self.provider_options = Some(options);
        self
    }

    /// 设置 HTTP 覆写选项。
    pub fn with_http(mut self, http: HttpOptions) -> Self {
        self.http = Some(http);
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_model() -> ModelRef {
        ModelRef::new(
            ModelId::new("claude-sonnet-4-20250514"),
            ProviderId::new("anthropic"),
            RouteId::new("anthropic-messages"),
            "https://api.anthropic.com/v1",
            ModelLimits {
                context_window: 200_000,
                max_output_tokens: 8192,
            },
        )
    }

    #[test]
    fn test_new_has_required_fields() {
        let m = sample_model();
        assert_eq!(m.id.as_str(), "claude-sonnet-4-20250514");
        assert_eq!(m.provider.as_str(), "anthropic");
        assert_eq!(m.route.as_str(), "anthropic-messages");
        assert_eq!(m.base_url, "https://api.anthropic.com/v1");
        assert_eq!(m.limits.context_window, 200_000);
        assert_eq!(m.limits.max_output_tokens, 8192);
    }

    #[test]
    fn test_new_optional_fields_are_none() {
        let m = sample_model();
        assert_eq!(m.display_name, None);
        assert_eq!(m.api_key, None);
        assert_eq!(m.headers, None);
        assert_eq!(m.generation, None);
        assert_eq!(m.pricing, None);
        assert_eq!(m.provider_options, None);
        assert_eq!(m.http, None);
    }

    #[test]
    fn test_builder_chain() {
        let m = sample_model()
            .with_display_name("Claude Sonnet 4")
            .with_api_key("sk-ant-xxx")
            .with_header("x-custom", "value")
            .with_query_param("version", "1")
            .with_generation(GenerationOptions::new().with_max_tokens(4096))
            .with_cache_policy(CachePolicy::Auto)
            .with_pricing(ModelPricing {
                input: 3.0,
                output: 15.0,
                cache_read: 0.30,
                cache_write: 3.75,
            });

        assert_eq!(m.display_name.as_deref(), Some("Claude Sonnet 4"));
        assert_eq!(m.api_key.as_deref(), Some("sk-ant-xxx"));
        assert_eq!(
            m.headers.as_ref().unwrap().get("x-custom").unwrap(),
            "value"
        );
        assert_eq!(
            m.generation.as_ref().unwrap().max_tokens,
            Some(4096)
        );
        assert_eq!(m.pricing.as_ref().unwrap().input, 3.0);
    }

    #[test]
    fn test_capabilities_default() {
        let m = sample_model();
        assert!(m.capabilities.supports_modality(InputModality::Text));
        assert!(!m.capabilities.supports_modality(InputModality::Image));
        assert!(m.capabilities.tool_calls);
        assert!(!m.capabilities.supports_thinking());
    }

    #[test]
    fn test_capabilities_with_thinking() {
        let m = sample_model().with_capabilities(ModelCapabilities {
            input_modalities: vec![InputModality::Text, InputModality::Image],
            tool_calls: true,
            streaming_tool_input: true,
            structured_output: false,
            prompt_caching: true,
            thinking: Some(ThinkingConfig {
                mode: ThinkingMode::Adaptive,
                default_budget: None,
                min_effort: None,
                max_effort: None,
            }),
        });

        assert!(m.capabilities.supports_thinking());
        assert!(m.capabilities.supports_modality(InputModality::Image));
        assert!(m.capabilities.streaming_tool_input);
    }

    #[test]
    fn test_serde_roundtrip_minimal() {
        let m = sample_model();
        let json = serde_json::to_string(&m).unwrap();
        let restored: ModelRef = serde_json::from_str(&json).unwrap();
        assert_eq!(m.id, restored.id);
        assert_eq!(m.provider, restored.provider);
        assert_eq!(m.limits, restored.limits);
    }

    #[test]
    fn test_serde_roundtrip_full() {
        let m = sample_model()
            .with_display_name("Claude Sonnet 4")
            .with_api_key("sk-test")
            .with_capabilities(ModelCapabilities {
                input_modalities: vec![InputModality::Text, InputModality::Image],
                tool_calls: true,
                streaming_tool_input: true,
                structured_output: true,
                prompt_caching: true,
                thinking: Some(ThinkingConfig {
                    mode: ThinkingMode::Budget,
                    default_budget: Some(10000),
                    min_effort: Some(ReasoningEffort::Low),
                    max_effort: Some(ReasoningEffort::High),
                }),
            })
            .with_generation(GenerationOptions::new().with_max_tokens(4096).with_temperature(0.7))
            .with_cache_policy(CachePolicy::Auto)
            .with_pricing(ModelPricing {
                input: 3.0,
                output: 15.0,
                cache_read: 0.30,
                cache_write: 3.75,
            })
            .with_provider_options(serde_json::json!({"region": "us-east-1"}))
            .with_http(HttpOptions::new().with_header("x-extra", "val"));

        let json = serde_json::to_string_pretty(&m).unwrap();
        let restored: ModelRef = serde_json::from_str(&json).unwrap();
        assert_eq!(m, restored);
    }

    #[test]
    fn test_serde_skips_none_fields() {
        let m = sample_model();
        let json = serde_json::to_string(&m).unwrap();
        // None fields with skip_serializing_if should not appear
        assert!(!json.contains("display_name"));
        assert!(!json.contains("api_key"));
        assert!(!json.contains("pricing"));
        assert!(!json.contains("provider_options"));
    }
}
