//! OpenAI Chat Completions API wire types.
//!
//! 这些结构体精确对应 OpenAI REST API 的 JSON 格式，
//! 仅用于序列化/反序列化，不暴露给外部消费者。
//! 看看能不能提出来？
use serde::{Deserialize, Serialize};

// ===========================================================================
// Request types
// ===========================================================================

/// POST /v1/chat/completions 请求体
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptionsParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolParam>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

/// Stop sequences — 单个或多个
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum StopParam {
    Single(String),
    Multiple(Vec<String>),
}

/// stream_options 参数
#[derive(Debug, Clone, Serialize)]
pub struct StreamOptionsParam {
    pub include_usage: bool,
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// OpenAI chat message (request)
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System {
        content: String,
    },
    User {
        content: UserContentParam,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<AssistantToolCallParam>>,
    },
    Tool {
        content: String,
        tool_call_id: String,
    },
}

/// User message content — string or array of parts
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum UserContentParam {
    Text(String),
    Parts(Vec<ContentPartParam>),
}

/// Content part for user messages
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPartParam {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrlParam,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageUrlParam {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Tool call in assistant message (request, 用于多轮对话重放)
#[derive(Debug, Clone, Serialize)]
pub struct AssistantToolCallParam {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCallParam,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionCallParam {
    pub name: String,
    pub arguments: String,
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// Tool definition for request
#[derive(Debug, Clone, Serialize)]
pub struct ToolParam {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefParam,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDefParam {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

// ---------------------------------------------------------------------------
// Tool choice
// ---------------------------------------------------------------------------

/// tool_choice parameter
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ToolChoiceParam {
    /// "none" | "auto" | "required"
    Mode(String),
    /// { "type": "function", "function": { "name": "..." } }
    Specific(NamedToolChoice),
}

#[derive(Debug, Clone, Serialize)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub function: NamedToolChoiceFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct NamedToolChoiceFunction {
    pub name: String,
}

// ===========================================================================
// Response types (non-streaming)
// ===========================================================================

/// ChatCompletion response object
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub created: u64,
    pub model: String,
    #[serde(default)]
    pub usage: Option<CompletionUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ResponseFunctionCall,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

// ===========================================================================
// Streaming response types
// ===========================================================================

/// chat.completion.chunk
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    pub created: u64,
    pub model: String,
    #[serde(default)]
    pub usage: Option<CompletionUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

/// Tool call delta in streaming
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkToolCall {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    pub call_type: Option<String>,
    #[serde(default)]
    pub function: Option<ChunkFunctionCall>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkFunctionCall {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

// ===========================================================================
// Shared types
// ===========================================================================

/// Usage statistics
///
/// 某些代理/provider 在非最终 chunk 中返回 `"usage":{}` 空对象，
/// 因此所有字段需要 `#[serde(default)]` 以容忍缺失。
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CompletionUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}

// ===========================================================================
// Error response
// ===========================================================================

/// OpenAI API error response body
#[derive(Debug, Clone, Deserialize)]
pub struct ApiErrorResponse {
    pub error: ApiError,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: Option<String>,
    pub code: Option<String>,
}
