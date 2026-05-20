//! # katu_core::event
//!
//! ## 职责
//! 定义 LLM 层的流式事件类型（OpenCode 风格的细粒度 start/delta/end 三段式）。
//!
//! ## 设计
//! `StreamEvent` 是 provider 无关的 LLM 流式输出契约：
//! - Provider adapter 将 provider 特定的 SSE/WebSocket 帧**翻译**为 `StreamEvent`
//! - Agent loop **消费** `StreamEvent` 流，驱动工具执行和状态更新
//! - UI/持久化层可直接订阅 `StreamEvent` 做实时渲染
//!
//! ## 事件生命周期
//! ```text
//! StepStart → [TextStart → TextDelta* → TextEnd]
//!           → [ReasoningStart → ReasoningDelta* → ReasoningEnd]
//!           → [ToolCallStart → ToolCallDelta* → ToolCallEnd]
//!           → StepFinish
//! (重复多个 Step，或直到 Finish / ProviderError)
//! ```
//!
//! ## 对外接口
//! - `StreamEvent` — LLM 流式事件枚举
//! - `ToolResultValue` — 工具返回值（json / text / error）
//!
//! ## 调用者
//! - `katu-llm` (future) — provider adapter 产出 StreamEvent
//! - `katu-agent` (future) — agent loop 消费 StreamEvent
//! - UI 层 — 实时渲染流式输出

use serde::{Deserialize, Serialize};

use crate::types::{FinishReason, ToolCallId};
use crate::usage::Usage;

// ===========================================================================
// ToolResultValue
// ===========================================================================

/// 工具执行返回值 — 区分 JSON 结构、纯文本、错误三种类型。
///
/// Provider adapter 在收到 `tool-result` 帧时构造此值，
/// agent loop 据此决定是否标记为工具错误
/// !tool.rs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultValue {
    /// JSON 结构化结果
    Json { value: serde_json::Value },
    /// 纯文本结果
    Text { value: String },
    /// 错误结果（工具执行失败）
    Error { value: String },
}

// ===========================================================================
// StreamEvent
// ===========================================================================

/// LLM 流式事件 — provider 无关的细粒度输出事件。
///
/// 遵循 OpenCode 的 start/delta/end 三段式设计：
/// - **start** — 标记一个内容块开始，消费者可据此创建 UI 占位
/// - **delta** — 增量数据，消费者追加到当前块
/// - **end** — 标记内容块结束，消费者可据此 finalize
///
/// 每个事件通过 `content_index` 标识其所属的内容块位置
/// （一次 LLM 回复可能包含多个并行内容块）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    // ----- Step 生命周期 -----
    /// 一个推理步骤开始（provider 可能在一次请求中产出多步）
    StepStart {
        index: u32,
    },

    /// 一个推理步骤结束，携带停止原因和 token 用量
    StepFinish {
        index: u32,
        finish_reason: FinishReason,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },

    // ----- Text 流 -----
    /// 文本内容块开始
    TextStart {
        content_index: usize,
    },

    /// 文本增量
    TextDelta {
        content_index: usize,
        delta: String,
    },

    /// 文本内容块结束
    TextEnd {
        content_index: usize,
    },

    // ----- Reasoning 流 -----
    /// 推理/思考内容块开始
    ReasoningStart {
        content_index: usize,
    },

    /// 推理增量
    ReasoningDelta {
        content_index: usize,
        delta: String,
    },

    /// 推理内容块结束
    ReasoningEnd {
        content_index: usize,
    },

    // ----- ToolCall 参数流 -----
    /// 工具调用开始（已知 id 和 name）
    ToolCallStart {
        content_index: usize,
        id: ToolCallId,
        name: String,
    },

    /// 工具调用参数增量（JSON 字符串片段）
    ToolCallDelta {
        content_index: usize,
        delta: String,
    },

    /// 工具调用参数流结束
    ToolCallEnd {
        content_index: usize,
    },

    // ----- Tool 执行结果（由 agent loop 注入事件流）-----
    /// 工具执行成功
    ToolResult {
        id: ToolCallId,
        name: String,
        result: ToolResultValue,
    },

    /// 工具执行失败
    ToolError {
        id: ToolCallId,
        name: String,
        message: String,
    },

    // ----- 终态 -----
    /// 整个 LLM 请求完成
    Finish {
        finish_reason: FinishReason,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },

    /// Provider 级别错误（网络、认证、限流等）
    ProviderError {
        message: String,
        retryable: bool,
    },
}

// ---------------------------------------------------------------------------
// StreamEvent — helper methods
// ---------------------------------------------------------------------------

impl StreamEvent {
    /// 是否为终态事件（Finish 或 ProviderError）。
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Finish { .. } | Self::ProviderError { .. })
    }

    /// 是否为文本增量事件。
    pub fn is_text_delta(&self) -> bool {
        matches!(self, Self::TextDelta { .. })
    }

    /// 是否为推理增量事件。
    pub fn is_reasoning_delta(&self) -> bool {
        matches!(self, Self::ReasoningDelta { .. })
    }

    /// 提取文本增量内容，非 TextDelta 事件返回 None。
    pub fn as_text_delta(&self) -> Option<&str> {
        match self {
            Self::TextDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        }
    }

    /// 提取推理增量内容，非 ReasoningDelta 事件返回 None。
    pub fn as_reasoning_delta(&self) -> Option<&str> {
        match self {
            Self::ReasoningDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_event_step_lifecycle_serde() {
        let start = StreamEvent::StepStart { index: 0 };
        let json = serde_json::to_string(&start).unwrap();
        assert!(json.contains(r#""type":"step_start""#));
        let restored: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(start, restored);

        let finish = StreamEvent::StepFinish {
            index: 0,
            finish_reason: FinishReason::Stop,
            usage: None,
        };
        let json = serde_json::to_string(&finish).unwrap();
        assert!(json.contains(r#""type":"step_finish""#));
        assert!(!json.contains("usage"));
        let restored: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(finish, restored);
    }

    #[test]
    fn test_stream_event_text_serde() {
        let delta = StreamEvent::TextDelta {
            content_index: 0,
            delta: "Hello".into(),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(json.contains(r#""type":"text_delta""#));
        assert!(json.contains(r#""delta":"Hello""#));
        let restored: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(delta, restored);
    }

    #[test]
    fn test_stream_event_reasoning_serde() {
        let delta = StreamEvent::ReasoningDelta {
            content_index: 1,
            delta: "thinking...".into(),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(json.contains(r#""type":"reasoning_delta""#));
        let restored: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(delta, restored);
    }

    #[test]
    fn test_stream_event_tool_call_serde() {
        let start = StreamEvent::ToolCallStart {
            content_index: 2,
            id: ToolCallId::new("call_abc"),
            name: "read_file".into(),
        };
        let json = serde_json::to_string(&start).unwrap();
        assert!(json.contains(r#""type":"tool_call_start""#));
        let restored: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(start, restored);
    }

    #[test]
    fn test_stream_event_tool_result_serde() {
        let event = StreamEvent::ToolResult {
            id: ToolCallId::new("call_1"),
            name: "bash".into(),
            result: ToolResultValue::Text {
                value: "exit 0".into(),
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"tool_result""#));
        let restored: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_stream_event_tool_error_serde() {
        let event = StreamEvent::ToolError {
            id: ToolCallId::new("call_2"),
            name: "write_file".into(),
            message: "permission denied".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"tool_error""#));
        let restored: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_stream_event_finish_with_usage() {
        let event = StreamEvent::Finish {
            finish_reason: FinishReason::ToolCalls,
            usage: Some(Usage {
                input_tokens: 100,
                output_tokens: 50,
                total_tokens: 150,
                ..Default::default()
            }),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"finish""#));
        assert!(json.contains("usage"));
        let restored: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    #[test]
    fn test_stream_event_provider_error_serde() {
        let event = StreamEvent::ProviderError {
            message: "rate limit exceeded".into(),
            retryable: true,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"provider_error""#));
        let restored: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, restored);
    }

    // -- ToolResultValue --

    #[test]
    fn test_tool_result_value_json_serde() {
        let v = ToolResultValue::Json {
            value: serde_json::json!({"count": 42}),
        };
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains(r#""type":"json""#));
        let restored: ToolResultValue = serde_json::from_str(&json).unwrap();
        assert_eq!(v, restored);
    }

    #[test]
    fn test_tool_result_value_text_serde() {
        let v = ToolResultValue::Text {
            value: "hello".into(),
        };
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains(r#""type":"text""#));
        let restored: ToolResultValue = serde_json::from_str(&json).unwrap();
        assert_eq!(v, restored);
    }

    #[test]
    fn test_tool_result_value_error_serde() {
        let v = ToolResultValue::Error {
            value: "not found".into(),
        };
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains(r#""type":"error""#));
        let restored: ToolResultValue = serde_json::from_str(&json).unwrap();
        assert_eq!(v, restored);
    }

    // -- Helper methods --

    #[test]
    fn test_is_terminal() {
        assert!(StreamEvent::Finish {
            finish_reason: FinishReason::Stop,
            usage: None,
        }
        .is_terminal());

        assert!(StreamEvent::ProviderError {
            message: "err".into(),
            retryable: false,
        }
        .is_terminal());

        assert!(!StreamEvent::TextDelta {
            content_index: 0,
            delta: "hi".into(),
        }
        .is_terminal());
    }

    #[test]
    fn test_as_text_delta() {
        let event = StreamEvent::TextDelta {
            content_index: 0,
            delta: "hello".into(),
        };
        assert_eq!(event.as_text_delta(), Some("hello"));

        let other = StreamEvent::ReasoningDelta {
            content_index: 0,
            delta: "think".into(),
        };
        assert_eq!(other.as_text_delta(), None);
    }

    #[test]
    fn test_as_reasoning_delta() {
        let event = StreamEvent::ReasoningDelta {
            content_index: 1,
            delta: "hmm".into(),
        };
        assert_eq!(event.as_reasoning_delta(), Some("hmm"));

        let other = StreamEvent::TextDelta {
            content_index: 0,
            delta: "hi".into(),
        };
        assert_eq!(other.as_reasoning_delta(), None);
    }
}
