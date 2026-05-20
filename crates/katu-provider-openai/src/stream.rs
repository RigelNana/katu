//! SSE 流解析：OpenAI chat.completion.chunk → katu StreamEvent。
//!
//! 负责处理 tool_call delta 累积、content_index 跟踪、
//! 以及 finish_reason / usage 的最终事件发射。

use std::collections::HashMap;

use futures_core::Stream;
use std::pin::Pin;

use katu_core::{Error, FinishReason, StreamEvent, ToolCallId};
use katu_provider_http::SseStream;

use crate::convert;
use crate::types::*;

// ===========================================================================
// StreamState — 流式解析状态机
// ===========================================================================

/// 跟踪流式响应中正在构建的 tool call。
#[derive(Debug)]
struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
    content_index: usize,
    started: bool,
}

/// 流式解析状态。
#[derive(Debug, Default)]
pub(crate) struct StreamState {
    /// 正在构建的 tool calls，key = OpenAI delta 中的 index
    tool_calls: HashMap<u32, ToolCallState>,
    /// 下一个可用的 content_index
    next_content_index: usize,
    /// 是否已发出 TextStart
    text_started: bool,
    /// 当前文本 content_index
    text_content_index: usize,
    /// 最终 finish reason
    finish_reason: Option<FinishReason>,
}

impl StreamState {
    pub fn new() -> Self {
        Self::default()
    }

    /// 处理一个 SSE chunk，产出零或多个 StreamEvent。
    pub fn process_chunk(&mut self, chunk: &ChatCompletionChunk) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        for choice in &chunk.choices {
            let delta = &choice.delta;

            // --- Text content delta ---
            if let Some(text) = &delta.content {
                if !text.is_empty() {
                    if !self.text_started {
                        self.text_content_index = self.next_content_index;
                        self.next_content_index += 1;
                        self.text_started = true;
                        events.push(StreamEvent::TextStart {
                            content_index: self.text_content_index,
                        });
                    }
                    events.push(StreamEvent::TextDelta {
                        content_index: self.text_content_index,
                        delta: text.clone(),
                    });
                }
            }

            // --- Tool call deltas ---
            if let Some(tool_calls) = &delta.tool_calls {
                for tc in tool_calls {
                    let state = self.tool_calls.entry(tc.index).or_insert_with(|| {
                        let ci = self.next_content_index;
                        self.next_content_index += 1;
                        ToolCallState {
                            id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            content_index: ci,
                            started: false,
                        }
                    });

                    if let Some(id) = &tc.id {
                        state.id = id.clone();
                    }
                    if let Some(func) = &tc.function {
                        if let Some(name) = &func.name {
                            state.name.push_str(name);
                        }
                        if let Some(args) = &func.arguments {
                            state.arguments.push_str(args);

                            // 首先，确保文本流已关闭
                            if self.text_started && !state.started {
                                events.push(StreamEvent::TextEnd {
                                    content_index: self.text_content_index,
                                });
                                self.text_started = false;
                            }

                            // 发出 ToolCallStart（仅首次）
                            if !state.started && !state.id.is_empty() && !state.name.is_empty() {
                                state.started = true;
                                events.push(StreamEvent::ToolCallStart {
                                    content_index: state.content_index,
                                    id: ToolCallId::new(&state.id),
                                    name: state.name.clone(),
                                });
                            }

                            // 发出参数增量
                            if state.started && !args.is_empty() {
                                events.push(StreamEvent::ToolCallDelta {
                                    content_index: state.content_index,
                                    delta: args.clone(),
                                });
                            }
                        }
                    }
                }
            }

            // --- Finish reason ---
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(convert::convert_finish_reason(reason));
            }
        }

        events
    }

    /// 流结束时产出最终事件。
    pub fn finish(&mut self, final_usage: Option<&CompletionUsage>) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        // 关闭文本流
        if self.text_started {
            events.push(StreamEvent::TextEnd {
                content_index: self.text_content_index,
            });
            self.text_started = false;
        }

        // 关闭所有 tool call 流
        let mut indices: Vec<u32> = self.tool_calls.keys().copied().collect();
        indices.sort();
        for idx in indices {
            if let Some(state) = self.tool_calls.get(&idx) {
                if state.started {
                    events.push(StreamEvent::ToolCallEnd {
                        content_index: state.content_index,
                    });
                }
            }
        }

        // Finish 事件
        let usage = final_usage.map(convert::convert_usage);
        let finish_reason = self.finish_reason.unwrap_or(FinishReason::Stop);

        events.push(StreamEvent::Finish {
            finish_reason,
            usage,
        });

        events
    }
}

// ===========================================================================
// create_event_stream — 从 SSE 事件流构建 StreamEvent 流
// ===========================================================================

/// 从 SSE 事件流构建 `StreamEvent` 异步流。
///
/// 接收 `katu_provider_http::SseStream`（已完成 SSE 协议解析），
/// 通过 `StreamState` 将 OpenAI chunk 转换为 katu StreamEvent。
pub fn create_event_stream(
    sse_stream: SseStream,
) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>> {
    use tokio_stream::StreamExt;

    Box::pin(async_stream::try_stream! {
        let mut state = StreamState::new();
        let mut final_usage: Option<CompletionUsage> = None;

        tokio::pin!(sse_stream);

        while let Some(event_result) = sse_stream.next().await {
            let event = event_result?;

            // [DONE] 标记流结束
            if event.data == "[DONE]" {
                for ev in state.finish(final_usage.as_ref()) {
                    yield ev;
                }
                return;
            }

            // 解析 chunk JSON
            let chunk: ChatCompletionChunk = match serde_json::from_str(&event.data) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(data = %event.data, error = %e, "failed to parse chunk");
                    continue;
                }
            };

            // 保存最终 usage（只取包含实际数据的 chunk）
            if let Some(u) = &chunk.usage {
                if u.total_tokens > 0 {
                    final_usage = Some(u.clone());
                }
            }

            // 处理 chunk → events
            for ev in state.process_chunk(&chunk) {
                yield ev;
            }
        }

        // 流意外结束（没有 [DONE]）
        for ev in state.finish(final_usage.as_ref()) {
            yield ev;
        }
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_text_chunk(text: &str, finish: Option<&str>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chatcmpl-test".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some(text.into()),
                    tool_calls: None,
                },
                finish_reason: finish.map(String::from),
            }],
            created: 0,
            model: "gpt-4o".into(),
            usage: None,
        }
    }

    fn make_tool_call_chunk(
        tc_index: u32,
        id: Option<&str>,
        name: Option<&str>,
        args: Option<&str>,
        finish: Option<&str>,
    ) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chatcmpl-test".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![ChunkToolCall {
                        index: tc_index,
                        id: id.map(String::from),
                        call_type: Some("function".into()),
                        function: Some(ChunkFunctionCall {
                            name: name.map(String::from),
                            arguments: args.map(String::from),
                        }),
                    }]),
                },
                finish_reason: finish.map(String::from),
            }],
            created: 0,
            model: "gpt-4o".into(),
            usage: None,
        }
    }

    #[test]
    fn test_text_stream() {
        let mut state = StreamState::new();

        let events = state.process_chunk(&make_text_chunk("Hello", None));
        assert_eq!(events.len(), 2); // TextStart + TextDelta
        assert!(matches!(&events[0], StreamEvent::TextStart { content_index: 0 }));
        assert!(matches!(&events[1], StreamEvent::TextDelta { delta, .. } if delta == "Hello"));

        let events = state.process_chunk(&make_text_chunk(" World", None));
        assert_eq!(events.len(), 1); // TextDelta only
        assert!(matches!(&events[0], StreamEvent::TextDelta { delta, .. } if delta == " World"));

        let events = state.process_chunk(&make_text_chunk("", Some("stop")));
        assert_eq!(events.len(), 0); // empty text, just sets finish_reason

        let events = state.finish(None);
        assert_eq!(events.len(), 2); // TextEnd + Finish
        assert!(matches!(&events[0], StreamEvent::TextEnd { content_index: 0 }));
        assert!(matches!(
            &events[1],
            StreamEvent::Finish { finish_reason: FinishReason::Stop, .. }
        ));
    }

    #[test]
    fn test_tool_call_stream() {
        let mut state = StreamState::new();

        // First chunk: tool call start
        let events =
            state.process_chunk(&make_tool_call_chunk(0, Some("call_1"), Some("bash"), Some("{\""), None));
        // ToolCallStart + ToolCallDelta
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallStart { name, .. } if name == "bash")));
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallDelta { delta, .. } if delta == "{\"")));

        // More arguments
        let events =
            state.process_chunk(&make_tool_call_chunk(0, None, None, Some("cmd\":\"ls\"}"), None));
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], StreamEvent::ToolCallDelta { delta, .. } if delta == "cmd\":\"ls\"}")
        );

        // Finish
        let events = state.process_chunk(&make_tool_call_chunk(0, None, None, None, Some("tool_calls")));

        let fin_events = state.finish(None);
        assert!(fin_events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallEnd { content_index: 0 })));
        assert!(fin_events.iter().any(|e| matches!(
            e,
            StreamEvent::Finish {
                finish_reason: FinishReason::ToolCalls,
                ..
            }
        )));
        let _ = events;
    }

    #[test]
    fn test_usage_forwarding() {
        let mut state = StreamState::new();
        state.finish_reason = Some(FinishReason::Stop);

        let usage = CompletionUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };

        let events = state.finish(Some(&usage));
        if let Some(StreamEvent::Finish { usage: Some(u), .. }) = events.last() {
            assert_eq!(u.input_tokens, 100);
            assert_eq!(u.output_tokens, 50);
        } else {
            panic!("expected Finish with usage");
        }
    }
}
