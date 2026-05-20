//! # stream_consumer
//!
//! ## 职责
//! 将 Provider 产出的 `StreamEvent` 流消费为完整的 `AssistantMessage`，
//! 同时实时通过 channel 发射 `AgentEvent` 供 UI/日志层订阅。
//!
//! ## 设计
//! - **关联函数入口** — `StreamConsumer::consume` 不持有跨调用状态
//! - **状态机驱动** — 内部 `StreamState` 管理 text/reasoning/tool_call 累积，事件处理为其方法
//! - **实时发射** — 每个 delta 立即通过 channel 发出，不缓存
//! - **异常鲁棒** — 取消或流中断时 flush 未关闭 buffers，不丢数据
//! - **保序** — `IndexMap` 保证 tool call 按 LLM 发出顺序排列
//!
//! ## 调用者
//! - `katu-agent::runner` (future) — Agent loop 核心循环

use chrono::Utc;
use indexmap::IndexMap;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tracing::{debug, warn};

use katu_core::agent_event::AgentEvent;
use katu_core::event::StreamEvent;
use katu_core::message::{AssistantBlock, AssistantMessage};
use katu_core::types::{FinishReason, MessageId, ToolCallId};
use katu_core::usage::Usage;
use katu_core::CancellationToken;

// ===========================================================================
// Public Types
// ===========================================================================

/// 流消费的最终结果 — 包含完整 AssistantMessage + 元数据。
#[derive(Debug, Clone)]
pub struct StreamResult {
    /// 累积出的完整 assistant 消息
    pub message: AssistantMessage,
    /// LLM 停止原因
    pub finish_reason: FinishReason,
    /// Token 用量
    pub usage: Usage,
}

/// 流消费错误。
#[derive(Debug, thiserror::Error)]
pub enum StreamConsumerError {
    /// 用户或系统取消了操作
    #[error("stream cancelled")]
    Cancelled,

    /// Provider 报告了不可恢复的错误
    #[error("provider error: {message}")]
    Provider { message: String, retryable: bool },

    /// 流意外结束（无 Finish 事件）
    #[error("stream ended unexpectedly without a finish event")]
    UnexpectedEnd,

    /// 事件发送失败（接收端已关闭）
    #[error("event channel closed")]
    ChannelClosed,
}

/// 模块级 Result 别名。
pub type Result<T, E = StreamConsumerError> = std::result::Result<T, E>;

// ===========================================================================
// EventStream type alias
// ===========================================================================

/// Provider 产出的流式事件流。
///
/// 这是一个 pinned, boxed async stream of `Result<StreamEvent, katu_core::Error>`。
pub type EventStream =
    std::pin::Pin<Box<dyn futures_core::Stream<Item = Result<StreamEvent, katu_core::Error>> + Send>>;

// ===========================================================================
// StreamConsumer
// ===========================================================================

/// 流消费器 — 将 `EventStream` 转为 `AssistantMessage` + 实时 `AgentEvent`。
///
/// 无状态设计：所有状态存在于一次 `consume` 调用的局部 `StreamState` 中。
pub struct StreamConsumer;

impl StreamConsumer {
    /// 消费完整的 EventStream。
    ///
    /// - 逐事件处理，实时通过 `event_tx` 发射 `AgentEvent`
    /// - 支持 `CancellationToken` 提前终止
    /// - 返回累积出的完整结果
    ///
    /// # Arguments
    /// - `stream` — Provider 产出的 StreamEvent 流
    /// - `event_tx` — AgentEvent 发射通道（unbounded，不反压）
    /// - `cancel` — 协作式取消令牌
    /// - `model` — 当前使用的模型标识（写入 AssistantMessage）
    /// - `provider` — 当前使用的 provider 标识
    ///
    /// # Errors
    /// - `Cancelled` — 取消令牌被触发
    /// - `Provider` — 流中收到 ProviderError 事件
    /// - `UnexpectedEnd` — 流在未收到 Finish 事件时结束
    pub async fn consume(
        stream: EventStream,
        event_tx: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
        model: String,
        provider: String,
    ) -> Result<StreamResult> {
        let mut state = StreamState::new(model, provider);

        tokio::pin!(stream);
        loop {
            if cancel.is_cancelled() {
                debug!("stream_consumer: cancelled before next event");
                return state.into_cancelled_result();
            }

            // tokio::select! 允许在等待下一事件时响应取消
            let item = tokio::select! {
                biased;
                _ = Self::wait_for_cancel(cancel) => {
                    debug!("stream_consumer: cancelled during await");
                    return state.into_cancelled_result();
                }
                item = stream.next() => item,
            };

            match item {
                Some(Ok(event)) => {
                    let terminal = state.handle_event(event, event_tx)?;
                    if terminal {
                        break;
                    }
                }
                Some(Err(e)) => {
                    warn!("stream_consumer: provider stream error: {e}");
                    return Err(StreamConsumerError::Provider {
                        message: e.to_string(),
                        retryable: e.retryable(),
                    });
                }
                None => {
                    // Stream ended without Finish event
                    warn!("stream_consumer: stream ended without finish event");
                    return state.into_result_or_unexpected_end();
                }
            }
        }

        state
            .into_result()
            .ok_or(StreamConsumerError::UnexpectedEnd)
    }

    /// 等待取消令牌触发的 future。
    ///
    /// 通过短间隔轮询实现（CancellationToken 基于 AtomicBool，无 async notify）。
    async fn wait_for_cancel(token: &CancellationToken) {
        loop {
            if token.is_cancelled() {
                return;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }
}

// ===========================================================================
// Internal State
// ===========================================================================

/// 单个 tool call 的累积器。
struct ToolCallAccumulator {
    name: String,
    arguments_buf: String,
    content_index: usize,
}

/// 流消费过程中累积的内部状态。
struct StreamState {
    /// 当前累积的 content blocks（按产出顺序）
    blocks: Vec<AssistantBlock>,
    /// 正在累积的文本 buffer
    text_buf: Option<TextAccumulator>,
    /// 正在累积的 reasoning buffer
    reasoning_buf: Option<ReasoningAccumulator>,
    /// 正在累积的 tool call buffers (ToolCallId → accumulator)
    tool_call_bufs: IndexMap<ToolCallId, ToolCallAccumulator>,
    /// finish reason (由 Finish/StepFinish 设置)
    finish_reason: Option<FinishReason>,
    /// usage (由 Finish/StepFinish 设置)
    usage: Option<Usage>,
    /// 模型标识
    model: String,
    /// Provider 标识
    provider: String,
}

#[allow(dead_code)]
struct TextAccumulator {
    content_index: usize,
    text: String,
}

#[allow(dead_code)]
struct ReasoningAccumulator {
    content_index: usize,
    text: String,
}

impl StreamState {
    fn new(model: String, provider: String) -> Self {
        Self {
            blocks: Vec::new(),
            text_buf: None,
            reasoning_buf: None,
            tool_call_bufs: IndexMap::new(),
            finish_reason: None,
            usage: None,
            model,
            provider,
        }
    }

    /// 强制 flush 所有未关闭的 buffers（用于取消/异常路径）。
    fn flush_open_buffers(&mut self) {
        if let Some(acc) = self.text_buf.take() {
            if !acc.text.is_empty() {
                self.blocks.push(AssistantBlock::Text { text: acc.text });
            }
        }
        if let Some(acc) = self.reasoning_buf.take() {
            if !acc.text.is_empty() {
                self.blocks.push(AssistantBlock::Reasoning {
                    text: acc.text,
                    signature: None,
                });
            }
        }
        for (id, acc) in self.tool_call_bufs.drain(..) {
            let arguments = serde_json::from_str(&acc.arguments_buf)
                .unwrap_or(serde_json::Value::Null);
            self.blocks.push(AssistantBlock::ToolCall {
                id,
                name: acc.name,
                arguments,
            });
        }
    }

    /// 正常完成时构建 StreamResult。
    fn into_result(self) -> Option<StreamResult> {
        let finish_reason = self.finish_reason?;
        let usage = self.usage.unwrap_or_default();
        let message = AssistantMessage {
            id: MessageId::new(),
            content: self.blocks,
            model: self.model,
            provider: self.provider,
            finish_reason,
            usage: Some(usage.clone()),
            timestamp: Utc::now(),
        };
        Some(StreamResult {
            message,
            finish_reason,
            usage,
        })
    }

    fn into_cancelled_result(mut self) -> Result<StreamResult> {
        self.flush_open_buffers();
        Err(StreamConsumerError::Cancelled)
    }

    /// 流意外结束时尝试构建结果（可能有部分数据）。
    fn into_result_or_unexpected_end(mut self) -> Result<StreamResult> {
        self.flush_open_buffers();
        // 即使没有 finish event，如果有数据就尝试返回
        let finish_reason = self.finish_reason.unwrap_or(FinishReason::Unknown);
        let usage = self.usage.unwrap_or_default();
        if self.blocks.is_empty() {
            return Err(StreamConsumerError::UnexpectedEnd);
        }
        let message = AssistantMessage {
            id: MessageId::new(),
            content: self.blocks,
            model: self.model,
            provider: self.provider,
            finish_reason,
            usage: Some(usage.clone()),
            timestamp: Utc::now(),
        };
        Ok(StreamResult {
            message,
            finish_reason,
            usage,
        })
    }

    // ─── Event Processing ────────────────────────────────────────────────────

    /// 处理单个 StreamEvent，更新状态并发射 AgentEvent。
    ///
    /// 返回 `true` 表示流已终止（收到 Finish 事件）。
    fn handle_event(
        &mut self,
        event: StreamEvent,
        tx: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<bool> {
        match event {
            // ─── Step 生命周期 ───
            StreamEvent::StepStart { .. } => {
                // Step 开始，目前不需要额外处理
            }
            StreamEvent::StepFinish {
                finish_reason,
                usage,
                ..
            } => {
                // 每个 step 完成时记录 finish_reason 和 usage
                // 对于多 step 场景，后续 step 会覆盖
                self.finish_reason = Some(finish_reason);
                if let Some(u) = usage {
                    self.usage = Some(u);
                }
            }

            // ─── Text 流 ───
            StreamEvent::TextStart { content_index } => {
                self.text_buf = Some(TextAccumulator {
                    content_index,
                    text: String::new(),
                });
                Self::send(tx, AgentEvent::TextStarted { content_index })?;
            }
            StreamEvent::TextDelta {
                content_index,
                delta,
            } => {
                if let Some(acc) = &mut self.text_buf {
                    acc.text.push_str(&delta);
                }
                Self::send(
                    tx,
                    AgentEvent::TextDelta {
                        content_index,
                        delta,
                    },
                )?;
            }
            StreamEvent::TextEnd { content_index } => {
                let text = if let Some(acc) = self.text_buf.take() {
                    let t = acc.text;
                    if !t.is_empty() {
                        self.blocks.push(AssistantBlock::Text { text: t.clone() });
                    }
                    t
                } else {
                    String::new()
                };
                Self::send(tx, AgentEvent::TextEnded { content_index, text })?;
            }

            // ─── Reasoning 流 ───
            StreamEvent::ReasoningStart { content_index } => {
                self.reasoning_buf = Some(ReasoningAccumulator {
                    content_index,
                    text: String::new(),
                });
                Self::send(tx, AgentEvent::ReasoningStarted { content_index })?;
            }
            StreamEvent::ReasoningDelta {
                content_index,
                delta,
            } => {
                if let Some(acc) = &mut self.reasoning_buf {
                    acc.text.push_str(&delta);
                }
                Self::send(
                    tx,
                    AgentEvent::ReasoningDelta {
                        content_index,
                        delta,
                    },
                )?;
            }
            StreamEvent::ReasoningEnd { content_index } => {
                let text = if let Some(acc) = self.reasoning_buf.take() {
                    let t = acc.text;
                    if !t.is_empty() {
                        self.blocks.push(AssistantBlock::Reasoning {
                            text: t.clone(),
                            signature: None,
                        });
                    }
                    t
                } else {
                    String::new()
                };
                Self::send(
                    tx,
                    AgentEvent::ReasoningEnded { content_index, text },
                )?;
            }
            StreamEvent::ToolCallStart {
                content_index,
                id,
                name,
            } => {
                self.tool_call_bufs.insert(
                    id.clone(),
                    ToolCallAccumulator {
                        name: name.clone(),
                        arguments_buf: String::new(),
                        content_index,
                    },
                );
                Self::send(
                    tx,
                    AgentEvent::ToolInputStarted {
                        call_id: id,
                        tool_name: name,
                    },
                )?;
            }
            StreamEvent::ToolCallDelta {
                content_index,
                delta,
            } => {
                // Delta 只携带 content_index，需要找到对应的 tool call
                // 通常同一时间只有一个 tool call 在接收 delta
                if let Some((call_id, acc)) = self
                    .tool_call_bufs
                    .iter_mut()
                    .find(|(_, acc)| acc.content_index == content_index)
                {
                    acc.arguments_buf.push_str(&delta);
                    Self::send(
                        tx,
                        AgentEvent::ToolInputDelta {
                            call_id: call_id.clone(),
                            delta,
                        },
                    )?;
                }
            }
            StreamEvent::ToolCallEnd { content_index } => {
                // 找到匹配 content_index 的 tool call 并 finalize
                let entry = self
                    .tool_call_bufs
                    .iter()
                    .find(|(_, acc)| acc.content_index == content_index)
                    .map(|(id, _)| id.clone());

                if let Some(call_id) = entry {
                    if let Some(acc) = self.tool_call_bufs.shift_remove(&call_id) {
                        let arguments: serde_json::Value =
                            serde_json::from_str(&acc.arguments_buf)
                                .unwrap_or(serde_json::Value::Null);

                        self.blocks.push(AssistantBlock::ToolCall {
                            id: call_id.clone(),
                            name: acc.name,
                            arguments: arguments.clone(),
                        });

                        Self::send(
                            tx,
                            AgentEvent::ToolInputEnded {
                                call_id,
                                arguments,
                            },
                        )?;
                    }
                }
            }

            // ─── 终态 ───
            StreamEvent::Finish {
                finish_reason,
                usage,
            } => {
                self.finish_reason = Some(finish_reason);
                if let Some(u) = usage {
                    self.usage = Some(u);
                }
                return Ok(true);
            }
            StreamEvent::ProviderError {
                message, retryable, ..
            } => {
                return Err(StreamConsumerError::Provider { message, retryable });
            }

            // ─── Agent loop 注入的事件（stream 阶段不应出现）───
            StreamEvent::ToolResult { .. } | StreamEvent::ToolError { .. } => {
                // 忽略 — 这些事件由 agent loop 注入到事件流中
                // stream_consumer 不处理
            }
        }
        Ok(false)
    }

    /// 通过 channel 发送 AgentEvent。
    fn send(
        tx: &mpsc::UnboundedSender<AgentEvent>,
        event: AgentEvent,
    ) -> Result<()> {
        tx.send(event).map_err(|_| StreamConsumerError::ChannelClosed)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use async_stream::stream;
    use katu_core::types::ToolCallId;

    /// 辅助：构造一个简单的文本流
    fn text_stream(text: &str) -> EventStream {
        let chunks: Vec<String> = text
            .chars()
            .collect::<Vec<_>>()
            .chunks(5)
            .map(|c| c.iter().collect())
            .collect();
        let s = stream! {
            yield Ok(StreamEvent::StepStart { index: 0 });
            yield Ok(StreamEvent::TextStart { content_index: 0 });
            for chunk in chunks {
                yield Ok(StreamEvent::TextDelta { content_index: 0, delta: chunk });
            }
            yield Ok(StreamEvent::TextEnd { content_index: 0 });
            yield Ok(StreamEvent::StepFinish {
                index: 0,
                finish_reason: FinishReason::Stop,
                usage: Some(Usage { input_tokens: 10, output_tokens: 5, total_tokens: 15, ..Default::default() }),
            });
            yield Ok(StreamEvent::Finish {
                finish_reason: FinishReason::Stop,
                usage: Some(Usage { input_tokens: 10, output_tokens: 5, total_tokens: 15, ..Default::default() }),
            });
        };
        Box::pin(s)
    }

    #[tokio::test]
    async fn test_simple_text_stream() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let result = StreamConsumer::consume(
            text_stream("Hello, world!"),
            &tx,
            &cancel,
            "gpt-4o".into(),
            "openai".into(),
        )
        .await
        .unwrap();

        assert_eq!(result.finish_reason, FinishReason::Stop);
        assert_eq!(result.message.text(), "Hello, world!");
        assert_eq!(result.message.model, "gpt-4o");
        assert_eq!(result.message.provider, "openai");
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);

        // 验证收到了 AgentEvent
        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        // 应该有 TextStarted, TextDelta*, TextEnded
        assert!(events.iter().any(|e| matches!(e, AgentEvent::TextStarted { .. })));
        assert!(events.iter().any(|e| matches!(e, AgentEvent::TextEnded { .. })));
        assert!(events.iter().any(|e| matches!(e, AgentEvent::TextDelta { .. })));
    }

    #[tokio::test]
    async fn test_tool_call_stream() {
        let s = stream! {
            yield Ok(StreamEvent::StepStart { index: 0 });
            yield Ok(StreamEvent::TextStart { content_index: 0 });
            yield Ok(StreamEvent::TextDelta { content_index: 0, delta: "Let me read that file.".into() });
            yield Ok(StreamEvent::TextEnd { content_index: 0 });
            yield Ok(StreamEvent::ToolCallStart {
                content_index: 1,
                id: ToolCallId::new("call_123"),
                name: "read_file".into(),
            });
            yield Ok(StreamEvent::ToolCallDelta { content_index: 1, delta: r#"{"path":"#.into() });
            yield Ok(StreamEvent::ToolCallDelta { content_index: 1, delta: r#""src/main.rs"}"#.into() });
            yield Ok(StreamEvent::ToolCallEnd { content_index: 1 });
            yield Ok(StreamEvent::Finish {
                finish_reason: FinishReason::ToolCalls,
                usage: Some(Usage { input_tokens: 50, output_tokens: 20, total_tokens: 70, ..Default::default() }),
            });
        };

        let (tx, mut rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let result = StreamConsumer::consume(
            Box::pin(s),
            &tx,
            &cancel,
            "claude-sonnet-4-20250514".into(),
            "anthropic".into(),
        )
        .await
        .unwrap();

        assert_eq!(result.finish_reason, FinishReason::ToolCalls);
        assert_eq!(result.message.content.len(), 2);

        // 第一个 block 是文本
        assert!(matches!(&result.message.content[0], AssistantBlock::Text { text } if text == "Let me read that file."));

        // 第二个 block 是 tool call
        match &result.message.content[1] {
            AssistantBlock::ToolCall { id, name, arguments } => {
                assert_eq!(id.as_str(), "call_123");
                assert_eq!(name, "read_file");
                assert_eq!(arguments["path"], "src/main.rs");
            }
            _ => panic!("expected ToolCall block"),
        }

        // 验证 AgentEvent
        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolInputStarted { tool_name, .. } if tool_name == "read_file")));
        assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolInputEnded { .. })));
    }

    #[tokio::test]
    async fn test_reasoning_stream() {
        let s = stream! {
            yield Ok(StreamEvent::StepStart { index: 0 });
            yield Ok(StreamEvent::ReasoningStart { content_index: 0 });
            yield Ok(StreamEvent::ReasoningDelta { content_index: 0, delta: "I need to think".into() });
            yield Ok(StreamEvent::ReasoningDelta { content_index: 0, delta: " about this carefully.".into() });
            yield Ok(StreamEvent::ReasoningEnd { content_index: 0 });
            yield Ok(StreamEvent::TextStart { content_index: 1 });
            yield Ok(StreamEvent::TextDelta { content_index: 1, delta: "Here is my answer.".into() });
            yield Ok(StreamEvent::TextEnd { content_index: 1 });
            yield Ok(StreamEvent::Finish {
                finish_reason: FinishReason::Stop,
                usage: None,
            });
        };

        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let result = StreamConsumer::consume(
            Box::pin(s),
            &tx,
            &cancel,
            "o1".into(),
            "openai".into(),
        )
        .await
        .unwrap();

        assert_eq!(result.message.content.len(), 2);
        assert_eq!(result.message.reasoning(), "I need to think about this carefully.");
        assert_eq!(result.message.text(), "Here is my answer.");
    }

    #[tokio::test]
    async fn test_cancellation() {
        let s = stream! {
            yield Ok(StreamEvent::StepStart { index: 0 });
            yield Ok(StreamEvent::TextStart { content_index: 0 });
            yield Ok(StreamEvent::TextDelta { content_index: 0, delta: "Hello".into() });
            // 无限等待 — 模拟长时间流
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            yield Ok(StreamEvent::TextEnd { content_index: 0 });
        };

        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });

        let result = StreamConsumer::consume(
            Box::pin(s),
            &tx,
            &cancel,
            "gpt-4o".into(),
            "openai".into(),
        )
        .await;

        assert!(matches!(result, Err(StreamConsumerError::Cancelled)));
    }

    #[tokio::test]
    async fn test_provider_error() {
        let s = stream! {
            yield Ok(StreamEvent::StepStart { index: 0 });
            yield Ok(StreamEvent::ProviderError {
                message: "rate limit exceeded".into(),
                retryable: true,
            });
        };

        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let result = StreamConsumer::consume(
            Box::pin(s),
            &tx,
            &cancel,
            "gpt-4o".into(),
            "openai".into(),
        )
        .await;

        match result {
            Err(StreamConsumerError::Provider { message, retryable }) => {
                assert_eq!(message, "rate limit exceeded");
                assert!(retryable);
            }
            _ => panic!("expected Provider error"),
        }
    }

    #[tokio::test]
    async fn test_unexpected_end_with_data() {
        // 流有数据但没有 Finish 事件就结束了
        let s = stream! {
            yield Ok(StreamEvent::StepStart { index: 0 });
            yield Ok(StreamEvent::TextStart { content_index: 0 });
            yield Ok(StreamEvent::TextDelta { content_index: 0, delta: "partial".into() });
            // 没有 TextEnd 或 Finish，直接结束
        };

        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let result = StreamConsumer::consume(
            Box::pin(s),
            &tx,
            &cancel,
            "gpt-4o".into(),
            "openai".into(),
        )
        .await
        .unwrap();

        // 有数据就返回部分结果
        assert_eq!(result.message.text(), "partial");
        assert_eq!(result.finish_reason, FinishReason::Unknown);
    }

    #[tokio::test]
    async fn test_unexpected_end_no_data() {
        // 流完全没有数据就结束
        let s = stream! {
            yield Ok(StreamEvent::StepStart { index: 0 });
            // 什么都没产出
        };

        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let result = StreamConsumer::consume(
            Box::pin(s),
            &tx,
            &cancel,
            "gpt-4o".into(),
            "openai".into(),
        )
        .await;

        assert!(matches!(result, Err(StreamConsumerError::UnexpectedEnd)));
    }

    #[tokio::test]
    async fn test_multiple_tool_calls() {
        let s = stream! {
            yield Ok(StreamEvent::StepStart { index: 0 });
            yield Ok(StreamEvent::ToolCallStart {
                content_index: 0,
                id: ToolCallId::new("call_1"),
                name: "read_file".into(),
            });
            yield Ok(StreamEvent::ToolCallDelta { content_index: 0, delta: r#"{"path":"a.rs"}"#.into() });
            yield Ok(StreamEvent::ToolCallEnd { content_index: 0 });
            yield Ok(StreamEvent::ToolCallStart {
                content_index: 1,
                id: ToolCallId::new("call_2"),
                name: "read_file".into(),
            });
            yield Ok(StreamEvent::ToolCallDelta { content_index: 1, delta: r#"{"path":"b.rs"}"#.into() });
            yield Ok(StreamEvent::ToolCallEnd { content_index: 1 });
            yield Ok(StreamEvent::Finish {
                finish_reason: FinishReason::ToolCalls,
                usage: None,
            });
        };

        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let result = StreamConsumer::consume(
            Box::pin(s),
            &tx,
            &cancel,
            "gpt-4o".into(),
            "openai".into(),
        )
        .await
        .unwrap();

        assert_eq!(result.message.content.len(), 2);
        assert!(result.message.has_tool_calls());

        // 验证顺序保持
        match &result.message.content[0] {
            AssistantBlock::ToolCall { id, arguments, .. } => {
                assert_eq!(id.as_str(), "call_1");
                assert_eq!(arguments["path"], "a.rs");
            }
            _ => panic!("expected ToolCall"),
        }
        match &result.message.content[1] {
            AssistantBlock::ToolCall { id, arguments, .. } => {
                assert_eq!(id.as_str(), "call_2");
                assert_eq!(arguments["path"], "b.rs");
            }
            _ => panic!("expected ToolCall"),
        }
    }
}
