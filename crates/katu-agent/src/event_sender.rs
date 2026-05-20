//! # event_sender
//!
//! ## 职责
//! 为 `mpsc::UnboundedSender<AgentEvent>` 提供扩展 trait，
//! 统一 channel 发送的错误处理。

use tokio::sync::mpsc;

use katu_core::agent_event::AgentEvent;

// ===========================================================================
// Error
// ===========================================================================

/// Channel 发送失败 — 接收端已关闭。
#[derive(Debug, thiserror::Error)]
#[error("event channel closed")]
pub struct ChannelClosedError;

// ===========================================================================
// Extension Trait
// ===========================================================================

/// `UnboundedSender<AgentEvent>` 扩展 — 提供统一的发送方法。
pub(crate) trait AgentEventSenderExt {
    /// 发送事件，接收端关闭时返回 `ChannelClosedError`。
    fn emit(&self, event: AgentEvent) -> Result<(), ChannelClosedError>;

    /// 发送事件，忽略失败（用于 spawn 内部无法传播错误的场景）。
    fn emit_lossy(&self, event: AgentEvent) {
        let _ = self.emit(event);
    }
}

impl AgentEventSenderExt for mpsc::UnboundedSender<AgentEvent> {
    fn emit(&self, event: AgentEvent) -> Result<(), ChannelClosedError> {
        self.send(event).map_err(|_| ChannelClosedError)
    }
}
