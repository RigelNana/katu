//! # error
//!
//! ## 职责
//! 定义 `katu-agent` 的统一错误类型 — Agent 执行层的所有可恢复与不可恢复错误。
//!
//! ## 设计
//! `AgentError` 是 **Agent 执行层** 的错误枚举，与 `katu_core::Error` 的关系：
//! - `katu_core::Error` — 基础设施层（Provider/Tool/Config 级别）
//! - `AgentError` — 编排层（实例构建、循环控制、流消费、工具编排）
//!
//! Agent loop 内部捕获 `katu_core::Error` 后，根据语义转换为 `AgentError`
//! 或直接处理（如 Provider RateLimit → 重试，不暴露为 AgentError）。
//!
//! ## 调用者
//! - `katu-agent::instance` — 构建失败
//! - `katu-agent::runner` (future) — 循环执行错误

use crate::stream_consumer::StreamConsumerError;
use crate::tool_executor::ToolExecutorError;

// ===========================================================================
// Result 别名
// ===========================================================================

/// `katu-agent` 模块级 Result 别名。
pub type Result<T, E = AgentError> = std::result::Result<T, E>;

// ===========================================================================
// AgentError
// ===========================================================================

/// Agent 执行层统一错误枚举。
///
/// # 错误分类
///
/// | 变体 | 严重度 | 来源 |
/// |------|--------|------|
/// | `Build` | 致命 | `InstanceBuilder::build()` |
/// | `Stream` | 可恢复 | `StreamConsumer::consume()` |
/// | `ToolExecution` | 可恢复 | `ToolExecutor::execute_batch()` |
/// | `Provider` | 视情况 | LLM Provider 调用 |
/// | `ContextOverflow` | 可恢复 | token 预算超限 |
/// | `MaxSteps` | 正常终止 | 步数上限 |
/// | `Cancelled` | 正常终止 | 用户/系统中断 |
/// | `Internal` | 致命 | 未预料的内部错误 |
///
/// # Examples
///
/// ```
/// use katu_agent::error::AgentError;
///
/// let err = AgentError::build("missing model reference");
/// assert!(err.to_string().contains("missing model reference"));
///
/// let err = AgentError::max_steps(50);
/// assert!(err.to_string().contains("50"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// 实例构建失败 — 缺少必要配置或参数不合法。
    #[error("agent build error: {message}")]
    Build {
        message: String,
    },

    /// 流消费错误 — StreamConsumer 层报告。
    #[error("stream error: {0}")]
    Stream(#[from] StreamConsumerError),

    /// 工具执行错误 — ToolExecutor 层报告。
    #[error("tool execution error: {0}")]
    ToolExecution(#[from] ToolExecutorError),

    /// LLM Provider 错误 — 从 `katu_core::Error` 传播。
    ///
    /// Agent loop 内部已处理可重试错误（RateLimit 等），
    /// 到达此处的是不可恢复的 Provider 错误。
    #[error("provider error: {message}")]
    Provider {
        message: String,
        /// 是否可重试（供上层决策）。
        retryable: bool,
    },

    /// 上下文窗口溢出 — 消息历史超过模型 token 上限。
    ///
    /// Runner 收到此错误后应触发压缩流程。
    #[error("context overflow: {used}/{limit} tokens")]
    ContextOverflow {
        used: u64,
        limit: u64,
    },

    /// 达到最大步数限制 — 正常终止条件。
    #[error("max steps reached: {steps}")]
    MaxSteps {
        steps: u32,
    },

    /// 用户或系统取消。
    #[error("cancelled")]
    Cancelled,

    /// 不可归类的内部错误。
    #[error("{0}")]
    Internal(Box<dyn std::error::Error + Send + Sync>),
}

// ---------------------------------------------------------------------------
// 构造辅助方法
// ---------------------------------------------------------------------------

impl AgentError {
    /// 构造构建错误。
    pub fn build(message: impl Into<String>) -> Self {
        Self::Build {
            message: message.into(),
        }
    }

    /// 构造 Provider 错误。
    pub fn provider(message: impl Into<String>, retryable: bool) -> Self {
        Self::Provider {
            message: message.into(),
            retryable,
        }
    }

    /// 构造上下文溢出错误。
    pub fn context_overflow(used: u64, limit: u64) -> Self {
        Self::ContextOverflow { used, limit }
    }

    /// 构造最大步数错误。
    pub fn max_steps(steps: u32) -> Self {
        Self::MaxSteps { steps }
    }

    /// 从任意 error 构造内部错误。
    pub fn internal(err: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        Self::Internal(err.into())
    }
}

// ---------------------------------------------------------------------------
// 分类查询
// ---------------------------------------------------------------------------

impl AgentError {
    /// 是否为致命错误（不可恢复，应终止 Agent loop）。
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::Build { .. } | Self::Internal(_))
    }

    /// 是否为正常终止条件（不算真正的错误）。
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::MaxSteps { .. } | Self::Cancelled)
    }

    /// 是否可重试。
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Provider { retryable, .. } => *retryable,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// From katu_core::Error
// ---------------------------------------------------------------------------

impl From<katu_core::Error> for AgentError {
    fn from(err: katu_core::Error) -> Self {
        match err {
            katu_core::Error::Cancelled => Self::Cancelled,
            katu_core::Error::ContextOverflow { used, limit } => {
                Self::ContextOverflow {
                    used: used as u64,
                    limit: limit as u64,
                }
            }
            katu_core::Error::Provider { ref kind, .. } => {
                Self::Provider {
                    message: err.to_string(),
                    retryable: kind.retryable(),
                }
            }
            katu_core::Error::Config { message } => {
                Self::Build { message }
            }
            other => Self::internal(other),
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
    fn test_build_error() {
        let err = AgentError::build("model not found");
        assert!(err.is_fatal());
        assert!(!err.is_terminal());
        assert!(err.to_string().contains("model not found"));
    }

    #[test]
    fn test_max_steps_is_terminal() {
        let err = AgentError::max_steps(50);
        assert!(err.is_terminal());
        assert!(!err.is_fatal());
    }

    #[test]
    fn test_cancelled_is_terminal() {
        let err = AgentError::Cancelled;
        assert!(err.is_terminal());
    }

    #[test]
    fn test_provider_retryable() {
        let err = AgentError::provider("rate limit", true);
        assert!(err.is_retryable());
        assert!(!err.is_fatal());
    }

    #[test]
    fn test_provider_not_retryable() {
        let err = AgentError::provider("auth failed", false);
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_context_overflow() {
        let err = AgentError::context_overflow(200_000, 128_000);
        assert!(!err.is_fatal());
        assert!(!err.is_terminal());
        assert!(err.to_string().contains("200000/128000"));
    }

    #[test]
    fn test_internal_error() {
        let err = AgentError::internal("something unexpected");
        assert!(err.is_fatal());
    }

    #[test]
    fn test_from_core_cancelled() {
        let core_err = katu_core::Error::Cancelled;
        let agent_err: AgentError = core_err.into();
        assert!(matches!(agent_err, AgentError::Cancelled));
    }

    #[test]
    fn test_from_core_context_overflow() {
        let core_err = katu_core::Error::ContextOverflow {
            used: 150_000,
            limit: 128_000,
        };
        let agent_err: AgentError = core_err.into();
        match agent_err {
            AgentError::ContextOverflow { used, limit } => {
                assert_eq!(used, 150_000);
                assert_eq!(limit, 128_000);
            }
            _ => panic!("expected ContextOverflow"),
        }
    }

    #[test]
    fn test_from_core_config() {
        let core_err = katu_core::Error::Config {
            message: "bad config".into(),
        };
        let agent_err: AgentError = core_err.into();
        assert!(matches!(agent_err, AgentError::Build { .. }));
    }
}
