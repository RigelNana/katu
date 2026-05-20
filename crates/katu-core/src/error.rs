//! # katu_core::error
//!
//! ## 职责
//! 定义全局错误类型。所有 crate 的错误最终汇聚到 `Error` 枚举。
//!
//! ## 依赖
//! - `katu_core::types` — ToolCallId
//!
//! ## 对外接口
//! - `Error` — 顶层错误枚举
//! - `Result<T>` — 类型别名
//! - `ProviderErrorKind` — Provider 错误分类（retryable 判定依据）
//! - `AuthErrorKind` — 认证错误细分
//!
//! ## 调用者
//! - `katu_core::message` — Message 构造可能失败
//! - `katu_core::event` — AgentEvent 携带错误信息
//! - `katu-provider` — 将 HTTP 错误映射为 ProviderErrorKind
//! - `katu-tools` — 工具执行失败时构造 Error::Tool
//! - `katu-agent` — Agent loop match Error 决定下一步行为

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::ToolCallId;

// ---------------------------------------------------------------------------
// Result 别名
// ---------------------------------------------------------------------------

/// Katu 全局 Result 别名。
///
/// # Examples
/// ```
/// use katu_core::error::Result;
///
/// fn do_something() -> Result<()> {
///     Ok(())
/// }
/// ```
pub type Result<T, E = Error> = std::result::Result<T, E>;

// ---------------------------------------------------------------------------
// Error — 顶层错误枚举
// ---------------------------------------------------------------------------

/// 顶层错误枚举。
///
/// Agent loop 对此做 match 决定行为：
/// - `Provider { kind: RateLimit { .. }, .. }` → 退避重试
/// - `Provider { kind: Authentication { .. }, .. }` → 终止并提示用户
/// - `Tool { .. }` → 将错误作为 ToolResult 回传给 LLM
/// - `ContextOverflow { .. }` → 触发上下文压缩
///
/// # Examples
/// ```
/// use katu_core::error::{Error, ProviderErrorKind};
/// use katu_core::types::ToolCallId;
///
/// // 构造一个工具错误
/// let err = Error::tool("read_file", ToolCallId::new("call_1"), "file not found");
/// assert!(err.to_string().contains("read_file"));
/// assert!(!err.retryable());
///
/// // 构造一个可重试的 Provider 错误
/// let err = Error::provider(
///     ProviderErrorKind::RateLimit {
///         message: "too many requests".into(),
///         retry_after: None,
///     },
///     "please wait 30 seconds",
/// );
/// assert!(err.retryable());
/// assert!(err.retry_after().is_some());
/// ```
#[derive(Debug, Error)]
pub enum Error {
    /// LLM Provider 返回的错误。
    #[error("provider error: {kind}")]
    Provider {
        kind: ProviderErrorKind,
        /// 可选的恢复建议，面向终端用户。
        suggestion: Option<String>,
    },

    /// 工具执行失败。
    /// 非致命：Agent loop 将此作为 tool_result 回传 LLM，让模型自行修正。
    #[error("tool `{name}` failed: {message}")]
    Tool {
        name: String,
        call_id: ToolCallId,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// 上下文窗口溢出。
    /// Agent loop 收到此错误后应触发上下文压缩策略。
    #[error("context overflow: {used}/{limit} tokens")]
    ContextOverflow { used: usize, limit: usize },

    /// 配置错误（缺失字段、无效值）。
    #[error("config error: {message}")]
    Config { message: String },

    /// 用户或系统取消操作。
    #[error("cancelled")]
    Cancelled,

    /// 不可归类的内部错误。
    #[error("{0}")]
    Internal(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// 构造一个 Tool 错误。
    ///
    /// # Examples
    /// ```
    /// use katu_core::error::Error;
    /// use katu_core::types::ToolCallId;
    ///
    /// let err = Error::tool("bash", ToolCallId::new("call_42"), "command not found");
    /// assert!(matches!(err, Error::Tool { .. }));
    /// ```
    pub fn tool(
        name: impl Into<String>,
        call_id: ToolCallId,
        message: impl Into<String>,
    ) -> Self {
        Self::Tool {
            name: name.into(),
            call_id,
            message: message.into(),
            source: None,
        }
    }

    /// 构造一个带 source 的 Tool 错误。
    pub fn tool_with_source(
        name: impl Into<String>,
        call_id: ToolCallId,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Tool {
            name: name.into(),
            call_id,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// 构造一个带恢复建议的 Provider 错误。
    ///
    /// # Examples
    /// ```
    /// use katu_core::error::{Error, ProviderErrorKind, AuthErrorKind};
    ///
    /// let err = Error::provider(
    ///     ProviderErrorKind::Authentication {
    ///         message: "invalid key".into(),
    ///         kind: AuthErrorKind::Invalid,
    ///     },
    ///     "check your API key in ~/.config/katu/config.toml",
    /// );
    /// assert!(!err.retryable());
    /// ```
    pub fn provider(kind: ProviderErrorKind, suggestion: impl Into<String>) -> Self {
        Self::Provider {
            kind,
            suggestion: Some(suggestion.into()),
        }
    }

    /// 该错误是否可重试。
    ///
    /// 仅 `RateLimit` 和 `ServerError` 返回 true。
    pub fn retryable(&self) -> bool {
        match self {
            Self::Provider { kind, .. } => kind.retryable(),
            _ => false,
        }
    }

    /// 建议的退避时间。
    ///
    /// 如果 Provider 在 header 中指定了 retry-after，返回该值；
    /// 否则按分类返回默认退避时间。
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Provider { kind, .. } => kind.retry_after(),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ProviderErrorKind — Provider 错误分类
// ---------------------------------------------------------------------------

/// Provider 错误分类。
///
/// 每种分类自带 retryable 语义，Agent loop 据此决定重试策略。
/// 可序列化以便持久化到 Session 记录中。
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum ProviderErrorKind {
    /// API Key 缺失、无效、过期、权限不足。
    #[error("authentication failed: {message} ({kind})")]
    Authentication {
        message: String,
        kind: AuthErrorKind,
    },

    /// 请求频率超限（HTTP 429）。
    #[error("rate limited: {message}")]
    RateLimit {
        message: String,
        /// Provider 建议的等待时间。
        #[serde(with = "option_duration_millis")]
        retry_after: Option<Duration>,
    },

    /// 配额/余额耗尽（需充值或切换账户）。
    #[error("quota exceeded: {message}")]
    QuotaExceeded { message: String },

    /// 请求参数非法（prompt 过长、无效模型名等）。
    #[error("invalid request: {message}")]
    InvalidRequest { message: String },

    /// 内容被安全策略过滤。
    #[error("content filtered: {message}")]
    ContentFiltered { message: String },

    /// Provider 服务端错误（5xx）。
    #[error("provider internal error ({status}): {message}")]
    ServerError {
        message: String,
        status: u16,
        #[serde(with = "option_duration_millis")]
        retry_after: Option<Duration>,
    },

    /// 网络/传输层错误（DNS 失败、连接超时、SSL 等）。
    #[error("transport error: {message}")]
    Transport { message: String },

    /// Provider 返回了无法解析的响应。
    #[error("invalid response: {message}")]
    InvalidResponse { message: String },

    /// 未知 Provider 错误。
    #[error("unknown provider error: {message}")]
    Unknown { message: String },
}

impl ProviderErrorKind {
    /// 该错误是否可重试。
    ///
    /// - `true` → RateLimit, ServerError
    /// - `false` → Authentication, QuotaExceeded, InvalidRequest, 其他
    pub fn retryable(&self) -> bool {
        matches!(self, Self::RateLimit { .. } | Self::ServerError { .. })
    }

    /// 建议的退避等待时间。
    ///
    /// 优先使用 Provider 返回的 retry-after 值；
    /// 无显式值时按分类使用默认值：
    /// - RateLimit → 30s
    /// - ServerError → 20s
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimit { retry_after, .. } => {
                retry_after.or(Some(Duration::from_secs(30)))
            }
            Self::ServerError { retry_after, .. } => {
                retry_after.or(Some(Duration::from_secs(20)))
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// AuthErrorKind — 认证错误细分
// ---------------------------------------------------------------------------

/// 认证错误细分。
///
/// 便于上层 UI 给出针对性的恢复建议：
/// - `Missing` → "请设置 API Key"
/// - `Expired` → "请刷新 Token"
/// - `InsufficientPermissions` → "当前 Key 无权使用该模型"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthErrorKind {
    /// API Key 未设置。
    Missing,
    /// API Key 格式正确但被拒绝。
    Invalid,
    /// Token 已过期。
    Expired,
    /// 权限不足（Key 有效但无权访问该模型）。
    InsufficientPermissions,
    /// 无法分类的认证错误。
    Unknown,
}

impl fmt::Display for AuthErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => write!(f, "missing"),
            Self::Invalid => write!(f, "invalid"),
            Self::Expired => write!(f, "expired"),
            Self::InsufficientPermissions => write!(f, "insufficient_permissions"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// Serde helper: Option<Duration> as milliseconds
// ---------------------------------------------------------------------------

/// 将 `Option<Duration>` 序列化为毫秒数（u64 或 null）。
mod option_duration_millis {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(value: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(d) => d.as_millis().serialize(serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<u64> = Option::deserialize(deserializer)?;
        Ok(opt.map(Duration::from_millis))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Error 顶层 --

    /// 验证 Error::tool 构造正确
    #[test]
    fn test_error_tool_construction() {
        let err = Error::tool("read_file", ToolCallId::new("call_1"), "not found");
        match &err {
            Error::Tool {
                name,
                call_id,
                message,
                source,
            } => {
                assert_eq!(name, "read_file");
                assert_eq!(call_id.as_str(), "call_1");
                assert_eq!(message, "not found");
                assert!(source.is_none());
            }
            _ => panic!("expected Error::Tool"),
        }
    }

    /// 验证 Error::tool_with_source 保留原始错误
    #[test]
    fn test_error_tool_with_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let err = Error::tool_with_source(
            "read_file",
            ToolCallId::new("call_2"),
            "failed",
            io_err,
        );
        match &err {
            Error::Tool { source, .. } => {
                assert!(source.is_some());
            }
            _ => panic!("expected Error::Tool"),
        }
    }

    /// 验证 Error::provider 构造正确
    #[test]
    fn test_error_provider_construction() {
        let err = Error::provider(
            ProviderErrorKind::Authentication {
                message: "bad key".into(),
                kind: AuthErrorKind::Invalid,
            },
            "run /login",
        );
        match &err {
            Error::Provider { kind, suggestion } => {
                assert!(matches!(kind, ProviderErrorKind::Authentication { .. }));
                assert_eq!(suggestion.as_deref(), Some("run /login"));
            }
            _ => panic!("expected Error::Provider"),
        }
    }

    /// 验证 Error::Cancelled 不可重试
    #[test]
    fn test_cancelled_not_retryable() {
        let err = Error::Cancelled;
        assert!(!err.retryable());
        assert!(err.retry_after().is_none());
    }

    /// 验证 Error::ContextOverflow 的 Display
    #[test]
    fn test_context_overflow_display() {
        let err = Error::ContextOverflow {
            used: 130000,
            limit: 128000,
        };
        assert_eq!(err.to_string(), "context overflow: 130000/128000 tokens");
    }

    /// 验证 Error::Config 的 Display
    #[test]
    fn test_config_error_display() {
        let err = Error::Config {
            message: "missing api_key".into(),
        };
        assert!(err.to_string().contains("missing api_key"));
    }

    // -- ProviderErrorKind --

    /// 验证 RateLimit 可重试
    #[test]
    fn test_rate_limit_retryable() {
        let kind = ProviderErrorKind::RateLimit {
            message: "429".into(),
            retry_after: Some(Duration::from_secs(60)),
        };
        assert!(kind.retryable());
        assert_eq!(kind.retry_after(), Some(Duration::from_secs(60)));
    }

    /// 验证 RateLimit 无显式 retry_after 时使用默认 30s
    #[test]
    fn test_rate_limit_default_retry_after() {
        let kind = ProviderErrorKind::RateLimit {
            message: "slow down".into(),
            retry_after: None,
        };
        assert_eq!(kind.retry_after(), Some(Duration::from_secs(30)));
    }

    /// 验证 ServerError 可重试且默认 20s
    #[test]
    fn test_server_error_retryable() {
        let kind = ProviderErrorKind::ServerError {
            message: "internal".into(),
            status: 500,
            retry_after: None,
        };
        assert!(kind.retryable());
        assert_eq!(kind.retry_after(), Some(Duration::from_secs(20)));
    }

    /// 验证 Authentication 不可重试
    #[test]
    fn test_authentication_not_retryable() {
        let kind = ProviderErrorKind::Authentication {
            message: "invalid".into(),
            kind: AuthErrorKind::Invalid,
        };
        assert!(!kind.retryable());
        assert!(kind.retry_after().is_none());
    }

    /// 验证 QuotaExceeded 不可重试
    #[test]
    fn test_quota_exceeded_not_retryable() {
        let kind = ProviderErrorKind::QuotaExceeded {
            message: "out of credits".into(),
        };
        assert!(!kind.retryable());
    }

    /// 验证 InvalidRequest 不可重试
    #[test]
    fn test_invalid_request_not_retryable() {
        let kind = ProviderErrorKind::InvalidRequest {
            message: "model not found".into(),
        };
        assert!(!kind.retryable());
    }

    /// 验证 ContentFiltered 不可重试
    #[test]
    fn test_content_filtered_not_retryable() {
        let kind = ProviderErrorKind::ContentFiltered {
            message: "blocked".into(),
        };
        assert!(!kind.retryable());
    }

    /// 验证 Transport 不可重试
    #[test]
    fn test_transport_not_retryable() {
        let kind = ProviderErrorKind::Transport {
            message: "dns failed".into(),
        };
        assert!(!kind.retryable());
    }

    /// 验证 InvalidResponse 不可重试
    #[test]
    fn test_invalid_response_not_retryable() {
        let kind = ProviderErrorKind::InvalidResponse {
            message: "bad json".into(),
        };
        assert!(!kind.retryable());
    }

    /// 验证 Unknown 不可重试
    #[test]
    fn test_unknown_not_retryable() {
        let kind = ProviderErrorKind::Unknown {
            message: "???".into(),
        };
        assert!(!kind.retryable());
    }

    // -- ProviderErrorKind serde --

    /// 验证 ProviderErrorKind 序列化/反序列化 roundtrip
    #[test]
    fn test_provider_error_kind_serde_roundtrip() {
        let kind = ProviderErrorKind::RateLimit {
            message: "429 too many".into(),
            retry_after: Some(Duration::from_millis(5000)),
        };
        let json = serde_json::to_string(&kind).unwrap();
        let restored: ProviderErrorKind = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            restored,
            ProviderErrorKind::RateLimit {
                retry_after: Some(d),
                ..
            } if d == Duration::from_millis(5000)
        ));
    }

    /// 验证 ProviderErrorKind::ServerError serde roundtrip
    #[test]
    fn test_server_error_serde_roundtrip() {
        let kind = ProviderErrorKind::ServerError {
            message: "overloaded".into(),
            status: 529,
            retry_after: None,
        };
        let json = serde_json::to_string(&kind).unwrap();
        let restored: ProviderErrorKind = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            restored,
            ProviderErrorKind::ServerError { status: 529, retry_after: None, .. }
        ));
    }

    // -- AuthErrorKind --

    /// 验证 AuthErrorKind 的 Display 输出
    #[test]
    fn test_auth_error_kind_display() {
        assert_eq!(AuthErrorKind::Missing.to_string(), "missing");
        assert_eq!(AuthErrorKind::Invalid.to_string(), "invalid");
        assert_eq!(AuthErrorKind::Expired.to_string(), "expired");
        assert_eq!(
            AuthErrorKind::InsufficientPermissions.to_string(),
            "insufficient_permissions"
        );
        assert_eq!(AuthErrorKind::Unknown.to_string(), "unknown");
    }

    /// 验证 AuthErrorKind serde roundtrip
    #[test]
    fn test_auth_error_kind_serde_roundtrip() {
        for kind in [
            AuthErrorKind::Missing,
            AuthErrorKind::Invalid,
            AuthErrorKind::Expired,
            AuthErrorKind::InsufficientPermissions,
            AuthErrorKind::Unknown,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let restored: AuthErrorKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, restored);
        }
    }

    // -- Error::retryable / retry_after 集成 --

    /// 验证 Error 层面的 retryable 委托到 ProviderErrorKind
    #[test]
    fn test_error_retryable_delegates() {
        let retryable_err = Error::Provider {
            kind: ProviderErrorKind::RateLimit {
                message: "wait".into(),
                retry_after: Some(Duration::from_secs(10)),
            },
            suggestion: None,
        };
        assert!(retryable_err.retryable());
        assert_eq!(retryable_err.retry_after(), Some(Duration::from_secs(10)));

        let non_retryable_err = Error::Provider {
            kind: ProviderErrorKind::Authentication {
                message: "bad".into(),
                kind: AuthErrorKind::Invalid,
            },
            suggestion: None,
        };
        assert!(!non_retryable_err.retryable());
        assert!(non_retryable_err.retry_after().is_none());
    }

    /// 验证非 Provider 错误的 retryable 始终为 false
    #[test]
    fn test_non_provider_errors_not_retryable() {
        assert!(!Error::Cancelled.retryable());
        assert!(!Error::Config { message: "x".into() }.retryable());
        assert!(!Error::ContextOverflow { used: 1, limit: 1 }.retryable());
        assert!(!Error::tool("t", ToolCallId::new("c"), "m").retryable());
    }

    // -- Display --

    /// 验证各错误变体的 Display 格式
    #[test]
    fn test_error_display_formats() {
        let tool_err = Error::tool("bash", ToolCallId::new("c1"), "permission denied");
        assert_eq!(
            tool_err.to_string(),
            "tool `bash` failed: permission denied"
        );

        let provider_err = Error::Provider {
            kind: ProviderErrorKind::Transport {
                message: "connection reset".into(),
            },
            suggestion: Some("check your network".into()),
        };
        assert_eq!(
            provider_err.to_string(),
            "provider error: transport error: connection reset"
        );
    }
}
