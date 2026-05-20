//! # katu_llm::cache
//!
//! ## 职责
//! 定义 prompt 缓存策略。
//!
//! ## 对外接口
//! - `CachePolicy` — 缓存控制策略

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CachePolicy
// ---------------------------------------------------------------------------

/// Prompt 缓存策略。
///
/// 控制 LLM 请求中哪些部分可被 provider 缓存以降低延迟和费用。
///
/// - `Auto` — 推荐的 agent loop 默认值，自动在 tools/system/messages 尾部放置缓存标记
/// - `None` — 不使用缓存
/// - `Custom` — 精确控制各部分的缓存行为
///
/// # Examples
/// ```
/// use katu_llm::CachePolicy;
///
/// let policy = CachePolicy::Auto;
/// assert!(policy.is_enabled());
///
/// let custom = CachePolicy::Custom {
///     tools: true,
///     system: true,
///     messages: false,
/// };
/// assert!(custom.is_enabled());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CachePolicy {
    /// 自动缓存（推荐默认值）。
    /// 在 tools、system prompt 尾部、最近 user message 放置缓存断点。
    #[default]
    Auto,
    /// 不使用缓存。
    None,
    /// 自定义缓存策略。
    Custom {
        /// 是否缓存工具定义
        tools: bool,
        /// 是否缓存 system prompt
        system: bool,
        /// 是否缓存消息尾部
        messages: bool,
    },
}

impl CachePolicy {
    /// 缓存是否启用（非 None）。
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_auto() {
        assert_eq!(CachePolicy::default(), CachePolicy::Auto);
    }

    #[test]
    fn test_is_enabled() {
        assert!(CachePolicy::Auto.is_enabled());
        assert!(!CachePolicy::None.is_enabled());
        assert!(CachePolicy::Custom {
            tools: false,
            system: false,
            messages: false,
        }
        .is_enabled());
    }

    #[test]
    fn test_serde_roundtrip_auto() {
        let policy = CachePolicy::Auto;
        let json = serde_json::to_string(&policy).unwrap();
        let restored: CachePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(policy, restored);
    }

    #[test]
    fn test_serde_roundtrip_custom() {
        let policy = CachePolicy::Custom {
            tools: true,
            system: true,
            messages: false,
        };
        let json = serde_json::to_string(&policy).unwrap();
        let restored: CachePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(policy, restored);
    }
}
