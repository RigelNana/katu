//! # prompt::section
//!
//! ## 职责
//! 定义 Prompt 段的基本数据类型 — PromptSection、CacheHint、PromptOutput。
//!
//! ## 对外接口
//! - `CacheHint` — 缓存策略提示（Static / Session / Volatile）
//! - `PromptSection` — 一个命名 prompt 段
//! - `PromptOutput` — 完整组装结果

use serde::{Deserialize, Serialize};

// ===========================================================================
// CacheHint
// ===========================================================================

/// Prompt 缓存提示 — 引导 Provider 层 prompt caching 策略。
///
/// 不直接控制缓存行为 — 最终由 Provider 适配层决定。
/// 例如 Anthropic 可将 Static 段标记为 `cache_control: ephemeral`。
///
/// # Examples
///
/// ```
/// use katu_agent::prompt::CacheHint;
///
/// let hint = CacheHint::Static;
/// assert!(hint.is_static());
/// assert!(!hint.is_volatile());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheHint {
    /// 静态段 — 跨会话不变，可全局缓存。
    /// 例: identity、core instructions
    Static,
    /// 会话级段 — 同一会话内不变，会话间可能不同。
    /// 例: environment info、user rules
    Session,
    /// 易变段 — 每轮可能变化，不宜缓存。
    /// 例: MCP instructions、动态工具列表
    Volatile,
}

impl CacheHint {
    /// 是否为静态段。
    pub fn is_static(&self) -> bool {
        matches!(self, Self::Static)
    }

    /// 是否为会话级段。
    pub fn is_session(&self) -> bool {
        matches!(self, Self::Session)
    }

    /// 是否为易变段。
    pub fn is_volatile(&self) -> bool {
        matches!(self, Self::Volatile)
    }
}

impl Default for CacheHint {
    fn default() -> Self {
        Self::Static
    }
}

// ===========================================================================
// PromptSection
// ===========================================================================

/// Prompt 段 — system prompt 的最小组成单元。
///
/// 每段有名称、内容和缓存属性。
/// 由 `PromptBuilder` 根据 `PromptSectionProvider::provide()` 的结果构造。
///
/// # Examples
///
/// ```
/// use katu_agent::prompt::{PromptSection, CacheHint};
///
/// let section = PromptSection {
///     name: "identity".into(),
///     content: "You are a coding assistant.".into(),
///     cache_hint: CacheHint::Static,
/// };
/// assert_eq!(section.name, "identity");
/// ```
#[derive(Debug, Clone)]
pub struct PromptSection {
    /// 段名（用于日志、调试、缓存键）。
    pub name: String,
    /// 段内容。
    pub content: String,
    /// 缓存提示。
    pub cache_hint: CacheHint,
}

// ===========================================================================
// PromptOutput
// ===========================================================================

/// Prompt 构建结果 — `PromptBuilder::build()` 的输出。
///
/// 包含有序段列表、拼接后的完整文本和缓存分界信息。
/// AgentRunner 将 `text` 设置为 `LlmRequest` 的 system prompt，
/// Provider 层可利用 `sections` 和 `cache_boundary` 优化缓存。
///
/// # Examples
///
/// ```
/// use katu_agent::prompt::PromptOutput;
///
/// let output = PromptOutput {
///     sections: vec![],
///     text: String::new(),
///     cache_boundary: None,
/// };
/// assert!(output.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct PromptOutput {
    /// 有序段列表（保留结构信息，供调试/缓存用）。
    pub sections: Vec<PromptSection>,
    /// 拼接后的完整 system prompt 文本（段间以 `\n\n` 分隔）。
    pub text: String,
    /// 静态/动态分界索引。
    ///
    /// `sections[0..boundary]` 为 Static 段，可被 Provider 全局缓存。
    /// None 表示没有明确分界。
    pub cache_boundary: Option<usize>,
}

impl PromptOutput {
    /// 是否为空（没有任何段）。
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// 段数量。
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hint_default_is_static() {
        assert_eq!(CacheHint::default(), CacheHint::Static);
    }

    #[test]
    fn test_cache_hint_predicates() {
        assert!(CacheHint::Static.is_static());
        assert!(!CacheHint::Static.is_session());
        assert!(!CacheHint::Static.is_volatile());

        assert!(!CacheHint::Session.is_static());
        assert!(CacheHint::Session.is_session());
        assert!(!CacheHint::Session.is_volatile());

        assert!(!CacheHint::Volatile.is_static());
        assert!(!CacheHint::Volatile.is_session());
        assert!(CacheHint::Volatile.is_volatile());
    }

    #[test]
    fn test_cache_hint_serde_roundtrip() {
        let hints = [CacheHint::Static, CacheHint::Session, CacheHint::Volatile];
        for hint in hints {
            let json = serde_json::to_string(&hint).unwrap();
            let restored: CacheHint = serde_json::from_str(&json).unwrap();
            assert_eq!(hint, restored);
        }
    }

    #[test]
    fn test_prompt_output_empty() {
        let output = PromptOutput {
            sections: vec![],
            text: String::new(),
            cache_boundary: None,
        };
        assert!(output.is_empty());
        assert_eq!(output.section_count(), 0);
    }

    #[test]
    fn test_prompt_output_with_sections() {
        let output = PromptOutput {
            sections: vec![PromptSection {
                name: "test".into(),
                content: "hello".into(),
                cache_hint: CacheHint::Static,
            }],
            text: "hello".into(),
            cache_boundary: Some(1),
        };
        assert!(!output.is_empty());
        assert_eq!(output.section_count(), 1);
    }
}
