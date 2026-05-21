//! # prompt::provider
//!
//! ## 职责
//! 定义 Prompt 段生成器 trait — Prompt 系统的核心扩展点。
//!
//! ## 对外接口
//! - `PromptSectionProvider` — 段生成器 trait
//!
//! ## 调用者
//! - `prompt::builder` — PromptBuilder 调用 provide() 收集段内容
//! - `prompt::builtin` — 内置段实现此 trait

use super::context::PromptContext;
use super::section::CacheHint;

// ===========================================================================
// PromptSectionProvider
// ===========================================================================

/// Prompt 段生成器 — 负责产出一段 prompt 内容。
///
/// 实现者根据 `PromptContext` 判断是否需要此段，并生成文本内容。
/// `PromptBuilder` 根据 `order()` 排序所有生成器，依次调用 `provide()`。
///
/// ## 设计参考
/// - Claude Code: `systemPromptSection(name, compute)` — 函数式段注册
/// - Oh-My-Pi: Handlebars 模板中的 `{{#if}}` 条件段
/// - 本设计: trait 对象，支持类型安全的条件生成和组合
///
/// ## 扩展方式
/// 上层 crate（如 `katu-app`）可实现自定义 `PromptSectionProvider`
/// 并通过 `PromptBuilder::with_provider()` 注册，无需修改 `katu-agent`。
///
/// # Examples
///
/// ```
/// use katu_agent::prompt::{PromptSectionProvider, PromptContext, CacheHint};
///
/// struct ProjectRulesSection;
///
/// impl PromptSectionProvider for ProjectRulesSection {
///     fn name(&self) -> &str { "project_rules" }
///
///     fn provide(&self, ctx: &PromptContext<'_>) -> Option<String> {
///         if ctx.user_instructions.is_empty() {
///             return None;
///         }
///         Some(format!("Follow these rules:\n{}", ctx.user_instructions.join("\n")))
///     }
///
///     fn cache_hint(&self) -> CacheHint { CacheHint::Session }
///     fn order(&self) -> u32 { 75 }
/// }
/// ```
pub trait PromptSectionProvider: Send + Sync {
    /// 段名（用于日志、调试、缓存键）。
    ///
    /// 同一 PromptBuilder 中不应注册重名的段。
    fn name(&self) -> &str;

    /// 生成段内容。
    ///
    /// 返回 `None` 表示此段在当前上下文中不适用（条件不满足）。
    /// 返回 `Some(text)` 的文本将按 `order()` 顺序拼接到 system prompt。
    /// 空字符串等同于 `None`，会被 PromptBuilder 过滤。
    fn provide(&self, ctx: &PromptContext<'_>) -> Option<String>;

    /// 缓存提示（默认 Static）。
    ///
    /// - `Static` — 首次计算后缓存，整个 PromptBuilder 生命周期内复用
    /// - `Session` — 同上（语义上标记为"会话级"，`clear_cache()` 时清除）
    /// - `Volatile` — 每次 `build()` 都重新计算
    fn cache_hint(&self) -> CacheHint {
        CacheHint::Static
    }

    /// 排序权重 — 越小越靠前（默认 100）。
    ///
    /// 内置段使用 10-90 范围。自定义段建议使用 100+。
    /// 靠前的段更容易被 Provider prompt cache 命中。
    fn order(&self) -> u32 {
        100
    }
}
