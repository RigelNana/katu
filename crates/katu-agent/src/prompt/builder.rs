//! # prompt::builder
//!
//! ## 职责
//! Prompt 组装器 — 管理段生成器，按序组装最终 system prompt。
//!
//! ## 对外接口
//! - `PromptBuilder` — Builder 模式组装器
//!
//! ## 调用者
//! - `katu-agent::runner` (future) — Agent loop 每轮调用 `build()`

use std::collections::HashMap;

use super::builtin;
use super::context::PromptContext;
use super::provider::PromptSectionProvider;
use super::section::{CacheHint, PromptOutput, PromptSection};

// ===========================================================================
// PromptBuilder
// ===========================================================================

/// Prompt 组装器 — 管理所有段生成器，按顺序组装最终 system prompt。
///
/// ## 两阶段使用
///
/// ```text
/// 配置阶段（Builder 模式，consume self）:
///     let mut builder = PromptBuilder::new()
///         .with_provider(SectionA)
///         .with_provider(SectionB);
///
/// 使用阶段（每轮调用 &mut self）:
///     let output = builder.build(&ctx);
/// ```
///
/// ## 缓存行为
/// - `CacheHint::Static` / `Session` 段 — 首次 `build()` 后缓存结果，
///   后续直接复用。调用 `clear_cache()` 清除。
/// - `CacheHint::Volatile` 段 — 每次 `build()` 都重新计算。
///
/// # Examples
///
/// ```
/// use katu_agent::prompt::{PromptBuilder, PromptContext, EnvironmentInfo};
/// use katu_core::{AgentDefinition, AgentRole, ModelId, ProviderId};
///
/// let agent = AgentDefinition::new("build", AgentRole::Primary)
///     .with_system_prompt("You are a coding assistant.");
/// let model_id = ModelId::new("gpt-4o");
/// let provider_id = ProviderId::new("openai");
/// let env = EnvironmentInfo::new("/tmp", "linux");
///
/// let mut builder = PromptBuilder::with_defaults();
/// let ctx = PromptContext::new(&agent, &model_id, &provider_id, &env);
/// let output = builder.build(&ctx);
///
/// assert!(!output.text.is_empty());
/// assert!(!output.sections.is_empty());
/// ```
pub struct PromptBuilder {
    /// 注册的段生成器。
    providers: Vec<Box<dyn PromptSectionProvider>>,
    /// 会话级缓存（段名 → 内容；None 表示该段返回了 None）。
    session_cache: HashMap<String, Option<String>>,
    /// providers 是否已按 order 排序。
    sorted: bool,
}

impl PromptBuilder {
    /// 创建空的 PromptBuilder（无任何段生成器）。
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            session_cache: HashMap::new(),
            sorted: true,
        }
    }

    /// 创建包含所有内置段的 PromptBuilder。
    ///
    /// 内置段按 order 排列：
    /// identity(10) → core_instructions(20) → tool_guidance(30) →
    /// agent_prompt(40) → environment(60) → user_instructions(70) →
    /// language(90)
    pub fn with_defaults() -> Self {
        Self::new()
            .with_provider(builtin::IdentitySection)
            .with_provider(builtin::CoreInstructionsSection)
            .with_provider(builtin::ToolGuidanceSection)
            .with_provider(builtin::AgentPromptSection)
            .with_provider(builtin::EnvironmentSection)
            .with_provider(builtin::UserInstructionsSection)
            .with_provider(builtin::LanguageSection)
    }

    /// 注册一个段生成器（Builder 模式，consume self）。
    pub fn with_provider(mut self, provider: impl PromptSectionProvider + 'static) -> Self {
        self.providers.push(Box::new(provider));
        self.sorted = false;
        self
    }

    /// 在配置完成后追加段生成器。
    pub fn register(&mut self, provider: impl PromptSectionProvider + 'static) {
        self.providers.push(Box::new(provider));
        self.sorted = false;
    }

    /// 组装最终 prompt。
    ///
    /// 每轮 LLM 调用前由 AgentRunner 调用。
    /// 按 `order()` 排序所有段生成器，依次调用 `provide()`，
    /// 跳过返回 None 或空字符串的段，拼接非空段为最终文本。
    pub fn build(&mut self, ctx: &PromptContext<'_>) -> PromptOutput {
        self.ensure_sorted();

        let mut sections = Vec::new();

        for provider in &self.providers {
            let name = provider.name().to_string();
            let hint = provider.cache_hint();

            // 获取内容：Volatile 每次重新计算，其余使用缓存
            let content = match hint {
                CacheHint::Volatile => provider.provide(ctx),
                _ => {
                    if let Some(cached) = self.session_cache.get(&name) {
                        cached.clone()
                    } else {
                        let result = provider.provide(ctx);
                        self.session_cache.insert(name.clone(), result.clone());
                        result
                    }
                }
            };

            // 跳过 None 和空内容
            if let Some(text) = content {
                if !text.is_empty() {
                    sections.push(PromptSection {
                        name,
                        content: text,
                        cache_hint: hint,
                    });
                }
            }
        }

        // 计算缓存分界
        let cache_boundary = Self::compute_cache_boundary(&sections);

        // 拼接文本（段间以双换行分隔）
        let text = sections
            .iter()
            .map(|s| s.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        PromptOutput {
            sections,
            text,
            cache_boundary,
        }
    }

    /// 清除会话缓存。
    ///
    /// 用于 `/clear` 命令或 compaction 后重建 prompt。
    /// 下次 `build()` 将重新计算所有非 Volatile 段。
    pub fn clear_cache(&mut self) {
        self.session_cache.clear();
    }

    /// 已注册的段生成器数量。
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    // -----------------------------------------------------------------------
    // 内部方法
    // -----------------------------------------------------------------------

    /// 确保 providers 按 order 排序。
    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.providers.sort_by_key(|p| p.order());
            self.sorted = true;
        }
    }

    /// 计算静态/动态分界位置。
    ///
    /// 从头开始找到连续的 Static 段数量。
    /// 第一个非 Static 段之前就是分界点。
    fn compute_cache_boundary(sections: &[PromptSection]) -> Option<usize> {
        let boundary = sections
            .iter()
            .position(|s| !s.cache_hint.is_static());

        match boundary {
            Some(0) => None,                                       // 第一段就不是 Static
            Some(n) => Some(n),                                    // n 段 Static 后出现非 Static
            None if !sections.is_empty() => Some(sections.len()),  // 全部 Static
            None => None,                                          // 空
        }
    }
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::EnvironmentInfo;
    use katu_core::{AgentDefinition, AgentRole, ModelId, ProviderId, ToolDefinition};

    fn test_agent() -> AgentDefinition {
        AgentDefinition::new("test", AgentRole::Primary)
            .with_system_prompt("You are a test assistant.")
    }

    fn test_env() -> EnvironmentInfo {
        EnvironmentInfo::new("/tmp/test", "linux")
            .with_date("2025-05-21")
    }

    fn test_ctx<'a>(
        agent: &'a AgentDefinition,
        model_id: &'a ModelId,
        provider_id: &'a ProviderId,
        env: &'a EnvironmentInfo,
    ) -> PromptContext<'a> {
        PromptContext::new(agent, model_id, provider_id, env)
    }

    // -- 自定义测试段 --

    struct StaticSection(&'static str);
    impl PromptSectionProvider for StaticSection {
        fn name(&self) -> &str { self.0 }
        fn provide(&self, _ctx: &PromptContext<'_>) -> Option<String> {
            Some(format!("Content of {}", self.0))
        }
        fn cache_hint(&self) -> CacheHint { CacheHint::Static }
        fn order(&self) -> u32 { 50 }
    }

    struct SessionSection(&'static str);
    impl PromptSectionProvider for SessionSection {
        fn name(&self) -> &str { self.0 }
        fn provide(&self, _ctx: &PromptContext<'_>) -> Option<String> {
            Some(format!("Session: {}", self.0))
        }
        fn cache_hint(&self) -> CacheHint { CacheHint::Session }
        fn order(&self) -> u32 { 60 }
    }

    struct VolatileCounter {
        name: &'static str,
    }
    impl PromptSectionProvider for VolatileCounter {
        fn name(&self) -> &str { self.name }
        fn provide(&self, ctx: &PromptContext<'_>) -> Option<String> {
            Some(format!("Step: {}", ctx.step_count))
        }
        fn cache_hint(&self) -> CacheHint { CacheHint::Volatile }
        fn order(&self) -> u32 { 70 }
    }

    struct SkippedSection;
    impl PromptSectionProvider for SkippedSection {
        fn name(&self) -> &str { "skipped" }
        fn provide(&self, _ctx: &PromptContext<'_>) -> Option<String> { None }
        fn order(&self) -> u32 { 55 }
    }

    // -- 基本功能 --

    #[test]
    fn test_empty_builder() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();
        let ctx = test_ctx(&agent, &model_id, &provider_id, &env);

        let mut builder = PromptBuilder::new();
        let output = builder.build(&ctx);

        assert!(output.is_empty());
        assert!(output.text.is_empty());
        assert!(output.cache_boundary.is_none());
    }

    #[test]
    fn test_with_defaults() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();
        let ctx = test_ctx(&agent, &model_id, &provider_id, &env);

        let mut builder = PromptBuilder::with_defaults();
        assert_eq!(builder.provider_count(), 7);

        let output = builder.build(&ctx);

        // 至少有 identity + core_instructions + agent_prompt + environment
        assert!(output.section_count() >= 4);
        assert!(!output.text.is_empty());
        assert!(output.text.contains("test")); // agent name
        assert!(output.text.contains("# System")); // core instructions
        assert!(output.text.contains("<environment>")); // environment
        assert!(output.text.contains("You are a test assistant.")); // agent prompt
    }

    #[test]
    fn test_provider_order() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();
        let ctx = test_ctx(&agent, &model_id, &provider_id, &env);

        // 注册顺序与 order 不同
        let mut builder = PromptBuilder::new()
            .with_provider(SessionSection("second"))
            .with_provider(StaticSection("first"));

        let output = builder.build(&ctx);

        // StaticSection(order=50) 应在 SessionSection(order=60) 之前
        assert_eq!(output.sections[0].name, "first");
        assert_eq!(output.sections[1].name, "second");
    }

    #[test]
    fn test_skipped_section() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();
        let ctx = test_ctx(&agent, &model_id, &provider_id, &env);

        let mut builder = PromptBuilder::new()
            .with_provider(StaticSection("a"))
            .with_provider(SkippedSection)
            .with_provider(SessionSection("b"));

        let output = builder.build(&ctx);

        // SkippedSection 不应出现
        assert_eq!(output.section_count(), 2);
        assert!(output.sections.iter().all(|s| s.name != "skipped"));
    }

    // -- 缓存行为 --

    #[test]
    fn test_session_cache() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();

        let mut builder = PromptBuilder::new()
            .with_provider(StaticSection("cached"));

        // 第一次 build
        let ctx1 = test_ctx(&agent, &model_id, &provider_id, &env);
        let output1 = builder.build(&ctx1);
        assert_eq!(output1.section_count(), 1);

        // 第二次 build — 应使用缓存
        let ctx2 = test_ctx(&agent, &model_id, &provider_id, &env);
        let output2 = builder.build(&ctx2);
        assert_eq!(output1.text, output2.text);
    }

    #[test]
    fn test_volatile_not_cached() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();

        let mut builder = PromptBuilder::new()
            .with_provider(VolatileCounter { name: "counter" });

        // step_count = 0
        let ctx1 = test_ctx(&agent, &model_id, &provider_id, &env);
        let output1 = builder.build(&ctx1);
        assert!(output1.text.contains("Step: 0"));

        // step_count = 5 — Volatile 应重新计算
        let ctx2 = test_ctx(&agent, &model_id, &provider_id, &env)
            .with_step_count(5);
        let output2 = builder.build(&ctx2);
        assert!(output2.text.contains("Step: 5"));
    }

    #[test]
    fn test_clear_cache() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();
        let ctx = test_ctx(&agent, &model_id, &provider_id, &env);

        let mut builder = PromptBuilder::new()
            .with_provider(StaticSection("a"));

        let output1 = builder.build(&ctx);
        builder.clear_cache();
        let output2 = builder.build(&ctx);

        // 内容应相同（虽然重新计算了）
        assert_eq!(output1.text, output2.text);
    }

    // -- 缓存分界 --

    #[test]
    fn test_cache_boundary_all_static() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();
        let ctx = test_ctx(&agent, &model_id, &provider_id, &env);

        let mut builder = PromptBuilder::new()
            .with_provider(StaticSection("a"))
            .with_provider(StaticSection("b"));

        let output = builder.build(&ctx);
        // 全部 Static → boundary = 段数
        assert_eq!(output.cache_boundary, Some(2));
    }

    #[test]
    fn test_cache_boundary_mixed() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();
        let ctx = test_ctx(&agent, &model_id, &provider_id, &env);

        let mut builder = PromptBuilder::new()
            .with_provider(StaticSection("a"))       // order=50, Static
            .with_provider(SessionSection("b"));     // order=60, Session

        let output = builder.build(&ctx);
        // 第一段 Static，第二段 Session → boundary = 1
        assert_eq!(output.cache_boundary, Some(1));
    }

    #[test]
    fn test_cache_boundary_no_static() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();
        let ctx = test_ctx(&agent, &model_id, &provider_id, &env);

        let mut builder = PromptBuilder::new()
            .with_provider(SessionSection("a"));

        let output = builder.build(&ctx);
        // 第一段就不是 Static → None
        assert!(output.cache_boundary.is_none());
    }

    // -- register --

    #[test]
    fn test_register_after_build() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();
        let ctx = test_ctx(&agent, &model_id, &provider_id, &env);

        let mut builder = PromptBuilder::new()
            .with_provider(StaticSection("a"));

        let output1 = builder.build(&ctx);
        assert_eq!(output1.section_count(), 1);

        // 追加段后重新 build
        builder.register(SessionSection("b"));
        builder.clear_cache(); // 清缓存以重新计算
        let output2 = builder.build(&ctx);
        assert_eq!(output2.section_count(), 2);
    }

    // -- 文本拼接 --

    #[test]
    fn test_text_join() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();
        let ctx = test_ctx(&agent, &model_id, &provider_id, &env);

        let mut builder = PromptBuilder::new()
            .with_provider(StaticSection("a"))
            .with_provider(SessionSection("b"));

        let output = builder.build(&ctx);

        // 段间以 \n\n 分隔
        assert!(output.text.contains("Content of a\n\nSession: b"));
    }

    // -- with_defaults 工具引导 --

    #[test]
    fn test_defaults_with_tools() {
        let agent = test_agent();
        let model_id = ModelId::new("gpt-4o");
        let provider_id = ProviderId::new("openai");
        let env = test_env();
        let tools = vec![
            ToolDefinition::no_params("read_file", "Read a file"),
            ToolDefinition::no_params("bash", "Run command"),
        ];
        let ctx = test_ctx(&agent, &model_id, &provider_id, &env).with_tools(&tools);

        let mut builder = PromptBuilder::with_defaults();
        let output = builder.build(&ctx);

        // 应包含 tool_guidance 段
        assert!(output.sections.iter().any(|s| s.name == "tool_guidance"));
        assert!(output.text.contains("# Tool Usage"));
    }
}
