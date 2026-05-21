//! # prompt
//!
//! ## 职责
//! System prompt 组装系统 — 分段构建、缓存管理、Provider 缓存优化。
//!
//! ## 架构
//! ```text
//! PromptBuilder (组装器, Builder 模式)
//!   ├── PromptSectionProvider (段生成器 trait)
//!   │   ├── builtin::IdentitySection          (order: 10, Static)
//!   │   ├── builtin::CoreInstructionsSection   (order: 20, Static)
//!   │   ├── builtin::ToolGuidanceSection       (order: 30, Session)
//!   │   ├── builtin::AgentPromptSection        (order: 40, Static)
//!   │   ├── builtin::EnvironmentSection        (order: 60, Session)
//!   │   ├── builtin::UserInstructionsSection   (order: 70, Session)
//!   │   └── builtin::LanguageSection           (order: 90, Session)
//!   └── build(&ctx) → PromptOutput
//! ```
//!
//! ## 使用方式
//! ```rust,no_run
//! use katu_agent::prompt::{PromptBuilder, PromptContext, EnvironmentInfo};
//! use katu_core::{AgentDefinition, AgentRole, ModelId, ProviderId};
//!
//! // 配置阶段（Builder 模式）
//! let mut builder = PromptBuilder::with_defaults();
//!
//! // 每轮 LLM 调用前
//! let agent = AgentDefinition::new("build", AgentRole::Primary);
//! let env = EnvironmentInfo::new("/project", "linux");
//! let model_id = ModelId::new("gpt-4o");
//! let provider_id = ProviderId::new("openai");
//! let ctx = PromptContext::new(
//!     &agent,
//!     &model_id,
//!     &provider_id,
//!     &env,
//! );
//! let output = builder.build(&ctx);
//! // output.text → 完整 system prompt
//! ```
//!
//! ## 扩展方式
//! 实现 `PromptSectionProvider` trait 并通过 `PromptBuilder::with_provider()`
//! 或 `PromptBuilder::register()` 注册自定义段。

mod builder;
pub mod builtin;
mod context;
mod provider;
mod section;

pub use builder::PromptBuilder;
pub use context::{EnvironmentInfo, PromptContext};
pub use provider::PromptSectionProvider;
pub use section::{CacheHint, PromptOutput, PromptSection};
