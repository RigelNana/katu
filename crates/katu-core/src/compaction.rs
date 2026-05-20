//! # katu_core::compaction
//!
//! ## 职责
//! 定义上下文压缩（Compaction）的配置与数据类型。
//!
//! ## 设计来源
//! 综合 oh-my-pi、opencode、claude-code 三个项目的压缩系统设计：
//!
//! | 维度         | oh-my-pi           | opencode        | claude-code     | katu          |
//! |-------------|--------------------|-----------------|-----------------| --------------|
//! | 触发方式     | threshold+overflow | 仅 overflow     | threshold       | 可配置        |
//! | 阈值模型     | 百分比/固定/reserve | 固定 buffer     | effectiveWindow | 统一三种      |
//! | 保留策略     | keepRecentTokens   | tail_turns+tokens| 无             | turns+tokens  |
//! | 修剪(Prune)  | 无                 | 旧工具输出修剪   | 无             | 可配置        |
//! | 策略         | summarize/handoff  | summarize       | summarize       | 可扩展        |
//! | 熔断器       | 无                 | 无              | 3次失败         | 可配置        |
//! | 压缩模型     | 可选               | compaction agent | 主模型          | 可选          |
//!
//! ## 分层原则
//! - **katu-core（本模块）** — 纯数据配置、结果类型、token 状态枚举
//! - **katu-agent（future）** — 运行时压缩逻辑、overflow 检测、LLM 调用、状态机
//!
//! ## 对外接口
//! - `CompactionConfig` — 压缩完整配置
//! - `CompactionThreshold` — 阈值配置
//! - `CompactionTriggerMode` — 触发模式
//! - `CompactionStrategy` — 压缩策略
//! - `PreserveConfig` — 消息保留策略
//! - `PruneConfig` — 旧工具输出修剪配置
//! - `CompactionResult` — 压缩执行结果
//! - `TokenBudgetState` — token 用量警告状态
//!
//! ## 调用者
//! - `katu-agent` (future) — Agent loop 读取配置驱动压缩
//! - `AgentDefinition` (future) — 可选嵌入 CompactionConfig
//! - UI 层 — 展示 TokenBudgetState 进度条

use serde::{Deserialize, Serialize};

use crate::agent::AgentModelRef;

// ===========================================================================
// CompactionConfig
// ===========================================================================

/// 上下文压缩完整配置。
///
/// 控制 Agent loop 何时触发压缩、如何保留近期上下文、是否修剪旧内容、
/// 以及压缩失败时的熔断行为。
///
/// ## 配置合并优先级
/// ```text
/// AgentDefinition.compaction > SessionConfig.compaction > 全局默认
/// ```
///
/// # Examples
///
/// ```
/// use katu_core::compaction::CompactionConfig;
///
/// // 默认配置：自动压缩开启，threshold 模式
/// let config = CompactionConfig::default();
/// assert!(config.auto_enabled);
///
/// // 禁用自动压缩
/// let manual_only = CompactionConfig::default().with_auto_enabled(false);
/// assert!(!manual_only.auto_enabled);
///
/// // 只在 overflow 时被动压缩（opencode 风格）
/// use katu_core::compaction::CompactionTriggerMode;
/// let passive = CompactionConfig::default()
///     .with_trigger_mode(CompactionTriggerMode::Overflow);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionConfig {
    // ── 开关 ──

    /// 是否启用自动压缩。
    ///
    /// false 时仅支持手动触发（如 `/compact` 命令）。
    /// 三个参考项目都默认开启。
    pub auto_enabled: bool,

    // ── 触发 ──

    /// 触发模式 — 何时启动自动压缩。
    pub trigger_mode: CompactionTriggerMode,

    /// 阈值配置 — 在 Threshold 模式下生效。
    pub threshold: CompactionThreshold,

    /// 为输出预留的 token 缓冲。
    ///
    /// 阈值 fallback 计算: threshold = context_window - reserve_tokens。
    /// 同时确保压缩过程本身不会因为摘要输出而 overflow。
    ///
    /// - oh-my-pi: 16,384
    /// - opencode: min(20,000, max_output_tokens)
    /// - claude-code: 13,000 (auto) / 3,000 (manual)
    pub reserve_tokens: u64,

    // ── 保留策略 ──

    /// 消息保留策略 — 压缩时哪些近期内容保持原文不总结。
    pub preserve: PreserveConfig,

    // ── 修剪 ──

    /// 旧工具输出修剪配置。
    ///
    /// 独立于压缩的轻量级优化：截断旧工具调用的输出内容，
    /// 释放 token 空间，延迟全量压缩的触发。
    /// 来源: opencode 的 prune 机制。
    pub prune: PruneConfig,

    // ── 策略 ──

    /// 压缩策略 — 如何处理旧消息。
    pub strategy: CompactionStrategy,

    // ── 行为 ──

    /// 压缩完成后是否自动继续 Agent loop。
    ///
    /// true: 压缩后自动发送 "continue" 消息继续执行。
    /// false: 压缩后等待用户输入。
    /// oh-my-pi 和 opencode 都默认 true。
    pub auto_continue: bool,

    /// 连续失败熔断次数。
    ///
    /// 连续 N 次自动压缩失败后停止尝试，防止无限循环。
    /// 手动压缩不受此限制。
    /// 来源: claude-code 的 MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3。
    /// 0 = 不限制。
    pub max_consecutive_failures: u32,

    // ── 压缩模型 ──

    /// 用于执行压缩摘要的模型。
    ///
    /// None = 使用 Agent 当前的主模型。
    /// 来源: opencode 有独立的 "compaction" agent 配置。
    pub model: Option<AgentModelRef>,

    // ── 摘要输出 ──

    /// 摘要的最大输出 token 数。
    ///
    /// 限制 LLM 生成摘要时的输出长度。
    /// 来源: claude-code MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20,000。
    pub summary_max_tokens: Option<u32>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            auto_enabled: true,
            trigger_mode: CompactionTriggerMode::default(),
            threshold: CompactionThreshold::default(),
            reserve_tokens: 16_384,
            preserve: PreserveConfig::default(),
            prune: PruneConfig::default(),
            strategy: CompactionStrategy::default(),
            auto_continue: true,
            max_consecutive_failures: 3,
            model: None,
            summary_max_tokens: Some(20_000),
        }
    }
}

impl CompactionConfig {
    /// 设置自动压缩开关。
    pub fn with_auto_enabled(mut self, enabled: bool) -> Self {
        self.auto_enabled = enabled;
        self
    }

    /// 设置触发模式。
    pub fn with_trigger_mode(mut self, mode: CompactionTriggerMode) -> Self {
        self.trigger_mode = mode;
        self
    }

    /// 设置阈值配置。
    pub fn with_threshold(mut self, threshold: CompactionThreshold) -> Self {
        self.threshold = threshold;
        self
    }

    /// 设置预留 token 数。
    pub fn with_reserve_tokens(mut self, tokens: u64) -> Self {
        self.reserve_tokens = tokens;
        self
    }

    /// 设置消息保留策略。
    pub fn with_preserve(mut self, preserve: PreserveConfig) -> Self {
        self.preserve = preserve;
        self
    }

    /// 设置修剪配置。
    pub fn with_prune(mut self, prune: PruneConfig) -> Self {
        self.prune = prune;
        self
    }

    /// 设置压缩策略。
    pub fn with_strategy(mut self, strategy: CompactionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// 设置是否压缩后自动继续。
    pub fn with_auto_continue(mut self, auto_continue: bool) -> Self {
        self.auto_continue = auto_continue;
        self
    }

    /// 设置连续失败熔断次数。
    pub fn with_max_consecutive_failures(mut self, max: u32) -> Self {
        self.max_consecutive_failures = max;
        self
    }

    /// 设置压缩模型。
    pub fn with_model(mut self, model: AgentModelRef) -> Self {
        self.model = Some(model);
        self
    }

    /// 设置摘要最大输出 token 数。
    pub fn with_summary_max_tokens(mut self, tokens: u32) -> Self {
        self.summary_max_tokens = Some(tokens);
        self
    }
}

// ===========================================================================
// CompactionTriggerMode
// ===========================================================================

/// 压缩触发模式 — 决定何时启动自动压缩。
///
/// 两种模式对应不同的产品哲学：
/// - `Threshold`: 主动式 — 接近上限时提前压缩，避免 overflow（oh-my-pi, claude-code）
/// - `Overflow`: 被动式 — 仅在实际溢出时才压缩，最大化上下文利用率（opencode）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionTriggerMode {
    /// 达到阈值时主动压缩。
    ///
    /// 在 token 用量超过阈值（百分比或固定值）时触发，
    /// 留出足够空间完成当前对话而不 overflow。
    /// 这是 oh-my-pi 和 claude-code 的做法。
    #[default]
    Threshold,

    /// 仅在 context overflow 时被动压缩。
    ///
    /// 不提前触发，等到 LLM 返回 prompt-too-long 错误时才压缩。
    /// 最大化上下文利用率，但用户可能感知到短暂中断。
    /// 这是 opencode 的做法。
    Overflow,
}

// ===========================================================================
// CompactionThreshold
// ===========================================================================

/// 压缩阈值配置 — 控制在 Threshold 模式下何时触发。
///
/// ## 解析优先级（与 oh-my-pi 一致）
/// ```text
/// tokens (固定值) > ratio (百分比) > fallback (context_window - reserve_tokens)
/// ```
///
/// # Examples
///
/// ```
/// use katu_core::compaction::CompactionThreshold;
///
/// // 固定阈值: 超过 150K tokens 时触发
/// let fixed = CompactionThreshold::fixed(150_000);
///
/// // 百分比阈值: 超过 context window 的 85% 时触发
/// let ratio = CompactionThreshold::ratio(0.85);
///
/// // 默认: 都不设 — 使用 fallback (context_window - reserve_tokens)
/// let fallback = CompactionThreshold::default();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct CompactionThreshold {
    /// 固定 token 阈值 — 优先级最高。
    ///
    /// 当 context_tokens > tokens 时触发压缩。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u64>,

    /// 百分比阈值 (0.0 ~ 1.0) — tokens 未设置时使用。
    ///
    /// 当 context_tokens > context_window * ratio 时触发。
    /// 来源: oh-my-pi thresholdPercent。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ratio: Option<f64>,
}

impl CompactionThreshold {
    /// 创建固定 token 阈值。
    pub fn fixed(tokens: u64) -> Self {
        Self {
            tokens: Some(tokens),
            ratio: None,
        }
    }

    /// 创建百分比阈值。
    pub fn ratio(ratio: f64) -> Self {
        Self {
            tokens: None,
            ratio: Some(ratio),
        }
    }

    /// 解析最终阈值 token 数。
    ///
    /// ## 优先级
    /// 1. `self.tokens` — 固定值，直接返回（clamp 到 [1, context_window-1]）
    /// 2. `self.ratio` — 百分比，返回 `context_window * ratio`
    /// 3. fallback — `context_window - reserve_tokens`
    pub fn resolve(&self, context_window: u64, reserve_tokens: u64) -> u64 {
        // 固定值优先
        if let Some(tokens) = self.tokens {
            return tokens.clamp(1, context_window.saturating_sub(1));
        }

        // 百分比
        if let Some(ratio) = self.ratio {
            let clamped = ratio.clamp(0.01, 0.99);
            return (context_window as f64 * clamped) as u64;
        }

        // Fallback: context_window - max(reserve_tokens, 15% of window)
        // 与 oh-my-pi 的 effectiveReserveTokens 一致
        let effective_reserve = reserve_tokens.max((context_window as f64 * 0.15) as u64);
        context_window.saturating_sub(effective_reserve)
    }
}

// ===========================================================================
// PreserveConfig
// ===========================================================================

/// 消息保留策略 — 压缩时保留哪些近期内容不总结。
///
/// 保留的消息保持原始内容，不被 LLM 重新摘要。
/// 这对保留最近的工具调用上下文、用户指令尤其重要。
///
/// ## 两种维度
/// - **turns** — 按 user turn 数量保留（opencode tail_turns=2）
/// - **tokens** — 按 token 预算保留（oh-my-pi keepRecentTokens=20K）
///
/// 两者取 **交集**：先按 turns 选出候选，再按 tokens 预算裁剪。
///
/// # Examples
///
/// ```
/// use katu_core::compaction::PreserveConfig;
///
/// // 保留最近 2 个 user turn，最多 8K tokens
/// let config = PreserveConfig::new(2, 8_000);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreserveConfig {
    /// 保留最近 N 个 user turn（含其后续的 assistant/tool 回复）。
    ///
    /// 0 = 不按 turn 保留。
    /// 来源: opencode DEFAULT_TAIL_TURNS = 2。
    pub recent_turns: u32,

    /// 保留最近内容的 token 预算上限。
    ///
    /// None = 自动计算（usable_tokens * 0.25，clamp 到 2K~8K）。
    /// 来源: oh-my-pi keepRecentTokens=20K, opencode preserve_recent_tokens。
    pub recent_tokens: Option<u64>,
}

impl Default for PreserveConfig {
    fn default() -> Self {
        Self {
            recent_turns: 2,
            recent_tokens: None, // 自动计算
        }
    }
}

impl PreserveConfig {
    /// 创建保留配置。
    pub fn new(recent_turns: u32, recent_tokens: u64) -> Self {
        Self {
            recent_turns,
            recent_tokens: Some(recent_tokens),
        }
    }

    /// 设置 turn 数量。
    pub fn with_recent_turns(mut self, turns: u32) -> Self {
        self.recent_turns = turns;
        self
    }

    /// 设置 token 预算。
    pub fn with_recent_tokens(mut self, tokens: u64) -> Self {
        self.recent_tokens = Some(tokens);
        self
    }

    /// 解析最终的保留 token 预算。
    ///
    /// 如果 `recent_tokens` 已设置，直接返回。
    /// 否则自动计算：`usable_tokens * 0.25`，clamp 到 [min, max]。
    ///
    /// # Arguments
    /// - `usable_tokens`: 可用 token 数（context_window - reserve - output）
    /// - `min`: 最小保留（默认 2,000）
    /// - `max`: 最大保留（默认 8,000）
    pub fn resolve_recent_tokens(&self, usable_tokens: u64, min: u64, max: u64) -> u64 {
        self.recent_tokens.unwrap_or_else(|| {
            let auto = (usable_tokens as f64 * 0.25) as u64;
            auto.clamp(min, max)
        })
    }
}

// ===========================================================================
// PruneConfig
// ===========================================================================

/// 旧工具输出修剪配置。
///
/// Prune 是一种**轻量级**的上下文优化手段，独立于全量压缩：
/// 把旧的、体积大的工具输出内容截断或标记为已压缩，
/// 释放 token 空间，延迟全量压缩的触发。
///
/// ## 算法（来源: opencode）
/// 1. 从最新消息向旧遍历，跳过最近 2 个 user turn
/// 2. 累计 tool output tokens，超过 `protect_tokens` 后开始标记
/// 3. 仅在总修剪量超过 `minimum_tokens` 时才实际执行
///
/// # Examples
///
/// ```
/// use katu_core::compaction::PruneConfig;
///
/// let config = PruneConfig::default();
/// assert!(config.enabled);
/// assert_eq!(config.protect_tokens, 40_000);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PruneConfig {
    /// 是否启用修剪。
    pub enabled: bool,

    /// 保护最近的 tool output token 数不被修剪。
    ///
    /// 从最新往旧遍历，累计超过此值后才开始标记修剪。
    /// 来源: opencode PRUNE_PROTECT = 40,000。
    pub protect_tokens: u64,

    /// 修剪的最小触发阈值。
    ///
    /// 仅当可修剪量超过此值时才执行，避免无意义的小量修剪。
    /// 来源: opencode PRUNE_MINIMUM = 20,000。
    pub minimum_tokens: u64,

    /// 修剪时工具输出截断的最大字符数。
    ///
    /// 超过此长度的工具输出在修剪时被截断。
    /// 来源: opencode TOOL_OUTPUT_MAX_CHARS = 2,000。
    pub tool_output_max_chars: usize,

    /// 不受修剪影响的工具名称列表。
    ///
    /// 某些工具（如 skill）的输出对上下文非常重要，不应被修剪。
    /// 来源: opencode PRUNE_PROTECTED_TOOLS = ["skill"]。
    #[serde(default)]
    pub protected_tools: Vec<String>,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            protect_tokens: 40_000,
            minimum_tokens: 20_000,
            tool_output_max_chars: 2_000,
            protected_tools: Vec::new(),
        }
    }
}

impl PruneConfig {
    /// 设置修剪开关。
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// 设置保护 token 数。
    pub fn with_protect_tokens(mut self, tokens: u64) -> Self {
        self.protect_tokens = tokens;
        self
    }

    /// 设置最小触发阈值。
    pub fn with_minimum_tokens(mut self, tokens: u64) -> Self {
        self.minimum_tokens = tokens;
        self
    }

    /// 设置工具输出截断字符数。
    pub fn with_tool_output_max_chars(mut self, chars: usize) -> Self {
        self.tool_output_max_chars = chars;
        self
    }

    /// 添加受保护的工具。
    pub fn add_protected_tool(mut self, tool: impl Into<String>) -> Self {
        self.protected_tools.push(tool.into());
        self
    }
}

// ===========================================================================
// CompactionStrategy
// ===========================================================================

/// 压缩策略 — 旧消息如何被处理。
///
/// # Examples
///
/// ```
/// use katu_core::compaction::CompactionStrategy;
///
/// let strategy = CompactionStrategy::Summarize;
/// assert!(strategy.is_summarize());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionStrategy {
    /// 用 LLM 总结旧消息，就地替换为摘要。
    ///
    /// 最常见的策略，三个项目都支持。
    /// 旧消息被丢弃，摘要作为新的 system/user message 注入。
    #[default]
    Summarize,

    /// 生成 handoff 文档，开始新会话。
    ///
    /// 将旧对话总结为一个完整的 "交接文档"，然后开启新 session。
    /// 来源: oh-my-pi strategy="handoff"。
    Handoff,
}

impl CompactionStrategy {
    /// 是否为 Summarize 策略。
    pub fn is_summarize(&self) -> bool {
        matches!(self, Self::Summarize)
    }

    /// 是否为 Handoff 策略。
    pub fn is_handoff(&self) -> bool {
        matches!(self, Self::Handoff)
    }
}

// ===========================================================================
// CompactionResult
// ===========================================================================

/// 压缩执行结果 — 一次压缩操作完成后的数据。
///
/// 由 `katu-agent` 层的压缩逻辑产出，用于：
/// - `AgentEvent::CompactionEnded` 事件
/// - 持久化到 session 历史
/// - UI 展示压缩效果
///
/// # Examples
///
/// ```
/// use katu_core::compaction::{CompactionResult, CompactTrigger};
///
/// let result = CompactionResult {
///     summary: "User asked about Rust ownership...".into(),
///     short_summary: Some("Discussed Rust ownership".into()),
///     trigger: CompactTrigger::Auto,
///     tokens_before: 150_000,
///     tokens_after: Some(5_000),
///     messages_compacted: 42,
///     messages_kept: 8,
///     success: true,
/// };
/// assert!(result.success);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionResult {
    /// 压缩生成的完整摘要文本。
    pub summary: String,

    /// 短摘要（用于 UI 显示，类似 PR title）。
    ///
    /// 来源: oh-my-pi shortSummary。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub short_summary: Option<String>,

    /// 触发原因。
    pub trigger: CompactTrigger,

    /// 压缩前的 prompt token 数。
    pub tokens_before: u64,

    /// 压缩后的估计 token 数。
    ///
    /// None = 未测量。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_after: Option<u64>,

    /// 被压缩掉的消息数。
    pub messages_compacted: usize,

    /// 保留不变的消息数（recent turns）。
    pub messages_kept: usize,

    /// 是否成功。
    ///
    /// false 时 summary 可能包含错误信息。
    pub success: bool,
}

impl CompactionResult {
    /// 计算节省的 token 数。
    pub fn tokens_saved(&self) -> Option<u64> {
        self.tokens_after
            .map(|after| self.tokens_before.saturating_sub(after))
    }

    /// 计算压缩比 (0.0 ~ 1.0)。
    ///
    /// 0.0 = 完全没减少，1.0 = 全部压缩掉。
    pub fn compression_ratio(&self) -> Option<f64> {
        self.tokens_after.map(|after| {
            if self.tokens_before == 0 {
                return 0.0;
            }
            1.0 - (after as f64 / self.tokens_before as f64)
        })
    }
}

// ===========================================================================
// CompactTrigger (moved from agent_event)
// ===========================================================================

/// 上下文压缩触发方式。
///
/// 用于 `AgentEvent::CompactionStarted` 和 `CompactionResult`，
/// 标识本次压缩是如何被触发的。
///
/// # Examples
///
/// ```
/// use katu_core::compaction::CompactTrigger;
///
/// let trigger = CompactTrigger::Auto;
/// assert!(trigger.is_auto());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactTrigger {
    /// 自动触发 — token 用量超过阈值。
    Auto,

    /// 用户手动触发 — 如 `/compact` 命令。
    Manual,

    /// Overflow 触发 — LLM 返回 prompt-too-long。
    ///
    /// 与 Auto 不同：Auto 是提前预防，Overflow 是事后补救。
    /// 来源: opencode overflow 标志、oh-my-pi "overflow" reason。
    Overflow,

    /// 空闲触发 — 用户一段时间无操作后预压缩。
    ///
    /// 来源: oh-my-pi idleEnabled + idleTimeoutSeconds。
    Idle,
}

impl CompactTrigger {
    /// 是否为自动触发（Auto 或 Overflow 或 Idle）。
    pub fn is_auto(&self) -> bool {
        !matches!(self, Self::Manual)
    }

    /// 是否为手动触发。
    pub fn is_manual(&self) -> bool {
        matches!(self, Self::Manual)
    }
}

// ===========================================================================
// TokenBudgetState
// ===========================================================================

/// Token 用量状态 — 当前上下文占用量的分级警告。
///
/// UI 层用此枚举渲染进度条颜色和警告提示。
/// Agent loop 用此判断是否触发自动压缩。
///
/// ## 阈值对照（来源: claude-code）
/// ```text
/// |------ Normal ------|-- Warning --|-- Error --|-- Blocking --|
/// 0%                 ~70%          ~85%        ~95%          100%
/// ```
///
/// # Examples
///
/// ```
/// use katu_core::compaction::TokenBudgetState;
///
/// let state = TokenBudgetState::from_usage(150_000, 200_000, 13_000);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "level", rename_all = "snake_case")]
pub enum TokenBudgetState {
    /// 正常 — 充足余量。
    Normal {
        /// 剩余可用百分比 (0.0 ~ 1.0)。
        percent_remaining: f64,
    },

    /// 警告 — 接近阈值，UI 显示黄色提示。
    Warning {
        percent_remaining: f64,
    },

    /// 危险 — 非常接近上限，UI 显示红色提示。
    Error {
        percent_remaining: f64,
    },

    /// 阻塞 — 已达到上限，应阻止新消息发送。
    Blocking,
}

impl TokenBudgetState {
    /// 根据当前 token 用量计算状态。
    ///
    /// # Arguments
    /// - `used_tokens`: 当前已使用的 token 数
    /// - `context_window`: 模型 context window 大小
    /// - `auto_compact_buffer`: 自动压缩缓冲区大小（reserve_tokens）
    ///
    /// # 阈值计算
    /// ```text
    /// effective_window = context_window - summary_reserve (通常 20K)
    /// auto_compact_threshold = effective_window - auto_compact_buffer
    /// warning_threshold = auto_compact_threshold - 20K
    /// error_threshold = effective_window - 20K
    /// ```
    pub fn from_usage(
        used_tokens: u64,
        context_window: u64,
        auto_compact_buffer: u64,
    ) -> Self {
        if context_window == 0 {
            return Self::Blocking;
        }

        let percent_remaining = 1.0 - (used_tokens as f64 / context_window as f64);

        // 阻塞: 已达到或超过 context window
        if used_tokens >= context_window {
            return Self::Blocking;
        }

        // 自动压缩阈值
        let auto_threshold = context_window.saturating_sub(auto_compact_buffer);

        // 错误阈值: 距离 context window 20K
        let error_threshold = context_window.saturating_sub(20_000);

        // 警告阈值: 距离自动压缩阈值 20K
        let warning_threshold = auto_threshold.saturating_sub(20_000);

        if used_tokens >= error_threshold {
            Self::Error { percent_remaining }
        } else if used_tokens >= warning_threshold {
            Self::Warning { percent_remaining }
        } else {
            Self::Normal { percent_remaining }
        }
    }

    /// 是否应触发自动压缩。
    pub fn should_auto_compact(&self) -> bool {
        matches!(self, Self::Error { .. } | Self::Blocking)
    }

    /// 是否应阻止新消息发送。
    pub fn is_blocking(&self) -> bool {
        matches!(self, Self::Blocking)
    }

    /// 是否处于警告或更严重状态。
    pub fn is_warning_or_worse(&self) -> bool {
        !matches!(self, Self::Normal { .. })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- CompactionConfig --

    #[test]
    fn test_default_config() {
        let config = CompactionConfig::default();
        assert!(config.auto_enabled);
        assert_eq!(config.trigger_mode, CompactionTriggerMode::Threshold);
        assert_eq!(config.reserve_tokens, 16_384);
        assert_eq!(config.preserve.recent_turns, 2);
        assert!(config.prune.enabled);
        assert_eq!(config.strategy, CompactionStrategy::Summarize);
        assert!(config.auto_continue);
        assert_eq!(config.max_consecutive_failures, 3);
        assert!(config.model.is_none());
        assert_eq!(config.summary_max_tokens, Some(20_000));
    }

    #[test]
    fn test_config_builder() {
        let config = CompactionConfig::default()
            .with_auto_enabled(false)
            .with_trigger_mode(CompactionTriggerMode::Overflow)
            .with_reserve_tokens(20_000)
            .with_auto_continue(false)
            .with_max_consecutive_failures(5);

        assert!(!config.auto_enabled);
        assert_eq!(config.trigger_mode, CompactionTriggerMode::Overflow);
        assert_eq!(config.reserve_tokens, 20_000);
        assert!(!config.auto_continue);
        assert_eq!(config.max_consecutive_failures, 5);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = CompactionConfig::default()
            .with_strategy(CompactionStrategy::Handoff)
            .with_prune(PruneConfig::default().with_enabled(false));

        let json = serde_json::to_string(&config).unwrap();
        let restored: CompactionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, restored);
    }

    // -- CompactionThreshold --

    #[test]
    fn test_threshold_fixed() {
        let t = CompactionThreshold::fixed(150_000);
        assert_eq!(t.resolve(200_000, 16_384), 150_000);
    }

    #[test]
    fn test_threshold_fixed_clamp() {
        let t = CompactionThreshold::fixed(300_000);
        // clamp to context_window - 1
        assert_eq!(t.resolve(200_000, 16_384), 199_999);
    }

    #[test]
    fn test_threshold_ratio() {
        let t = CompactionThreshold::ratio(0.85);
        assert_eq!(t.resolve(200_000, 16_384), 170_000);
    }

    #[test]
    fn test_threshold_fallback() {
        let t = CompactionThreshold::default();
        // fallback = context_window - max(reserve, 15% of window)
        // max(16_384, 200_000 * 0.15 = 30_000) = 30_000
        // 200_000 - 30_000 = 170_000
        assert_eq!(t.resolve(200_000, 16_384), 170_000);
    }

    #[test]
    fn test_threshold_fallback_small_window() {
        let t = CompactionThreshold::default();
        // max(16_384, 50_000 * 0.15 = 7_500) = 16_384
        // 50_000 - 16_384 = 33_616
        assert_eq!(t.resolve(50_000, 16_384), 33_616);
    }

    // -- PreserveConfig --

    #[test]
    fn test_preserve_default() {
        let p = PreserveConfig::default();
        assert_eq!(p.recent_turns, 2);
        assert!(p.recent_tokens.is_none());
    }

    #[test]
    fn test_preserve_resolve_auto() {
        let p = PreserveConfig::default();
        // usable = 100_000, auto = 25_000, clamp to [2K, 8K] => 8_000
        assert_eq!(p.resolve_recent_tokens(100_000, 2_000, 8_000), 8_000);
        // usable = 4_000, auto = 1_000, clamp to [2K, 8K] => 2_000
        assert_eq!(p.resolve_recent_tokens(4_000, 2_000, 8_000), 2_000);
        // usable = 20_000, auto = 5_000, clamp to [2K, 8K] => 5_000
        assert_eq!(p.resolve_recent_tokens(20_000, 2_000, 8_000), 5_000);
    }

    #[test]
    fn test_preserve_resolve_explicit() {
        let p = PreserveConfig::new(3, 15_000);
        // explicit 值直接返回，不受 clamp 影响
        assert_eq!(p.resolve_recent_tokens(100_000, 2_000, 8_000), 15_000);
    }

    // -- PruneConfig --

    #[test]
    fn test_prune_default() {
        let p = PruneConfig::default();
        assert!(p.enabled);
        assert_eq!(p.protect_tokens, 40_000);
        assert_eq!(p.minimum_tokens, 20_000);
        assert_eq!(p.tool_output_max_chars, 2_000);
        assert!(p.protected_tools.is_empty());
    }

    #[test]
    fn test_prune_builder() {
        let p = PruneConfig::default()
            .with_enabled(false)
            .with_protect_tokens(50_000)
            .add_protected_tool("skill")
            .add_protected_tool("memory");

        assert!(!p.enabled);
        assert_eq!(p.protect_tokens, 50_000);
        assert_eq!(p.protected_tools, vec!["skill", "memory"]);
    }

    // -- CompactionStrategy --

    #[test]
    fn test_strategy_predicates() {
        assert!(CompactionStrategy::Summarize.is_summarize());
        assert!(!CompactionStrategy::Summarize.is_handoff());
        assert!(CompactionStrategy::Handoff.is_handoff());
        assert!(!CompactionStrategy::Handoff.is_summarize());
    }

    #[test]
    fn test_strategy_serde() {
        let json = serde_json::to_string(&CompactionStrategy::Handoff).unwrap();
        assert_eq!(json, r#""handoff""#);
        let restored: CompactionStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, CompactionStrategy::Handoff);
    }

    // -- CompactTrigger --

    #[test]
    fn test_trigger_is_auto() {
        assert!(CompactTrigger::Auto.is_auto());
        assert!(CompactTrigger::Overflow.is_auto());
        assert!(CompactTrigger::Idle.is_auto());
        assert!(!CompactTrigger::Manual.is_auto());
    }

    #[test]
    fn test_trigger_serde() {
        for trigger in [
            CompactTrigger::Auto,
            CompactTrigger::Manual,
            CompactTrigger::Overflow,
            CompactTrigger::Idle,
        ] {
            let json = serde_json::to_string(&trigger).unwrap();
            let restored: CompactTrigger = serde_json::from_str(&json).unwrap();
            assert_eq!(trigger, restored);
        }
    }

    // -- CompactionResult --

    #[test]
    fn test_result_tokens_saved() {
        let result = CompactionResult {
            summary: "test".into(),
            short_summary: None,
            trigger: CompactTrigger::Auto,
            tokens_before: 150_000,
            tokens_after: Some(5_000),
            messages_compacted: 40,
            messages_kept: 8,
            success: true,
        };
        assert_eq!(result.tokens_saved(), Some(145_000));
    }

    #[test]
    fn test_result_compression_ratio() {
        let result = CompactionResult {
            summary: "test".into(),
            short_summary: None,
            trigger: CompactTrigger::Auto,
            tokens_before: 100_000,
            tokens_after: Some(10_000),
            messages_compacted: 30,
            messages_kept: 5,
            success: true,
        };
        let ratio = result.compression_ratio().unwrap();
        assert!((ratio - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_result_no_tokens_after() {
        let result = CompactionResult {
            summary: "test".into(),
            short_summary: None,
            trigger: CompactTrigger::Manual,
            tokens_before: 100_000,
            tokens_after: None,
            messages_compacted: 20,
            messages_kept: 5,
            success: true,
        };
        assert!(result.tokens_saved().is_none());
        assert!(result.compression_ratio().is_none());
    }

    #[test]
    fn test_result_serde_roundtrip() {
        let result = CompactionResult {
            summary: "The user asked about Rust ownership...".into(),
            short_summary: Some("Discussed Rust ownership".into()),
            trigger: CompactTrigger::Overflow,
            tokens_before: 180_000,
            tokens_after: Some(8_000),
            messages_compacted: 50,
            messages_kept: 6,
            success: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        let restored: CompactionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, restored);
    }

    // -- TokenBudgetState --

    #[test]
    fn test_budget_state_normal() {
        let state = TokenBudgetState::from_usage(50_000, 200_000, 13_000);
        assert!(matches!(state, TokenBudgetState::Normal { .. }));
        assert!(!state.should_auto_compact());
        assert!(!state.is_blocking());
        assert!(!state.is_warning_or_worse());
    }

    #[test]
    fn test_budget_state_warning() {
        // warning threshold = (200K - 13K) - 20K = 167K
        let state = TokenBudgetState::from_usage(170_000, 200_000, 13_000);
        assert!(matches!(state, TokenBudgetState::Warning { .. }));
        assert!(!state.should_auto_compact());
        assert!(state.is_warning_or_worse());
    }

    #[test]
    fn test_budget_state_error() {
        // error threshold = 200K - 20K = 180K
        let state = TokenBudgetState::from_usage(185_000, 200_000, 13_000);
        assert!(matches!(state, TokenBudgetState::Error { .. }));
        assert!(state.should_auto_compact());
        assert!(state.is_warning_or_worse());
    }

    #[test]
    fn test_budget_state_blocking() {
        let state = TokenBudgetState::from_usage(200_000, 200_000, 13_000);
        assert!(matches!(state, TokenBudgetState::Blocking));
        assert!(state.should_auto_compact());
        assert!(state.is_blocking());
    }

    #[test]
    fn test_budget_state_zero_window() {
        let state = TokenBudgetState::from_usage(0, 0, 0);
        assert!(matches!(state, TokenBudgetState::Blocking));
    }

    #[test]
    fn test_budget_state_serde_roundtrip() {
        let states = vec![
            TokenBudgetState::Normal {
                percent_remaining: 0.75,
            },
            TokenBudgetState::Warning {
                percent_remaining: 0.15,
            },
            TokenBudgetState::Error {
                percent_remaining: 0.05,
            },
            TokenBudgetState::Blocking,
        ];
        for state in states {
            let json = serde_json::to_string(&state).unwrap();
            let restored: TokenBudgetState = serde_json::from_str(&json).unwrap();
            assert_eq!(state, restored);
        }
    }
}
