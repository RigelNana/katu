//! # retry
//!
//! ## 职责
//! 定义 Agent 执行循环的重试策略 — 配置 (`RetryConfig`) + 执行状态 (`RetryState`)。
//!
//! ## 设计
//! - `RetryConfig` — 纯配置，描述退避参数
//! - `RetryState` — 有状态的重试控制器，跟踪已重试次数并计算下次退避
//!
//! 退避策略采用截断指数退避 + 等概率全域抖动（full jitter）：
//! ```text
//! base = min(base_delay * 2^attempt, max_delay)
//! delay = uniform(base/2, base)
//! ```
//!
//! 当 Provider 返回 `retry_after` 时，使用 Provider 指定的时间（不低于计算值）。
//!
//! ## 调用者
//! - `katu-agent::instance` — `RunConfig` 持有 `RetryConfig`
//! - `katu-agent::runner` (future) — 通过 `RetryState` 驱动重试

use std::time::Duration;

use rand::Rng;

// ===========================================================================
// RetryConfig
// ===========================================================================

/// 重试策略配置 — 控制 Agent loop 对可重试错误的退避行为。
///
/// 默认配置适用于大多数 LLM Provider：
/// - 最多重试 3 次
/// - 初始退避 1 秒
/// - 最大退避 60 秒
///
/// # Examples
///
/// ```
/// use katu_agent::retry::RetryConfig;
/// use std::time::Duration;
///
/// // 使用默认配置
/// let config = RetryConfig::default();
/// assert_eq!(config.max_retries(), 3);
///
/// // 自定义配置
/// let config = RetryConfig::new()
///     .with_max_retries(5)
///     .with_base_delay(Duration::from_millis(500))
///     .with_max_delay(Duration::from_secs(120));
///
/// assert_eq!(config.max_retries(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// 最大重试次数（不含首次尝试）。
    max_retries: u32,

    /// 初始退避时间。
    base_delay: Duration,

    /// 最大退避时间（截断上限）。
    max_delay: Duration,
}

/// 默认最大重试次数。
const DEFAULT_MAX_RETRIES: u32 = 3;

/// 默认初始退避 — 1 秒。
const DEFAULT_BASE_DELAY: Duration = Duration::from_secs(1);

/// 默认最大退避 — 60 秒。
const DEFAULT_MAX_DELAY: Duration = Duration::from_secs(60);

impl RetryConfig {
    /// 创建默认重试配置。
    pub fn new() -> Self {
        Self::default()
    }

    /// 禁用重试 — `max_retries = 0`。
    pub fn disabled() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    /// 设置最大重试次数。
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// 设置初始退避时间。
    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// 设置最大退避时间。
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }
}

// ---------------------------------------------------------------------------
// 读取方法
// ---------------------------------------------------------------------------

impl RetryConfig {
    /// 最大重试次数。
    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// 初始退避时间。
    pub fn base_delay(&self) -> Duration {
        self.base_delay
    }

    /// 最大退避时间。
    pub fn max_delay(&self) -> Duration {
        self.max_delay
    }

    /// 是否启用重试（`max_retries > 0`）。
    pub fn is_enabled(&self) -> bool {
        self.max_retries > 0
    }

    /// 创建基于此配置的 `RetryState`。
    pub fn into_state(self) -> RetryState {
        RetryState::new(self)
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: DEFAULT_MAX_RETRIES,
            base_delay: DEFAULT_BASE_DELAY,
            max_delay: DEFAULT_MAX_DELAY,
        }
    }
}

// ===========================================================================
// RetryState
// ===========================================================================

/// 重试控制器 — 有状态，跟踪重试进度并计算退避时间。
///
/// 由 `RetryConfig::into_state()` 创建，每次 Agent run 使用独立的 state。
///
/// # Examples
///
/// ```
/// use katu_agent::retry::{RetryConfig, RetryState};
/// use std::time::Duration;
///
/// let mut state = RetryConfig::new()
///     .with_max_retries(3)
///     .with_base_delay(Duration::from_secs(1))
///     .into_state();
///
/// assert_eq!(state.attempt(), 0);
///
/// // 第一次重试
/// if let Some(delay) = state.next_delay(None) {
///     assert!(delay >= Duration::from_millis(500));
///     assert!(delay <= Duration::from_secs(1));
/// }
/// assert_eq!(state.attempt(), 1);
///
/// // 用完重试次数后返回 None
/// state.next_delay(None);
/// state.next_delay(None);
/// assert!(state.next_delay(None).is_none());
/// ```
#[derive(Debug, Clone)]
pub struct RetryState {
    /// 关联的配置。
    config: RetryConfig,
    /// 已重试次数（0 = 尚未重试）。
    attempt: u32,
}

impl RetryState {
    /// 从配置创建初始状态。
    pub fn new(config: RetryConfig) -> Self {
        Self { config, attempt: 0 }
    }

    /// 当前已重试次数。
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// 关联的配置引用。
    pub fn config(&self) -> &RetryConfig {
        &self.config
    }

    /// 是否还有重试机会。
    pub fn has_remaining(&self) -> bool {
        self.attempt < self.config.max_retries
    }

    /// 计算下次退避时间并推进计数器。
    ///
    /// - 返回 `Some(delay)` — 应等待 `delay` 后重试
    /// - 返回 `None` — 已用尽重试次数
    ///
    /// ## `retry_after` 参数
    /// Provider 通过 `Retry-After` header 指定的最小等待时间。
    /// 当 Provider 给出的值大于计算值时，使用 Provider 的值（但不超过 max_delay）。
    pub fn next_delay(&mut self, retry_after: Option<Duration>) -> Option<Duration> {
        if !self.has_remaining() {
            return None;
        }

        let delay = self.compute_delay(self.attempt, retry_after);
        self.attempt += 1;
        Some(delay)
    }

    /// 重置状态（新一轮重试场景）。
    pub fn reset(&mut self) {
        self.attempt = 0;
    }

    /// 计算退避时间 — 截断指数退避 + full jitter，尊重 retry_after。
    fn compute_delay(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        // 指数退避：base_delay * 2^attempt，截断到 max_delay
        let multiplier = 2u64.saturating_pow(attempt);
        let base_nanos = self.config.base_delay.as_nanos() as u64;
        let exp_nanos = base_nanos.saturating_mul(multiplier);
        let max_nanos = self.config.max_delay.as_nanos() as u64;
        let capped_nanos = exp_nanos.min(max_nanos);

        // Full jitter: uniform [capped/2, capped]
        let half = capped_nanos / 2;
        let jitter_range = capped_nanos - half;
        let jitter = if jitter_range > 0 {
            rand::rng().random_range(0..jitter_range)
        } else {
            0
        };
        let delay_nanos = half + jitter;

        let delay = Duration::from_nanos(delay_nanos);

        // 尊重 Provider 的 retry_after（取两者较大值，但不超过 max_delay）
        match retry_after {
            Some(ra) => delay.max(ra).min(self.config.max_delay),
            None => delay,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ─── RetryConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries(), 3);
        assert_eq!(config.base_delay(), Duration::from_secs(1));
        assert_eq!(config.max_delay(), Duration::from_secs(60));
        assert!(config.is_enabled());
    }

    #[test]
    fn test_config_disabled() {
        let config = RetryConfig::disabled();
        assert_eq!(config.max_retries(), 0);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_config_builder_chain() {
        let config = RetryConfig::new()
            .with_max_retries(5)
            .with_base_delay(Duration::from_millis(500))
            .with_max_delay(Duration::from_secs(120));

        assert_eq!(config.max_retries(), 5);
        assert_eq!(config.base_delay(), Duration::from_millis(500));
        assert_eq!(config.max_delay(), Duration::from_secs(120));
    }

    // ─── RetryState ─────────────────────────────────────────────────────────

    #[test]
    fn test_state_initial() {
        let state = RetryConfig::new().into_state();
        assert_eq!(state.attempt(), 0);
        assert!(state.has_remaining());
    }

    #[test]
    fn test_state_disabled_returns_none() {
        let mut state = RetryConfig::disabled().into_state();
        assert!(!state.has_remaining());
        assert!(state.next_delay(None).is_none());
    }

    #[test]
    fn test_state_exhaustion() {
        let mut state = RetryConfig::new()
            .with_max_retries(2)
            .into_state();

        assert!(state.next_delay(None).is_some()); // attempt 0 → 1
        assert!(state.next_delay(None).is_some()); // attempt 1 → 2
        assert!(state.next_delay(None).is_none()); // exhausted
        assert_eq!(state.attempt(), 2);
    }

    #[test]
    fn test_state_delay_in_jitter_range() {
        let mut state = RetryConfig::new()
            .with_max_retries(5)
            .with_base_delay(Duration::from_secs(2))
            .with_max_delay(Duration::from_secs(60))
            .into_state();

        // attempt 0: base = 2s, jitter range [1s, 2s]
        let delay = state.next_delay(None).unwrap();
        assert!(delay >= Duration::from_secs(1));
        assert!(delay <= Duration::from_secs(2));
    }

    #[test]
    fn test_state_exponential_growth() {
        // 多次采样验证指数增长趋势
        let config = RetryConfig::new()
            .with_max_retries(5)
            .with_base_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(60));

        // attempt 0: base=1s → [0.5s, 1s]
        // attempt 1: base=2s → [1s, 2s]
        // attempt 2: base=4s → [2s, 4s]
        // 验证上界不超过 base * 2^attempt
        for attempt in 0..5 {
            let mut state = config.clone().into_state();
            // 推进到目标 attempt
            for _ in 0..attempt {
                state.next_delay(None);
            }
            let delay = state.next_delay(None).unwrap();
            let max_expected = Duration::from_secs(1 << attempt).min(Duration::from_secs(60));
            assert!(delay <= max_expected, "attempt {attempt}: {delay:?} > {max_expected:?}");
        }
    }

    #[test]
    fn test_state_capped_at_max_delay() {
        let mut state = RetryConfig::new()
            .with_max_retries(10)
            .with_base_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(30))
            .into_state();

        // 推进到高 attempt，验证截断
        for _ in 0..8 {
            state.next_delay(None);
        }
        let delay = state.next_delay(None).unwrap();
        assert!(delay <= Duration::from_secs(30));
    }

    #[test]
    fn test_state_retry_after_respected() {
        let mut state = RetryConfig::new()
            .with_max_retries(3)
            .with_base_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(60))
            .into_state();

        // Provider 要求等 10 秒，大于计算值 [0.5s, 1s]
        let delay = state.next_delay(Some(Duration::from_secs(10))).unwrap();
        assert!(delay >= Duration::from_secs(10));
    }

    #[test]
    fn test_state_retry_after_capped_by_max() {
        let mut state = RetryConfig::new()
            .with_max_retries(3)
            .with_base_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(30))
            .into_state();

        // Provider 要求 120 秒，但 max_delay 是 30 秒
        let delay = state.next_delay(Some(Duration::from_secs(120))).unwrap();
        assert_eq!(delay, Duration::from_secs(30));
    }

    #[test]
    fn test_state_reset() {
        let mut state = RetryConfig::new()
            .with_max_retries(2)
            .into_state();

        state.next_delay(None);
        state.next_delay(None);
        assert!(!state.has_remaining());

        state.reset();
        assert_eq!(state.attempt(), 0);
        assert!(state.has_remaining());
    }

    #[test]
    fn test_state_overflow_safety() {
        let mut state = RetryConfig::new()
            .with_max_retries(40)
            .with_base_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(60))
            .into_state();

        // 推进到高 attempt（2^35 会溢出 u64 纳秒）
        for _ in 0..35 {
            state.next_delay(None);
        }
        let delay = state.next_delay(None).unwrap();
        assert!(delay <= Duration::from_secs(60));
    }
}
