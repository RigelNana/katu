//! # katu_core::generation
//!
//! ## 职责
//! 定义 provider 无关的 LLM 生成参数。
//!
//! ## 对外接口
//! - `GenerationOptions` — 生成控制参数

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// GenerationOptions
// ---------------------------------------------------------------------------

/// Provider 无关的 LLM 生成参数。
///
/// 所有字段均为 `Option`，`None` 表示使用 provider/model 默认值。
/// 支持多级合并：Request 级 > Agent 级 > Model 级 > Provider 默认。
///
/// # Examples
/// ```
/// use katu_core::GenerationOptions;
///
/// let opts = GenerationOptions::new()
///     .with_max_tokens(4096)
///     .with_temperature(0.7);
/// assert_eq!(opts.max_tokens, Some(4096));
/// assert_eq!(opts.temperature, Some(0.7));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationOptions {
    /// 最大输出 token 数
    pub max_tokens: Option<u32>,
    /// 采样温度 (0.0 = 确定性, 2.0 = 最大随机)
    pub temperature: Option<f32>,
    /// Top-p (nucleus) 采样阈值
    pub top_p: Option<f32>,
    /// Top-k 采样（部分 provider 支持）
    pub top_k: Option<u32>,
    /// 频率惩罚 (-2.0 ~ 2.0)
    pub frequency_penalty: Option<f32>,
    /// 存在惩罚 (-2.0 ~ 2.0)
    pub presence_penalty: Option<f32>,
    /// 停止序列
    pub stop: Option<Vec<String>>,
    /// 随机种子（可复现生成）
    pub seed: Option<u64>,
}

impl GenerationOptions {
    /// 创建空的 GenerationOptions（所有字段为 None）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置 max_tokens。
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// 设置 temperature。
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// 设置 top_p。
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// 设置 top_k。
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// 设置 frequency_penalty。
    pub fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// 设置 presence_penalty。
    pub fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// 设置停止序列。
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// 设置随机种子。
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// 合并两个 GenerationOptions，`other` 的非 None 值覆盖 `self`。
    ///
    /// 用于实现参数合并链：Request > Agent > Model > Provider。
    pub fn merge(&self, other: &GenerationOptions) -> GenerationOptions {
        GenerationOptions {
            max_tokens: other.max_tokens.or(self.max_tokens),
            temperature: other.temperature.or(self.temperature),
            top_p: other.top_p.or(self.top_p),
            top_k: other.top_k.or(self.top_k),
            frequency_penalty: other.frequency_penalty.or(self.frequency_penalty),
            presence_penalty: other.presence_penalty.or(self.presence_penalty),
            stop: other.stop.clone().or_else(|| self.stop.clone()),
            seed: other.seed.or(self.seed),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_all_none() {
        let opts = GenerationOptions::new();
        assert_eq!(opts.max_tokens, None);
        assert_eq!(opts.temperature, None);
        assert_eq!(opts.top_p, None);
        assert_eq!(opts.top_k, None);
        assert_eq!(opts.frequency_penalty, None);
        assert_eq!(opts.presence_penalty, None);
        assert_eq!(opts.stop, None);
        assert_eq!(opts.seed, None);
    }

    #[test]
    fn test_builder_methods() {
        let opts = GenerationOptions::new()
            .with_max_tokens(4096)
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_top_k(40)
            .with_frequency_penalty(0.5)
            .with_presence_penalty(-0.5)
            .with_stop(vec!["END".into()])
            .with_seed(42);

        assert_eq!(opts.max_tokens, Some(4096));
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.top_p, Some(0.9));
        assert_eq!(opts.top_k, Some(40));
        assert_eq!(opts.frequency_penalty, Some(0.5));
        assert_eq!(opts.presence_penalty, Some(-0.5));
        assert_eq!(opts.stop, Some(vec!["END".to_string()]));
        assert_eq!(opts.seed, Some(42));
    }

    #[test]
    fn test_merge_other_overrides_self() {
        let base = GenerationOptions::new()
            .with_max_tokens(1024)
            .with_temperature(0.5);

        let override_opts = GenerationOptions::new()
            .with_max_tokens(4096)
            .with_top_p(0.9);

        let merged = base.merge(&override_opts);
        assert_eq!(merged.max_tokens, Some(4096)); // overridden
        assert_eq!(merged.temperature, Some(0.5)); // kept from base
        assert_eq!(merged.top_p, Some(0.9)); // new from override
    }

    #[test]
    fn test_merge_none_does_not_override() {
        let base = GenerationOptions::new()
            .with_max_tokens(2048)
            .with_seed(123);

        let empty = GenerationOptions::new();
        let merged = base.merge(&empty);
        assert_eq!(merged.max_tokens, Some(2048));
        assert_eq!(merged.seed, Some(123));
    }

    #[test]
    fn test_serde_roundtrip() {
        let opts = GenerationOptions::new()
            .with_max_tokens(4096)
            .with_temperature(0.8)
            .with_stop(vec!["<|end|>".into(), "STOP".into()]);

        let json = serde_json::to_string(&opts).unwrap();
        let restored: GenerationOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, restored);
    }

    #[test]
    fn test_serde_skips_none_fields() {
        let opts = GenerationOptions::new().with_max_tokens(100);
        let json = serde_json::to_string(&opts).unwrap();
        let restored: GenerationOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.max_tokens, Some(100));
        assert_eq!(restored.temperature, None);
    }
}
