//! # katu_core::usage
//!
//! ## 职责
//! 定义 LLM 请求的 token 用量与费用计量类型。
//!
//! ## 依赖
//! 无（本模块是 katu-core 的底层类型）
//!
//! ## 对外接口
//! - `Usage` — token 用量统计
//! - `Cost` — 费用明细（美元）
//!
//! ## 调用者
//! - `katu_core::message` — AssistantMessage 持有 Usage
//! - 上层 crate 通过 `katu_core::usage::*` 使用

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Cost
// ---------------------------------------------------------------------------

/// 费用明细（单位：美元）。
///
/// 按 token 类别拆分，`total` 为各项之和。
/// 仅在上层持有定价表时可计算，因此 `Usage` 中为 `Option<Cost>`。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// 输入 token 费用
    pub input: f64,
    /// 输出 token 费用
    pub output: f64,
    /// 缓存读取 token 费用
    pub cache_read: f64,
    /// 缓存写入 token 费用
    pub cache_write: f64,
    /// 总费用（各项之和）
    pub total: f64,
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// LLM 请求的 token 用量统计。
///
/// 字段语义遵循"包含式总量 + 非重叠分解"惯例
/// （与 OpenAI / Anthropic / AI SDK 对齐）：
///
/// - `input_tokens` — 总输入 token，**包含**缓存读/写
/// - `output_tokens` — 总输出 token，**包含**推理 token
/// - `total_tokens` — provider 报告的总量，或 `input + output`
///
/// 非重叠分解：
/// - `cache_read_tokens` + `cache_write_tokens` + 非缓存部分 = `input_tokens`
/// - `reasoning_tokens` ≤ `output_tokens`
///
/// 部分 provider 不报告细分字段，此时为 `0` 或由上层推断。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// 总输入 token（含缓存读/写）
    pub input_tokens: u32,
    /// 总输出 token（含推理）
    pub output_tokens: u32,
    /// 从缓存读取的输入 token
    pub cache_read_tokens: u32,
    /// 写入缓存的输入 token
    pub cache_write_tokens: u32,
    /// 推理/思考 token（output 的子集），`None` 表示 provider 未报告
    pub reasoning_tokens: Option<u32>,
    /// 总 token（provider 报告值或 input + output）
    pub total_tokens: u32,
    /// 费用明细，仅在上层持有定价表时可计算
    pub cost: Option<Cost>,
}

impl Usage {
    /// 可见输出 token = output - reasoning，至少为 0。
    pub fn visible_output_tokens(&self) -> u32 {
        self.output_tokens
            .saturating_sub(self.reasoning_tokens.unwrap_or(0))
    }

    /// 非缓存输入 token = input - cache_read - cache_write，至少为 0。
    pub fn non_cached_input_tokens(&self) -> u32 {
        self.input_tokens
            .saturating_sub(self.cache_read_tokens)
            .saturating_sub(self.cache_write_tokens)
    }
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_default_is_zero() {
        let u = Usage::default();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
        assert_eq!(u.total_tokens, 0);
        assert!(u.cost.is_none());
    }

    #[test]
    fn test_visible_output_tokens() {
        let u = Usage {
            output_tokens: 100,
            reasoning_tokens: Some(40),
            ..Default::default()
        };
        assert_eq!(u.visible_output_tokens(), 60);
    }

    #[test]
    fn test_visible_output_tokens_no_reasoning() {
        let u = Usage {
            output_tokens: 100,
            reasoning_tokens: None,
            ..Default::default()
        };
        assert_eq!(u.visible_output_tokens(), 100);
    }

    #[test]
    fn test_visible_output_tokens_saturating() {
        let u = Usage {
            output_tokens: 10,
            reasoning_tokens: Some(999),
            ..Default::default()
        };
        assert_eq!(u.visible_output_tokens(), 0);
    }

    #[test]
    fn test_non_cached_input_tokens() {
        let u = Usage {
            input_tokens: 1000,
            cache_read_tokens: 600,
            cache_write_tokens: 200,
            ..Default::default()
        };
        assert_eq!(u.non_cached_input_tokens(), 200);
    }

    #[test]
    fn test_non_cached_input_tokens_saturating() {
        let u = Usage {
            input_tokens: 100,
            cache_read_tokens: 80,
            cache_write_tokens: 80,
            ..Default::default()
        };
        assert_eq!(u.non_cached_input_tokens(), 0);
    }

    #[test]
    fn test_usage_serde_roundtrip() {
        let u = Usage {
            input_tokens: 500,
            output_tokens: 200,
            cache_read_tokens: 100,
            cache_write_tokens: 50,
            reasoning_tokens: Some(30),
            total_tokens: 700,
            cost: Some(Cost {
                input: 0.005,
                output: 0.010,
                cache_read: 0.001,
                cache_write: 0.002,
                total: 0.018,
            }),
        };
        let json = serde_json::to_string(&u).unwrap();
        let restored: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(u, restored);
    }

    #[test]
    fn test_cost_serde_roundtrip() {
        let c = Cost {
            input: 0.01,
            output: 0.03,
            cache_read: 0.001,
            cache_write: 0.002,
            total: 0.043,
        };
        let json = serde_json::to_string(&c).unwrap();
        let restored: Cost = serde_json::from_str(&json).unwrap();
        assert_eq!(c, restored);
    }
}
