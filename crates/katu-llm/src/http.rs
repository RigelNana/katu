//! # katu_llm::http
//!
//! ## 职责
//! 定义 HTTP 传输层的覆写选项。
//!
//! ## 对外接口
//! - `HttpOptions` — 额外的请求头、查询参数、body 字段

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// HttpOptions
// ---------------------------------------------------------------------------

/// HTTP 传输层覆写选项。
///
/// 允许在 ModelRef 或 LlmRequest 级别注入额外的 HTTP 信息，
/// 用于应对特殊部署需求（如 Azure api-version、OpenRouter routing、
/// 自定义 auth header 等）。
///
/// # Examples
/// ```
/// use katu_llm::HttpOptions;
///
/// let opts = HttpOptions::new()
///     .with_header("x-custom-header", "value")
///     .with_query_param("api-version", "2024-02-01");
/// assert_eq!(opts.headers.as_ref().unwrap().get("x-custom-header").unwrap(), "value");
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HttpOptions {
    /// 额外请求头（合并到 provider 默认头上）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// URL 查询参数（如 Azure `api-version`）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_params: Option<HashMap<String, String>>,
    /// 额外 body 字段（直接合并到请求体顶层）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<serde_json::Value>,
}

impl HttpOptions {
    /// 创建空的 HttpOptions。
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加一个请求头。
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }

    /// 添加一个查询参数。
    pub fn with_query_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }

    /// 设置额外 body 字段。
    pub fn with_extra_body(mut self, body: serde_json::Value) -> Self {
        self.extra_body = Some(body);
        self
    }

    /// 合并两个 HttpOptions，`other` 的值覆盖 `self`。
    pub fn merge(&self, other: &HttpOptions) -> HttpOptions {
        let headers = match (&self.headers, &other.headers) {
            (Some(base), Some(over)) => {
                let mut merged = base.clone();
                merged.extend(over.iter().map(|(k, v)| (k.clone(), v.clone())));
                Some(merged)
            }
            (None, Some(h)) | (Some(h), None) => Some(h.clone()),
            (None, None) => None,
        };

        let query_params = match (&self.query_params, &other.query_params) {
            (Some(base), Some(over)) => {
                let mut merged = base.clone();
                merged.extend(over.iter().map(|(k, v)| (k.clone(), v.clone())));
                Some(merged)
            }
            (None, Some(q)) | (Some(q), None) => Some(q.clone()),
            (None, None) => None,
        };

        let extra_body = other.extra_body.clone().or_else(|| self.extra_body.clone());

        HttpOptions {
            headers,
            query_params,
            extra_body,
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
    fn test_default_is_empty() {
        let opts = HttpOptions::new();
        assert_eq!(opts.headers, None);
        assert_eq!(opts.query_params, None);
        assert_eq!(opts.extra_body, None);
    }

    #[test]
    fn test_builder_methods() {
        let opts = HttpOptions::new()
            .with_header("Authorization", "Bearer token123")
            .with_query_param("api-version", "2024-02-01")
            .with_extra_body(serde_json::json!({"routing": {"order": ["anthropic"]}}));

        assert_eq!(
            opts.headers.as_ref().unwrap().get("Authorization").unwrap(),
            "Bearer token123"
        );
        assert_eq!(
            opts.query_params.as_ref().unwrap().get("api-version").unwrap(),
            "2024-02-01"
        );
        assert!(opts.extra_body.is_some());
    }

    #[test]
    fn test_merge_headers() {
        let base = HttpOptions::new()
            .with_header("x-base", "1")
            .with_header("x-shared", "base");

        let over = HttpOptions::new()
            .with_header("x-over", "2")
            .with_header("x-shared", "override");

        let merged = base.merge(&over);
        let headers = merged.headers.unwrap();
        assert_eq!(headers.get("x-base").unwrap(), "1");
        assert_eq!(headers.get("x-over").unwrap(), "2");
        assert_eq!(headers.get("x-shared").unwrap(), "override");
    }

    #[test]
    fn test_serde_roundtrip() {
        let opts = HttpOptions::new()
            .with_header("key", "val")
            .with_query_param("v", "1");

        let json = serde_json::to_string(&opts).unwrap();
        let restored: HttpOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, restored);
    }
}
