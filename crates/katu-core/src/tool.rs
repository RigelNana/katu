//! # katu_core::tool
//!
//! ## 职责
//! 定义工具系统的数据类型与执行契约。
//!
//! ## 设计原则
//! - **Provider 无关** — 参数使用 JSON Schema（`serde_json::Value`），适配所有 LLM provider
//! - **Serde 友好** — 数据类型可序列化/反序列化
//! - **最小 trait** — `Tool` trait 只含必要方法，扩展能力留给上层
//!
//! ## 对外接口
//! - `ToolDefinition` — 发送给 LLM 的工具 schema（name + description + parameters JSON Schema）
//! - `ToolOutput` — 工具执行结果（content + metadata + is_error）
//! - `ToolChoice` — 工具选择策略（auto / none / required / specific）
//! - `Tool` — 工具执行 trait（definition + validate + execute + concurrency_mode）
//! - `ToolCallContext` — 执行上下文（call_id + cancellation + extra）
//! - `CancellationToken` — 协作式取消令牌
//! - `ConcurrencyMode` — 并发调度标记
//!
//! ## 调用者
//! - `katu-llm` — `LlmRequest` 持有 `Vec<ToolDefinition>` + `ToolChoice`
//! - `katu-agent` (future) — Agent loop 通过 `Tool` trait 调用工具
//! - `katu-core::event` — `StreamEvent::ToolResult` 可从 `ToolOutput` 构造

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::types::ToolCallId;

// ===========================================================================
// ToolDefinition
// ===========================================================================

/// 工具定义 — 发送给 LLM 的 schema。
///
/// 对应 LLM API 中 `tools` 数组的每个元素，包含名称、描述和参数 JSON Schema。
/// LLM 据此决定何时调用工具以及如何构造参数。
///
/// # Examples
///
/// ```
/// use katu_core::ToolDefinition;
/// use serde_json::json;
///
/// let tool = ToolDefinition::new(
///     "read_file",
///     "Read the contents of a file at the given path",
///     json!({
///         "type": "object",
///         "properties": {
///             "path": { "type": "string", "description": "File path" }
///         },
///         "required": ["path"]
///     }),
/// );
/// assert_eq!(tool.name, "read_file");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// 工具名称 — LLM 在 tool_call 中引用的唯一标识。
    ///
    /// 命名约定：`snake_case`，如 `"read_file"`, `"bash"`, `"web_search"`。
    pub name: String,

    /// 工具描述 — LLM 据此决定何时以及为什么调用此工具。
    ///
    /// 应该清晰描述工具的功能、适用场景和限制。
    pub description: String,

    /// 参数的 JSON Schema — 定义工具接受的输入格式。
    ///
    /// 必须是一个 `{"type": "object", "properties": {...}}` 形式的 JSON Schema。
    /// LLM 根据此 schema 构造 `tool_call.arguments`。
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    /// 创建新的工具定义。
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// 创建无参数的工具定义。
    ///
    /// 等价于 `parameters: {"type": "object", "properties": {}}`。
    pub fn no_params(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }
}

// ===========================================================================
// ToolOutput
// ===========================================================================

/// 工具执行结果 — 返回给 agent loop 的输出。
///
/// 设计参考 OpenCode 的 `ExecuteResult`：
/// - `content` 总是 string（LLM 只理解文本）
/// - `metadata` 为结构化数据（UI/日志/遥测用，不发送给 LLM）
/// - `is_error` 标记非抛异常的失败（如工具内部捕获的错误）
///
/// # Examples
///
/// ```
/// use katu_core::ToolOutput;
/// use serde_json::json;
///
/// // 成功结果
/// let output = ToolOutput::success("File contents here");
/// assert!(!output.is_error);
///
/// // 错误结果
/// let output = ToolOutput::error("Permission denied: /etc/shadow");
/// assert!(output.is_error);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    /// 标题 — UI 显示用的简短描述。
    ///
    /// 例如 `"Read file: src/main.rs"`, `"Bash: ls -la"`。
    #[serde(default)]
    pub title: String,

    /// 主输出内容 — 发送回 LLM 的文本。
    ///
    /// 这是 LLM 在下一轮推理中看到的 tool result 内容。
    pub content: String,

    /// 结构化元数据 — UI/日志/遥测用，**不**发送给 LLM。
    ///
    /// 例如执行耗时、文件路径、diff 统计等。
    #[serde(default = "default_metadata")]
    pub metadata: serde_json::Value,

    /// 是否为错误结果。
    ///
    /// `true` 时 agent loop 将 `content` 作为错误信息反馈给 LLM，
    /// LLM 可据此修正策略。区别于 `Err(...)` 的不可恢复错误，
    /// `is_error = true` 表示工具执行完成但结果是失败的。
    #[serde(default)]
    pub is_error: bool,
}

fn default_metadata() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

impl ToolOutput {
    /// 创建成功的工具输出。
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            title: String::new(),
            content: content.into(),
            metadata: default_metadata(),
            is_error: false,
        }
    }

    /// 创建带标题的成功工具输出。
    pub fn success_with_title(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            content: content.into(),
            metadata: default_metadata(),
            is_error: false,
        }
    }

    /// 创建错误的工具输出。
    ///
    /// 不同于 `Err(...)` — 这里工具执行完成了，但结果是失败的。
    /// LLM 会看到错误信息并可以据此调整策略。
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            title: String::new(),
            content: content.into(),
            metadata: default_metadata(),
            is_error: true,
        }
    }

    /// 设置元数据（builder 模式）。
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// 设置标题（builder 模式）。
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }
}

// ===========================================================================
// ToolChoice
// ===========================================================================

/// 工具选择策略 — 控制 LLM 是否以及如何使用工具。
///
/// 发送给 LLM 的 `tool_choice` 参数，不同 provider 有不同的映射方式，
/// 由 provider adapter 负责转换。
///
/// # Examples
///
/// ```
/// use katu_core::ToolChoice;
///
/// let choice = ToolChoice::Auto;
/// assert!(choice.allows_tools());
///
/// let choice = ToolChoice::None;
/// assert!(!choice.allows_tools());
///
/// let choice = ToolChoice::specific("bash");
/// assert!(choice.allows_tools());
/// assert_eq!(choice.required_tool(), Some("bash"));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// 模型自行决定是否使用工具。
    #[default]
    Auto,

    /// 禁止使用工具 — 模型只生成文本。
    None,

    /// 必须使用工具 — 模型必须至少调用一个工具。
    Required,

    /// 强制使用指定工具。
    Specific {
        /// 必须调用的工具名称。
        name: String,
    },
}

impl ToolChoice {
    /// 创建强制使用指定工具的选择策略。
    pub fn specific(name: impl Into<String>) -> Self {
        Self::Specific { name: name.into() }
    }

    /// 是否允许工具调用（`None` 时不允许）。
    pub fn allows_tools(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// 如果是 `Specific`，返回指定的工具名称。
    pub fn required_tool(&self) -> Option<&str> {
        match self {
            Self::Specific { name } => Some(name.as_str()),
            _ => Option::None,
        }
    }

    /// 是否强制使用工具（`Required` 或 `Specific`）。
    pub fn is_forced(&self) -> bool {
        matches!(self, Self::Required | Self::Specific { .. })
    }
}


// ===========================================================================
// CancellationToken
// ===========================================================================

/// 协作式取消令牌。
///
/// Agent loop 在需要取消工具时调用 `cancel()`，
/// 工具在长运行循环中周期性检查 `is_cancelled()` 并提前退出。
///
/// ## 设计选择
/// - **轻量 AtomicBool** — katu-core 保持 runtime 无关，不依赖 tokio
/// - **polling 模式** — 覆盖长循环检查场景；async 等待由 agent loop 的
///   `tokio::select!` 实现
/// - **Clone 共享** — Arc 内部共享，agent loop 和 tool 各持一端
///
/// # Examples
///
/// ```
/// use katu_core::CancellationToken;
///
/// let token = CancellationToken::new();
/// let token2 = token.clone();
///
/// assert!(!token.is_cancelled());
/// token2.cancel();
/// assert!(token.is_cancelled());
/// ```
/// !TODO
#[derive(Debug, Clone)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// 创建未取消的令牌。
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// 触发取消。
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// 检查是否已取消。
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// ConcurrencyMode
// ===========================================================================

/// 工具并发模式 — Agent loop 调度同批次 tool_call 时的行为。
///
/// 当 LLM 在一次响应中请求多个 tool_call 时：
/// - 所有 `Shared` 工具可以并行执行
/// - 遇到 `Exclusive` 工具时，等待前面的工具完成后独占执行
///
/// # Examples
///
/// ```
/// use katu_core::ConcurrencyMode;
///
/// let mode = ConcurrencyMode::default();
/// assert_eq!(mode, ConcurrencyMode::Shared);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConcurrencyMode {
    /// 可与其他 Shared 工具并行执行（如 read_file, grep）。
    #[default]
    Shared,
    /// 独占执行，不与其他工具并行（如 write_file, bash）。
    Exclusive,
}

// ===========================================================================
// ToolCallContext
// ===========================================================================

/// 工具执行上下文 — Agent loop 构造后传入 `Tool::execute`。
///
/// ## Builder 模式
/// 必填字段通过 `new(call_id)` 提供，可选字段通过 `with_*` 链式设置。
///
/// # Examples
///
/// ```
/// use katu_core::{ToolCallContext, ToolCallId, CancellationToken};
/// use serde_json::json;
///
/// let token = CancellationToken::new();
/// let ctx = ToolCallContext::new(ToolCallId::new("call_1"))
///     .with_cancellation(token.clone())
///     .with_extra(json!({"cwd": "/home/user/project"}));
///
/// assert_eq!(ctx.call_id.as_str(), "call_1");
/// assert!(!ctx.cancellation.is_cancelled());
/// ```
pub struct ToolCallContext {
    /// 本次 tool_call 的唯一标识（由 LLM 或 agent loop 分配）。
    pub call_id: ToolCallId,

    /// 取消令牌 — 工具在长循环中检查是否需要提前退出。
    pub cancellation: CancellationToken,

    /// 扩展数据 — 上层应用注入的额外上下文。
    ///
    /// 如当前工作目录、环境变量、session 信息等。
    /// 基座不预设上层需求，工具按 key 取值。
    pub extra: serde_json::Value,
}

impl ToolCallContext {
    /// 创建上下文 — 只需 call_id，其余使用默认值。
    pub fn new(call_id: ToolCallId) -> Self {
        Self {
            call_id,
            cancellation: CancellationToken::new(),
            extra: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// 注入已有的取消令牌（Agent loop 持有另一端用于触发取消）。
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = token;
        self
    }

    /// 注入扩展数据。
    pub fn with_extra(mut self, extra: serde_json::Value) -> Self {
        self.extra = extra;
        self
    }
}

// ===========================================================================
// Tool trait
// ===========================================================================

/// 工具执行 trait — 所有可被 Agent 调用的工具必须实现此 trait。
///
/// ## 方法职责
/// - `definition()` — "我是谁"（schema 信息，注册给 LLM）
/// - `validate()` — "参数合法吗"（可选的补充校验，默认通过）
/// - `execute()` — "执行动作"（核心业务逻辑）
/// - `concurrency_mode()` — "我的调度约束"（给 agent loop 的调度提示）
/// - `permission_request()` — "细粒度权限"（动态构建权限请求，覆盖默认逻辑）
///
/// ## 返回值约定
/// - `Ok(ToolOutput { is_error: false })` — 工具成功
/// - `Ok(ToolOutput { is_error: true })` — 业务失败（如"文件不存在"），
///   agent loop 回传 LLM 让模型调整
/// - `Err(Error::Cancelled)` — 被取消
/// - `Err(Error::Internal(..))` — 工具崩溃，agent loop 决定重试或终止
///
/// ## Object Safety
/// 通过 `#[async_trait]` 实现 dyn dispatch，支持 `Arc<dyn Tool>` 在 registry 中存储。
///
/// # Examples
///
/// ```
/// use async_trait::async_trait;
/// use katu_core::{Tool, ToolDefinition, ToolOutput, ToolCallContext, ConcurrencyMode, Result};
/// use serde_json::{json, Value};
///
/// struct GetTimeTool;
///
/// static GET_TIME_DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
///     ToolDefinition::no_params("get_time", "Get current UTC time")
/// });
///
/// #[async_trait]
/// impl Tool for GetTimeTool {
///     fn definition(&self) -> &ToolDefinition {
///         &GET_TIME_DEF
///     }
///
///     async fn execute(&self, _args: Value, _ctx: &ToolCallContext) -> Result<ToolOutput> {
///         Ok(ToolOutput::success("2025-01-01T00:00:00Z"))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// 返回工具定义 — 名称、描述、参数 JSON Schema。
    fn definition(&self) -> &ToolDefinition;

    /// 参数补充验证 — 在 execute 前调用。
    ///
    /// 默认实现返回 `Ok(())` — 信任 LLM 已按 JSON Schema 生成参数。
    /// 需要额外校验的工具（如路径安全检查）可覆盖此方法。
    ///
    /// 返回 `Err` 时，agent loop 将错误信息作为 tool result 返回给 LLM，
    /// 不会调用 `execute`。
    async fn validate(&self, _args: &serde_json::Value, _ctx: &ToolCallContext) -> Result<()> {
        Ok(())
    }

    /// 执行工具 — 核心业务逻辑。
    ///
    /// ## 取消约定
    /// 长运行工具应周期性检查 `ctx.cancellation.is_cancelled()`，
    /// 检测到取消后返回 `Err(Error::Cancelled)` 并清理资源。
    async fn execute(&self, args: serde_json::Value, ctx: &ToolCallContext) -> Result<ToolOutput>;

    /// 并发模式 — 告知 agent loop 此工具的调度约束。
    ///
    /// 默认 `Shared` — 可与其他工具并行。
    /// 写入类工具应返回 `Exclusive`。
    fn concurrency_mode(&self) -> ConcurrencyMode {
        ConcurrencyMode::Shared
    }

    /// 权限 key — 权限规则匹配时使用的标识。
    ///
    /// 默认返回工具名称。某些工具可能细分权限：
    /// 如 bash 工具根据子命令前缀返回 `"bash"` 或 `"bash:git"`。
    fn permission_key(&self) -> &str {
        &self.definition().name
    }

    /// 工具级权限检查 — 在规则引擎求值后、用户交互前调用。
    ///
    /// ## 用途
    /// 工具实现可据此检查：
    /// - 路径安全性（如禁止写入 `.git/`、`.katu/` 目录）
    /// - 命令安全性（如禁止 `rm -rf /`）
    /// - URL 白名单等
    ///
    /// ## 返回值
    /// - `Passthrough` — 不做判断，交由规则引擎（**默认**）
    /// - `Allow` — 工具认为此操作安全
    /// - `Deny { message }` — 工具明确拒绝
    /// - `Ask { message }` — 工具建议询问用户
    ///
    /// ## 与 validate 的区别
    /// - `validate()` = 参数**格式**是否合法（类型检查）
    /// - `check_permissions()` = 操作**是否被允许**（授权检查）
    ///
    /// ## 调用顺序
    /// ```text
    /// Hook(PreToolUse) → check_permissions() → 规则引擎 → 用户交互 → validate() → execute()
    /// ```
    fn check_permissions(
        &self,
        _args: &serde_json::Value,
        _ctx: &ToolCallContext,
    ) -> crate::permission::PermissionResult {
        crate::permission::PermissionResult::Passthrough
    }

    /// 构建细粒度权限请求 — 工具可据此定制 permission key 和 pattern。
    ///
    /// 默认返回 `None` — 框架使用 `permission_key()` + `args.to_string()` 的默认逻辑。
    ///
    /// ## 用途
    /// 需要细粒度权限控制的工具（如 bash）可覆盖此方法，提供：
    /// - 更精确的 permission key（如 `"bash:git"` 而非 `"bash"`）
    /// - 有意义的 pattern（如 `"git push origin main"` 而非序列化后的 JSON）
    /// - always-allow 模式（如 `"git push *"`）
    /// - UI 展示用的元数据
    ///
    /// ## 与 permission_key() 的关系
    /// - `permission_key()` 返回固定的 `&str`，适合简单工具
    /// - `permission_request()` 返回动态构造的 `PermissionRequest`，适合需要
    ///   根据参数内容变化 key 和 pattern 的复杂工具
    /// - 如果两者都实现，`permission_request()` 优先
    ///
    /// ## 调用时机
    /// 在 `check_permissions()` 返回 `Passthrough` 后、Ruleset 求值前调用。
    ///
    /// # Examples
    ///
    /// ```ignore
    /// fn permission_request(
    ///     &self,
    ///     args: &serde_json::Value,
    ///     _ctx: &ToolCallContext,
    /// ) -> Option<crate::permission::PermissionRequest> {
    ///     let command = args["command"].as_str()?;
    ///     Some(crate::permission::PermissionRequest::new("bash:git", command)
    ///         .with_tool_name("bash")
    ///         .with_always_allow(vec!["git push *"]))
    /// }
    /// ```
    fn permission_request(
        &self,
        _args: &serde_json::Value,
        _ctx: &ToolCallContext,
    ) -> Option<crate::permission::PermissionRequest> {
        None
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- ToolDefinition --

    #[test]
    fn test_tool_definition_new() {
        let def = ToolDefinition::new(
            "read_file",
            "Read file contents",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        );
        assert_eq!(def.name, "read_file");
        assert_eq!(def.description, "Read file contents");
        assert!(def.parameters["properties"]["path"]["type"]
            .as_str()
            .unwrap()
            == "string");
    }

    #[test]
    fn test_tool_definition_no_params() {
        let def = ToolDefinition::no_params("get_time", "Get current time");
        assert_eq!(def.name, "get_time");
        assert_eq!(def.parameters["type"], "object");
        assert!(def.parameters["properties"]
            .as_object()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_tool_definition_serde_roundtrip() {
        let def = ToolDefinition::new(
            "bash",
            "Run a shell command",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                },
                "required": ["command"]
            }),
        );
        let json_str = serde_json::to_string(&def).unwrap();
        let restored: ToolDefinition = serde_json::from_str(&json_str).unwrap();
        assert_eq!(def, restored);
    }

    // -- ToolOutput --

    #[test]
    fn test_tool_output_success() {
        let out = ToolOutput::success("hello world");
        assert_eq!(out.content, "hello world");
        assert!(!out.is_error);
        assert!(out.title.is_empty());
    }

    #[test]
    fn test_tool_output_success_with_title() {
        let out = ToolOutput::success_with_title("Read file", "file contents");
        assert_eq!(out.title, "Read file");
        assert_eq!(out.content, "file contents");
        assert!(!out.is_error);
    }

    #[test]
    fn test_tool_output_error() {
        let out = ToolOutput::error("not found");
        assert_eq!(out.content, "not found");
        assert!(out.is_error);
    }

    #[test]
    fn test_tool_output_builder() {
        let out = ToolOutput::success("ok")
            .with_title("Done")
            .with_metadata(json!({"elapsed_ms": 42}));
        assert_eq!(out.title, "Done");
        assert_eq!(out.metadata["elapsed_ms"], 42);
        assert!(!out.is_error);
    }

    #[test]
    fn test_tool_output_serde_roundtrip() {
        let out = ToolOutput::success_with_title("Read", "contents")
            .with_metadata(json!({"lines": 100}));
        let json_str = serde_json::to_string(&out).unwrap();
        let restored: ToolOutput = serde_json::from_str(&json_str).unwrap();
        assert_eq!(out, restored);
    }

    #[test]
    fn test_tool_output_serde_defaults() {
        // 反序列化时缺少可选字段应该使用默认值
        let json_str = r#"{"content":"hello"}"#;
        let out: ToolOutput = serde_json::from_str(json_str).unwrap();
        assert_eq!(out.content, "hello");
        assert!(!out.is_error);
        assert!(out.title.is_empty());
        assert!(out.metadata.is_object());
    }

    // -- ToolChoice --

    #[test]
    fn test_tool_choice_auto_default() {
        let choice = ToolChoice::default();
        assert_eq!(choice, ToolChoice::Auto);
    }

    #[test]
    fn test_tool_choice_allows_tools() {
        assert!(ToolChoice::Auto.allows_tools());
        assert!(!ToolChoice::None.allows_tools());
        assert!(ToolChoice::Required.allows_tools());
        assert!(ToolChoice::specific("bash").allows_tools());
    }

    #[test]
    fn test_tool_choice_required_tool() {
        assert_eq!(ToolChoice::Auto.required_tool(), Option::None);
        assert_eq!(ToolChoice::None.required_tool(), Option::None);
        assert_eq!(ToolChoice::Required.required_tool(), Option::None);
        assert_eq!(ToolChoice::specific("bash").required_tool(), Some("bash"));
    }

    #[test]
    fn test_tool_choice_is_forced() {
        assert!(!ToolChoice::Auto.is_forced());
        assert!(!ToolChoice::None.is_forced());
        assert!(ToolChoice::Required.is_forced());
        assert!(ToolChoice::specific("bash").is_forced());
    }

    #[test]
    fn test_tool_choice_serde_roundtrip() {
        for choice in [
            ToolChoice::Auto,
            ToolChoice::None,
            ToolChoice::Required,
            ToolChoice::specific("read_file"),
        ] {
            let json_str = serde_json::to_string(&choice).unwrap();
            let restored: ToolChoice = serde_json::from_str(&json_str).unwrap();
            assert_eq!(choice, restored);
        }
    }

    #[test]
    fn test_tool_choice_serde_format() {
        let json_str = serde_json::to_string(&ToolChoice::Auto).unwrap();
        assert!(json_str.contains(r#""type":"auto""#));

        let json_str = serde_json::to_string(&ToolChoice::specific("bash")).unwrap();
        assert!(json_str.contains(r#""type":"specific""#));
        assert!(json_str.contains(r#""name":"bash""#));
    }

    // -- CancellationToken --

    #[test]
    fn test_cancellation_token_new() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_cancel() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_clone_shares_state() {
        let token = CancellationToken::new();
        let token2 = token.clone();

        assert!(!token.is_cancelled());
        assert!(!token2.is_cancelled());

        token.cancel();

        assert!(token.is_cancelled());
        assert!(token2.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_default() {
        let token = CancellationToken::default();
        assert!(!token.is_cancelled());
    }

    // -- ConcurrencyMode --

    #[test]
    fn test_concurrency_mode_default() {
        assert_eq!(ConcurrencyMode::default(), ConcurrencyMode::Shared);
    }

    #[test]
    fn test_concurrency_mode_serde_roundtrip() {
        for mode in [ConcurrencyMode::Shared, ConcurrencyMode::Exclusive] {
            let json_str = serde_json::to_string(&mode).unwrap();
            let restored: ConcurrencyMode = serde_json::from_str(&json_str).unwrap();
            assert_eq!(mode, restored);
        }
    }

    #[test]
    fn test_concurrency_mode_serde_format() {
        assert_eq!(
            serde_json::to_string(&ConcurrencyMode::Shared).unwrap(),
            r#""shared""#
        );
        assert_eq!(
            serde_json::to_string(&ConcurrencyMode::Exclusive).unwrap(),
            r#""exclusive""#
        );
    }

    // -- ToolCallContext --

    #[test]
    fn test_tool_call_context_new() {
        let ctx = ToolCallContext::new(ToolCallId::new("call_1"));
        assert_eq!(ctx.call_id.as_str(), "call_1");
        assert!(!ctx.cancellation.is_cancelled());
        assert!(ctx.extra.is_object());
    }

    #[test]
    fn test_tool_call_context_builder() {
        let token = CancellationToken::new();
        let ctx = ToolCallContext::new(ToolCallId::new("call_2"))
            .with_cancellation(token.clone())
            .with_extra(json!({"cwd": "/tmp", "env": {"DEBUG": "1"}}));

        assert_eq!(ctx.call_id.as_str(), "call_2");
        assert_eq!(ctx.extra["cwd"], "/tmp");
        assert_eq!(ctx.extra["env"]["DEBUG"], "1");

        // 共享 token
        token.cancel();
        assert!(ctx.cancellation.is_cancelled());
    }

    // -- Tool trait --

    struct EchoTool;

    static ECHO_DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
        ToolDefinition::new(
            "echo",
            "Echoes the input message",
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"]
            }),
        )
    });

    #[async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> &ToolDefinition {
            &ECHO_DEF
        }

        async fn execute(
            &self,
            args: serde_json::Value,
            _ctx: &ToolCallContext,
        ) -> Result<ToolOutput> {
            let message = args["message"]
                .as_str()
                .unwrap_or("(no message)");
            Ok(ToolOutput::success(message))
        }
    }

    struct ExclusiveTool;

    static EXCLUSIVE_DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
        ToolDefinition::new(
            "write_file",
            "Write content to a file",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        )
    });

    #[async_trait]
    impl Tool for ExclusiveTool {
        fn definition(&self) -> &ToolDefinition {
            &EXCLUSIVE_DEF
        }

        async fn validate(
            &self,
            args: &serde_json::Value,
            _ctx: &ToolCallContext,
        ) -> Result<()> {
            let path = args["path"].as_str().unwrap_or("");
            if path.starts_with("/etc/") {
                return Err(crate::Error::tool(
                    "write_file",
                    _ctx.call_id.clone(),
                    "cannot write to /etc/",
                ));
            }
            Ok(())
        }

        async fn execute(
            &self,
            args: serde_json::Value,
            ctx: &ToolCallContext,
        ) -> Result<ToolOutput> {
            if ctx.cancellation.is_cancelled() {
                return Err(crate::Error::Cancelled);
            }
            let path = args["path"].as_str().unwrap_or("?");
            Ok(ToolOutput::success(format!("wrote to {path}"))
                .with_title(format!("Write: {path}")))
        }

        fn concurrency_mode(&self) -> ConcurrencyMode {
            ConcurrencyMode::Exclusive
        }
    }

    #[tokio::test]
    async fn test_tool_echo_execute() {
        let tool = EchoTool;
        let ctx = ToolCallContext::new(ToolCallId::new("c1"));
        let result = tool
            .execute(json!({"message": "hello"}), &ctx)
            .await
            .unwrap();
        assert_eq!(result.content, "hello");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn test_tool_echo_default_concurrency() {
        let tool = EchoTool;
        assert_eq!(tool.concurrency_mode(), ConcurrencyMode::Shared);
    }

    #[tokio::test]
    async fn test_tool_echo_default_validate() {
        let tool = EchoTool;
        let ctx = ToolCallContext::new(ToolCallId::new("c1"));
        // 默认 validate 应通过
        assert!(tool.validate(&json!({}), &ctx).await.is_ok());
    }

    #[tokio::test]
    async fn test_tool_definition_matches() {
        let tool = EchoTool;
        assert_eq!(tool.definition().name, "echo");
        assert!(!tool.definition().description.is_empty());
    }

    #[tokio::test]
    async fn test_tool_exclusive_concurrency() {
        let tool = ExclusiveTool;
        assert_eq!(tool.concurrency_mode(), ConcurrencyMode::Exclusive);
    }

    #[tokio::test]
    async fn test_tool_validate_rejects_invalid() {
        let tool = ExclusiveTool;
        let ctx = ToolCallContext::new(ToolCallId::new("c2"));
        let result = tool
            .validate(&json!({"path": "/etc/shadow", "content": "x"}), &ctx)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_tool_validate_accepts_valid() {
        let tool = ExclusiveTool;
        let ctx = ToolCallContext::new(ToolCallId::new("c3"));
        let result = tool
            .validate(&json!({"path": "/tmp/test.txt", "content": "x"}), &ctx)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_tool_execute_with_cancellation() {
        let tool = ExclusiveTool;
        let token = CancellationToken::new();
        let ctx = ToolCallContext::new(ToolCallId::new("c4"))
            .with_cancellation(token.clone());

        // 未取消 → 成功
        let result = tool
            .execute(json!({"path": "/tmp/a.txt", "content": "hi"}), &ctx)
            .await
            .unwrap();
        assert_eq!(result.content, "wrote to /tmp/a.txt");

        // 取消后 → 错误
        token.cancel();
        let ctx2 = ToolCallContext::new(ToolCallId::new("c5"))
            .with_cancellation(token.clone());
        let result = tool
            .execute(json!({"path": "/tmp/b.txt", "content": "hi"}), &ctx2)
            .await;
        assert!(matches!(result, Err(crate::Error::Cancelled)));
    }

    #[tokio::test]
    async fn test_tool_dyn_dispatch() {
        // 验证 Tool trait 支持 dyn dispatch（object safety）
        let tool: Arc<dyn Tool> = Arc::new(EchoTool);
        let ctx = ToolCallContext::new(ToolCallId::new("c6"));
        let result = tool
            .execute(json!({"message": "dynamic"}), &ctx)
            .await
            .unwrap();
        assert_eq!(result.content, "dynamic");
    }
}
