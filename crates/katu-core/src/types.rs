//! # katu_core::types
//!
//! ## 职责
//! 定义全局 newtype ID 类型与基础枚举，防止裸 String/u64 混用。
//!
//! ## 依赖
//! 无（本模块是 katu-core 的最底层）
//!
//! ## 对外接口
//! - `SessionId`, `MessageId`, `AgentId`, `ToolCallId` — newtype ID
//! - `ProviderId`, `ModelId`, `RouteId` — LLM 模型相关 ID
//! - `Role` — 消息角色枚举
//! - `FinishReason` — LLM 停止原因
//!
//! ## 调用者
//! - `katu_core::error` — 在错误类型中引用 ID
//! - `katu_core::message` — Message 结构体持有 MessageId、Role
//! - `katu_core::event` — AgentEvent 引用 AgentId、ToolCallId
//! - 所有上层 crate 通过 `katu_core::types::*` 使用

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// SessionId
// ---------------------------------------------------------------------------

/// 会话唯一标识符。
///
/// 每个 Session 在创建时生成，贯穿整个对话生命周期。
/// 使用 UUID v7（时间有序）便于按时间排序。
///
/// # Examples
/// ```
/// use katu_core::SessionId;
///
/// let id = SessionId::new();
/// println!("session: {id}");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(Uuid);

impl SessionId {
    /// 生成一个新的时间有序 SessionId (UUID v7)。
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// 获取内部 UUID 的引用。
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// ---------------------------------------------------------------------------
// MessageId
// ---------------------------------------------------------------------------

/// 单条消息的唯一标识符。
///
/// 在 Agent loop 中每条 User/Assistant/Tool 消息各持有一个。
///
/// # Examples
/// ```
/// use katu_core::MessageId;
///
/// let id = MessageId::new();
/// assert!(!id.to_string().is_empty());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(Uuid);

impl MessageId {
    /// 生成一个新的时间有序 MessageId (UUID v7)。
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// 获取内部 UUID 的引用。
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// ---------------------------------------------------------------------------
// AgentId
// ---------------------------------------------------------------------------

/// Agent 实例标识符。
///
/// 主 Agent 和每个 SubAgent 各持有独立 ID，
/// 用于事件追踪和上下文隔离。
///
/// # Examples
/// ```
/// use katu_core::AgentId;
///
/// let main_agent = AgentId::new();
/// let sub_agent = AgentId::new();
/// assert_ne!(main_agent, sub_agent);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(Uuid);

impl AgentId {
    /// 生成一个新的时间有序 AgentId (UUID v7)。
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// 获取内部 UUID 的引用。
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// ---------------------------------------------------------------------------
// ToolCallId
// ---------------------------------------------------------------------------

/// 单次工具调用的唯一标识符。
///
/// 由 LLM 在 tool_call 响应中生成（字符串格式），
/// 用于关联 ToolCall 请求与 ToolResult 响应。
///
/// # Examples
/// ```
/// use katu_core::ToolCallId;
///
/// let id = ToolCallId::new("call_abc123");
/// assert_eq!(id.as_str(), "call_abc123");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolCallId(String);

impl ToolCallId {
    /// 从 LLM 返回的 tool call id 字符串创建。
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// 获取内部字符串的引用。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// ---------------------------------------------------------------------------
// ProviderId
// ---------------------------------------------------------------------------

/// LLM Provider 标识符。
///
/// 标识一个 LLM 服务提供商，如 `"anthropic"`, `"openai"`, `"deepseek"`。
/// 用于路由选择、API key 查找、费率匹配等。
///
/// # Examples
/// ```
/// use katu_core::ProviderId;
///
/// let p = ProviderId::new("anthropic");
/// assert_eq!(p.as_str(), "anthropic");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderId(String);

impl ProviderId {
    /// 从字符串创建 ProviderId。
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// 获取内部字符串的引用。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// ---------------------------------------------------------------------------
// ModelId
// ---------------------------------------------------------------------------

/// LLM 模型标识符。
///
/// 对应发送给 API 的 `model` 字段值，
/// 如 `"claude-sonnet-4-20250514"`, `"gpt-4o"`, `"deepseek-chat"`。
///
/// # Examples
/// ```
/// use katu_core::ModelId;
///
/// let m = ModelId::new("claude-sonnet-4-20250514");
/// assert_eq!(m.as_str(), "claude-sonnet-4-20250514");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(String);

impl ModelId {
    /// 从字符串创建 ModelId。
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// 获取内部字符串的引用。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// ---------------------------------------------------------------------------
// RouteId
// ---------------------------------------------------------------------------

/// 路由标识符。
///
/// 指向一个 Protocol 实现，决定如何将 `LlmRequest` 转换为
/// provider 原生请求格式。如 `"openai-chat"`, `"anthropic-messages"`。
///
/// 多个 Provider 可共享同一 RouteId（例如 DeepSeek、Together
/// 等兼容 OpenAI 的 provider 均使用 `"openai-chat"` 路由）。
///
/// # Examples
/// ```
/// use katu_core::RouteId;
///
/// let r = RouteId::new("openai-chat");
/// assert_eq!(r.as_str(), "openai-chat");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RouteId(String);

impl RouteId {
    /// 从字符串创建 RouteId。
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// 获取内部字符串的引用。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RouteId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

/// 消息角色。
///
/// 对应 LLM API 中的 `role` 字段：
/// - `System` — 系统指令，定义 Agent 行为
/// - `User` — 用户输入
/// - `Assistant` — LLM 回复（可包含文本和 tool_call）
/// - `Tool` — 工具执行结果
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Tool => write!(f, "tool"),
        }
    }
}

// ---------------------------------------------------------------------------
// FinishReason
// ---------------------------------------------------------------------------

/// LLM 停止生成的原因。
///
/// Agent loop 根据此值决定下一步行为：
/// - `ToolCalls` → 执行工具后继续循环
/// - `Stop` → 返回最终回复
/// - `Length` → 触发上下文压缩
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// 正常结束
    Stop,
    /// 达到 token 上限
    Length,
    /// 模型请求调用工具
    ToolCalls,
    /// 内容被过滤
    ContentFilter,
    /// 发生错误
    Error,
    /// 未知原因
    Unknown,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stop => write!(f, "stop"),
            Self::Length => write!(f, "length"),
            Self::ToolCalls => write!(f, "tool_calls"),
            Self::ContentFilter => write!(f, "content_filter"),
            Self::Error => write!(f, "error"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SessionId --

    /// 验证 SessionId::new() 生成的 ID 互不相同
    #[test]
    fn test_session_id_uniqueness() {
        let a = SessionId::new();
        let b = SessionId::new();
        assert_ne!(a, b);
    }

    /// 验证 SessionId 的 Display 输出非空
    #[test]
    fn test_session_id_display_not_empty() {
        let id = SessionId::new();
        assert!(!id.to_string().is_empty());
    }

    /// 验证 SessionId 可序列化/反序列化并保持一致
    #[test]
    fn test_session_id_serde_roundtrip() {
        let id = SessionId::new();
        let json = serde_json::to_string(&id).unwrap();
        let restored: SessionId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    // -- MessageId --

    /// 验证 MessageId::new() 生成的 ID 互不相同
    #[test]
    fn test_message_id_uniqueness() {
        let a = MessageId::new();
        let b = MessageId::new();
        assert_ne!(a, b);
    }

    /// 验证 MessageId 可序列化/反序列化并保持一致
    #[test]
    fn test_message_id_serde_roundtrip() {
        let id = MessageId::new();
        let json = serde_json::to_string(&id).unwrap();
        let restored: MessageId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    // -- AgentId --

    /// 验证两个 AgentId 不相等（主 Agent vs SubAgent 场景）
    #[test]
    fn test_agent_id_uniqueness() {
        let main = AgentId::new();
        let sub = AgentId::new();
        assert_ne!(main, sub);
    }

    /// 验证 AgentId 可序列化/反序列化并保持一致
    #[test]
    fn test_agent_id_serde_roundtrip() {
        let id = AgentId::new();
        let json = serde_json::to_string(&id).unwrap();
        let restored: AgentId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    // -- ToolCallId --

    /// 验证 ToolCallId 从字符串构造并正确返回
    #[test]
    fn test_tool_call_id_from_string() {
        let id = ToolCallId::new("call_abc123");
        assert_eq!(id.as_str(), "call_abc123");
        assert_eq!(id.to_string(), "call_abc123");
    }

    /// 验证 ToolCallId 接受 String 和 &str
    #[test]
    fn test_tool_call_id_impl_into() {
        let _from_str = ToolCallId::new("abc");
        let _from_string = ToolCallId::new(String::from("def"));
    }

    /// 验证 ToolCallId 可序列化/反序列化并保持一致
    #[test]
    fn test_tool_call_id_serde_roundtrip() {
        let id = ToolCallId::new("call_xyz");
        let json = serde_json::to_string(&id).unwrap();
        let restored: ToolCallId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    // -- Role --

    /// 验证 Role 的 serde 序列化为小写字符串
    #[test]
    fn test_role_serde_lowercase() {
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
    }

    /// 验证 Role 的 Display 输出与 serde 一致
    #[test]
    fn test_role_display() {
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
        assert_eq!(Role::Tool.to_string(), "tool");
    }

    /// 验证 Role 可从 JSON 字符串反序列化
    #[test]
    fn test_role_deserialize() {
        let role: Role = serde_json::from_str("\"assistant\"").unwrap();
        assert_eq!(role, Role::Assistant);
    }

    /// 验证 Role 反序列化非法值时报错
    #[test]
    fn test_role_deserialize_invalid() {
        let result = serde_json::from_str::<Role>("\"invalid\"");
        assert!(result.is_err());
    }

    // -- FinishReason --

    /// 验证 FinishReason 的 serde 序列化为 snake_case
    #[test]
    fn test_finish_reason_serde() {
        assert_eq!(
            serde_json::to_string(&FinishReason::Stop).unwrap(),
            "\"stop\""
        );
        assert_eq!(
            serde_json::to_string(&FinishReason::ToolCalls).unwrap(),
            "\"tool_calls\""
        );
        assert_eq!(
            serde_json::to_string(&FinishReason::ContentFilter).unwrap(),
            "\"content_filter\""
        );
    }

    /// 验证 FinishReason 的 Display 输出
    #[test]
    fn test_finish_reason_display() {
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::Length.to_string(), "length");
        assert_eq!(FinishReason::ToolCalls.to_string(), "tool_calls");
    }

    /// 验证 FinishReason 可从 JSON 反序列化
    #[test]
    fn test_finish_reason_roundtrip() {
        let reason = FinishReason::ToolCalls;
        let json = serde_json::to_string(&reason).unwrap();
        let restored: FinishReason = serde_json::from_str(&json).unwrap();
        assert_eq!(reason, restored);
    }

    // -- ProviderId --

    /// 验证 ProviderId 从字符串构造并正确返回
    #[test]
    fn test_provider_id_from_string() {
        let id = ProviderId::new("anthropic");
        assert_eq!(id.as_str(), "anthropic");
        assert_eq!(id.to_string(), "anthropic");
    }

    /// 验证 ProviderId 可序列化/反序列化并保持一致
    #[test]
    fn test_provider_id_serde_roundtrip() {
        let id = ProviderId::new("openai");
        let json = serde_json::to_string(&id).unwrap();
        let restored: ProviderId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    // -- ModelId --

    /// 验证 ModelId 从字符串构造并正确返回
    #[test]
    fn test_model_id_from_string() {
        let id = ModelId::new("claude-sonnet-4-20250514");
        assert_eq!(id.as_str(), "claude-sonnet-4-20250514");
        assert_eq!(id.to_string(), "claude-sonnet-4-20250514");
    }

    /// 验证 ModelId 可序列化/反序列化并保持一致
    #[test]
    fn test_model_id_serde_roundtrip() {
        let id = ModelId::new("gpt-4o");
        let json = serde_json::to_string(&id).unwrap();
        let restored: ModelId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }

    // -- RouteId --

    /// 验证 RouteId 从字符串构造并正确返回
    #[test]
    fn test_route_id_from_string() {
        let id = RouteId::new("openai-chat");
        assert_eq!(id.as_str(), "openai-chat");
        assert_eq!(id.to_string(), "openai-chat");
    }

    /// 验证 RouteId 可序列化/反序列化并保持一致
    #[test]
    fn test_route_id_serde_roundtrip() {
        let id = RouteId::new("anthropic-messages");
        let json = serde_json::to_string(&id).unwrap();
        let restored: RouteId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, restored);
    }
}
