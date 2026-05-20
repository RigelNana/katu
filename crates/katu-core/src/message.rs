//! # katu_core::message
//!
//! ## 职责
//! 定义对话消息类型：内容块、角色特化的消息结构、以及统一的 `Message` 枚举。
//!
//! ## 设计原则
//! - **Provider 无关** — 不绑定任何特定 LLM SDK
//! - **角色特化** — User / Assistant / ToolResult 各有独立 struct，
//!   编译期禁止非法组合（如 User 消息不可能包含 ToolCall）
//! - **Serde 友好** — 所有类型可序列化/反序列化，便于持久化和网络传输
//! - **双层 Content** — `ContentBlock`（文本+图片）用于 User/ToolResult，
//!   `AssistantBlock`（文本+推理+工具调用）用于 Assistant
//!
//! ## 对外接口
//! - `ContentBlock` — 基础内容块（文本、图片）
//! - `AssistantBlock` — Assistant 专用内容块（文本、推理、工具调用）
//! - `UserContent` — 用户内容（纯文本或多模态块）
//! - `UserMessage`, `AssistantMessage`, `ToolResultMessage` — 角色特化消息
//! - `Message` — 统一消息枚举
//!
//! ## 调用者
//! - `katu_core::event` — StreamEvent 引用 AssistantMessage
//! - `katu_core::request` — LlmRequest 持有 Vec<Message>
//! - 所有上层 crate 通过 `katu_core::message::*` 使用

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::types::{FinishReason, MessageId, Role, ToolCallId};
use crate::usage::Usage;

// ===========================================================================
// Content Blocks
// ===========================================================================

/// 基础内容块 — 可出现在 User 消息和 ToolResult 中。
///
/// 使用 `type` 字段作为 serde 标签区分变体。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// 纯文本
    Text {
        text: String,
    },

    /// 图片（base64 编码的二进制数据）
    Image {
        /// MIME 类型，如 `"image/png"`, `"image/jpeg"`
        media_type: String,
        /// base64 编码的图片数据
        data: String,
    },
}

/// Assistant 专用内容块 — 比基础块多了推理和工具调用。
///
/// 编译期保证只有 Assistant 消息可以包含 `ToolCall` 和 `Reasoning`。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantBlock {
    /// 文本输出
    Text {
        text: String,
    },

    /// 推理/思考内容（extended thinking）
    Reasoning {
        text: String,
        /// Provider 不透明签名（如 Anthropic redacted thinking, Google thought signature）
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },

    /// 工具调用请求
    ToolCall {
        /// LLM 生成的工具调用 ID，用于关联 ToolResultMessage
        id: ToolCallId,
        /// 工具名称
        name: String,
        /// 工具参数（JSON 值）
        arguments: serde_json::Value,
    },
}

// ===========================================================================
// User Content
// ===========================================================================

/// 用户消息内容 — 支持纯文本快捷方式或多模态内容块。
///
/// 90% 的场景是纯文本，`Text` 变体避免了
/// `vec![ContentBlock::Text { text }]` 的样板代码。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    /// 纯文本内容
    Text(String),
    /// 多模态内容块（文本 + 图片）
    Blocks(Vec<ContentBlock>),
}

impl UserContent {
    /// 将内容规范化为 `ContentBlock` 数组。
    pub fn into_blocks(self) -> Vec<ContentBlock> {
        match self {
            Self::Text(text) => vec![ContentBlock::Text { text }],
            Self::Blocks(blocks) => blocks,
        }
    }

    /// 提取纯文本内容；多模态时拼接所有文本块。
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

impl From<String> for UserContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for UserContent {
    fn from(text: &str) -> Self {
        Self::Text(text.to_owned())
    }
}

impl From<Vec<ContentBlock>> for UserContent {
    fn from(blocks: Vec<ContentBlock>) -> Self {
        Self::Blocks(blocks)
    }
}

// ===========================================================================
// Role-Specific Messages
// ===========================================================================

/// 用户消息。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    /// 消息唯一标识
    pub id: MessageId,
    /// 消息内容
    pub content: UserContent,
    /// 创建时间
    pub timestamp: DateTime<Utc>,
}

/// LLM 回复消息。
///
/// 直接携带 `model`、`provider`、`usage` 等上下文信息，
/// 便于多模型场景下追踪每轮回复的来源和开销。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// 消息唯一标识
    pub id: MessageId,
    /// 内容块数组（文本、推理、工具调用）
    pub content: Vec<AssistantBlock>,
    /// 使用的模型标识（如 `"gpt-4o"`, `"claude-sonnet-4-20250514"`）
    pub model: String,
    /// Provider 标识（如 `"openai"`, `"anthropic"`）
    pub provider: String,
    /// 停止原因
    pub finish_reason: FinishReason,
    /// Token 用量统计
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// 创建时间
    pub timestamp: DateTime<Utc>,
}

/// 工具执行结果消息。
///
/// 作为独立角色存在（非嵌入 content 数组），
/// 通过 `tool_call_id` 关联对应的 `AssistantBlock::ToolCall`。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultMessage {
    /// 消息唯一标识
    pub id: MessageId,
    /// 关联的工具调用 ID
    pub tool_call_id: ToolCallId,
    /// 工具名称
    pub tool_name: String,
    /// 执行结果内容（支持文本 + 图片）
    pub content: Vec<ContentBlock>,
    /// 是否为错误结果
    pub is_error: bool,
    /// 创建时间
    pub timestamp: DateTime<Utc>,
}

// ===========================================================================
// Unified Message Enum
// ===========================================================================

/// 统一消息枚举 — 对话历史中的一条消息。
///
/// 使用角色特化的变体，编译期保证类型安全：
/// - `User` — 用户输入（文本或多模态）
/// - `Assistant` — LLM 回复（文本、推理、工具调用）
/// - `ToolResult` — 工具执行结果
///
/// System prompt 不在此枚举中，它属于请求级别。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

// ---------------------------------------------------------------------------
// Message — accessor methods
// ---------------------------------------------------------------------------

impl Message {
    /// 返回消息的角色。
    pub fn role(&self) -> Role {
        match self {
            Self::User(_) => Role::User,
            Self::Assistant(_) => Role::Assistant,
            Self::ToolResult(_) => Role::Tool,
        }
    }

    /// 返回消息的唯一标识。
    pub fn id(&self) -> &MessageId {
        match self {
            Self::User(m) => &m.id,
            Self::Assistant(m) => &m.id,
            Self::ToolResult(m) => &m.id,
        }
    }

    /// 返回消息的时间戳。
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::User(m) => m.timestamp,
            Self::Assistant(m) => m.timestamp,
            Self::ToolResult(m) => m.timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// Message — convenience constructors
// ---------------------------------------------------------------------------

impl Message {
    /// 创建一个纯文本 User 消息。
    pub fn user(content: impl Into<UserContent>) -> Self {
        Self::User(UserMessage {
            id: MessageId::new(),
            content: content.into(),
            timestamp: Utc::now(),
        })
    }

    /// 创建一个纯文本 Assistant 消息（用于测试或合成消息）。
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::Assistant(AssistantMessage {
            id: MessageId::new(),
            content: vec![AssistantBlock::Text {
                text: text.into(),
            }],
            model: String::new(),
            provider: String::new(),
            finish_reason: FinishReason::Stop,
            usage: None,
            timestamp: Utc::now(),
        })
    }

    /// 创建一个工具结果消息。
    pub fn tool_result(
        tool_call_id: ToolCallId,
        tool_name: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::ToolResult(ToolResultMessage {
            id: MessageId::new(),
            tool_call_id,
            tool_name: tool_name.into(),
            content: vec![ContentBlock::Text {
                text: content.into(),
            }],
            is_error,
            timestamp: Utc::now(),
        })
    }
}

// ---------------------------------------------------------------------------
// AssistantMessage — helper methods
// ---------------------------------------------------------------------------

impl AssistantMessage {
    /// 提取所有文本内容，拼接为单个字符串。
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// 提取所有推理内容，拼接为单个字符串。
    pub fn reasoning(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::Reasoning { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// 提取所有工具调用。
    pub fn tool_calls(&self) -> Vec<&AssistantBlock> {
        self.content
            .iter()
            .filter(|b| matches!(b, AssistantBlock::ToolCall { .. }))
            .collect()
    }

    /// 是否包含工具调用。
    pub fn has_tool_calls(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, AssistantBlock::ToolCall { .. }))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ContentBlock --

    #[test]
    fn test_content_block_text_serde() {
        let block = ContentBlock::Text {
            text: "hello".into(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"text""#));
        let restored: ContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, restored);
    }

    #[test]
    fn test_content_block_image_serde() {
        let block = ContentBlock::Image {
            media_type: "image/png".into(),
            data: "iVBOR...".into(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"image""#));
        let restored: ContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, restored);
    }

    // -- AssistantBlock --

    #[test]
    fn test_assistant_block_text_serde() {
        let block = AssistantBlock::Text {
            text: "hi".into(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"text""#));
        let restored: AssistantBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, restored);
    }

    #[test]
    fn test_assistant_block_reasoning_serde() {
        let block = AssistantBlock::Reasoning {
            text: "let me think...".into(),
            signature: Some("sig123".into()),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"reasoning""#));
        let restored: AssistantBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, restored);
    }

    #[test]
    fn test_assistant_block_reasoning_no_signature() {
        let block = AssistantBlock::Reasoning {
            text: "thinking".into(),
            signature: None,
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(!json.contains("signature"));
    }

    #[test]
    fn test_assistant_block_tool_call_serde() {
        let block = AssistantBlock::ToolCall {
            id: ToolCallId::new("call_123"),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/tmp/test.rs"}),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"tool_call""#));
        let restored: AssistantBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, restored);
    }

    // -- UserContent --

    #[test]
    fn test_user_content_text() {
        let content = UserContent::from("hello");
        assert_eq!(content.text(), "hello");
        let blocks = content.into_blocks();
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "hello"));
    }

    #[test]
    fn test_user_content_blocks() {
        let content = UserContent::Blocks(vec![
            ContentBlock::Text {
                text: "look at this".into(),
            },
            ContentBlock::Image {
                media_type: "image/png".into(),
                data: "base64data".into(),
            },
        ]);
        assert_eq!(content.text(), "look at this");
    }

    #[test]
    fn test_user_content_serde_text() {
        let content = UserContent::Text("hello".into());
        let json = serde_json::to_string(&content).unwrap();
        assert_eq!(json, r#""hello""#);
        let restored: UserContent = serde_json::from_str(&json).unwrap();
        assert_eq!(content, restored);
    }

    #[test]
    fn test_user_content_serde_blocks() {
        let content = UserContent::Blocks(vec![ContentBlock::Text {
            text: "hi".into(),
        }]);
        let json = serde_json::to_string(&content).unwrap();
        let restored: UserContent = serde_json::from_str(&json).unwrap();
        assert_eq!(content, restored);
    }

    // -- Message constructors --

    #[test]
    fn test_message_user_constructor() {
        let msg = Message::user("hello");
        assert_eq!(msg.role(), Role::User);
        if let Message::User(u) = &msg {
            assert_eq!(u.content.text(), "hello");
        } else {
            panic!("expected User");
        }
    }

    #[test]
    fn test_message_assistant_constructor() {
        let msg = Message::assistant("hi there");
        assert_eq!(msg.role(), Role::Assistant);
        if let Message::Assistant(a) = &msg {
            assert_eq!(a.text(), "hi there");
            assert_eq!(a.finish_reason, FinishReason::Stop);
        } else {
            panic!("expected Assistant");
        }
    }

    #[test]
    fn test_message_tool_result_constructor() {
        let msg = Message::tool_result(
            ToolCallId::new("call_1"),
            "read_file",
            "file contents here",
            false,
        );
        assert_eq!(msg.role(), Role::Tool);
        if let Message::ToolResult(t) = &msg {
            assert_eq!(t.tool_call_id, ToolCallId::new("call_1"));
            assert_eq!(t.tool_name, "read_file");
            assert!(!t.is_error);
        } else {
            panic!("expected ToolResult");
        }
    }

    #[test]
    fn test_message_tool_result_error() {
        let msg = Message::tool_result(
            ToolCallId::new("call_2"),
            "write_file",
            "permission denied",
            true,
        );
        if let Message::ToolResult(t) = &msg {
            assert!(t.is_error);
        } else {
            panic!("expected ToolResult");
        }
    }

    // -- Message accessors --

    #[test]
    fn test_message_id_accessor() {
        let msg = Message::user("test");
        let id = msg.id().clone();
        assert_eq!(msg.id(), &id);
    }

    #[test]
    fn test_message_timestamp_accessor() {
        let before = Utc::now();
        let msg = Message::user("test");
        let after = Utc::now();
        assert!(msg.timestamp() >= before);
        assert!(msg.timestamp() <= after);
    }

    // -- AssistantMessage helpers --

    #[test]
    fn test_assistant_message_text() {
        let msg = AssistantMessage {
            id: MessageId::new(),
            content: vec![
                AssistantBlock::Reasoning {
                    text: "hmm".into(),
                    signature: None,
                },
                AssistantBlock::Text {
                    text: "Hello ".into(),
                },
                AssistantBlock::Text {
                    text: "World".into(),
                },
            ],
            model: "gpt-4o".into(),
            provider: "openai".into(),
            finish_reason: FinishReason::Stop,
            usage: None,
            timestamp: Utc::now(),
        };
        assert_eq!(msg.text(), "Hello World");
        assert_eq!(msg.reasoning(), "hmm");
    }

    #[test]
    fn test_assistant_message_tool_calls() {
        let msg = AssistantMessage {
            id: MessageId::new(),
            content: vec![
                AssistantBlock::Text {
                    text: "I'll read that file.".into(),
                },
                AssistantBlock::ToolCall {
                    id: ToolCallId::new("call_1"),
                    name: "read_file".into(),
                    arguments: serde_json::json!({"path": "foo.rs"}),
                },
                AssistantBlock::ToolCall {
                    id: ToolCallId::new("call_2"),
                    name: "grep".into(),
                    arguments: serde_json::json!({"pattern": "fn main"}),
                },
            ],
            model: "claude-sonnet-4-20250514".into(),
            provider: "anthropic".into(),
            finish_reason: FinishReason::ToolCalls,
            usage: None,
            timestamp: Utc::now(),
        };
        assert!(msg.has_tool_calls());
        assert_eq!(msg.tool_calls().len(), 2);
    }

    #[test]
    fn test_assistant_message_no_tool_calls() {
        let msg = AssistantMessage {
            id: MessageId::new(),
            content: vec![AssistantBlock::Text {
                text: "done".into(),
            }],
            model: String::new(),
            provider: String::new(),
            finish_reason: FinishReason::Stop,
            usage: None,
            timestamp: Utc::now(),
        };
        assert!(!msg.has_tool_calls());
        assert!(msg.tool_calls().is_empty());
    }

    // -- Message serde roundtrip --

    #[test]
    fn test_message_serde_roundtrip_user() {
        let msg = Message::user("hello world");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""role":"user""#));
        let restored: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg.role(), restored.role());
    }

    #[test]
    fn test_message_serde_roundtrip_assistant() {
        let msg = Message::assistant("reply");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""role":"assistant""#));
        let restored: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg.role(), restored.role());
    }

    #[test]
    fn test_message_serde_roundtrip_tool_result() {
        let msg = Message::tool_result(
            ToolCallId::new("call_x"),
            "bash",
            "exit code 0",
            false,
        );
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""role":"tool_result""#));
        let restored: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg.role(), restored.role());
    }
}
