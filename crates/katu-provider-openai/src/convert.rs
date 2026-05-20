//! LlmRequest ↔ OpenAI wire format 转换。

use katu_core::{
    AssistantBlock, ContentBlock, FinishReason, Message, ToolCallId, ToolChoice, ToolDefinition,
    Usage, UserContent,
};
use katu_llm::request::LlmRequest;

use crate::types::*;

// ===========================================================================
// LlmRequest → ChatCompletionRequest
// ===========================================================================

/// 将通用 `LlmRequest` 转换为 OpenAI wire format 请求。
pub fn build_request(req: &LlmRequest, stream: bool) -> ChatCompletionRequest {
    let gen_opts = req.resolved_generation();

    let mut messages = Vec::new();

    // System prompt
    if let Some(system) = &req.system {
        messages.push(ChatMessage::System {
            content: system.clone(),
        });
    }

    // Conversation messages
    for msg in &req.messages {
        messages.push(ChatMessage::from(msg));
    }

    // Tools
    let tools = if req.tools.is_empty() {
        None
    } else {
        Some(req.tools.iter().map(ToolParam::from).collect())
    };

    // Tool choice
    let tool_choice = req.tool_choice.as_ref().map(ToolChoiceParam::from);

    ChatCompletionRequest {
        model: req.model.id.as_str().to_owned(),
        messages,
        temperature: gen_opts.temperature,
        top_p: gen_opts.top_p,
        max_completion_tokens: gen_opts.max_tokens,
        frequency_penalty: gen_opts.frequency_penalty,
        presence_penalty: gen_opts.presence_penalty,
        stop: gen_opts.stop.map(|s| {
            if s.len() == 1 {
                StopParam::Single(s.into_iter().next().unwrap())
            } else {
                StopParam::Multiple(s)
            }
        }),
        seed: gen_opts.seed,
        stream: if stream { Some(true) } else { None },
        stream_options: if stream {
            Some(StreamOptionsParam {
                include_usage: true,
            })
        } else {
            None
        },
        tools,
        tool_choice,
        reasoning_effort: None, // TODO: 从 ModelRef.capabilities.thinking 推导
    }
}

// ===========================================================================
// From impls: katu types → OpenAI wire types
// ===========================================================================

impl From<&Message> for ChatMessage {
    fn from(msg: &Message) -> Self {
        match msg {
            Message::User(u) => ChatMessage::User {
                content: UserContentParam::from(&u.content),
            },
            Message::Assistant(a) => {
                let text = a.text();
                let content = if text.is_empty() { None } else { Some(text) };

                let tool_calls: Vec<AssistantToolCallParam> = a
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        AssistantBlock::ToolCall {
                            id,
                            name,
                            arguments,
                        } => Some(AssistantToolCallParam {
                            id: id.to_string(),
                            call_type: "function".to_owned(),
                            function: FunctionCallParam {
                                name: name.clone(),
                                arguments: arguments.to_string(),
                            },
                        }),
                        _ => None,
                    })
                    .collect();

                let tool_calls = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                };

                ChatMessage::Assistant {
                    content,
                    tool_calls,
                }
            }
            Message::ToolResult(t) => {
                let content_text = t
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                ChatMessage::Tool {
                    content: content_text,
                    tool_call_id: t.tool_call_id.to_string(),
                }
            }
        }
    }
}

impl From<&UserContent> for UserContentParam {
    fn from(content: &UserContent) -> Self {
        match content {
            UserContent::Text(text) => UserContentParam::Text(text.clone()),
            UserContent::Blocks(blocks) => {
                let parts = blocks
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } => {
                            ContentPartParam::Text { text: text.clone() }
                        }
                        ContentBlock::Image { data, .. } => ContentPartParam::ImageUrl {
                            image_url: ImageUrlParam {
                                url: format!("data:image/png;base64,{data}"),
                                detail: None,
                            },
                        },
                    })
                    .collect();
                UserContentParam::Parts(parts)
            }
        }
    }
}

impl From<&ToolDefinition> for ToolParam {
    fn from(tool: &ToolDefinition) -> Self {
        ToolParam {
            tool_type: "function".to_owned(),
            function: FunctionDefParam {
                name: tool.name.clone(),
                description: Some(tool.description.clone()),
                parameters: Some(tool.parameters.clone()),
                strict: None,
            },
        }
    }
}

impl From<&ToolChoice> for ToolChoiceParam {
    fn from(choice: &ToolChoice) -> Self {
        match choice {
            ToolChoice::Auto => ToolChoiceParam::Mode("auto".to_owned()),
            ToolChoice::None => ToolChoiceParam::Mode("none".to_owned()),
            ToolChoice::Required => ToolChoiceParam::Mode("required".to_owned()),
            ToolChoice::Specific { name } => ToolChoiceParam::Specific(NamedToolChoice {
                choice_type: "function".to_owned(),
                function: NamedToolChoiceFunction { name: name.clone() },
            }),
        }
    }
}

// ===========================================================================
// OpenAI response → katu types
// ===========================================================================

/// 将 OpenAI finish_reason 字符串转换为 katu FinishReason。
pub fn convert_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Unknown,
    }
}

/// 将 OpenAI CompletionUsage 转换为 katu Usage。
pub fn convert_usage(u: &CompletionUsage) -> Usage {
    Usage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
        cache_read_tokens: u
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0),
        reasoning_tokens: u
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        ..Default::default()
    }
}

/// 将 OpenAI tool call ID 字符串转为 katu ToolCallId。
pub fn to_tool_call_id(id: &str) -> ToolCallId {
    ToolCallId::new(id)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use katu_core::{ModelId, ProviderId, RouteId};
    use katu_llm::model::ModelLimits;
    use katu_llm::model::ModelRef;
    use katu_llm::GenerationOptions;

    fn sample_model() -> ModelRef {
        ModelRef::new(
            ModelId::new("gpt-4o"),
            ProviderId::new("openai"),
            RouteId::new("openai-chat"),
            "https://api.openai.com/v1",
            ModelLimits {
                context_window: 128_000,
                max_output_tokens: 4096,
            },
        )
    }

    #[test]
    fn test_build_request_basic() {
        let req = LlmRequest::new(sample_model())
            .with_system("You are helpful.")
            .with_message(Message::user("Hello"))
            .with_generation(GenerationOptions::new().with_max_tokens(1024).with_temperature(0.7));

        let wire = build_request(&req, true);
        assert_eq!(wire.model, "gpt-4o");
        assert_eq!(wire.messages.len(), 2); // system + user
        assert_eq!(wire.temperature, Some(0.7));
        assert_eq!(wire.max_completion_tokens, Some(1024));
        assert_eq!(wire.stream, Some(true));
        assert!(wire.stream_options.is_some());
        assert!(wire.tools.is_none());
    }

    #[test]
    fn test_build_request_with_tools() {
        let tools = vec![ToolDefinition::new(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        )];

        let req = LlmRequest::new(sample_model())
            .with_tools(tools)
            .with_tool_choice(ToolChoice::Auto);

        let wire = build_request(&req, false);
        assert!(wire.tools.is_some());
        assert_eq!(wire.tools.as_ref().unwrap().len(), 1);
        assert_eq!(wire.tools.as_ref().unwrap()[0].function.name, "read_file");
        assert!(wire.stream.is_none());
    }

    #[test]
    fn test_build_request_tool_choice_specific() {
        let req = LlmRequest::new(sample_model()).with_tool_choice(ToolChoice::specific("bash"));

        let wire = build_request(&req, false);
        let tc = wire.tool_choice.unwrap();
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "bash");
    }

    #[test]
    fn test_convert_finish_reason_mapping() {
        assert_eq!(convert_finish_reason("stop"), FinishReason::Stop);
        assert_eq!(convert_finish_reason("length"), FinishReason::Length);
        assert_eq!(convert_finish_reason("tool_calls"), FinishReason::ToolCalls);
        assert_eq!(
            convert_finish_reason("content_filter"),
            FinishReason::ContentFilter
        );
        assert_eq!(
            convert_finish_reason("unknown_reason"),
            FinishReason::Unknown
        );
    }

    #[test]
    fn test_convert_usage() {
        let openai_usage = CompletionUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: Some(20),
            }),
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(10),
            }),
        };

        let usage = convert_usage(&openai_usage);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.cache_read_tokens, 20);
        assert_eq!(usage.reasoning_tokens, Some(10));
    }

    #[test]
    fn test_request_serialization_snapshot() {
        let req = LlmRequest::new(sample_model())
            .with_system("Be concise.")
            .with_message(Message::user("Hi"));

        let wire = build_request(&req, true);
        let json = serde_json::to_value(&wire).unwrap();

        assert_eq!(json["model"], "gpt-4o");
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][0]["content"], "Be concise.");
        assert_eq!(json["messages"][1]["role"], "user");
        assert_eq!(json["stream"], true);
        assert_eq!(json["stream_options"]["include_usage"], true);
    }
}
