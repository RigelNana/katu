//! OpenAI Chat Completions 示例 — 演示流式和非流式调用。
//!
//! 运行方式：
//! ```bash
//! export OPENAI_API_KEY="sk-..."
//! cargo run -p katu-provider-openai --example chat
//! ```

use std::io::Write;

use tokio_stream::StreamExt;

use katu_core::{Message, ModelId, ProviderId, RouteId, StreamEvent};
use katu_llm::GenerationOptions;
use katu_llm::model::{ModelLimits, ModelRef};
use katu_llm::provider::Provider;
use katu_llm::request::LlmRequest;
use katu_provider_openai::OpenAiProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("请设置环境变量 OPENAI_API_KEY");

    let provider = OpenAiProvider::new();

    let model = ModelRef::new(
        ModelId::new("gemini-2.5-pro"),
        ProviderId::new("openai"),
        RouteId::new("openai-chat"),
        "https://yunwu.ai/v1",
        ModelLimits {
            context_window: 128_000,
            max_output_tokens: 16_384,
        },
    )
    .with_api_key(api_key);
    let request = LlmRequest::new(model)
        .with_system("You are a helpful assistant.")
        .with_message(Message::user("解释什么是 Rust 的所有权系统。"))
        .with_generation(
            GenerationOptions::new()
                .with_max_tokens(1024)
                .with_temperature(0.7),
        );

    let mut stream = provider.stream(request).await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta { delta, .. } => {
                print!("{delta}");
                std::io::stdout().flush().ok();
            }
            StreamEvent::Finish {
                finish_reason,
                usage,
                ..
            } => {
                println!("\n\n[完成] reason={finish_reason}");
                if let Some(u) = usage {
                    println!(
                        "[用量] input={} output={} total={}",
                        u.input_tokens, u.output_tokens, u.total_tokens
                    );
                }
            }
            _ => {}
        }
    }

    println!();

    Ok(())
}
