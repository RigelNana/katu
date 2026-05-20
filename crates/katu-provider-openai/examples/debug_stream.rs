//! 调试用：打印所有 StreamEvent 查看流式输出
//!
//! cargo run -p katu-provider-openai --example debug_stream

use tokio_stream::StreamExt;

use katu_core::{Message, ModelId, ProviderId, RouteId};
use katu_llm::model::{ModelLimits, ModelRef};
use katu_llm::provider::Provider;
use katu_llm::request::LlmRequest;
use katu_llm::GenerationOptions;
use katu_provider_openai::OpenAiProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("需要 OPENAI_API_KEY");

    let model = ModelRef::new(
        ModelId::new("gpt-4o"),
        ProviderId::new("openai"),
        RouteId::new("openai-chat"),
        "https://yunwu.ai/v1",
        ModelLimits {
            context_window: 128_000,
            max_output_tokens: 16_384,
        },
    )
    .with_api_key(&api_key);

    let request = LlmRequest::new(model)
        .with_message(Message::user("Say hello in 3 words."))
        .with_generation(GenerationOptions::new().with_max_tokens(32));

    let provider = OpenAiProvider::new();
    let mut stream = provider.stream(request).await?;

    let mut idx = 0;
    while let Some(event) = stream.next().await {
        let event = event?;
        println!("[{idx}] {event:?}");
        idx += 1;
    }

    Ok(())
}
