//! 调试用：打印原始 SSE 事件数据，不经过 chunk 解析
//!
//! cargo run -p katu-provider-openai --example debug_raw_sse

use tokio_stream::StreamExt;

use katu_core::{Message, ModelId, ProviderId, RouteId};
use katu_llm::model::{ModelLimits, ModelRef};
use katu_llm::request::LlmRequest;
use katu_llm::GenerationOptions;
use katu_provider_http::HttpProviderClient;

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

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Say hello in 3 words."}],
        "max_completion_tokens": 32,
        "stream": true,
        "stream_options": {"include_usage": true}
    });

    let request = LlmRequest::new(model)
        .with_message(Message::user("Say hello in 3 words."))
        .with_generation(GenerationOptions::new().with_max_tokens(32));

    let http = HttpProviderClient::new();
    let mut sse = http.post_sse(&request, "/chat/completions", &body).await?;

    let mut idx = 0;
    while let Some(event) = sse.next().await {
        match event {
            Ok(ev) => {
                println!("[{idx}] data={}", ev.data);
                // 尝试解析看能否成功
                match serde_json::from_str::<serde_json::Value>(&ev.data) {
                    Ok(v) => {
                        if let Some(choices) = v.get("choices") {
                            println!("     choices={choices}");
                        }
                        if let Some(usage) = v.get("usage") {
                            println!("     usage={usage}");
                        }
                    }
                    Err(e) => println!("     parse_err={e}"),
                }
            }
            Err(e) => {
                eprintln!("[{idx}] ERROR: {e}");
            }
        }
        idx += 1;
    }

    println!("\n共 {idx} 个 SSE 事件");
    Ok(())
}
