//! # Agent 完整调用流程示例 — OpenAI Provider（多轮交互）
//!
//! 演示如何使用 katu-agent 构建并运行一个**多轮对话** Agent loop。
//!
//! ## 流程
//! ```text
//! 1. 定义工具 (get_time, read_file)
//! 2. 创建 AgentDefinition + ModelRef + OpenAiProvider
//! 3. 通过 InstanceBuilder 构建 AgentInstance
//! 4. 创建 DefaultCompactor
//! 5. 循环:
//!    a. 读取用户输入 (stdin)
//!    b. 注入用户消息
//!    c. Runner::run() 驱动主循环
//!    d. 打印结果
//! 6. 异步消费 AgentEvent 事件流
//! ```
//!
//! ## 运行
//! ```bash
//! export OPENAI_API_KEY="sk-..."
//! cargo run -p katu-agent --example agent_openai
//! ```
//!
//! 输入 `quit` / `exit` / 空行退出。

use std::io::Write;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;

use katu_core::agent_event::AgentEvent;
use katu_core::{
    AgentDefinition, AgentRole, ModelId, ProviderId, Result, RouteId,
    Tool, ToolCallContext, ToolDefinition, ToolOutput,
};
use katu_llm::model::{ModelLimits, ModelRef};

use katu_agent::compaction::DefaultCompactor;
use katu_agent::instance::{InstanceBuilder, RunConfig};
use katu_agent::runner::Runner;

// ===========================================================================
// 工具定义
// ===========================================================================

/// 获取当前时间的工具。
struct GetTimeTool {
    def: ToolDefinition,
}

impl GetTimeTool {
    fn new() -> Self {
        Self {
            def: ToolDefinition::no_params("get_time", "获取当前 UTC 时间"),
        }
    }
}

#[async_trait]
impl Tool for GetTimeTool {
    fn definition(&self) -> &ToolDefinition {
        &self.def
    }

    async fn execute(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolCallContext,
    ) -> Result<ToolOutput> {
        let now = chrono::Utc::now().to_rfc3339();
        Ok(ToolOutput::success(now))
    }
}

/// 模拟读取文件的工具。
struct ReadFileTool {
    def: ToolDefinition,
}

impl ReadFileTool {
    fn new() -> Self {
        Self {
            def: ToolDefinition::new(
                "read_file",
                "读取指定路径的文件内容",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "文件路径"
                        }
                    },
                    "required": ["path"]
                }),
            ),
        }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> &ToolDefinition {
        &self.def
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: &ToolCallContext,
    ) -> Result<ToolOutput> {
        let path = args["path"]
            .as_str()
            .unwrap_or("<unknown>");

        // 真实实现会读取文件，这里模拟返回
        match std::fs::read_to_string(path) {
            Ok(content) => {
                // 截断过长内容
                if content.len() > 4096 {
                    Ok(ToolOutput::success(format!(
                        "{}...\n\n[截断: 原文 {} 字节]",
                        &content[..4096],
                        content.len()
                    )))
                } else {
                    Ok(ToolOutput::success(content))
                }
            }
            Err(e) => Ok(ToolOutput::error(format!(
                "无法读取文件 `{path}`: {e}"
            ))),
        }
    }
}

// ===========================================================================
// 事件打印
// ===========================================================================

/// 异步消费事件流，打印关键信息。
async fn print_events(mut rx: mpsc::UnboundedReceiver<AgentEvent>) {
    while let Some(event) = rx.recv().await {
        match &event {
            AgentEvent::AgentStarted {
                agent_name,
                model_id,
                ..
            } => {
                println!("\n╔══════════════════════════════════════════╗");
                println!("║  Agent 启动: {agent_name}");
                println!("║  模型: {model_id}");
                println!("╚══════════════════════════════════════════╝\n");
            }

            AgentEvent::StepStarted { step_index, .. } => {
                println!("── Step {step_index} ──────────────────────────");
            }

            AgentEvent::TextDelta { delta, .. } => {
                print!("{delta}");
            }

            AgentEvent::TextEnded { .. } => {
                println!();
            }

            AgentEvent::ToolCalled {
                tool_name, arguments, ..
            } => {
                let args_str = arguments.to_string();
                let args_short = if args_str.len() > 120 {
                    format!("{}...", &args_str[..120])
                } else {
                    args_str
                };
                println!("  🔧 调用工具: {tool_name}({args_short})");
            }

            AgentEvent::ToolSucceeded {
                tool_name, output, ..
            } => {
                let content = &output.content;
                let output_short = if content.len() > 200 {
                    format!("{}...", &content[..200])
                } else {
                    content.clone()
                };
                println!("  ✅ {tool_name} → {output_short}");
            }

            AgentEvent::ToolFailed {
                tool_name, error, ..
            } => {
                println!("  ❌ {tool_name} 失败: {error}");
            }

            AgentEvent::Retried {
                attempt,
                error,
                delay_ms,
            } => {
                println!("  ⟳ 重试 #{attempt} (等待 {delay_ms}ms): {error}");
            }

            AgentEvent::CompactionStarted { trigger, .. } => {
                println!("  📦 压缩开始 (trigger: {trigger:?})");
            }

            AgentEvent::CompactionEnded { result } => {
                println!(
                    "  📦 压缩完成: {}条消息 → 摘要 ({}→{:?} tokens)",
                    result.messages_compacted, result.tokens_before, result.tokens_after
                );
            }

            AgentEvent::AgentEnded {
                finish_reason,
                steps,
                total_usage,
                ..
            } => {
                println!("\n╔══════════════════════════════════════════╗");
                println!("║  Agent 结束: {finish_reason:?}");
                println!("║  总步数: {steps}");
                if let Some(usage) = total_usage {
                    println!(
                        "║  用量: input={} output={} total={}",
                        usage.input_tokens, usage.output_tokens, usage.total_tokens
                    );
                }
                println!("╚══════════════════════════════════════════╝");
            }

            // 其他事件静默忽略
            _ => {}
        }
    }
}

// ===========================================================================
// main
// ===========================================================================

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // ── 1. 环境变量 ──────────────────────────────────────────
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("请设置环境变量 OPENAI_API_KEY");

    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let model_name = std::env::var("OPENAI_MODEL")
        .unwrap_or_else(|_| "gpt-4o".to_string());

    // ── 2. 定义 Agent ────────────────────────────────────────
    let agent_def = AgentDefinition::new("katu-demo", AgentRole::Primary)
        .with_description("演示 Agent — 支持获取时间和读取文件")
        .with_max_steps(10);

    // ── 3. 配置模型 + Provider ───────────────────────────────
    let model = ModelRef::new(
        ModelId::new(&model_name),
        ProviderId::new("openai"),
        RouteId::new("openai-chat"),
        &base_url,
        ModelLimits {
            context_window: 128_000,
            max_output_tokens: 16_384,
        },
    )
    .with_api_key(&api_key);

    let provider: Arc<dyn katu_llm::Provider> =
        Arc::new(katu_provider_openai::OpenAiProvider::new());

    // ── 4. 创建工具 ─────────────────────────────────────────
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(GetTimeTool::new()),
        Arc::new(ReadFileTool::new()),
    ];

    // ── 5. 事件 channel ─────────────────────────────────────
    let (event_tx, event_rx) = mpsc::unbounded_channel::<AgentEvent>();

    // ── 6. 构建 AgentInstance ───────────────────────────────
    let config = RunConfig::new()
        .with_max_steps(10)
        .with_auto_compact(false); // 示例中禁用自动压缩

    let mut instance = InstanceBuilder::new(agent_def, model.clone(), provider.clone())
        .with_tools(tools)
        .with_event_sender(event_tx)
        .with_config(config)
        .build()?;

    // ── 7. 创建压缩器（与主模型共用） ──────────────────────
    let compactor = DefaultCompactor::new(provider.clone(), model.clone());

    // ── 8. 启动事件消费任务 ─────────────────────────────────
    let event_handle = tokio::spawn(print_events(event_rx));

    // ── 9. 多轮交互循环 ─────────────────────────────────────
    println!("╔══════════════════════════════════════════╗");
    println!("║  Katu Agent — 多轮交互示例                ║");
    println!("║  模型: {model_name:<34}║");
    println!("║  输入 quit/exit 或空行退出               ║");
    println!("╚══════════════════════════════════════════╝\n");

    let mut turn = 0u32;
    loop {
        turn += 1;

        // 读取用户输入
        print!("[Turn {turn}] 📝 You> ");
        std::io::stdout().flush().ok();

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();

        // 退出条件
        if input.is_empty() || input == "quit" || input == "exit" {
            println!("\n👋 再见！");
            break;
        }

        // 注入用户消息
        instance.session_mut().push_user(input);

        // 运行 Agent loop
        let outcome = Runner::run(&mut instance, &compactor).await;

        // 打印本轮结果
        println!("\n  📋 结果: {outcome:?}");

        if let Some(reply) = instance.session().last_assistant() {
            let text = reply.text();
            if !text.is_empty() {
                println!("  💬 回复: {text}");
            }
        }

        println!(
            "  📊 累计用量: input={} output={} total={}",
            instance.session().total_usage().input_tokens,
            instance.session().total_usage().output_tokens,
            instance.session().total_usage().total_tokens,
        );
        println!(
            "  📨 消息数: {}\n",
            instance.session().message_count(),
        );
    }

    // ── 10. 清理 ────────────────────────────────────────────
    drop(instance); // 释放 event_tx，使事件消费任务退出
    let _ = event_handle.await;

    Ok(())
}
