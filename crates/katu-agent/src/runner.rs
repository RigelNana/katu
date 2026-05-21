//! # runner
//!
//! ## 职责
//! Agent Loop 核心驱动器 — 编排 LLM 调用、工具执行、重试与压缩，
//! 将 `AgentInstance` 与 `Session` 串联为自动化循环。
//!
//! ## 设计
//! - **单 while 循环** — 每次迭代 = 1 step（LLM 调用 + 工具执行 + 压缩检查）
//! - **无状态 Runner** — 所有可变状态在 `run()` 局部管理
//! - **取消协作** — 通过 Session 的 `CancellationToken` 响应中断
//! - **重试内嵌** — LLM 调用处用 `RetryState` 驱动退避重试
//! - **2 层压缩** — L1 Prune（每步）+ L2 Compact（按需）
//!
//! ## 流程
//! ```text
//! Runner::run(instance, compactor)
//!   loop {
//!     check_termination()
//!     build_llm_request()
//!     stream_with_retry()  → StreamResult
//!     if tool_calls → execute_tools → prune → compact?
//!     else → break Completed
//!   }
//! ```
//!
//! ## 调用者
//! - 应用层 — 构建 `AgentInstance` 后调用 `Runner::run()`

use tracing::{debug, info, warn};

use katu_core::agent_event::{AgentEvent, AgentFinishReason};
use katu_core::compaction::CompactTrigger;

use katu_llm::request::LlmRequest;

use crate::compaction::Compactor;
use crate::error::AgentError;
use crate::event_sender::AgentEventSenderExt;
use crate::instance::AgentInstance;
use crate::prompt::PromptContext;
use crate::retry::RetryState;
use crate::stream_consumer::{StreamConsumer, StreamConsumerError, StreamResult};
use crate::tool_executor::ToolExecutor;

// ===========================================================================
// RunOutcome
// ===========================================================================

/// Agent loop 终止原因。
///
/// 类型安全的结束信号 — 区分正常完成、资源限制、取消与错误。
///
/// # Examples
///
/// ```
/// use katu_agent::runner::RunOutcome;
///
/// let outcome = RunOutcome::completed(5);
/// assert!(outcome.is_completed());
/// assert_eq!(outcome.steps(), Some(5));
///
/// let outcome = RunOutcome::max_steps(50);
/// assert!(!outcome.is_completed());
/// ```
#[derive(Debug)]
pub enum RunOutcome {
    /// 正常结束 — 最后的 assistant 消息无 tool_use。
    Completed {
        steps: u32,
    },
    /// 达到步数上限。
    MaxSteps {
        limit: u32,
    },
    /// 用户/系统取消。
    Cancelled,
    /// 不可恢复的上下文溢出（压缩后仍超限）。
    ContextOverflow,
    /// 不可恢复的错误。
    Error(AgentError),
}

impl RunOutcome {
    /// 正常完成。
    pub fn completed(steps: u32) -> Self {
        Self::Completed { steps }
    }

    /// 达到步数上限。
    pub fn max_steps(limit: u32) -> Self {
        Self::MaxSteps { limit }
    }

    /// 是否正常完成。
    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }

    /// 返回步数（仅 Completed 时有值）。
    pub fn steps(&self) -> Option<u32> {
        match self {
            Self::Completed { steps } => Some(*steps),
            _ => None,
        }
    }

    /// 转换为 `AgentFinishReason`（用于事件发射）。
    pub fn finish_reason(&self) -> AgentFinishReason {
        match self {
            Self::Completed { .. } => AgentFinishReason::Completed,
            Self::MaxSteps { .. } => AgentFinishReason::MaxSteps,
            Self::Cancelled => AgentFinishReason::UserAbort,
            Self::ContextOverflow => AgentFinishReason::TokenBudget,
            Self::Error(_) => AgentFinishReason::Error,
        }
    }
}

// ===========================================================================
// Runner
// ===========================================================================

/// Agent Loop 驱动器 — 编排 LLM 调用与工具执行的核心循环。
///
/// ## 设计选择
/// - **无状态** — 一次 `run()` 调用内局部管理所有状态
/// - **不持有 Instance** — `run()` 接收 `&mut AgentInstance`，调用者保留所有权
/// - **依赖注入** — 通过 Instance 间接获取 provider/tools/hooks
///
/// # Examples
///
/// ```ignore
/// use katu_agent::runner::Runner;
///
/// let outcome = Runner::run(&mut instance, &compactor).await;
/// match outcome {
///     RunOutcome::Completed { steps } => println!("done in {steps} steps"),
///     RunOutcome::Cancelled => println!("cancelled"),
///     _ => {}
/// }
/// ```
pub struct Runner;

impl Runner {
    /// 执行 Agent loop 直到自然终止或错误。
    ///
    /// # 前置条件
    /// - `instance.session().status().is_idle()` — 会话空闲
    /// - instance 已通过 `InstanceBuilder` 完整构建
    ///
    /// # 行为
    /// 1. 进入 Running 状态，发射 `AgentStarted`
    /// 2. 循环：LLM 调用 → 工具执行 → 压缩检查
    /// 3. 退出时回到 Idle 状态，发射 `AgentEnded`
    ///
    /// # 取消
    /// 通过 `instance.session_mut().cancel()` 触发协作式取消。
    pub async fn run(
        instance: &mut AgentInstance,
        compactor: &dyn Compactor,
    ) -> RunOutcome {
        // ── 初始化 ──────────────────────────────────────────────
        instance.session_mut().begin_run();

        let session_id = instance.session().id().clone();
        let agent_name = instance.agent().name.to_string();
        let model_id = instance.model().id.clone();

        // 发射 AgentStarted
        instance.event_sender().emit_lossy(AgentEvent::AgentStarted {
            session_id: session_id.clone(),
            agent_name: agent_name.clone(),
            model_id: model_id.clone(),
        });

        info!(
            agent = %agent_name,
            model = %model_id,
            session = %session_id,
            "agent loop started"
        );

        // ── 主循环 ──────────────────────────────────────────────
        let outcome = Self::run_loop(instance, compactor).await;

        // ── 清理 ────────────────────────────────────────────────
        let session = instance.session_mut();
        if session.status().is_cancelled() {
            session.reset_after_cancel();
        } else {
            session.end_run();
        }

        // 发射 AgentEnded
        instance.event_sender().emit_lossy(AgentEvent::AgentEnded {
            session_id,
            finish_reason: outcome.finish_reason(),
            total_usage: Some(instance.session().total_usage().clone()),
            steps: instance.session().step_count(),
        });

        info!(
            agent = %agent_name,
            outcome = ?outcome.finish_reason(),
            steps = instance.session().step_count(),
            "agent loop ended"
        );

        outcome
    }

    /// 内部循环 — 分离以便 `run()` 专注生命周期管理。
    async fn run_loop(
        instance: &mut AgentInstance,
        compactor: &dyn Compactor,
    ) -> RunOutcome {
        // 解析 max_steps：RunConfig 覆盖 > Session 默认
        let max_steps = instance
            .config()
            .max_steps_override()
            .unwrap_or_else(|| instance.session().max_steps());

        loop {
            // ── 1. 终止条件检查 ─────────────────────────────────
            if let Some(outcome) = Self::check_termination(instance, max_steps) {
                return outcome;
            }

            let step = instance.session().step_count();

            // ── 2. 构建 LLM 请求 ───────────────────────────────
            let request = Self::build_llm_request(instance);

            // 发射 StepStarted
            instance.event_sender().emit_lossy(AgentEvent::StepStarted {
                step_index: step,
                model_id: instance.model().id.clone(),
                agent_name: instance.agent().name.to_string(),
            });

            // ── 3. 流式 LLM 调用（含重试） ─────────────────────
            let stream_result = Self::stream_with_retry(instance, request).await;

            let stream_result = match stream_result {
                Ok(result) => result,
                Err(outcome) => {
                    // 发射 StepFailed
                    let error_msg = match &outcome {
                        RunOutcome::Cancelled => "cancelled".to_string(),
                        RunOutcome::Error(e) => e.to_string(),
                        other => format!("{:?}", other.finish_reason()),
                    };
                    instance.event_sender().emit_lossy(AgentEvent::StepFailed {
                        step_index: step,
                        error: error_msg,
                    });
                    return outcome;
                }
            };

            // ── 4. 处理 LLM 响应 ───────────────────────────────
            let finish_reason = stream_result.finish_reason;
            let usage = stream_result.usage.clone();
            let assistant_msg = stream_result.message;

            // 更新 context tokens（来自 LLM 报告的 input_tokens）
            instance
                .session_mut()
                .set_context_tokens(usage.input_tokens as u64);

            // 检查是否有工具调用
            let has_tool_calls = assistant_msg.has_tool_calls();

            // 写入 session
            instance.session_mut().push_assistant(assistant_msg.clone());

            // 发射 StepEnded
            instance.event_sender().emit_lossy(AgentEvent::StepEnded {
                step_index: step,
                finish_reason,
                usage: Some(usage),
            });

            // ── 5. 无工具调用 → 正常结束 ───────────────────────
            if !has_tool_calls {
                return RunOutcome::completed(instance.session().step_count());
            }

            // ── 6. 递增步数 ─────────────────────────────────────
            instance.session_mut().increment_step();

            // ── 7. 执行工具 ─────────────────────────────────────
            let tool_result = {
                let executor = ToolExecutor::new(
                    instance.tools(),
                    instance.hooks(),
                    instance.ruleset(),
                    instance.event_sender(),
                    instance.session().cancel_token(),
                    instance.config().tool_executor(),
                );
                executor.execute_batch(&assistant_msg).await
            };

            match tool_result {
                Ok(batch) => {
                    // 写入工具结果
                    instance.session_mut().push_tool_results(batch.tool_results);

                    // 工具执行被中断
                    if batch.interrupted {
                        if instance.session().cancel_token().is_cancelled() {
                            return RunOutcome::Cancelled;
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "tool execution error");
                    return RunOutcome::Error(e.into());
                }
            }

            // ── 8. L1 Prune（每步） ────────────────────────────
            if let Err(e) = Self::run_prune(instance, compactor).await {
                warn!(error = %e, "prune failed (non-fatal)");
            }

            // ── 9. L2 Compact（按需） ──────────────────────────
            if instance.config().auto_compact() && instance.session().should_compact() {
                match Self::run_compact(instance, compactor, CompactTrigger::Auto).await {
                    Ok(()) => {}
                    Err(RunOutcome::ContextOverflow) => return RunOutcome::ContextOverflow,
                    Err(other) => return other,
                }
            }
        } // loop
    }

    // =====================================================================
    // 子步骤实现
    // =====================================================================

    /// 检查终止条件 — 返回 `Some(outcome)` 表示应退出循环。
    fn check_termination(
        instance: &AgentInstance,
        max_steps: u32,
    ) -> Option<RunOutcome> {
        let session = instance.session();

        // 取消
        if session.cancel_token().is_cancelled() {
            return Some(RunOutcome::Cancelled);
        }

        // 步数上限
        if session.step_count() >= max_steps {
            return Some(RunOutcome::max_steps(max_steps));
        }

        // 压缩熔断器（连续失败过多）
        let cs = session.compaction_state();
        let max_failures = session.compaction_config().max_consecutive_failures;
        if cs.is_circuit_broken(max_failures) {
            warn!(
                failures = cs.consecutive_failures(),
                "compaction circuit breaker tripped"
            );
            return Some(RunOutcome::ContextOverflow);
        }

        None
    }

    /// 构建 LLM 请求 — 组装 system prompt + 消息历史 + 工具定义。
    fn build_llm_request(instance: &mut AgentInstance) -> LlmRequest {
        // 需要先收集所有不可变引用所需的数据
        let agent = instance.agent().clone();
        let model_id = instance.model().id.clone();
        let provider_id = instance.model().provider.clone();
        let environment = instance.environment().clone();
        let tool_definitions = instance.tool_definitions().to_vec();
        let step_count = instance.session().step_count();
        let message_count = instance.session().message_count();

        // 构建 PromptContext
        let prompt_ctx = PromptContext::new(
            &agent,
            &model_id,
            &provider_id,
            &environment,
        )
        .with_tools(&tool_definitions)
        .with_step_count(step_count)
        .with_message_count(message_count);

        // 组装 system prompt（需要 &mut self）
        let prompt_output = instance.prompt_builder_mut().build(&prompt_ctx);

        // 构建请求
        let mut request = LlmRequest::new(instance.model().clone());
        request.system = Some(prompt_output.text);
        request.messages = instance.session().message_slice().to_vec();
        request.tools = instance.tool_definitions().to_vec();

        request
    }

    /// 流式 LLM 调用 — 含重试逻辑。
    ///
    /// 成功返回 `Ok(StreamResult)`，不可恢复时返回 `Err(RunOutcome)`。
    async fn stream_with_retry(
        instance: &mut AgentInstance,
        request: LlmRequest,
    ) -> Result<StreamResult, RunOutcome> {
        let mut retry_state = RetryState::new(instance.config().retry().clone());

        loop {
            // 取消检查
            if instance.session().cancel_token().is_cancelled() {
                return Err(RunOutcome::Cancelled);
            }

            // 调用 Provider 获取流
            let stream = match instance.provider().stream(request.clone()).await {
                Ok(s) => s,
                Err(e) => {
                    let agent_err: AgentError = e.into();
                    if agent_err.is_retryable() {
                        if let Some(delay) = retry_state.next_delay(None) {
                            instance.event_sender().emit_lossy(AgentEvent::Retried {
                                attempt: retry_state.attempt(),
                                error: agent_err.to_string(),
                                delay_ms: delay.as_millis() as u64,
                            });
                            debug!(
                                attempt = retry_state.attempt(),
                                delay_ms = delay.as_millis(),
                                error = %agent_err,
                                "retrying provider call"
                            );
                            tokio::time::sleep(delay).await;
                            continue;
                        }
                    }
                    return Err(RunOutcome::Error(agent_err));
                }
            };

            // 消费流
            let cancel = instance.session().cancel_token().clone();
            let model = instance.model().id.to_string();
            let provider = instance.model().provider.to_string();

            match StreamConsumer::consume(
                stream,
                instance.event_sender(),
                &cancel,
                model,
                provider,
            )
            .await
            {
                Ok(result) => return Ok(result),
                Err(StreamConsumerError::Cancelled) => {
                    return Err(RunOutcome::Cancelled);
                }
                Err(StreamConsumerError::Provider {
                    message,
                    retryable: true,
                }) => {
                    if let Some(delay) = retry_state.next_delay(None) {
                        instance.event_sender().emit_lossy(AgentEvent::Retried {
                            attempt: retry_state.attempt(),
                            error: message.clone(),
                            delay_ms: delay.as_millis() as u64,
                        });
                        debug!(
                            attempt = retry_state.attempt(),
                            delay_ms = delay.as_millis(),
                            error = %message,
                            "retrying after stream error"
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    return Err(RunOutcome::Error(AgentError::provider(message, true)));
                }
                Err(e) => {
                    return Err(RunOutcome::Error(e.into()));
                }
            }
        }
    }

    /// L1 Prune — 截断旧工具输出。
    async fn run_prune(
        instance: &mut AgentInstance,
        compactor: &dyn Compactor,
    ) -> crate::error::Result<()> {
        let outcome = compactor.prune(instance.session_mut()).await?;
        if outcome.has_effect() {
            debug!(
                tokens_freed = outcome.tokens_freed,
                parts_pruned = outcome.parts_pruned,
                "prune completed"
            );
            instance.event_sender().emit_lossy(AgentEvent::PruneCompleted {
                tokens_freed: outcome.tokens_freed,
                parts_pruned: outcome.parts_pruned,
            });
        }
        Ok(())
    }

    /// L2 Compact — 调用 LLM 生成摘要，重建消息历史。
    ///
    /// 成功返回 `Ok(())`，不可恢复时返回 `Err(RunOutcome)`。
    async fn run_compact(
        instance: &mut AgentInstance,
        compactor: &dyn Compactor,
        trigger: CompactTrigger,
    ) -> Result<(), RunOutcome> {
        let step = instance.session().step_count();

        // 防重复触发
        if instance.session().compaction_state().already_compacted_at(step) {
            debug!(step, "skipping duplicate compaction");
            return Ok(());
        }

        let tokens_before = instance.session().context_tokens();
        let strategy = instance.session().compaction_config().strategy;

        // 发射 CompactionStarted
        instance.event_sender().emit_lossy(AgentEvent::CompactionStarted {
            trigger: trigger.clone(),
            strategy,
            tokens_before,
        });

        info!(
            trigger = ?trigger,
            tokens_before,
            "starting compaction"
        );

        match compactor.compact(instance.session_mut(), trigger).await {
            Ok(result) => {
                let tokens_after = result.tokens_after;

                // 更新压缩状态
                let session = instance.session_mut();
                session.compaction_state_mut().record_success();
                session.compaction_state_mut().mark_compacted(step, tokens_after);

                // 更新 context tokens
                if let Some(after) = tokens_after {
                    session.set_context_tokens(after);
                }

                info!(
                    tokens_before = result.tokens_before,
                    tokens_after = ?result.tokens_after,
                    messages_compacted = result.messages_compacted,
                    "compaction completed"
                );

                // 发射 CompactionEnded
                instance.event_sender().emit_lossy(AgentEvent::CompactionEnded {
                    result,
                });

                Ok(())
            }
            Err(e) => {
                warn!(error = %e, "compaction failed");
                instance.session_mut().compaction_state_mut().record_failure();

                // 检查是否触发熔断器
                let max_failures = instance
                    .session()
                    .compaction_config()
                    .max_consecutive_failures;

                if instance.session().compaction_state().is_circuit_broken(max_failures) {
                    Err(RunOutcome::ContextOverflow)
                } else {
                    // 非致命 — 继续循环，下一步再试
                    Ok(())
                }
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_outcome_completed() {
        let outcome = RunOutcome::completed(5);
        assert!(outcome.is_completed());
        assert_eq!(outcome.steps(), Some(5));
        assert_eq!(outcome.finish_reason(), AgentFinishReason::Completed);
    }

    #[test]
    fn test_run_outcome_max_steps() {
        let outcome = RunOutcome::max_steps(50);
        assert!(!outcome.is_completed());
        assert_eq!(outcome.steps(), None);
        assert_eq!(outcome.finish_reason(), AgentFinishReason::MaxSteps);
    }

    #[test]
    fn test_run_outcome_cancelled() {
        let outcome = RunOutcome::Cancelled;
        assert!(!outcome.is_completed());
        assert_eq!(outcome.finish_reason(), AgentFinishReason::UserAbort);
    }

    #[test]
    fn test_run_outcome_context_overflow() {
        let outcome = RunOutcome::ContextOverflow;
        assert_eq!(outcome.finish_reason(), AgentFinishReason::TokenBudget);
    }

    #[test]
    fn test_run_outcome_error() {
        let outcome = RunOutcome::Error(AgentError::provider("test", false));
        assert!(!outcome.is_completed());
        assert_eq!(outcome.finish_reason(), AgentFinishReason::Error);
    }
}
