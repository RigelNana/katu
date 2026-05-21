//! # executor
//!
//! ## 职责
//! Shell 命令子进程管理 — spawn、流式输出收集、超时、取消。
//!
//! ## 设计
//! - **ShellExecutor** — 无状态执行器，持有配置（shell 路径、环境变量）
//! - **ShellResult** — 执行结果（exit code、输出快照、是否超时/取消）
//! - 使用 `tokio::process::Command` 异步 spawn
//! - 通过 `CancellationToken`（tokio_util）或 `AbortSignal` 实现外部取消
//! - 输出通过 `OutputCollector` 流式收集，自动截断

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::io::AsyncBufReadExt;
use tokio::process::Command;

use super::env::NonInteractiveEnv;
use super::output::{OutputCollector, OutputSnapshot};

// ===========================================================================
// ShellResult
// ===========================================================================

/// 命令执行结果。
#[derive(Debug, Clone)]
pub struct ShellResult {
    /// 退出码。`None` 表示被信号终止或未能获取。
    pub exit_code: Option<i32>,
    /// 输出快照。
    pub output: OutputSnapshot,
    /// 是否因超时被终止。
    pub timed_out: bool,
    /// 是否因外部取消而终止。
    pub cancelled: bool,
}

impl ShellResult {
    /// 命令是否成功（exit_code == 0）。
    pub fn is_success(&self) -> bool {
        self.exit_code == Some(0)
    }

    /// 输出文本。
    pub fn stdout(&self) -> &str {
        self.output.text()
    }
}

// ===========================================================================
// ShellExecutor
// ===========================================================================

/// Shell 命令执行器。
///
/// 负责 spawn 子进程、环境注入、输出流收集、超时/取消管理。
///
/// # Examples
///
/// ```no_run
/// use katu_tool::shell::{ShellExecutor, ShellResult};
/// use tokio_util::sync::CancellationToken;
///
/// # async fn example() {
/// let executor = ShellExecutor::new();
/// let result = executor.run(
///     "echo hello",
///     "/tmp",
///     std::time::Duration::from_secs(30),
///     CancellationToken::new(),
/// ).await;
/// assert!(result.is_success());
/// assert!(result.stdout().contains("hello"));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ShellExecutor {
    /// Shell 二进制路径（默认 /bin/bash）。
    shell: PathBuf,
    /// 非交互式环境变量。
    env: NonInteractiveEnv,
    /// 额外环境变量（用户传入）。
    extra_env: HashMap<String, String>,
    /// 输出 head 最大字节数。
    output_head_bytes: usize,
    /// 输出 tail 最大字节数。
    output_tail_bytes: usize,
}

impl Default for ShellExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ShellExecutor {
    /// 使用默认配置创建。
    pub fn new() -> Self {
        Self {
            shell: PathBuf::from("/bin/bash"),
            env: NonInteractiveEnv::default(),
            extra_env: HashMap::new(),
            output_head_bytes: 32 * 1024,
            output_tail_bytes: 32 * 1024,
        }
    }

    /// 设置 shell 路径。
    pub fn with_shell(mut self, shell: impl Into<PathBuf>) -> Self {
        self.shell = shell.into();
        self
    }

    /// 设置自定义非交互式环境。
    pub fn with_env(mut self, env: NonInteractiveEnv) -> Self {
        self.env = env;
        self
    }

    /// 添加额外环境变量。
    pub fn with_extra_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_env.insert(key.into(), value.into());
        self
    }

    /// 设置输出限制。
    pub fn with_output_limits(mut self, head_bytes: usize, tail_bytes: usize) -> Self {
        self.output_head_bytes = head_bytes;
        self.output_tail_bytes = tail_bytes;
        self
    }

    /// 执行命令。
    ///
    /// - `command` — Shell 命令字符串
    /// - `cwd` — 工作目录
    /// - `timeout` — 超时时长
    /// - `cancel` — 取消令牌
    pub async fn run(
        &self,
        command: &str,
        cwd: impl AsRef<Path>,
        timeout: Duration,
        cancel: tokio_util::sync::CancellationToken,
    ) -> ShellResult {
        self.run_with_env(command, cwd, timeout, cancel, None).await
    }

    /// 执行命令（带额外环境变量）。
    pub async fn run_with_env(
        &self,
        command: &str,
        cwd: impl AsRef<Path>,
        timeout: Duration,
        cancel: tokio_util::sync::CancellationToken,
        extra_env: Option<&HashMap<String, String>>,
    ) -> ShellResult {
        let collector = OutputCollector::new(self.output_head_bytes, self.output_tail_bytes);

        // 构建环境
        let mut env_map: HashMap<String, String> = self.env.vars().clone();
        for (k, v) in &self.extra_env {
            env_map.insert(k.clone(), v.clone());
        }
        if let Some(extra) = extra_env {
            for (k, v) in extra {
                env_map.insert(k.clone(), v.clone());
            }
        }

        // Spawn 子进程
        let mut child = match Command::new(&self.shell)
            .arg("-c")
            .arg(command)
            .current_dir(cwd.as_ref())
            .envs(&env_map)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .stdin(std::process::Stdio::null())
            // 设置进程组以便后续 kill 整个树
            .process_group(0)
            .spawn()
        {
            Ok(child) => child,
            Err(e) => {
                let error_output = OutputCollector::new(1024, 0);
                error_output.push(&format!("Failed to spawn shell: {e}"));
                return ShellResult {
                    exit_code: Some(126),
                    output: error_output.snapshot(),
                    timed_out: false,
                    cancelled: false,
                };
            }
        };

        // 获取输出流
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        // 异步读取 stdout + stderr
        let collector_for_stdout = collector.clone();
        let stdout_task = tokio::spawn(async move {
            if let Some(stdout) = stdout {
                let reader = tokio::io::BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    collector_for_stdout.push(&line);
                    collector_for_stdout.push("\n");
                }
            }
        });

        let collector_for_stderr = collector.clone();
        let stderr_task = tokio::spawn(async move {
            if let Some(stderr) = stderr {
                let reader = tokio::io::BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    collector_for_stderr.push(&line);
                    collector_for_stderr.push("\n");
                }
            }
        });

        // 等待完成 / 超时 / 取消
        let result = tokio::select! {
            status = child.wait() => {
                // 正常完成
                let _ = stdout_task.await;
                let _ = stderr_task.await;
                match status {
                    Ok(s) => ShellResult {
                        exit_code: s.code(),
                        output: collector.snapshot(),
                        timed_out: false,
                        cancelled: false,
                    },
                    Err(e) => {
                        collector.push(&format!("\nProcess wait error: {e}"));
                        ShellResult {
                            exit_code: None,
                            output: collector.snapshot(),
                            timed_out: false,
                            cancelled: false,
                        }
                    }
                }
            }
            _ = tokio::time::sleep(timeout) => {
                // 超时
                Self::kill_child(&mut child).await;
                let _ = stdout_task.await;
                let _ = stderr_task.await;
                collector.push(&format!(
                    "\n\nCommand timed out after {} seconds",
                    timeout.as_secs()
                ));
                ShellResult {
                    exit_code: None,
                    output: collector.snapshot(),
                    timed_out: true,
                    cancelled: false,
                }
            }
            _ = cancel.cancelled() => {
                // 外部取消
                Self::kill_child(&mut child).await;
                let _ = stdout_task.await;
                let _ = stderr_task.await;
                ShellResult {
                    exit_code: None,
                    output: collector.snapshot(),
                    timed_out: false,
                    cancelled: true,
                }
            }
        };

        result
    }

    /// 终止子进程（发送 SIGKILL 到进程组）。
    async fn kill_child(child: &mut tokio::process::Child) {
        // 尝试 kill 进程组
        if let Some(pid) = child.id() {
            let pgid = pid as i32;
            // 先 SIGTERM
            unsafe {
                libc::kill(-pgid, libc::SIGTERM);
            }
            // 等待一小段时间让进程优雅退出
            tokio::time::sleep(Duration::from_millis(200)).await;
            // 再 SIGKILL
            unsafe {
                libc::kill(-pgid, libc::SIGKILL);
            }
        }
        // 回收 zombie
        let _ = child.wait().await;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn test_simple_echo() {
        let executor = ShellExecutor::new();
        let result = executor
            .run(
                "echo hello",
                "/tmp",
                Duration::from_secs(10),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_success());
        assert!(result.stdout().contains("hello"));
    }

    #[tokio::test]
    async fn test_exit_code() {
        let executor = ShellExecutor::new();
        let result = executor
            .run(
                "exit 42",
                "/tmp",
                Duration::from_secs(10),
                CancellationToken::new(),
            )
            .await;
        assert_eq!(result.exit_code, Some(42));
        assert!(!result.is_success());
    }

    #[tokio::test]
    async fn test_timeout() {
        let executor = ShellExecutor::new();
        let result = executor
            .run(
                "sleep 60",
                "/tmp",
                Duration::from_millis(200),
                CancellationToken::new(),
            )
            .await;
        assert!(result.timed_out);
        assert!(!result.cancelled);
        assert_eq!(result.exit_code, None);
    }

    #[tokio::test]
    async fn test_cancel() {
        let executor = ShellExecutor::new();
        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();

        // 在短时间后取消
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            cancel2.cancel();
        });

        let result = executor
            .run(
                "sleep 60",
                "/tmp",
                Duration::from_secs(60),
                cancel,
            )
            .await;
        assert!(result.cancelled);
        assert!(!result.timed_out);
    }

    #[tokio::test]
    async fn test_stderr_captured() {
        let executor = ShellExecutor::new();
        let result = executor
            .run(
                "echo error >&2",
                "/tmp",
                Duration::from_secs(10),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_success());
        assert!(result.stdout().contains("error"));
    }

    #[tokio::test]
    async fn test_cwd() {
        let executor = ShellExecutor::new();
        let result = executor
            .run(
                "pwd",
                "/tmp",
                Duration::from_secs(10),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_success());
        // /tmp 可能是 /private/tmp 的符号链接 (macOS)
        assert!(
            result.stdout().contains("/tmp") || result.stdout().contains("/private/tmp"),
            "unexpected pwd: {}",
            result.stdout()
        );
    }

    #[tokio::test]
    async fn test_env_injection() {
        let executor = ShellExecutor::new();
        let result = executor
            .run(
                "echo $KATU",
                "/tmp",
                Duration::from_secs(10),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_success());
        assert!(result.stdout().contains("1"), "KATU env should be set to 1");
    }

    #[tokio::test]
    async fn test_multi_line_output() {
        let executor = ShellExecutor::new();
        let result = executor
            .run(
                "echo line1; echo line2; echo line3",
                "/tmp",
                Duration::from_secs(10),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_success());
        let text = result.stdout();
        assert!(text.contains("line1"));
        assert!(text.contains("line2"));
        assert!(text.contains("line3"));
    }

    #[tokio::test]
    async fn test_output_truncation() {
        // 极小的 output limit
        let executor = ShellExecutor::new().with_output_limits(20, 20);
        let result = executor
            .run(
                "seq 1 1000",
                "/tmp",
                Duration::from_secs(10),
                CancellationToken::new(),
            )
            .await;
        assert!(result.is_success());
        assert!(result.output.is_truncated());
    }

    #[tokio::test]
    async fn test_nonexistent_cwd() {
        let executor = ShellExecutor::new();
        let result = executor
            .run(
                "echo hello",
                "/nonexistent_dir_xyz_12345",
                Duration::from_secs(10),
                CancellationToken::new(),
            )
            .await;
        // spawn 应该失败
        assert_eq!(result.exit_code, Some(126));
        assert!(result.stdout().contains("spawn"));
    }
}
