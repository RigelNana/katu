//! # output
//!
//! ## 职责
//! 命令输出收集与截断 — 参考 claude-code TaskOutput、oh-my-pi OutputSink、opencode tail()。
//!
//! ## 设计
//! - **OutputCollector** — 流式收集 stdout + stderr，保留 tail 和 head 两端
//! - 支持最大字节数限制，超出后丢弃中间部分
//! - 线程安全（`Arc<Mutex<...>>`）以支持异步 spawn 的输出回调

use std::sync::{Arc, Mutex};

// ===========================================================================
// OutputCollector
// ===========================================================================

/// 命令输出收集器。
///
/// 策略：保留前 `head_bytes` + 后 `tail_bytes`，中间截断。
/// 适用于命令输出可能极长（如 `find /` 或 `cat huge_file`）的场景。
///
/// # Examples
///
/// ```
/// use katu_tool::shell::OutputCollector;
///
/// let collector = OutputCollector::new(100, 100);
/// collector.push("hello world\n");
/// let snapshot = collector.snapshot();
/// assert_eq!(snapshot.text(), "hello world\n");
/// assert!(!snapshot.is_truncated());
/// ```
#[derive(Debug, Clone)]
pub struct OutputCollector {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Debug)]
struct Inner {
    /// 保留的头部。
    head: String,
    /// 保留的尾部（环形缓冲区 — 简单实现用 VecDeque<u8> 存储）。
    tail: TailBuffer,
    /// head 最大字节数。
    head_max: usize,
    /// 总接收字节数。
    total_bytes: usize,
    /// 总接收行数。
    total_lines: usize,
    /// head 是否已满。
    head_full: bool,
}

/// 简单 tail 缓冲区 — 保留最后 N 字节。
#[derive(Debug)]
struct TailBuffer {
    buf: Vec<u8>,
    max: usize,
}

impl TailBuffer {
    fn new(max: usize) -> Self {
        Self {
            buf: Vec::new(),
            max,
        }
    }

    fn push(&mut self, data: &[u8]) {
        if self.max == 0 {
            return;
        }
        self.buf.extend_from_slice(data);
        if self.buf.len() > self.max {
            let excess = self.buf.len() - self.max;
            self.buf.drain(..excess);
        }
    }

    fn as_str(&self) -> &str {
        // 跳过不完整的 UTF-8 前缀
        match std::str::from_utf8(&self.buf) {
            Ok(s) => s,
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                // 从第一个有效位置开始切割
                if valid_up_to == 0 {
                    // 尝试跳过无效前缀
                    for i in 1..4.min(self.buf.len()) {
                        if let Ok(s) = std::str::from_utf8(&self.buf[i..]) {
                            return s;
                        }
                    }
                    ""
                } else {
                    // 这种情况不该出现（末尾不完整），取 valid 部分
                    unsafe { std::str::from_utf8_unchecked(&self.buf[..valid_up_to]) }
                }
            }
        }
    }

    fn len(&self) -> usize {
        self.buf.len()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

impl OutputCollector {
    /// 创建收集器。
    ///
    /// - `head_bytes` — 保留头部最大字节数
    /// - `tail_bytes` — 保留尾部最大字节数
    pub fn new(head_bytes: usize, tail_bytes: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                head: String::new(),
                tail: TailBuffer::new(tail_bytes),
                head_max: head_bytes,
                total_bytes: 0,
                total_lines: 0,
                head_full: false,
            })),
        }
    }

    /// 使用默认限制创建（head 32KB + tail 32KB = 64KB 最大返回）。
    pub fn with_default_limits() -> Self {
        Self::new(32 * 1024, 32 * 1024)
    }

    /// 推入一段输出文本。
    pub fn push(&self, chunk: &str) {
        let mut inner = self.inner.lock().unwrap();
        inner.total_bytes += chunk.len();
        inner.total_lines += chunk.chars().filter(|&c| c == '\n').count();

        if !inner.head_full {
            let remaining = inner.head_max - inner.head.len();
            if chunk.len() <= remaining {
                inner.head.push_str(chunk);
                return;
            }
            // head 填满
            inner.head.push_str(&chunk[..remaining]);
            inner.head_full = true;
            // 剩余部分进入 tail
            let rest = &chunk[remaining..];
            inner.tail.push(rest.as_bytes());
        } else {
            inner.tail.push(chunk.as_bytes());
        }
    }

    /// 获取当前输出快照。
    pub fn snapshot(&self) -> OutputSnapshot {
        let inner = self.inner.lock().unwrap();
        let total_bytes = inner.total_bytes;
        let total_lines = inner.total_lines;

        if !inner.head_full {
            // 所有内容都在 head 中
            return OutputSnapshot {
                text: inner.head.clone(),
                total_bytes,
                total_lines,
                truncated: false,
            };
        }

        // head + tail，中间截断
        let tail_str = inner.tail.as_str();
        let kept_bytes = inner.head.len() + inner.tail.len();
        let truncated = total_bytes > kept_bytes;

        if truncated {
            let dropped = total_bytes - kept_bytes;
            let text = format!(
                "{}\n\n… [{} bytes truncated] …\n\n{}",
                inner.head.trim_end(),
                dropped,
                tail_str.trim_start()
            );
            OutputSnapshot {
                text,
                total_bytes,
                total_lines,
                truncated: true,
            }
        } else {
            // 没有实际截断（total == head + tail 还没溢出）
            let mut text = inner.head.clone();
            text.push_str(tail_str);
            OutputSnapshot {
                text,
                total_bytes,
                total_lines,
                truncated: false,
            }
        }
    }

    /// 当前总字节数。
    pub fn total_bytes(&self) -> usize {
        self.inner.lock().unwrap().total_bytes
    }

    /// 当前总行数。
    pub fn total_lines(&self) -> usize {
        self.inner.lock().unwrap().total_lines
    }

    /// 是否已截断。
    pub fn is_truncated(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.head_full && inner.total_bytes > (inner.head.len() + inner.tail.len())
    }
}

// ===========================================================================
// OutputSnapshot
// ===========================================================================

/// 输出快照 — `OutputCollector::snapshot()` 的返回值。
#[derive(Debug, Clone)]
pub struct OutputSnapshot {
    text: String,
    total_bytes: usize,
    total_lines: usize,
    truncated: bool,
}

impl OutputSnapshot {
    /// 输出文本（可能截断中间部分）。
    pub fn text(&self) -> &str {
        &self.text
    }

    /// 消费并返回文本。
    pub fn into_text(self) -> String {
        self.text
    }

    /// 原始总字节数。
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// 原始总行数。
    pub fn total_lines(&self) -> usize {
        self.total_lines
    }

    /// 是否发生了截断。
    pub fn is_truncated(&self) -> bool {
        self.truncated
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_output_no_truncation() {
        let collector = OutputCollector::new(100, 100);
        collector.push("hello ");
        collector.push("world\n");
        let snap = collector.snapshot();
        assert_eq!(snap.text(), "hello world\n");
        assert!(!snap.is_truncated());
        assert_eq!(snap.total_bytes(), 12);
        assert_eq!(snap.total_lines(), 1);
    }

    #[test]
    fn test_truncation() {
        let collector = OutputCollector::new(10, 10);
        // head: 10 bytes, tail: 10 bytes, middle gets dropped
        collector.push("AAAAAAAAAA"); // 10 bytes → fills head
        collector.push("BBBBBBBBBBBBBBBBBBBB"); // 20 bytes → overflow to tail
        collector.push("CCCCCCCCCC"); // 10 bytes → tail shifts

        let snap = collector.snapshot();
        assert!(snap.is_truncated());
        assert!(snap.text().contains("truncated"));
        assert_eq!(snap.total_bytes(), 40);
    }

    #[test]
    fn test_exact_head_fit() {
        let collector = OutputCollector::new(5, 5);
        collector.push("12345");
        let snap = collector.snapshot();
        assert_eq!(snap.text(), "12345");
        assert!(!snap.is_truncated());
    }

    #[test]
    fn test_head_and_tail_no_truncation() {
        // head=5, tail=5, push 8 bytes → head fills at 5, tail gets 3
        // total = 8, kept = 5 + 3 = 8 → no truncation
        let collector = OutputCollector::new(5, 5);
        collector.push("12345678");
        let snap = collector.snapshot();
        assert!(!snap.is_truncated());
        assert_eq!(snap.text(), "12345678");
    }

    #[test]
    fn test_head_and_tail_with_truncation() {
        // head=5, tail=5, push 5 + 10 + 5 = 20 bytes
        // head: "12345", tail: last 5 of remaining 15 = "EEEEE"
        // total=20, kept=10, truncated=true
        let collector = OutputCollector::new(5, 5);
        collector.push("12345"); // fills head
        collector.push("AAAAAAAAAA"); // 10 to tail
        collector.push("EEEEE"); // 5 to tail (tail keeps last 5)

        let snap = collector.snapshot();
        assert!(snap.is_truncated());
        assert!(snap.text().starts_with("12345"));
        assert!(snap.text().ends_with("EEEEE"));
        assert_eq!(snap.total_bytes(), 20);
    }

    #[test]
    fn test_line_counting() {
        let collector = OutputCollector::new(1000, 1000);
        collector.push("line1\nline2\nline3\n");
        assert_eq!(collector.total_lines(), 3);
    }

    #[test]
    fn test_empty() {
        let collector = OutputCollector::new(100, 100);
        let snap = collector.snapshot();
        assert_eq!(snap.text(), "");
        assert!(!snap.is_truncated());
        assert_eq!(snap.total_bytes(), 0);
    }

    #[test]
    fn test_thread_safe() {
        let collector = OutputCollector::new(1000, 1000);
        let c2 = collector.clone();
        let handle = std::thread::spawn(move || {
            c2.push("from thread");
        });
        handle.join().unwrap();
        assert_eq!(collector.snapshot().text(), "from thread");
    }
}
