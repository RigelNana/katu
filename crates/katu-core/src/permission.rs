//! # katu_core::permission
//!
//! ## 职责
//! 定义权限系统的类型与规则引擎 — 控制工具执行的授权机制。
//!
//! ## 设计
//! 参考三个参考实现的取舍：
//! - **Claude-Code 的层次化规则** — 多来源规则 + 优先级（简化为 5 层）
//! - **OpenCode 的通配符匹配** — 模式匹配规则引擎（采纳 last-match-wins 语义）
//! - **Oh-My-Pi 的简洁性** — 会话级缓存 + 4 种用户回复
//!
//! ## 权限检查管线
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        Agent Loop                                    │
//! │                                                                      │
//! │  1. 规则引擎求值 (Ruleset::evaluate)                                  │
//! │     Policy deny → 立即 Deny (不可覆盖)                               │
//! │     其他 deny  → Deny                                                │
//! │                                                                      │
//! │  2. Tool.check_permissions() (工具级检查)                             │
//! │     deny → Deny                                                      │
//! │     ask  → 继续                                                      │
//! │                                                                      │
//! │  3. 安全检查 (SafetyCheck)                                           │
//! │     敏感路径/危险操作 → Ask                                           │
//! │                                                                      │
//! │  4. 会话缓存 (SessionCache)                                          │
//! │     always_allow → Allow                                             │
//! │     always_deny  → Deny                                              │
//! │                                                                      │
//! │  5. 模式检查 + allow 规则                                             │
//! │     bypass 模式 → Allow                                               │
//! │     allow 规则  → Allow                                               │
//! │                                                                      │
//! │  6. 回退                                                              │
//! │     → Ask (请求用户决策)                                              │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## 对外接口
//! - `PermissionBehavior` — 权限行为三态 (allow / deny / ask)
//! - `PermissionMode` — 全局权限模式
//! - `RuleSource` — 规则来源（5 层优先级）
//! - `PermissionRule` — 单条权限规则
//! - `Ruleset` — 规则集合 + 求值引擎
//! - `PermissionRequest` — 权限请求
//! - `PermissionDecision` — 权限决策
//! - `PermissionReason` — 决策原因
//! - `PermissionReply` — 用户回复
//! - `SessionPermissionCache` — 会话级权限缓存
//! - `PermissionUpdate` — 权限规则更新动作
//!
//! ## 与 Hook 系统的关系
//! ```text
//! PreToolUse Hook → HookPermission (allow/deny/ask)
//!                         ↓
//! Permission System → 聚合 Hook 决策 + 规则 + 工具检查 → 最终决策
//! ```
//!
//! ## 调用者
//! - `katu-agent` (future) — Agent loop 在工具执行前调用
//! - Hook 系统 — HookPermission 作为输入之一

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::hook::HookPermission;
use crate::types::ToolCallId;

// ===========================================================================
// PermissionBehavior
// ===========================================================================

/// 权限行为三态。
///
/// 这是权限系统中最基本的决策单元：
/// - `Allow` — 允许执行
/// - `Deny` — 拒绝执行
/// - `Ask` — 需要用户确认
///
/// # 优先级
/// 在规则冲突时：`Deny > Ask > Allow`。
///
/// # Examples
///
/// ```
/// use katu_core::permission::PermissionBehavior;
///
/// let behavior = PermissionBehavior::Allow;
/// assert!(behavior.is_allow());
/// assert!(!behavior.is_deny());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionBehavior {
    /// 允许执行。
    Allow,
    /// 拒绝执行。
    Deny,
    /// 需要用户确认。
    Ask,
}

impl PermissionBehavior {
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }

    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Deny)
    }

    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask)
    }

    /// 严格度数值 — 用于冲突解决。
    ///
    /// Deny(2) > Ask(1) > Allow(0)。
    pub fn strictness(&self) -> u8 {
        match self {
            Self::Allow => 0,
            Self::Ask => 1,
            Self::Deny => 2,
        }
    }
}

// ===========================================================================
// PermissionMode
// ===========================================================================

/// 全局权限模式 — 控制 Agent 的整体权限策略。
///
/// 模式决定了"未被规则覆盖"的操作如何处理。
///
/// # Examples
///
/// ```
/// use katu_core::permission::PermissionMode;
///
/// let mode = PermissionMode::Default;
/// assert!(!mode.is_bypass());
/// assert!(PermissionMode::Bypass.is_bypass());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    /// 默认模式 — 危险操作询问用户。
    #[default]
    Default,

    /// 计划模式 — 只规划不执行，所有写操作 deny。
    Plan,

    /// 自动允许编辑 — 文件编辑类操作自动 allow，其他仍 ask。
    AcceptEdits,

    /// 绕过权限 — 跳过用户询问，直接 allow。
    ///
    /// # 安全限制
    /// - Policy deny 规则**不可**被绕过
    /// - 安全检查（敏感路径）**不可**被绕过
    Bypass,

    /// 不询问模式 — 需要 ask 的操作直接 deny（用于非交互环境）。
    NonInteractive,
}

impl PermissionMode {
    /// 是否为绕过模式。
    pub fn is_bypass(&self) -> bool {
        matches!(self, Self::Bypass)
    }

    /// 是否为非交互模式。
    pub fn is_non_interactive(&self) -> bool {
        matches!(self, Self::NonInteractive)
    }

    /// 是否为计划模式。
    pub fn is_plan(&self) -> bool {
        matches!(self, Self::Plan)
    }

    /// 对 ask 决策应用模式变换。
    ///
    /// - `Bypass` → ask 变为 allow
    /// - `NonInteractive` → ask 变为 deny
    /// - 其他 → 保持 ask
    pub fn transform_ask(&self) -> PermissionBehavior {
        match self {
            Self::Bypass => PermissionBehavior::Allow,
            Self::NonInteractive => PermissionBehavior::Deny,
            _ => PermissionBehavior::Ask,
        }
    }
}

// ===========================================================================
// RuleSource
// ===========================================================================

/// 规则来源 — 5 层优先级，高优先级来源的决策不可被低优先级覆盖。
///
/// ```text
/// Policy (最高) → User → Project → Session → Default (最低)
/// ```
///
/// # 设计参考
/// - Claude-Code: 7 种来源
/// - OpenCode: 无来源概念（flat ruleset）
/// - katu: 5 层，兼顾灵活性与简洁性
///
/// # Examples
///
/// ```
/// use katu_core::permission::RuleSource;
///
/// assert!(RuleSource::Policy.priority() > RuleSource::User.priority());
/// assert!(RuleSource::User.priority() > RuleSource::Project.priority());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleSource {
    /// 管理员策略 — 最高优先级，不可被用户覆盖。
    ///
    /// 来自企业管理平台或 `policySettings`。
    Policy,

    /// 用户全局设置 — 用户级偏好。
    ///
    /// 来自 `~/.config/katu/settings.toml`。
    User,

    /// 项目设置 — 项目级偏好。
    ///
    /// 来自 `.katu/settings.toml`（版本控制中共享）。
    Project,

    /// 会话 / 运行时 — 本次运行中动态添加的规则。
    ///
    /// 来自用户 "always allow" 选择或命令行参数。
    Session,

    /// 默认规则 — 最低优先级，内置兜底。
    Default,
}

impl RuleSource {
    /// 返回此来源的优先级数值（越大越高）。
    pub fn priority(&self) -> u8 {
        match self {
            Self::Policy => 100,
            Self::User => 80,
            Self::Project => 60,
            Self::Session => 40,
            Self::Default => 0,
        }
    }

    /// 此来源的规则是否不可被低优先级来源覆盖。
    pub fn is_immutable(&self) -> bool {
        matches!(self, Self::Policy)
    }
}

impl PartialOrd for RuleSource {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RuleSource {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority().cmp(&other.priority())
    }
}

// ===========================================================================
// PermissionRule
// ===========================================================================

/// 权限规则 — 基于模式匹配的授权条目。
///
/// 一条规则匹配 `(permission_key, content_pattern)` 二元组。
///
/// # 模式匹配语法
/// - 精确匹配：`"bash"`, `"/etc/shadow"`
/// - 通配符：`"read_*"`, `"*.rs"`, `"/home/*/project/*"`
/// - 全匹配：`"*"`
///
/// # Examples
///
/// ```
/// use katu_core::permission::{PermissionRule, PermissionBehavior, RuleSource};
///
/// // 允许所有 read 操作
/// let rule = PermissionRule::new(RuleSource::User, PermissionBehavior::Allow, "read", "*");
///
/// // 拒绝 bash 执行 rm 命令
/// let rule = PermissionRule::new(RuleSource::Project, PermissionBehavior::Deny, "bash", "rm *");
///
/// // 编辑 .rs 文件需要确认
/// let rule = PermissionRule::new(RuleSource::User, PermissionBehavior::Ask, "edit", "*.rs");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionRule {
    /// 规则来源。
    pub source: RuleSource,

    /// 权限行为。
    pub behavior: PermissionBehavior,

    /// 权限 key 的匹配模式（如 `"bash"`, `"edit"`, `"read_*"`）。
    pub permission: String,

    /// 内容的匹配模式（如 `"rm *"`, `"*.rs"`, `"/etc/*"`）。
    pub pattern: String,
}

impl PermissionRule {
    /// 创建新规则。
    pub fn new(
        source: RuleSource,
        behavior: PermissionBehavior,
        permission: impl Into<String>,
        pattern: impl Into<String>,
    ) -> Self {
        Self {
            source,
            behavior,
            permission: permission.into(),
            pattern: pattern.into(),
        }
    }

    /// 快捷构造：allow 规则。
    pub fn allow(source: RuleSource, permission: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self::new(source, PermissionBehavior::Allow, permission, pattern)
    }

    /// 快捷构造：deny 规则。
    pub fn deny(source: RuleSource, permission: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self::new(source, PermissionBehavior::Deny, permission, pattern)
    }

    /// 快捷构造：ask 规则。
    pub fn ask(source: RuleSource, permission: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self::new(source, PermissionBehavior::Ask, permission, pattern)
    }

    /// 检查此规则是否匹配给定的 (permission_key, content)。
    pub fn matches(&self, permission_key: &str, content: &str) -> bool {
        wildcard_match(permission_key, &self.permission)
            && wildcard_match(content, &self.pattern)
    }
}

// ===========================================================================
// Ruleset
// ===========================================================================

/// 规则集合 — 有序规则列表 + 求值引擎。
///
/// ## 求值语义
/// 按来源优先级分层求值：
/// 1. 从最高优先级来源开始
/// 2. 在同一来源内，最后匹配的规则获胜（last-match-wins，参考 OpenCode）
/// 3. 高优先级来源的结果不可被低优先级覆盖
/// 4. 无匹配 → 返回 None（由调用者决定默认行为）
///
/// # Examples
///
/// ```
/// use katu_core::permission::*;
///
/// let mut ruleset = Ruleset::new();
///
/// // 用户允许所有 read
/// ruleset.add(PermissionRule::allow(RuleSource::User, "read", "*"));
/// // 项目禁止读取 .env
/// ruleset.add(PermissionRule::deny(RuleSource::Project, "read", "*.env"));
///
/// // 求值：User(allow read *) 优先于 Project(deny read *.env)
/// let result = ruleset.evaluate("read", ".env");
/// // User 优先级更高，返回 Allow
/// assert_eq!(result, Some(PermissionBehavior::Allow));
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Ruleset {
    rules: Vec<PermissionRule>,
}

impl Ruleset {
    /// 创建空规则集。
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// 添加规则。
    pub fn add(&mut self, rule: PermissionRule) {
        self.rules.push(rule);
    }

    /// 批量添加规则。
    pub fn extend(&mut self, rules: impl IntoIterator<Item = PermissionRule>) {
        self.rules.extend(rules);
    }

    /// 移除指定来源的所有规则。
    pub fn remove_source(&mut self, source: RuleSource) {
        self.rules.retain(|r| r.source != source);
    }

    /// 规则数量。
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// 获取所有规则的只读引用。
    pub fn rules(&self) -> &[PermissionRule] {
        &self.rules
    }

    /// 求值 — 对给定的 (permission_key, content) 查找匹配的规则。
    ///
    /// ## 算法
    /// 1. 按来源优先级从高到低分层
    /// 2. 每层内找最后一条匹配规则（last-match-wins）
    /// 3. 最高优先级层的结果获胜
    /// 4. 无匹配返回 None
    ///
    /// # Examples
    ///
    /// ```
    /// use katu_core::permission::*;
    ///
    /// let mut ruleset = Ruleset::new();
    /// ruleset.add(PermissionRule::deny(RuleSource::Policy, "bash", "rm *"));
    /// ruleset.add(PermissionRule::allow(RuleSource::Session, "bash", "*"));
    ///
    /// // Policy deny 不可被 Session allow 覆盖
    /// assert_eq!(ruleset.evaluate("bash", "rm -rf /"), Some(PermissionBehavior::Deny));
    /// // 非 rm 命令被 Session allow 覆盖
    /// assert_eq!(ruleset.evaluate("bash", "ls -la"), Some(PermissionBehavior::Allow));
    /// ```
    pub fn evaluate(&self, permission_key: &str, content: &str) -> Option<PermissionBehavior> {
        // 按来源分层，从高优先级到低优先级
        for source in &[
            RuleSource::Policy,
            RuleSource::User,
            RuleSource::Project,
            RuleSource::Session,
            RuleSource::Default,
        ] {
            // 在此来源层内，找最后一条匹配的规则 (last-match-wins)
            let last_match = self
                .rules
                .iter()
                .filter(|r| r.source == *source)
                .filter(|r| r.matches(permission_key, content))
                .last();

            if let Some(rule) = last_match {
                return Some(rule.behavior);
            }
        }

        None
    }

    /// 快速检查 — 是否存在针对某 permission_key 的 deny 规则。
    ///
    /// 用于热路径的快速短路（不做 content 匹配）。
    pub fn has_deny_for(&self, permission_key: &str) -> bool {
        self.rules.iter().any(|r| {
            r.behavior.is_deny() && wildcard_match(permission_key, &r.permission)
        })
    }
}

// ===========================================================================
// PermissionRequest
// ===========================================================================

/// 权限请求 — 工具执行前向权限系统提交的授权请求。
///
/// 由 Agent loop 构造，传入权限系统进行求值。
///
/// # Examples
///
/// ```
/// use katu_core::permission::PermissionRequest;
/// use katu_core::ToolCallId;
/// use serde_json::json;
///
/// let req = PermissionRequest::new("bash", "rm -rf /tmp/cache")
///     .with_tool_name("bash")
///     .with_call_id(ToolCallId::new("call_1"))
///     .with_metadata(json!({"working_dir": "/home/user"}));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionRequest {
    /// 权限 key（如 `"bash"`, `"edit"`, `"read"`）。
    pub permission: String,

    /// 需检查的内容模式列表。
    ///
    /// 例如 bash 命令的完整内容、文件路径等。
    /// 所有 pattern 必须通过才能 allow。
    pub patterns: Vec<String>,

    /// 工具名称。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,

    /// 关联的 tool call ID。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<ToolCallId>,

    /// 额外元数据（用于 UI 展示和日志）。
    #[serde(default)]
    pub metadata: serde_json::Value,

    /// 如果用户选择 "always allow"，应持久化的模式列表。
    ///
    /// 可能与 `patterns` 不同 — 例如 bash 可能请求检查 `"rm -rf /tmp/cache"`，
    /// 但 always-allow 应记录为 `"rm *"`（更宽泛的模式）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub always_allow_patterns: Vec<String>,
}

impl PermissionRequest {
    /// 创建权限请求 — 单一 pattern。
    pub fn new(permission: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            permission: permission.into(),
            patterns: vec![pattern.into()],
            tool_name: None,
            call_id: None,
            metadata: serde_json::Value::Null,
            always_allow_patterns: Vec::new(),
        }
    }

    /// 创建权限请求 — 多 pattern。
    pub fn with_patterns(
        permission: impl Into<String>,
        patterns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            permission: permission.into(),
            patterns: patterns.into_iter().map(Into::into).collect(),
            tool_name: None,
            call_id: None,
            metadata: serde_json::Value::Null,
            always_allow_patterns: Vec::new(),
        }
    }

    /// 设置工具名称（builder 模式）。
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = Some(name.into());
        self
    }

    /// 设置 call ID（builder 模式）。
    pub fn with_call_id(mut self, id: ToolCallId) -> Self {
        self.call_id = Some(id);
        self
    }

    /// 设置元数据（builder 模式）。
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// 设置 always-allow 模式列表（builder 模式）。
    pub fn with_always_allow(mut self, patterns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.always_allow_patterns = patterns.into_iter().map(Into::into).collect();
        self
    }
}

// ===========================================================================
// PermissionDecision
// ===========================================================================

/// 权限决策 — 权限系统对请求的最终裁定。
///
/// # Examples
///
/// ```
/// use katu_core::permission::{PermissionDecision, PermissionReason, RuleSource};
///
/// let decision = PermissionDecision::allow(PermissionReason::Rule {
///     source: RuleSource::User,
/// });
/// assert!(decision.is_allow());
///
/// let decision = PermissionDecision::deny(
///     PermissionReason::Rule { source: RuleSource::Policy },
///     "Operation blocked by admin policy",
/// );
/// assert!(decision.is_deny());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "behavior", rename_all = "snake_case")]
pub enum PermissionDecision {
    /// 允许执行。
    Allow {
        reason: PermissionReason,
        /// Hook 或工具修改后的输入（如有）。
        #[serde(default, skip_serializing_if = "Option::is_none")]
        updated_input: Option<serde_json::Value>,
    },

    /// 拒绝执行。
    Deny {
        reason: PermissionReason,
        /// 展示给用户/LLM 的拒绝消息。
        message: String,
    },

    /// 需要用户确认。
    Ask {
        /// 展示给用户的确认消息。
        message: String,
        /// 建议的权限更新（用户选择 always 时应用）。
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        suggestions: Vec<PermissionUpdate>,
    },
}

impl PermissionDecision {
    /// 创建 Allow 决策。
    pub fn allow(reason: PermissionReason) -> Self {
        Self::Allow {
            reason,
            updated_input: None,
        }
    }

    /// 创建带修改输入的 Allow 决策。
    pub fn allow_with_input(reason: PermissionReason, updated_input: serde_json::Value) -> Self {
        Self::Allow {
            reason,
            updated_input: Some(updated_input),
        }
    }

    /// 创建 Deny 决策。
    pub fn deny(reason: PermissionReason, message: impl Into<String>) -> Self {
        Self::Deny {
            reason,
            message: message.into(),
        }
    }

    /// 创建 Ask 决策。
    pub fn ask(message: impl Into<String>) -> Self {
        Self::Ask {
            message: message.into(),
            suggestions: Vec::new(),
        }
    }

    /// 创建带建议的 Ask 决策。
    pub fn ask_with_suggestions(
        message: impl Into<String>,
        suggestions: Vec<PermissionUpdate>,
    ) -> Self {
        Self::Ask {
            message: message.into(),
            suggestions,
        }
    }

    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow { .. })
    }

    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Deny { .. })
    }

    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask { .. })
    }

    /// 获取决策的行为枚举。
    pub fn behavior(&self) -> PermissionBehavior {
        match self {
            Self::Allow { .. } => PermissionBehavior::Allow,
            Self::Deny { .. } => PermissionBehavior::Deny,
            Self::Ask { .. } => PermissionBehavior::Ask,
        }
    }
}

// ===========================================================================
// PermissionReason
// ===========================================================================

/// 权限决策原因 — 说明为什么做出此决策。
///
/// 用于审计日志和 UI 展示。
///
/// # Examples
///
/// ```
/// use katu_core::permission::{PermissionReason, RuleSource};
///
/// let reason = PermissionReason::Rule { source: RuleSource::Policy };
/// assert!(matches!(reason, PermissionReason::Rule { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PermissionReason {
    /// 匹配了某条规则。
    Rule { source: RuleSource },
    /// 权限模式决定（如 Bypass 自动 allow）。
    Mode,
    /// 工具级 check_permissions 返回。
    ToolCheck,
    /// Hook 系统决定。
    Hook,
    /// 会话缓存命中（用户之前选择了 always）。
    SessionCache,
    /// 安全检查触发（敏感路径等）。
    SafetyCheck,
    /// 用户直接决定（回复 allow/deny）。
    UserDecision,
}

// ===========================================================================
// PermissionResult
// ===========================================================================

/// 工具级权限检查结果 — `Tool::check_permissions()` 的返回值。
///
/// 与 `PermissionDecision` 不同，多了 `Passthrough` 变体表示
/// "我不关心，交给框架规则引擎决定"。
///
/// # Examples
///
/// ```
/// use katu_core::permission::PermissionResult;
///
/// // 工具不关心权限（大多数工具的默认行为）
/// let result = PermissionResult::Passthrough;
/// assert!(result.is_passthrough());
///
/// // 工具明确拒绝（如路径安全检查失败）
/// let result = PermissionResult::deny("Cannot write to /etc/");
/// assert!(result.is_deny());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PermissionResult {
    /// 允许执行。
    Allow,
    /// 拒绝执行。
    Deny { message: String },
    /// 需要用户确认。
    Ask { message: String },
    /// 不做决定 — 交由框架的规则引擎处理。
    Passthrough,
}

impl PermissionResult {
    /// 创建 Deny 结果。
    pub fn deny(message: impl Into<String>) -> Self {
        Self::Deny {
            message: message.into(),
        }
    }

    /// 创建 Ask 结果。
    pub fn ask(message: impl Into<String>) -> Self {
        Self::Ask {
            message: message.into(),
        }
    }

    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow)
    }

    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Deny { .. })
    }

    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask { .. })
    }

    pub fn is_passthrough(&self) -> bool {
        matches!(self, Self::Passthrough)
    }
}

// ===========================================================================
// PermissionReply
// ===========================================================================

/// 用户回复 — 对 Ask 决策的用户响应。
///
/// # 设计参考
/// - Oh-My-Pi: allow_once / allow_always / reject_once / reject_always (4 种)
/// - OpenCode: once / always / reject (3 种)
/// - katu: 4 种 + 可选反馈（参考 OpenCode 的 CorrectedError）
///
/// # Examples
///
/// ```
/// use katu_core::permission::PermissionReply;
///
/// let reply = PermissionReply::AllowOnce;
/// assert!(reply.is_allow());
///
/// let reply = PermissionReply::DenyWithFeedback {
///     feedback: "Use a safer command instead".into(),
/// };
/// assert!(reply.is_deny());
/// assert!(reply.has_feedback());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PermissionReply {
    /// 本次允许（不记忆）。
    AllowOnce,

    /// 始终允许（写入会话缓存 / 持久化）。
    AllowAlways,

    /// 本次拒绝。
    DenyOnce,

    /// 始终拒绝（写入会话缓存）。
    DenyAlways,

    /// 拒绝 + 反馈（LLM 可据此修正策略）。
    DenyWithFeedback { feedback: String },
}

impl PermissionReply {
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::AllowOnce | Self::AllowAlways)
    }

    pub fn is_deny(&self) -> bool {
        matches!(self, Self::DenyOnce | Self::DenyAlways | Self::DenyWithFeedback { .. })
    }

    pub fn is_always(&self) -> bool {
        matches!(self, Self::AllowAlways | Self::DenyAlways)
    }

    /// 是否携带用户反馈。
    pub fn has_feedback(&self) -> bool {
        matches!(self, Self::DenyWithFeedback { .. })
    }

    /// 获取反馈内容（如有）。
    pub fn feedback(&self) -> Option<&str> {
        match self {
            Self::DenyWithFeedback { feedback } => Some(feedback.as_str()),
            _ => None,
        }
    }

    /// 转换为行为枚举。
    pub fn behavior(&self) -> PermissionBehavior {
        if self.is_allow() {
            PermissionBehavior::Allow
        } else {
            PermissionBehavior::Deny
        }
    }
}

// ===========================================================================
// PermissionUpdate
// ===========================================================================

/// 权限规则更新动作 — 描述对规则集的变更。
///
/// 用于：
/// 1. 用户选择 "always allow" 后持久化规则
/// 2. Hook 建议的权限变更
/// 3. 配置文件同步
///
/// # Examples
///
/// ```
/// use katu_core::permission::*;
///
/// // 添加规则：always allow bash(ls *)
/// let update = PermissionUpdate::add_rule(
///     RuleSource::Session,
///     PermissionRule::allow(RuleSource::Session, "bash", "ls *"),
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum PermissionUpdate {
    /// 添加规则。
    AddRule {
        destination: RuleSource,
        rule: PermissionRule,
    },
    /// 移除匹配规则。
    RemoveRules {
        destination: RuleSource,
        permission: String,
        pattern: String,
    },
    /// 设置全局模式。
    SetMode {
        mode: PermissionMode,
    },
}

impl PermissionUpdate {
    /// 创建添加规则的更新。
    pub fn add_rule(destination: RuleSource, rule: PermissionRule) -> Self {
        Self::AddRule { destination, rule }
    }

    /// 创建移除规则的更新。
    pub fn remove_rules(
        destination: RuleSource,
        permission: impl Into<String>,
        pattern: impl Into<String>,
    ) -> Self {
        Self::RemoveRules {
            destination,
            permission: permission.into(),
            pattern: pattern.into(),
        }
    }

    /// 创建设置模式的更新。
    pub fn set_mode(mode: PermissionMode) -> Self {
        Self::SetMode { mode }
    }
}

// ===========================================================================
// SessionPermissionCache
// ===========================================================================

/// 会话级权限缓存 — 记录用户 "always" 决策。
///
/// ## 设计参考
/// - Oh-My-Pi: `Map<string, "allow_always" | "reject_always">`
/// - OpenCode: `approved: Ruleset` 累积规则
/// - katu: `Ruleset` 子集，只包含 Session 来源的 allow/deny 规则
///
/// ## 线程安全
/// 作为纯数据结构，线程安全由持有者负责（如 `Arc<RwLock<SessionPermissionCache>>`）。
///
/// # Examples
///
/// ```
/// use katu_core::permission::*;
///
/// let mut cache = SessionPermissionCache::new();
///
/// // 用户选择 "always allow bash(git *)"
/// cache.allow_always("bash", "git *");
///
/// // 下次检查时命中缓存
/// assert_eq!(cache.check("bash", "git pull"), Some(PermissionBehavior::Allow));
/// assert_eq!(cache.check("bash", "rm -rf /"), None); // 不在缓存中
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionPermissionCache {
    rules: Vec<CacheEntry>,
}

/// 缓存条目。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CacheEntry {
    permission: String,
    pattern: String,
    behavior: PermissionBehavior,
}

impl SessionPermissionCache {
    /// 创建空缓存。
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// 记录 "always allow"。
    pub fn allow_always(&mut self, permission: impl Into<String>, pattern: impl Into<String>) {
        self.rules.push(CacheEntry {
            permission: permission.into(),
            pattern: pattern.into(),
            behavior: PermissionBehavior::Allow,
        });
    }

    /// 记录 "always deny"。
    pub fn deny_always(&mut self, permission: impl Into<String>, pattern: impl Into<String>) {
        self.rules.push(CacheEntry {
            permission: permission.into(),
            pattern: pattern.into(),
            behavior: PermissionBehavior::Deny,
        });
    }

    /// 检查缓存 — 返回匹配的缓存决策。
    ///
    /// 使用 last-match-wins 语义。
    pub fn check(&self, permission_key: &str, content: &str) -> Option<PermissionBehavior> {
        self.rules
            .iter()
            .filter(|e| {
                wildcard_match(permission_key, &e.permission)
                    && wildcard_match(content, &e.pattern)
            })
            .last()
            .map(|e| e.behavior)
    }

    /// 缓存条目数量。
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// 清空缓存。
    pub fn clear(&mut self) {
        self.rules.clear();
    }

    /// 转换为可持久化的 Ruleset（Session 来源）。
    pub fn to_ruleset(&self) -> Ruleset {
        let mut ruleset = Ruleset::new();
        for entry in &self.rules {
            ruleset.add(PermissionRule::new(
                RuleSource::Session,
                entry.behavior,
                &entry.permission,
                &entry.pattern,
            ));
        }
        ruleset
    }
}

// ===========================================================================
// DenialTracker
// ===========================================================================

/// 连续拒绝追踪器 — 防止 LLM 陷入反复被拒绝的循环。
///
/// 参考 Claude-Code 的 circuit breaker 机制：
/// 连续 N 次拒绝后，向 LLM 注入警告或强制中断。
///
/// # Examples
///
/// ```
/// use katu_core::permission::DenialTracker;
///
/// let mut tracker = DenialTracker::new(3); // 阈值 = 3
///
/// tracker.record_denial();
/// tracker.record_denial();
/// assert!(!tracker.is_tripped());
///
/// tracker.record_denial();
/// assert!(tracker.is_tripped()); // 连续 3 次，触发
///
/// tracker.record_allow();
/// assert!(!tracker.is_tripped()); // 一次 allow 重置
/// ```
#[derive(Debug, Clone)]
pub struct DenialTracker {
    /// 连续拒绝次数。
    consecutive_denials: u32,
    /// 触发阈值。
    threshold: u32,
    /// 总拒绝次数（不重置）。
    total_denials: u32,
}

impl DenialTracker {
    /// 创建新的追踪器。
    pub fn new(threshold: u32) -> Self {
        Self {
            consecutive_denials: 0,
            threshold,
            total_denials: 0,
        }
    }

    /// 记录一次拒绝。
    pub fn record_denial(&mut self) {
        self.consecutive_denials += 1;
        self.total_denials += 1;
    }

    /// 记录一次允许（重置连续计数）。
    pub fn record_allow(&mut self) {
        self.consecutive_denials = 0;
    }

    /// 是否已触发阈值。
    pub fn is_tripped(&self) -> bool {
        self.consecutive_denials >= self.threshold
    }

    /// 当前连续拒绝次数。
    pub fn consecutive_count(&self) -> u32 {
        self.consecutive_denials
    }

    /// 总拒绝次数。
    pub fn total_count(&self) -> u32 {
        self.total_denials
    }

    /// 重置。
    pub fn reset(&mut self) {
        self.consecutive_denials = 0;
    }
}

// ===========================================================================
// Hook 集成 — 类型桥接
// ===========================================================================

/// `HookPermission` → `PermissionBehavior` 转换。
///
/// Hook 系统的权限决策映射到权限系统的行为三态。
///
/// # Examples
///
/// ```
/// use katu_core::hook::HookPermission;
/// use katu_core::permission::PermissionBehavior;
///
/// let behavior: PermissionBehavior = HookPermission::Allow.into();
/// assert_eq!(behavior, PermissionBehavior::Allow);
///
/// let behavior: PermissionBehavior = HookPermission::Deny { reason: Some("no".into()) }.into();
/// assert_eq!(behavior, PermissionBehavior::Deny);
/// ```
impl From<HookPermission> for PermissionBehavior {
    fn from(hook_perm: HookPermission) -> Self {
        match hook_perm {
            HookPermission::Allow => Self::Allow,
            HookPermission::Deny { .. } => Self::Deny,
            HookPermission::Ask { .. } => Self::Ask,
        }
    }
}

impl From<&HookPermission> for PermissionBehavior {
    fn from(hook_perm: &HookPermission) -> Self {
        match hook_perm {
            HookPermission::Allow => Self::Allow,
            HookPermission::Deny { .. } => Self::Deny,
            HookPermission::Ask { .. } => Self::Ask,
        }
    }
}

/// `PermissionBehavior` → `HookPermission` 转换。
impl From<PermissionBehavior> for HookPermission {
    fn from(behavior: PermissionBehavior) -> Self {
        match behavior {
            PermissionBehavior::Allow => Self::Allow,
            PermissionBehavior::Deny => Self::Deny { reason: None },
            PermissionBehavior::Ask => Self::Ask { message: None },
        }
    }
}

// ===========================================================================
// PermissionEngine — 管线协调器 trait
// ===========================================================================

/// 权限检查的完整输入上下文。
///
/// Agent loop 在工具执行前构造此结构，传入 `PermissionEngine::check()`。
#[derive(Debug, Clone)]
pub struct PermissionCheckInput {
    /// 权限请求。
    pub request: PermissionRequest,

    /// Hook 系统的聚合决策（如果 PreToolUse hook 有返回）。
    ///
    /// `None` 表示没有 Hook 参与或所有 Hook 都 passthrough。
    pub hook_decision: Option<HookPermission>,

    /// 工具级权限检查结果。
    ///
    /// `None` 表示工具未实现 `check_permissions`（等同于 Passthrough）。
    pub tool_check: Option<PermissionResult>,

    /// 当前权限模式。
    pub mode: PermissionMode,
}

impl PermissionCheckInput {
    /// 创建最小输入 — 只有请求和模式。
    pub fn new(request: PermissionRequest, mode: PermissionMode) -> Self {
        Self {
            request,
            hook_decision: None,
            tool_check: None,
            mode,
        }
    }

    /// 设置 Hook 决策（builder 模式）。
    pub fn with_hook_decision(mut self, decision: HookPermission) -> Self {
        self.hook_decision = Some(decision);
        self
    }

    /// 设置工具检查结果（builder 模式）。
    pub fn with_tool_check(mut self, result: PermissionResult) -> Self {
        self.tool_check = Some(result);
        self
    }
}

/// 权限引擎 trait — 定义完整的权限检查管线契约。
///
/// `katu-agent` 实现此 trait，协调规则引擎 + Hook 决策 + 工具检查 + 用户交互。
///
/// ## 管线执行顺序
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────────┐
/// │  PermissionEngine::check(input)                                   │
/// │                                                                   │
/// │  Phase 1: 不可覆盖层                                               │
/// │    ├─ Policy deny 规则 → 立即 Deny                                │
/// │    └─ Tool deny (check_permissions) → 立即 Deny                   │
/// │                                                                   │
/// │  Phase 2: Hook 决策                                               │
/// │    ├─ Hook deny → Deny (Hook deny 不可被规则 allow 覆盖)           │
/// │    └─ Hook ask → 记录 (后续可能被规则 allow 覆盖)                   │
/// │                                                                   │
/// │  Phase 3: 规则求值                                                 │
/// │    ├─ Rule deny → Deny                                            │
/// │    ├─ Rule allow → 但 Hook deny 仍然 deny                         │
/// │    └─ Rule ask → Ask                                              │
/// │                                                                   │
/// │  Phase 4: 缓存 + 模式                                             │
/// │    ├─ Session cache allow → Allow                                 │
/// │    ├─ Bypass mode → Allow (Policy deny 除外)                      │
/// │    └─ NonInteractive mode → Deny                                  │
/// │                                                                   │
/// │  Phase 5: 回退                                                    │
/// │    └─ → Ask                                                       │
/// └──────────────────────────────────────────────────────────────────┘
/// ```
///
/// ## 重要约束
/// - **Hook allow 不能绕过 Rule deny** — 规则是安全底线
/// - **Hook deny 不能被 Rule allow 覆盖** — Hook 拦截是显式安全决策
/// - **Policy deny 不可被任何方式覆盖** — 管理员策略是最终权威
///
/// # Examples
///
/// ```ignore
/// // katu-agent 中的实现示例（伪代码）
/// struct DefaultPermissionEngine {
///     ruleset: Ruleset,
///     cache: SessionPermissionCache,
///     denial_tracker: DenialTracker,
/// }
///
/// #[async_trait]
/// impl PermissionEngine for DefaultPermissionEngine {
///     async fn check(&self, input: PermissionCheckInput) -> PermissionDecision {
///         // Phase 1: Policy deny
///         if let Some(PermissionBehavior::Deny) = self.ruleset.evaluate_source(
///             RuleSource::Policy, &input.request.permission, &pattern
///         ) {
///             return PermissionDecision::deny(PermissionReason::Rule { source: RuleSource::Policy }, "Blocked by policy");
///         }
///
///         // Phase 2: Hook decision
///         if let Some(HookPermission::Deny { reason }) = &input.hook_decision {
///             return PermissionDecision::deny(PermissionReason::Hook, reason.as_deref().unwrap_or("Blocked by hook"));
///         }
///
///         // ... etc
///     }
/// }
/// ```
#[async_trait]
pub trait PermissionEngine: Send + Sync {
    /// 执行完整的权限检查管线。
    ///
    /// 返回 `PermissionDecision`：
    /// - `Allow` → Agent loop 继续执行工具
    /// - `Deny` → 将拒绝信息作为 ToolResult 反馈给 LLM
    /// - `Ask` → 调用 `prompt_user()` 获取用户回复
    async fn check(&self, input: PermissionCheckInput) -> PermissionDecision;

    /// 向用户提示权限确认（交互式模式）。
    ///
    /// 由 Agent loop 在收到 `Ask` 决策后调用。
    /// 返回用户的回复，Agent loop 据此决定下一步：
    /// - `AllowOnce` → 继续执行
    /// - `AllowAlways` → 更新缓存/规则 + 继续执行
    /// - `DenyOnce` → 中止本次
    /// - `DenyAlways` → 更新缓存 + 中止
    /// - `DenyWithFeedback` → 中止 + 将 feedback 注入 LLM
    async fn prompt_user(&self, decision: &PermissionDecision) -> PermissionReply;

    /// 应用用户回复 — 更新内部状态（缓存、规则等）。
    ///
    /// `request` 用于确定 always-allow 应持久化的 pattern。
    async fn apply_reply(
        &self,
        request: &PermissionRequest,
        reply: &PermissionReply,
    );
}

/// 默认权限检查算法 — 纯函数版本，不涉及用户交互。
///
/// 适合在不实现完整 `PermissionEngine` trait 的场景下使用。
/// 只执行 Phase 1-4（规则 + Hook + 工具 + 缓存 + 模式），不处理 Ask 的用户交互。
///
/// # Returns
/// - `Allow` / `Deny` — 确定性决策
/// - `Ask` — 需要上层调用 `prompt_user()`
///
/// # Examples
///
/// ```
/// use katu_core::hook::HookPermission;
/// use katu_core::permission::*;
///
/// let mut ruleset = Ruleset::new();
/// ruleset.add(PermissionRule::deny(RuleSource::Policy, "bash", "rm *"));
///
/// let cache = SessionPermissionCache::new();
///
/// let request = PermissionRequest::new("bash", "rm -rf /");
/// let input = PermissionCheckInput::new(request, PermissionMode::Default);
///
/// let decision = evaluate_permission(&ruleset, &cache, &input);
/// assert!(decision.is_deny());
/// ```
pub fn evaluate_permission(
    ruleset: &Ruleset,
    cache: &SessionPermissionCache,
    input: &PermissionCheckInput,
) -> PermissionDecision {
    let permission_key = &input.request.permission;

    // 对所有 pattern 进行检查（所有 pattern 必须 allow 才能 allow）
    // 任一 pattern deny → deny；任一 ask → ask（除非有 deny）
    let mut overall_behavior: Option<PermissionBehavior> = None;

    for pattern in &input.request.patterns {
        let decision = evaluate_single(ruleset, cache, input, permission_key, pattern);
        match decision {
            PermissionBehavior::Deny => {
                // 任何 deny → 立即返回
                let reason = determine_deny_reason(ruleset, input, permission_key, pattern);
                return PermissionDecision::deny(
                    reason,
                    format!("Permission denied: {}({})", permission_key, pattern),
                );
            }
            PermissionBehavior::Ask => {
                if overall_behavior != Some(PermissionBehavior::Deny) {
                    overall_behavior = Some(PermissionBehavior::Ask);
                }
            }
            PermissionBehavior::Allow => {
                if overall_behavior.is_none() {
                    overall_behavior = Some(PermissionBehavior::Allow);
                }
            }
        }
    }

    // 如果 patterns 为空，用 "*" 做检查
    if input.request.patterns.is_empty() {
        let decision = evaluate_single(ruleset, cache, input, permission_key, "*");
        overall_behavior = Some(decision);
        if decision.is_deny() {
            let reason = determine_deny_reason(ruleset, input, permission_key, "*");
            return PermissionDecision::deny(reason, format!("Permission denied: {}", permission_key));
        }
    }

    match overall_behavior.unwrap_or(PermissionBehavior::Ask) {
        PermissionBehavior::Allow => {
            PermissionDecision::allow(PermissionReason::Rule { source: RuleSource::Session })
        }
        PermissionBehavior::Ask => {
            PermissionDecision::ask(format!(
                "Allow {} to execute {}?",
                input.request.tool_name.as_deref().unwrap_or(permission_key),
                input.request.patterns.first().map(|s| s.as_str()).unwrap_or("*"),
            ))
        }
        PermissionBehavior::Deny => {
            // 已在循环中 early-return 处理，逻辑上不应到达这里
            unreachable!("Deny should have been returned early in the loop")
        }
    }
}

/// 单个 (permission_key, pattern) 的权限求值 — 完整管线。
fn evaluate_single(
    ruleset: &Ruleset,
    cache: &SessionPermissionCache,
    input: &PermissionCheckInput,
    permission_key: &str,
    pattern: &str,
) -> PermissionBehavior {
    // Phase 1: Policy deny（不可覆盖）
    let policy_rules: Vec<_> = ruleset
        .rules()
        .iter()
        .filter(|r| r.source == RuleSource::Policy && r.behavior.is_deny())
        .filter(|r| r.matches(permission_key, pattern))
        .collect();
    if !policy_rules.is_empty() {
        return PermissionBehavior::Deny;
    }

    // Phase 1b: Tool-level deny
    if let Some(ref tool_result) = input.tool_check {
        match tool_result {
            PermissionResult::Deny { .. } => return PermissionBehavior::Deny,
            PermissionResult::Ask { .. } => { /* 继续，后面处理 */ }
            _ => {}
        }
    }

    // Phase 2: Hook deny（不可被规则 allow 覆盖）
    if let Some(ref hook_decision) = input.hook_decision {
        if hook_decision.is_deny() {
            return PermissionBehavior::Deny;
        }
    }

    // Phase 3: 规则求值
    if let Some(rule_behavior) = ruleset.evaluate(permission_key, pattern) {
        match rule_behavior {
            PermissionBehavior::Deny => return PermissionBehavior::Deny,
            PermissionBehavior::Allow => {
                // 规则 allow — 但 Hook ask 可能要求确认
                if let Some(ref hook_decision) = input.hook_decision {
                    if hook_decision.is_ask() {
                        return PermissionBehavior::Ask;
                    }
                }
                // 工具 ask 也需要确认
                if let Some(PermissionResult::Ask { .. }) = input.tool_check {
                    return PermissionBehavior::Ask;
                }
                return PermissionBehavior::Allow;
            }
            PermissionBehavior::Ask => { /* 继续到缓存+模式检查 */ }
        }
    }

    // Phase 4: Session cache
    if let Some(cached) = cache.check(permission_key, pattern) {
        match cached {
            PermissionBehavior::Allow => return PermissionBehavior::Allow,
            PermissionBehavior::Deny => return PermissionBehavior::Deny,
            _ => {}
        }
    }

    // Phase 4b: Hook allow（规则没有 deny 时，Hook allow 可以生效）
    if let Some(ref hook_decision) = input.hook_decision {
        if hook_decision.is_allow() {
            return PermissionBehavior::Allow;
        }
    }

    // Phase 4c: Mode 变换
    match input.mode {
        PermissionMode::Bypass => return PermissionBehavior::Allow,
        PermissionMode::NonInteractive => return PermissionBehavior::Deny,
        PermissionMode::Plan => return PermissionBehavior::Deny,
        _ => {}
    }

    // Phase 5: 回退
    PermissionBehavior::Ask
}

/// 确定 deny 的具体原因。
fn determine_deny_reason(
    ruleset: &Ruleset,
    input: &PermissionCheckInput,
    permission_key: &str,
    pattern: &str,
) -> PermissionReason {
    // 检查是否是 Policy deny
    let is_policy = ruleset
        .rules()
        .iter()
        .any(|r| r.source == RuleSource::Policy && r.behavior.is_deny() && r.matches(permission_key, pattern));
    if is_policy {
        return PermissionReason::Rule { source: RuleSource::Policy };
    }

    // 检查是否是 Hook deny
    if let Some(ref hook) = input.hook_decision {
        if hook.is_deny() {
            return PermissionReason::Hook;
        }
    }

    // 检查是否是 Tool deny
    if let Some(PermissionResult::Deny { .. }) = &input.tool_check {
        return PermissionReason::ToolCheck;
    }

    // 检查规则来源
    for source in &[RuleSource::User, RuleSource::Project, RuleSource::Session, RuleSource::Default] {
        let has_deny = ruleset
            .rules()
            .iter()
            .any(|r| r.source == *source && r.behavior.is_deny() && r.matches(permission_key, pattern));
        if has_deny {
            return PermissionReason::Rule { source: *source };
        }
    }

    // Mode deny
    if input.mode.is_non_interactive() || input.mode.is_plan() {
        return PermissionReason::Mode;
    }

    PermissionReason::Rule { source: RuleSource::Default }
}

// ===========================================================================
// 通配符匹配
// ===========================================================================

/// 通配符模式匹配。
///
/// 支持 `*` 作为通配符（匹配零个或多个字符）。
/// 不支持 `?`（单字符通配符）— 保持简单。
///
/// # Examples
///
/// ```
/// use katu_core::permission::wildcard_match;
///
/// assert!(wildcard_match("hello", "hello"));
/// assert!(wildcard_match("hello", "*"));
/// assert!(wildcard_match("hello world", "hello *"));
/// assert!(wildcard_match("foo.rs", "*.rs"));
/// assert!(wildcard_match("/etc/shadow", "/etc/*"));
/// assert!(!wildcard_match("foo.ts", "*.rs"));
/// ```
pub fn wildcard_match(value: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if !pattern.contains('*') {
        return value == pattern;
    }

    let parts: Vec<&str> = pattern.split('*').collect();

    // 检查首段（必须是前缀）
    if !parts[0].is_empty() && !value.starts_with(parts[0]) {
        return false;
    }

    // 检查尾段（必须是后缀）
    let last = parts[parts.len() - 1];
    if !last.is_empty() && !value.ends_with(last) {
        return false;
    }

    // 逐段贪心匹配中间部分
    let mut pos = parts[0].len();
    for part in &parts[1..parts.len() - 1] {
        if part.is_empty() {
            continue;
        }
        match value[pos..].find(part) {
            Some(idx) => pos += idx + part.len(),
            None => return false,
        }
    }

    // 确保尾段不与中间段重叠
    if !last.is_empty() {
        let tail_start = value.len() - last.len();
        if pos > tail_start {
            return false;
        }
    }

    true
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- PermissionBehavior --

    #[test]
    fn test_behavior_variants() {
        assert!(PermissionBehavior::Allow.is_allow());
        assert!(PermissionBehavior::Deny.is_deny());
        assert!(PermissionBehavior::Ask.is_ask());
    }

    #[test]
    fn test_behavior_strictness_order() {
        assert!(PermissionBehavior::Deny.strictness() > PermissionBehavior::Ask.strictness());
        assert!(PermissionBehavior::Ask.strictness() > PermissionBehavior::Allow.strictness());
    }

    #[test]
    fn test_behavior_serde_roundtrip() {
        for b in [
            PermissionBehavior::Allow,
            PermissionBehavior::Deny,
            PermissionBehavior::Ask,
        ] {
            let json_str = serde_json::to_string(&b).unwrap();
            let restored: PermissionBehavior = serde_json::from_str(&json_str).unwrap();
            assert_eq!(b, restored);
        }
    }

    // -- PermissionMode --

    #[test]
    fn test_mode_default() {
        assert_eq!(PermissionMode::default(), PermissionMode::Default);
    }

    #[test]
    fn test_mode_transform_ask() {
        assert_eq!(PermissionMode::Default.transform_ask(), PermissionBehavior::Ask);
        assert_eq!(PermissionMode::Bypass.transform_ask(), PermissionBehavior::Allow);
        assert_eq!(PermissionMode::NonInteractive.transform_ask(), PermissionBehavior::Deny);
        assert_eq!(PermissionMode::Plan.transform_ask(), PermissionBehavior::Ask);
        assert_eq!(PermissionMode::AcceptEdits.transform_ask(), PermissionBehavior::Ask);
    }

    #[test]
    fn test_mode_serde_roundtrip() {
        for mode in [
            PermissionMode::Default,
            PermissionMode::Plan,
            PermissionMode::AcceptEdits,
            PermissionMode::Bypass,
            PermissionMode::NonInteractive,
        ] {
            let json_str = serde_json::to_string(&mode).unwrap();
            let restored: PermissionMode = serde_json::from_str(&json_str).unwrap();
            assert_eq!(mode, restored);
        }
    }

    // -- RuleSource --

    #[test]
    fn test_rule_source_priority_order() {
        assert!(RuleSource::Policy.priority() > RuleSource::User.priority());
        assert!(RuleSource::User.priority() > RuleSource::Project.priority());
        assert!(RuleSource::Project.priority() > RuleSource::Session.priority());
        assert!(RuleSource::Session.priority() > RuleSource::Default.priority());
    }

    #[test]
    fn test_rule_source_ord() {
        let mut sources = vec![
            RuleSource::Session,
            RuleSource::Policy,
            RuleSource::Default,
            RuleSource::User,
            RuleSource::Project,
        ];
        sources.sort();
        assert_eq!(
            sources,
            vec![
                RuleSource::Default,
                RuleSource::Session,
                RuleSource::Project,
                RuleSource::User,
                RuleSource::Policy,
            ]
        );
    }

    #[test]
    fn test_rule_source_immutable() {
        assert!(RuleSource::Policy.is_immutable());
        assert!(!RuleSource::User.is_immutable());
        assert!(!RuleSource::Session.is_immutable());
    }

    // -- PermissionRule --

    #[test]
    fn test_rule_new() {
        let rule = PermissionRule::new(
            RuleSource::User,
            PermissionBehavior::Allow,
            "read",
            "*",
        );
        assert_eq!(rule.source, RuleSource::User);
        assert_eq!(rule.behavior, PermissionBehavior::Allow);
        assert_eq!(rule.permission, "read");
        assert_eq!(rule.pattern, "*");
    }

    #[test]
    fn test_rule_shortcuts() {
        let allow = PermissionRule::allow(RuleSource::User, "read", "*");
        assert!(allow.behavior.is_allow());

        let deny = PermissionRule::deny(RuleSource::Policy, "bash", "rm *");
        assert!(deny.behavior.is_deny());

        let ask = PermissionRule::ask(RuleSource::Project, "edit", "*.env");
        assert!(ask.behavior.is_ask());
    }

    #[test]
    fn test_rule_matches_exact() {
        let rule = PermissionRule::deny(RuleSource::Policy, "bash", "rm -rf /");
        assert!(rule.matches("bash", "rm -rf /"));
        assert!(!rule.matches("bash", "ls -la"));
        assert!(!rule.matches("edit", "rm -rf /"));
    }

    #[test]
    fn test_rule_matches_wildcard() {
        let rule = PermissionRule::deny(RuleSource::Policy, "bash", "rm *");
        assert!(rule.matches("bash", "rm -rf /"));
        assert!(rule.matches("bash", "rm file.txt"));
        assert!(!rule.matches("bash", "ls -la"));
    }

    #[test]
    fn test_rule_matches_permission_wildcard() {
        let rule = PermissionRule::allow(RuleSource::User, "*", "*");
        assert!(rule.matches("bash", "anything"));
        assert!(rule.matches("edit", "anything"));
    }

    #[test]
    fn test_rule_serde_roundtrip() {
        let rule = PermissionRule::deny(RuleSource::Policy, "bash", "rm *");
        let json_str = serde_json::to_string(&rule).unwrap();
        let restored: PermissionRule = serde_json::from_str(&json_str).unwrap();
        assert_eq!(rule, restored);
    }

    // -- Ruleset --

    #[test]
    fn test_ruleset_empty() {
        let ruleset = Ruleset::new();
        assert!(ruleset.is_empty());
        assert_eq!(ruleset.evaluate("bash", "ls"), None);
    }

    #[test]
    fn test_ruleset_single_rule() {
        let mut ruleset = Ruleset::new();
        ruleset.add(PermissionRule::allow(RuleSource::User, "read", "*"));
        assert_eq!(ruleset.evaluate("read", "file.txt"), Some(PermissionBehavior::Allow));
        assert_eq!(ruleset.evaluate("bash", "ls"), None);
    }

    #[test]
    fn test_ruleset_last_match_wins_same_source() {
        let mut ruleset = Ruleset::new();
        ruleset.add(PermissionRule::allow(RuleSource::User, "bash", "*"));
        ruleset.add(PermissionRule::deny(RuleSource::User, "bash", "rm *"));

        // "rm -rf /" 匹配两条规则，last-match-wins → deny
        assert_eq!(ruleset.evaluate("bash", "rm -rf /"), Some(PermissionBehavior::Deny));
        // "ls" 只匹配第一条 → allow
        assert_eq!(ruleset.evaluate("bash", "ls"), Some(PermissionBehavior::Allow));
    }

    #[test]
    fn test_ruleset_higher_source_wins() {
        let mut ruleset = Ruleset::new();
        // Session: allow bash *
        ruleset.add(PermissionRule::allow(RuleSource::Session, "bash", "*"));
        // Policy: deny bash(rm *)
        ruleset.add(PermissionRule::deny(RuleSource::Policy, "bash", "rm *"));

        // Policy > Session: "rm -rf /" → deny (Policy wins)
        assert_eq!(ruleset.evaluate("bash", "rm -rf /"), Some(PermissionBehavior::Deny));
        // "ls" 只匹配 Session allow → allow
        assert_eq!(ruleset.evaluate("bash", "ls"), Some(PermissionBehavior::Allow));
    }

    #[test]
    fn test_ruleset_policy_deny_cannot_be_overridden() {
        let mut ruleset = Ruleset::new();
        ruleset.add(PermissionRule::deny(RuleSource::Policy, "bash", "rm *"));
        ruleset.add(PermissionRule::allow(RuleSource::User, "bash", "*"));
        ruleset.add(PermissionRule::allow(RuleSource::Session, "bash", "rm *"));

        // Policy deny 不可被任何低优先级覆盖
        assert_eq!(ruleset.evaluate("bash", "rm file"), Some(PermissionBehavior::Deny));
    }

    #[test]
    fn test_ruleset_remove_source() {
        let mut ruleset = Ruleset::new();
        ruleset.add(PermissionRule::allow(RuleSource::User, "read", "*"));
        ruleset.add(PermissionRule::deny(RuleSource::Session, "bash", "*"));
        assert_eq!(ruleset.len(), 2);

        ruleset.remove_source(RuleSource::Session);
        assert_eq!(ruleset.len(), 1);
        assert_eq!(ruleset.evaluate("bash", "ls"), None);
    }

    #[test]
    fn test_ruleset_has_deny_for() {
        let mut ruleset = Ruleset::new();
        ruleset.add(PermissionRule::deny(RuleSource::Policy, "bash", "rm *"));
        ruleset.add(PermissionRule::allow(RuleSource::User, "read", "*"));

        assert!(ruleset.has_deny_for("bash"));
        assert!(!ruleset.has_deny_for("read"));
        assert!(!ruleset.has_deny_for("edit"));
    }

    // -- PermissionRequest --

    #[test]
    fn test_permission_request_builder() {
        let req = PermissionRequest::new("bash", "rm -rf /tmp")
            .with_tool_name("bash")
            .with_call_id(ToolCallId::new("call_1"))
            .with_metadata(json!({"cwd": "/home/user"}))
            .with_always_allow(["rm *"]);

        assert_eq!(req.permission, "bash");
        assert_eq!(req.patterns, vec!["rm -rf /tmp"]);
        assert_eq!(req.tool_name, Some("bash".into()));
        assert_eq!(req.call_id.unwrap().as_str(), "call_1");
        assert_eq!(req.always_allow_patterns, vec!["rm *"]);
    }

    #[test]
    fn test_permission_request_multi_pattern() {
        let req = PermissionRequest::with_patterns("edit", ["src/main.rs", "src/lib.rs"]);
        assert_eq!(req.patterns.len(), 2);
    }

    // -- PermissionDecision --

    #[test]
    fn test_decision_allow() {
        let d = PermissionDecision::allow(PermissionReason::Rule {
            source: RuleSource::User,
        });
        assert!(d.is_allow());
        assert_eq!(d.behavior(), PermissionBehavior::Allow);
    }

    #[test]
    fn test_decision_deny() {
        let d = PermissionDecision::deny(
            PermissionReason::Rule {
                source: RuleSource::Policy,
            },
            "Blocked by policy",
        );
        assert!(d.is_deny());
        if let PermissionDecision::Deny { message, .. } = &d {
            assert_eq!(message, "Blocked by policy");
        }
    }

    #[test]
    fn test_decision_ask() {
        let d = PermissionDecision::ask("Allow bash to run 'rm -rf /tmp'?");
        assert!(d.is_ask());
    }

    #[test]
    fn test_decision_ask_with_suggestions() {
        let d = PermissionDecision::ask_with_suggestions(
            "Allow?",
            vec![PermissionUpdate::add_rule(
                RuleSource::Session,
                PermissionRule::allow(RuleSource::Session, "bash", "rm *"),
            )],
        );
        if let PermissionDecision::Ask { suggestions, .. } = &d {
            assert_eq!(suggestions.len(), 1);
        }
    }

    // -- PermissionResult --

    #[test]
    fn test_permission_result_variants() {
        assert!(PermissionResult::Allow.is_allow());
        assert!(PermissionResult::deny("no").is_deny());
        assert!(PermissionResult::ask("confirm?").is_ask());
        assert!(PermissionResult::Passthrough.is_passthrough());
    }

    // -- PermissionReply --

    #[test]
    fn test_reply_is_allow() {
        assert!(PermissionReply::AllowOnce.is_allow());
        assert!(PermissionReply::AllowAlways.is_allow());
        assert!(!PermissionReply::DenyOnce.is_allow());
    }

    #[test]
    fn test_reply_is_deny() {
        assert!(PermissionReply::DenyOnce.is_deny());
        assert!(PermissionReply::DenyAlways.is_deny());
        assert!(PermissionReply::DenyWithFeedback {
            feedback: "use a safer command".into()
        }
        .is_deny());
        assert!(!PermissionReply::AllowOnce.is_deny());
    }

    #[test]
    fn test_reply_is_always() {
        assert!(PermissionReply::AllowAlways.is_always());
        assert!(PermissionReply::DenyAlways.is_always());
        assert!(!PermissionReply::AllowOnce.is_always());
        assert!(!PermissionReply::DenyOnce.is_always());
    }

    #[test]
    fn test_reply_feedback() {
        let reply = PermissionReply::DenyWithFeedback {
            feedback: "try ls instead".into(),
        };
        assert!(reply.has_feedback());
        assert_eq!(reply.feedback(), Some("try ls instead"));

        assert!(!PermissionReply::DenyOnce.has_feedback());
        assert_eq!(PermissionReply::DenyOnce.feedback(), None);
    }

    #[test]
    fn test_reply_serde_roundtrip() {
        for reply in [
            PermissionReply::AllowOnce,
            PermissionReply::AllowAlways,
            PermissionReply::DenyOnce,
            PermissionReply::DenyAlways,
            PermissionReply::DenyWithFeedback {
                feedback: "feedback".into(),
            },
        ] {
            let json_str = serde_json::to_string(&reply).unwrap();
            let restored: PermissionReply = serde_json::from_str(&json_str).unwrap();
            assert_eq!(reply, restored);
        }
    }

    // -- SessionPermissionCache --

    #[test]
    fn test_cache_empty() {
        let cache = SessionPermissionCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.check("bash", "ls"), None);
    }

    #[test]
    fn test_cache_allow_always() {
        let mut cache = SessionPermissionCache::new();
        cache.allow_always("bash", "git *");

        assert_eq!(cache.check("bash", "git pull"), Some(PermissionBehavior::Allow));
        assert_eq!(cache.check("bash", "git push"), Some(PermissionBehavior::Allow));
        assert_eq!(cache.check("bash", "rm -rf /"), None);
    }

    #[test]
    fn test_cache_deny_always() {
        let mut cache = SessionPermissionCache::new();
        cache.deny_always("bash", "rm *");

        assert_eq!(cache.check("bash", "rm file.txt"), Some(PermissionBehavior::Deny));
        assert_eq!(cache.check("bash", "ls"), None);
    }

    #[test]
    fn test_cache_last_match_wins() {
        let mut cache = SessionPermissionCache::new();
        cache.allow_always("bash", "*");
        cache.deny_always("bash", "rm *");

        // "rm file" 匹配两条，last-match deny 获胜
        assert_eq!(cache.check("bash", "rm file"), Some(PermissionBehavior::Deny));
        // "ls" 只匹配第一条 → allow
        assert_eq!(cache.check("bash", "ls"), Some(PermissionBehavior::Allow));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = SessionPermissionCache::new();
        cache.allow_always("bash", "*");
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.check("bash", "ls"), None);
    }

    #[test]
    fn test_cache_to_ruleset() {
        let mut cache = SessionPermissionCache::new();
        cache.allow_always("bash", "git *");
        cache.deny_always("bash", "rm *");

        let ruleset = cache.to_ruleset();
        assert_eq!(ruleset.len(), 2);
        assert_eq!(ruleset.evaluate("bash", "git pull"), Some(PermissionBehavior::Allow));
        assert_eq!(ruleset.evaluate("bash", "rm file"), Some(PermissionBehavior::Deny));
    }

    // -- DenialTracker --

    #[test]
    fn test_denial_tracker_basic() {
        let tracker = DenialTracker::new(3);
        assert!(!tracker.is_tripped());
        assert_eq!(tracker.consecutive_count(), 0);
        assert_eq!(tracker.total_count(), 0);
    }

    #[test]
    fn test_denial_tracker_trips_at_threshold() {
        let mut tracker = DenialTracker::new(3);
        tracker.record_denial();
        tracker.record_denial();
        assert!(!tracker.is_tripped());

        tracker.record_denial();
        assert!(tracker.is_tripped());
        assert_eq!(tracker.consecutive_count(), 3);
        assert_eq!(tracker.total_count(), 3);
    }

    #[test]
    fn test_denial_tracker_allow_resets() {
        let mut tracker = DenialTracker::new(3);
        tracker.record_denial();
        tracker.record_denial();
        tracker.record_allow(); // 重置连续计数

        assert!(!tracker.is_tripped());
        assert_eq!(tracker.consecutive_count(), 0);
        assert_eq!(tracker.total_count(), 2); // 总数不重置
    }

    #[test]
    fn test_denial_tracker_reset() {
        let mut tracker = DenialTracker::new(2);
        tracker.record_denial();
        tracker.record_denial();
        assert!(tracker.is_tripped());

        tracker.reset();
        assert!(!tracker.is_tripped());
    }

    // -- wildcard_match --

    #[test]
    fn test_wildcard_exact() {
        assert!(wildcard_match("hello", "hello"));
        assert!(!wildcard_match("hello", "world"));
    }

    #[test]
    fn test_wildcard_star_all() {
        assert!(wildcard_match("anything", "*"));
        assert!(wildcard_match("", "*"));
    }

    #[test]
    fn test_wildcard_prefix() {
        assert!(wildcard_match("hello world", "hello *"));
        assert!(wildcard_match("hello", "hello*"));
        assert!(!wildcard_match("world hello", "hello *"));
    }

    #[test]
    fn test_wildcard_suffix() {
        assert!(wildcard_match("file.rs", "*.rs"));
        assert!(wildcard_match("test.rs", "*.rs"));
        assert!(!wildcard_match("file.ts", "*.rs"));
    }

    #[test]
    fn test_wildcard_middle() {
        assert!(wildcard_match("/etc/shadow", "/etc/*"));
        assert!(wildcard_match("/home/user/file.txt", "/home/*/file.txt"));
        assert!(!wildcard_match("/tmp/file.txt", "/home/*/file.txt"));
    }

    #[test]
    fn test_wildcard_multiple_stars() {
        assert!(wildcard_match("/home/user/project/src/main.rs", "/home/*/project/*.rs"));
        assert!(!wildcard_match("/home/user/other/src/main.rs", "/home/*/project/*.rs"));
    }

    #[test]
    fn test_wildcard_empty_pattern_segment() {
        // "**" has empty segment between stars — should still work
        assert!(wildcard_match("anything", "**"));
    }

    // -- PermissionUpdate --

    #[test]
    fn test_permission_update_add_rule() {
        let update = PermissionUpdate::add_rule(
            RuleSource::Session,
            PermissionRule::allow(RuleSource::Session, "bash", "git *"),
        );
        assert!(matches!(update, PermissionUpdate::AddRule { .. }));
    }

    #[test]
    fn test_permission_update_serde_roundtrip() {
        let update = PermissionUpdate::set_mode(PermissionMode::Bypass);
        let json_str = serde_json::to_string(&update).unwrap();
        let restored: PermissionUpdate = serde_json::from_str(&json_str).unwrap();
        assert_eq!(update, restored);
    }

    // -- Integration: Ruleset + SessionCache --

    // ================================================================
    // 集成测试: evaluate_permission — Hook + 规则 + 工具 + 缓存 + 模式
    // ================================================================
    //
    // 完整管线流程（katu-agent 中的 Agent Loop 按此顺序调用）：
    //
    //   LLM 请求 tool_call(bash, {"command": "rm -rf /"})
    //     │
    //     ▼
    //   ① Hook(PreToolUse) → HookPermission (allow/deny/ask)
    //     │                    ↓ aggregated
    //     ▼
    //   ② Tool.check_permissions(args) → PermissionResult
    //     │                    ↓
    //     ▼
    //   ③ PermissionCheckInput { request, hook_decision, tool_check, mode }
    //     │                    ↓
    //     ▼
    //   ④ evaluate_permission(ruleset, cache, input) → PermissionDecision
    //     │
    //     ├─ Allow → tool.validate() → tool.execute()
    //     ├─ Deny  → ToolOutput::error(message) → 反馈 LLM
    //     └─ Ask   → prompt_user() → apply_reply() → Allow/Deny
    //                                                    │
    //     ⑤ Hook(PostToolUse) ←────────────────────────────

    #[test]
    fn test_pipeline_policy_deny_beats_everything() {
        let mut ruleset = Ruleset::new();
        ruleset.add(PermissionRule::deny(RuleSource::Policy, "bash", "rm *"));

        let cache = SessionPermissionCache::new();

        // 即使 Hook allow + Bypass mode，Policy deny 仍然获胜
        let input = PermissionCheckInput::new(
            PermissionRequest::new("bash", "rm -rf /"),
            PermissionMode::Bypass,
        )
        .with_hook_decision(HookPermission::Allow);

        let decision = evaluate_permission(&ruleset, &cache, &input);
        assert!(decision.is_deny());
        if let PermissionDecision::Deny { reason, .. } = &decision {
            assert!(matches!(reason, PermissionReason::Rule { source: RuleSource::Policy }));
        }
    }

    #[test]
    fn test_pipeline_hook_deny_beats_rule_allow() {
        let mut ruleset = Ruleset::new();
        ruleset.add(PermissionRule::allow(RuleSource::User, "bash", "*"));

        let cache = SessionPermissionCache::new();

        // 规则说 allow，但 Hook 说 deny → deny
        let input = PermissionCheckInput::new(
            PermissionRequest::new("bash", "curl evil.com"),
            PermissionMode::Default,
        )
        .with_hook_decision(HookPermission::Deny {
            reason: Some("Suspicious URL detected".into()),
        });

        let decision = evaluate_permission(&ruleset, &cache, &input);
        assert!(decision.is_deny());
        if let PermissionDecision::Deny { reason, .. } = &decision {
            assert!(matches!(reason, PermissionReason::Hook));
        }
    }

    #[test]
    fn test_pipeline_hook_allow_cannot_bypass_rule_deny() {
        let mut ruleset = Ruleset::new();
        ruleset.add(PermissionRule::deny(RuleSource::User, "bash", "rm *"));

        let cache = SessionPermissionCache::new();

        // Hook 说 allow，但规则说 deny → deny（规则是安全底线）
        let input = PermissionCheckInput::new(
            PermissionRequest::new("bash", "rm file.txt"),
            PermissionMode::Default,
        )
        .with_hook_decision(HookPermission::Allow);

        let decision = evaluate_permission(&ruleset, &cache, &input);
        assert!(decision.is_deny());
    }

    #[test]
    fn test_pipeline_tool_deny_is_immediate() {
        let ruleset = Ruleset::new(); // 无规则
        let cache = SessionPermissionCache::new();

        // 工具自身拒绝（如路径安全检查）
        let input = PermissionCheckInput::new(
            PermissionRequest::new("edit", "/etc/shadow"),
            PermissionMode::Bypass, // 即使 bypass
        )
        .with_tool_check(PermissionResult::deny("Cannot edit system files"));

        let decision = evaluate_permission(&ruleset, &cache, &input);
        assert!(decision.is_deny());
        if let PermissionDecision::Deny { reason, .. } = &decision {
            assert!(matches!(reason, PermissionReason::ToolCheck));
        }
    }

    #[test]
    fn test_pipeline_hook_ask_overrides_rule_allow() {
        let mut ruleset = Ruleset::new();
        ruleset.add(PermissionRule::allow(RuleSource::User, "bash", "*"));

        let cache = SessionPermissionCache::new();

        // 规则 allow，但 Hook 说 ask → ask（Hook 要求确认）
        let input = PermissionCheckInput::new(
            PermissionRequest::new("bash", "git push --force"),
            PermissionMode::Default,
        )
        .with_hook_decision(HookPermission::Ask {
            message: Some("Force push detected, confirm?".into()),
        });

        let decision = evaluate_permission(&ruleset, &cache, &input);
        assert!(decision.is_ask());
    }

    #[test]
    fn test_pipeline_cache_allows_after_user_always() {
        let ruleset = Ruleset::new(); // 无规则 → 默认 ask
        let mut cache = SessionPermissionCache::new();
        cache.allow_always("bash", "git *");

        let input = PermissionCheckInput::new(
            PermissionRequest::new("bash", "git pull"),
            PermissionMode::Default,
        );

        let decision = evaluate_permission(&ruleset, &cache, &input);
        assert!(decision.is_allow());
    }

    #[test]
    fn test_pipeline_bypass_mode_auto_allows() {
        let ruleset = Ruleset::new();
        let cache = SessionPermissionCache::new();

        let input = PermissionCheckInput::new(
            PermissionRequest::new("bash", "anything"),
            PermissionMode::Bypass,
        );

        let decision = evaluate_permission(&ruleset, &cache, &input);
        assert!(decision.is_allow());
    }

    #[test]
    fn test_pipeline_non_interactive_auto_denies() {
        let ruleset = Ruleset::new();
        let cache = SessionPermissionCache::new();

        let input = PermissionCheckInput::new(
            PermissionRequest::new("bash", "anything"),
            PermissionMode::NonInteractive,
        );

        let decision = evaluate_permission(&ruleset, &cache, &input);
        assert!(decision.is_deny());
        if let PermissionDecision::Deny { reason, .. } = &decision {
            assert!(matches!(reason, PermissionReason::Mode));
        }
    }

    #[test]
    fn test_pipeline_no_rules_no_hook_defaults_to_ask() {
        let ruleset = Ruleset::new();
        let cache = SessionPermissionCache::new();

        let input = PermissionCheckInput::new(
            PermissionRequest::new("bash", "ls -la"),
            PermissionMode::Default,
        );

        let decision = evaluate_permission(&ruleset, &cache, &input);
        assert!(decision.is_ask());
    }

    #[test]
    fn test_pipeline_hook_allow_works_when_no_rule_conflicts() {
        let ruleset = Ruleset::new(); // 无规则
        let cache = SessionPermissionCache::new();

        // 无冲突规则时，Hook allow 生效
        let input = PermissionCheckInput::new(
            PermissionRequest::new("read", "file.txt"),
            PermissionMode::Default,
        )
        .with_hook_decision(HookPermission::Allow);

        let decision = evaluate_permission(&ruleset, &cache, &input);
        assert!(decision.is_allow());
    }

    #[test]
    fn test_pipeline_tool_ask_overrides_rule_allow() {
        let mut ruleset = Ruleset::new();
        ruleset.add(PermissionRule::allow(RuleSource::User, "edit", "*"));

        let cache = SessionPermissionCache::new();

        // 规则 allow，但工具要求确认（如 .git/ 路径）
        let input = PermissionCheckInput::new(
            PermissionRequest::new("edit", ".git/config"),
            PermissionMode::Default,
        )
        .with_tool_check(PermissionResult::ask("Editing .git/ files requires confirmation"));

        let decision = evaluate_permission(&ruleset, &cache, &input);
        assert!(decision.is_ask());
    }

    // -- HookPermission ↔ PermissionBehavior 转换 --

    #[test]
    fn test_hook_permission_to_behavior() {
        assert_eq!(
            PermissionBehavior::from(HookPermission::Allow),
            PermissionBehavior::Allow
        );
        assert_eq!(
            PermissionBehavior::from(HookPermission::Deny { reason: None }),
            PermissionBehavior::Deny
        );
        assert_eq!(
            PermissionBehavior::from(HookPermission::Ask { message: None }),
            PermissionBehavior::Ask
        );
    }

    #[test]
    fn test_behavior_to_hook_permission() {
        let perm: HookPermission = PermissionBehavior::Allow.into();
        assert!(perm.is_allow());

        let perm: HookPermission = PermissionBehavior::Deny.into();
        assert!(perm.is_deny());
    }
}
