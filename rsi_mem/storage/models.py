# -*- coding: utf-8 -*-
"""Pydantic 数据模型：三层记忆条目、轨迹、基因组、待确认队列。

这些模型是 ``rsi_mem`` 的公共数据形状，被存储层（SQLite 序列化）、
记忆层、RSI 引擎与连接器共用。JSON 字段（links/source 等）在
``database.py`` 里以 ``json.dumps`` 持久化。
"""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

import shortuuid

MemoryTier = Literal["muscle", "short", "long"]
"""记忆层级：肌肉 / 短期 / 长期。"""

LongTermLayer = Literal["L0_raw", "L1_atom", "L2_scenario", "L3_core"]
"""长期记忆内部分层（参考 TencentDB-Agent-Memory）：

- ``L0_raw`` 原始证据（对话摘录、工具输出片段），只作溯源；
- ``L1_atom`` 原子事实/事件；
- ``L2_scenario`` 场景/主题块（如"某集群磁盘告警处置"）；
- ``L3_core`` 核心档案（环境摘要、用户偏好），小容量、每轮注入。
"""

MuscleKind = Literal["reflex", "sop"]
"""肌肉记忆种类：条件反射（短触发规则）/ SOP（分步流程）。"""

MemoryStatus = Literal["active", "pending", "candidate", "archived", "promoted"]
"""条目状态。``candidate`` 专供 RSI 产生的未部署候选 SOP。"""

LinkKind = Literal["related", "provenance", "promotes"]
"""记忆间链接类型：``related`` 短期→长期相关索引；``provenance``
新条目指向其证据来源；``promotes`` 晋升后指向晋升前条目。"""


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return shortuuid.uuid()


class MemoryLink(BaseModel):
    """一条记忆→记忆的索引链接。"""

    target_id: str
    kind: LinkKind = "related"
    note: str = ""


class MemoryRecord(BaseModel):
    """三层记忆的统一条目模型。"""

    id: str = Field(default_factory=_new_id)
    tier: MemoryTier
    layer: LongTermLayer | MuscleKind | None = None
    kind: str = "observation"
    """条目种类：fact / incident / sop / profile / observation / decision …"""
    title: str = ""
    content: str = ""
    triggers: list[str] = Field(default_factory=list)
    """肌肉记忆触发器：关键词、正则或工具名。"""
    scope: str = "global"
    """作用域（Harmony 的 gen/env 思路）：``global`` 或会话/环境标识。"""
    session_id: str | None = None
    """短期记忆所属会话。"""
    confidence: float = 0.5
    status: MemoryStatus = "active"
    links: list[MemoryLink] = Field(default_factory=list)
    source: dict[str, Any] = Field(default_factory=dict)
    """溯源：{trace_id, reply_id, origin(…), …}"""
    use_count: int = 0
    hit_count: int = 0
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)
    expires_at: float | None = None
    """短期记忆 TTL 戳（epoch 秒），过期由固化管线清理。"""
    version: int = 1

    def display(self, max_chars: int = 500) -> str:
        """渲染为注入用的一行/多行摘要文本。"""
        head = f"[{self.tier}:{self.layer or self.kind}] {self.title}".rstrip()
        body = self.content if len(self.content) <= max_chars else (
            self.content[:max_chars] + "…"
        )
        return f"{head}\n{body}" if self.title else body


# ---------------------------------------------------------------------------
# RSI 轨迹（发现树）
# ---------------------------------------------------------------------------


class ToolCallNode(BaseModel):
    """一次工具调用节点。"""

    name: str
    input_digest: str = ""
    state: Literal["ok", "error", "interrupted"] = "ok"
    latency_s: float = 0.0
    at: float = Field(default_factory=_now)


class ModelCallStat(BaseModel):
    """一次模型调用统计。"""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class PermissionEvent(BaseModel):
    """一次权限决策记录（安全目标信号）。"""

    tool: str
    behavior: str
    at: float = Field(default_factory=_now)


class OutcomeRecord(BaseModel):
    """轨迹结局评分（质量目标的输入）。"""

    score: float = 0.5
    """0..1 归一化任务质量估计。"""
    feedback: int | None = None
    """用户显式反馈：1 赞 / -1 踩 / None。"""
    signals: dict[str, Any] = Field(default_factory=dict)
    """max_iters_exceeded / tool_error_ratio / has_final_answer …"""


class TraceRecord(BaseModel):
    """一条 reply 轨迹 —— Dream-RSI"发现树"的在线录制形态。"""

    id: str = Field(default_factory=_new_id)
    session_id: str | None = None
    query: str = ""
    digest: str = ""
    """最终回复摘要（供回放与评审）。"""
    started_at: float = Field(default_factory=_now)
    finished_at: float | None = None
    tool_calls: list[ToolCallNode] = Field(default_factory=list)
    model_calls: list[ModelCallStat] = Field(default_factory=list)
    permissions: list[PermissionEvent] = Field(default_factory=list)
    outcome: OutcomeRecord | None = None
    used_memory_ids: list[str] = Field(default_factory=list)
    """该轨迹注入过哪些记忆（回放回归检查用）。"""

    @property
    def total_tokens(self) -> int:
        return sum(
            m.input_tokens + m.output_tokens for m in self.model_calls
        )

    @property
    def duration_s(self) -> float:
        if self.finished_at is None:
            return 0.0
        return self.finished_at - self.started_at


# ---------------------------------------------------------------------------
# RSI 声明式基因组（Harmony Harness 的可演化组件集）
# ---------------------------------------------------------------------------


class PromptFragment(BaseModel):
    """一段可注入系统提示词的逻辑组件（logic_evolve 的对象）。"""

    id: str = Field(default_factory=_new_id)
    slot: Literal["always", "reflex"] = "always"
    """``always`` 每轮注入；``reflex`` 仅在触发词命中时注入。"""
    triggers: list[str] = Field(default_factory=list)
    text: str
    enabled: bool = True
    origin: Literal["seed", "rsi"] = "seed"
    locked: bool = False
    """锁定片段（安全规则等）任何候选不得删除或改写。"""


class RetrievalParams(BaseModel):
    """检索与注入参数（retrieval_evolve 的对象），全部带安全边界。"""

    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    char_budget: int = Field(default=2000, ge=200, le=8000)
    short_inject_recent: int = Field(default=6, ge=0, le=30)
    reflex_budget: int = Field(default=2048, ge=256, le=8192)
    include_l3: bool = True


class GenomeSpec(BaseModel):
    """声明式基因组：RSI 的演化对象。

    对应 Harmony Harness 的 {tools, skills, logic} 组件集 × gen/env 作用域
    思想，但全部是**数据**而非任意代码 —— 这是"不改动源码"约束下的安全
    演化边界。Dream-RSI 的"探索策略"在此体现为这些参数的组合。
    """

    version: int = 0
    """0 表示种子版本。"""
    created_at: float = Field(default_factory=_now)
    parent_version: int | None = None
    operator: str = "seed"
    """产生该版本的算子：seed / skill_evolve / logic_evolve /
    retrieval_evolve / generalize。"""
    notes: str = ""
    prompt_fragments: list[PromptFragment] = Field(default_factory=list)
    retrieval: RetrievalParams = Field(default_factory=RetrievalParams)
    active_sops: list[str] = Field(default_factory=list)
    """激活的肌肉记忆条目 id（优先级顺序）。"""
    scores: dict[str, Any] = Field(default_factory=dict)
    """部署时记录的各子问题 Tchebycheff 分数与三目标向量。"""

    @classmethod
    def seed(cls) -> GenomeSpec:
        """构造种子基因组：带一条锁定的安全片段。"""
        return cls(
            version=0,
            operator="seed",
            notes="seed genome",
            prompt_fragments=[
                PromptFragment(
                    id="safety-core",
                    slot="always",
                    text=(
                        "## 安全底线\n"
                        "生产环境操作前必须：确认目标（主机/集群/命名空间），"
                        "使用只读命令核实现状，变更类操作先说明影响面。"
                        "禁止未经确认的删除/格式化/重启生产服务。"
                    ),
                    origin="seed",
                    locked=True,
                ),
            ],
        )

    def fragment_text(self) -> str:
        """渲染所有启用片段（按 always → reflex 排序）。"""
        parts = [
            f.text
            for f in self.prompt_fragments
            if f.enabled and f.slot == "always"
        ]
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 待确认队列（dsh-memory-evolve：AI 只提议，你确认）
# ---------------------------------------------------------------------------

ReviewAction = Literal[
    "promote_longterm",
    "create_longterm",
    "create_muscle",
    "archive_memory",
    "demote_muscle",
    "merge_memories",
]


class PendingReview(BaseModel):
    """一条待人工确认的提议。"""

    id: str = Field(default_factory=_new_id)
    action: ReviewAction
    memory_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.5
    created_at: float = Field(default_factory=_now)
    resolved_at: float | None = None
    resolved_by: Literal["user", "auto", "rejected"] | None = None

    def summarize(self) -> str:
        return f"[{self.action}] {self.reason} (confidence={self.confidence:.2f})"
