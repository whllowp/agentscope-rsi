# -*- coding: utf-8 -*-
"""固化审查管线（周期触发，dsh-memory-evolve 的"记忆审查"）：

1. **晋升提议**：会话内置信度较高、出现多次的短期条目 → 提议晋升长期；
2. **SOP 蒸馏**：同一会话中反复出现的工具序列（尤其含失败后重试成功）
   → 提议固化为肌肉记忆 SOP（重复踩坑自动成流程）；
3. **去重 / 归档**：L1 明显重复、肌肉记忆长期不命中 → 提议合并/降级。

所有提议走 :meth:`TierManager.propose` 的分级确认：高置信低风险自动
执行，SOP 与安全敏感内容必须人工确认。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from agentscope._logging import logger
from agentscope.model import ChatModelBase

from ..config import RSIMemConfig
from ..storage import PendingReview, TraceRecord
from .tiers import TierManager

_SOP_MIN_REPEAT = 2


class Consolidator:
    """周期固化审查。"""

    def __init__(self, config: RSIMemConfig, tiers: TierManager) -> None:
        self._config = config
        self._tiers = tiers

    async def review_session(
        self,
        session_id: str | None,
        traces: list[TraceRecord],
        *,
        model: ChatModelBase | None = None,
    ) -> list[PendingReview]:
        """对一个会话执行审查，返回新入队的提议（含自动执行的）。"""
        proposals: list[PendingReview] = []
        proposals.extend(await self._propose_promotions(session_id))
        sop = await self._distill_sop(session_id, traces)
        if sop is not None:
            proposals.append(sop)
        proposals.extend(await self._propose_maintenance())
        out: list[PendingReview] = []
        for p in proposals:
            try:
                state = await self._tiers.propose(p)
                if state == "auto_applied":
                    p.resolved_by = "auto"
                out.append(p)
            except Exception as e:  # noqa: BLE001
                logger.warning("propose %s failed: %s", p.action, e)
        await self._tiers.maintenance()
        return out

    # ------------------------------------------------------------------
    # 短期 → 长期 晋升
    # ------------------------------------------------------------------
    async def _propose_promotions(
        self, session_id: str | None,
    ) -> list[PendingReview]:
        if session_id is None:
            return []
        shorts = await self._tiers.shortterm.recent(
            session_id, self._config.short_term.max_per_session,
        )
        out: list[PendingReview] = []
        for m in shorts:
            looks_durable = (
                m.confidence >= 0.7 and m.kind in ("fact", "incident",
                                                   "decision")
            ) or m.hit_count >= 2
            if not looks_durable:
                continue
            out.append(PendingReview(
                action="promote_longterm",
                memory_id=m.id,
                payload={"memory_id": m.id, "layer": "L1_atom"},
                reason=f"短期记忆[{m.title[:40]}]反复出现/高置信，"
                       f"建议晋升长期",
                confidence=min(0.95, m.confidence + 0.1),
            ))
        return out[:5]

    # ------------------------------------------------------------------
    # 重复工具模式 → 肌肉记忆 SOP
    # ------------------------------------------------------------------
    async def _distill_sop(
        self,
        session_id: str | None,
        traces: list[TraceRecord],
    ) -> PendingReview | None:
        """从轨迹中找重复出现 ≥ N 次的工具序列，蒸馏 SOP 提议。"""
        session_traces = [
            t for t in traces
            if session_id is None or t.session_id == session_id
        ]
        if len(session_traces) < self._config.review.min_turns_for_sop:
            return None
        seqs: Counter[tuple[str, ...]] = Counter()
        sample_query = ""
        for t in session_traces:
            names = tuple(c.name for c in t.tool_calls if c.state == "ok")
            if len(names) >= 2:
                seqs[names] += 1
                if not sample_query:
                    sample_query = t.query[:60]
        if not seqs:
            return None
        seq, count = seqs.most_common(1)[0]
        if count < _SOP_MIN_REPEAT:
            return None
        steps = "\n".join(f"{i}. 执行 `{n}`" for i, n in enumerate(seq, 1))
        content = (
            f"处理类似「{sample_query}」的问题时按以下流程：\n{steps}\n"
            f"（该流程在 {count} 条轨迹中重复出现，已验证有效）"
        )
        return PendingReview(
            action="create_muscle",
            payload={
                "title": f"SOP: {sample_query or '常见运维流程'}",
                "content": content,
                "triggers": [w for w in sample_query.split() if w][:5],
                "origin": "consolidation",
            },
            reason=f"工具序列 {' → '.join(seq)} 重复出现 {count} 次，"
                   f"建议固化为 SOP",
            confidence=min(0.95, 0.6 + 0.1 * count),
        )

    # ------------------------------------------------------------------
    # 维护类提议：重复合并 / 肌肉记忆降级
    # ------------------------------------------------------------------
    async def _propose_maintenance(self) -> list[PendingReview]:
        out: list[PendingReview] = []
        atoms = await self._tiers.retriever._db.list_memories(
            tier="long", layer="L1_atom", status="active", limit=200,
        )
        by_title: dict[str, list[str]] = {}
        for a in atoms:
            by_title.setdefault(a.title, []).append(a.id)
        for title, ids in by_title.items():
            if len(ids) >= 2:
                out.append(PendingReview(
                    action="merge_memories",
                    payload={"primary_id": ids[0], "duplicate_id": ids[1]},
                    reason=f"长期记忆标题重复：{title[:40]}",
                    confidence=0.8,
                ))
        muscles = await self._tiers.retriever._db.list_memories(
            tier="muscle", status="active", limit=100,
        )
        for m in muscles:
            if m.use_count >= 10 and m.hit_count == 0:
                out.append(PendingReview(
                    action="demote_muscle",
                    memory_id=m.id,
                    payload={"memory_id": m.id},
                    reason=f"肌肉记忆[{m.title[:30]}]长期未命中，"
                           f"建议降级为长期场景知识",
                    confidence=0.6,
                ))
        return out[:8]


def summarize_feedback(traces: list[TraceRecord]) -> dict[str, Any]:
    """给 RSI 诊断用的聚合统计（质量/成本/安全三目标信号源）。"""
    if not traces:
        return {"count": 0}
    scores = [t.outcome.score for t in traces if t.outcome]
    feedbacks = [t.outcome.feedback for t in traces
                 if t.outcome and t.outcome.feedback is not None]
    tool_errors = sum(
        1 for t in traces for c in t.tool_calls if c.state == "error"
    )
    tool_total = sum(len(t.tool_calls) for t in traces)
    tokens = sum(t.total_tokens for t in traces)
    durations = [t.duration_s for t in traces]
    denied = sum(
        1 for t in traces for p in t.permissions
        if p.behavior not in ("allow", "ALLOW")
    )
    return {
        "count": len(traces),
        "mean_outcome": sum(scores) / len(scores) if scores else None,
        "positive_feedback": sum(1 for f in feedbacks if f > 0),
        "negative_feedback": sum(1 for f in feedbacks if f < 0),
        "tool_error_ratio": (
            round(tool_errors / tool_total, 3) if tool_total else 0.0
        ),
        "total_tokens": tokens,
        "mean_duration_s": (
            round(sum(durations) / len(durations), 2) if durations else 0.0
        ),
        "permission_denied": denied,
    }
