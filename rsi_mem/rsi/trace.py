# -*- coding: utf-8 -*-
"""轨迹录制：Dream-RSI"发现树"的在线录制形态。

RSIMiddleware 在一次 reply 的各 hook 里向 :class:`TraceRecorder` 喂
模型调用 / 工具调用 / 权限事件，回合结束时结算 outcome 评分并落库。
这些轨迹就是回放模拟器（``replay.py``）与 RSI 诊断的数据来源。
"""
from __future__ import annotations

from typing import Any

from ..config import RSIMemConfig
from ..storage import (
    ModelCallStat,
    OutcomeRecord,
    PermissionEvent,
    RSIMemDatabase,
    ToolCallNode,
    TraceRecord,
)
from ..telemetry import CAT_TRACE, LocalEventLog


def score_outcome(
    *,
    has_answer: bool,
    max_iters_exceeded: bool,
    tool_error_ratio: float,
    tool_ok_count: int,
    interrupted: bool,
) -> float:
    """启发式结局评分（0..1），用户反馈在其上叠加。"""
    score = 0.5
    if has_answer:
        score += 0.2
    if max_iters_exceeded:
        score -= 0.2
    if interrupted:
        score -= 0.3
    score -= 0.15 * min(1.0, tool_error_ratio)
    if tool_ok_count > 0:
        score += min(0.1, 0.02 * tool_ok_count)
    return max(0.0, min(1.0, score))


class TraceRecorder:
    """单次 reply 的轨迹收集器。"""

    def __init__(
        self,
        db: RSIMemDatabase,
        config: RSIMemConfig,
        *,
        session_id: str | None,
        query: str,
        events: LocalEventLog | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._events = events
        self.record = TraceRecord(session_id=session_id, query=query)

    def add_model_call(self, stat: ModelCallStat) -> None:
        self.record.model_calls.append(stat)

    def add_tool_call(
        self,
        name: str,
        input_digest: str,
        state: str,
        latency_s: float,
    ) -> None:
        self.record.tool_calls.append(
            ToolCallNode(
                name=name,
                input_digest=input_digest[:200],
                state=state,  # type: ignore[arg-type]
                latency_s=round(latency_s, 3),
            ),
        )

    def add_permission(self, tool: str, behavior: str) -> None:
        self.record.permissions.append(
            PermissionEvent(tool=tool, behavior=behavior),
        )

    def note_used_memories(self, memory_ids: list[str]) -> None:
        for mid in memory_ids:
            if mid not in self.record.used_memory_ids:
                self.record.used_memory_ids.append(mid)

    async def finish(
        self,
        *,
        digest: str,
        has_answer: bool,
        max_iters_exceeded: bool = False,
        interrupted: bool = False,
        injected_memory_ids: list[str] | None = None,
    ) -> TraceRecord:
        """结算评分并落库。"""
        if injected_memory_ids:
            self.note_used_memories(injected_memory_ids)
        calls = self.record.tool_calls
        error_ratio = (
            sum(1 for c in calls if c.state == "error") / len(calls)
            if calls else 0.0
        )
        score = score_outcome(
            has_answer=has_answer,
            max_iters_exceeded=max_iters_exceeded,
            tool_error_ratio=error_ratio,
            tool_ok_count=sum(1 for c in calls if c.state == "ok"),
            interrupted=interrupted,
        )
        self.record.digest = digest[:500]
        self.record.outcome = OutcomeRecord(
            score=round(score, 3),
            signals={
                "has_answer": has_answer,
                "max_iters_exceeded": max_iters_exceeded,
                "tool_error_ratio": round(error_ratio, 3),
                "tool_calls": len(calls),
                "model_calls": len(self.record.model_calls),
            },
        )
        await self.upsert()
        if self._events is not None:
            self._events.emit_sync(
                CAT_TRACE,
                {
                    "trace_id": self.record.id,
                    "session_id": self.record.session_id,
                    "query": self.record.query[:120],
                    "score": self.record.outcome.score,
                    "tokens": self.record.total_tokens,
                    "tool_calls": len(calls),
                    "tool_error_ratio": round(error_ratio, 3),
                    "max_iters_exceeded": max_iters_exceeded,
                },
            )
        return self.record

    async def upsert(self) -> None:
        await self._db.upsert_trace(self.record)


async def attach_feedback(
    db: RSIMemDatabase,
    trace_id: str,
    feedback: int,
) -> TraceRecord | None:
    """rsi_feedback 工具：把用户 👍/👎 附加到轨迹 outcome。"""
    traces = await db.list_traces(limit=200)
    trace = next((t for t in traces if t.id == trace_id), None)
    if trace is None or trace.outcome is None:
        return None
    trace.outcome.feedback = feedback
    trace.outcome.score = round(
        max(0.0, min(1.0, trace.outcome.score + 0.3 * feedback)), 3,
    )
    await db.upsert_trace(trace)
    return trace


def trace_cost_summary(traces: list[TraceRecord]) -> dict[str, Any]:
    """成本侧聚合（供 objectives 使用）。"""
    if not traces:
        return {"mean_tokens": 0.0, "mean_calls": 0.0, "mean_duration": 0.0}
    return {
        "mean_tokens": sum(t.total_tokens for t in traces) / len(traces),
        "mean_calls": sum(
            len(t.model_calls) for t in traces
        ) / len(traces),
        "mean_duration": sum(t.duration_s for t in traces) / len(traces),
    }
