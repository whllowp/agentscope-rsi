# -*- coding: utf-8 -*-
"""RSI 工具集（middleware ``list_tools`` 注入给 Agent）。"""
from __future__ import annotations

import json
from typing import Any

from agentscope.message import TextBlock, ToolResultState
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import ToolBase, ToolChunk

from ..context import RSIMemContext
from ..storage import RSIMemDatabase
from .trace import attach_feedback


class _RSIToolBase(ToolBase):
    """RSI 工具基类：自动放行。"""

    is_external_tool: bool = False
    is_state_injected: bool = False
    is_mcp: bool = False
    mcp_name: str | None = None

    def __init__(self, ctx: RSIMemContext) -> None:
        self._ctx = ctx

    async def check_permissions(
        self, *_args: Any, **_kwargs: Any,
    ) -> PermissionDecision:
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="auto-allowed: rsi_mem rsi tool",
        )


def _text_chunk(message: str) -> ToolChunk:
    return ToolChunk(content=[TextBlock(type="text", text=message)])


def _error_chunk(message: str) -> ToolChunk:
    return ToolChunk(
        content=[TextBlock(type="text", text=message)],
        state=ToolResultState.ERROR,
    )


class RSIStatusTool(_RSIToolBase):
    """RSI 与记忆整体状态。"""

    name: str = "rsi_status"
    description: str = (
        "查看 RSI 自我改进引擎状态：基因组版本、上次 dream 报告、"
        "三层记忆统计、待确认提议数。"
    )
    is_concurrency_safe: bool = True
    is_read_only: bool = True

    def __init__(self, ctx: RSIMemContext) -> None:
        super().__init__(ctx)
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {},
        }

    async def __call__(self) -> ToolChunk:
        snap = await self._ctx.snapshot()
        report_raw = await self._ctx.db.get_meta("last_dream_report")
        try:
            report = json.loads(report_raw) if report_raw else None
        except json.JSONDecodeError:
            report = None
        return _text_chunk(json.dumps(
            {"snapshot": snap, "last_dream_report": report},
            ensure_ascii=False, indent=2,
        ))


class RSIFeedbackTool(_RSIToolBase):
    """用户反馈（质量目标信号）。"""

    name: str = "rsi_feedback"
    description: str = (
        "对刚才的处理结果打分：1=满意解决，-1=没解决/处理糟糕。"
        "反馈会写入轨迹并驱动 RSI 自我改进。"
    )
    is_concurrency_safe: bool = False
    is_read_only: bool = False

    def __init__(self, ctx: RSIMemContext) -> None:
        super().__init__(ctx)
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "enum": [1, -1],
                    "description": "1 赞 / -1 踩",
                },
                "comment": {
                    "type": "string",
                    "description": "可选备注",
                },
                "trace_id": {
                    "type": "string",
                    "description": "可选：指定轨迹 id（默认最近一条）",
                },
            },
            "required": ["score"],
        }

    async def __call__(
        self,
        score: int,
        comment: str = "",
        trace_id: str = "",
    ) -> ToolChunk:
        db: RSIMemDatabase = self._ctx.db
        if not trace_id:
            traces = await db.list_traces(limit=1)
            if not traces:
                return _error_chunk("还没有可评价的轨迹")
            trace_id = traces[0].id
        trace = await attach_feedback(db, trace_id, score)
        if trace is None:
            return _error_chunk(f"未找到轨迹 {trace_id[:8]}")
        if comment:
            trace.outcome.signals["feedback_comment"] = comment[:300]
            await db.upsert_trace(trace)
        return _text_chunk(
            f"已记录反馈 {score:+d} → 轨迹 {trace_id[:8]}，"
            f"新评分 {trace.outcome.score}",
        )


class RSIEvolveTool(_RSIToolBase):
    """手动触发一次 dream（离线改进评估）。"""

    name: str = "rsi_evolve"
    description: str = (
        "手动触发一次 RSI 离线改进（dream）：诊断近期轨迹 → 生成候选"
        "基因组 → 回放评估 → 部署不劣于当前的最佳候选。"
    )
    is_concurrency_safe: bool = False
    is_read_only: bool = False

    def __init__(self, ctx: RSIMemContext) -> None:
        super().__init__(ctx)
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "rollback": {
                    "type": "boolean",
                    "description": "True 则回滚到上一版基因组而非演化",
                    "default": False,
                },
            },
        }

    async def __call__(self, rollback: bool = False) -> ToolChunk:
        if rollback:
            prev = await self._ctx.engine.rollback()
            if prev is None:
                return _error_chunk("没有可回滚的历史版本")
            return _text_chunk(f"已回滚：当前基因组 v{prev.version}")
        report = await self._ctx.engine.dream(trigger="manual")
        deployed = report.get("deployed")
        if deployed:
            return _text_chunk(
                f"已部署基因组 v{report['version']}"
                f"（{report['operator']}）：{report['notes']}",
            )
        return _text_chunk(
            f"评估完成但未部署：{report.get('reason')}。"
            f"候选数 {report.get('candidates')}。",
        )


class RSIHistoryTool(_RSIToolBase):
    """基因组版本历史。"""

    name: str = "rsi_history"
    description: str = "查看基因组演化历史（版本/算子/评分）。"
    is_concurrency_safe: bool = True
    is_read_only: bool = True

    def __init__(self, ctx: RSIMemContext) -> None:
        super().__init__(ctx)
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {},
        }

    async def __call__(self) -> ToolChunk:
        genomes = await self._ctx.db.list_genomes()
        if not genomes:
            return _text_chunk("(尚无基因组)")
        lines = []
        for g in genomes:
            scores = g.scores.get("primary_tchebycheff")
            lines.append(
                f"- v{g.version} [{g.operator}] sops={len(g.active_sops)}"
                f" fragments={len(g.prompt_fragments)}"
                f"{' score=%.4f' % scores if scores is not None else ''}"
                f" — {g.notes[:60]}",
            )
        return _text_chunk("\n".join(lines))


def build_rsi_tools(ctx: RSIMemContext) -> list[ToolBase]:
    """构造全部 RSI 工具。"""
    return [
        RSIStatusTool(ctx),
        RSIFeedbackTool(ctx),
        RSIEvolveTool(ctx),
        RSIHistoryTool(ctx),
    ]
