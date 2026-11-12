# -*- coding: utf-8 -*-
"""记忆工具集（middleware ``list_tools`` 注入给 Agent）。

- ``memory_search``：跨层检索（长期/短期/肌肉），短期命中自动展开
  其链接的长期条目；
- ``memory_note``：用户显式"记一下"—— 写入**待确认队列**（dsh 原则：
  AI 只提议，你确认）；
- ``memory_confirm``：确认/拒绝待确认提议；
- ``memory_link``：手工为短期记忆补一条指向长期的索引链接；
- ``memory_status``：三层记忆一览。
"""
from __future__ import annotations

from typing import Any

from agentscope.message import TextBlock, ToolResultState
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import ToolBase, ToolChunk

from ..memory.tiers import TierManager
from ..storage import MemoryLink, PendingReview


class _MemoryToolBase(ToolBase):
    """记忆工具基类：自动放行（与 ReMe 中间件工具同款约定）。"""

    is_external_tool: bool = False
    is_state_injected: bool = False
    is_mcp: bool = False
    mcp_name: str | None = None

    def __init__(self, tiers: TierManager) -> None:
        self._tiers = tiers

    async def check_permissions(
        self, *_args: Any, **_kwargs: Any,
    ) -> PermissionDecision:
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="auto-allowed: rsi_mem memory tool",
        )


def _text_chunk(message: str) -> ToolChunk:
    return ToolChunk(content=[TextBlock(type="text", text=message)])


def _error_chunk(message: str) -> ToolChunk:
    return ToolChunk(
        content=[TextBlock(type="text", text=message)],
        state=ToolResultState.ERROR,
    )


class MemorySearchTool(_MemoryToolBase):
    """跨层记忆检索工具。"""

    name: str = "memory_search"
    description: str = (
        "检索三层运维记忆（肌肉记忆SOP/短期/长期L0-L3），短期命中会连带"
        "其索引链接的长期知识。需要历史处置经验、环境事实或 SOP 时调用。"
    )
    is_concurrency_safe: bool = True
    is_read_only: bool = True

    def __init__(self, tiers: TierManager) -> None:
        super().__init__(tiers)
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词或问题。",
                },
                "limit": {
                    "type": "integer",
                    "description": "最大返回条数（默认 5）。",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def __call__(self, query: str, limit: int = 5) -> ToolChunk:
        if not query.strip():
            return _error_chunk("query 不能为空")
        results = await self._tiers.search_for_tool(query, limit)
        if not results:
            return _text_chunk("(没有命中相关记忆)")
        lines = []
        for r in results:
            link = (
                f" ← 链接自短期记忆 {r.parent_id[:8]}"
                if r.parent_id else ""
            )
            lines.append(
                f"- {r.record.display(400)}{link}",
            )
        return _text_chunk("\n".join(lines))


class MemoryNoteTool(_MemoryToolBase):
    """显式记忆提议工具（写入待确认队列）。"""

    name: str = "memory_note"
    description: str = (
        "提议写入一条持久记忆（SOP/事实/经验）。写入需用户确认后才"
        "生效——调用后提示用户可用 memory_confirm 确认。"
        "title 简洁，content 完整，kind ∈ fact/sop/incident/decision。"
    )
    is_concurrency_safe: bool = False
    is_read_only: bool = False

    def __init__(self, tiers: TierManager) -> None:
        super().__init__(tiers)
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "标题（≤40字）"},
                "content": {"type": "string", "description": "内容"},
                "kind": {
                    "type": "string",
                    "enum": ["fact", "sop", "incident", "decision"],
                    "description": "记忆种类",
                },
                "tier": {
                    "type": "string",
                    "enum": ["muscle", "long", "short"],
                    "description": "目标层：muscle=SOP反射，"
                                   "long=长期知识，short=短期临时",
                    "default": "long",
                },
                "triggers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "muscle 层触发关键词（可选）",
                },
            },
            "required": ["title", "content", "kind"],
        }

    async def __call__(
        self,
        title: str,
        content: str,
        kind: str = "fact",
        tier: str = "long",
        triggers: list[str] | None = None,
    ) -> ToolChunk:
        if tier == "short":
            rec = await self._tiers.shortterm.record(
                session_id=None,
                title=title,
                content=content,
                kind=kind,
                confidence=0.8,
                source={"origin": "memory_note"},
            )
            return _text_chunk(
                f"已写入短期记忆 {rec.id[:8]}（TTL 内有效）。",
            )
        if tier == "muscle" or kind == "sop":
            review = PendingReview(
                action="create_muscle",
                payload={
                    "title": title,
                    "content": content,
                    "triggers": triggers or [title],
                    "origin": "memory_note",
                },
                reason=f"用户/Agent 提议 SOP：{title[:40]}",
                confidence=0.75,
            )
        else:
            review = PendingReview(
                action="create_longterm",
                payload={
                    "title": title,
                    "content": content,
                    "layer": "L1_atom",
                    "kind": kind,
                },
                reason=f"提议长期记忆：{title[:40]}",
                confidence=0.75,
            )
        state = await self._tiers.propose(review)
        if state == "auto_applied":
            return _text_chunk(f"已自动写入（{review.action}）。")
        return _text_chunk(
            f"已加入待确认队列（review {review.id[:8]}），"
            f"用户可用 memory_confirm 确认。",
        )


class MemoryConfirmTool(_MemoryToolBase):
    """确认/拒绝待确认提议。"""

    name: str = "memory_confirm"
    description: str = (
        "确认或拒绝一条记忆提议。不带 review_id 时先列出全部待确认项。"
    )
    is_concurrency_safe: bool = False
    is_read_only: bool = False

    def __init__(self, tiers: TierManager) -> None:
        super().__init__(tiers)
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "review_id": {
                    "type": "string",
                    "description": "待确认项 id（空 = 列出全部）",
                },
                "approved": {
                    "type": "boolean",
                    "description": "True 确认 / False 拒绝",
                    "default": True,
                },
            },
        }

    async def __call__(
        self,
        review_id: str = "",
        approved: bool = True,
    ) -> ToolChunk:
        if not review_id:
            pend = await self._tiers.pending()
            if not pend:
                return _text_chunk("(当前没有待确认的记忆提议)")
            return _text_chunk("\n".join(
                f"- {r.id[:8]}: {r.summarize()}" for r in pend
            ))
        result = await self._tiers.resolve(review_id, approved)
        if result is None and approved:
            return _error_chunk(f"未找到或已处理：{review_id}")
        verb = "已确认并执行" if approved else "已拒绝"
        return _text_chunk(f"{verb}：{review_id[:8]}")


class MemoryLinkTool(_MemoryToolBase):
    """手工补建 短期→长期 索引链接。"""

    name: str = "memory_link"
    description: str = (
        "把一条短期记忆与一条长期记忆建立索引链接（related）。"
        "两个 id 都可从 memory_search 结果取。"
    )
    is_concurrency_safe: bool = False
    is_read_only: bool = False

    def __init__(self, tiers: TierManager) -> None:
        super().__init__(tiers)
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "short_id": {"type": "string"},
                "long_id": {"type": "string"},
            },
            "required": ["short_id", "long_id"],
        }

    async def __call__(self, short_id: str, long_id: str) -> ToolChunk:
        db = self._tiers.retriever._db
        rec = await db.get_memory(short_id)
        target = await db.get_memory(long_id)
        if rec is None or target is None:
            return _error_chunk("id 不存在")
        if any(l.target_id == long_id for l in rec.links):
            return _text_chunk("链接已存在。")
        rec.links.append(
            MemoryLink(target_id=long_id, kind="related", note="manual"),
        )
        await db.update_memory(rec)
        return _text_chunk(
            f"已链接 {short_id[:8]} → {long_id[:8]}。",
        )


class MemoryStatusTool(_MemoryToolBase):
    """三层记忆状态一览。"""

    name: str = "memory_status"
    description: str = "查看三层记忆统计（条数/待确认/基因组版本）。"
    is_concurrency_safe: bool = True
    is_read_only: bool = True

    def __init__(self, tiers: TierManager) -> None:
        super().__init__(tiers)
        self.input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def __call__(self) -> ToolChunk:
        stats = await self._tiers.stats()
        return _text_chunk(str(stats))


def build_memory_tools(tiers: TierManager) -> list[ToolBase]:
    """构造全部记忆工具。"""
    return [
        MemorySearchTool(tiers),
        MemoryNoteTool(tiers),
        MemoryConfirmTool(tiers),
        MemoryLinkTool(tiers),
        MemoryStatusTool(tiers),
    ]
