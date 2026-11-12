# -*- coding: utf-8 -*-
"""短期记忆：会话临时层（TTL + 每会话 LRU 容量 + 链接索引）。

写入即建立指向长期记忆的 ``related`` 索引链接（用户的核心需求）；
高价值条目经固化审查晋升为长期（晋升后保留 ``promotes`` 溯源链）。
"""
from __future__ import annotations

import time

from agentscope._logging import logger

from ..config import RSIMemConfig
from ..storage import MemoryLink, MemoryRecord, RSIMemDatabase
from .retrieval import Retriever


class ShortTermStore:
    """短期记忆读写与生命周期。"""

    def __init__(
        self,
        db: RSIMemDatabase,
        config: RSIMemConfig,
        retriever: Retriever | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._retriever = retriever

    async def record(
        self,
        *,
        session_id: str | None,
        title: str,
        content: str,
        kind: str = "observation",
        confidence: float = 0.5,
        source: dict | None = None,
        link_query: str | None = None,
        max_links: int = 3,
    ) -> MemoryRecord:
        """写入一条短期记忆，并自动检索长期记忆建立索引链接。"""
        now = time.time()
        record = MemoryRecord(
            tier="short",
            kind=kind,
            title=title[:200],
            content=content[:2000],
            session_id=session_id,
            confidence=confidence,
            expires_at=now + self._config.short_term.ttl_hours * 3600,
            source=source or {},
        )
        # —— 短期 → 长期 索引链接 ——
        if self._retriever is not None:
            query = link_query or f"{title}\n{content}"
            try:
                related = await self._retriever.search(
                    query, tiers=("long",), top_k=max_links,
                    score_threshold=0.02, char_budget=10_000,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("short-term linking failed: %s", e)
                related = []
            record.links = [
                MemoryLink(
                    target_id=r.record.id, kind="related",
                    note=f"via:{r.via}",
                )
                for r in related
            ]
        await self._db.insert_memory(record)
        await self._enforce_capacity(session_id)
        return record

    async def _enforce_capacity(self, session_id: str | None) -> None:
        """超出每会话容量时按 updated_at 淘汰最旧条目（LRU）。"""
        if session_id is None:
            return
        cap = self._config.short_term.max_per_session
        items = await self._db.list_memories(
            tier="short", session_id=session_id, status="active", limit=cap + 50,
        )
        for stale in items[cap:]:
            await self._db.delete_memory(stale.id)

    async def recent(
        self, session_id: str | None, n: int,
    ) -> list[MemoryRecord]:
        """最近 n 条短期记忆（注入用）。"""
        if session_id is None or n <= 0:
            return []
        items = await self._db.list_memories(
            tier="short", session_id=session_id, status="active", limit=n,
        )
        return list(reversed(items))  # 时间正序

    async def cleanup(self) -> int:
        """TTL 过期清理。"""
        return await self._db.expire_short_term(time.time())

    async def promote(
        self,
        memory_id: str,
        *,
        layer: str = "L1_atom",
        confidence: float | None = None,
    ) -> MemoryRecord | None:
        """把短期条目晋升为长期：状态翻转 + ``promotes`` 溯源 + 建向量。"""
        rec = await self._db.get_memory(memory_id)
        if rec is None or rec.tier != "short":
            return None
        old_id = rec.id
        rec.tier = "long"
        rec.layer = layer  # type: ignore[assignment]
        rec.status = "active"
        rec.expires_at = None
        rec.version += 1
        if confidence is not None:
            rec.confidence = confidence
        rec.links = [
            l for l in rec.links
        ] + [MemoryLink(target_id=old_id, kind="provenance",
                        note="promoted from short-term")]
        # 保持 id 不变以维持已有链接；晋升标记放 source
        rec.source = {"promoted_from": "short", "original_id": old_id,
                      **rec.source}
        await self._db.update_memory(rec)
        if self._retriever is not None:
            await self._retriever.embed_and_store(rec)
        return rec

    @staticmethod
    def render_recent(items: list[MemoryRecord]) -> str:
        """渲染近期短期记忆注入块。"""
        if not items:
            return ""
        lines = ["## 本会话短期记忆（临时，含指向长期记忆的链接）"]
        for m in items:
            link_note = ""
            if m.links:
                link_note = (
                    " [linked→长期记忆 ×%d]" % len(m.links)
                )
            body = m.content if len(m.content) <= 200 else m.content[:200] + "…"
            lines.append(f"- {m.title or m.kind}:{link_note} {body}")
        return "\n".join(lines)
