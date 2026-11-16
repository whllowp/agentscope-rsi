# -*- coding: utf-8 -*-
"""长期记忆：大量、低频知识层（L0-L3 分层）。

分层语义（参考 TencentDB-Agent-Memory）：

- ``L0_raw`` 原始证据（对话/工具输出摘录）——只作溯源，不参与注入；
- ``L1_atom`` 原子事实/事件——精确回捞；
- ``L2_scenario`` 场景块（"某类故障的处置上下文"）——快速恢复工作上下文；
- ``L3_core`` 核心档案（环境摘要、偏好、约定）——小容量、每轮注入。
"""
from __future__ import annotations

from ..storage import MemoryLink, MemoryRecord, MirrorStore, RSIMemDatabase
from .retrieval import Retriever


class LongTermStore:
    """长期记忆写入与分层查询。"""

    def __init__(
        self,
        db: RSIMemDatabase,
        mirror: MirrorStore,
        retriever: Retriever | None = None,
    ) -> None:
        self._db = db
        self._mirror = mirror
        self._retriever = retriever

    async def add(
        self,
        *,
        title: str,
        content: str,
        layer: str = "L1_atom",
        kind: str = "fact",
        confidence: float = 0.7,
        scope: str = "global",
        source: dict | None = None,
        links: list | None = None,
    ) -> MemoryRecord:
        """写入长期记忆（含去重：同层同标题只更新内容）。"""
        existing = await self._db.list_memories(
            tier="long", layer=layer, limit=500,
        )
        for e in existing:
            if e.title == title:
                e.content = content
                e.confidence = confidence
                e.version += 1
                e.updated_at = e.updated_at
                await self._db.update_memory(e)
                self._mirror.write_longterm(e)
                return e
        record = MemoryRecord(
            tier="long",
            layer=layer,  # type: ignore[arg-type]
            kind=kind,
            title=title,
            content=content,
            confidence=confidence,
            scope=scope,
            links=links or [],
            source=source or {},
        )
        await self._db.insert_memory(record)
        self._mirror.write_longterm(record)
        if self._retriever is not None:
            await self._retriever.embed_and_store(record)
        return record

    async def add_raw_evidence(
        self, *, text: str, source: dict | None = None,
    ) -> MemoryRecord:
        """写入 L0 原始证据（溯源用）。"""
        record = MemoryRecord(
            tier="long",
            layer="L0_raw",
            kind="evidence",
            title=text[:80],
            content=text[:4000],
            confidence=1.0,
            source=source or {},
        )
        await self._db.insert_memory(record)
        return record

    async def archive(self, memory_id: str) -> MemoryRecord | None:
        rec = await self._db.get_memory(memory_id)
        if rec is None or rec.tier != "long":
            return None
        rec.status = "archived"
        await self._db.update_memory(rec)
        self._mirror.write_longterm(rec)
        return rec

    async def merge(
        self, primary_id: str, duplicate_id: str,
    ) -> MemoryRecord | None:
        """合并重复条目：把 duplicate 的要点并入 primary 后归档。

        归档前写 ``source["merged_into"]=primary_id`` 并**即时重定向**
        指向 duplicate 的 related 链接（U3：merge 的链接语义闭环）；
        断链修复插槽对漏网链接按 ``merged_into`` 重定向。幂等：已带
        ``merged_into`` 的 duplicate 直接返回 primary。
        """
        primary = await self._db.get_memory(primary_id)
        duplicate = await self._db.get_memory(duplicate_id)
        if primary is None or duplicate is None:
            return None
        if duplicate.source.get("merged_into"):
            return primary  # 已合并过（幂等）
        if primary.content and duplicate.content not in primary.content:
            primary.content = (
                f"{primary.content}\n\n[合并自 {duplicate_id[:8]}] "
                f"{duplicate.content}"
            )
        primary.version += 1
        await self._db.update_memory(primary)
        self._mirror.write_longterm(primary)
        # 归档前留 merged_into（断链修复的重定向锚点）
        duplicate.source["merged_into"] = primary_id
        await self._db.update_memory(duplicate)
        await self._redirect_links(duplicate_id, primary_id)
        dup = await self.archive(duplicate_id)
        assert dup is not None
        return primary

    async def _redirect_links(
        self, duplicate_id: str, primary_id: str,
    ) -> None:
        """把指向 duplicate 的 related 链接即时重定向到 primary。"""
        hosts = await self._db.list_memories(
            tier="long,short", status="active,promoted", limit=1000,
        )
        for host in hosts:
            if not any(
                l.target_id == duplicate_id and l.kind == "related"
                for l in host.links
            ):
                continue
            kept: list[MemoryLink] = []
            for link in host.links:
                if link.target_id == duplicate_id and link.kind == "related":
                    if not any(x.target_id == primary_id for x in kept):
                        kept.append(MemoryLink(
                            target_id=primary_id, kind="related",
                            note=f"redirect:{duplicate_id[:8]}",
                        ))
                    # 已有指向 primary 的链接则直接丢弃重复链
                else:
                    kept.append(link)
            host.links = kept
            await self._db.update_memory(host)
