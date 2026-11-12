# -*- coding: utf-8 -*-
"""TierManager：三层记忆统一门面。

中间件、记忆工具、固化管线与 RSI 引擎都只跟这个类打交道。职责：

- 组装注入区块（dsh-memory-evolve 的多轨注入：核心档案轨 / 肌肉反射轨 /
  短期近况轨 / 检索轨）；
- 待确认提议的分级确认策略（高置信低风险自动执行，安全敏感一律人工）；
- 三层之间的晋升 / 归档 / 降级 / 合并动作。
"""
from __future__ import annotations

import time
from typing import Any

from agentscope._logging import logger

from ..config import RSIMemConfig
from ..storage import (
    GenomeSpec,
    MemoryRecord,
    MirrorStore,
    PendingReview,
    RSIMemDatabase,
)
from .longterm import LongTermStore
from .muscle import MuscleMatcher
from .retrieval import Retriever, RetrievedMemory
from .shortterm import ShortTermStore

_DANGEROUS_PATTERNS = (
    "rm -rf", "mkfs", "drop table", "drop database", "shutdown",
    "reboot", "systemctl restart", "systemctl stop", "kill -9",
    "truncate table", "del /f", "remove-item", "format ", "drain",
    "cordon", "scale --replicas=0", "删库", "格式化", "下线", "重启生产",
)


class TierManager:
    """三层记忆的统一门面。"""

    def __init__(
        self,
        db: RSIMemDatabase,
        mirror: MirrorStore,
        config: RSIMemConfig,
        retriever: Retriever | None = None,
    ) -> None:
        self._db = db
        self._mirror = mirror
        self._config = config
        self.retriever = retriever or Retriever(db, config,
                                                config.embedding_model)
        self.muscle = MuscleMatcher(db, mirror)
        self.shortterm = ShortTermStore(db, config, self.retriever)
        self.longterm = LongTermStore(db, mirror, self.retriever)

    # ------------------------------------------------------------------
    # 注入区块组装
    # ------------------------------------------------------------------
    async def system_prompt_block(
        self,
        *,
        query: str,
        genome: GenomeSpec,
        tool_hint: str | None = None,
    ) -> str:
        """系统提示词区块：基因组 always 片段 + L3 核心档案 + 命中反射。"""
        sections: list[str] = []

        fragments = genome.fragment_text()
        if fragments:
            sections.append(fragments)

        if genome.retrieval.include_l3:
            core = await self.retriever.core_profile()
            if core:
                sections.append("## 核心档案（长期记忆 L3，每轮可见）")
                sections.extend(f"- {c.display(200)}" for c in core)

        reflexes = await self.muscle.match(
            query,
            tool_hint=tool_hint,
            active_sops=genome.active_sops,
            max_hits=self._config.muscle.max_hits,
            budget_bytes=genome.retrieval.reflex_budget,
        )
        rendered = self.muscle.render_hits(reflexes)
        if rendered:
            sections.append(rendered)

        return "\n\n".join(sections)

    async def reasoning_hint_block(
        self,
        *,
        query: str,
        session_id: str | None,
        genome: GenomeSpec,
        exclude_sessions: tuple[str, ...] = (),
    ) -> tuple[str, list[str]]:
        """推理前注入区块：近期短期记忆（含链接说明）+ 混合检索结果。

        Returns:
            ``(注入文本, 实际注入的记忆 id 列表)`` —— id 列表供 RSI
            轨迹记录（回放回归检查用）。
        """
        sections: list[str] = []
        used_ids: list[str] = []

        recent_n = min(
            self._config.short_term.inject_recent,
            genome.retrieval.short_inject_recent,
        )
        recent = await self.shortterm.recent(session_id, recent_n)
        rendered = self.shortterm.render_recent(recent)
        if rendered:
            sections.append(rendered)
        used_ids.extend(m.id for m in recent)

        results = await self.retriever.search(
            query,
            tiers=("long", "short"),
            top_k=genome.retrieval.top_k,
            score_threshold=genome.retrieval.score_threshold,
            char_budget=genome.retrieval.char_budget,
            exclude_sessions=exclude_sessions,
        )
        rendered = Retriever.render_results(results)
        if rendered:
            sections.append(rendered)
        used_ids.extend(r.record.id for r in results)
        return "\n\n".join(sections), used_ids

    async def search_for_tool(
        self, query: str, limit: int | None,
    ) -> list[RetrievedMemory]:
        """memory_search 工具入口：跨层检索。"""
        return await self.retriever.search(
            query,
            tiers=("long", "short", "muscle"),
            top_k=limit,
        )

    # ------------------------------------------------------------------
    # 待确认提议（分级确认策略）
    # ------------------------------------------------------------------
    @staticmethod
    def is_safety_critical(text: str) -> bool:
        """内容涉及高危操作时必须人工确认。"""
        text_l = text.casefold()
        return any(p in text_l for p in _DANGEROUS_PATTERNS)

    async def propose(self, review: PendingReview) -> str:
        """入队一条提议；高置信且非安全敏感时自动执行。

        Returns:
            ``"auto_applied"`` / ``"pending"``。
        """
        evidence = " ".join(
            [review.reason, str(review.payload.get("content", "")),
             str(review.payload.get("title", ""))],
        )
        auto_ok = (
            review.confidence
            >= self._config.review.auto_accept_confidence
            and not self.is_safety_critical(evidence)
            and review.action != "create_muscle"
        )
        if auto_ok:
            await self.apply_review(review, by="auto")
            return "auto_applied"
        await self._db.enqueue_review(review)
        return "pending"

    async def apply_review(self, review: PendingReview, by: str) -> Any:
        """执行一条提议并标记解决。"""
        result: Any = None
        try:
            result = await self._dispatch(review)
        except Exception as e:  # noqa: BLE001
            logger.warning("apply review %s failed: %s", review.id, e)
            raise
        finally:
            await self._db.resolve_review(review.id, by, time.time())
        return result

    async def _dispatch(self, review: PendingReview) -> Any:
        action = review.action
        payload = review.payload
        if action == "promote_longterm":
            out = await self.shortterm.promote(
                payload["memory_id"],
                layer=payload.get("layer", "L1_atom"),
                confidence=review.confidence,
            )
            if out is not None:
                self._mirror.write_longterm(out)
            return out

        if action == "create_muscle":
            return await self.muscle.note(
                title=payload["title"],
                content=payload["content"],
                triggers=payload.get("triggers", []),
                confidence=review.confidence,
                origin=payload.get("origin", "consolidation"),
                status=payload.get("status", "active"),
                source={"review_id": review.id},
            )

        if action == "create_longterm":
            return await self.longterm.add(
                title=payload["title"],
                content=payload["content"],
                layer=payload.get("layer", "L1_atom"),
                kind=payload.get("kind", "fact"),
                confidence=review.confidence,
                source={"review_id": review.id,
                        **payload.get("source", {})},
            )

        if action == "archive_memory":
            rec = await self._db.get_memory(payload["memory_id"])
            if rec is None:
                return None
            if rec.tier == "muscle":
                return await self.muscle.archive(rec.id)
            return await self.longterm.archive(rec.id)

        if action == "demote_muscle":
            rec = await self._db.get_memory(payload["memory_id"])
            if rec is None:
                return None
            # 降级：肌肉 → 长期 L2 场景（保留知识、退出反射注入）
            scenario = await self.longterm.add(
                title=f"[降级SOP] {rec.title}",
                content=rec.content,
                layer="L2_scenario",
                kind="sop",
                confidence=rec.confidence,
                source={"demoted_from": rec.id},
            )
            await self.muscle.archive(rec.id)
            return scenario

        if action == "merge_memories":
            return await self.longterm.merge(
                payload["primary_id"], payload["duplicate_id"],
            )

        raise ValueError(f"unknown review action: {action}")

    async def pending(self, limit: int = 20) -> list[PendingReview]:
        return await self._db.pending_reviews(limit)

    async def resolve(self, review_id: str, approved: bool) -> Any:
        """用户确认/拒绝一条待确认提议。"""
        review = await self._db.get_review(review_id)
        if review is None or review.resolved_at is not None:
            return None
        if approved:
            return await self.apply_review(review, by="user")
        await self._db.resolve_review(review_id, "rejected", time.time())
        return None

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------
    async def maintenance(self) -> dict[str, int]:
        """TTL 清理等轻量维护，返回清理统计。"""
        expired = await self.shortterm.cleanup()
        return {"expired_short_term": expired}

    async def stats(self) -> dict[str, Any]:
        return await self._db.stats()
