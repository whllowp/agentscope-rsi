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
    MemoryLink,
    MemoryRecord,
    MirrorStore,
    PendingReview,
    RSIMemDatabase,
)
from ..telemetry import CAT_MEMORY, LocalEventLog
from .guards import (
    broken_link_reason,
    flag_contradiction_pair,
    set_verification,
    verification_of,
)
from .longterm import LongTermStore
from .muscle import MuscleMatcher
from .retrieval import Retriever, RetrievedMemory
from .salience import dream_priority, settle_predictions, tag_of
from .shortterm import ShortTermStore

_DANGEROUS_PATTERNS = (
    "rm -rf", "mkfs", "drop table", "drop database", "shutdown",
    "reboot", "systemctl restart", "systemctl stop", "kill -9",
    "truncate table", "del /f", "remove-item", "format ", "drain",
    "cordon", "scale --replicas=0", "删库", "格式化", "下线", "重启生产",
)

L3_INDEX_MARGIN = 400
"""L3 合计预算中的 U4 索引段预留余量（合计口径 = L3 段 + U4 索引段 ≤
``l3_char_budget``）。U3 瘦身插槽与 U4 注入路径共用本常量，防止两处
口径漂移。"""


class TierManager:
    """三层记忆的统一门面。"""

    def __init__(
        self,
        db: RSIMemDatabase,
        mirror: MirrorStore,
        config: RSIMemConfig,
        retriever: Retriever | None = None,
        *,
        events: LocalEventLog | None = None,
    ) -> None:
        self._db = db
        self._mirror = mirror
        self._config = config
        self._events = events
        self.retriever = retriever or Retriever(db, config,
                                                config.embedding_model)
        self.muscle = MuscleMatcher(db, mirror)
        self.shortterm = ShortTermStore(db, config, self.retriever)
        self.longterm = LongTermStore(db, mirror, self.retriever)

    # ------------------------------------------------------------------
    # 注入区块组装
    # ------------------------------------------------------------------
    async def _core_context_sections(
        self,
        *,
        query: str,
        genome: GenomeSpec,
        tool_hint: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """中间件与代理共用的渲染路径（U4）。

        组装：基因组 always 片段 + L3 核心档案（超取 + l3_eligible
        过滤 + **运行时合计截断**）+ 顶部小索引 + 命中反射 —— 同一
        ``l3_char_budget`` 合计预算（索引段字符在 ``L3_INDEX_MARGIN``
        余量内）。L3 段运行时截断与 U3 瘦身插槽同值同口径（字符）：
        预算 ``l3_char_budget − L3_INDEX_MARGIN``，超限按
        (trace_verified, hit_count) 排序保留头部 —— 瘦身提案降级的是
        库内条目，截断兜底的是两次审查之间的注入面（U1 债务⑥）。

        Returns:
            ``(sections, 注入记忆 id 列表)``。
        """
        sections: list[str] = []
        used_ids: list[str] = []

        fragments = genome.fragment_text()
        if fragments:
            sections.append(fragments)

        if genome.retrieval.include_l3:
            core = await self.retriever.core_profile()
            if core:
                sections.append("## 核心档案（长期记忆 L3，每轮可见）")
                budget = (
                    self._config.review.l3_char_budget - L3_INDEX_MARGIN
                )
                ranked = sorted(
                    core,
                    key=lambda c: (
                        verification_of(c).state == "trace_verified",
                        c.hit_count,
                    ),
                    reverse=True,
                )
                kept: list[MemoryRecord] = []
                used = 0
                for c in ranked:
                    cost = len(c.display(200))
                    if used + cost > budget and kept:
                        break  # 保留头部：首个必留，其后按序装满即止
                    used += cost
                    kept.append(c)
                sections.extend(f"- {c.display(200)}" for c in kept)
                used_ids.extend(c.id for c in kept)

        # —— 顶部小索引（U4）：注入不依赖 rebuild，DB 直渲染 ——
        # 索引段按字符口径（与 L3 段合计口径统一；INDEX.md 写盘
        # 仍走 rebuild_wiki 的默认字节口径）
        if self._config.wiki.enabled:
            long_recs = await self._db.list_memories(
                tier="long", status="active,promoted", limit=500,
            )
            index_memories = [
                m for m in long_recs
                if m.layer in ("L1_atom", "L2_scenario")
            ]
            if index_memories:
                block, index_ids = (
                    self._mirror.render_index_block_with_ids(
                        index_memories,
                        max_lines=self._config.wiki.index_max_lines,
                        max_chars=L3_INDEX_MARGIN,
                        unit="chars",
                    )
                )
                if block:
                    sections.append(block)
                    used_ids.extend(index_ids)

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
        used_ids.extend(r.id for r in reflexes)
        return sections, used_ids

    async def system_prompt_block(
        self,
        *,
        query: str,
        genome: GenomeSpec,
        tool_hint: str | None = None,
    ) -> str:
        """系统提示词区块：基因组 always 片段 + L3 核心档案 + 记忆索引
        小节 + 命中反射（U4：L3 段后追加「## 记忆索引」）。"""
        block, _ids = await self.system_prompt_block_with_ids(
            query=query, genome=genome, tool_hint=tool_hint,
        )
        return block

    async def system_prompt_block_with_ids(
        self,
        *,
        query: str,
        genome: GenomeSpec,
        tool_hint: str | None = None,
    ) -> tuple[str, list[str]]:
        """同 :meth:`system_prompt_block`，并返回注入记忆 id 列表。

        id 列表是 U2 验证/隔离机制在**中间件主路径**的信号源：中间件
        把系统提示注入面（L3 / 索引入选 / 命中反射）经
        ``engine.note_injected`` 计入轨迹 ``used_memory_ids`` ——
        使用型 trace_verified、负反馈隔离、漂移再验证都依赖它。
        注入即真实使用：对注入 id 批量 settle ``"hit"``（内部自带
        错误隔离，失败不阻断注入）。
        """
        sections, used_ids = await self._core_context_sections(
            query=query, genome=genome, tool_hint=tool_hint,
        )
        await settle_predictions(self._db, used_ids, "hit")
        return "\n\n".join(sections), used_ids

    async def compact_context_block(
        self,
        *,
        query: str,
        genome: GenomeSpec,
    ) -> tuple[str, list[str]]:
        """代理路径的紧凑上下文（与 system_prompt_block 同一渲染路径）。

        Returns:
            ``(注入文本, 注入记忆 id 列表)`` —— ids 经代理
            ``_record_trace → _save_trace(used_memory_ids)`` 链路落进
            轨迹（使用型验证可达）。注入即真实使用：settle ``hit``。
        """
        sections, used_ids = await self._core_context_sections(
            query=query, genome=genome,
        )
        await settle_predictions(self._db, used_ids, "hit")
        return "\n\n".join(sections), used_ids

    async def reasoning_hint_block(
        self,
        *,
        query: str,
        session_id: str | None,
        genome: GenomeSpec,
        exclude_sessions: tuple[str, ...] = (),
    ) -> tuple[str, list[str]]:
        """推理前注入区块：近期短期记忆（含链接说明）+ 混合检索结果。

        注入即真实使用：收集 ``used_ids`` 后 settle ``"hit"``（settle
        内部错误隔离，失败只告警、不阻断区块返回）。

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
        await settle_predictions(self._db, used_ids, "hit")
        return "\n\n".join(sections), used_ids

    async def search_for_tool(
        self, query: str, limit: int | None,
    ) -> list[RetrievedMemory]:
        """memory_search 工具入口：跨层检索。

        agent 主动检索是真实使用：命中结果同样 settle ``"hit"``（计入
        U2 驻留豁免；不授予 trace_verified —— used_memory_ids 只收注入 id）。
        """
        results = await self.retriever.search(
            query,
            tiers=("long", "short", "muscle"),
            top_k=limit,
        )
        await settle_predictions(
            self._db, [r.record.id for r in results], "hit",
        )
        return results

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

        例外（U2/U3 污染防御）：``quarantine_memory`` 与 ``demote_l3``
        的目标是用户手写（``origin ∈ memory_note/user``）时**强制人工
        确认** —— 手写核心档案只有主人能删/降级（注入会累积 hit_count
        自证存留，但静默自动降级被禁止）；其余污染防御提议（负反馈
        证据 / 驻留上限 / 矛盾 / 非手写瘦身）置信度 0.9 > 默认
        ``auto_accept_confidence``=0.85，自动执行，目标不静默降级。

        Returns:
            ``"auto_applied"`` / ``"pending"``。
        """
        evidence = " ".join(
            [review.reason, str(review.payload.get("content", "")),
             str(review.payload.get("title", ""))],
        )
        hand_written = (
            review.action in ("quarantine_memory", "demote_l3")
            and review.payload.get("origin") in ("user", "memory_note")
        )
        auto_ok = (
            review.confidence
            >= self._config.review.auto_accept_confidence
            and not self.is_safety_critical(evidence)
            and review.action != "create_muscle"
            and not hand_written
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
            # 幂等守卫：同 title 的 active SOP 已存在则直接返回，
            # 不重复 INSERT（配合 _distill_sop 的提议前查重；精确
            # 查询，不吃 list_memories 的 limit 截断 —— U1 债务⑤）
            dup = await self._db.find_by_title(
                tier="muscle", title=payload["title"],
                status="active,promoted",
            )
            if dup is not None:
                return dup
            source: dict = {
                "review_id": review.id,
                "origin": payload.get("origin", "consolidation"),
            }
            if payload.get("trace_ids"):
                # 蒸馏来源轨迹：U2 污染防御的 SOP 关联锚点
                source["trace_ids"] = list(payload["trace_ids"])
            return await self.muscle.note(
                title=payload["title"],
                content=payload["content"],
                triggers=payload.get("triggers", []),
                confidence=review.confidence,
                origin=payload.get("origin", "consolidation"),
                status=payload.get("status", "active"),
                source=source,
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
            if rec is None or rec.status == "archived":
                return None  # 幂等：已降级（归档）不再执行
            # 幂等守卫：降级场景已存在则不重复创建
            scenarios = await self._db.list_memories(
                tier="long", layer="L2_scenario", status="active,promoted",
                limit=200,
            )
            existed = next(
                (
                    s for s in scenarios
                    if s.source.get("demoted_from") == rec.id
                ), None,
            )
            if existed is not None:
                await self.muscle.archive(rec.id)
                return existed
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

        if action == "quarantine_memory":
            # 污染隔离：归档退出一切注入池（muscle.match / 检索 / L3），
            # 并写入 refuted 验证态（retrieval 的 is_injectable 双保险）。
            rec = await self._db.get_memory(payload["memory_id"])
            if rec is None:
                return None
            if (
                rec.status == "archived"
                and verification_of(rec).state == "refuted"
            ):
                return rec  # 幂等：已隔离
            set_verification(
                rec, "refuted",
                [f"quarantine:{review.reason[:120]}"],
            )
            if rec.tier == "muscle":
                await self._db.update_memory(rec)
                return await self.muscle.archive(rec.id)
            rec.status = "archived"
            await self._db.update_memory(rec)
            return rec

        if action == "flag_contradiction":
            # 矛盾旗标：两侧互插旗标；已 trace_verified 的降级待再验证。
            a = await self._db.get_memory(payload["memory_id"])
            b = await self._db.get_memory(payload["other_id"])
            if a is None or b is None:
                return None
            flag_contradiction_pair(a, b, payload.get("detail", ""))
            await self._db.update_memory(a)
            await self._db.update_memory(b)
            return a

        if action == "demote_l3":
            # L3 驻留上限降级：L3_core → L1_atom（tier/status 不动，回放
            # 回归检查 replay.py 只认 tier+status）；先删旧层镜像再写新层。
            rec = await self._db.get_memory(payload["memory_id"])
            if rec is None or rec.layer != "L3_core":
                return None
            self._mirror.remove_longterm(rec)
            rec.layer = "L1_atom"
            rec.version += 1
            rec.source.pop("l3_reviews", None)
            await self._db.update_memory(rec)
            self._mirror.write_longterm(rec)
            # 剩余 L3 追加 related 指向被降级条目：降级知识仍可从
            # L3 注入面经链接展开回捞（检索层 link 通道）。
            cores = await self._db.list_memories(
                tier="long", layer="L3_core",
                status="active,promoted", limit=30,
            )
            for core in cores:
                if core.id == rec.id or any(
                    l.target_id == rec.id for l in core.links
                ):
                    continue
                core.links.append(MemoryLink(
                    target_id=rec.id, kind="related",
                    note="demoted from L3 (residency cap)",
                ))
                await self._db.update_memory(core)
            return rec

        if action == "repair_links":
            # 断链修复（U3 插槽 3 的执行侧）：逐链**重验**后重定向/删链，
            # 不盲信提议时的快照 —— 天然幂等（并发 last-write-wins 基础）。
            host = await self._db.get_memory(payload["memory_id"])
            if host is None:
                return None
            now = time.time()
            kept: list[MemoryLink] = []
            changed = False
            for link in host.links:
                target = await self._db.get_memory(link.target_id)
                if broken_link_reason(target, now) is None:
                    kept.append(link)
                    continue
                changed = True
                merged_into = (
                    (target.source or {}).get("merged_into")
                    if target is not None else None
                )
                primary = (
                    await self._db.get_memory(merged_into)
                    if merged_into else None
                )
                if (
                    primary is not None
                    and broken_link_reason(primary, now) is None
                    and not any(x.target_id == primary.id for x in kept)
                ):
                    kept.append(MemoryLink(
                        target_id=primary.id, kind=link.kind,
                        note=f"redirect:{link.target_id[:8]}",
                    ))
                # 无重定向锚点（或锚点也失效）则删链
            if changed:
                host.links = kept
                await self._db.update_memory(host)
            return host

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
        """TTL 清理等轻量维护，返回清理统计。

        过期结局走遥测：先 ``expired_short_ids`` 取 id、读显著性标注、
        emit ``salience_expired`` 事件（CAT_MEMORY），再真正删除 —— 不写
        predictions 表（predictor 会载入全部 realized 行，混入会污染世界
        模型 MAE）。

        U4：末尾触发 wiki 全量重建 —— review_session 末尾经本方法到达
        （中间件与代理两条链路都触发 review），首建最迟第
        ``every_n_turns`` 轮。
        """
        now = time.time()
        expired_ids = await self._db.expired_short_ids(now)
        if expired_ids and self._events is not None:
            await self._emit_expired(expired_ids)
        expired = await self.shortterm.cleanup()
        try:
            await self.rebuild_wiki()
        except Exception as e:  # noqa: BLE001 — 镜像失败不阻塞维护
            logger.warning("wiki rebuild failed: %s", e)
        return {"expired_short_term": expired}

    async def rebuild_wiki(self) -> int:
        """全量重建长期记忆 wiki 镜像，返回写入的主题页数。

        INDEX.md 首建前注入侧仍可从 DB 直渲染（render_index_block
        不依赖 rebuild，保持写穿副本语义）；rebuild 只维护镜像。
        """
        if not self._config.wiki.enabled:
            return 0
        memories = await self._db.list_memories(
            tier="long", status="active,promoted", limit=1000,
        )
        memories = [
            m for m in memories if m.layer in ("L1_atom", "L2_scenario")
        ]
        return self._mirror.rebuild_wiki(
            memories,
            index_max_lines=self._config.wiki.index_max_lines,
            index_max_bytes=self._config.wiki.index_max_bytes,
        )

    async def _emit_expired(self, ids: list[str]) -> None:
        """过期短期记忆的显著性结局落遥测（尽力而为，失败只告警）。"""
        assert self._events is not None
        try:
            items = []
            for mid in ids:
                rec = await self._db.get_memory(mid)
                tag = tag_of(rec) if rec is not None else None
                items.append({
                    "id": mid,
                    "kind": tag.kind if tag else None,
                    "volatile": tag.volatile if tag else None,
                    "predicted_value": (
                        tag.predicted_value if tag else None
                    ),
                    "priority": dream_priority(tag) if tag else None,
                })
            await self._events.emit(CAT_MEMORY, {
                "event": "salience_expired",
                "items": items,
            })
        except Exception as e:  # noqa: BLE001 — 遥测失败不阻塞维护
            logger.warning("salience_expired telemetry failed: %s", e)

    async def stats(self) -> dict[str, Any]:
        return await self._db.stats()
