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

import time
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from agentscope._logging import logger
from agentscope.model import ChatModelBase

from ..config import RSIMemConfig
from ..storage import MemoryRecord, PendingReview, TraceRecord
from .guards import (
    broken_link_reason,
    find_contradictions,
    l3_eligible,
    llm_recheck,
    mark_stale,
    set_verification,
    verification_of,
)
from .salience import dream_priority, tag_of
from .tiers import L3_INDEX_MARGIN, TierManager

_SOP_MIN_REPEAT = 2

_DEDUP_SCAN_LIMIT = 300
"""内容相似去重的 O(n²) 扫描上限（防大库膨胀；标题精确去重不受限）。"""


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
        """对一个会话执行审查（以轨迹为 ground truth），返回新提议。

        插槽结构与预算口径（config.ReviewConfig docstring 同步写死）：

        - **晋升插槽**：``_propose_promotions`` 维持 ``out[:5]``，不计数；
        - **SOP 蒸馏插槽**：``_distill_sop`` 单条（含同 title 查重），
          不计数；
        - **维护+污染插槽**（合计 ``maintenance_budget``=12 封顶）：
          (1) ``_verify_against_traces``（U2 污染防御原样收编：LLM 复核
          / 使用型验证、负反馈隔离、L3 驻留、漂移再验证）
          (2) ``_propose_dedup``（标题精确 + 内容 difflib ≥0.85 相似）
          (3) ``_propose_link_repair``（断链重定向/删除）
          (4) ``_propose_injection_slimming``（L3 合计预算瘦身）
          + ``_propose_maintenance``（肌肉降级 + L0_raw 回收）。
          long 层超 ``long_soft_cap`` 时去重/瘦身插槽提到队首 —— 软信号
          提升瘦身优先级，非写入墙。
        """
        proposals: list[PendingReview] = []
        proposals.extend(await self._propose_promotions(session_id))
        sop = await self._distill_sop(session_id, traces)
        if sop is not None:
            proposals.append(sop)

        verify = await self._verify_against_traces(traces, model=model)
        dedup = await self._propose_dedup()
        link_repair = await self._propose_link_repair()
        slimming = await self._propose_injection_slimming()
        legacy = await self._propose_maintenance()

        stats = await self._tiers._db.stats()
        over_cap = (
            stats.get("long", 0) > self._config.review.long_soft_cap
        )
        maintenance = (
            [*dedup, *slimming, *verify, *link_repair, *legacy]
            if over_cap else
            [*verify, *dedup, *link_repair, *slimming, *legacy]
        )
        proposals.extend(
            maintenance[: self._config.review.maintenance_budget],
        )
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
    # 维护扫描游标（U1 债务⑤）
    # ------------------------------------------------------------------
    async def _cursor_scan(
        self, slot: str, limit: int, **filters: Any,
    ) -> list[MemoryRecord]:
        """判定型维护扫描的游标轮转。

        读 ``meta["scan_cursor:<slot>"]``（``"<updated_at>:<id>"`` 复
        合游标）作 ``list_memories`` 的 ``(updated_before, cursor_id)``
        边界——主边界 ``<=`` 含边界行（判定型扫描幂等，重复无害）；
        并列时间戳组超过 ``limit`` 时纯时间戳边界会永远重复同一窗
        （SQLite 稳定排序，本会话实测），组内按 id 走窗保证不跳行。
        本轮结果数 < ``limit`` 视为扫到尾，游标重置（空串）下一轮回
        到最新端。防单轮 ``list_memories(limit=…)`` 只见最新窗口、
        老记录永远轮不到审查的截断问题。

        注意：**存在性检查不得走本方法**（游标窗口外漏判）——如 SOP
        查重用 :meth:`RSIMemDatabase.find_by_title` 精确查询。
        """
        db = self._tiers._db
        key = f"scan_cursor:{slot}"
        raw = await db.get_meta(key)
        cursor_ts: float | None = None
        cursor_id: str | None = None
        if raw:
            ts_text, _, id_text = raw.partition(":")
            cursor_ts = float(ts_text)
            cursor_id = id_text or None
        batch = await db.list_memories(
            limit=limit, updated_before=cursor_ts, cursor_id=cursor_id,
            **filters,
        )
        if len(batch) < limit:
            await db.set_meta(key, "")  # 扫到尾：游标归位，下轮从头
        else:
            last = batch[-1]  # updated_at DESC, id DESC 序的窗口末位
            await db.set_meta(key, f"{last.updated_at}:{last.id}")
        return batch

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
        """从轨迹中找重复出现 ≥ N 次的工具序列，蒸馏 SOP 提议。

        提议前查重（U3 无界增长点修复 a）：同 title 的 active SOP
        已存在则跳过 —— 否则 ``create_muscle`` 走 ``muscle.note``
        无条件 INSERT，每 N 轮可重复插入同一 SOP。
        """
        session_traces = [
            t for t in traces
            if session_id is None or t.session_id == session_id
        ]
        if len(session_traces) < self._config.review.min_turns_for_sop:
            return None
        seqs: Counter[tuple[str, ...]] = Counter()
        sample_query = ""
        contributing: list[str] = []
        for t in session_traces:
            names = tuple(c.name for c in t.tool_calls if c.state == "ok")
            if len(names) >= 2:
                seqs[names] += 1
                contributing.append(t.id)
                if not sample_query:
                    sample_query = t.query[:60]
        if not seqs:
            return None
        seq, count = seqs.most_common(1)[0]
        if count < _SOP_MIN_REPEAT:
            return None
        # —— SOP 查重：同 title active SOP 已存在 → 不再提议 ——
        # 精确查询（U1 债务⑤）：存在性检查不能走游标窗口（窗口外
        # 漏判重复），也不能靠 list_memories(limit=100) 截断扫描
        title = f"SOP: {sample_query or '常见运维流程'}"
        if await self._tiers._db.find_by_title(
            tier="muscle", title=title, status="active,promoted",
        ) is not None:
            return None
        steps = "\n".join(f"{i}. 执行 `{n}`" for i, n in enumerate(seq, 1))
        content = (
            f"处理类似「{sample_query}」的问题时按以下流程：\n{steps}\n"
            f"（该流程在 {count} 条轨迹中重复出现，已验证有效）"
        )
        return PendingReview(
            action="create_muscle",
            payload={
                "title": title,
                "content": content,
                "triggers": [w for w in sample_query.split() if w][:5],
                "origin": "consolidation",
                # 来源轨迹 id：U2 污染防御的 SOP ↔ 轨迹关联锚点
                # （写入 muscle.source.trace_ids，负反馈可回溯隔离）
                "trace_ids": contributing,
            },
            reason=f"工具序列 {' → '.join(seq)} 重复出现 {count} 次，"
                   f"建议固化为 SOP",
            confidence=min(0.95, 0.6 + 0.1 * count),
        )

    # ------------------------------------------------------------------
    # 维护+污染插槽（U3 四插槽）：轨迹校验 / 去重 / 断链修复 / 瘦身
    # ------------------------------------------------------------------
    async def _verify_against_traces(
        self,
        traces: list[TraceRecord],
        *,
        model: ChatModelBase | None = None,
    ) -> list[PendingReview]:
        """插槽 (1)：以轨迹为 ground truth 的校验（U2 语义原样收编）。"""
        """对注入面（L3 档案 + active SOP）执行污染防御检查。

        - **判定顺序**：``feedback==-1`` 先于一切 score 判定 —— 关联
          轨迹（L3 被注入 / SOP 蒸馏来源）收到 👎 即提议
          ``quarantine_memory``（0.9，默认阈值下自动执行；用户手写
          origin 强制人工），本轮不再做验证判定；
        - **trace_verified 授予**（创建轨迹的回合级 score 不单独授予）：
          (a) backend_model 对照创建轨迹 query/digest 复核通过，或
          (b) 使用型：``id ∈ t.used_memory_ids`` 且 ``t.id ≠
          source.trace_id`` 且（feedback==1 或 score≥0.7）；
        - **L3 驻留**：unverified 且 hit_count==0 每轮 ``l3_reviews``+1，
          trace_verified 或有命中清零，达上限提议 ``demote_l3``（0.9）；
        - **漂移再验证**：trace_verified 连续 ``verify_refresh_reviews``
          轮无新命中 → mark_stale 重置 unverified（保留 evidence 历史）；
        - **矛盾检测**：L3 面内标题相似且关键事实互斥 →
          ``flag_contradiction``（0.9）。
        """
        db = self._tiers._db
        model = model or self._config.backend_model
        proposals: list[PendingReview] = []
        memories = [
            # L3 注入面小（≤30），直接超取；muscle 面可大 → 游标轮转
            *await db.list_memories(
                tier="long", layer="L3_core",
                status="active,promoted", limit=30,
            ),
            *await self._cursor_scan(
                "verify_muscle", 100,
                tier="muscle", status="active,promoted",
            ),
        ]
        pending_keys = {
            (p.action, p.memory_id)
            for p in await self._tiers.pending(100)
        }
        scored = [t for t in traces if t.outcome is not None]
        for rec in memories:
            if verification_of(rec).state == "refuted":
                continue  # 已隔离（隔离执行写 refuted）
            origin = str(rec.source.get("origin") or "")
            used = [t for t in scored if rec.id in t.used_memory_ids]
            sop_linked = [
                t for t in scored
                if rec.source.get("trace_ids")
                and t.id in rec.source["trace_ids"]
            ]
            # —— 负反馈先于一切 score 判定 ——
            neg = next(
                (t for t in (*used, *sop_linked)
                 if t.outcome.feedback == -1),
                None,
            )
            if neg is not None:
                if ("quarantine_memory", rec.id) not in pending_keys:
                    proposals.append(PendingReview(
                        action="quarantine_memory",
                        memory_id=rec.id,
                        payload={"memory_id": rec.id, "origin": origin},
                        reason=(
                            f"记忆[{rec.title[:40]}]关联轨迹 "
                            f"{neg.id[:8]} 收到用户负反馈，疑似污染"
                        ),
                        confidence=0.9,
                    ))
                continue
            # —— trace_verified 授予 ——
            creation_id = rec.source.get("trace_id")
            verified_by = next(
                (
                    t for t in used
                    if t.id != creation_id
                    and (t.outcome.feedback == 1
                         or t.outcome.score >= 0.7)
                ),
                None,
            )
            dirty = False
            if verified_by is not None:
                set_verification(
                    rec, "trace_verified", [f"trace:{verified_by.id}"],
                )
                rec.source["v_reviews"] = 0
                rec.source["v_hits"] = rec.hit_count
                rec.source.pop("l3_reviews", None)
                dirty = True
            elif verified_by is None and model is not None and creation_id:
                ct = next(
                    (t for t in traces if t.id == creation_id), None,
                )
                if ct is not None and await llm_recheck(model, rec, ct):
                    set_verification(
                        rec, "trace_verified", [f"llm_recheck:{ct.id}"],
                    )
                    rec.source["v_reviews"] = 0
                    rec.source["v_hits"] = rec.hit_count
                    rec.source.pop("l3_reviews", None)
                    dirty = True
            state = verification_of(rec).state
            if state == "trace_verified":
                # —— 漂移再验证：连续 N 轮无新命中即重置 ——
                reviews = int(rec.source.get("v_reviews") or 0) + 1
                if rec.hit_count > int(rec.source.get("v_hits") or 0):
                    rec.source["v_hits"] = rec.hit_count
                    rec.source["v_reviews"] = 0
                elif (
                    reviews > self._config.review.verify_refresh_reviews
                ):
                    mark_stale(
                        rec, f"连续 {reviews} 次审查无新命中，待再验证",
                    )
                    rec.source.pop("v_reviews", None)
                    rec.source.pop("v_hits", None)
                else:
                    rec.source["v_reviews"] = reviews
                dirty = True
            elif rec.tier == "long" and rec.layer == "L3_core":
                # —— L3 驻留上限计数 ——
                if rec.hit_count > 0:
                    rec.source.pop("l3_reviews", None)
                else:
                    n = int(rec.source.get("l3_reviews") or 0) + 1
                    rec.source["l3_reviews"] = n
                    if (
                        n >= self._config.review.l3_unverified_max_reviews
                        and ("demote_l3", rec.id) not in pending_keys
                    ):
                        proposals.append(PendingReview(
                            action="demote_l3",
                            memory_id=rec.id,
                            payload={
                                "memory_id": rec.id,
                                # 手写 origin → propose 强制人工
                                "origin": origin,
                            },
                            reason=(
                                f"L3 档案[{rec.title[:40]}]连续 {n} 次"
                                f"审查未验证且零命中，触发驻留上限降级"
                            ),
                            confidence=0.9,
                        ))
                dirty = True
            if dirty:
                await db.update_memory(rec)
        # —— 矛盾检测（L3 注入面；已插过旗标的对不重复提议）——
        l3s = [m for m in memories if m.tier == "long"]
        for a, b, detail in await find_contradictions(db, l3s, model=model):
            already = any(
                isinstance(f, dict) and f.get("with") == b.id
                for f in (a.source.get("contradiction_flags") or [])
            )
            if already or ("flag_contradiction", a.id) in pending_keys:
                continue
            proposals.append(PendingReview(
                action="flag_contradiction",
                memory_id=a.id,
                payload={
                    "memory_id": a.id, "other_id": b.id, "detail": detail,
                },
                reason=(
                    f"L3 档案疑似矛盾：{a.title[:30]} ↔ {b.title[:30]}"
                    f"（{detail}）"
                ),
                confidence=0.9,
            ))
        return proposals

    # ------------------------------------------------------------------
    # 插槽 (2)：内容级去重
    # ------------------------------------------------------------------
    async def _propose_dedup(self) -> list[PendingReview]:
        """插槽 (2)：标题精确重复 + difflib 内容相似 ≥ 阈值 → merge。

        ``long_soft_cap`` 超限时阈值放宽 0.85→0.75（软信号促合并，
        非写入墙）；内容相似为 O(n²)，限 ``_DEDUP_SCAN_LIMIT`` 条防
        大库扫描膨胀（标题精确去重不受限）。
        """
        db = self._tiers._db
        atoms = await db.list_memories(
            tier="long", layer="L1_atom",
            status="active,promoted", limit=800,
        )
        out: list[PendingReview] = []
        # —— 标题精确重复（原 _propose_maintenance 逻辑收编）——
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
        # —— 内容相似（软上限超限时放宽阈值）——
        stats = await db.stats()
        over_cap = (
            stats.get("long", 0) > self._config.review.long_soft_cap
        )
        threshold = 0.75 if over_cap else 0.85
        scan = atoms[:_DEDUP_SCAN_LIMIT]
        proposed: set[tuple[str, str]] = set()
        for i, a in enumerate(scan):
            for b in scan[i + 1:]:
                if a.title == b.title:
                    continue  # 标题重复已提议
                key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                if key in proposed:
                    continue
                if SequenceMatcher(
                    None, a.content or "", b.content or "",
                ).ratio() >= threshold:
                    proposed.add(key)
                    out.append(PendingReview(
                        action="merge_memories",
                        payload={"primary_id": a.id, "duplicate_id": b.id},
                        reason=(
                            f"长期记忆内容相似度 ≥{threshold}："
                            f"{a.title[:30]} ≈ {b.title[:30]}"
                        ),
                        confidence=0.8,
                    ))
        return out

    # ------------------------------------------------------------------
    # 插槽 (3)：断链修复
    # ------------------------------------------------------------------
    async def _propose_link_repair(self) -> list[PendingReview]:
        """插槽 (3)：扫描 long+short 的链接，断链 → ``repair_links``。

        扫描依赖 ``list_memories`` 的 tier 逗号拆分（U3 接缝修复：
        此前传 ``tier="long,short"`` 得空集、修复静默 no-op）。断链
        判定见 :func:`guards.broken_link_reason`；target 带
        ``merged_into`` → 重定向 primary，否则删链。执行侧逐链重验，
        幂等。
        """
        db = self._tiers._db
        now = time.time()
        hosts = await self._cursor_scan(
            "link_repair", 500, tier="long,short", status="active,promoted",
        )
        out: list[PendingReview] = []
        for host in hosts:
            if not host.links:
                continue
            repairs: list[dict] = []
            for link in host.links:
                target = await db.get_memory(link.target_id)
                if broken_link_reason(target, now) is None:
                    continue
                merged_into = (
                    (target.source or {}).get("merged_into")
                    if target is not None else None
                )
                primary = (
                    await db.get_memory(merged_into)
                    if merged_into else None
                )
                if primary is not None and broken_link_reason(
                    primary, now,
                ) is None:
                    repairs.append({
                        "target": link.target_id,
                        "action": "redirect",
                        "to": primary.id,
                    })
                else:
                    repairs.append({
                        "target": link.target_id, "action": "remove",
                    })
            if repairs:
                out.append(PendingReview(
                    action="repair_links",
                    memory_id=host.id,
                    payload={"memory_id": host.id, "repairs": repairs},
                    reason=(
                        f"记忆[{host.title[:40]}]存在 {len(repairs)} 条"
                        f"断链，提议重定向/删除"
                    ),
                    confidence=0.9,
                ))
        return out

    # ------------------------------------------------------------------
    # 插槽 (4)：L3 注入面合计预算瘦身
    # ------------------------------------------------------------------
    async def _propose_injection_slimming(self) -> list[PendingReview]:
        """插槽 (4)：L3 段渲染字符合计超 ``l3_char_budget`` → demote_l3。

        合计口径（写死）：L3 段 + U4 索引段 ≤ ``l3_char_budget``=1200，
        实现预留 ``L3_INDEX_MARGIN``=400 索引余量（与 U4 注入路径共用
        同一常量）。超限按
        (trace_verified, hit_count, dream_priority) 排序保留头部，其余
        复用 U2 ``demote_l3``（tier/status 不动保证回放兼容；留 related
        链接保持 L3 面可达）。
        """
        db = self._tiers._db
        cores = await db.list_memories(
            tier="long", layer="L3_core",
            status="active,promoted", limit=30,
        )
        eligible = [c for c in cores if l3_eligible(c)]
        if not eligible:
            return []
        budget = self._config.review.l3_char_budget - L3_INDEX_MARGIN
        total = sum(len(c.display(200)) for c in eligible)
        if total <= budget:
            return []
        ranked = sorted(
            eligible,
            key=lambda c: (
                verification_of(c).state == "trace_verified",
                c.hit_count,
                (dream_priority(tag) if (tag := tag_of(c)) else 0.0),
            ),
            reverse=True,
        )
        keep: set[str] = set()
        used = 0
        for c in ranked:
            cost = len(c.display(200))
            if used + cost > budget and keep:
                break  # 保留头部：首个必留，其后按序装满即止
            used += cost
            keep.add(c.id)
        return [
            PendingReview(
                action="demote_l3",
                memory_id=c.id,
                payload={
                    "memory_id": c.id,
                    # 手写 origin → propose 强制人工（主人才能降级）
                    "origin": str(c.source.get("origin") or ""),
                },
                reason=(
                    f"L3 注入段合计 {total} 字符超预算 {budget}，"
                    f"瘦身降级低价值条目"
                ),
                confidence=0.9,
            )
            for c in eligible if c.id not in keep
        ]

    # ------------------------------------------------------------------
    # 既有维护：肌肉降级 + L0_raw 回收
    # ------------------------------------------------------------------
    async def _propose_maintenance(self) -> list[PendingReview]:
        out: list[PendingReview] = []
        muscles = await self._cursor_scan(
            "maintenance_muscle", 100, tier="muscle", status="active",
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
        out.extend(await self._propose_l0_recycling())
        return out[:8]

    async def _propose_l0_recycling(self) -> list[PendingReview]:
        """L0_raw 回收（U3 无界增长点修复 b）。

        未被任何 related/provenance 链接引用且总量超
        ``long_soft_cap/2`` → ``archive_memory`` 提议（0.6 人工确认）。
        """
        db = self._tiers._db
        l0s = await self._cursor_scan(
            "l0_recycle", 1000,
            tier="long", layer="L0_raw", status="active,promoted",
        )
        if len(l0s) <= self._config.review.long_soft_cap // 2:
            return []
        referenced: set[str] = set()
        for rec in await db.list_memories(
            tier="long,short", status="active,promoted", limit=2000,
        ):
            for link in rec.links:
                if link.kind in ("related", "provenance"):
                    referenced.add(link.target_id)
        return [
            PendingReview(
                action="archive_memory",
                memory_id=l0.id,
                payload={"memory_id": l0.id},
                reason=(
                    f"L0 原始证据[{l0.title[:30]}]未被任何链接引用，"
                    f"建议回收归档"
                ),
                confidence=0.6,
            )
            for l0 in l0s if l0.id not in referenced
        ]


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
