# -*- coding: utf-8 -*-
"""肌肉记忆：长期且高效的 SOP / 条件反射层。

对应运维同学"自己的 SOP 和条件反射"：触发器（关键词 / ``re:`` 前缀
正则 / ``tool:`` 前缀工具名）命中即注入，全程**规则匹配、零 LLM 调用**
—— 这是它"高效"的来源。条目同时镜像为可加载的 SKILL.md 技能。
"""
from __future__ import annotations

import re

from ..storage import MemoryRecord, MirrorStore, RSIMemDatabase


def _match_trigger(trigger: str, text: str, tool_hint: str | None) -> bool:
    """判断单个触发器是否命中。"""
    if trigger.startswith("re:"):
        try:
            return re.search(trigger[3:], text) is not None
        except re.error:
            return False
    if trigger.startswith("tool:"):
        return tool_hint is not None and trigger[5:] == tool_hint
    return trigger.casefold() in text


class MuscleMatcher:
    """肌肉记忆匹配与写入。"""

    def __init__(
        self,
        db: RSIMemDatabase,
        mirror: MirrorStore,
    ) -> None:
        self._db = db
        self._mirror = mirror

    async def match(
        self,
        text: str,
        *,
        tool_hint: str | None = None,
        active_sops: list[str] | None = None,
        max_hits: int = 3,
        budget_bytes: int = 2048,
    ) -> list[MemoryRecord]:
        """对当前轮文本做反射匹配。

        Args:
            text: 本轮用户输入 + 最近上下文摘要。
            tool_hint: 上一轮失败/常用工具名（``tool:`` 触发器用）。
            active_sops: 基因组激活的 SOP id；空列表表示全部 active。
            max_hits / budget_bytes: 注入上限。
        """
        candidates = await self._db.list_memories(
            tier="muscle", status="active,promoted,candidate", limit=100,
        )
        if active_sops:
            order = {mid: i for i, mid in enumerate(active_sops)}
            candidates = [
                c for c in candidates if c.id in order
            ] or candidates
            candidates.sort(key=lambda c: order.get(c.id, 1 << 30))
        text_l = text.casefold()
        hits = [
            c for c in candidates
            if any(_match_trigger(t, text_l, tool_hint) for t in c.triggers)
        ]
        # 命中的候选 SOP（RSI 未部署产物）不注入
        hits = [h for h in hits if h.status != "candidate"]
        hits.sort(key=lambda c: -c.hit_count)
        out: list[MemoryRecord] = []
        used = 0
        for h in hits:
            cost = len((h.title + h.content).encode("utf-8"))
            if len(out) >= max_hits or used + cost > budget_bytes:
                break
            used += cost
            out.append(h)
        if out:
            await self._db.touch_hits([h.id for h in out])
        return out

    async def note(
        self,
        *,
        title: str,
        content: str,
        triggers: list[str],
        kind: str = "sop",
        confidence: float = 0.9,
        origin: str = "user",
        status: str = "active",
        source: dict | None = None,
    ) -> MemoryRecord:
        """写入一条肌肉记忆并镜像为 SKILL.md。"""
        record = MemoryRecord(
            tier="muscle",
            layer="sop" if kind == "sop" else "reflex",
            kind=kind,
            title=title,
            content=content,
            triggers=triggers,
            confidence=confidence,
            status=status,  # type: ignore[arg-type]
            source={"origin": origin, **(source or {})},
        )
        await self._db.insert_memory(record)
        self._mirror.write_muscle(record)
        return record

    async def archive(self, memory_id: str) -> MemoryRecord | None:
        """归档肌肉记忆（同时清理镜像，回退为长期记忆语义由调用方处理）。"""
        rec = await self._db.get_memory(memory_id)
        if rec is None or rec.tier != "muscle":
            return None
        rec.status = "archived"
        await self._db.update_memory(rec)
        self._mirror.remove_muscle(rec.id, rec.title)
        return rec

    @staticmethod
    def render_hits(hits: list[MemoryRecord]) -> str:
        """渲染为注入系统提示词的反射区块。"""
        if not hits:
            return ""
        lines = ["## 肌肉记忆（命中的 SOP / 条件反射，按流程执行）"]
        for h in hits:
            head = h.title or "(untitled sop)"
            lines.append(f"### {head}")
            lines.append(h.content.strip())
        return "\n".join(lines)
