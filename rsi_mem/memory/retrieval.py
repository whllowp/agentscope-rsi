# -*- coding: utf-8 -*-
"""混合检索：FTS5 bm25 + 可选向量 + RRF 融合 + 短期→长期链接展开。

参考 TencentDB-Agent-Memory 的检索设计：多路召回、RRF 融合、条数/字符
预算封顶。短期记忆命中时顺带浮出其链接（``links``）指向的长期记忆 ——
这是"短期记忆建立索引连接到长期记忆"在检索侧的体现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from agentscope._logging import logger
from agentscope.embedding import EmbeddingModelBase

from ..config import RSIMemConfig
from ..storage import MemoryRecord, RSIMemDatabase

RetrievalVia = Literal["bm25", "vector", "link", "l3", "reflex"]

_RRF_K = 60
_LINK_SCORE = 0.8
"""链接展开得分为父命中分 × 该系数（低于直接命中）。"""


@dataclass
class RetrievedMemory:
    """一条检索结果。"""

    record: MemoryRecord
    score: float
    via: RetrievalVia
    parent_id: str | None = field(default=None)

    def render(self, max_chars: int = 400) -> str:
        src = "" if self.parent_id is None else f" ←linked:{self.parent_id[:8]}"
        return f"({self.via} {self.score:.2f}){src} {self.record.display(max_chars)}"


class Retriever:
    """三层记忆的统一检索入口。"""

    def __init__(
        self,
        db: RSIMemDatabase,
        config: RSIMemConfig,
        embedding_model: EmbeddingModelBase | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._embedding = embedding_model

    # ------------------------------------------------------------------
    # 向量维护
    # ------------------------------------------------------------------
    async def embed_and_store(self, record: MemoryRecord) -> None:
        """为长期/肌肉条目建立向量（短期层不建，易变且量小）。"""
        if self._embedding is None or record.tier == "short":
            return
        text = f"{record.title}\n{record.content}"[:2000]
        try:
            resp = await self._embedding([text])
            await self._db.upsert_vector(record.id, resp.embeddings[0])
        except Exception as e:  # noqa: BLE001
            logger.warning("embed failed for %s: %s", record.id, e)

    async def _vector_scores(
        self, query: str, tiers: Sequence[str],
    ) -> dict[str, float]:
        """向量余弦相似度，返回 {memory_id: cos}。"""
        if self._embedding is None:
            return {}
        try:
            resp = await self._embedding([query[:2000]])
            qvec = np.asarray(resp.embeddings[0], dtype=float)
        except Exception as e:  # noqa: BLE001
            logger.warning("query embed failed: %s", e)
            return {}
        out: dict[str, float] = {}
        for mid, vec in await self._db.load_vectors(tiers):
            v = np.asarray(vec, dtype=float)
            denom = float(np.linalg.norm(qvec) * np.linalg.norm(v))
            if denom > 0:
                out[mid] = float(np.dot(qvec, v) / denom)
        return out

    # ------------------------------------------------------------------
    # 主检索
    # ------------------------------------------------------------------
    async def search(
        self,
        query: str,
        *,
        tiers: Sequence[str] = ("long", "short"),
        top_k: int | None = None,
        score_threshold: float | None = None,
        char_budget: int | None = None,
        expand_links: bool = True,
        exclude_sessions: Sequence[str] = (),
    ) -> list[RetrievedMemory]:
        """混合检索并对结果做预算裁剪。

        Args:
            query: 查询文本。
            tiers: 参与检索的层。
            top_k / score_threshold / char_budget: 默认取配置，可被
                基因组检索参数覆盖（RSI 回放时传入候选参数）。
            expand_links: 是否把短期命中的长期链接展开进结果。
            exclude_sessions: 排除这些会话的短期记忆（跨会话隔离）。
        """
        cfg = self._config.long_term
        top_k = top_k if top_k is not None else cfg.top_k
        threshold = (
            score_threshold
            if score_threshold is not None else 0.0
        )
        budget = char_budget if char_budget is not None else cfg.char_budget
        if not query.strip():
            return []

        bm25 = await self._db.search_text(query, tiers=tiers, limit=top_k * 3)
        vec = await self._vector_scores(query, tiers)

        # RRF 融合（各自排名 → 1/(k+rank) 求和）
        fused: dict[str, float] = {}
        via: dict[str, RetrievalVia] = {}
        for rank, (mid, _) in enumerate(bm25):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            via.setdefault(mid, "bm25")
        ranked_vec = sorted(vec.items(), key=lambda x: -x[1])
        for rank, (mid, _) in enumerate(ranked_vec[: top_k * 3]):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if mid in vec and (mid not in via or vec[mid] > 0.5):
                via[mid] = "vector"

        results: list[RetrievedMemory] = []
        seen: set[str] = set()
        used_chars = 0
        ordered = sorted(fused.items(), key=lambda x: -x[1])
        for mid, raw in ordered:
            if len(results) >= top_k:
                break
            rec = await self._db.get_memory(mid)
            if rec is None or rec.status not in ("active", "promoted"):
                continue
            if (
                rec.tier == "short"
                and rec.session_id
                and rec.session_id in exclude_sessions
            ):
                continue
            score = min(1.0, raw * 10)  # RRF 原始分很小，放大便于阈值
            if score < threshold:
                continue
            display_len = len(rec.display())
            if used_chars + display_len > budget:
                continue
            used_chars += display_len
            results.append(
                RetrievedMemory(record=rec, score=round(score, 4),
                                via=via.get(mid, "bm25")),
            )
            seen.add(mid)

        # 短期→长期链接展开（不占 top_k 名额，但受字符预算约束）
        if expand_links:
            for hit in list(results):
                if hit.record.tier != "short":
                    continue
                for link in hit.record.links:
                    if link.target_id in seen or link.kind != "related":
                        continue
                    target = await self._db.get_memory(link.target_id)
                    if (
                        target is None
                        or target.status not in ("active", "promoted")
                        or target.tier not in tiers
                    ):
                        continue
                    display_len = len(target.display())
                    if used_chars + display_len > budget:
                        continue
                    used_chars += display_len
                    seen.add(link.target_id)
                    results.append(
                        RetrievedMemory(
                            record=target,
                            score=round(hit.score * _LINK_SCORE, 4),
                            via="link",
                            parent_id=hit.record.id,
                        ),
                    )

        if results:
            await self._db.touch_hits([r.record.id for r in results])
        return results

    async def core_profile(self) -> list[MemoryRecord]:
        """长期记忆 L3 核心档案（小容量、每轮注入）。"""
        return await self._db.list_memories(
            tier="long", layer="L3_core", status="active,promoted", limit=10,
        )

    @staticmethod
    def render_results(results: list[RetrievedMemory]) -> str:
        """把结果渲染为注入文本。"""
        if not results:
            return ""
        lines = ["## 记忆检索结果（短期/长期联动）"]
        for r in results:
            lines.append(f"- {r.render()}")
        return "\n".join(lines)
