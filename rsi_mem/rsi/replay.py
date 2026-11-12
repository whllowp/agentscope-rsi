# -*- coding: utf-8 -*-
"""回放模拟器（Dream-RSI §3 的"离线 dreaming"）。

Dream-RSI 把在线录制的历史当作"回放世界"：候选策略不必重新上线跑，
而是在录制轨迹上做**反事实评估**，一次昂贵在线运行换取多次廉价评估。
本项目里可确定性回放的部分：

- **检索重放**：用候选基因组的检索参数在历史轨迹的查询上重跑检索
  （记忆库不变，检索是纯函数），得到候选会注入什么；
- **成本重估**：注入字符差折算 token 差，叠加该轨迹真实 token 用量；
- **回归检查**：成功轨迹实际用到（``used_memory_ids``）的可检索记忆，
  候选参数下必须仍然召回 —— 这是 Harmony 的回归保障；
- **LLM 评审**（可选后台模型）：基于证据的预期质量增量。

安全目标与结构合法性由校验器保证；三目标合成见 ``objectives.py``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json
from typing import Any

import json_repair
from agentscope._logging import logger
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase

from ..config import RSIMemConfig
from ..memory.retrieval import Retriever
from ..storage import GenomeSpec, RSIMemDatabase, TraceRecord
from .objectives import (
    ObjectiveVector,
    cost_efficiency,
    safety_from_signals,
)

_CHARS_PER_TOKEN = 2.0
"""注入字符 → token 的粗折算（中文偏保守）。"""
_COVERAGE_PENALTY = 0.2
"""丢失一条成功轨迹用过的记忆对质量的惩罚（0..1 区间内扣减）。"""

_JUDGE_SYSTEM = (
    "你是 Agent 自我改进的回放评审员。基于给出的【证据】（真实轨迹"
    "统计与失败模式）评估【候选变更】的预期效果。只输出 JSON："
    "{\"quality_delta\": -1..1, \"reason\": \"一句话\"}。"
    "拿不准就给 0。禁止编造证据里没有的结论。"
)


@dataclass
class ReplayReport:
    """一次回放评估的明细（写入 dream 报告）。"""

    candidate_op: str
    objective: ObjectiveVector
    coverage: float = 1.0
    est_tokens_mean: float = 0.0
    judge_reason: str = ""
    judge_delta: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def _fragment_chars(genome: GenomeSpec) -> int:
    return sum(len(f.text) for f in genome.prompt_fragments if f.enabled)


def _first_text(resp: Any) -> str:
    for block in getattr(resp, "content", []):
        if getattr(block, "type", "") == "text" and block.text:
            return block.text
    return ""


class ReplayEvaluator:
    """在录制轨迹集上评估候选基因组。"""

    def __init__(
        self,
        config: RSIMemConfig,
        db: RSIMemDatabase,
        retriever: Retriever,
    ) -> None:
        self._config = config
        self._db = db
        self._retriever = retriever

    async def evaluate(
        self,
        candidate: GenomeSpec,
        parent: GenomeSpec,
        traces: list[TraceRecord],
        *,
        validation_passed: bool,
        judge_model: ChatModelBase | None = None,
        diagnosis_summary: dict[str, Any] | None = None,
    ) -> ReplayReport:
        """评估候选，返回三目标向量与明细。"""
        scored = [t for t in traces if t.outcome is not None]
        sample = scored[: self._config.rsi.regression_traces * 2]

        cand_ids_per_trace: list[tuple[set[str], set[str]]] = []
        est_tokens: list[float] = []
        for trace in sample:
            res_c = await self._retriever.search(
                trace.query,
                tiers=("long", "short"),
                top_k=candidate.retrieval.top_k,
                score_threshold=candidate.retrieval.score_threshold,
                char_budget=candidate.retrieval.char_budget,
            )
            res_p = await self._retriever.search(
                trace.query,
                tiers=("long", "short"),
                top_k=parent.retrieval.top_k,
                score_threshold=parent.retrieval.score_threshold,
                char_budget=parent.retrieval.char_budget,
            )
            cand_ids_per_trace.append(
                ({r.record.id for r in res_c},
                 {r.record.id for r in res_p}),
            )
            chars_c = sum(
                len(r.record.display()) for r in res_c
            ) + _fragment_chars(candidate)
            chars_p = sum(
                len(r.record.display()) for r in res_p
            ) + _fragment_chars(parent)
            est_tokens.append(
                max(0.0, trace.total_tokens + (chars_c - chars_p)
                    / _CHARS_PER_TOKEN),
            )

        # ---- 回归覆盖：成功轨迹用过的、父代当前能召回的记忆，
        # ---- 候选参数下必须仍然召回（回归保障）。
        coverage_penalty = 0.0
        checks = 0
        for trace, (cand_ids, parent_ids) in zip(sample, cand_ids_per_trace):
            used = {
                mid for mid in trace.used_memory_ids
                if mid in parent_ids  # 父代当前能召回的才纳入检查
            }
            used = {
                mid for mid in used
                if (rec := await self._db.get_memory(mid)) is not None
                and rec.tier in ("long", "short")
                and rec.status in ("active", "promoted")
            }
            if not used:
                continue
            checks += 1
            lost = len(used - cand_ids) / len(used)
            coverage_penalty += _COVERAGE_PENALTY * lost
        coverage = (
            1.0 - coverage_penalty / checks if checks else 1.0
        )

        base_quality = (
            sum(t.outcome.score for t in scored) / len(scored)
            if scored else 0.5
        )

        judge_delta, judge_reason = 0.0, ""
        model = judge_model or self._config.backend_model
        if model is not None and candidate.operator != "seed":
            judge_delta, judge_reason = await self._judge(
                model, candidate, parent, diagnosis_summary or {},
            )

        quality = max(
            0.0,
            min(
                1.0,
                base_quality * coverage
                + 0.15 * max(-1.0, min(1.0, judge_delta)),
            ),
        )
        mean_tokens = (
            sum(est_tokens) / len(est_tokens) if est_tokens else 0.0
        )
        denied = sum(
            1 for t in traces for p in t.permissions
            if p.behavior not in ("allow", "ALLOW")
        )
        perm_total = sum(len(t.permissions) for t in traces)

        objective = ObjectiveVector(
            quality=round(quality, 4),
            cost_eff=round(cost_efficiency(mean_tokens), 4),
            safety=round(
                safety_from_signals(
                    permission_denied=denied,
                    permission_total=perm_total,
                    validation_passed=validation_passed,
                ),
                4,
            ),
            components={
                "base_quality": round(base_quality, 4),
                "coverage": round(coverage, 4),
                "judge_delta": round(judge_delta, 4),
                "est_tokens_mean": round(mean_tokens, 1),
            },
        )
        return ReplayReport(
            candidate_op=candidate.operator,
            objective=objective,
            coverage=round(coverage, 4),
            est_tokens_mean=round(mean_tokens, 1),
            judge_delta=round(judge_delta, 4),
            judge_reason=judge_reason,
            details={"traces_evaluated": len(sample)},
        )

    async def _judge(
        self,
        model: ChatModelBase,
        candidate: GenomeSpec,
        parent: GenomeSpec,
        diagnosis: dict[str, Any],
    ) -> tuple[float, str]:
        """LLM 评审：基于证据的预期质量增量。"""
        evidence = {
            "diagnosis": diagnosis,
            "parent": {
                "retrieval": parent.retrieval.model_dump(),
                "fragments": [
                    f.text[:120] for f in parent.prompt_fragments
                ],
                "active_sops": len(parent.active_sops),
            },
            "candidate": {
                "operator": candidate.operator,
                "retrieval": candidate.retrieval.model_dump(),
                "fragment_changes": [
                    f.text[:160] for f in candidate.prompt_fragments
                    if f.origin == "rsi"
                ],
                "active_sops": len(candidate.active_sops),
            },
        }
        prompt = (
            f"【证据】\n{json.dumps(evidence, ensure_ascii=False)}"
            f"\n\n【候选变更】operator={candidate.operator}，"
            f"说明={candidate.notes}"
        )
        try:
            resp = await model(
                messages=[
                    Msg(name="system", role="system",
                        content=[TextBlock(type="text",
                                           text=_JUDGE_SYSTEM)]),
                    Msg(name="user", role="user",
                        content=[TextBlock(type="text", text=prompt)]),
                ],
            )
            parsed = json_repair.loads(_first_text(resp))
            if isinstance(parsed, dict) and "quality_delta" in parsed:
                delta = float(parsed["quality_delta"])
                return (
                    max(-1.0, min(1.0, delta)),
                    str(parsed.get("reason", ""))[:200],
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("replay judge failed: %s", e)
        return 0.0, ""
