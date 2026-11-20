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
    pareto_frontier_auc,
    parallel_penalty,
    safety_from_signals,
)
from .trace import serial_rounds_k

_CHARS_PER_TOKEN = 2.0
"""注入字符 → token 的粗折算（中文偏保守）。"""
_COVERAGE_PENALTY = 0.2
"""丢失一条成功轨迹用过的记忆对质量的惩罚（0..1 区间内扣减）。"""
_SOP_COVERAGE_BONUS = 0.1
"""候选激活的 SOP 触发词命中有据可查的近期轨迹查询时，给质量的
确定性加成（E0：SOP-查询对齐度，RSI-Anything 的预测性收益信号）。"""

_JUDGE_SYSTEM = (
    "你是 Agent 自我改进的回放评审员。基于给出的【证据】（真实轨迹"
    "统计与失败模式）评估【候选变更】的预期效果。只输出 JSON："
    "{\"quality_delta\": -1..1, \"reason\": \"一句话\"}。"
    "拿不准就给 0。禁止编造证据里没有的结论。"
    "评审只依据【证据】中已揭示的**前缀**统计（前缀工具序列 / 工具"
    "错误分布 / 查询摘录 / (n,k*) 数据）；证据不提供任何结局分数，"
    "禁止引用或推断结局绝对分数（附录 D.2 前缀盲评约束）。"
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
    pareto_auc: float = 1.0
    """Eq.1 Pareto 前沿下面积（ρ 扫描，[0,1] 越大越好）。空样本默认
    1.0（无点集无从计算，保持旧测试口径）。"""
    points: list[dict[str, Any]] = field(default_factory=list)
    """ρ 扫描点集：每轨迹 {trace_id, points: [[n,a]×|ρ|], budget}。"""
    is_v0: bool = False
    """engine 对 v0 条目置 True（U3-3b 起回放报告显式标注基准位）。"""
    quality_e2: float = 0.5
    """E2 质量分量（U2 双门）：clamp(base_quality×coverage + 0.1×
    sop_coverage, 0, 1)——即 quality_floor 剥掉 0.15×judge_delta 项。
    仅 E1（LLM 评审）证据支持的质量优势不得据此越过 E2/E0 门。
    judge 缺席时 judge_delta≡0，quality_e2 == quality，双门退化单门
    （现有测试与 bench 行为不变）。"""


def _fragment_chars(genome: GenomeSpec) -> int:
    return sum(len(f.text) for f in genome.prompt_fragments if f.enabled)


_PREFIX_STEPS = 5
"""前缀视图里每条被引用轨迹揭示的工具步数上限。"""


def _prefix_view(
    diagnosis: dict[str, Any] | None,
    traces: list[TraceRecord] | None = None,
) -> dict[str, Any]:
    """judge 证据的前缀视图（U3-3a）：剥离结局分数，(n,k*) 数据前置。

    - 剥离 ``worst/best_traces[].score`` 与 ``stats.mean_outcome`` ——
      结局分数不进评审视野（防评审锚定绝对分数；诊断全量数据仍留给
      策略侧 ``hypothesize``，improver 不动）；
    - 保留 ``error_tools`` / ``ok_tools`` / ``query`` 摘录，并按轨迹
      id 匹配 ``traces`` 前置每条摘录的**前缀工具序列**（``step_index``
      序前 5 步 name+state，无任何分数）与 ``calls_n`` /
      ``serial_rounds_k``（(n,k*) 数据前置）。

    ``traces`` 为 None 时只做剥离（无前缀序列可挂）。返回浅拷贝，
    不改入参。
    """
    if not isinstance(diagnosis, dict):
        return {}
    out = dict(diagnosis)
    stats = out.get("stats")
    if isinstance(stats, dict):
        out["stats"] = {
            k: v for k, v in stats.items() if k != "mean_outcome"
        }
    by_id = {t.id: t for t in (traces or [])}
    for key in ("worst_traces", "best_traces"):
        rows = out.get(key)
        if not isinstance(rows, list):
            continue
        kept_rows = []
        for r in rows:
            if not isinstance(r, dict):
                kept_rows.append(r)
                continue
            row = {k: v for k, v in r.items() if k != "score"}
            trace = by_id.get(str(row.get("id") or ""))
            if trace is not None:
                steps = sorted(
                    trace.tool_calls,
                    key=lambda c: (
                        c.step_index if c.step_index is not None else 1 << 30,
                        c.at,
                    ),
                )[:_PREFIX_STEPS]
                row["prefix_tools"] = [
                    {
                        "step": (
                            c.step_index
                            if c.step_index is not None else i
                        ),
                        "name": c.name,
                        "state": c.state,
                    }
                    for i, c in enumerate(steps)
                ]
                row["calls_n"] = len(trace.tool_calls)
                row["serial_rounds_k"] = serial_rounds_k(trace.tool_calls)
            kept_rows.append(row)
        out[key] = kept_rows
    return out


def _lerp(a: float, b: float, t: float) -> float:
    """线性插值 a→b（t∈[0,1]）。"""
    return a + (b - a) * t


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
        """评估候选，返回三目标向量、ρ 扫描点集与 Pareto 前沿面积。

        U3-3b：检索参数在 ``RetrievalParams`` 边界内按 ρ∈``rho_grid``
        插值（top_k/char_budget 线性、threshold 不动），逐 (trace, ρ)
        得归一化质量-成本点 (n_iρ, a_iρ)；``pareto_auc`` = Eq.1 前沿
        面积均值 − 并行度惩罚（k*/n 由 ``at`` 间隔 >0.5s 聚类估计）。
        judge 只调一次（与 ρ 无关），证据为前缀视图（3a）。
        ``objective`` 单点产出保持：quality/cost 用 ρ=1 点（= 候选原
        参数），兼容既有分量与测试。
        """
        scored = [t for t in traces if t.outcome is not None]
        sample = scored[: self._config.rsi.regression_traces * 2]
        model = judge_model or self._config.backend_model

        # ---- judge 一次（与 ρ 无关：rho_grid 不放大评审调用）----
        judge_delta, judge_reason = 0.0, ""
        if model is not None and candidate.operator != "seed":
            judge_delta, judge_reason = await self._judge(
                model, candidate, parent, diagnosis_summary or {},
                traces=sample,
            )

        base_quality = (
            sum(t.outcome.score for t in scored) / len(scored)
            if scored else 0.5
        )
        # ---- SOP 触发覆盖加成（E0 确定性）：候选激活的 SOP 与近期
        # ---- 查询的对齐度越高，预期质量加成越大。
        sop_coverage = await self._sop_trigger_coverage(
            candidate, sample,
        )
        quality_floor = (
            _SOP_COVERAGE_BONUS * sop_coverage
            + 0.15 * max(-1.0, min(1.0, judge_delta))
        )

        # ---- per-(trace, ρ) 双层扫描（U3-3b）----
        rho_grid = list(self._config.rsi.rho_grid) or [1.0]
        full_rho = 1.0 if 1.0 in rho_grid else max(rho_grid)
        coverage_penalty: dict[float, float] = {r: 0.0 for r in rho_grid}
        checks = 0
        points: list[dict[str, Any]] = []
        est_tokens_full: list[float] = []
        aucs: list[float] = []
        n_effective_traces = 0
        k_over_n: list[float] = []
        # S = 池内最高 outcome.score（a 的归一分母；P10：S≤0 → a≡0）
        max_score = max(
            (t.outcome.score for t in scored), default=0.0,
        )
        for trace in sample:
            res_p = await self._retriever.search(
                trace.query,
                tiers=("long", "short"),
                top_k=parent.retrieval.top_k,
                score_threshold=parent.retrieval.score_threshold,
                char_budget=parent.retrieval.char_budget,
            )
            parent_ids = {r.record.id for r in res_p}
            chars_p = (
                sum(len(r.record.display()) for r in res_p)
                + _fragment_chars(parent)
            )
            # 回归覆盖锚点（父代可召回的才纳入检查；与 ρ 无关）
            used = {
                mid for mid in trace.used_memory_ids
                if mid in parent_ids
            }
            used = {
                mid for mid in used
                if (rec := await self._db.get_memory(mid)) is not None
                and rec.tier in ("long", "short")
                and rec.status in ("active", "promoted")
            }
            if used:
                checks += 1
            per_rho: list[tuple[float, float]] = []  # (n_iρ, a_iρ)
            per_rho_n: dict[float, float] = {}
            for rho in rho_grid:
                top_k = max(
                    1, round(_lerp(1, candidate.retrieval.top_k, rho)),
                )
                char_budget = max(
                    200,
                    round(_lerp(200, candidate.retrieval.char_budget, rho)),
                )
                res_c = await self._retriever.search(
                    trace.query,
                    tiers=("long", "short"),
                    top_k=top_k,
                    score_threshold=candidate.retrieval.score_threshold,
                    char_budget=char_budget,
                )
                cand_ids = {r.record.id for r in res_c}
                chars_c = (
                    sum(len(r.record.display()) for r in res_c)
                    + _fragment_chars(candidate)
                )
                per_rho_n[rho] = max(
                    0.0,
                    trace.total_tokens
                    + (chars_c - chars_p) / _CHARS_PER_TOKEN,
                )
                lost = (
                    len(used - cand_ids) / len(used) if used else 0.0
                )
                if used:
                    coverage_penalty[rho] += _COVERAGE_PENALTY * lost
                # 本点质量：coverage×base + SOP + judge 分量（ρ 只经
                # coverage 起作用；judge/SOP 与 ρ 无关）
                q_i_rho = max(
                    0.0,
                    min(
                        1.0,
                        base_quality * (1.0 - _COVERAGE_PENALTY * lost)
                        + quality_floor,
                    ),
                )
                a_i_rho = (
                    max(0.0, min(1.0, q_i_rho / max_score))
                    if max_score > 0.0 else 0.0
                )
                per_rho.append((per_rho_n[rho], a_i_rho))
            # B_i = ρ=1（候选满预算）点的 est tokens
            budget_i = per_rho_n.get(full_rho, 0.0)
            est_tokens_full.append(budget_i)
            # U3 n_effective：该轨迹 a 有区分度的 ρ 点数（不同 a 值数）。
            # 全平（n_effective<2）的轨迹前沿面积恒 0——计入均值会把
            # 信号稀释成恒定低值（实测 5 轮 bench 全轮 AUC=0，严格递减
            # 哨兵永盲），故 AUC 均值只含有区分度的轨迹
            n_effective = len({round(a, 6) for _, a in per_rho})
            if n_effective >= 2:
                aucs.append(pareto_frontier_auc(per_rho, budget_i))
                n_effective_traces += 1
            points.append({
                "trace_id": trace.id[:8],
                "points": [
                    [round(n, 1), round(a, 4)] for n, a in per_rho
                ],
                "budget": round(budget_i, 1),
                "n_effective": n_effective,
            })
            if trace.tool_calls:
                k_over_n.append(
                    serial_rounds_k(trace.tool_calls)
                    / len(trace.tool_calls),
                )

        # ---- 覆盖与单点三目标（ρ=1 全局口径，兼容旧语义）----
        coverage = (
            1.0 - coverage_penalty.get(full_rho, 0.0) / checks
            if checks else 1.0
        )
        quality = max(
            0.0,
            min(
                1.0,
                base_quality * coverage + quality_floor,
            ),
        )
        # U2 双门的 E2 质量分量：剥 judge_delta（E1）项，只留确定性
        # E2/E0 分量（base×coverage + SOP 触发覆盖加成）
        quality_e2 = max(
            0.0,
            min(
                1.0,
                base_quality * coverage
                + _SOP_COVERAGE_BONUS * sop_coverage,
            ),
        )
        mean_tokens = (
            sum(est_tokens_full) / len(est_tokens_full)
            if est_tokens_full else 0.0
        )
        denied = sum(
            1 for t in traces for p in t.permissions
            if p.behavior not in ("allow", "ALLOW")
        )
        perm_total = sum(len(t.permissions) for t in traces)

        # ---- Pareto 前沿面积（口径）----
        # 有样本但零区分度（全被 n_effective 过滤）→ 0.0：前沿无从
        # 谈起且不能伪装成"满分"；空样本（无轨迹）保持默认 1.0
        # （旧测试口径：无点集无从计算）
        if aucs:
            pareto_auc = max(
                0.0,
                min(
                    1.0,
                    sum(aucs) / len(aucs)
                    - parallel_penalty(
                        k_over_n, self._config.rsi.parallel_lambda,
                    ),
                ),
            )
        elif sample:
            pareto_auc = 0.0
        else:
            pareto_auc = 1.0

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
                "sop_coverage": round(sop_coverage, 4),
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
            quality_e2=round(quality_e2, 4),
            judge_reason=judge_reason,
            details={
                "traces_evaluated": len(sample),
                "rho_grid": list(rho_grid),
                # U3：AUC 均值口径里的有效轨迹数（n_effective≥2）
                "n_effective_traces": n_effective_traces,
            },
            pareto_auc=round(pareto_auc, 4),
            points=points,
        )

    async def _sop_trigger_coverage(
        self,
        candidate: GenomeSpec,
        traces: list[TraceRecord],
    ) -> float:
        """候选激活 SOP 的触发词对近期轨迹查询的覆盖率（0..1）。

        确定性计算（E0）：加载 ``active_sops`` 对应的肌肉记忆条目，
        统计查询被任一触发词（子串语义）命中的轨迹占比。SOP 候选
        以 ``status=candidate`` 存库，同样参与计算。
        """
        if not candidate.active_sops or not traces:
            return 0.0
        sops = []
        for mid in candidate.active_sops:
            rec = await self._db.get_memory(mid)
            if (
                rec is not None
                and rec.tier == "muscle"
                and rec.status in ("active", "candidate", "promoted")
            ):
                sops.append(rec)
        if not sops:
            return 0.0
        hit = 0
        for t in traces:
            q = t.query.casefold()
            if any(
                tr.casefold() in q
                for s in sops for tr in s.triggers
            ):
                hit += 1
        return hit / len(traces)

    async def _judge(
        self,
        model: ChatModelBase,
        candidate: GenomeSpec,
        parent: GenomeSpec,
        diagnosis: dict[str, Any],
        *,
        traces: list[TraceRecord] | None = None,
    ) -> tuple[float, str]:
        """LLM 评审：基于**前缀**证据的预期质量增量（U3-3a）。

        证据经 :func:`_prefix_view` 剥离结局分数并前置 (n,k*) 数据
        ——评审只能看前缀统计，不能锚定绝对分数。v0 仅首轮
        （operator=='seed'）免评，第二轮起 v0 同被 judge：公平性按
        「全部条目同视野同评审」论证。
        """
        evidence = {
            "diagnosis": _prefix_view(diagnosis, traces),
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
