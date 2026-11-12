# -*- coding: utf-8 -*-
"""结果预测器：RSI-Anything 世界模型（W1）的轻量类比。

论文把 RSI 建模在"策略-世界模型对"上：候选动作执行**之前**，由世界
模型预测其结果；预测-观测对不断积累，既指导候选选择，也用预测误差
校准世界模型自身。本模块是其声明式基因组版：

- **特征**：候选 vs 父代的结构差（算子类型、检索参数差、片段/SOP
  增量）—— 确定性、可复现（E1 证据）；
- **预测**：三目标增量向量 + 不确定性（样本不足时高不确定）；
- **在线校准**：部署过的候选在下一轮 dream 拿到真实回放结果时回填
  ``realized``，按算子维护在线均值与 MAE；
- **可靠性门（retention gate）**：样本足够且 MAE 超阈 → 引擎进入
  保守模式（只放行 retrieval_evolve 小步候选），直至重新校准。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agentscope._logging import logger

from ..storage import GenomeSpec, RSIMemDatabase
from .objectives import ObjectiveVector

_CONSERVATIVE_OPERATORS = ("retrieval_evolve",)


@dataclass
class CandidateFeatures:
    """候选基因组相对父代的确定性结构特征。"""

    operator: str = "seed"
    top_k_delta: int = 0
    threshold_delta: float = 0.0
    char_budget_delta: int = 0
    frag_chars_delta: int = 0
    sop_count_delta: int = 0

    def as_dict(self) -> dict:
        return {
            "operator": self.operator,
            "top_k_delta": self.top_k_delta,
            "threshold_delta": self.threshold_delta,
            "char_budget_delta": self.char_budget_delta,
            "frag_chars_delta": self.frag_chars_delta,
            "sop_count_delta": self.sop_count_delta,
        }

    @classmethod
    def extract(
        cls, candidate: GenomeSpec, parent: GenomeSpec,
    ) -> "CandidateFeatures":
        frag = sum(
            len(f.text) for f in candidate.prompt_fragments if f.enabled
        ) - sum(
            len(f.text) for f in parent.prompt_fragments if f.enabled
        )
        return cls(
            operator=candidate.operator,
            top_k_delta=(
                candidate.retrieval.top_k - parent.retrieval.top_k
            ),
            threshold_delta=(
                candidate.retrieval.score_threshold
                - parent.retrieval.score_threshold
            ),
            char_budget_delta=(
                candidate.retrieval.char_budget
                - parent.retrieval.char_budget
            ),
            frag_chars_delta=frag,
            sop_count_delta=(
                len(candidate.active_sops) - len(parent.active_sops)
            ),
        )


@dataclass
class Prediction:
    """一次预测输出（预测的三目标 + 不确定性）。"""

    quality_delta: float
    cost_eff_delta: float
    safety_delta: float
    uncertainty: float
    samples: int = 0

    def as_dict(self) -> dict:
        return {
            "quality_delta": round(self.quality_delta, 4),
            "cost_eff_delta": round(self.cost_eff_delta, 4),
            "safety_delta": round(self.safety_delta, 4),
            "uncertainty": round(self.uncertainty, 4),
            "samples": self.samples,
        }


@dataclass
class OperatorStats:
    """按算子的在线统计（预测增量 vs 实测增量的均值）。"""

    n: int = 0
    abs_err_sum: float = 0.0
    quality_delta_sum: float = 0.0
    cost_eff_delta_sum: float = 0.0
    safety_delta_sum: float = 0.0

    @property
    def mae(self) -> float | None:
        return self.abs_err_sum / self.n if self.n else None

    def deltas(self) -> tuple[float, float, float]:
        if not self.n:
            return 0.0, 0.0, 0.0
        return (
            self.quality_delta_sum / self.n,
            self.cost_eff_delta_sum / self.n,
            self.safety_delta_sum / self.n,
        )


# 先验：无样本时按算子给中性偏保守的预期（可被实测快速覆盖）
_PRIOR: dict[str, tuple[float, float, float]] = {
    "skill_evolve": (0.05, -0.01, 0.0),
    "logic_evolve": (0.02, -0.02, 0.0),
    "retrieval_evolve": (0.0, 0.03, 0.0),
    "generalize": (0.03, 0.0, 0.0),
    "seed": (0.0, 0.0, 0.0),
}


class OutcomePredictor:
    """候选基因组 → 预期三目标增量（在线校准）。"""

    def __init__(self, db: RSIMemDatabase) -> None:
        self._db = db
        self._stats: dict[str, OperatorStats] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    async def _ensure_loaded(self) -> None:
        """从 predictions 表恢复按算子统计（幂等）。"""
        if self._loaded:
            return
        try:
            rows = await self._db.load_predictions(realized_only=True)
        except Exception:  # noqa: BLE001 — 兼容空库
            rows = []
        for row in rows:
            stats = self._stats.setdefault(
                row["operator"], OperatorStats(),
            )
            pred = row["predicted"]
            real = row["realized"]
            stats.n += 1
            stats.abs_err_sum += abs(
                pred.get("quality_delta", 0.0)
                - real.get("quality_delta", 0.0),
            )
            stats.quality_delta_sum += real.get("quality_delta", 0.0)
            stats.cost_eff_delta_sum += real.get("cost_eff_delta", 0.0)
            stats.safety_delta_sum += real.get("safety_delta", 0.0)
        self._loaded = True

    def predict(self, features: CandidateFeatures) -> Prediction:
        """预测候选的三目标增量（同步、纯函数式）。"""
        stats = self._stats.get(features.operator)
        prior = _PRIOR.get(features.operator, (0.0, 0.0, 0.0))
        if stats and stats.n:
            qd, cd, sd = stats.deltas()
            uncertainty = 1.0 / (1.0 + stats.n)
        else:
            qd, cd, sd = prior
            uncertainty = 0.8
        # 结构性直觉（可与在线统计叠加的小修正）：
        # 收紧注入 → 成本效率上升；扩大 → 下降
        cd += -features.char_budget_delta / 20000.0
        cd += -features.top_k_delta * 0.01
        return Prediction(
            quality_delta=qd,
            cost_eff_delta=cd,
            safety_delta=sd,
            uncertainty=uncertainty,
            samples=stats.n if stats else 0,
        )

    def update(
        self,
        features: CandidateFeatures,
        realized: ObjectiveVector,
        parent_objective: ObjectiveVector,
    ) -> float:
        """回填实测结果，返回该样本的绝对预测误差。"""
        real = (
            realized.quality - parent_objective.quality,
            realized.cost_eff - parent_objective.cost_eff,
            realized.safety - parent_objective.safety,
        )
        pred = self.predict(features)
        err = abs(pred.quality_delta - real[0])
        stats = self._stats.setdefault(
            features.operator, OperatorStats(),
        )
        stats.n += 1
        stats.abs_err_sum += err
        stats.quality_delta_sum += real[0]
        stats.cost_eff_delta_sum += real[1]
        stats.safety_delta_sum += real[2]
        return err

    # ------------------------------------------------------------------
    # 可靠性门（RSI-Anything retention gate）
    # ------------------------------------------------------------------
    def mae(self) -> float | None:
        total_n = sum(s.n for s in self._stats.values())
        if not total_n:
            return None
        return (
            sum(s.abs_err_sum for s in self._stats.values()) / total_n
        )

    def sample_count(self) -> int:
        return sum(s.n for s in self._stats.values())

    def reliable(
        self, *, min_samples: int, mae_threshold: float,
    ) -> bool:
        """样本不足 → 视为"未证实不可靠"（保持探索，论文语义）；
        样本足够且 MAE 超阈 → 不可靠 → 引擎转保守模式。"""
        mae = self.mae()
        if mae is None or self.sample_count() < min_samples:
            return True
        return mae <= mae_threshold

    def stats_summary(self) -> dict:
        overall = self.mae()
        return {
            "samples": self.sample_count(),
            "mae": round(overall, 4) if overall is not None else None,
            "per_operator": {
                op: {"n": s.n, "mae": round(s.mae, 4) if s.mae
                     is not None else None}
                for op, s in self._stats.items()
            },
        }


def is_conservative_operator(operator: str) -> bool:
    """保守模式下唯一放行的算子（小步、可回滚、无新增内容）。"""
    return operator in _CONSERVATIVE_OPERATORS


async def record_prediction(
    db: RSIMemDatabase,
    *,
    genome_version: int | None,
    operator: str,
    features: CandidateFeatures,
    prediction: Prediction,
) -> str:
    """把一次预测写入 predictions 表（realized 置空待回填）。"""
    return await db.insert_prediction(
        operator=operator,
        genome_version=genome_version,
        features=features.as_dict(),
        predicted=prediction.as_dict(),
    )


async def backfill_realized(
    db: RSIMemDatabase,
    *,
    prediction_id: str,
    realized: dict,
) -> None:
    await db.update_prediction_realized(prediction_id, realized)


def realized_dict(
    realized: ObjectiveVector, parent: ObjectiveVector,
) -> dict:
    return {
        "quality_delta": round(realized.quality - parent.quality, 4),
        "cost_eff_delta": round(realized.cost_eff - parent.cost_eff, 4),
        "safety_delta": round(realized.safety - parent.safety, 4),
    }
