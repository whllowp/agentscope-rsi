# -*- coding: utf-8 -*-
"""多目标与 Tchebycheff 聚合（Harmony Harness 的 MOEA/D 思路）。

三个目标（全部归一化到 [0,1]，越大越好）：

- ``quality``：轨迹结局评分 + 回放回归惩罚 + LLM 评审增量；
- ``cost_eff``：成本效率 —— 注入预算与调用次数的确定估计；
- ``safety``：安全 —— 权限拒绝率与校验通过情况。

每个权重向量 λ 定义一个"子问题"；Tchebycheff 聚合取
``max_i w_i·|z*_i − f_i|``（越小越好），理想点 z* 动态维护。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObjectiveVector:
    """一次评估的三目标向量。"""

    quality: float = 0.5
    cost_eff: float = 0.5
    safety: float = 1.0
    components: dict[str, float] = field(default_factory=dict)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.quality, self.cost_eff, self.safety)


@dataclass
class IdealPoint:
    """动态理想点 z*（历史最优，单调度不减）。"""

    quality: float = 0.0
    cost_eff: float = 0.0
    safety: float = 0.0

    def update(self, v: ObjectiveVector) -> None:
        self.quality = max(self.quality, v.quality)
        self.cost_eff = max(self.cost_eff, v.cost_eff)
        self.safety = max(self.safety, v.safety)


def tchebycheff(
    v: ObjectiveVector,
    weights: list[float],
    ideal: IdealPoint,
) -> float:
    """Tchebycheff 标量化（越小越好）。

    Args:
        v: 目标向量。
        weights: [quality, cost, safety] 权重（和不必为 1）。
        ideal: 理想点。
    """
    f = v.as_tuple()
    z = (ideal.quality, ideal.cost_eff, ideal.safety)
    return max(
        w * abs(zi - fi) for w, zi, fi in zip(weights, z, f)
    )


def safety_from_signals(
    *,
    permission_denied: int,
    permission_total: int,
    validation_passed: bool,
) -> float:
    """安全目标：校验通过为前提，权限拒绝率拉低得分。"""
    if not validation_passed:
        return 0.0
    if permission_total == 0:
        return 1.0
    return max(0.0, 1.0 - permission_denied / permission_total)


def cost_efficiency(mean_est_tokens: float) -> float:
    """成本目标：估计平均每轨迹 token 越少效率越高。

    以 2000 token 为"半效率"锚点（``1/(1+t/2000)``）。
    """
    return 1.0 / (1.0 + max(0.0, mean_est_tokens) / 2000.0)
