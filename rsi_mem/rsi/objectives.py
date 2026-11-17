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


def tprime(
    v: ObjectiveVector,
    weights: list[float],
    ideal: IdealPoint,
    uncertainty: float = 0.0,
    uncertainty_lambda: float = 0.0,
) -> float:
    """U2 统一子问题度量（越小越好）：t′ = tchebycheff + λu·u。

    即 Eq.7 ``J′ = J − λu·u``（J = −tchebycheff，奖励域越大越好）转
    写到本库的距离域——距离越小越好，惩罚项必须**加**：u 越大 t′ 越
    大越差（施工单原文的减号写在奖励域；若按字面在距离域做减法，
    高不确定候选反而被奖励，「λu>0 压过 v0」的验收语义不可能成立）。

    同一本账三处共用，无口径分歧：λ₀ 质量门、per-λ 邻域替换比较、
    ``v0_score``/``winner_score`` 报告键。``uncertainty_lambda=0`` 时
    退化为 :func:`tchebycheff`；v0 与无预测条目由调用方传
    ``uncertainty=0``（已验证真实值不惩罚）。
    """
    t = tchebycheff(v, weights, ideal)
    return t + uncertainty_lambda * max(0.0, uncertainty)


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


def pareto_frontier_auc(
    points: list[tuple[float, float]], budget: float,
) -> float:
    """Eq.1 Pareto 前沿下面积（归一 [0,1]，越大越好；U3-3b）。

    ``points``：归一化质量-成本点集 ``(n, a)``（n=est tokens 越小越
    好，a=质量/池内最高分 越大越好）；``budget``：n 的归一化基准 B
    （该轨迹 ρ=1 满预算点的 est tokens）。

    非支配过滤（n 小且 a 大者胜，被支配点剔除）→ n 升序 →
    ``AUC = Σ_j (ā_j − ā_{j−1})·(B − n̄_j)/B``（j 自第二个前沿点
    起，步进增量 × 剩余预算比例）。单点/空集返回 0（无步进可言）。
    """
    if len(points) < 2 or budget <= 0.0:
        return 0.0
    front = [
        (n, a) for n, a in points
        if not any(
            (n2 <= n and a2 > a) or (n2 < n and a2 >= a)
            for n2, a2 in points if (n2, a2) != (n, a)
        )
    ]
    if len(front) < 2:
        return 0.0
    front.sort(key=lambda p: p[0])  # n 升序（前沿上 a 随 n 不减）
    auc = 0.0
    for j in range(1, len(front)):
        step = front[j][1] - front[j - 1][1]
        if step <= 0.0:
            continue
        auc += step * max(0.0, (budget - front[j][0]) / budget)
    return max(0.0, min(1.0, auc))


def parallel_penalty(k_over_n: list[float], lam: float) -> float:
    """并行度惩罚（U3-3b）：``lam·mean(k*/n)``。

    ``k*/n``：串行轮数 / 总调用数（大 = 偏串行 = 罚）；``lam=0``
    关闭（默认）。空列表返回 0。
    """
    if not k_over_n or lam <= 0.0:
        return 0.0
    return lam * (sum(k_over_n) / len(k_over_n))
