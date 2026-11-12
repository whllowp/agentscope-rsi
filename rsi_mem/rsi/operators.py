# -*- coding: utf-8 -*-
"""演化算子（Harmony Harness：tool/skill/logic 组件 × gen/env 作用域）。

在声明式基因组上的四个算子：

- ``skill_evolve``：新增/修订 SOP（写入肌肉记忆 ``candidate`` 条目并
  挂到候选基因组的 ``active_sops``，部署获胜才转 active）；
- ``logic_evolve``：新增/修订提示词片段；
- ``retrieval_evolve``：调整检索参数（top_k / 阈值 / 预算）；
- ``generalize``：作用域升降级（会话 SOP → 全局，或反向收缩）。

算子是**纯结构变换**：输入当前基因组 + 变更计划，输出候选基因组；
LLM 只负责在 improver 里产出"计划"，不直接改对象。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..memory.tiers import TierManager
from ..storage import GenomeSpec, MemoryRecord, PromptFragment

OperatorName = Literal[
    "skill_evolve", "logic_evolve", "retrieval_evolve", "generalize",
]


@dataclass
class OperatorPlan:
    """一条来自诊断/假设的变更计划。"""

    operator: OperatorName
    rationale: str = ""
    # skill_evolve
    sop_title: str = ""
    sop_content: str = ""
    sop_triggers: list[str] = field(default_factory=list)
    sop_replaces_id: str | None = None
    # logic_evolve
    fragment_id: str | None = None
    fragment_text: str = ""
    fragment_triggers: list[str] = field(default_factory=list)
    # retrieval_evolve
    retrieval_changes: dict[str, Any] = field(default_factory=dict)
    # generalize
    sop_promote_id: str | None = None

    def describe(self) -> str:
        if self.operator == "skill_evolve":
            return f"skill_evolve: {self.sop_title[:60]}"
        if self.operator == "logic_evolve":
            return (
                f"logic_evolve: {'改' if self.fragment_id else '增'}片段"
                f" {self.fragment_text[:40]}"
            )
        if self.operator == "retrieval_evolve":
            return f"retrieval_evolve: {self.retrieval_changes}"
        return f"generalize: {self.sop_promote_id}"


@dataclass
class MutationOutcome:
    """一次算子应用的结果（含需要部署时才生效的 SOP 候选）。"""

    genome: GenomeSpec
    sop_candidates: list[MemoryRecord] = field(default_factory=list)
    sop_discards: list[str] = field(default_factory=list)


async def apply_plan(
    current: GenomeSpec,
    plan: OperatorPlan,
    tiers: TierManager,
) -> MutationOutcome | None:
    """应用一条计划，返回变异结果；结构非法时返回 ``None``。"""
    if plan.operator == "skill_evolve":
        return await _skill_evolve(current, plan, tiers)
    if plan.operator == "logic_evolve":
        return _logic_evolve(current, plan)
    if plan.operator == "retrieval_evolve":
        return _retrieval_evolve(current, plan)
    if plan.operator == "generalize":
        return await _generalize(current, plan, tiers)
    return None


# ----------------------------------------------------------------------
# skill_evolve
# ----------------------------------------------------------------------
async def _skill_evolve(
    current: GenomeSpec,
    plan: OperatorPlan,
    tiers: TierManager,
) -> MutationOutcome | None:
    if not plan.sop_title or not plan.sop_content:
        return None
    sop = await tiers.muscle.note(
        title=plan.sop_title,
        content=plan.sop_content,
        triggers=plan.sop_triggers or [plan.sop_title],
        kind="sop",
        confidence=0.7,
        origin="rsi",
        status="candidate",
        source={"plan": plan.rationale[:200]},
    )
    sops = [
        s for s in current.active_sops
        if s != plan.sop_replaces_id
    ]
    sops.insert(0, sop.id)
    genome = current.model_copy(deep=True)
    genome.active_sops = sops[:20]
    genome.operator = "skill_evolve"
    genome.notes = plan.rationale[:300]
    return MutationOutcome(genome=genome, sop_candidates=[sop])


# ----------------------------------------------------------------------
# logic_evolve
# ----------------------------------------------------------------------
def _logic_evolve(
    current: GenomeSpec,
    plan: OperatorPlan,
) -> MutationOutcome | None:
    if not plan.fragment_text.strip():
        return None
    genome = current.model_copy(deep=True)
    if plan.fragment_id:
        frag = next(
            (f for f in genome.prompt_fragments if f.id == plan.fragment_id),
            None,
        )
        if frag is None or frag.locked:
            if frag is not None and frag.locked:
                return None  # 锁定片段不可改
            return None
        frag.text = plan.fragment_text
        frag.origin = "rsi"
    else:
        genome.prompt_fragments.append(PromptFragment(
            slot="reflex" if plan.fragment_triggers else "always",
            triggers=plan.fragment_triggers,
            text=plan.fragment_text,
            origin="rsi",
        ))
    if len(genome.prompt_fragments) > 12:
        return None
    genome.operator = "logic_evolve"
    genome.notes = plan.rationale[:300]
    return MutationOutcome(genome=genome)


# ----------------------------------------------------------------------
# retrieval_evolve
# ----------------------------------------------------------------------
def _retrieval_evolve(
    current: GenomeSpec,
    plan: OperatorPlan,
) -> MutationOutcome | None:
    if not plan.retrieval_changes:
        return None
    genome = current.model_copy(deep=True)
    updates = genome.retrieval.model_dump()
    updates.update(plan.retrieval_changes)
    try:
        genome.retrieval = genome.retrieval.__class__.model_validate(
            updates,
        )
    except Exception:  # noqa: BLE001 — 参数越界直接拒绝该计划
        return None
    genome.operator = "retrieval_evolve"
    genome.notes = plan.rationale[:300]
    return MutationOutcome(genome=genome)


# ----------------------------------------------------------------------
# generalize（作用域升降级）
# ----------------------------------------------------------------------
async def _generalize(
    current: GenomeSpec,
    plan: OperatorPlan,
    tiers: TierManager,
) -> MutationOutcome | None:
    if not plan.sop_promote_id:
        return None
    db = tiers.retriever._db
    rec = await db.get_memory(plan.sop_promote_id)
    if rec is None or rec.tier != "muscle":
        return None
    genome = current.model_copy(deep=True)
    if plan.sop_promote_id not in genome.active_sops:
        genome.active_sops.append(plan.sop_promote_id)
        genome.operator = "generalize"
        genome.notes = (
            f"提升 SOP {plan.sop_promote_id} 为全局反射: "
            f"{plan.rationale[:200]}"
        )
        return MutationOutcome(genome=genome)
    return None
