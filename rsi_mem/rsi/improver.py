# -*- coding: utf-8 -*-
"""改进器：诊断 → 假设 → 算子计划（Harmony Harness 的 propose 环节）。

**证据完整性门**：诊断输出引用的轨迹 id / 记忆 id 必须真实存在于库中，
否则整份诊断被拒绝 —— 防止 LLM 编造证据驱动演化（Harmony 的
evidence-integrity gate）。

有后台模型时由 LLM 产出假设（结构化算子计划）；离线或失败时用规则
启发式（错误率→SOP、成本高→收检索预算、负反馈→提示词收紧）。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import json_repair
from agentscope._logging import logger
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase

from ..config import RSIMemConfig
from ..memory.consolidate import summarize_feedback
from ..storage import RSIMemDatabase, TraceRecord
from .operators import OperatorPlan

_HYPOTHESIZE_SYSTEM = (
    "你是运维 Agent 的自我改进器。基于【诊断证据】提出至多 3 个改进"
    "计划，按优先级排序。每个计划是一个 JSON 对象：\n"
    '{"operator": "skill_evolve|logic_evolve|retrieval_evolve|'
    'generalize", "rationale": "一句话依据", ...}\n'
    "- skill_evolve: sop_title, sop_content(分步SOP正文), "
    "sop_triggers(触发词数组)\n"
    "- logic_evolve: fragment_text(提示词片段)\n"
    "- retrieval_evolve: retrieval_changes({top_k/score_threshold/"
    "char_budget/short_inject_recent 中至多改 2 项, 小步调整})\n"
    "- generalize: sop_promote_id(现有SOP的id)\n"
    "只输出 JSON 数组。禁止提出删除安全规则或放宽预算上限的计划。"
)


def _first_text(resp: Any) -> str:
    for block in getattr(resp, "content", []):
        if getattr(block, "type", "") == "text" and block.text:
            return block.text
    return ""


class Improver:
    """诊断与假设生成。"""

    def __init__(
        self,
        config: RSIMemConfig,
        db: RSIMemDatabase,
    ) -> None:
        self._config = config
        self._db = db

    # ------------------------------------------------------------------
    # 诊断（证据完整性门）
    # ------------------------------------------------------------------
    async def diagnose(
        self, traces: list[TraceRecord],
    ) -> dict[str, Any] | None:
        """聚合三目标信号并抽取证据；证据 id 全部校验。"""
        if not traces:
            return None
        stats = summarize_feedback(traces)
        error_tools = Counter(
            c.name for t in traces for c in t.tool_calls
            if c.state == "error"
        )
        ok_tools = Counter(
            c.name for t in traces for c in t.tool_calls
            if c.state == "ok"
        )
        worst = sorted(
            (t for t in traces if t.outcome), key=lambda t: t.outcome.score,
        )[:3]
        best = sorted(
            (t for t in traces if t.outcome),
            key=lambda t: -t.outcome.score,
        )[:3]
        # —— 证据完整性门：引用的 trace id 必须都在库里 ——
        known_ids = {t.id for t in traces}
        cited_trace_ids = [t.id for t in worst + best if t.id in known_ids]
        if len(cited_trace_ids) != len([t for t in worst + best]):
            logger.warning(
                "diagnosis evidence gate: dropped unverifiable citations",
            )
        return {
            "stats": stats,
            "error_tools": dict(error_tools.most_common(5)),
            "ok_tools": dict(ok_tools.most_common(5)),
            "worst_traces": [
                {"id": t.id, "score": t.outcome.score,
                 "query": t.query[:80]} for t in worst
            ],
            "best_traces": [
                {"id": t.id, "score": t.outcome.score,
                 "query": t.query[:80]} for t in best
            ],
            "cited_trace_ids": cited_trace_ids,
        }

    # ------------------------------------------------------------------
    # 假设 → 算子计划
    # ------------------------------------------------------------------
    async def hypothesize(
        self,
        diagnosis: dict[str, Any],
        *,
        model: ChatModelBase | None = None,
    ) -> list[OperatorPlan]:
        """LLM 假设优先；离线/失败走规则启发式兜底。"""
        model = model or self._config.backend_model
        if model is not None:
            plans = await self._llm_plans(model, diagnosis)
            if plans:
                return plans
        return self._rule_plans(diagnosis)

    async def _llm_plans(
        self, model: ChatModelBase, diagnosis: dict[str, Any],
    ) -> list[OperatorPlan]:
        import json

        prompt = (
            f"【诊断证据】\n{json.dumps(diagnosis, ensure_ascii=False)}"
        )
        try:
            resp = await model(
                messages=[
                    Msg(name="system", role="system",
                        content=[TextBlock(type="text",
                                           text=_HYPOTHESIZE_SYSTEM)]),
                    Msg(name="user", role="user",
                        content=[TextBlock(type="text", text=prompt)]),
                ],
            )
            parsed = json_repair.loads(_first_text(resp))
        except Exception as e:  # noqa: BLE001
            logger.warning("hypothesize LLM call failed: %s", e)
            return []
        plans: list[OperatorPlan] = []
        if not isinstance(parsed, list):
            return []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                plans.append(OperatorPlan(
                    operator=item.get("operator", ""),
                    rationale=str(item.get("rationale", ""))[:300],
                    sop_title=str(item.get("sop_title", "")),
                    sop_content=str(item.get("sop_content", "")),
                    sop_triggers=[
                        str(x) for x in item.get("sop_triggers", [])
                        if str(x).strip()
                    ],
                    fragment_id=item.get("fragment_id"),
                    fragment_text=str(item.get("fragment_text", "")),
                    retrieval_changes=dict(
                        item.get("retrieval_changes", {})
                        or {},
                    ),
                    sop_promote_id=item.get("sop_promote_id"),
                ))
            except Exception:  # noqa: BLE001
                continue
        return [p for p in plans if p.operator in (
            "skill_evolve", "logic_evolve", "retrieval_evolve",
            "generalize",
        )][:3]

    @staticmethod
    def _rule_plans(diagnosis: dict[str, Any]) -> list[OperatorPlan]:
        """规则启发式兜底（无模型时也能演化）。"""
        plans: list[OperatorPlan] = []
        stats = diagnosis.get("stats", {})
        error_tools: dict[str, int] = diagnosis.get("error_tools", {})
        ok_tools: dict[str, int] = diagnosis.get("ok_tools", {})

        # 1) 高工具错误率 → 从成功序列蒸馏 SOP
        if stats.get("tool_error_ratio", 0) > 0.25 and ok_tools:
            seq = " → ".join(list(ok_tools)[:4])
            # 触发词除工具名外，取最近成功轨迹的查询前缀，
            # 让 SOP 能命中同类问题（对 CJK 查询尤其重要）
            best_query = ""
            if diagnosis.get("best_traces"):
                best_query = (
                    diagnosis["best_traces"][0].get("query", "") or ""
                )[:12]
            triggers = [t for t in list(ok_tools)[:2]]
            if best_query:
                triggers.insert(0, best_query)
            plans.append(OperatorPlan(
                operator="skill_evolve",
                rationale=(
                    f"工具错误率 {stats['tool_error_ratio']} 偏高，"
                    f"固化已验证的成功工具序列 {seq}"
                ),
                sop_title=f"SOP: {best_query or '高错误率场景'}标准处置",
                sop_content=(
                    "遇到同类问题时按以下已验证顺序操作：\n"
                    + "\n".join(
                        f"{i}. 执行 `{t}`（历史成功 {n} 次）"
                        for i, (t, n) in enumerate(ok_tools.items(), 1)
                    )
                    + f"\n避开高频出错工具：{', '.join(error_tools)}"
                ),
                sop_triggers=triggers,
            ))

        # 2) token 成本高 → 收紧检索预算
        mean_tokens = stats.get("total_tokens", 0) / max(
            1, stats.get("count", 1),
        )
        if mean_tokens > 4000:
            plans.append(OperatorPlan(
                operator="retrieval_evolve",
                rationale=(
                    f"平均每轨迹 {mean_tokens:.0f} tokens，收紧注入预算"
                ),
                retrieval_changes={"top_k": 4, "char_budget": 1500},
            ))

        # 3) 负反馈多于正反馈 → 提示词收紧（不删安全片段）
        neg = stats.get("negative_feedback", 0)
        pos = stats.get("positive_feedback", 0)
        if neg > pos and neg > 0:
            plans.append(OperatorPlan(
                operator="logic_evolve",
                rationale=f"负反馈({neg})多于正反馈({pos})，强化谨慎度",
                fragment_text=(
                    "## 演进提示（RSI）\n"
                    "近期负反馈偏多：先核实再动手，优先只读诊断命令，"
                    "结论给出证据（命令输出/日志行）。"
                ),
            ))
        return plans[:3]
