# -*- coding: utf-8 -*-
"""分级证据（RSI-Anything 的 E0-E3 证据层级）。

论文为每条用于改进决策的信号标注来源与保真层级：

- **E0** 规则/确定性检查（零模型依赖，完全可信）；
- **E1** 学习模型预测（本项目的 OutcomePredictor、LLM 评审）；
- **E2** 模拟器回放（本项目的 replay 检索重放/回归检查）；
- **E3** 外部真实测量（真实轨迹 outcome、用户 rsi_feedback）。

诊断与 dream 报告里的每个信号都带层级标注 —— 快速发现（doctor /
langfuse）时可直接按证据强度排序、过滤"仅凭 E1 就动手术"的高风险
决策。
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any


class EvidenceLevel(IntEnum):
    """证据保真层级（数值越大越接近真实）。"""

    E0_RULE = 0
    E1_MODEL = 1
    E2_SIMULATION = 2
    E3_EXTERNAL = 3


# 诊断字段的默认证据层级（doctor / 报告展示用）
_DIAGNOSIS_EVIDENCE_MAP: dict[str, EvidenceLevel] = {
    "stats": EvidenceLevel.E3_EXTERNAL,      # 真实轨迹聚合
    "stats.tool_error_ratio": EvidenceLevel.E0_RULE,
    "stats.positive_feedback": EvidenceLevel.E3_EXTERNAL,
    "stats.negative_feedback": EvidenceLevel.E3_EXTERNAL,
    "error_tools": EvidenceLevel.E0_RULE,    # 工具状态是确定性记录
    "ok_tools": EvidenceLevel.E0_RULE,
    "worst_traces": EvidenceLevel.E3_EXTERNAL,
    "best_traces": EvidenceLevel.E3_EXTERNAL,
}

# dream 报告各成分的证据层级
_REPORT_EVIDENCE_MAP: dict[str, EvidenceLevel] = {
    "coverage": EvidenceLevel.E2_SIMULATION,
    "est_tokens_mean": EvidenceLevel.E2_SIMULATION,
    "judge_delta": EvidenceLevel.E1_MODEL,
    "objectives": EvidenceLevel.E2_SIMULATION,
    "predictor": EvidenceLevel.E1_MODEL,
}


def tag_evidence(mapping: dict[str, EvidenceLevel]) -> dict[str, int]:
    """把层级映射序列化为 {key: level}。"""
    return {k: int(v) for k, v in mapping.items()}


def annotate_diagnosis(diagnosis: dict[str, Any]) -> dict[str, Any]:
    """给诊断字典附加 ``evidence`` 层级标注（原字段不变）。"""
    if diagnosis is None:
        return diagnosis
    diagnosis["evidence"] = tag_evidence(_DIAGNOSIS_EVIDENCE_MAP)
    return diagnosis


def evidence_legend() -> dict[int, str]:
    """层级说明（CLI / 报告用）。"""
    return {
        0: "E0 规则/确定性检查",
        1: "E1 学习模型预测",
        2: "E2 模拟器回放",
        3: "E3 外部真实测量",
    }
