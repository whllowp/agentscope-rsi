# -*- coding: utf-8 -*-
"""RSI 引擎：轨迹、基因组、算子、校验、多目标、回放、改进器与工具。

插件化（RSI-Anything domain-adapter 思路）：计划生成收敛到
:class:`PluginRegistry`；预测器（世界模型类比）与证据分级见
:mod:`.predictor` / :mod:`.evidence`。
"""
from .engine import RSIEngine
from .evidence import EvidenceLevel, evidence_legend, tag_evidence
from .improver import Improver
from .objectives import IdealPoint, ObjectiveVector, tchebycheff
from .operators import OperatorPlan, MutationOutcome, apply_plan
from .plugins import (
    ENTRY_POINT_GROUP,
    BuiltinImproverPlugin,
    PluginRegistry,
    RSIPlugin,
    build_default_registry,
)
from .predictor import (
    CandidateFeatures,
    OutcomePredictor,
    Prediction,
    is_conservative_operator,
)
from .replay import ReplayEvaluator, ReplayReport
from .trace import TraceRecorder, attach_feedback
from .validator import GenomeValidator, ValidationReport

__all__ = [
    "RSIEngine",
    "Improver",
    "IdealPoint",
    "ObjectiveVector",
    "tchebycheff",
    "OperatorPlan",
    "MutationOutcome",
    "apply_plan",
    "ReplayEvaluator",
    "ReplayReport",
    "TraceRecorder",
    "attach_feedback",
    "GenomeValidator",
    "ValidationReport",
    "EvidenceLevel",
    "evidence_legend",
    "tag_evidence",
    "ENTRY_POINT_GROUP",
    "BuiltinImproverPlugin",
    "PluginRegistry",
    "RSIPlugin",
    "build_default_registry",
    "CandidateFeatures",
    "OutcomePredictor",
    "Prediction",
    "is_conservative_operator",
]
