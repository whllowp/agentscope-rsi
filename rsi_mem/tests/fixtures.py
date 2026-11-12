# -*- coding: utf-8 -*-
"""测试夹具：可被配置装载的示例 RSI 插件。"""
from __future__ import annotations

from typing import Any

from agentscope.model import ChatModelBase

from rsi_mem.rsi.evidence import EvidenceLevel
from rsi_mem.rsi.operators import OperatorPlan
from rsi_mem.rsi.plugins import RSIPlugin


class FixedPlanPlugin(RSIPlugin):
    """总是贡献一条 logic_evolve 计划的固定插件（测试用）。"""

    name = "fixed-plan"
    description = "test fixture plugin"

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self, ctx) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def propose_plans(
        self,
        diagnosis: dict[str, Any] | None,
        current,
        traces,
        *,
        model: ChatModelBase | None = None,
    ):
        if diagnosis is None:
            return []
        return [
            OperatorPlan(
                operator="logic_evolve",
                rationale="fixture: 强化只读诊断",
                fragment_text=(
                    "## fixture 插件片段\n"
                    "动手前先用只读命令核实现状。"
                ),
            ),
        ]

    def evidence_tags(self) -> dict[str, EvidenceLevel]:
        return {"fixture_signal": EvidenceLevel.E0_RULE}


class BrokenPlugin(RSIPlugin):
    """propose 抛异常的坏插件：不应拖垮 dream。"""

    name = "broken"

    async def propose_plans(self, *args, **kwargs):
        raise RuntimeError("boom")


class NotAPlugin:
    """非 RSIPlugin：装载时应被拒绝而不是崩溃。"""
