# -*- coding: utf-8 -*-
"""U4 L1-L5 分级自评测试：事件历史双源 verify_executed 与诚实级别。"""
from rsi_mem.rsi.level import assess, collect

from .rsi_test import (
    _PlansPlugin,
    _cost_plan,
    _seed_cost_stage,
    _u2_ctx,
)


async def test_level_assessor_event_history(tmp_path):
    """末轮未部署，但 CAT_DREAM 事件历史含 execution_gate.passed=True
    （∪ 基因组 scores 双源）→ verify_executed 仍 True；当前库诚实输出
    L3 + failing 含 imp_mutable/domain_generality；dream 报告带
    level_assessment 段。"""
    ctx = _u2_ctx(tmp_path, "u4_level")
    try:
        await _seed_cost_stage(ctx)
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        r1 = await ctx.engine.dream(trigger="u4-lv1")
        assert r1["deployed"] is True, "前置：历史上有部署与执行门通过"

        # 末轮：一次性插件已 fired、incumbent=is_current → 无新候选
        # → 不部署（verify_executed 只能来自事件历史/基因组 scores）
        r2 = await ctx.engine.dream(trigger="u4-lv2")
        assert r2["deployed"] is False
        assert r2["level_assessment"]["level"] == "L3", \
            "dream 报告应挂 level_assessment 段"
        assert r2["level_assessment"]["criteria"]["verify_executed"] \
            is True

        probes = await collect(ctx.db, ctx.events, ctx.config)
        assessment = assess(probes)
        assert assessment["criteria"]["verify_executed"] is True, \
            "末轮未部署不抹掉历史执行门证据（双源）"
        assert assessment["level"] == "L3", "三项自主全真 → L3"
        assert {"imp_mutable", "domain_generality"} <= set(
            assessment["failing"],
        ), "L4/L5 卡点如实列出"
        assert assessment["evidence"] == "E0"
        assert assessment["criteria"]["propose_autonomous"] is True
        assert assessment["criteria"]["implement_autonomous"] is True
        assert assessment["criteria"]["verify_autonomous"] is True
    finally:
        await ctx.close()


async def test_level_empty_library(tmp_path):
    """空库（无 dream）：L1，全部自主判据未达。"""
    ctx = _u2_ctx(tmp_path, "u4_level_empty")
    try:
        assessment = assess(await collect(ctx.db, ctx.events, ctx.config))
        assert assessment["level"] == "L1"
        assert not assessment["criteria"]["propose_autonomous"]
        assert not assessment["criteria"]["implement_autonomous"]
        assert not assessment["criteria"]["verify_autonomous"]
        assert not assessment["criteria"]["verify_executed"]
    finally:
        await ctx.close()
