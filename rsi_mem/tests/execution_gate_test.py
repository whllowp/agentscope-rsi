# -*- coding: utf-8 -*-
"""U2 执行级验证门测试。

覆盖：validator 锁定片段全字段锁、kw 探针必召回（真实隔离栈，非
mock）、锁定片段必注入、绝对预算臂（无相对臂）、隔离栈无泄漏、
引擎拒绝路径与 incumbent 驱逐、门绿真实部署、黄金数据单源。
"""
import json
import os

import pytest

from rsi_mem.config import RSIMemConfig
from rsi_mem.context import RSIMemContext
from rsi_mem.rsi.execution_gate import (
    ExecutionGateResult,
    run_execution_gate,
)
from rsi_mem.rsi.golden import GOLDEN_KNOWLEDGE, GOLDEN_SCENARIOS
from rsi_mem.rsi.operators import OperatorPlan
from rsi_mem.rsi.validator import GenomeValidator
from rsi_mem.storage import (
    GenomeSpec,
    PromptFragment,
    RSIMemDatabase,
)
from rsi_mem.telemetry import CAT_DREAM

from .fixtures import FixedPlanPlugin
from .rsi_test import (
    _PlansPlugin,
    _cost_plan,
    _seed_cost_stage,
    _token_trace,
    _u2_ctx,
)


@pytest.fixture
async def ctx(tmp_path):
    context = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem")),
    )
    yield context
    await context.close()


# ---------------------------------------------------------------------------
# ① validator 全字段锁
# ---------------------------------------------------------------------------

async def test_validator_full_field_lock(ctx):
    """锁定片段的 enabled/slot/triggers 篡改全部 FAIL@integrity 且
    problems 指明字段（旧版只比 text，可静默退出渲染面）。"""
    seed = await ctx.db.active_genome()
    locked = next(f for f in seed.prompt_fragments if f.locked)

    def _mutate(**updates):
        cand = seed.model_copy(deep=True)
        cand.version = 99
        frag = next(
            f for f in cand.prompt_fragments if f.id == locked.id
        )
        for k, v in updates.items():
            setattr(frag, k, v)
        return cand

    for updates, field in (
        ({"enabled": False}, "enabled"),
        ({"slot": "reflex"}, "slot"),
        ({"triggers": ["x"]}, "triggers"),
        ({"text": locked.text + "x"}, "text"),  # 回归：改写仍拒
    ):
        report = await GenomeValidator(ctx.db).validate(
            _mutate(**updates), seed,
        )
        assert not report.passed and report.stage == "integrity"
        assert any(
            "锁定片段" in p and field in p for p in report.problems
        ), f"{field} 篡改应指明字段（实际 {report.problems}）"

    # 未触锁定字段的新增片段候选不受影响（防误伤合法演化）
    cand = seed.model_copy(deep=True)
    cand.version = 99
    cand.prompt_fragments.append(PromptFragment(
        id="ok-frag", text="普通演进片段", origin="rsi",
    ))
    assert await GenomeValidator(ctx.db).validate(cand, seed)


# ---------------------------------------------------------------------------
# ② 必召回：kw 探针（真实隔离栈）
# ---------------------------------------------------------------------------

async def test_gate_must_recall_kw_probe(tmp_path):
    """默认参数 kw 探针 5/5 场景全命中（隔离栈真实 FTS/RRF 检索）；
    score_threshold=1.0 → 探针 0 命中 → 必召回否决。"""
    cfg = RSIMemConfig(data_dir=str(tmp_path / "gate_recall"))
    seed = GenomeSpec.seed()
    result = await run_execution_gate(seed, seed, cfg)
    assert result.passed is True, \
        f"默认参数 kw 探针应全命中: {result.failed_checks}"
    assert result.details["max_block_chars"] > 0

    bad = seed.model_copy(deep=True)
    bad.retrieval = bad.retrieval.model_copy(
        update={"score_threshold": 1.0},
    )
    result2 = await run_execution_gate(bad, seed, cfg)
    assert not result2.passed
    missed = [
        c for c in result2.failed_checks if c.startswith("kw_recall:")
    ]
    assert len(missed) == len(GOLDEN_SCENARIOS), \
        "threshold=1.0 应杀掉全部 kw 探针命中"
    # 拒绝只来自必召回（必注入/预算不受 threshold 影响）
    assert "locked_injection" not in result2.failed_checks
    assert "inject_budget" not in result2.failed_checks


# ---------------------------------------------------------------------------
# ③ 必注入：锁定片段禁用旁路
# ---------------------------------------------------------------------------

async def test_gate_locked_injection(tmp_path):
    """锁定片段被禁用（enabled=False）的 winner 直喂执行门 →
    必注入失败（渲染面不含父代锁定片段 text）。"""
    cfg = RSIMemConfig(data_dir=str(tmp_path / "gate_lock"))
    parent = GenomeSpec.seed()
    evil = parent.model_copy(deep=True)
    frag = next(f for f in evil.prompt_fragments if f.locked)
    frag.enabled = False
    result = await run_execution_gate(evil, parent, cfg)
    assert not result.passed
    assert "locked_injection" in result.failed_checks
    assert result.details["missing_locked"], "细节应记录缺失的锁定片段"


# ---------------------------------------------------------------------------
# ④ 绝对预算臂（无相对臂）
# ---------------------------------------------------------------------------

async def test_gate_budget_absolute(tmp_path):
    """拼接片段超 governance.max_inject_chars（默认 12000）→ fail；
    正常 seed（~160 chars 渲染）照过——无相对臂，短父代不再误杀
    内容增量候选。"""
    cfg = RSIMemConfig(data_dir=str(tmp_path / "gate_budget"))
    seed = GenomeSpec.seed()
    big = seed.model_copy(deep=True)
    big.prompt_fragments.append(PromptFragment(
        id="bloat", text="冗余" * 7000, origin="rsi",
    ))
    result = await run_execution_gate(big, seed, cfg)
    assert not result.passed
    assert "inject_budget" in result.failed_checks
    assert result.details["max_block_chars"] > cfg.governance.max_inject_chars

    ok = await run_execution_gate(seed, seed, cfg)
    assert ok.passed, f"正常 seed 不触预算臂: {ok.failed_checks}"


# ---------------------------------------------------------------------------
# ⑤ 隔离性：无泄漏
# ---------------------------------------------------------------------------

async def test_gate_isolated_no_leak(ctx, monkeypatch):
    """跑门后：活 ctx 无黄金知识条目、临时目录被清理、隔离 db 关闭。"""
    import rsi_mem.rsi.execution_gate as gate_mod

    closed: list[str] = []

    class _SpyDB(RSIMemDatabase):
        async def close(self) -> None:
            closed.append(self.path)
            await super().close()

    monkeypatch.setattr(gate_mod, "RSIMemDatabase", _SpyDB)
    result = await run_execution_gate(
        GenomeSpec.seed(), GenomeSpec.seed(), ctx.config,
    )
    assert result.passed
    assert len(closed) == 1, "隔离 db 应恰好关闭一次"
    assert not os.path.isdir(os.path.dirname(closed[0])), \
        "临时目录应随门结束清理"
    # 活 ctx 的记忆库不残留黄金知识
    for kw in GOLDEN_KNOWLEDGE:
        rec = await ctx.db.find_by_title(
            tier="long", title=f"{kw}处置要点",
            status="active,promoted,candidate",
        )
        assert rec is None, f"黄金知识 {kw} 泄漏进活 ctx"


# ---------------------------------------------------------------------------
# ⑦ 引擎拒绝路径 + incumbent 驱逐
# ---------------------------------------------------------------------------

async def test_engine_rejects_and_evicts_incumbent(tmp_path, monkeypatch):
    """执行门拒绝：部署否决（版本不变/SOP 归档/报告带门结论）且
    incumbent 槽位驱逐被拒 winner（防下轮重入→重赢→重拒死循环）。"""
    import rsi_mem.rsi.engine as engine_mod

    async def _fail_gate(winner, parent, config, **kwargs):
        return ExecutionGateResult(
            passed=False, failed_checks=["kw_recall:磁盘"], details={},
        )

    monkeypatch.setattr(engine_mod, "run_execution_gate", _fail_gate)
    ctx = _u2_ctx(tmp_path, "u2_exec_reject", replace_cap=1)
    try:
        for q in ("磁盘告警 A", "redis 内存 B", "磁盘告警 C", "证书续期 D"):
            await _token_trace(ctx, q, 0.55, tokens=0)
        ctx.engine.registry.add(_PlansPlugin("strong-sop", [OperatorPlan(
            operator="skill_evolve", rationale="U2g: 强 SOP（覆盖 2/4）",
            sop_title="SOP: u2g 强", sop_content="1. df -h",
            sop_triggers=["磁盘"],
        )]))
        ctx.engine.registry.add(_PlansPlugin("weak-sop", [OperatorPlan(
            operator="skill_evolve", rationale="U2g: 弱 SOP（覆盖 1/4）",
            sop_title="SOP: u2g 弱", sop_content="1. free -m",
            sop_triggers=["证书"],
        )]))

        r1 = await ctx.engine.dream(trigger="u2g-1")
        assert r1["deployed"] is False
        assert "执行门拒绝" in r1["reason"]
        assert r1["execution_gate"]["passed"] is False
        assert r1["execution_gate"]["failed_checks"] == ["kw_recall:磁盘"]
        assert (await ctx.db.active_genome()).version == 0, \
            "拒绝后 active 版本不变"
        strong = await ctx.db.find_by_title(
            tier="muscle", title="SOP: u2g 强", status="archived",
        )
        assert strong is not None, "被拒 winner 的 SOP 应归档"
        # 驱逐：λ₀ 槽位（被拒 winner 所持）重置 v0 并记录
        assert r1["moead"]["evicted"] == [0]

        pool = json.loads(await ctx.db.get_meta("rsi_moead_incumbents"))

        def _pool_sops(p):
            return {
                sop
                for item in p.values()
                for sop in json.loads(item["spec_json"])
                .get("active_sops", [])
            }

        assert strong.id not in _pool_sops(pool), \
            "被拒 winner 不得残留在任何 incumbent 槽位（驱逐生效）"

        # 次轮：被拒 spec 不经 incumbent 槽重入
        r2 = await ctx.engine.dream(trigger="u2g-2")
        assert r2["deployed"] is False
        pool2 = json.loads(await ctx.db.get_meta("rsi_moead_incumbents"))
        assert strong.id not in _pool_sops(pool2), \
            "次轮仍不得出现被拒 winner 的 spec"
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# ⑧ 门绿：真实部署端到端
# ---------------------------------------------------------------------------

async def test_gate_green_real_deploy_end_to_end(tmp_path):
    """默认参数真实 dream：优质 retrieval 变更照常部署（门不误杀），
    报告/CAT_DREAM/基因组 scores 三处带 execution_gate 通过标记。
    bench（rsi_anything_test 断言 deploys≥1）作持续回归护栏。"""

    class _RetrievalFixedPlan(FixedPlanPlugin):
        """FixedPlanPlugin 变体：一次性优质 retrieval 变更。"""

        name = "fixed-retrieval-plan"

        def __init__(self, plan: OperatorPlan) -> None:
            super().__init__()
            self._plan = plan
            self._fired = False

        async def propose_plans(
            self, diagnosis, current, traces, *, model=None,
        ):
            if diagnosis is None or self._fired:
                return []
            self._fired = True
            return [self._plan]

    ctx = _u2_ctx(tmp_path, "u2_gate_green")
    try:
        await _seed_cost_stage(ctx)
        ctx.engine.registry.add(_RetrievalFixedPlan(_cost_plan()))
        report = await ctx.engine.dream(trigger="u2g-green")
        assert report["deployed"] is True, \
            f"默认参数优质 retrieval 变更应照常部署: {report.get('reason')}"
        assert report["execution_gate"]["passed"] is True
        assert report["execution_gate"]["failed_checks"] == []
        genome = await ctx.db.active_genome()
        assert genome.scores["execution_gate"] is True, \
            "部署基因组打 execution_gate 标（U4 verify_executed 探针源）"
        ends = [
            e["payload"]
            for e in ctx.events.read(category=CAT_DREAM, limit=10)
            if e["payload"].get("phase") == "end"
        ]
        assert ends and ends[-1]["execution_gate"]["passed"] is True, \
            "CAT_DREAM end 载荷应带门结论"
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# ⑩（复核修订）注入预算构造期快照：插件重赋值 config 不旁路
# ---------------------------------------------------------------------------

async def test_gate_budget_snapshot_immune_to_reassignment(tmp_path):
    """插件在 start(ctx) 里整字段重赋值 config.governance（pydantic
    放行）放宽 max_inject_chars —— 引擎路径按构造期快照执行，超大
    片段照拒。"""
    from rsi_mem.config import GovernanceConfig

    ctx = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / "gate_snap")),
    )
    try:
        # 对抗动作：模拟插件把活配置预算改到 10**9
        ctx.config.governance = GovernanceConfig(max_inject_chars=10 ** 9)
        assert ctx.config.governance.max_inject_chars == 10 ** 9, \
            "前置：整字段重赋值确实被放行（边界如实）"
        # 引擎构造期快照不受影响，门按快照拒
        assert ctx.engine._max_inject_chars == 12000
        big = GenomeSpec.seed()
        big.prompt_fragments.append(PromptFragment(
            id="bloat", text="冗余" * 7000, origin="rsi",
        ))
        result = await run_execution_gate(
            big, GenomeSpec.seed(), ctx.config,
            max_inject_chars=ctx.engine._max_inject_chars,
        )
        assert not result.passed
        assert "inject_budget" in result.failed_checks
        # 对照：直接按被改宽的活配置跑（旁路路径）会放行——证明
        # 快照是必要防线而非多余参数
        bypassed = await run_execution_gate(
            big, GenomeSpec.seed(), ctx.config,
        )
        assert bypassed.passed, "前置：活配置路径确实可被旁路"
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# ⑨ 黄金数据单源
# ---------------------------------------------------------------------------

def test_golden_single_source():
    """bench.rounds 的场景/知识与 rsi.golden 同源（leaf 冻结原值）。"""
    from rsi_mem.bench import rounds as bench_rounds
    from rsi_mem.rsi import golden

    assert bench_rounds.SCENARIOS == golden.GOLDEN_SCENARIOS
    assert bench_rounds._KNOWLEDGE == golden.GOLDEN_KNOWLEDGE
    assert len(golden.GOLDEN_SCENARIOS) == 5
