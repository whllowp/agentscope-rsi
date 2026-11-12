# -*- coding: utf-8 -*-
"""RSI 引擎测试：目标聚合、校验门、算子、回放回归、非退化与回滚。"""
import pytest

from rsi_mem.config import RSIMemConfig
from rsi_mem.context import RSIMemContext
from rsi_mem.rsi.engine import RSIEngine
from rsi_mem.rsi.objectives import (
    IdealPoint,
    ObjectiveVector,
    cost_efficiency,
    safety_from_signals,
    tchebycheff,
)
from rsi_mem.rsi.operators import OperatorPlan, apply_plan
from rsi_mem.rsi.replay import ReplayEvaluator
from rsi_mem.rsi.validator import GenomeValidator
from rsi_mem.storage import (
    GenomeSpec,
    OutcomeRecord,
    ToolCallNode,
    TraceRecord,
)


@pytest.fixture
async def ctx(tmp_path):
    config = RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem"))
    config.rsi.dream_every_n_traces = 100  # 测试里手动 dream
    context = RSIMemContext(config)
    yield context
    await context.close()


async def _mk_trace(ctx, query, score, *, used=(), error=False):
    t = TraceRecord(session_id="s1", query=query, digest=f"answer:{query}")
    t.tool_calls = [
        ToolCallNode(
            name="bash", input_digest="df -h",
            state="error" if error else "ok",
        ),
    ]
    t.outcome = OutcomeRecord(score=score)
    t.used_memory_ids = list(used)
    await ctx.db.upsert_trace(t)
    return t


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------

def test_tchebycheff_and_helpers():
    ideal = IdealPoint()
    ideal.update(ObjectiveVector(quality=1.0, cost_eff=1.0, safety=1.0))
    good = ObjectiveVector(quality=1.0, cost_eff=1.0, safety=1.0)
    bad = ObjectiveVector(quality=0.0, cost_eff=0.0, safety=0.0)
    assert tchebycheff(good, [0.5, 0.3, 0.2], ideal) == 0.0
    assert tchebycheff(bad, [0.5, 0.3, 0.2], ideal) == pytest.approx(0.5)
    assert cost_efficiency(0) == 1.0
    assert cost_efficiency(2000) == pytest.approx(0.5)
    assert safety_from_signals(
        permission_denied=1, permission_total=4, validation_passed=True,
    ) == pytest.approx(0.75)
    assert safety_from_signals(
        permission_denied=0, permission_total=0, validation_passed=False,
    ) == 0.0


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------

async def test_validator_locked_fragment_guard(ctx):
    from rsi_mem.storage import PromptFragment

    seed = await ctx.db.active_genome()
    evil = seed.model_copy(deep=True)
    # 保留一个普通片段（通过结构段），删掉锁定安全片段
    evil.prompt_fragments = [
        PromptFragment(text="普通演进片段", origin="rsi"),
    ]
    report = await GenomeValidator(ctx.db).validate(evil, seed)
    assert not report.passed and report.stage == "integrity"
    assert any("锁定片段" in p for p in report.problems)
    # 修复也无法恢复锁定片段
    assert await GenomeValidator(ctx.db).repair(evil, seed) is None


async def test_validator_rejects_identity_and_dangling_sop(ctx):
    from rsi_mem.storage import PromptFragment

    seed = await ctx.db.active_genome()
    same = seed.model_copy(deep=True)
    report = await GenomeValidator(ctx.db).validate(same, seed)
    assert report.passed  # version 相同 → v0 自评允许
    # version 不同且内容相同 → 拒绝（空演化）
    same.version = 99
    report = await GenomeValidator(ctx.db).validate(same, seed)
    assert not report.passed and any(
        "完全相同" in p for p in report.problems
    )

    # 悬挂 SOP 引用 → smoke 段拒绝；修复剔除后应通过（有真实变更）
    dangling = seed.model_copy(deep=True)
    dangling.version = 100
    dangling.prompt_fragments.append(
        PromptFragment(text="收紧诊断提示", origin="rsi"),
    )
    dangling.active_sops = ["no-such-sop"]
    report = await GenomeValidator(ctx.db).validate(dangling, seed)
    assert not report.passed and report.stage == "smoke"
    repaired = await GenomeValidator(ctx.db).repair(dangling, seed)
    assert repaired is not None and repaired.active_sops == []
    final = await GenomeValidator(ctx.db).validate(repaired, seed)
    assert final.passed


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

async def test_retrieval_evolve_bounds(ctx):
    seed = await ctx.db.active_genome()
    plan = OperatorPlan(
        operator="retrieval_evolve",
        retrieval_changes={"top_k": 999},  # 越界
    )
    assert await apply_plan(seed, plan, ctx.tiers) is None
    plan = OperatorPlan(
        operator="retrieval_evolve",
        retrieval_changes={"top_k": 3, "char_budget": 1200},
    )
    out = await apply_plan(seed, plan, ctx.tiers)
    assert out is not None
    assert out.genome.retrieval.top_k == 3
    assert out.genome.operator == "retrieval_evolve"


async def test_skill_evolve_candidate_sop(ctx):
    seed = await ctx.db.active_genome()
    plan = OperatorPlan(
        operator="skill_evolve",
        rationale="高频故障固化",
        sop_title="SOP: 磁盘告警",
        sop_content="1. df -h\n2. 清日志\n3. 扩容评估",
        sop_triggers=["磁盘", "disk"],
    )
    out = await apply_plan(seed, plan, ctx.tiers)
    assert out is not None
    sop = out.sop_candidates[0]
    assert sop.status == "candidate", "RSI SOP 候选部署前不得激活"
    assert sop.id in out.genome.active_sops
    # candidate 不参与反射注入
    hits = await ctx.tiers.muscle.match("磁盘告警")
    assert not any(h.id == sop.id for h in hits)


# ---------------------------------------------------------------------------
# replay：回归覆盖检测
# ---------------------------------------------------------------------------

async def test_replay_detects_coverage_regression(ctx):
    memory = await ctx.tiers.longterm.add(
        title="磁盘满处置要点",
        content="先查大文件：du -sh /var/*；清日志前确认轮转策略。",
    )
    trace = await _mk_trace(
        ctx, "磁盘满了", 0.9, used=[memory.id],
    )
    seed = await ctx.db.active_genome()
    # 父代（top_k=5）能召回该记忆
    base = await ctx.tiers.retriever.search(
        "磁盘满了", tiers=("long", "short"), top_k=5,
    )
    assert any(r.record.id == memory.id for r in base)

    # 候选：top_k=1 + 高阈值 → 丢覆盖
    harsh = seed.model_copy(deep=True)
    harsh.retrieval = harsh.retrieval.model_copy(
        update={"top_k": 1, "score_threshold": 0.9},
    )
    evaluator = ReplayEvaluator(ctx.config, ctx.db, ctx.tiers.retriever)
    report = await evaluator.evaluate(
        harsh, seed, [trace], validation_passed=True,
    )
    assert report.coverage < 1.0, "收紧参数丢掉成功轨迹用过的记忆应被检出"

    keep = seed.model_copy(deep=True)
    keep.retrieval = keep.retrieval.model_copy(update={"top_k": 5})
    report_keep = await evaluator.evaluate(
        keep, seed, [trace], validation_passed=True,
    )
    assert report_keep.coverage == 1.0


# ---------------------------------------------------------------------------
# engine：dream 全流程 + 非退化 + 回滚
# ---------------------------------------------------------------------------

async def _seed_traces(ctx, n=6):
    for i in range(n):
        await _mk_trace(
            ctx, f"线上磁盘告警 {i}", 0.7, error=(i % 2 == 0),
        )


async def test_engine_dream_deploys_or_keeps_v0(ctx):
    await _seed_traces(ctx)
    engine = ctx.engine
    report = await engine.dream(trigger="manual")
    assert report["candidates"] >= 1  # 至少 v0
    assert "v0_score" in report
    # 非退化：若部署，胜者分数（Tchebycheff 距离）必须 ≤ v0
    if report["deployed"]:
        assert report["winner_score"] <= report["v0_score"] + 1e-9
        genome = await ctx.db.active_genome()
        assert genome.version == report["version"]
        assert genome.operator == report["operator"]
        # 部署的候选 SOP 变为 active
        for sop_id in genome.active_sops:
            rec = await ctx.db.get_memory(sop_id)
            if rec is not None:
                assert rec.status in ("active", "promoted")


async def test_engine_non_regression_when_no_candidates(ctx):
    # 无轨迹 → 诊断为 None → 只有 v0 → 不部署
    report = await ctx.engine.dream(trigger="manual")
    assert report["deployed"] is False
    assert report["reason"]


async def test_engine_rollback(ctx):
    seed = await ctx.db.active_genome()
    v1 = seed.model_copy(
        update={
            "version": 1, "parent_version": 0,
            "operator": "retrieval_evolve", "notes": "收紧",
        },
    )
    v1.retrieval = v1.retrieval.model_copy(update={"top_k": 3})
    await ctx.db.save_genome(v1)
    prev = await ctx.engine.rollback()
    assert prev is not None
    assert prev.version == 2 and prev.operator == "rollback"
    assert prev.retrieval.top_k == seed.retrieval.top_k


async def test_improver_evidence_gate_and_rule_plans(ctx):
    await _seed_traces(ctx, 4)
    improver = ctx.engine._improver
    traces = await ctx.db.list_traces(limit=10)
    diagnosis = await improver.diagnose(traces)
    assert diagnosis is not None
    # 引用的轨迹 id 全部真实存在
    all_ids = {t.id for t in traces}
    assert set(diagnosis["cited_trace_ids"]) <= all_ids
    # 离线（无模型）规则计划：错误率高 → 蒸馏 SOP
    plans = await improver.hypothesize(diagnosis, model=None)
    assert plans, "规则兜底应产出计划"
    assert all(p.operator in (
        "skill_evolve", "logic_evolve", "retrieval_evolve", "generalize",
    ) for p in plans)
