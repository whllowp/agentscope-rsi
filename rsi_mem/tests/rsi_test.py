# -*- coding: utf-8 -*-
"""RSI 引擎测试：目标聚合、校验门、算子、回放回归、非退化与回滚。"""
import json

import pytest

from rsi_mem.config import RSIMemConfig
from rsi_mem.context import RSIMemContext
from rsi_mem.rsi.engine import RSIEngine
from rsi_mem.rsi.objectives import (
    IdealPoint,
    ObjectiveVector,
    cost_efficiency,
    parallel_penalty,
    pareto_frontier_auc,
    safety_from_signals,
    tchebycheff,
    tprime,
)
from rsi_mem.rsi.operators import OperatorPlan, apply_plan
from rsi_mem.rsi.plugins import RSIPlugin
from rsi_mem.rsi.replay import ReplayEvaluator
from rsi_mem.rsi.tools import build_rsi_tools
from rsi_mem.rsi.trace import TraceRecorder, serial_rounds_k
from rsi_mem.rsi.validator import GenomeValidator
from rsi_mem.storage import (
    GenomeSpec,
    ModelCallStat,
    OutcomeRecord,
    PromptFragment,
    ToolCallNode,
    TraceRecord,
)

from .conftest import DummyChatModel


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


def test_tprime_unified_metric():
    """U2 统一子问题度量 t′：λu=0 退化为 tchebycheff；λu>0 惩罚 u。

    距离域（越小越好）下惩罚项相加——即 Eq.7 奖励域 J′=J−λu·u 的
    转写；u_v0≡0（任何 λu 下已验证真实值不惩罚）。
    """
    ideal = IdealPoint()
    ideal.update(ObjectiveVector(quality=1.0, cost_eff=1.0, safety=1.0))
    v = ObjectiveVector(quality=0.8, cost_eff=0.8, safety=1.0)
    w = [0.5, 0.3, 0.2]
    assert tprime(v, w, ideal, 0.9, 0.0) == tchebycheff(v, w, ideal)
    assert tprime(v, w, ideal, 0.9, 0.5) == pytest.approx(
        tchebycheff(v, w, ideal) + 0.45,
    )
    assert tprime(v, w, ideal, 0.0, 10.0) == tchebycheff(v, w, ideal)


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


# ---------------------------------------------------------------------------
# U2：MOEA/D 选择（per-λ incumbent + 邻域替换 + 平衡部署 + t′ 惩罚）
# ---------------------------------------------------------------------------

class _PlansPlugin(RSIPlugin):
    """按脚本吐固定计划的一次性插件（精确构造候选几何用）。

    一次性：只在首个 dream 提议（跨轮回归测试需要第二轮无新候选）。
    轨迹面（零错误率 / token < 4000 / 无负反馈）下内置规则计划不触
    发，候选面完全由本插件控制。
    """

    def __init__(self, name: str, plans: list[OperatorPlan]) -> None:
        self.name = name
        self._plans = list(plans)
        self._fired = False
        self.started = False
        self.stopped = False

    async def start(self, ctx) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def propose_plans(
        self, diagnosis, current, traces, *, model=None,
    ):
        if diagnosis is None or self._fired:
            return []
        self._fired = True
        return list(self._plans)

    def evidence_tags(self):
        return {}


def _u2_ctx(tmp_path, name: str, **rsi_overrides) -> RSIMemContext:
    config = RSIMemConfig(data_dir=str(tmp_path / name))
    config.rsi.dream_every_n_traces = 100
    for k, v in rsi_overrides.items():
        setattr(config.rsi, k, v)
    return RSIMemContext(config)


async def _token_trace(ctx, query: str, score: float, *, tokens: int = 1800):
    t = TraceRecord(session_id="u2", query=query, digest="ok")
    if tokens:
        t.model_calls = [ModelCallStat(
            model="u2", input_tokens=tokens, output_tokens=100,
            latency_s=0.1,
        )]
    t.tool_calls = [
        ToolCallNode(name="bash", input_digest="x", state="ok"),
    ]
    t.outcome = OutcomeRecord(score=score)
    await ctx.db.upsert_trace(t)
    return t


async def _seed_cost_stage(ctx, n_traces: int = 4) -> None:
    """成本几何舞台：两条 ~470 字符磁盘记忆 + n 条带 token 成功轨迹。

    ``used_memory_ids`` 为空 ⇒ coverage 恒 1.0（收紧预算不伤质量检
    查）；v0（char_budget 2000）注入两条记忆（~940 字符），收紧到
    200 的候选把它们全部裁掉 → est tokens 降 ~470 → cost_eff ↑。
    """
    for i in (1, 2):
        await ctx.tiers.longterm.add(
            title=f"磁盘处置档案{i}",
            content="磁盘告警处置要点：" + "细节" * 220,
        )
    for i in range(n_traces):
        await _token_trace(ctx, f"磁盘告警 {i}", 0.7)


def _cost_plan() -> OperatorPlan:
    return OperatorPlan(
        operator="retrieval_evolve",
        rationale="U2 测试：收紧注入预算（成本优先候选）",
        retrieval_changes={"char_budget": 200},
    )


async def test_moead_lambda1_incumbent_occupied_and_gate(tmp_path):
    """①激活：成本效率极高质量≈v0 的候选占据 λ₁，部署经 λ₀ 质量门。"""
    ctx = _u2_ctx(tmp_path, "u2_activate")
    try:
        await _seed_cost_stage(ctx)
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        report = await ctx.engine.dream(trigger="u2-1")
        inc = {x["lambda"]: x for x in report["moead"]["incumbents"]}
        assert inc[1]["restore"] == "fresh"
        assert inc[1]["holder"] == "cost-tightener", \
            "λ₁（成本优先子问题）应被成本候选占据"
        assert report["moead"]["gate_size"] == 2, "成本候选过 λ₀ 质量门"
        assert report["deployed"] is True
        assert report["winner_score"] <= report["v0_score"] + 1e-9, \
            "既有非退化断言语义：门内胜者 λ₀-t′ ≤ v0 的 λ₀-t′"
    finally:
        await ctx.close()


async def test_moead_uncertainty_penalty(tmp_path):
    """②t′ 不确定性惩罚：λu>0 高不确定伪优候选被压过 v0；λu=0 不受影响。

    retrieval_evolve 零样本 → 预测器先验 uncertainty=0.8，即「高不
    确定伪优候选」（真实回放占优全在 cost 维 ~0.05，惩罚 0.8 远超）。
    """
    ctx = _u2_ctx(tmp_path, "u2_pen", uncertainty_lambda=1.0)
    try:
        await _seed_cost_stage(ctx)
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        report = await ctx.engine.dream(trigger="u2-2")
        assert report["deployed"] is False, "λu>0 应把高不确定候选压过 v0"
        assert "无候选优于" in report["reason"]
        assert report["moead"]["gate_size"] == 1, "高不确定候选进不了门"
        inc = {x["lambda"]: x for x in report["moead"]["incumbents"]}
        assert inc[1]["holder"] == "v0", "λ₁ 同样被惩罚压过"
        assert all(
            p["predicted"]["uncertainty"] == 0.8
            for p in report["candidate_predictions"]
        ), "u 取自 candidate_predictions 的预测不确定性"
    finally:
        await ctx.close()
    # 对照组：λu=0 同几何下候选正常胜出（不受惩罚影响）
    ctx0 = _u2_ctx(tmp_path, "u2_pen0")
    try:
        await _seed_cost_stage(ctx0)
        ctx0.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        r0 = await ctx0.engine.dream(trigger="u2-2b")
        assert r0["deployed"] is True, "λu=0 时不惩罚，成本候选照常部署"
    finally:
        await ctx0.close()


async def test_moead_replace_cap(tmp_path):
    """③邻域替换 nr 封顶：单候选至多替换 replace_cap 个 incumbent。"""
    ctx = _u2_ctx(tmp_path, "u2_cap", replace_cap=1)
    try:
        await _seed_cost_stage(ctx)
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        ctx.engine.registry.add(_PlansPlugin("sop-booster", [OperatorPlan(
            operator="skill_evolve",
            rationale="U2 测试：SOP 触发覆盖加成（质量候选）",
            sop_title="SOP: u2 磁盘", sop_content="1. df -h",
            sop_triggers=["磁盘"],
        )]))
        report = await ctx.engine.dream(trigger="u2-3")
        reps = report["moead"]["replacements"]
        cost_reps = [r for r in reps if r["origin"] == "cost-tightener"]
        assert len(cost_reps) <= 1, "单候选替换数 ≤ replace_cap=1"
        inc = {x["lambda"]: x for x in report["moead"]["incumbents"]}
        assert {inc[0]["holder"], inc[1]["holder"]} == {
            "cost-tightener", "sop-booster",
        }, "cap 后两 λ 各归其主（成本候选只占 λ₁）"
    finally:
        await ctx.close()


async def test_moead_cross_dream_incumbent_sop_archived(tmp_path):
    """④跨 dream 回归：落选 incumbent 的 SOP 被归档 → 下轮恢复被
    validator 拒（P2）→ 回退 v0，第二轮绝不出现「保留门拒绝」。"""
    ctx = _u2_ctx(tmp_path, "u2_cross", replace_cap=1)
    try:
        # 零 token 轨迹：cost 恒 1.0，质量 = 0.55 + 0.1×SOP 触发覆盖率
        for q in ("磁盘告警 A", "redis 内存 B", "磁盘告警 C", "证书续期 D"):
            await _token_trace(ctx, q, 0.55, tokens=0)
        ctx.engine.registry.add(_PlansPlugin("strong-sop", [OperatorPlan(
            operator="skill_evolve", rationale="U2: 强 SOP（覆盖 2/4）",
            sop_title="SOP: u2 强", sop_content="1. df -h",
            sop_triggers=["磁盘"],
        )]))
        ctx.engine.registry.add(_PlansPlugin("weak-sop", [OperatorPlan(
            operator="skill_evolve", rationale="U2: 弱 SOP（覆盖 1/4）",
            sop_title="SOP: u2 弱", sop_content="1. free -m",
            sop_triggers=["证书"],
        )]))

        r1 = await ctx.engine.dream(trigger="u2-4a")
        assert r1["deployed"] is True, "强 SOP（min(f̂) 最高）应部署"
        inc1 = {x["lambda"]: x for x in r1["moead"]["incumbents"]}
        assert inc1[0]["holder"] == "strong-sop", "λ₀ 改进最大，cap=1 先占"
        assert inc1[1]["holder"] == "weak-sop", "弱 SOP 拿走 λ₁"

        # 落选 SOP 已归档（weak-sop 的 incumbent spec 引用它）
        pool = json.loads(await ctx.db.get_meta("rsi_moead_incumbents"))
        weak_spec = GenomeSpec.model_validate_json(pool["1"]["spec_json"])
        sop_rec = await ctx.db.get_memory(weak_spec.active_sops[0])
        assert sop_rec is not None and sop_rec.status == "archived", \
            "落选候选的 SOP 应已归档"

        r2 = await ctx.engine.dream(trigger="u2-4b")
        reason = str(r2.get("reason", ""))
        assert "保留门拒绝" not in reason, "恢复失败应回退 v0，非保留门路径"
        assert r2["deployed"] is False and "无候选优于" in reason
        inc2 = {x["lambda"]: x for x in r2["moead"]["incumbents"]}
        assert inc2[1]["restore"].startswith("dropped:"), \
            "引用已归档 SOP 的 incumbent 应被 validator 拒收并重置 v0"
        assert inc2[0]["restore"] == "is_current", \
            "上轮已部署的 λ₀ incumbent 即当前版，等价 v0 基准"
    finally:
        await ctx.close()


async def test_moead_ideal_recomputed_not_stored(tmp_path):
    """⑤【P1】跨轮 incumbent 一律当前 ideal 重算：旧 objectives 不进门。

    存档 objectives 伪装 (0.99,0.99,0.99)（在旧 ideal 下 t′ 极小）；
    真实重评下该 incumbent cost 更差（+400 字符片段 → est +200
    tokens）。高分轨迹移动 ideal 后部署决策只按当前 ideal 的本轮回
    放数字：劣化 incumbent 不进门、不部署。
    """
    ctx = _u2_ctx(tmp_path, "u2_ideal")
    try:
        for i in range(3):
            await _token_trace(ctx, f"低分轨迹 {i}", 0.6)
        current = await ctx.db.active_genome()
        stale = current.model_copy(deep=True)
        stale.prompt_fragments.append(PromptFragment(
            id="u2-stale", text="stale" * 80, origin="rsi",
        ))
        await ctx.db.set_meta("rsi_moead_incumbents", json.dumps({
            "0": {
                "spec_json": stale.model_dump_json(),
                "objectives": [0.99, 0.99, 0.99],
                "dream_seq": 0,
            },
        }))
        # 高分轨迹抬高质量维度 ideal
        for i in range(3):
            await _token_trace(ctx, f"高分轨迹 {i}", 0.95)

        report = await ctx.engine.dream(trigger="u2-5")
        assert report["moead"]["gate_size"] == 1, \
            "存档 objectives 若被采信会轻松过门；按当前 ideal 重算后" \
            "劣化 incumbent 被排除"
        assert report["deployed"] is False
        assert "保留门拒绝" not in str(report.get("reason", ""))
        inc = {x["lambda"]: x for x in report["moead"]["incumbents"]}
        assert inc[0]["holder"] == "v0", "劣化 incumbent 不占 λ₀"
    finally:
        await ctx.close()


def _u2_ctx_from_dir(data_dir: str, **rsi_overrides) -> RSIMemContext:
    config = RSIMemConfig(data_dir=data_dir)
    config.rsi.dream_every_n_traces = 100
    for k, v in rsi_overrides.items():
        setattr(config.rsi, k, v)
    return RSIMemContext(config)


async def test_moead_incumbents_persist_across_engine_restart(tmp_path):
    """⑥持久化：同 data_dir 重建 RSIEngine，incumbent 恢复且 objectives
    为最新轮值。"""
    data_dir = str(tmp_path / "u2_persist" / ".rsi_mem")
    ctx = _u2_ctx_from_dir(data_dir, uncertainty_lambda=0.05, replace_cap=1)
    try:
        await _seed_cost_stage(ctx)
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        r1 = await ctx.engine.dream(trigger="u2-6a")
        # λu=0.05：λ₀ t′ = 0 + 0.05×0.8 = 0.04 > v0 的 0.014 → 不部署；
        # λ₁ t′ = 0.04 < v0 的 0.049 → 仍占据 λ₁
        assert r1["deployed"] is False
        inc1 = {x["lambda"]: x for x in r1["moead"]["incumbents"]}
        assert inc1[1]["holder"] == "cost-tightener"
    finally:
        await ctx.close()

    ctx2 = _u2_ctx_from_dir(data_dir, uncertainty_lambda=0.05, replace_cap=1)
    try:
        r2 = await ctx2.engine.dream(trigger="u2-6b")
        inc2 = {x["lambda"]: x for x in r2["moead"]["incumbents"]}
        assert inc2[1]["restore"] == "reentered", \
            "跨引擎重启应恢复 incumbent 并进入本轮重评"
        assert inc2[1]["holder"] == "incumbent:1", "恢复后守住 λ₁"
        pool = json.loads(await ctx2.db.get_meta("rsi_moead_incumbents"))
        assert pool["1"]["dream_seq"] == 2, "每轮 dream_seq 自增打标"
        assert pool["1"]["objectives"] == inc2[1]["objectives"], \
            "objectives 刷新为最新轮回放值（P1）"
        spec = json.loads(pool["1"]["spec_json"])
        assert spec["retrieval"]["char_budget"] == 200, \
            "恢复的就是上轮收紧预算的 incumbent"
    finally:
        await ctx2.close()


async def test_moead_report_keys_and_tool_compat(tmp_path):
    """⑦键名兼容：v0_score/winner_score（λ₀ t′）+ 工具消费
    primary_tchebycheff。"""
    ctx = _u2_ctx(tmp_path, "u2_keys")
    try:
        await _seed_cost_stage(ctx)
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        tools = build_rsi_tools(ctx)
        evolve = next(t for t in tools if t.name == "rsi_evolve")
        chunk = await evolve()
        assert "已部署" in chunk.content[0].text
        report = ctx.engine.last_report
        assert "v0_score" in report and "winner_score" in report
        assert report["winner_score"] <= report["v0_score"] + 1e-9
        genome = await ctx.db.active_genome()
        assert "primary_tchebycheff" in genome.scores, \
            "rsi_history 工具消费的键必须保留"
        assert genome.scores["selector"] == "balanced"
        assert "per_lambda_tchebycheff" in genome.scores
        history = next(t for t in tools if t.name == "rsi_history")
        text = (await history()).content[0].text
        assert "score=" in text, "rsi_history 应渲染 primary_tchebycheff"
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# U3-3a：judge 前缀净化 + (n,k*) 数据前置 + ToolCallNode 序号
# ---------------------------------------------------------------------------

def _judge_evidence_of(text: str) -> dict:
    """从 judge 收到的 user 消息里抠出 evidence JSON。"""
    return json.loads(
        text.split("【证据】\n", 1)[1].split("\n\n【候选变更】", 1)[0],
    )


async def test_replay_judge_prefix_view(ctx):
    """3a-①：judge 证据为前缀视图：无结局分数、含前缀工具序列与
    (n,k*)；judge_delta 仍解析；system 含附录 D.2 前缀约束语。"""
    traces = []
    for q, score in (
        ("磁盘告警 甲", 0.9),
        ("redis 内存 乙", 0.5),
        ("磁盘告警 丙", 0.3),
    ):
        t = TraceRecord(session_id="u3", query=q, digest="ok")
        for j in range(3):
            t.tool_calls.append(ToolCallNode(
                name=f"bash{j}", input_digest="d", state="ok",
            ))
        t.outcome = OutcomeRecord(score=score)
        await ctx.db.upsert_trace(t)
        traces.append(t)
    diagnosis = await ctx.engine._improver.diagnose(traces)
    assert diagnosis is not None

    seed = await ctx.db.active_genome()
    cand = seed.model_copy(deep=True)
    cand.operator = "logic_evolve"
    cand.prompt_fragments.append(PromptFragment(
        id="u3-frag", text="U3 前缀评审测试片段", origin="rsi",
    ))
    judge = DummyChatModel(
        script=['{"quality_delta": 0.2, "reason": "前缀证据下小幅改进"}'],
    )
    evaluator = ReplayEvaluator(ctx.config, ctx.db, ctx.tiers.retriever)
    report = await evaluator.evaluate(
        cand, seed, traces, validation_passed=True,
        judge_model=judge, diagnosis_summary=diagnosis,
    )
    assert report.judge_delta == pytest.approx(0.2), "judge_delta 仍解析"

    system_text = judge.received[0][0].get_text_content()
    assert "前缀" in system_text
    assert "禁止引用或推断结局绝对分数" in system_text, \
        "附录 D.2 前缀盲评约束语"

    evidence = _judge_evidence_of(judge.received[0][1].get_text_content())
    flat = json.dumps(evidence, ensure_ascii=False)
    assert "mean_outcome" not in flat, "stats.mean_outcome 已剥离"
    assert '"score"' not in flat, "worst/best_traces[].score 已剥离"
    rows = (
        evidence["diagnosis"]["worst_traces"]
        + evidence["diagnosis"]["best_traces"]
    )
    assert rows, "诊断应引用最差/最佳轨迹"
    for row in rows:
        assert "prefix_tools" in row and "calls_n" in row \
            and "serial_rounds_k" in row, "(n,k*) 数据前置"
        assert len(row["prefix_tools"]) == 3, "前 5 步上限（本例 3 步）"
        assert all(
            set(p) <= {"step", "name", "state"}
            for p in row["prefix_tools"]
        ), "前缀步只有 step/name/state，无任何分数"
    assert "error_tools" in evidence["diagnosis"], "error_tools 保留"


async def test_replay_judge_second_round_v0_same_view(tmp_path):
    """3a-①【P5 补】：首轮部署后第二轮 dream 的 v0 条目也产生 judge
    调用（同前缀视野）；评审调用数不随 rho_grid 放大。"""
    ctx = _u2_ctx(tmp_path, "u3_p5")
    try:
        await _seed_cost_stage(ctx)
        judge = DummyChatModel()  # 脚本耗尽后回默认文案 → delta 0
        ctx.config.backend_model = judge
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        assert len(ctx.config.rsi.rho_grid) == 5, "前置：5 个 ρ"

        def _judge_calls():
            # backend_model 也会喂 improver 的 hypothesize（LLM 假设），
            # 按 judge 的 system 文案甄别真正的回放评审调用
            return [
                m for m in judge.received
                if "回放评审员" in m[0].get_text_content()
            ]

        r1 = await ctx.engine.dream(trigger="u3-a")
        assert r1["deployed"] is True
        assert len(_judge_calls()) == 1, \
            "首轮：v0(seed) 免评，仅候选 1 次（未随 ρ 放大）"
        r2 = await ctx.engine.dream(trigger="u3-b")
        assert len(_judge_calls()) == 2, \
            "第二轮：v0（已部署，operator≠seed）同被 judge"
        evidence = _judge_evidence_of(
            _judge_calls()[-1][1].get_text_content(),
        )
        flat = json.dumps(evidence, ensure_ascii=False)
        assert "mean_outcome" not in flat and '"score"' not in flat, \
            "全部条目同为前缀视图（同视野）"
    finally:
        await ctx.close()


async def test_toolcall_step_index_and_serial_rounds(ctx):
    """3a-②：step_index 自动填入、旧 JSON 兼容、signals 含 (n,k*)。"""
    legacy = ToolCallNode.model_validate(json.loads(
        '{"name": "bash", "input_digest": "x", "state": "ok",'
        ' "latency_s": 0.1, "at": 123.0}',
    ))
    assert legacy.batch_id is None and legacy.step_index is None, \
        "旧库 JSON 无新键照常加载"
    t = TraceRecord(query="q")
    t.tool_calls.append(legacy)
    await ctx.db.upsert_trace(t)
    loaded = next(
        x for x in await ctx.db.list_traces(limit=10) if x.id == t.id
    )
    assert loaded.tool_calls[0].name == "bash"

    recorder = TraceRecorder(
        ctx.db, ctx.config, session_id="u3", query="q",
    )
    for i in range(4):
        recorder.add_tool_call(f"bash{i}", "d", "ok", 0.1)
    assert [c.step_index for c in recorder.record.tool_calls] == [0, 1, 2, 3]
    rec = await recorder.finish(digest="d", has_answer=True)
    assert rec.outcome.signals["calls_n"] == 4
    assert rec.outcome.signals["serial_rounds_k"] == 1, \
        "4 次连续调用（间隔 <0.5s）→ 单批 k*=1"
    assert serial_rounds_k(recorder.record.tool_calls) == 1

    recorder2 = TraceRecorder(
        ctx.db, ctx.config, session_id="u3", query="q2",
    )
    for i in range(4):
        recorder2.add_tool_call(f"bash{i}", "d", "ok", 0.1)
    for i, c in enumerate(recorder2.record.tool_calls):
        c.at = 1000.0 + 2.0 * i  # 间隔 2s → 全串行
    rec2 = await recorder2.finish(digest="d", has_answer=True)
    assert rec2.outcome.signals["serial_rounds_k"] == 4, "间隔 2s → k*=4"


# ---------------------------------------------------------------------------
# U3-3b：ρ 预算扫描 + Eq.1 Pareto 前沿下面积
# ---------------------------------------------------------------------------

def test_pareto_frontier_auc_and_parallel_penalty():
    """3b-③：非支配过滤与前沿面积（含被支配点剔除、单点/空集=0）、
    并行度惩罚线性。"""
    B = 1000.0
    stair = [(200.0, 0.2), (600.0, 0.5), (1000.0, 0.8)]
    # AUC = (0.5−0.2)·(1000−600)/1000 + (0.8−0.5)·0 = 0.12
    assert pareto_frontier_auc(stair, B) == pytest.approx(0.12)
    # 被支配点剔除：(800, 0.4) 被 (600, 0.5) 支配 → AUC 不变
    assert pareto_frontier_auc(
        stair + [(800.0, 0.4)], B,
    ) == pytest.approx(0.12)
    assert pareto_frontier_auc([(500.0, 0.5)], B) == 0.0, "单点 → 0"
    assert pareto_frontier_auc([], B) == 0.0, "空集 → 0"
    assert pareto_frontier_auc(stair, 0.0) == 0.0, "零预算 → 0"

    assert parallel_penalty([0.25, 0.75], 0.4) == pytest.approx(0.2)
    assert parallel_penalty([1.0], 0.0) == 0.0, "λ=0 关闭"
    assert parallel_penalty([], 0.5) == 0.0


async def test_replay_rho_scan_points_and_auc(ctx):
    """3b-④：3 条不同难度轨迹的 ρ 扫描点集与前沿面积；【P10】缩放
    不变性（全池 score 同乘 0.5 → AUC 不变）与全零分池 guard。"""
    mem_ids = []
    for i in (1, 2):
        m = await ctx.tiers.longterm.add(
            title=f"磁盘处置档案{i}",
            content="磁盘告警处置要点：" + "细节" * 220,
        )
        mem_ids.append(m.id)
    traces = []
    for q, score in (
        ("磁盘告警 甲", 0.9),
        ("redis 内存 乙", 0.5),
        ("磁盘告警 丙", 0.3),
    ):
        t = TraceRecord(session_id="u3", query=q, digest="ok")
        t.model_calls = [ModelCallStat(
            model="m", input_tokens=1800, output_tokens=100, latency_s=0.1,
        )]
        t.tool_calls = [
            ToolCallNode(name="bash", input_digest="x", state="ok"),
        ]
        t.outcome = OutcomeRecord(score=score)
        if "磁盘" in q:
            t.used_memory_ids = list(mem_ids)
        await ctx.db.upsert_trace(t)
        traces.append(t)

    seed = await ctx.db.active_genome()
    cand = seed.model_copy(deep=True)
    cand.operator = "retrieval_evolve"
    cand.retrieval = cand.retrieval.model_copy(update={"char_budget": 2000})
    evaluator = ReplayEvaluator(ctx.config, ctx.db, ctx.tiers.retriever)
    report = await evaluator.evaluate(
        cand, seed, traces, validation_passed=True,
    )
    assert report.judge_delta == 0.0, "前置：无 backend_model，judge 不跑"
    assert len(report.points) == 3
    tight_rows = 0
    for row in report.points:
        pts = row["points"]
        assert len(pts) == 5, "覆盖 5 个 ρ"
        ns = [p[0] for p in pts]
        assert ns[0] <= ns[-1], "n 随 ρ 单调不减"
        if ns[0] < ns[-1]:
            tight_rows += 1
        assert row["budget"] == pytest.approx(max(ns)), "B=ρ=1 点 est tokens"
        assert all(0.0 <= p[1] <= 1.0 for p in pts)
    assert tight_rows >= 1, \
        "有记忆依赖的轨迹（磁盘×2）低 ρ 端 est tokens 更小（收紧注入）"
    assert 0.0 < report.pareto_auc <= 1.0, \
        "难度梯田上升（低 ρ 丢覆盖）→ 前沿面积 > 0"

    # 【P10】缩放不变性条件显式化：judge 不跑（无 backend_model、
    # judge_delta=0）且候选/父代均无 active_sops（sop_coverage=0），
    # a = 质量/S 分子分母同缩放
    assert not cand.active_sops and not seed.active_sops
    scaled = []
    for t in traces:
        t2 = TraceRecord(session_id=t.session_id, query=t.query, digest="d")
        t2.model_calls = t.model_calls
        t2.tool_calls = t.tool_calls
        t2.used_memory_ids = t.used_memory_ids
        t2.outcome = OutcomeRecord(score=round(t.outcome.score * 0.5, 4))
        scaled.append(t2)
    report2 = await evaluator.evaluate(
        cand, seed, scaled, validation_passed=True,
    )
    assert report2.pareto_auc == pytest.approx(report.pareto_auc), \
        "全池同缩放时 AUC 不变"

    # 【P10】全零分池：S=0 → a≡0（不除零）
    zero = []
    for t in scaled:
        t3 = TraceRecord(session_id=t.session_id, query=t.query, digest="d")
        t3.model_calls = t.model_calls
        t3.tool_calls = t.tool_calls
        t3.used_memory_ids = t.used_memory_ids
        t3.outcome = OutcomeRecord(score=0.0)
        zero.append(t3)
    report0 = await evaluator.evaluate(
        cand, seed, zero, validation_passed=True,
    )
    assert report0.pareto_auc == 0.0
    assert all(
        p[1] == 0.0 for row in report0.points for p in row["points"]
    ), "全零分池 → a 恒 0"


async def test_engine_pareto_alpha_zero_matches_old_behavior(tmp_path):
    """3b-⑥：pareto_alpha=0 回退旧行为（成本候选照常过门部署）。"""
    ctx = _u2_ctx(tmp_path, "u3_alpha0", pareto_alpha=0.0)
    try:
        await _seed_cost_stage(ctx)
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        report = await ctx.engine.dream(trigger="u3-a0")
        assert report["deployed"] is True
        assert report["winner_score"] <= report["v0_score"] + 1e-9
        assert "pareto_auc_v0" in report and "pareto_auc_winner" in report
    finally:
        await ctx.close()


async def test_engine_alpha_uncertainty_double_suppression(tmp_path):
    """3b-⑥：α=0.5 且 λu>0 同开（λu×α 交互）：v0 劣化场景仍不部
    署、伪优高不确定候选被双重压制。"""
    ctx = _u2_ctx(
        tmp_path, "u3_double", pareto_alpha=0.5, uncertainty_lambda=1.0,
    )
    try:
        await _seed_cost_stage(ctx)
        plans = [
            _cost_plan(),  # 伪优：回放占优但预测 u=0.8 → λu 压制
            OperatorPlan(  # 劣化：大片段推高 est tokens → 成本更差
                operator="logic_evolve",
                rationale="U3: 劣化候选（膨胀注入）",
                fragment_text="冗余" * 300,
            ),
        ]
        ctx.engine.registry.add(_PlansPlugin("u3-cands", plans))
        report = await ctx.engine.dream(trigger="u3-dbl")
        assert report["deployed"] is False, "两类候选都不得部署"
        assert report["moead"]["gate_size"] == 1, \
            "劣化出门、伪优被 λu 压出门"
        assert "无候选优于" in report["reason"]
        assert report["pareto_auc_v0"] == pytest.approx(0.0), \
            "前置：本舞台 used 为空 → a 平坦 → AUC 0"
    finally:
        await ctx.close()
