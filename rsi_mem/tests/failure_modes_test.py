# -*- coding: utf-8 -*-
"""U3 失败模式防御测试：自证回路、多样性坍缩、分布监测、遥测接线
与帧锁定审计。"""
import random

import pytest

from rsi_mem.config import RSIMemConfig
from rsi_mem.context import RSIMemContext
from rsi_mem.rsi.engine import _explore_retrieval_changes, _render_hash
from rsi_mem.rsi.predictor import CandidateFeatures, OutcomePredictor
from rsi_mem.rsi.replay import ReplayEvaluator
from rsi_mem.rsi.trace import TraceRecorder
from rsi_mem.storage import (
    GenomeSpec,
    ModelCallStat,
    OutcomeRecord,
    PromptFragment,
    ToolCallNode,
    TraceRecord,
)
from rsi_mem.telemetry import CAT_DREAM

from .conftest import DummyChatModel
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
# ① E3 回填：production / replay source 分类
# ---------------------------------------------------------------------------

async def test_e3_backfill_production_source(tmp_path):
    """部署记 pending（genome_version 锚定）→ ≥5 条部署后真实轨迹
    → 次轮 dream 回填 realized.source=="production"；同轮回放回填行
    source=="replay"。"""
    ctx = _u2_ctx(tmp_path, "u3_e3")
    try:
        await _seed_cost_stage(ctx)  # 4 条部署前轨迹（0.7 分）
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        r1 = await ctx.engine.dream(trigger="u3-e3-1")
        assert r1["deployed"] is True
        genome = await ctx.db.active_genome()
        assert genome.version == 1
        # 部署即记 pending 预测；报告带分布/多样性观测面
        preds = await ctx.db.load_predictions(realized_only=False)
        e3_rows = [
            p for p in preds if p["genome_version"] == 1
        ]
        assert e3_rows, "部署应记 pending E3 预测行"
        assert all(p["realized"] is None for p in e3_rows)
        assert r1["distribution"]["source_ratio"].get("production", 0) > 0
        assert r1["diversity"]["origins"].get("cost-tightener") == 1

        # 部署后 5 条真实轨迹（0.85 分）→ 次轮回填 production
        for i in range(5):
            await _token_trace(ctx, f"部署后真实 {i}", 0.85)
        await ctx.engine.dream(trigger="u3-e3-2")
        preds2 = await ctx.db.load_predictions(realized_only=False)
        e3 = [p for p in preds2 if p["genome_version"] == 1]
        assert e3 and all(
            p["realized"] is not None for p in e3
        ), "样本足够时 pending 应被回填"
        assert all(
            p["realized"]["source"] == "production" for p in e3
        ), "E3 回填标 production"
        # post 0.85 − pre 0.7 ≈ +0.15（cost/safety 置 0）
        assert e3[0]["realized"]["quality_delta"] == pytest.approx(
            0.15, abs=0.01,
        )
        assert e3[0]["realized"]["cost_eff_delta"] == 0.0
        # 同轮回放回填行标 replay（r1 的候选预测）
        replay_rows = [
            p for p in preds2
            if p["realized"] and p["realized"].get("source") == "replay"
        ]
        assert replay_rows, "候选回放回填行应标 replay"
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# ② predictor 跨重启过滤 production 行
# ---------------------------------------------------------------------------

async def test_production_rows_excluded_after_restart(ctx):
    """production realized 行不进 predictor 在线统计——过滤在
    _ensure_loaded 加载层，新实例（跨重启）同库加载同样生效。"""
    pid_replay = await ctx.db.insert_prediction(
        operator="retrieval_evolve", genome_version=None,
        features={"operator": "retrieval_evolve"},
        predicted={"quality_delta": 0.0},
    )
    await ctx.db.update_prediction_realized(pid_replay, {
        "quality_delta": 0.1, "cost_eff_delta": 0.0,
        "safety_delta": 0.0, "source": "replay",
    })
    pid_prod = await ctx.db.insert_prediction(
        operator="retrieval_evolve", genome_version=1,
        features={"operator": "retrieval_evolve"},
        predicted={"quality_delta": 0.5},
    )
    await ctx.db.update_prediction_realized(pid_prod, {
        "quality_delta": 0.9, "cost_eff_delta": 0.0,
        "safety_delta": 0.0, "source": "production",
    })

    predictor = OutcomePredictor(ctx.db)  # 跨重启 = 同库新实例
    await predictor._ensure_loaded()
    assert predictor.sample_count() == 1, "production 行不得进在线统计"
    assert predictor.mae() == pytest.approx(0.1)  # 仅 |0.0−0.1|
    pred = predictor.predict(CandidateFeatures(operator="retrieval_evolve"))
    assert pred.quality_delta == pytest.approx(0.1)
    assert pred.samples == 1


# ---------------------------------------------------------------------------
# ③ source 分类学
# ---------------------------------------------------------------------------

async def test_trace_source_taxonomy(tmp_path):
    """TraceRecorder 主路径缺省 production；bench 轨迹自带 "bench"；
    diagnose 分布段占比正确；doctor 全自产才告警。"""
    from rsi_mem.bench.rounds import OpsEnv
    from rsi_mem.cli import doctor

    ctx = RSIMemContext(RSIMemConfig(data_dir=str(tmp_path / "u3_src")))
    try:
        # a) 真实轨迹（录制主路径）→ production
        recorder = TraceRecorder(
            ctx.db, ctx.config, session_id="s", query="磁盘告警",
        )
        recorder.add_tool_call("bash", "df -h", "ok", 0.1)
        rec = await recorder.finish(digest="done", has_answer=True)
        assert rec.outcome.signals["source"] == "production"

        # b) 合成轨迹显式自带 source（bench）
        await OpsEnv(ctx).emit_traces(2, 0)
        all_traces = await ctx.db.list_traces(limit=20)
        bench_traces = [
            t for t in all_traces
            if (t.outcome.signals or {}).get("source") == "bench"
        ]
        assert bench_traces, "bench 轨迹应带 source=bench"
        assert all(
            t.outcome.signals["source"] == "bench" for t in bench_traces
        )

        # c) 一条无 source 键的直写轨迹 → 计 production 且计入 untagged
        t = TraceRecord(session_id="s", query="直写", digest="x")
        t.outcome = OutcomeRecord(score=0.5)
        await ctx.db.upsert_trace(t)

        diagnosis = await ctx.engine._improver.diagnose(
            await ctx.db.list_traces(limit=30),
        )
        dist = diagnosis["distribution"]
        assert dist["source_ratio"].get("production", 0) > 0
        assert dist["source_ratio"].get("bench", 0) > 0
        assert dist["untagged"] >= 1
        assert 0.0 < dist["unique_query_ratio"] <= 1.0

        # d) 混合来源 → doctor 不报外部信号消失
        result = await doctor(ctx)
        assert not any(
            "外部信号消失" in f["message"] for f in result["findings"]
        )
    finally:
        await ctx.close()

    # e) 全合成（bench）→ info
    ctx2 = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / "u3_bench_only")),
    )
    try:
        await OpsEnv(ctx2).emit_traces(3, 0)
        result2 = await doctor(ctx2)
        hit = [
            f for f in result2["findings"]
            if "外部信号消失" in f["message"]
        ]
        assert hit and hit[0]["severity"] == "info"
    finally:
        await ctx2.close()


# ---------------------------------------------------------------------------
# ④ judge_delta_mean 载荷与 doctor 评审哨兵
# ---------------------------------------------------------------------------

async def test_judge_delta_payload_and_doctor(tmp_path):
    from rsi_mem.cli import doctor

    ctx = _u2_ctx(tmp_path, "u3_judge")
    try:
        await _seed_cost_stage(ctx)
        ctx.engine.registry.add(FixedPlanPlugin())
        judge = DummyChatModel(script=[
            '{"quality_delta": 0.2, "reason": "评审证据"}',
        ] * 3)
        await ctx.engine.dream(trigger="u3-j1", judge_model=judge)
        ends = [
            e["payload"]
            for e in ctx.events.read(category="dream", limit=30)
            if e["payload"].get("phase") == "end"
        ]
        assert ends[-1]["judge_delta_mean"] > 0, \
            "judge 跑过时 end 载荷应带正的 judge_delta_mean"
        # 补第 5 条轨迹：E3 回填到期（min_traces_after_deploy=5），
        # 自参考哨兵此时才应发声（复核修订的冷启动豁免之下）
        await _token_trace(ctx, "磁盘告警 补充", 0.7)

        # 之后 5 轮无 judge → 恒 0 → doctor 报『评审未激活』（judge
        # 曾激活后沉默 = warn）；自参考哨兵不报——本场景从未部署，
        # E3 行不可能存在（复核修订：永久冷启动不误报）
        for i in range(5):
            await ctx.engine.dream(trigger=f"u3-j{i + 2}")
        result = await doctor(ctx)
        msgs = [f["message"] for f in result["findings"]]
        assert any("评审未激活" in m for m in msgs)
        assert not any("预测器自参考" in m for m in msgs), \
            "从未部署的库不可能有 production 行（E3 未到期不算故障）"
    finally:
        await ctx.close()

    # 旧事件（无 judge_delta_mean 键）静默降级不误报
    ctx2 = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / "u3_judge_old")),
    )
    try:
        for _ in range(6):
            ctx2.events.emit_sync(
                CAT_DREAM, {"phase": "end", "deployed": False},
            )
        result2 = await doctor(ctx2)
        assert not any(
            "评审未激活" in f["message"]
            for f in result2["findings"]
        )
    finally:
        await ctx2.close()


# ---------------------------------------------------------------------------
# ④b（复核修订）冷启动不误报：E3 回填未到期时自参考哨兵静默
# ---------------------------------------------------------------------------

async def test_cold_start_no_self_reference_noise(tmp_path):
    """部署后真实轨迹 < min_traces_after_deploy（E3 回填未到期）→
    doctor 不报『预测器自参考』（此前冷启动全程误 warn）。"""
    from rsi_mem.cli import doctor

    ctx = _u2_ctx(tmp_path, "u3_coldstart")
    try:
        await _seed_cost_stage(ctx)  # 4 条轨迹（<5）
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        r1 = await ctx.engine.dream(trigger="u3-cold")
        assert r1["deployed"] is True
        # 部署后无新轨迹 → 无 production 回填属预期
        result = await doctor(ctx)
        assert not any(
            "预测器自参考" in f["message"] for f in result["findings"]
        ), "E3 回填未到期（部署后 0 条 < 5）不报自参考"
        # 到期（≥5 条部署后轨迹）后跑一次 dream → E3 真回填 production
        # → 依旧健康不报（正例：曾部署的系统拿回了外部证据）
        for i in range(5):
            await _token_trace(ctx, f"部署后 {i}", 0.7)
        await ctx.engine.dream(trigger="u3-cold2")
        preds = await ctx.db.load_predictions(realized_only=True)
        assert any(
            (p["realized"] or {}).get("source") == "production"
            for p in preds
        ), "前置：E3 已到期回填"
        result2 = await doctor(ctx)
        assert not any(
            "预测器自参考" in f["message"] for f in result2["findings"]
        ), "有 production 行 → 非自参考（到期不误报的正例）"
    finally:
        await ctx.close()

    # 正例：已部署 + E3 到期但回填缺席（模拟回填链路失效）→ 应报
    ctx2 = _u2_ctx(tmp_path, "u3_selfref_pos")
    try:
        await _seed_cost_stage(ctx2)
        ctx2.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        r = await ctx2.engine.dream(trigger="sr1")
        assert r["deployed"] is True
        for i in range(5):
            await _token_trace(ctx2, f"部署后 {i}", 0.7)

        async def _no_backfill(traces, current):
            return None  # 模拟 E3 回填失效

        orig_backfill = ctx2.engine._backfill_e3
        ctx2.engine._backfill_e3 = _no_backfill
        try:
            await ctx2.engine.dream(trigger="sr2")
        finally:
            ctx2.engine._backfill_e3 = orig_backfill
        preds = await ctx2.db.load_predictions(realized_only=True)
        assert not any(
            (p["realized"] or {}).get("source") == "production"
            for p in preds
        ), "前置：回填被模拟失效，realized 全 replay"
        result3 = await doctor(ctx2)
        assert any(
            "预测器自参考" in f["message"]
            and f["severity"] == "warn"
            for f in result3["findings"]
        ), "已部署 + 到期 + 全 replay → 自参考 warn"
    finally:
        await ctx2.close()


# ---------------------------------------------------------------------------
# ⑤ pareto n_effective 与恒定序列哨兵
# ---------------------------------------------------------------------------

async def test_pareto_n_effective(ctx, tmp_path):
    from rsi_mem.cli import doctor

    mem_ids = []
    for i in (1, 2):
        m = await ctx.tiers.longterm.add(
            title=f"磁盘处置档案{i}",
            content="磁盘告警处置要点：" + "细节" * 220,
        )
        mem_ids.append(m.id)
    traces = []
    for q, used in (("磁盘告警 甲", mem_ids), ("redis 内存 乙", None)):
        t = TraceRecord(session_id="u3ne", query=q, digest="ok")
        t.model_calls = [ModelCallStat(
            model="m", input_tokens=1800, output_tokens=100, latency_s=0.1,
        )]
        t.tool_calls = [
            ToolCallNode(name="bash", input_digest="x", state="ok"),
        ]
        t.outcome = OutcomeRecord(score=0.7)
        if used:
            t.used_memory_ids = list(used)
        await ctx.db.upsert_trace(t)
        traces.append(t)

    seed = await ctx.db.active_genome()
    cand = seed.model_copy(deep=True)
    cand.operator = "retrieval_evolve"
    cand.retrieval = cand.retrieval.model_copy(
        update={"char_budget": 2000},
    )
    evaluator = ReplayEvaluator(ctx.config, ctx.db, ctx.tiers.retriever)
    report = await evaluator.evaluate(
        cand, seed, traces, validation_passed=True,
    )
    # 甲轨迹低 ρ 丢覆盖 → a 有区分度；乙轨迹全平 → 不计入均值
    n_eff = {row["trace_id"]: row["n_effective"] for row in report.points}
    assert report.details["n_effective_traces"] == 1
    assert any(v >= 2 for v in n_eff.values())
    assert any(v < 2 for v in n_eff.values())

    # 全平轨迹（零分池）：不进均值，且 auc=0.0（非空样本不伪装满分）
    flat = []
    for t in traces:
        t2 = TraceRecord(
            session_id=t.session_id, query=t.query, digest="d",
        )
        t2.model_calls = t.model_calls
        t2.tool_calls = t.tool_calls
        t2.used_memory_ids = t.used_memory_ids
        t2.outcome = OutcomeRecord(score=0.0)
        flat.append(t2)
    rep0 = await evaluator.evaluate(
        cand, seed, flat, validation_passed=True,
    )
    assert rep0.details["n_effective_traces"] == 0
    assert rep0.pareto_auc == 0.0

    # 恒定序列（含恒 0）+ 新轨迹持续流入 → doctor warn；严格递减
    # 哨兵不误触
    ctx2 = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / "u3_auc_const")),
    )
    try:
        for n in (10, 12, 14, 16, 18):  # traces_evaluated 递增=有新数据
            ctx2.events.emit_sync(CAT_DREAM, {
                "phase": "end", "deployed": True,
                "pareto_auc_winner": 0.0,
                "traces_evaluated": n,
            })
        result = await doctor(ctx2)
        assert any(
            "持续无变化" in f["message"] for f in result["findings"]
        ), "有新数据而 AUC 恒定 → 前沿失明信号"
        assert not any(
            "连续下滑" in f["message"] for f in result["findings"]
        )
    finally:
        await ctx2.close()

    # 复核修订：空闲但健康（零新轨迹，回放确定性 → 逐轮完全相同）
    # 不再误报「前沿失明」
    ctx3 = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / "u3_auc_idle")),
    )
    try:
        for _ in range(5):
            ctx3.events.emit_sync(CAT_DREAM, {
                "phase": "end", "deployed": False,
                "pareto_auc_winner": 0.0,
                "traces_evaluated": 20,  # 恒定：没有新轨迹流入
            })
        result3 = await doctor(ctx3)
        assert not any(
            "持续无变化" in f["message"] for f in result3["findings"]
        ), "空闲系统的恒定 AUC 是确定性回放的预期输出，非故障"
    finally:
        await ctx3.close()


# ---------------------------------------------------------------------------
# ⑥ 探索注入
# ---------------------------------------------------------------------------

def test_explore_changes_deterministic_and_bounded():
    """扰动生成纯函数：同种子同输出、参数在 RetrievalParams 边界内
    且必与当前值可区分。"""
    seed = GenomeSpec.seed()
    assert (
        _explore_retrieval_changes(random.Random(42), seed)
        == _explore_retrieval_changes(random.Random(42), seed)
    )
    distinct = set()
    for v in range(20):
        ch = _explore_retrieval_changes(random.Random(v), seed)
        assert 1 <= ch["top_k"] <= 20
        assert ch["top_k"] != seed.retrieval.top_k, "必改 top_k（防空演化）"
        if "char_budget" in ch:
            assert 200 <= ch["char_budget"] <= 8000
        if "score_threshold" in ch:
            assert 0.0 <= ch["score_threshold"] <= 1.0
        distinct.add(ch["top_k"])
    assert len(distinct) >= 2, "不同种子应产生不同扰动"


async def test_exploration_injection(tmp_path):
    """连续同算子部署 ≥3 → dream 注入 origin="explore" 候选。"""
    ctx = _u2_ctx(tmp_path, "u3_explore")
    try:
        seed = await ctx.db.active_genome()
        prev = seed
        for v in (1, 2, 3):  # 手工造 3 版连续 retrieval_evolve 部署史
            g = prev.model_copy(update={
                "version": v, "parent_version": prev.version,
                "operator": "retrieval_evolve",
                "notes": f"u3 同算子 {v}",
            })
            g.retrieval = g.retrieval.model_copy(
                update={"top_k": max(1, 5 - v)},
            )
            await ctx.db.save_genome(g)
            prev = g
        await _token_trace(ctx, "磁盘告警 0", 0.7)

        report = await ctx.engine.dream(trigger="u3-explore")
        assert report["diversity"]["consecutive_same_operator"] == 3
        origins = [
            p.get("origin_plugin")
            for p in report["candidate_predictions"]
        ]
        assert "explore" in origins, f"应注入 explore 候选: {origins}"
        assert report["diversity"]["origins"].get("explore") == 1
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# ⑦ 算子集中度哨兵
# ---------------------------------------------------------------------------

async def test_operator_concentration_sentinel(tmp_path):
    from rsi_mem.cli import doctor

    # a) 6 版全 logic_evolve → warn
    ctx = RSIMemContext(RSIMemConfig(data_dir=str(tmp_path / "u3_conc")))
    try:
        seed = await ctx.db.active_genome()
        prev = seed
        for v in range(1, 7):
            g = prev.model_copy(update={
                "version": v, "parent_version": prev.version,
                "operator": "logic_evolve",
            })
            g.prompt_fragments = [
                *g.prompt_fragments,
                PromptFragment(
                    id=f"c{v}", text=f"片段{v}" + "细" * v, origin="rsi",
                ),
            ]
            await ctx.db.save_genome(g)
            prev = g
        result = await doctor(ctx)
        assert any(
            "算子集中度" in f["message"] and f["severity"] == "warn"
            for f in result["findings"]
        )
    finally:
        await ctx.close()

    # b) 混合算子 → 不报
    ctx2 = RSIMemContext(RSIMemConfig(data_dir=str(tmp_path / "u3_mix")))
    try:
        seed = await ctx2.db.active_genome()
        prev = seed
        for v, op in enumerate(
            ("logic_evolve", "retrieval_evolve", "skill_evolve",
             "logic_evolve", "retrieval_evolve", "generalize"), 1,
        ):
            g = prev.model_copy(update={
                "version": v, "parent_version": prev.version,
                "operator": op,
            })
            g.prompt_fragments = [
                *g.prompt_fragments,
                PromptFragment(
                    id=f"m{v}", text=f"混合{v}" + "节" * v, origin="rsi",
                ),
            ]
            await ctx2.db.save_genome(g)
            prev = g
        result2 = await doctor(ctx2)
        assert not any(
            "算子集中度" in f["message"] for f in result2["findings"]
        )
    finally:
        await ctx2.close()


# ---------------------------------------------------------------------------
# ⑧ CAT_ERROR / CAT_REVIEW 遥测接线
# ---------------------------------------------------------------------------

async def test_error_and_review_events(ctx, monkeypatch):
    """后台 dream 失败 → events.jsonl 落 category=error；审查自动执行
    → 落 category=review。"""
    # a) dream 内部异常 → _guarded_dream 捕获并 emit CAT_ERROR
    async def _boom(*args, **kwargs):
        raise RuntimeError("u3 dream boom")

    monkeypatch.setattr(ctx.engine._improver, "diagnose", _boom)
    await ctx.engine._guarded_dream(auto=True)
    errors = ctx.events.read(category="error", limit=10)
    assert errors, "dream 失败应落 CAT_ERROR 事件"
    assert any("u3 dream boom" in str(e["payload"]) for e in errors)
    monkeypatch.undo()

    # b) review_session 自动执行提议 → CAT_REVIEW（action+resolution）
    # 舞台：6 条 trace_verified L3_core 超合计预算 → demote_l3 置信
    # 0.9 > auto_accept_confidence=0.85 → 自动执行（merge 提议只有
    # 0.8 会 pending，这里要的是确定走 auto 的动作）
    from rsi_mem.memory.guards import set_verification

    for i in range(6):
        rec = await ctx.tiers.longterm.add(
            title=f"L3 档案{i}", layer="L3_core",
            content=f"档案{i}要点：" + "细节" * 150,
        )
        loaded = await ctx.db.get_memory(rec.id)
        set_verification(loaded, "trace_verified", ["trace:seed"])
        await ctx.db.update_memory(loaded)
    await ctx.consolidator.review_session(None, [])
    reviews = ctx.events.read(category="review", limit=30)
    assert reviews, "审查执行应落 CAT_REVIEW 事件"
    assert any(
        r["payload"]["action"] == "demote_l3"
        and r["payload"]["resolution"] == "auto_applied"
        for r in reviews
    ), [(r["payload"]["action"], r["payload"]["resolution"])
        for r in reviews]


# ---------------------------------------------------------------------------
# ⑨ render_hash 与外部审计包
# ---------------------------------------------------------------------------

async def test_render_hash(tmp_path):
    """部署基因组 scores 记 render_hash=sha256(fragment_text())[:16]；
    不同渲染面指纹不同（内容增量才改指纹）；doctor 输出审计包段。"""
    from rsi_mem.cli import doctor

    ctx = _u2_ctx(tmp_path, "u3_hash")
    try:
        await _seed_cost_stage(ctx)
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        report = await ctx.engine.dream(trigger="u3-h1")
        assert report["deployed"] is True
        genome = await ctx.db.active_genome()
        rh = genome.scores["render_hash"]
        assert isinstance(rh, str) and len(rh) == 16
        int(rh, 16)  # 16 位 hex
        assert rh == _render_hash(genome)

        # 渲染面语义：retrieval/SOP 类部署不改片段 → 指针稳定（v0 与
        # v1 同 hash 属预期）；内容增量（logic_evolve）才移动指纹
        seed = GenomeSpec.seed()
        changed = seed.model_copy(deep=True)
        changed.prompt_fragments.append(PromptFragment(
            id="h", text="新渲染片段", origin="rsi",
        ))
        assert _render_hash(seed) != _render_hash(changed)

        result = await doctor(ctx)
        audit = [
            f for f in result["findings"] if "外部审计包" in f["message"]
        ]
        assert audit, "doctor 应输出外部审计包发现项"
        rows = audit[0]["detail"]["rows"]
        assert len(rows) == 2 and {
            r["version"] for r in rows
        } == {0, 1}
        assert any(r["render_hash"] for r in rows)
    finally:
        await ctx.close()
