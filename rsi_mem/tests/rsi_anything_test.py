# -*- coding: utf-8 -*-
"""RSI-Anything 优化项测试：插件化、预测器、遥测、基准与 CLI。"""
import asyncio
import json
import os

import pytest

from rsi_mem.config import RSIMemConfig
from rsi_mem.context import RSIMemContext
from rsi_mem.rsi.objectives import ObjectiveVector
from rsi_mem.rsi.plugins import PluginRegistry, build_default_registry
from rsi_mem.rsi.predictor import (
    CandidateFeatures,
    OutcomePredictor,
    is_conservative_operator,
)
from rsi_mem.storage import (
    GenomeSpec,
    OutcomeRecord,
    ToolCallNode,
    TraceRecord,
)
from rsi_mem.telemetry import LangfuseBridge, LocalEventLog

from .fixtures import BrokenPlugin, FixedPlanPlugin, NotAPlugin


@pytest.fixture
async def ctx(tmp_path):
    config = RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem"))
    config.rsi.dream_every_n_traces = 100
    context = RSIMemContext(config)
    yield context
    await context.close()


async def _mk_trace(ctx, query, score=0.6, *, error=False, tokens=0):
    from rsi_mem.storage import ModelCallStat

    t = TraceRecord(session_id="s1", query=query, digest="ok")
    t.tool_calls = [
        ToolCallNode(name="bash", input_digest="x", state="ok"),
    ]
    if error:
        t.tool_calls.append(
            ToolCallNode(
                name="restart_service", input_digest="x", state="error",
            ),
        )
    if tokens:
        t.model_calls = [
            ModelCallStat(model="m", input_tokens=tokens, output_tokens=1),
        ]
    t.outcome = OutcomeRecord(score=score)
    await ctx.db.upsert_trace(t)
    return t


# ---------------------------------------------------------------------------
# 插件化
# ---------------------------------------------------------------------------

async def test_registry_load_and_tolerate_bad_specs(ctx):
    registry = PluginRegistry()
    loaded = registry.load_from_config([
        "rsi_mem.tests.fixtures:FixedPlanPlugin",
        "rsi_mem.tests.fixtures:NotAPlugin",      # 非 RSIPlugin → 拒绝
        "no.such.module:Foo",                     # 导入失败 → 拒绝
        "rsi_mem.tests.fixtures",                 # 缺 attr → 拒绝
    ])
    assert loaded == ["fixed-plan"]
    assert len(registry) == 1
    # 坏插件运行期异常也不拖垮
    registry.add(BrokenPlugin())
    plans = await registry.propose_all({"x": 1}, GenomeSpec.seed(), [])
    assert all(p.operator != "skill_evolve" for _, p in plans)
    # 证据标注合并（带插件名前缀）
    tags = registry.evidence_tags()
    assert "plugin:fixed-plan:fixture_signal" in tags


async def test_default_registry_contains_builtin(ctx):
    registry = build_default_registry(ctx.config, ctx.engine._improver)
    assert "builtin-improver" in registry.names()


async def test_custom_plugin_flows_into_dream(ctx, tmp_path):
    """配置装载的插件计划应进入 dream 候选评估（origin 溯源）。"""
    cfg = RSIMemConfig(data_dir=str(tmp_path / "d2"))
    cfg.rsi.plugins = ["rsi_mem.tests.fixtures:FixedPlanPlugin"]
    cfg.rsi.candidates = 6  # 给插件留预算
    context = RSIMemContext(cfg)
    try:
        # 制造诊断素材（有轨迹 → 诊断非 None）
        for i in range(3):
            await _mk_trace(context, f"磁盘告警 {i}", 0.7)
        report = await context.engine.dream(trigger="test")
        origins = [
            p.get("origin_plugin")
            for p in report.get("candidate_predictions", [])
        ]
        assert "fixed-plan" in origins, origins
        assert "fixed-plan" in report["plugins"]
        # 插件生命周期（首个 dream 懒启动）
        plugin = context.engine.registry.get("fixed-plan")
        assert plugin.started is True
    finally:
        await context.close()
        assert plugin.stopped is True


# ---------------------------------------------------------------------------
# 预测器（世界模型）与保守门
# ---------------------------------------------------------------------------

async def test_predictor_prior_then_calibrates(ctx):
    predictor = OutcomePredictor(ctx.db)
    features = CandidateFeatures(
        operator="skill_evolve", sop_count_delta=1,
    )
    pred = predictor.predict(features)
    assert pred.samples == 0 and pred.uncertainty == 0.8  # 先验
    assert pred.quality_delta == pytest.approx(0.05)  # skill 先验

    for realized_q in (0.7, 0.75, 0.8):
        predictor.update(
            features,
            ObjectiveVector(quality=realized_q),
            ObjectiveVector(quality=0.65),
        )
    after = predictor.predict(features)
    assert after.samples == 3
    assert after.uncertainty < 0.8
    # 实测增量均值 (0.05+0.10+0.15)/3 = 0.10
    assert after.quality_delta == pytest.approx(0.10, abs=0.01)
    assert predictor.mae() is not None and predictor.sample_count() == 3
    # 可靠性门：样本不足时保持"可靠"（允许探索）
    assert predictor.reliable(min_samples=10, mae_threshold=0.01)


async def test_conservative_gate_blocks_risky_plans(ctx):
    """预测不可靠 + 样本足够 → 保守模式拦截 skill_evolve。"""
    # 制造高错误率轨迹（触发 skill_evolve 计划）
    for i in range(4):
        await _mk_trace(ctx, f"磁盘告警 {i}", 0.5, error=True)
    # 人为注入"预测很差"的历史统计 → reliable=False
    from rsi_mem.rsi.predictor import OperatorStats

    stats = OperatorStats(n=5, abs_err_sum=5 * 0.4)
    ctx.engine.predictor._stats["skill_evolve"] = stats
    assert not ctx.engine.predictor.reliable(
        min_samples=ctx.config.rsi.predictor_min_samples,
        mae_threshold=ctx.config.rsi.predictor_mae_threshold,
    )
    report = await ctx.engine.dream(trigger="gate-test")
    assert report["conservative_mode"] is True
    assert report["dropped_by_gate"] >= 1
    # 保守模式下部署的（若有）只能是保守算子
    if report["deployed"]:
        assert is_conservative_operator(report["operator"])


async def test_predictions_persisted_with_realized(ctx):
    for i in range(4):
        await _mk_trace(ctx, f"磁盘告警 {i}", 0.5, error=True)
    await ctx.engine.dream(trigger="pred-test")
    preds = await ctx.db.load_predictions()
    assert preds, "dream 应写入预测记录"
    realized = [p for p in preds if p["realized"] is not None]
    assert realized, "回放结果应回填为观测"
    assert report_has_prediction(ctx)


async def report_has_prediction(ctx) -> bool:
    raw = await ctx.db.get_meta("last_dream_report")
    report = json.loads(raw)
    assert report["predictor"]["samples"] >= 1
    assert report["evidence"]["legend"]["1"] == "E1 学习模型预测"
    return True


# ---------------------------------------------------------------------------
# 遥测：本地事件日志 + langfuse 桥
# ---------------------------------------------------------------------------

async def test_local_event_log_roundtrip(tmp_path):
    log = LocalEventLog(str(tmp_path / "logs" / "events.jsonl"))
    await log.emit("trace", {"score": 0.7})
    await log.emit("dream", {"deployed": True}, level="warn")
    log.emit_sync("plugin", {"name": "x"})
    all_events = log.read(limit=10)
    assert len(all_events) == 3
    only_warn = log.read(level="warn")
    assert len(only_warn) == 1 and only_warn[0]["category"] == "dream"
    # 禁用时不落盘
    off = LocalEventLog(str(tmp_path / "off.jsonl"), enabled=False)
    await off.emit("x", {})
    assert off.read() == []


async def test_langfuse_noop_when_not_configured(ctx):
    assert ctx.langfuse.active is False  # 无密钥 → no-op
    with ctx.langfuse.observe("job"):
        pass  # 不抛异常


def test_langfake_client_receives_events():
    class FakeClient:
        def __init__(self):
            self.events = []

        def emit_event(self, name, payload=None, **kw):
            self.events.append((name, payload))

    fake = FakeClient()
    bridge = LangfuseBridge(enabled=False)
    bridge.inject_client(fake)
    assert bridge.active
    bridge.track("rsi_mem:dream", {"deployed": True})
    assert fake.events == [("rsi_mem:dream", {"deployed": True})]
    with bridge.observe("memory_extract"):
        pass
    assert fake.events[-1][0] == "rsi_mem:llm_job"
    assert fake.events[-1][1]["job"] == "memory_extract"


async def test_engine_emits_dream_and_trace_events(ctx):
    await _mk_trace(ctx, "磁盘告警", 0.7)
    await ctx.engine.dream(trigger="evt-test")
    dreams = ctx.events.read(category="dream", limit=10)
    assert any(e["payload"].get("phase") == "start" for e in dreams)
    assert any(e["payload"].get("phase") == "end" for e in dreams)


# ---------------------------------------------------------------------------
# 诊断证据分级
# ---------------------------------------------------------------------------

async def test_diagnosis_annotated_with_evidence(ctx):
    for i in range(3):
        await _mk_trace(ctx, f"q{i}", 0.6)
    traces = await ctx.db.list_traces()
    diagnosis = await ctx.engine._improver.diagnose(traces)
    assert diagnosis is not None
    from rsi_mem.rsi.evidence import annotate_diagnosis

    annotated = annotate_diagnosis(diagnosis)
    assert annotated["evidence"]["stats"] == 3  # E3
    assert annotated["evidence"]["error_tools"] == 0  # E0


# ---------------------------------------------------------------------------
# 10 轮基准与 CLI
# ---------------------------------------------------------------------------

async def test_bench_ten_rounds_non_degrading(tmp_path):
    from rsi_mem.bench.rounds import run_bench

    cfg = RSIMemConfig(data_dir=str(tmp_path / "bench"))
    context = RSIMemContext(cfg)
    try:
        report = await run_bench(
            context, rounds=10, traces_per_round=3,
        )
        assert len(report["rows"]) == 10
        s = report["summary"]
        assert s["deploys"] >= 1, "至少一轮应部署改进"
        assert s["non_degraded"] is True
        assert s["prediction_samples"] >= 5, "预测-观测对应持续积累"
        first = report["rows"][0]
        last = report["rows"][-1]
        assert last["quality_after"] >= first["quality_after"] - 0.02
        assert os.path.isfile(report["path"])
        # 报告可序列化往返
        json.loads(json.dumps(report))
    finally:
        await context.close()


async def test_cli_doctor_and_report(tmp_path, capsys):
    from rsi_mem import cli

    d = str(tmp_path / "cli")
    cfg = RSIMemConfig(data_dir=d)
    context = RSIMemContext(cfg)
    for i in range(3):
        await _mk_trace(context, f"磁盘 {i}", 0.6, error=(i == 0))
    await context.engine.dream(trigger="cli")
    await context.close()

    # cli.main 是同步入口（内部 asyncio.run）→ 在无事件循环的线程里跑
    rc = await asyncio.to_thread(cli.main, ["doctor", "--data-dir", d])
    assert rc == 0
    out = capsys.readouterr().out
    assert "findings" in out and "evidence_legend" in out
    rc = await asyncio.to_thread(cli.main, ["report", "--data-dir", d])
    assert rc == 0
    out = capsys.readouterr().out
    assert "genome_history" in out
