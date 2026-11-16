# -*- coding: utf-8 -*-
"""U1 显著性标注 +「记忆即预测」测试：写入、真实使用点结算与错误隔离。

覆盖：

- 标注写读 roundtrip 与脏数据容忍（tag_of → None 不抛）；
- 反向保障：内部检索（建链 / retriever.search / 回放）不回填预测；
  reasoning 注入后**仅注入 id** 结算 ``hit``；
- ``search_for_tool``（agent 主动 memory_search）命中即 ``hit``；
- settle 失败（db.update_memory 抛异常）绝不阻断注入区块；
- 规则兜底按内容归一化去重（恒定标题不误杀）；
- LLM salience 透传 / 脏 salience 记录照写、tag_of 返 None；
- promote 回填 ``promoted``；过期结局走 ``salience_expired`` 遥测；
- trace_id 接缝：单挂 OpsMemory 时 None 安全降级（pop 语义）。
"""
import asyncio
import json

import pytest

from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit

from rsi_mem.config import RSIMemConfig
from rsi_mem.connector import OpsMemoryMiddleware
from rsi_mem.context import RSIMemContext
from rsi_mem.memory.extract import TurnExtractor
from rsi_mem.memory.salience import (
    annotate,
    dream_priority,
    is_settled,
    tag_of,
)
from rsi_mem.rsi.replay import ReplayEvaluator
from rsi_mem.storage import (
    OutcomeRecord,
    SalienceTag,
    TraceRecord,
)

from .conftest import DummyChatModel


@pytest.fixture
async def ctx(tmp_path):
    config = RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem"))
    context = RSIMemContext(config)
    yield context
    await context.close()


def _inc(user_text: str, assistant_text: str) -> list[Msg]:
    return [
        Msg(name="user", role="user",
            content=[TextBlock(type="text", text=user_text)]),
        Msg(name="assistant", role="assistant",
            content=[TextBlock(type="text", text=assistant_text)]),
    ]


# ---------------------------------------------------------------------------
# ① 标注写读 roundtrip 与容忍
# ---------------------------------------------------------------------------

async def test_tag_roundtrip_and_tolerance(ctx):
    rec = await ctx.tiers.shortterm.record(
        session_id="s1", title="磁盘扩容策略",
        content="磁盘超 85% 先清理再扩容",
        source={"salience": SalienceTag(
            kind="strategy", rationale="处置入口",
            predicted_use="磁盘告警轮次", predicted_value=0.8,
        ).model_dump()},
    )
    tag = tag_of(await ctx.db.get_memory(rec.id))
    assert tag is not None
    assert tag.kind == "strategy" and tag.volatile is False
    assert tag.predicted_use == "磁盘告警轮次"
    assert tag.predicted_value == pytest.approx(0.8)
    assert tag.settled is False and tag.settled_outcome is None

    # 缺省标注：不带 salience 写入 → 默认 state
    rec2 = await ctx.tiers.shortterm.record(
        session_id="s1", title="t", content="c",
    )
    assert tag_of(await ctx.db.get_memory(rec2.id)).kind == "state"

    # 脏标注：非 dict / 字段类型错 → tag_of 返 None（记录照常存在）
    rec3 = await ctx.tiers.shortterm.record(
        session_id="s1", title="t2", content="c2",
        source={"salience": "oops"},
    )
    loaded3 = await ctx.db.get_memory(rec3.id)
    assert loaded3 is not None and tag_of(loaded3) is None
    rec4 = await ctx.tiers.shortterm.record(
        session_id="s1", title="t3", content="c3",
        source={"salience": {"kind": "strategy", "predicted_value": "high"}},
    )
    assert tag_of(await ctx.db.get_memory(rec4.id)) is None

    # dream_priority：strategy > state；volatile 显著降权
    assert dream_priority(
        SalienceTag(kind="strategy", predicted_value=0.8),
    ) > dream_priority(SalienceTag(kind="state", predicted_value=0.8))
    assert dream_priority(
        SalienceTag(kind="strategy", predicted_value=0.8),
    ) > dream_priority(
        SalienceTag(kind="strategy", predicted_value=0.8, volatile=True),
    )

    # annotate 部分更新 roundtrip（其余字段保留）
    annotate(rec, settled=True, settled_outcome="hit")
    tag2 = tag_of(rec)
    assert tag2.settled and tag2.settled_outcome == "hit"
    assert tag2.kind == "strategy" and tag2.rationale == "处置入口"


# ---------------------------------------------------------------------------
# ② 内部使用不回填；注入后仅注入 id 结算
# ---------------------------------------------------------------------------

async def test_internal_usage_never_settles_only_injection_does(ctx):
    tiers = ctx.tiers
    long_a = await tiers.longterm.add(
        title="核心交换机档案",
        content="CX-9 双机热备，主备同步延迟阈值 50ms。",
        source={"salience": SalienceTag(kind="env_fact").model_dump()},
    )
    long_b = await tiers.longterm.add(
        title="备份策略档案", content="Bacula 每周日全量备份，保留 4 周。",
        source={"salience": SalienceTag(kind="env_fact").model_dump()},
    )
    short = await tiers.shortterm.record(
        session_id="s1", title="Redis 内存告警",
        content="redis-prod 内存 90%，关联到核心交换机档案",
        kind="incident", confidence=0.8,
        link_query="核心交换机 CX-9 双机热备",
    )
    assert any(
        l.target_id == long_a.id for l in short.links
    ), "写入建链（内部检索）应发生"

    # 内部检索之二：直接 retriever.search（bench / 工具内部同路径）
    results = await tiers.retriever.search(
        "Redis 内存告警", tiers=("long", "short"),
    )
    assert results
    # 内部检索之三：回放评估（replay 内部对轨迹查询重跑检索）
    replay = ReplayEvaluator(ctx.config, ctx.db, tiers.retriever)
    trace = TraceRecord(session_id="s1", query="Redis 内存告警")
    trace.outcome = OutcomeRecord(score=0.7)
    genome = await ctx.db.active_genome()
    await replay.evaluate(genome, genome, [trace], validation_passed=True)

    for rec in (long_a, long_b, short):
        loaded = await ctx.db.get_memory(rec.id)
        assert not is_settled(loaded), "建链/检索/回放不应回填使用预测"

    # 真实使用：reasoning 注入
    block, used_ids = await tiers.reasoning_hint_block(
        query="Redis 内存告警", session_id="s1", genome=genome,
        exclude_sessions=("s1",),
    )
    assert block.strip() and used_ids
    settled_ids = {
        r.id for r in await ctx.db.list_memories(limit=100)
        if is_settled(r)
    }
    assert settled_ids == set(used_ids), "仅真实注入的 id 结算 hit"
    assert long_b.id not in settled_ids  # 未注入的长期条目不结算
    assert tag_of(await ctx.db.get_memory(short.id)).settled_outcome == "hit"


# ---------------------------------------------------------------------------
# ③-0 无标注不结算（复核修复：预测面契约）
# ---------------------------------------------------------------------------

async def test_settle_skips_untagged_records(ctx):
    """无标注不结算：settle 不凭空创建 settled=True 的默认标注。"""
    tiers = ctx.tiers
    untagged = await tiers.longterm.add(
        title="无标注档案", content="无标注内容：磁盘告警处置。",
    )
    tagged = await tiers.shortterm.record(
        session_id="s1", title="有标注", content="有标注内容：磁盘告警。",
        source={"salience": SalienceTag(
            kind="env_fact", predicted_value=0.7,
        ).model_dump()},
    )
    genome = await ctx.db.active_genome()
    block, used_ids = await tiers.reasoning_hint_block(
        query="磁盘告警处置", session_id="s1", genome=genome,
        exclude_sessions=("s1",),
    )
    assert untagged.id in used_ids and tagged.id in used_ids
    loaded_untagged = await ctx.db.get_memory(untagged.id)
    assert tag_of(loaded_untagged) is None, \
        "无标注记录不应被凭空创建 settled 标注"
    assert not is_settled(loaded_untagged)
    assert is_settled(await ctx.db.get_memory(tagged.id)), \
        "有标注记录正常结算"


# ---------------------------------------------------------------------------
# ③ 主动检索（memory_search）= 真实使用
# ---------------------------------------------------------------------------

async def test_tool_search_settles_hit(ctx):
    tiers = ctx.tiers
    rec = await tiers.shortterm.record(
        session_id="s1", title="磁盘处置",
        content="磁盘告警先 df -h 核实再清理日志",
    )
    results = await tiers.search_for_tool("磁盘告警", 5)
    assert any(r.record.id == rec.id for r in results)
    tag = tag_of(await ctx.db.get_memory(rec.id))
    assert tag is not None and tag.settled is True
    assert tag.settled_outcome == "hit"
    # 幂等：再次检索仍为 hit，不产生其它结局
    await tiers.search_for_tool("磁盘告警", 5)
    tag = tag_of(await ctx.db.get_memory(rec.id))
    assert tag.settled is True and tag.settled_outcome == "hit"


# ---------------------------------------------------------------------------
# ④ settle 失败绝不阻断注入
# ---------------------------------------------------------------------------

async def test_settle_failure_never_blocks_injection(ctx, monkeypatch):
    tiers = ctx.tiers
    await tiers.shortterm.record(
        session_id="s1", title="磁盘", content="磁盘告警处置记录",
    )
    await tiers.muscle.note(
        title="磁盘SOP", content="1. df -h 核实\n2. 清理日志",
        triggers=["磁盘"],
    )
    genome = await ctx.db.active_genome()

    async def _boom(m):
        raise RuntimeError("db down")

    monkeypatch.setattr(ctx.db, "update_memory", _boom)
    block, used_ids = await tiers.reasoning_hint_block(
        query="磁盘告警", session_id="s1", genome=genome,
    )
    assert block.strip() and used_ids, "settle 抛异常时注入区块必须完整"
    sys_block = await tiers.system_prompt_block(
        query="磁盘告警", genome=genome,
    )
    assert "磁盘SOP" in sys_block, "系统提示区块（含命中反射）不受影响"


# ---------------------------------------------------------------------------
# ⑤ 规则兜底去重
# ---------------------------------------------------------------------------

async def test_rule_fallback_dedup_by_content(ctx):
    extractor = TurnExtractor(ctx.config, ctx.tiers)

    first = await extractor.extract_turn(
        _inc("磁盘使用率 95% 怎么处理", "先清理日志再评估扩容"),
        session_id="s1", model=None,
    )
    assert len(first) == 2
    assert len(
        await ctx.db.list_memories(tier="short", session_id="s1"),
    ) == 2

    # 同增量二次写入：内容归一化命中 → 不增计数
    second = await extractor.extract_turn(
        _inc("磁盘使用率 95% 怎么处理", "先清理日志再评估扩容"),
        session_id="s1", model=None,
    )
    assert second == []
    assert len(
        await ctx.db.list_memories(tier="short", session_id="s1"),
    ) == 2

    # 不同内容不被恒定标题（本轮问题/本轮结论）误杀
    third = await extractor.extract_turn(
        _inc("CPU 使用率 100% 怎么处理", "先 top 定位进程"),
        session_id="s1", model=None,
    )
    assert len(third) == 2
    assert len(
        await ctx.db.list_memories(tier="short", session_id="s1"),
    ) == 4

    # 兜底产物默认 kind=state / volatile=True
    for rec in third:
        tag = tag_of(rec)
        assert tag is not None
        assert tag.kind == "state" and tag.volatile is True


# ---------------------------------------------------------------------------
# ⑥ LLM salience 透传与脏数据
# ---------------------------------------------------------------------------

async def test_llm_salience_passthrough_and_dirty(ctx):
    extractor = TurnExtractor(ctx.config, ctx.tiers)
    good = json.dumps([{
        "title": "磁盘扩容策略", "content": "磁盘超 85% 先清理再扩容",
        "kind": "fact", "confidence": 0.8,
        "salience": {"kind": "strategy", "volatile": False,
                     "rationale": "处置入口", "predicted_use": "磁盘告警轮次",
                     "predicted_value": 0.8},
    }], ensure_ascii=False)
    recs = await extractor.extract_turn(
        _inc("磁盘快满了", "按策略先清理再扩容"),
        session_id="s1", model=DummyChatModel([good]),
    )
    assert len(recs) == 1
    tag = tag_of(await ctx.db.get_memory(recs[0].id))
    assert tag is not None and tag.kind == "strategy"
    assert tag.predicted_use == "磁盘告警轮次"
    assert tag.predicted_value == pytest.approx(0.8)

    dirty = json.dumps([{
        "title": "脏标注", "content": "salience 不是 dict 的候选",
        "kind": "fact", "confidence": 0.6, "salience": "oops",
    }], ensure_ascii=False)
    recs2 = await extractor.extract_turn(
        _inc("再来一轮", "记录一条脏标注记忆"),
        session_id="s2", model=DummyChatModel([dirty]),
    )
    assert len(recs2) == 1, "脏 salience 不应阻断写入"
    loaded = await ctx.db.get_memory(recs2[0].id)
    assert loaded is not None and tag_of(loaded) is None


# ---------------------------------------------------------------------------
# ⑦ promote 回填 promoted
# ---------------------------------------------------------------------------

async def test_promote_settles_promoted(ctx):
    short = await ctx.tiers.shortterm.record(
        session_id="s1", title="核心交换机型号", content="核心是 CX-9",
        kind="fact", confidence=0.9,
    )
    out = await ctx.tiers.shortterm.promote(short.id, layer="L1_atom")
    assert out is not None
    tag = tag_of(await ctx.db.get_memory(short.id))
    assert tag is not None and tag.settled is True
    assert tag.settled_outcome == "promoted"


# ---------------------------------------------------------------------------
# ⑧ 过期结局走遥测（不写库、不进 predictions）
# ---------------------------------------------------------------------------

async def test_expired_outcome_goes_to_telemetry(ctx):
    rec = await ctx.tiers.shortterm.record(
        session_id="s1", title="临时状态", content="马上过期的易变状态",
    )
    loaded = await ctx.db.get_memory(rec.id)
    loaded.expires_at = 1000.0
    await ctx.db.update_memory(loaded)

    stats = await ctx.tiers.maintenance()
    assert stats["expired_short_term"] == 1
    assert await ctx.db.get_memory(rec.id) is None

    events = ctx.events.read(category="memory", limit=20)
    expired = [
        e for e in events if e["payload"].get("event") == "salience_expired"
    ]
    assert expired, "过期结局应落 salience_expired 遥测事件"
    items = expired[0]["payload"]["items"]
    entry = next(it for it in items if it["id"] == rec.id)
    assert entry["kind"] == "state"  # record 默认标注随事件落档
    assert entry["priority"] == pytest.approx(0.125)  # 0.25 × 0.5
    # 不写 predictions 表（避免污染世界模型 MAE）
    assert await ctx.db.load_predictions() == []


# ---------------------------------------------------------------------------
# ⑨ trace_id 接缝：单挂 OpsMemory 时 None 安全降级
# ---------------------------------------------------------------------------

async def test_ops_only_trace_id_none(tmp_path):
    config = RSIMemConfig(data_dir=str(tmp_path / "ops_only"))
    ctx = RSIMemContext(config)
    try:
        # 接缝本身：note → pop（单值、取走即清）
        assert ctx.engine.last_trace_id("nope") is None
        ctx.engine.note_trace_finished("s9", "tid-1")
        assert ctx.engine.last_trace_id("s9") == "tid-1"
        assert ctx.engine.last_trace_id("s9") is None  # pop 语义

        mem_mw = OpsMemoryMiddleware(ctx)
        agent = Agent(
            name="SoloAgent", system_prompt="你是运维助手。",
            model=DummyChatModel(), toolkit=Toolkit(),
            middlewares=[mem_mw],
        )
        await agent.reply(Msg(
            name="user", role="user",
            content=[TextBlock(type="text", text="磁盘快满了怎么办")],
        ))
        await asyncio.sleep(0.6)  # 等待后台写回（不阻塞回复）
        shorts = await ctx.db.list_memories(tier="short", limit=10)
        assert shorts, "单挂 OpsMemory 回合写回应照常发生"
        for s in shorts:
            assert s.source.get("trace_id") is None  # 无 RSI 上报 → None
            assert tag_of(s) is not None
        assert ctx.engine.last_trace_id(agent.state.session_id) is None
    finally:
        await ctx.close()
