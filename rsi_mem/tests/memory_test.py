# -*- coding: utf-8 -*-
"""memory 层测试：三层存储、链接、检索、固化与提取。"""
import os

import pytest

from rsi_mem.config import RSIMemConfig
from rsi_mem.context import RSIMemContext
from rsi_mem.memory import TierManager
from rsi_mem.memory.consolidate import Consolidator, summarize_feedback
from rsi_mem.memory.extract import TurnExtractor
from rsi_mem.storage import (
    MemoryRecord,
    MirrorStore,
    OutcomeRecord,
    PendingReview,
    RSIMemDatabase,
    ToolCallNode,
    TraceRecord,
)


@pytest.fixture
async def ctx(tmp_path):
    config = RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem"))
    context = RSIMemContext(config)
    yield context
    await context.close()


async def test_muscle_match_and_mirror(ctx):
    tiers = ctx.tiers
    sop = await tiers.muscle.note(
        title="磁盘告警处置SOP",
        content="1. df -h 确认\n2. 清理 /var/log\n3. 评估扩容",
        triggers=["磁盘", "re:disk\\s*full", "tool:bash"],
    )
    hits = await tiers.muscle.match("线上磁盘快满了怎么办")
    assert any(h.id == sop.id for h in hits)
    hits = await tiers.muscle.match("n/a", tool_hint="bash")
    assert any(h.id == sop.id for h in hits)
    hits = await tiers.muscle.match("完全无关的话题")
    assert not any(h.id == sop.id for h in hits)

    # 镜像为可加载 SKILL.md
    skill_md = os.path.join(
        ctx.mirror.muscle_dir, "磁盘告警处置SOP", "SKILL.md",
    )
    assert os.path.isfile(skill_md)
    with open(skill_md, encoding="utf-8") as f:
        head = f.read(200)
    assert "磁盘告警处置SOP" in head

    # 渲染注入区块
    rendered = tiers.muscle.render_hits(hits if hits else [sop])
    assert "肌肉记忆" in rendered


async def test_short_term_links_to_long_term(ctx):
    tiers = ctx.tiers
    # 长期条目刻意不含查询关键词（验证链接展开而非直接命中）
    long_rec = await tiers.longterm.add(
        title="核心交换机档案",
        content="CX-9 双机热备，主备同步延迟阈值 50ms。",
    )
    short = await tiers.shortterm.record(
        session_id="s1",
        title="Redis 内存告警",
        content="redis-prod 内存 90%，关联到核心交换机档案",
        kind="incident",
        confidence=0.8,
        link_query="核心交换机 CX-9 双机热备",
    )
    assert any(
        l.target_id == long_rec.id and l.kind == "related"
        for l in short.links
    ), "短期记忆应自动建立指向长期记忆的索引链接"

    # 检索命中短期时展开长期链接
    results = await tiers.retriever.search(
        "Redis 内存告警", tiers=("long", "short"),
    )
    ids = {r.record.id for r in results}
    assert short.id in ids
    assert long_rec.id in ids, "应通过短期记忆的链接浮出长期条目"
    via = {r.record.id: r.via for r in results}
    assert via[long_rec.id] == "link"
    linked = [r for r in results if r.record.id == long_rec.id]
    assert linked[0].parent_id == short.id


async def test_short_term_ttl_and_lru(ctx):
    tiers = ctx.tiers
    rec = await tiers.shortterm.record(
        session_id="s1", title="临时", content="马上过期", kind="todo",
    )
    # 手动把过期时间改成过去
    rec.expires_at = 1000.0
    await ctx.db.update_memory(rec)
    assert await tiers.shortterm.cleanup() == 1
    assert await ctx.db.get_memory(rec.id) is None


async def test_promote_short_to_long(ctx):
    tiers = ctx.tiers
    short = await tiers.shortterm.record(
        session_id="s1", title="核心交换机型号",
        content="核心是 CX-9，双机热备", kind="fact", confidence=0.9,
    )
    promoted = await tiers.shortterm.promote(short.id, layer="L1_atom")
    assert promoted is not None
    assert promoted.tier == "long" and promoted.status == "active"
    assert any(
        l.kind == "provenance" for l in promoted.links
    ), "晋升后应保留溯源链"
    assert await ctx.db.get_memory(short.id) is not None  # id 未变


async def test_propose_tiered_confirmation(ctx):
    tiers = ctx.tiers
    # 高置信普通事实（promote 类）→ 自动执行
    short = await tiers.shortterm.record(
        session_id="s1", title="k8s 版本", content="1.31", kind="fact",
        confidence=0.95,
    )
    r1 = PendingReview(
        action="promote_longterm",
        memory_id=short.id,
        payload={"memory_id": short.id, "layer": "L1_atom"},
        reason="高置信事实", confidence=0.95,
    )
    assert await tiers.propose(r1) == "auto_applied"

    # SOP → 一律进队列
    r2 = PendingReview(
        action="create_muscle",
        payload={"title": "重启SOP", "content": "systemctl restart api",
                 "triggers": ["重启"]},
        reason="提议", confidence=0.99,
    )
    assert await tiers.propose(r2) == "pending"
    pend = await tiers.pending()
    assert len(pend) == 1

    # 安全敏感内容 → 进队列
    r3 = PendingReview(
        action="create_longterm",
        payload={"title": "清理", "content": "执行 rm -rf /var/lib"},
        reason="危险", confidence=0.99,
    )
    assert await tiers.propose(r3) == "pending"

    # 用户确认执行 create_muscle
    await tiers.resolve(pend[0].id, approved=True)
    muscles = await ctx.db.list_memories(tier="muscle", status="active")
    assert any(m.title == "重启SOP" for m in muscles)


async def test_consolidator_distills_sop(ctx):
    traces = []
    for i in range(4):
        t = TraceRecord(session_id="s1", query=f"磁盘告警 {i}")
        t.tool_calls = [
            ToolCallNode(name="bash", input_digest="df -h", state="ok"),
            ToolCallNode(name="read_file", input_digest="/var/log", state="ok"),
        ]
        t.outcome = OutcomeRecord(score=0.8)
        await ctx.db.upsert_trace(t)
        traces.append(t)
    consolidator = Consolidator(ctx.config, ctx.tiers)
    proposals = await consolidator.review_session("s1", traces)
    sop_proposals = [p for p in proposals if p.action == "create_muscle"]
    assert sop_proposals, "重复工具序列应蒸馏出 SOP 提议"
    # create_muscle 永远待确认
    assert all(p.resolved_by is None for p in sop_proposals)
    pend = await ctx.tiers.pending()
    assert any(p.action == "create_muscle" for p in pend)


async def test_extractor_rule_fallback(ctx):
    from agentscope.message import Msg, TextBlock

    increment = [
        Msg(name="user", role="user",
            content=[TextBlock(type="text", text="磁盘使用率 95% 怎么处理")]),
        Msg(name="assistant", role="assistant",
            content=[TextBlock(type="text", text="先清理日志再评估扩容")]),
    ]
    extractor = TurnExtractor(ctx.config, ctx.tiers)
    records = await extractor.extract_turn(
        increment, session_id="sx", model=None,
    )
    assert records, "规则兜底也应产出短期记忆"
    assert any("磁盘" in (r.title + r.content) for r in records)


async def test_summarize_feedback(ctx):
    traces = []
    for fb, score in [(1, 0.8), (-1, 0.3), (None, 0.6)]:
        t = TraceRecord(query="q")
        t.outcome = OutcomeRecord(score=score, feedback=fb)
        traces.append(t)
    s = summarize_feedback(traces)
    assert s["count"] == 3
    assert s["positive_feedback"] == 1 and s["negative_feedback"] == 1
    assert 0.5 < s["mean_outcome"] < 0.6
