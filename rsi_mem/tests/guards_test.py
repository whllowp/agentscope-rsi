# -*- coding: utf-8 -*-
"""U2 记忆污染防御测试：验证分级 + 矛盾检测 + 隔离（含肌肉层与代理）。

覆盖：l3_eligible 矩阵（不依赖 confidence）、负反馈先于 score 的判定
顺序、创建轨迹不验证 / 后续成功轨迹验证、默认阈值自动执行（手写强制
人工）、SOP 蒸馏 trace_ids → 👎 隔离退出注入池与镜像、L3 驻留计数、
漂移再验证、矛盾正/反例、demote_l3 镜像与链接、未知 action。
"""
import os

import pytest

from rsi_mem.context import RSIMemContext
from rsi_mem.memory.guards import (
    is_injectable,
    l3_eligible,
    mark_stale,
    set_verification,
    verification_of,
)
from rsi_mem.rsi.trace import attach_feedback
from rsi_mem.storage import (
    MemoryRecord,
    OutcomeRecord,
    PendingReview,
    ToolCallNode,
    TraceRecord,
)


@pytest.fixture
async def ctx(tmp_path):
    from rsi_mem.config import RSIMemConfig

    context = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem")),
    )
    yield context
    await context.close()


def _l3(source: dict | None = None, **kw) -> MemoryRecord:
    return MemoryRecord(
        tier="long", layer="L3_core", source=source or {}, **kw,
    )


async def _review(ctx, traces=None):
    return await ctx.consolidator.review_session(None, traces or [])


def _skill_dirs(mirror) -> list[str]:
    root = mirror.muscle_dir
    return [
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "SKILL.md"))
    ]


# ---------------------------------------------------------------------------
# ① l3_eligible 矩阵（不依赖 confidence）
# ---------------------------------------------------------------------------

def test_l3_eligible_matrix_and_tolerance():
    # unverified 无 origin：confidence 再高也不准入
    rec = _l3(confidence=0.95)
    assert is_injectable(rec) and not l3_eligible(rec)

    # 用户手写（memory_note / user）二选一准入
    assert l3_eligible(_l3(source={"origin": "memory_note"},
                           confidence=0.3))
    assert l3_eligible(_l3(source={"origin": "user"}))

    # trace_verified（无 origin）
    rec = _l3()
    set_verification(rec, "trace_verified", ["trace:t1"])
    assert l3_eligible(rec)
    assert verification_of(rec).state == "trace_verified"

    # refuted / archived → 一票否决
    rec = _l3(source={"origin": "memory_note"})
    set_verification(rec, "refuted", ["quarantine:thumbs_down"])
    assert not is_injectable(rec) and not l3_eligible(rec)
    assert not l3_eligible(MemoryRecord(
        tier="long", layer="L3_core", status="archived",
        source={"origin": "memory_note"},
    ))

    # 缺省验证态 & 脏数据容忍
    assert verification_of(_l3()).state == "unverified"
    dirty = _l3(source={"verification": "oops"})
    assert verification_of(dirty).state == "unverified"

    # mark_stale 重置 unverified 且保留 evidence 历史
    rec = _l3()
    set_verification(rec, "trace_verified", ["trace:t1"])
    mark_stale(rec, "drift")
    vs = verification_of(rec)
    assert vs.state == "unverified"
    assert "trace:t1" in vs.evidence
    assert any(e.startswith("stale:") for e in vs.evidence)


async def test_core_profile_filters_unverified(ctx):
    await ctx.tiers.longterm.add(
        title="手写档案", content="内容。", layer="L3_core",
        source={"origin": "memory_note"},
    )
    await ctx.tiers.longterm.add(
        title="自动档案", content="内容。", layer="L3_core",
    )
    cores = await ctx.tiers.retriever.core_profile()
    assert [c.title for c in cores] == ["手写档案"], \
        "L3 注入面只留 trace_verified / 用户手写"


# ---------------------------------------------------------------------------
# ② 负反馈先于 score：0.9 分轨迹被 👎 后 → 隔离而非验证
# ---------------------------------------------------------------------------

async def test_negative_feedback_beats_score(ctx):
    l3 = await ctx.tiers.longterm.add(
        title="边界档案", content="内容。", layer="L3_core",
    )
    t = TraceRecord(session_id="s1", query="q", digest="d")
    t.used_memory_ids = [l3.id]
    t.outcome = OutcomeRecord(score=0.9)
    await ctx.db.upsert_trace(t)
    await attach_feedback(ctx.db, t.id, -1)  # 0.9-0.3=0.6, feedback=-1

    proposals = await ctx.consolidator.review_session(
        "s1", await ctx.db.list_traces(limit=5),
    )
    q = [p for p in proposals if p.action == "quarantine_memory"]
    assert q, "负反馈关联应产生隔离提议"
    assert q[0].resolved_by == "auto", "0.9>0.85 默认自动执行"
    loaded = await ctx.db.get_memory(l3.id)
    vs = verification_of(loaded)
    assert vs.state == "refuted"
    assert loaded.status == "archived"
    assert not is_injectable(loaded)


# ---------------------------------------------------------------------------
# ③ 创建轨迹不验证 / 后续成功轨迹注入使用验证
# ---------------------------------------------------------------------------

async def test_creation_trace_does_not_verify_usage_does(ctx):
    t1 = TraceRecord(session_id="s1", query="q1", digest="d1")
    t1.outcome = OutcomeRecord(score=0.9)
    await ctx.db.upsert_trace(t1)
    rec = await ctx.tiers.longterm.add(
        title="创建轨迹档案", content="内容。", layer="L3_core",
        source={"trace_id": t1.id},
    )
    t1.used_memory_ids = [rec.id]
    await ctx.db.upsert_trace(t1)

    await ctx.consolidator.review_session("s1", [t1])
    assert verification_of(
        await ctx.db.get_memory(rec.id),
    ).state == "unverified", "创建轨迹的回合级 score 不单独授予"

    t2 = TraceRecord(session_id="s1", query="q2", digest="d2")
    t2.used_memory_ids = [rec.id]
    t2.outcome = OutcomeRecord(score=0.8)
    await ctx.db.upsert_trace(t2)
    await ctx.consolidator.review_session("s1", [t1, t2])
    vs = verification_of(await ctx.db.get_memory(rec.id))
    assert vs.state == "trace_verified"
    assert f"trace:{t2.id}" in vs.evidence


# ---------------------------------------------------------------------------
# ④ 默认配置自动执行（auto_accept_confidence=0.85 不改）
# ---------------------------------------------------------------------------

async def test_default_threshold_auto_executes(ctx):
    assert ctx.config.review.auto_accept_confidence == pytest.approx(0.85)

    victim = await ctx.tiers.shortterm.record(
        session_id="s1", title="污染", content="脏数据",
    )
    r1 = PendingReview(
        action="quarantine_memory", memory_id=victim.id,
        payload={"memory_id": victim.id, "origin": ""},
        reason="负反馈证据", confidence=0.9,
    )
    assert await ctx.tiers.propose(r1) == "auto_applied"
    assert (await ctx.db.get_memory(victim.id)).status == "archived"

    l3 = await ctx.tiers.longterm.add(
        title="驻留", content="内容。", layer="L3_core",
    )
    r2 = PendingReview(
        action="demote_l3", memory_id=l3.id,
        payload={"memory_id": l3.id},
        reason="驻留上限", confidence=0.9,
    )
    assert await ctx.tiers.propose(r2) == "auto_applied"
    assert (await ctx.db.get_memory(l3.id)).layer == "L1_atom"

    hand = await ctx.tiers.longterm.add(
        title="手写", content="用户手写档案。", layer="L3_core",
        source={"origin": "memory_note"},
    )
    r3 = PendingReview(
        action="quarantine_memory", memory_id=hand.id,
        payload={"memory_id": hand.id, "origin": "memory_note"},
        reason="负反馈证据", confidence=0.9,
    )
    assert await ctx.tiers.propose(r3) == "pending", \
        "origin=memory_note 手写隔离强制人工"
    assert (await ctx.db.get_memory(hand.id)).status == "active"
    await ctx.tiers.resolve(r3.id, approved=False)


# ---------------------------------------------------------------------------
# ⑤ 肌肉层防线：蒸馏 SOP 带 trace_ids → 👎 → 隔离退出注入池与镜像
# ---------------------------------------------------------------------------

async def test_muscle_sop_pollution_quarantine(ctx):
    traces = []
    for i in range(3):
        t = TraceRecord(session_id="s1", query=f"磁盘 告警 {i}")
        t.tool_calls = [
            ToolCallNode(name="bash", input_digest="df -h", state="ok"),
            ToolCallNode(name="read_file", input_digest="logs",
                         state="ok"),
        ]
        t.outcome = OutcomeRecord(score=0.8)
        await ctx.db.upsert_trace(t)
        traces.append(t)
    await attach_feedback(ctx.db, traces[0].id, -1)  # 来源轨迹 👎

    refreshed = await ctx.db.list_traces(limit=10)
    proposals = await ctx.consolidator.review_session("s1", refreshed)
    sop = next(p for p in proposals if p.action == "create_muscle")
    muscle = await ctx.tiers.resolve(sop.id, approved=True)
    assert muscle is not None
    assert set(muscle.source.get("trace_ids") or []) == {
        t.id for t in refreshed
    }, "蒸馏 SOP 应落 source.trace_ids（来源轨迹关联）"
    assert _skill_dirs(ctx.mirror)
    assert any(
        h.id == muscle.id
        for h in await ctx.tiers.muscle.match("磁盘 告警")
    )

    # 只带 👎 轨迹复审 → 隔离自动执行 → 退出注入池 + 镜像清理
    bad = next(t for t in refreshed if t.outcome.feedback == -1)
    proposals2 = await ctx.consolidator.review_session("s1", [bad])
    q = [p for p in proposals2 if p.action == "quarantine_memory"]
    assert q and q[0].resolved_by == "auto"
    assert not any(
        h.id == muscle.id
        for h in await ctx.tiers.muscle.match("磁盘 告警")
    ), "隔离后不应再命中注入"
    assert _skill_dirs(ctx.mirror) == [], "SKILL.md 镜像应清理"
    loaded = await ctx.db.get_memory(muscle.id)
    assert loaded.status == "archived"
    assert verification_of(loaded).state == "refuted"


# ---------------------------------------------------------------------------
# ⑥ L3 驻留计数：三次审查未验证零命中 → demote_l3；验证/命中清零
# ---------------------------------------------------------------------------

async def test_l3_residency_counter(ctx):
    l3 = await ctx.tiers.longterm.add(
        title="驻留档案", content="内容。", layer="L3_core",
    )
    await _review(ctx)
    assert (await ctx.db.get_memory(l3.id)).source["l3_reviews"] == 1
    p2 = await _review(ctx)
    assert (await ctx.db.get_memory(l3.id)).source["l3_reviews"] == 2
    assert not [p for p in p2 if p.action == "demote_l3"]

    p3 = await _review(ctx)
    demos = [p for p in p3 if p.action == "demote_l3"]
    assert demos and demos[0].resolved_by == "auto"
    loaded = await ctx.db.get_memory(l3.id)
    assert loaded.layer == "L1_atom"
    assert "l3_reviews" not in loaded.source

    # 有命中（hit_count>0）不计数
    hit = await ctx.tiers.longterm.add(
        title="命中档案", content="内容。", layer="L3_core",
    )
    await ctx.db.touch_hits([hit.id])
    await _review(ctx)
    assert "l3_reviews" not in (
        await ctx.db.get_memory(hit.id)
    ).source

    # trace_verified 也不计数
    verified = await ctx.tiers.longterm.add(
        title="已验证档案", content="内容。", layer="L3_core",
    )
    set_verification(verified, "trace_verified", ["trace:t9"])
    await ctx.db.update_memory(verified)
    await _review(ctx)
    assert "l3_reviews" not in (
        await ctx.db.get_memory(verified.id)
    ).source


# ---------------------------------------------------------------------------
# ⑦ 漂移再验证：trace_verified 连续无新命中 → 重置 unverified
# ---------------------------------------------------------------------------

async def test_drift_refresh_resets_verification(ctx):
    stale = await ctx.tiers.longterm.add(
        title="漂移档案", content="内容。", layer="L3_core",
    )
    set_verification(stale, "trace_verified", ["trace:tseed"])
    await ctx.db.update_memory(stale)

    for _ in range(5):
        await _review(ctx)
    assert verification_of(
        await ctx.db.get_memory(stale.id),
    ).state == "trace_verified"

    for _ in range(6):  # 第 11 次审查：11 > verify_refresh_reviews=10
        await _review(ctx)
    vs = verification_of(await ctx.db.get_memory(stale.id))
    assert vs.state == "unverified", "超期无新命中应重置待再验证"
    assert "trace:tseed" in vs.evidence, "evidence 历史保留"

    # 期间有新命中 → 不重置
    fresh = await ctx.tiers.longterm.add(
        title="常新档案", content="内容。", layer="L3_core",
    )
    set_verification(fresh, "trace_verified", ["trace:tseed"])
    await ctx.db.update_memory(fresh)
    for i in range(11):
        await _review(ctx)
        if i == 4:
            await ctx.db.touch_hits([fresh.id])
    assert verification_of(
        await ctx.db.get_memory(fresh.id),
    ).state == "trace_verified"


# ---------------------------------------------------------------------------
# ⑧ 矛盾正/反例
# ---------------------------------------------------------------------------

async def test_contradiction_positive_and_negative(ctx):
    # 正例：相似标题 + 同属性不同取值
    a = await ctx.tiers.longterm.add(
        title="核心交换机档案",
        content="核心交换机型号：CX-9 双机热备。", layer="L3_core",
        source={"origin": "memory_note"},
    )
    b = await ctx.tiers.longterm.add(
        title="核心交换机档案2",
        content="核心交换机型号：CX-7 单机部署。", layer="L3_core",
        source={"origin": "memory_note"},
    )
    proposals = await _review(ctx)
    flags = [p for p in proposals if p.action == "flag_contradiction"]
    assert flags and flags[0].resolved_by == "auto"
    pair = {a.id, b.id}
    for rid in pair:
        loaded = await ctx.db.get_memory(rid)
        marked = {
            f.get("with") for f in
            (loaded.source.get("contradiction_flags") or [])
        }
        assert marked == pair - {rid}, "两侧应互插旗标"

    # 反例：相似标题但关键事实一致 → 不提议不插旗
    d1 = await ctx.tiers.longterm.add(
        title="备份策略档案",
        content="备份保留期：30天 全量周日。", layer="L3_core",
        source={"origin": "memory_note"},
    )
    d2 = await ctx.tiers.longterm.add(
        title="备份策略档案2",
        content="备份保留期：30天 增量每日。", layer="L3_core",
        source={"origin": "memory_note"},
    )
    proposals2 = await _review(ctx)
    assert not [
        p for p in proposals2 if p.action == "flag_contradiction"
    ]
    for rid in (d1.id, d2.id):
        loaded = await ctx.db.get_memory(rid)
        assert not (
            loaded.source.get("contradiction_flags") or []
        )


# ---------------------------------------------------------------------------
# ⑨ demote_l3 执行：旧 L3 镜像删除、新 L1 镜像、剩余 L3 related 链接
# ---------------------------------------------------------------------------

async def test_demote_l3_mirror_and_links(ctx):
    rec = await ctx.tiers.longterm.add(
        title="环境概要", content="生产集群 prod-1。", layer="L3_core",
        source={"origin": "memory_note"},
    )
    sibling = await ctx.tiers.longterm.add(
        title="值班概要", content="值班表 2026。", layer="L3_core",
        source={"origin": "memory_note"},
    )
    old = os.path.join(ctx.mirror.longterm_dir, "L3_core-环境概要.md")
    new = os.path.join(ctx.mirror.longterm_dir, "L1_atom-环境概要.md")
    assert os.path.isfile(old)  # 前置：write_longterm 同名规则

    review = PendingReview(
        action="demote_l3", memory_id=rec.id,
        payload={"memory_id": rec.id},
        reason="驻留上限", confidence=0.9,
    )
    assert await ctx.tiers.propose(review) == "auto_applied"
    assert not os.path.exists(old), "旧 L3 镜像应删除"
    assert os.path.isfile(new), "新 L1 镜像应写入"

    loaded = await ctx.db.get_memory(rec.id)
    assert loaded.layer == "L1_atom"
    assert loaded.tier == "long" and loaded.status == "active"
    assert loaded.version == 2

    sib = await ctx.db.get_memory(sibling.id)
    assert any(
        l.target_id == rec.id and l.kind == "related" for l in sib.links
    ), "剩余 L3 应追加 related 指向被降级条目"


# ---------------------------------------------------------------------------
# ⑪ 未知 action 抛 ValueError
# ---------------------------------------------------------------------------

async def test_dispatch_unknown_action_raises(ctx):
    review = PendingReview.model_construct(
        id="rx", action="no_such_action", payload={},
        reason="", confidence=0.9,
    )
    with pytest.raises(ValueError, match="unknown review action"):
        await ctx.tiers._dispatch(review)


# ---------------------------------------------------------------------------
# 复核修复 3：demote_l3 用户手写豁免（0.9 也不自动降级）
# ---------------------------------------------------------------------------

async def test_handwritten_l3_demote_requires_manual(ctx):
    hand = await ctx.tiers.longterm.add(
        title="手写档案", content="用户手写内容。", layer="L3_core",
        source={"origin": "memory_note"},
    )
    for _ in range(3):  # 驻留满限（无验证零命中）
        await _review(ctx)
    loaded = await ctx.db.get_memory(hand.id)
    assert loaded.layer == "L3_core", "手写 L3 不应被静默自动降级"
    assert loaded.status == "active"
    pend = [
        p for p in await ctx.tiers.pending(50)
        if p.action == "demote_l3" and p.memory_id == hand.id
    ]
    assert pend, "驻留满限仍应出提议（人工确认通道）"


# ---------------------------------------------------------------------------
# 复核修复 2：memory_note layer=L3_core 是 L3 的生产创建路径
# ---------------------------------------------------------------------------

async def test_memory_note_l3_production_path(ctx):
    from rsi_mem.memory.guards import l3_eligible
    from rsi_mem.memory.tools import build_memory_tools

    tools = build_memory_tools(ctx.tiers)
    note = next(t for t in tools if t.name == "memory_note")
    chunk = await note(
        title="值班约定", content="值班交接每天早 10 点。",
        kind="fact", tier="long", layer="L3_core",
    )
    assert "待确认" in chunk.content[0].text
    pend = [
        p for p in await ctx.tiers.pending(20)
        if p.action == "create_longterm"
        and p.payload.get("layer") == "L3_core"
    ]
    assert pend, "L3_core 提议应进入待确认队列"

    await ctx.tiers.resolve(pend[0].id, approved=True)
    rec = next(
        m for m in await ctx.db.list_memories(tier="long", layer="L3_core")
        if m.title == "值班约定"
    )
    assert rec.source.get("origin") == "memory_note"
    assert l3_eligible(rec), "用户手写 L3 凭 origin 白名单准入"
    assert any(
        c.id == rec.id
        for c in await ctx.tiers.retriever.core_profile()
    ), "确认后的手写 L3 进入每轮注入面"
