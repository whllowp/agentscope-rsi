# -*- coding: utf-8 -*-
"""U3 固化管线测试：以轨迹为 ground truth 的校验、去重、断链修复、
L3 合计预算瘦身、SOP 查重、L0 回收、预算口径、并发冒烟与 e2e。

预算口径（config.ReviewConfig docstring 同步）：maintenance_budget=12
只封维护+污染类提议；晋升维持 out[:5]、SOP 单条，不计数。
"""
import asyncio
import os
from difflib import SequenceMatcher

import pytest

from agentscope.agent import Agent
from agentscope.tool import Toolkit

from rsi_mem.config import RSIMemConfig
from rsi_mem.connector import OpsMemoryMiddleware
from rsi_mem.context import RSIMemContext
from rsi_mem.memory.guards import set_verification, verification_of
from rsi_mem.rsi.tools import build_rsi_tools
from rsi_mem.storage import (
    MemoryLink,
    MemoryRecord,
    OutcomeRecord,
    ToolCallNode,
    TraceRecord,
)

from .conftest import DummyChatModel

_L3_TITLES = [
    "集群档案", "网络档案", "存储档案", "值班档案", "容量档案",
    "备份档案", "安全档案", "中间件档案", "数据库档案", "日志档案",
    "告警档案", "发布档案",
]

_MAINTENANCE_ACTIONS = {
    "merge_memories", "demote_muscle", "archive_memory",
    "quarantine_memory", "flag_contradiction", "demote_l3",
    "repair_links",
}


@pytest.fixture
async def ctx(tmp_path):
    context = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem")),
    )
    yield context
    await context.close()


def _mk_e2e_agent(ctx, name="E2EAgent"):
    mem_mw = OpsMemoryMiddleware(ctx)
    agent = Agent(
        name=name, system_prompt="你是运维助手。",
        model=DummyChatModel(), toolkit=Toolkit(),
        middlewares=[mem_mw],
    )
    return mem_mw, agent


# ---------------------------------------------------------------------------
# ① 使用型验证正/反例
# ---------------------------------------------------------------------------

async def test_usage_verification_positive_and_negative(ctx):
    rec = await ctx.tiers.longterm.add(
        title="使用验证档案", content="内容。", layer="L3_core",
        source={"trace_id": "t-create"},
    )
    t_low = TraceRecord(session_id="s1", query="q1", digest="d1")
    t_low.used_memory_ids = [rec.id]
    t_low.outcome = OutcomeRecord(score=0.5)  # 低分反例
    await ctx.db.upsert_trace(t_low)
    await ctx.consolidator.review_session("s1", [t_low])
    assert verification_of(
        await ctx.db.get_memory(rec.id),
    ).state == "unverified", "低分轨迹不应授予 trace_verified"

    t_ok = TraceRecord(session_id="s1", query="q2", digest="d2")
    t_ok.used_memory_ids = [rec.id]
    t_ok.outcome = OutcomeRecord(score=0.8)  # 成功轨迹正例
    await ctx.db.upsert_trace(t_ok)
    await ctx.consolidator.review_session("s1", [t_low, t_ok])
    assert verification_of(
        await ctx.db.get_memory(rec.id),
    ).state == "trace_verified"


# ---------------------------------------------------------------------------
# ② merge：merged_into 落盘 + 第三方链接重定向 + 幂等
# ---------------------------------------------------------------------------

async def test_merge_writes_merged_into_and_redirects(ctx):
    primary = await ctx.tiers.longterm.add(title="主条目", content="A 内容。")
    dup = await ctx.tiers.longterm.add(title="重复条目", content="B 内容。")
    third = await ctx.tiers.longterm.add(
        title="第三方", content="C 内容。",
        links=[MemoryLink(target_id=dup.id, kind="related", note="x")],
    )
    merged = await ctx.tiers.longterm.merge(primary.id, dup.id)
    assert merged is not None

    loaded_dup = await ctx.db.get_memory(dup.id)
    assert loaded_dup.source.get("merged_into") == primary.id
    assert loaded_dup.status == "archived"

    third_loaded = await ctx.db.get_memory(third.id)
    assert [l.target_id for l in third_loaded.links] == [primary.id], \
        "指向 duplicate 的 related 链接应即时重定向到 primary"

    again = await ctx.tiers.longterm.merge(primary.id, dup.id)
    assert again.content == merged.content, "重复 merge 幂等不追加"


# ---------------------------------------------------------------------------
# ③ 断链三边界 → repair_links 执行后无悬空
# ---------------------------------------------------------------------------

async def test_link_repair_three_boundaries(ctx):
    host = await ctx.tiers.longterm.add(title="宿主", content="H 内容。")
    gone = await ctx.tiers.longterm.add(title="已删目标", content="G。")
    archived = await ctx.tiers.longterm.add(title="归档目标", content="A。")
    await ctx.tiers.longterm.archive(archived.id)
    expired = await ctx.tiers.shortterm.record(
        session_id="s1", title="过期短期", content="E。",
    )
    loaded_exp = await ctx.db.get_memory(expired.id)
    loaded_exp.expires_at = 1000.0
    await ctx.db.update_memory(loaded_exp)
    # 归档目标留 merged_into → repair 应重定向而非删链
    primary = await ctx.tiers.longterm.add(title="重定向主", content="P。")
    archived_rec = await ctx.db.get_memory(archived.id)
    archived_rec.source["merged_into"] = primary.id
    await ctx.db.update_memory(archived_rec)

    host_rec = await ctx.db.get_memory(host.id)
    host_rec.links = [
        MemoryLink(target_id=gone.id, kind="related"),
        MemoryLink(target_id=archived.id, kind="related"),
        MemoryLink(target_id=expired.id, kind="related"),
    ]
    await ctx.db.update_memory(host_rec)
    await ctx.db.delete_memory(gone.id)
    assert (await ctx.db.stats())["broken_links"] == 3

    proposals = await ctx.consolidator.review_session(None, [])
    repairs = [p for p in proposals if p.action == "repair_links"]
    assert repairs and repairs[0].resolved_by == "auto"

    fixed = await ctx.db.get_memory(host.id)
    targets = [l.target_id for l in fixed.links]
    assert gone.id not in targets, "缺失目标 → 删链"
    assert expired.id not in targets, "过期短期 → 删链"
    assert primary.id in targets, "merged_into → 重定向 primary"
    assert (await ctx.db.stats())["broken_links"] == 0, "无悬空链接"


# ---------------------------------------------------------------------------
# ④ 内容相似去重（0.85 阈值 + 软上限放宽 0.75）
# ---------------------------------------------------------------------------

async def test_content_similarity_dedup(ctx):
    a = await ctx.tiers.longterm.add(
        title="磁盘清理甲",
        content="大盘磁盘使用率超过 90% 时先清理 /var/log 大文件再评估扩容方案。",
    )
    b = await ctx.tiers.longterm.add(
        title="磁盘清理乙",
        content="大盘磁盘使用率超过 90% 时先清理 /var/log 大文件再评估扩容方案！",
    )
    assert SequenceMatcher(None, a.content, b.content).ratio() >= 0.85

    proposals = await ctx.consolidator.review_session(None, [])
    merges = [p for p in proposals if p.action == "merge_memories"]
    assert merges, "内容相似 ≥0.85 应产生 merge 提议"
    assert merges[0].resolved_by is None, "0.8 置信度走人工确认"

    # 软上限超限 → 阈值放宽 0.75，次相似对（0.79）也被提议
    ctx.config.review.long_soft_cap = 1  # long 层现量 > 1 → over cap
    c = await ctx.tiers.longterm.add(
        title="网络排查甲",
        content="生产排查网络延迟先看 pod 状态再抓包定位丢包点。",
    )
    d = await ctx.tiers.longterm.add(
        title="网络排查乙",
        content="生产排查网络延迟先看 pod 状态再抓包分析根因链路。",
    )
    sim = SequenceMatcher(None, c.content, d.content).ratio()
    assert 0.75 <= sim < 0.85, "该对只在放宽阈值下命中"
    proposals2 = await ctx.consolidator.review_session(None, [])
    merges2 = [p for p in proposals2 if p.action == "merge_memories"]
    assert any(
        {p.payload["primary_id"], p.payload["duplicate_id"]} == {c.id, d.id}
        for p in merges2
    ), f"软上限超限应放宽阈值（相似度 {sim:.2f}）"


# ---------------------------------------------------------------------------
# ⑤ L3 合计预算瘦身
# ---------------------------------------------------------------------------

async def test_l3_slimming_aggregate_budget(ctx):
    from rsi_mem.memory.guards import l3_eligible

    l3s = []
    for i, title in enumerate(_L3_TITLES):
        rec = await ctx.tiers.longterm.add(
            title=title, layer="L3_core",
            content=f"{title}：属性{i}：值{i}。" + "细节" * 30,
        )
        l3s.append(rec)
    # 全部 trace_verified：l3_eligible 准入且 demote 走自动档
    # （复核修复后 origin=memory_note 手写 L3 的降级须人工确认）
    for rec in l3s:
        loaded = await ctx.db.get_memory(rec.id)
        set_verification(loaded, "trace_verified", ["trace:tseed"])
        await ctx.db.update_memory(loaded)
    # 前 2 条有命中：排名最前，不应被瘦身降级
    verified = l3s[:2]
    await ctx.db.touch_hits([r.id for r in verified])

    # 前置条件：12 条全部 l3_eligible（core_profile 渲染另有 [:10]
    # 条数上限，瘦身插槽按全量 30 超取计算合计字符）
    eligible = [
        c for c in await ctx.db.list_memories(
            tier="long", layer="L3_core",
            status="active,promoted", limit=30,
        ) if l3_eligible(c)
    ]
    assert len(eligible) == 12, "前置条件：全部 l3_eligible"
    total_before = sum(len(c.display(200)) for c in eligible)
    assert total_before > ctx.config.review.l3_char_budget - 400, \
        "前置条件：L3 段合计超预算"

    proposals = await ctx.consolidator.review_session(None, [])
    demos = [p for p in proposals if p.action == "demote_l3"]
    assert demos and all(p.resolved_by == "auto" for p in demos)

    cores_after = await ctx.tiers.retriever.core_profile()
    total_after = sum(len(c.display(200)) for c in cores_after)
    margin = 400  # _L3_INDEX_MARGIN（U4 索引段预留）
    assert total_after <= ctx.config.review.l3_char_budget, \
        "合计口径：L3 段 + 索引段 ≤ l3_char_budget"
    assert total_after <= ctx.config.review.l3_char_budget - margin

    demoted_ids = {
        r.id for r in l3s
        if (await ctx.db.get_memory(r.id)).layer == "L1_atom"
    }
    assert demoted_ids, "应发生瘦身降级"
    assert not (demoted_ids & {r.id for r in verified}), \
        "trace_verified 排名最前应保留"

    for rid in demoted_ids:
        loaded = await ctx.db.get_memory(rid)
        assert loaded.tier == "long" and loaded.status == "active", \
            "降级不动 tier/status（回放兼容）"
        # 旧镜像已清、新镜像已写
        title = loaded.title
        assert not os.path.isfile(os.path.join(
            ctx.mirror.longterm_dir, f"L3_core-{title}.md",
        ))
        assert os.path.isfile(os.path.join(
            ctx.mirror.longterm_dir, f"L1_atom-{title}.md",
        ))

    # 留下的 L3 含 related 链接指向被降级条目
    for c in cores_after:
        loaded = await ctx.db.get_memory(c.id)
        assert any(
            l.kind == "related" and l.target_id in demoted_ids
            for l in loaded.links
        )

    # 被降级条目检索仍召回
    results = await ctx.tiers.retriever.search(
        "细节", tiers=("long",), top_k=20,
    )
    hit_ids = {r.record.id for r in results}
    assert hit_ids & demoted_ids, "降级条目仍可经检索回捞"


# ---------------------------------------------------------------------------
# ⑥ SOP 查重：同一工具序列两次 review 只产一条 SOP
# ---------------------------------------------------------------------------

async def test_sop_distill_dedup(ctx):
    for i in range(3):
        t = TraceRecord(session_id="s1", query=f"磁盘 告警 {i}")
        t.tool_calls = [
            ToolCallNode(name="bash", input_digest="df -h", state="ok"),
            ToolCallNode(name="read_file", input_digest="logs",
                         state="ok"),
        ]
        t.outcome = OutcomeRecord(score=0.8)
        await ctx.db.upsert_trace(t)
    refreshed = await ctx.db.list_traces(limit=10)

    p1 = await ctx.consolidator.review_session("s1", refreshed)
    sop1 = next(p for p in p1 if p.action == "create_muscle")
    await ctx.tiers.resolve(sop1.id, approved=True)
    sops = [
        m for m in await ctx.db.list_memories(tier="muscle")
        if m.title.startswith("SOP:")
    ]
    assert len(sops) == 1

    p2 = await ctx.consolidator.review_session("s1", refreshed)
    assert not [p for p in p2 if p.action == "create_muscle"], \
        "同 title active SOP 已存在 → 不再重复提议"
    sops = [
        m for m in await ctx.db.list_memories(tier="muscle")
        if m.title.startswith("SOP:")
    ]
    assert len(sops) == 1, "同一 SOP 不应重复 INSERT"


# ---------------------------------------------------------------------------
# ⑦ L0_raw 回收：未引用超限 → archive 提议；被引用的不出提议
# ---------------------------------------------------------------------------

async def test_l0_raw_recycling(ctx):
    ctx.config.review.long_soft_cap = 4  # 回收阈值 = cap // 2 = 2
    free1 = await ctx.tiers.longterm.add_raw_evidence(text="无引用证据一")
    free2 = await ctx.tiers.longterm.add_raw_evidence(text="无引用证据二")
    cited = await ctx.tiers.longterm.add_raw_evidence(text="被引用证据")
    await ctx.tiers.longterm.add(
        title="引用方", content="C。",
        links=[MemoryLink(target_id=cited.id, kind="provenance")],
    )

    proposals = await ctx.consolidator.review_session(None, [])
    recycles = [p for p in proposals if p.action == "archive_memory"]
    assert {p.memory_id for p in recycles} == {free1.id, free2.id}
    assert all(p.confidence == pytest.approx(0.6) for p in recycles)
    assert all(p.resolved_by is None for p in recycles), "0.6 人工确认"
    assert (await ctx.db.get_memory(cited.id)).status == "active"


# ---------------------------------------------------------------------------
# ⑧ 预算口径：维护+污染 ≤ maintenance_budget，晋升不计数
# ---------------------------------------------------------------------------

async def test_maintenance_budget_cap(ctx):
    for i in range(15):
        title = f"重复标题{i:02d}"
        await ctx.tiers.longterm.add(title=title, content=f"内容{i}。")
        await ctx.tiers.longterm.add(title=title, content=f"内容{i}b。")
    for i in range(6):
        await ctx.tiers.shortterm.record(
            session_id="s1", title=f"晋升{i}", content=f"晋升内容{i}。",
            kind="fact", confidence=0.9,
        )

    proposals = await ctx.consolidator.review_session("s1", [])
    maintenance = [
        p for p in proposals if p.action in _MAINTENANCE_ACTIONS
    ]
    promotions = [
        p for p in proposals if p.action == "promote_longterm"
    ]
    assert len(maintenance) <= ctx.config.review.maintenance_budget, \
        "维护+污染类提议单次 ≤ maintenance_budget"
    assert promotions, "晋升插槽不受预算影响"
    assert len(promotions) <= 5, "晋升维持 out[:5]"
    assert len(proposals) == len(maintenance) + len(promotions), \
        "SOP 单条不计数（本次无 SOP）"


# ---------------------------------------------------------------------------
# ⑨ 并发冒烟：gather 两个 review_session 共享 ctx
# ---------------------------------------------------------------------------

async def test_concurrent_review_sessions(ctx):
    l3 = await ctx.tiers.longterm.add(
        title="并发档案", content="内容。", layer="L3_core",
    )
    await ctx.consolidator.review_session(None, [])
    await ctx.consolidator.review_session(None, [])
    assert (await ctx.db.get_memory(l3.id)).source["l3_reviews"] == 2

    r1, r2 = await asyncio.gather(
        ctx.consolidator.review_session(None, []),
        ctx.consolidator.review_session(None, []),
    )
    proposals = list(r1) + list(r2)
    budget = ctx.config.review.maintenance_budget
    assert len(proposals) <= budget * 2, "提议总数 ≤ budget×2"
    loaded = await ctx.db.get_memory(l3.id)
    assert loaded.layer == "L1_atom", "驻留满限应降级"
    assert loaded.version == 2, "并发双 demote 只翻转一次（幂等）"


# ---------------------------------------------------------------------------
# ⑩ middleware e2e：demote 后 L3 段变短、检索段仍含该条
# ---------------------------------------------------------------------------

async def test_middleware_prompt_shorter_after_demote(tmp_path):
    ctx = RSIMemContext(RSIMemConfig(data_dir=str(tmp_path / "e2e_u3")))
    try:
        for i, title in enumerate(_L3_TITLES):
            await ctx.tiers.longterm.add(
                title=title, layer="L3_core",
                content=f"{title}：属性{i}：值{i}。" + "细节" * 30,
            )
        # trace_verified：l3_eligible 准入 + demote 自动档
        # （手写 origin 的降级须人工，见 guards_test 手写豁免用例）
        for m in await ctx.db.list_memories(
            tier="long", layer="L3_core", limit=30,
        ):
            set_verification(m, "trace_verified", ["trace:tseed"])
            await ctx.db.update_memory(m)
        mem_mw, agent = _mk_e2e_agent(ctx)
        session = agent.state.session_id
        mem_mw._current_query[session] = "集群档案 环境"
        before = await mem_mw.on_system_prompt(agent, "基础提示词")
        # core_profile 条数上限 [:10] 截 10 条后，U1 债务⑥ 的运行时
        # 合计截断再收口：每条 display(200) 92~96 字符，预算
        # 1200−400=800 字符 → 任意 9 条 ≥ 828 > 800、8 条恒容纳
        # （≤768）→ 恒注入 8 条（确定性）
        assert before.count("档案：属性") == 8, "前置：L3 段按字符合计预算注入"

        await ctx.consolidator.review_session(None, [])
        after = await mem_mw.on_system_prompt(agent, "基础提示词")
        # U1 债务⑥ 后「demote 使注入变短」的前提被运行时截断取代：
        # 瘦身插槽与注入截断同值同口径（800 字符 + 同一贪心头部），
        # demote 后的剩余集合恰好仍装满预算 → 条数不降反稳。改为
        # 断言新不变量：审查前后 L3 段注入条数恒 ≤ 预算容纳数，且
        # 段内字符合计（每条 display(200) 口径）恒 ≤ 预算。
        budget = ctx.config.review.l3_char_budget - 400  # L3_INDEX_MARGIN

        def _l3_chars(prompt: str) -> tuple[int, int]:
            seg = prompt.split(
                "## 核心档案（长期记忆 L3，每轮可见）", 1,
            )[1]
            seg = seg.split("\n\n## ", 1)[0]  # 截到下一节标题为止
            items = [l for l in seg.splitlines() if l.startswith("- ")]
            return len(items), sum(len(l) - 2 for l in items)

        for prompt in (before, after):
            count, chars = _l3_chars(prompt)
            assert count <= 8 and chars <= budget, \
                "L3 段注入恒在字符合计预算内（运行时截断兜底）"
        assert after.count("档案：属性") <= 8

        demoted_ids = {
            r.id for r in await ctx.db.list_memories(tier="long", limit=50)
            if r.layer == "L1_atom" and r.title in _L3_TITLES
        }
        assert demoted_ids, "超预算的 L3 条目应被瘦身降级"
        genome = await ctx.db.active_genome()
        _block, ids = await ctx.tiers.reasoning_hint_block(
            query="细节 集群档案", session_id=None, genome=genome,
        )
        assert ids and set(ids) & demoted_ids, \
            "被降级条目应仍出现在检索注入段"
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# ⑫ U1 债务⑤：维护扫描游标轮转（link_repair 超单轮 limit 覆盖 + 扫尾重置）
# ---------------------------------------------------------------------------

async def test_link_repair_cursor_covers_beyond_single_limit(ctx):
    """600 条带断链宿主：两轮 _propose_link_repair 合计覆盖 > 单轮 limit。

    单轮 ``list_memories(limit=500)`` 只见最新 500 条，最老的 100 条
    永远轮不到；游标轮转后第二轮扫到余量并重置，两轮并集全覆盖。
    """
    base = 1700000000.0
    host_ids = set()
    for i in range(600):
        host = MemoryRecord(
            tier="long", layer="L1_atom",
            title=f"宿主{i:03d}", content="x",
            links=[MemoryLink(target_id=f"gone-{i}", kind="related")],
            created_at=base + i, updated_at=base + i,  # 时间戳互异
        )
        await ctx.db.insert_memory(host)
        host_ids.add(host.id)

    r1 = await ctx.consolidator._propose_link_repair()
    r2 = await ctx.consolidator._propose_link_repair()
    assert len(r1) == 500, "第一轮扫满单轮 limit"
    assert len(r1) + len(r2) > 500, "两轮合计覆盖超单轮 limit"
    covered = {p.memory_id for p in (*r1, *r2)}
    assert covered == host_ids, "两轮并集应覆盖全部 600 条宿主"
    # 第二轮扫到尾（101 < 500）→ 游标重置
    assert await ctx.db.get_meta("scan_cursor:link_repair") == ""


async def test_cursor_resets_after_tail_and_restarts_from_newest(ctx):
    """limit+1 条：第二轮扫到尾后游标归位，第三轮回到最新端。"""
    base = 1800000000.0
    newest_id = None
    for i in range(501):  # link_repair 单轮 limit=500
        host = MemoryRecord(
            tier="long", layer="L1_atom",
            title=f"宿主{i:03d}", content="x",
            links=[MemoryLink(target_id=f"gone-{i}", kind="related")],
            created_at=base + i, updated_at=base + i,
        )
        await ctx.db.insert_memory(host)
        if i == 500:
            newest_id = host.id

    r1 = await ctx.consolidator._propose_link_repair()
    assert len(r1) == 500
    assert await ctx.db.get_meta("scan_cursor:link_repair") != ""

    r2 = await ctx.consolidator._propose_link_repair()
    assert 0 < len(r2) < 500, "第二轮扫余量（并列边界行 + 最老一条）"
    assert await ctx.db.get_meta("scan_cursor:link_repair") == "", \
        "扫到尾后游标应重置为空（下一轮从头）"

    r3 = await ctx.consolidator._propose_link_repair()
    assert len(r3) == 500, "第三轮无游标 → 回到最新端扫满"
    assert r3[0].memory_id == newest_id, "第三轮首批应含最新一条"


# ---------------------------------------------------------------------------
# ⑬ U1 债务⑤：SOP 查重精确查询——第 101 条开外的同 title 老记录也命中
# ---------------------------------------------------------------------------

async def test_sop_dedup_exact_query_beyond_list_limit(ctx):
    """150 条 active muscle 中，同 title 老记录在最新 100 条窗口外。

    旧的 ``list_memories(limit=100)`` 扫描只见最新 100 条（本例中
    i=50..149），第 101 条开外的同 title 老记录（i=0，最老）漏判 →
    重复 INSERT；``find_by_title`` 精确查询无窗口。
    """
    dup_title = "SOP: 磁盘 告警 0"
    base = 1700000000.0
    for i in range(150):
        rec = MemoryRecord(
            tier="muscle", kind="sop",
            title=dup_title if i == 0 else f"其他SOP{i:03d}",
            content="x",
            created_at=base + i, updated_at=base + i,
        )
        await ctx.db.insert_memory(rec)

    traces = []
    for _ in range(3):
        t = TraceRecord(session_id="s1", query="磁盘 告警 0")
        t.tool_calls = [
            ToolCallNode(name="bash", input_digest="df -h", state="ok"),
            ToolCallNode(name="read_file", input_digest="l", state="ok"),
        ]
        traces.append(t)
    assert await ctx.consolidator._distill_sop("s1", traces) is None, \
        "窗口外的同 title 老记录也应被精确查重命中（不重复提议）"


# ---------------------------------------------------------------------------
# ⑪ U2→U3 组合验收：注入 → rsi_feedback(-1) → 隔离 → 不再注入/召回
# ---------------------------------------------------------------------------

async def test_u2_u3_pollution_path_e2e(tmp_path):
    ctx = RSIMemContext(RSIMemConfig(data_dir=str(tmp_path / "e2e_poll")))
    try:
        # 无 origin：须经使用型验证挣得 L3 注入资格（l3_eligible），
        # 而非用户手写白名单 —— 后续负反馈隔离也因此走自动档
        polluted = await ctx.tiers.longterm.add(
            title="污染档案", layer="L3_core",
            content="污染档案：错误结论，声称生产集群是单节点部署。",
        )
        t_ok = TraceRecord(session_id="s-poll", query="污染档案 查询",
                           digest="回答了结论")
        t_ok.used_memory_ids = [polluted.id]
        t_ok.outcome = OutcomeRecord(score=0.8)
        await ctx.db.upsert_trace(t_ok)
        await ctx.consolidator.review_session(None, [t_ok])
        assert verification_of(
            await ctx.db.get_memory(polluted.id),
        ).state == "trace_verified"

        mem_mw, agent = _mk_e2e_agent(ctx, name="PollAgent")
        session = agent.state.session_id
        mem_mw._current_query[session] = "污染档案 环境"
        prompt_before = await mem_mw.on_system_prompt(agent, "基础提示词")
        assert "污染档案" in prompt_before, "前置：L3 注入该条"

        t_bad = TraceRecord(session_id="s-poll", query="污染档案 复查",
                            digest="又回答了错误结论")
        t_bad.used_memory_ids = [polluted.id]
        t_bad.outcome = OutcomeRecord(score=0.9)
        await ctx.db.upsert_trace(t_bad)

        tools = build_rsi_tools(ctx)  # 同步构造
        feedback = next(
            tool for tool in tools if tool.name == "rsi_feedback"
        )
        chunk = await feedback(
            score=-1, comment="完全错了", trace_id=t_bad.id,
        )
        assert "已记录反馈 -1" in chunk.content[0].text

        # 从库重读（rsi_feedback 写的是 DB 行，内存对象不反映）
        await ctx.consolidator.review_session(
            None, await ctx.db.list_traces(limit=5),
        )
        loaded = await ctx.db.get_memory(polluted.id)
        assert loaded.status == "archived"
        assert verification_of(loaded).state == "refuted"

        prompt_after = await mem_mw.on_system_prompt(agent, "基础提示词")
        assert "污染档案" not in prompt_after, "隔离后不再注入"
        results = await ctx.tiers.retriever.search(
            "污染档案", tiers=("long",),
        )
        assert not any(
            r.record.id == polluted.id for r in results
        ), "隔离后检索不召回"
    finally:
        await ctx.close()
