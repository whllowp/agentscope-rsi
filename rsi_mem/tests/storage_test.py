# -*- coding: utf-8 -*-
"""storage 层测试：CRUD、FTS 中文检索、TTL、基因组版本、待确认队列。"""
import asyncio
import os
import tempfile

import pytest

from rsi_mem.storage import (
    GenomeSpec,
    MemoryRecord,
    PendingReview,
    RSIMemDatabase,
    TraceRecord,
)


@pytest.fixture
async def db():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mem.db")
        database = RSIMemDatabase(path)
        yield database
        await database.close()


async def test_memory_crud_and_status_filter(db):
    m = MemoryRecord(
        tier="long",
        layer="L1_atom",
        title="Redis 内存淘汰策略",
        content="集群 redis-prod 使用 allkeys-lru，改动需先备份。",
    )
    await db.insert_memory(m)
    got = await db.get_memory(m.id)
    assert got is not None and got.title == m.title

    got.status = "archived"
    await db.update_memory(got)
    active = await db.list_memories(tier="long", status="active")
    assert all(x.id != m.id for x in active)
    archived = await db.list_memories(tier="long", status="archived")
    assert any(x.id == m.id for x in archived)


async def test_fts_chinese_and_english(db):
    await db.insert_memory(MemoryRecord(
        tier="long", layer="L1_atom", title="磁盘告警",
        content="大盘磁盘使用率超过 90% 时先清理 /var/log 大文件再扩容",
    ))
    await db.insert_memory(MemoryRecord(
        tier="long", layer="L1_atom", title="证书续期",
        content="ingress 证书由 cert-manager 自动续期，无需人工处理",
    ))
    zh = await db.search_text("磁盘使用率")
    assert zh and zh[0][1] > 0
    hit = await db.get_memory(zh[0][0])
    assert hit is not None and hit.title == "磁盘告警"

    en = await db.search_text("cert-manager renewal")
    assert en and (await db.get_memory(en[0][0])).title == "证书续期"

    # 两字中文词走 LIKE 兜底
    short = await db.search_text("磁盘")
    assert short and (await db.get_memory(short[0][0])).title == "磁盘告警"


async def test_short_term_ttl(db):
    m = MemoryRecord(
        tier="short", title="tmp", content="x",
        expires_at=1000.0,  # 早已过期
    )
    await db.insert_memory(m)
    assert await db.expire_short_term(now=2000.0) == 1
    assert await db.get_memory(m.id) is None


async def test_hit_counter(db):
    m = MemoryRecord(tier="short", title="t", content="c")
    await db.insert_memory(m)
    await db.touch_hits([m.id])
    got = await db.get_memory(m.id)
    assert got.hit_count == 1 and got.use_count == 1


async def test_genome_versions(db):
    seed = await db.active_genome()
    assert seed.version == 0
    v1 = seed.model_copy(
        update={"version": 1, "parent_version": 0, "operator": "logic_evolve"},
    )
    await db.save_genome(v1)
    assert (await db.active_genome()).version == 1
    versions = [g.version for g in await db.list_genomes()]
    assert versions == [1, 0]


async def test_trace_roundtrip(db):
    from rsi_mem.storage import ModelCallStat

    t = TraceRecord(query="磁盘满了怎么办", digest="建议先清理日志")
    t.model_calls.append(ModelCallStat(
        model="qwen", input_tokens=10, output_tokens=5, latency_s=0.1,
    ))
    await db.upsert_trace(t)
    traces = await db.list_traces()
    assert len(traces) == 1
    assert traces[0].total_tokens == 15
    assert await db.count_traces_since(0) == 1


async def test_pending_reviews(db):
    r = PendingReview(
        action="create_muscle", reason="重复出现磁盘告警处置模式",
        confidence=0.9,
    )
    await db.enqueue_review(r)
    pend = await db.pending_reviews()
    assert len(pend) == 1
    await db.resolve_review(r.id, "user", 123.0)
    assert await db.pending_reviews() == []
    got = await db.get_review(r.id)
    assert got is not None and got.resolved_by == "user"


async def test_genome_seed_has_locked_fragment():
    seed = GenomeSpec.seed()
    assert any(f.locked for f in seed.prompt_fragments)
    assert seed.retrieval.top_k >= 1


async def test_stats(db):
    await db.insert_memory(MemoryRecord(tier="muscle", title="sop"))
    await db.active_genome()  # 落种子基因组
    s = await db.stats()
    assert s["muscle"] == 1 and s["genome_version"] == 0
    assert s["broken_links"] == 0
    # 造一条断链（指向不存在的 id）→ 计数
    m = MemoryRecord(tier="long", layer="L1_atom", title="t", content="c")
    from rsi_mem.storage import MemoryLink

    m.links.append(MemoryLink(target_id="nope", kind="related"))
    await db.insert_memory(m)
    assert (await db.stats())["broken_links"] == 1


async def test_list_memories_tier_comma_split(db):
    """U3 扫描接缝修复的直接断言：tier 支持逗号拆分。

    此前 tier 是精确 AND 匹配（仅 status 拆分），传
    ``tier="long,short"`` 得空集、link_repair 静默 no-op。
    """
    await db.insert_memory(MemoryRecord(
        tier="long", layer="L1_atom", title="长期条目", content="x",
    ))
    await db.insert_memory(MemoryRecord(
        tier="short", title="短期条目", content="y",
    ))
    await db.insert_memory(MemoryRecord(
        tier="muscle", title="肌肉条目", content="z",
    ))
    both = await db.list_memories(
        tier="long,short", status="active,promoted",
    )
    assert {m.tier for m in both} == {"long", "short"}, \
        "tier 逗号拆分应返回两层记录"
    assert all(m.tier != "muscle" for m in both)
    # 单值语义不回归
    only_long = await db.list_memories(tier="long")
    assert {m.tier for m in only_long} == {"long"}


# ---------------------------------------------------------------------------
# U1 债务②③④⑤：stats 预过滤/脏 JSON、空拆分语义锁定、游标边界、
# bump_meta 原子性、find_by_title
# ---------------------------------------------------------------------------

async def test_stats_prefilters_empty_links_and_skips_dirty_json(db):
    """200 条空链 + 2 条脏 JSON：stats 不抛错、broken_links 只计真实断链。

    脏 JSON（外部写坏 / 损坏页）计入新返回键 ``broken_json``，跳过
    而非让整个 stats 崩掉；空链行经 SQL 预过滤不再进 Python 侧循环。
    """
    from rsi_mem.storage import MemoryLink

    for i in range(200):
        await db.insert_memory(MemoryRecord(
            tier="long", layer="L1_atom", title=f"空链{i:03d}",
            content="x",
        ))
    broken = MemoryRecord(
        tier="long", layer="L1_atom", title="真断链", content="x",
    )
    broken.links.append(MemoryLink(target_id="nope", kind="related"))
    await db.insert_memory(broken)
    # 空链行也可以是合法 target：指向空链行的链接不是断链
    empty_linked = MemoryRecord(
        tier="long", layer="L1_atom", title="指向空链行", content="x",
    )
    empty_linked.links.append(MemoryLink(target_id=broken.id, kind="related"))
    await db.insert_memory(empty_linked)

    def _insert_dirty(idx: int) -> None:
        db._conn.execute(
            "INSERT INTO memories (id, tier, kind, title, content,"
            " triggers, scope, confidence, status, links, source,"
            " use_count, hit_count, created_at, updated_at, version)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"dirty-{idx}", "long", "observation", f"脏JSON{idx}", "x",
             "[]", "global", 0.5, "active", "{oops", "{}", 0, 0,
             1000.0, 1000.0, 1),
        )

    await db._run(_insert_dirty, 1)
    await db._run(_insert_dirty, 2)

    s = await db.stats()
    assert s["broken_links"] == 1, "只计真实断链（脏 JSON 不计入）"
    assert s["broken_json"] == 2, "脏 JSON 行计入 broken_json 返回键"
    assert s["long"] == 204  # 空链 200 + 真断链 + 指向方 + 脏 2


async def test_list_memories_empty_split_returns_empty(db):
    """tier/status 空拆分 → 空集（U1 债务③：语义锁定与防御）。

    当前 SQLite 实测 ``IN ()`` 合法返回空集（本会话 3.50.4 验证，
    评审环境 3.53.1 同），此处非回归修复：实现已显式 ``AND 1=0``，
    不依赖方言行为，防止未来把空拆分误扫全表。
    """
    await db.insert_memory(MemoryRecord(
        tier="long", layer="L1_atom", title="t", content="c",
    ))
    assert await db.list_memories(tier=",") == []
    assert await db.list_memories(status=",,") == []
    assert await db.list_memories(tier=" , ,") == []
    # 非空拆分语义不受影响
    assert len(await db.list_memories(tier="long")) == 1


async def test_list_memories_updated_before_cursor_boundary(db):
    """游标边界：并列 updated_at + limit=2 扫两轮无遗漏。

    纯时间戳 ``<=`` 边界含边界行（并列组内允许重复），但并列组
    大小超过 limit 时会永远重复同一窗（SQLite 稳定排序，本会话
    3.50.4 实测）——复合游标 ``(updated_at, id)`` 在组内按 id 走窗：
    边界行允许重复一次、绝不跳行。
    """
    ts = 2000.0
    ids = set()
    for i in range(3):
        m = MemoryRecord(
            tier="long", layer="L1_atom", title=f"t{i}", content="c",
            created_at=ts, updated_at=ts,  # 三条完全同时间戳
        )
        await db.insert_memory(m)
        ids.add(m.id)

    # 纯 <= 边界：含边界行（3 条同戳全入窗，窗口 ≥ 组大小时无遗漏）
    full = await db.list_memories(tier="long", limit=10, updated_before=ts)
    assert {m.id for m in full} == ids, "<= 含边界行（< 会整组跳过）"

    # 复合游标按 limit=2 走两轮：边界行重复允许、不跳行
    seen: set[str] = set()
    cursor_ts: float | None = None
    cursor_id: str | None = None
    for _ in range(2):
        batch = await db.list_memories(
            tier="long", limit=2,
            updated_before=cursor_ts, cursor_id=cursor_id,
        )
        assert batch, "复合游标不应提前返回空窗"
        seen.update(m.id for m in batch)
        cursor_ts, cursor_id = batch[-1].updated_at, batch[-1].id
    assert seen == ids, "两轮扫描必须覆盖全部 3 条（组内按 id 走窗）"

    # 走到尾：第三轮只剩最后一条（< limit → 调用方据此重置游标）
    third = await db.list_memories(
        tier="long", limit=2,
        updated_before=cursor_ts, cursor_id=cursor_id,
    )
    assert len(third) == 1 and seen.issuperset({m.id for m in third})


async def test_bump_meta_atomic_under_concurrency(db):
    """并发 20 次自增：终值 20 且返回值恰为 1..20（单锁原子）。"""
    results = await asyncio.gather(
        *(db.bump_meta("proxy_turns") for _ in range(20)),
    )
    assert sorted(results) == list(range(1, 21)), "无丢更新、无重复值"
    assert await db.get_meta("proxy_turns") == "20"
    # 与 set_meta 的既有语义可共存（UPSERT 覆盖）
    await db.set_meta("proxy_turns", "0")
    assert await db.bump_meta("proxy_turns") == 1


async def test_find_by_title(db):
    """精确标题查询：命中 / 未命中 / 状态滤集外不算命中。"""
    m = MemoryRecord(tier="muscle", kind="sop", title="SOP: 磁盘", content="x")
    await db.insert_memory(m)
    got = await db.find_by_title(
        tier="muscle", title="SOP: 磁盘", status="active,promoted",
    )
    assert got is not None and got.id == m.id
    assert await db.find_by_title(
        tier="muscle", title="不存在", status="active",
    ) is None
    assert await db.find_by_title(
        tier="long", title="SOP: 磁盘", status="active",
    ) is None, "tier 滤集不匹配不算命中"

    m.status = "archived"
    await db.update_memory(m)
    assert await db.find_by_title(
        tier="muscle", title="SOP: 磁盘", status="active,promoted",
    ) is None, "archived 不在 active,promoted 滤集内"
    got = await db.find_by_title(
        tier="muscle", title="SOP: 磁盘", status="archived",
    )
    assert got is not None and got.id == m.id
