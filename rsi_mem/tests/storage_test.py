# -*- coding: utf-8 -*-
"""storage 层测试：CRUD、FTS 中文检索、TTL、基因组版本、待确认队列。"""
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
