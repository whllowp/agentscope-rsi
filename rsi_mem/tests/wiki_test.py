# -*- coding: utf-8 -*-
"""U4 测试：wiki 索引化镜像（防撞/互链/幂等）与顶部小索引注入
（同一渲染器、双上限、尾行提示永在、合计预算、代理路径共用）。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentscope.agent import Agent
from agentscope.tool import Toolkit

from rsi_mem.config import RSIMemConfig
from rsi_mem.connector import OpsMemoryMiddleware
from rsi_mem.context import RSIMemContext
from rsi_mem.memory.tiers import L3_INDEX_MARGIN
from rsi_mem.storage import MemoryLink, MemoryRecord
from rsi_mem.storage.mirror import _safe_name

from .conftest import DummyChatModel

_L3_TITLES = [
    "集群档案", "网络档案", "存储档案", "值班档案", "容量档案",
    "备份档案", "安全档案", "中间件档案", "数据库档案", "日志档案",
    "告警档案", "发布档案",
]

_INDEX_TAIL = "更多用 memory_search 检索"


@pytest.fixture
async def ctx(tmp_path):
    context = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem")),
    )
    yield context
    await context.close()


def _read(ctx, name: str) -> str:
    with open(
        os.path.join(ctx.mirror.wiki_dir, name), encoding="utf-8",
    ) as f:
        return f.read()


def _read_all(ctx) -> dict[str, str]:
    return {
        name: _read(ctx, name)
        for name in sorted(os.listdir(ctx.mirror.wiki_dir))
    }


# ---------------------------------------------------------------------------
# ① rebuild 幂等（先清空再重建）
# ---------------------------------------------------------------------------

async def test_rebuild_idempotent_and_clears_stale(ctx):
    await ctx.tiers.longterm.add(
        title="磁盘场景", content="场景内容", layer="L2_scenario",
    )
    await ctx.tiers.longterm.add(title="原子事实", content="原子内容")
    stale = os.path.join(ctx.mirror.wiki_dir, "stale-page.md")
    with open(stale, "w", encoding="utf-8") as f:
        f.write("陈旧页")

    n1 = await ctx.tiers.rebuild_wiki()
    files1 = sorted(os.listdir(ctx.mirror.wiki_dir))
    assert "stale-page.md" not in files1, "重建前先清空"
    assert "INDEX.md" in files1
    texts1 = _read_all(ctx)

    n2 = await ctx.tiers.rebuild_wiki()
    assert n1 == n2
    assert sorted(os.listdir(ctx.mirror.wiki_dir)) == files1
    assert _read_all(ctx) == texts1, "两次重建结果完全一致（幂等）"


# ---------------------------------------------------------------------------
# ② 文件名防撞：_safe_name 截断同名 → 两个不同文件、内容各自完整
# ---------------------------------------------------------------------------

async def test_filename_collision_proof(ctx):
    head = "主" * 48  # 恰好 48 字符（_safe_name 截断上限）
    topic_a, topic_b = head + "甲", head + "乙"
    assert _safe_name(topic_a, "x") == _safe_name(topic_b, "x") == head, \
        "前置：截断后同名（互覆风险）"

    l2a = await ctx.tiers.longterm.add(
        title=topic_a, content="甲主题完整内容", layer="L2_scenario",
    )
    l2b = await ctx.tiers.longterm.add(
        title=topic_b, content="乙主题完整内容", layer="L2_scenario",
    )
    await ctx.tiers.rebuild_wiki()

    fa = os.path.join(ctx.mirror.wiki_dir, f"{head}-{l2a.id[:8]}.md")
    fb = os.path.join(ctx.mirror.wiki_dir, f"{head}-{l2b.id[:8]}.md")
    assert fa != fb and os.path.isfile(fa) and os.path.isfile(fb), \
        "锚点 id 后缀保证防撞"
    text_a, text_b = _read(ctx, os.path.basename(fa)), _read(
        ctx, os.path.basename(fb),
    )
    assert "甲主题完整内容" in text_a
    assert "乙主题完整内容" not in text_a
    assert "乙主题完整内容" in text_b
    assert "甲主题完整内容" not in text_b


# ---------------------------------------------------------------------------
# ③ L2 锚点页 + 挂链 L1 同页互链（交叉引用）
# ---------------------------------------------------------------------------

async def test_l2_anchor_page_with_linked_l1(ctx):
    l2a = await ctx.tiers.longterm.add(
        title="磁盘处置场景", content="磁盘场景锚点", layer="L2_scenario",
    )
    l2b = await ctx.tiers.longterm.add(
        title="网络处置场景", content="网络场景锚点", layer="L2_scenario",
    )
    await ctx.tiers.longterm.add(
        title="磁盘清理原子", content="先清理 /var/log",
        links=[
            MemoryLink(target_id=l2a.id, kind="related"),
            MemoryLink(target_id=l2b.id, kind="related"),
        ],
    )
    await ctx.tiers.longterm.add(title="孤儿原子", content="无链接")

    await ctx.tiers.rebuild_wiki()
    page_a = _read(ctx, f"磁盘处置场景-{l2a.id[:8]}.md")
    assert "磁盘场景锚点" in page_a
    assert "磁盘清理原子" in page_a, "挂链 L1 归入锚点页"
    assert "↔ 网络处置场景" in page_a, "交叉引用指向另一主题页"
    assert "返回索引" in page_a
    assert "孤儿原子" in _read(ctx, "其他-misc.md"), "孤儿进「其他」页"
    index = _read(ctx, "INDEX.md")
    assert "磁盘处置场景" in index and "其他" in index


# ---------------------------------------------------------------------------
# ④ 截断边界：恰超 1 行 / 1 字节 → 尾行提示仍在、渲染确定
# ---------------------------------------------------------------------------

async def test_index_truncation_boundary(ctx):
    records = [
        MemoryRecord(
            tier="long", layer="L2_scenario",
            title=f"主题{c}", content="x",
        ) for c in "ABCDE"
    ] + [
        MemoryRecord(tier="long", layer="L1_atom",
                     title=f"原子{i}", content="y")
        for i in range(3)
    ]
    mirror = ctx.mirror

    # 行上限恰超 1 行：max_lines=2 → 头行 + 1 行主题（尾行不计入）
    block = mirror.render_index_block(records, max_lines=2, max_chars=8192)
    lines = block.splitlines()
    assert lines[0] == "## 记忆索引（知道什么、去哪找）"
    assert len([l for l in lines if l.startswith("- ")]) == 1
    assert lines[-1] == _INDEX_TAIL, "尾行提示永在"

    # 字节上限恰超 1 字节 → 少一行、尾行仍在（字节按渲染器口径：
    # 头行 + 主题行拼接、不含换行与尾行）
    full = mirror.render_index_block(records, max_lines=60, max_chars=8192)
    full_lines = full.splitlines()
    content = "".join(
        [full_lines[0]]
        + [l for l in full_lines if l.startswith("- ")],
    )
    tight = mirror.render_index_block(
        records, max_lines=60,
        max_chars=len(content.encode("utf-8")) - 1,
    )
    assert tight.endswith(_INDEX_TAIL)
    assert len(
        [l for l in tight.splitlines() if l.startswith("- ")],
    ) == len(
        [l for l in full.splitlines() if l.startswith("- ")],
    ) - 1, "恰超 1 字节 → 前缀截断少一行"

    # 稳定键（主题名排序）确定性：两次渲染一致
    assert mirror.render_index_block(records, 2, 8192) == block


# ---------------------------------------------------------------------------
# ⑤ 100 条长期记忆：INDEX 上限 + 注入合计预算 + 复跑 U3⑤ slimming
# ---------------------------------------------------------------------------

async def test_hundred_memories_index_caps_and_budget(ctx):
    for i, title in enumerate(_L3_TITLES):
        await ctx.tiers.longterm.add(
            title=title, layer="L3_core",
            content=f"{title}：属性{i}：值{i}。" + "细节" * 30,
        )
    # 全部 trace_verified：l3_eligible 准入且瘦身 demote 走自动档
    # （复核修复后 origin=memory_note 手写 L3 的降级须人工确认）
    for m in await ctx.db.list_memories(
        tier="long", layer="L3_core", limit=30,
    ):
        from rsi_mem.memory.guards import set_verification

        set_verification(m, "trace_verified", ["trace:tseed"])
        await ctx.db.update_memory(m)
    # 88 条 L1 孤儿：内容互不相似（首字符块各不相同，避开内容去重）
    for i in range(88):
        await ctx.tiers.longterm.add(
            title=f"原子知识{i:03d}",
            content=chr(0x4E00 + i) * 20 + "：固定运维知识描述。",
        )

    proposals = await ctx.consolidator.review_session(None, [])
    demos = [p for p in proposals if p.action == "demote_l3"]
    assert demos, "复跑 U3⑤：L3 合计超预算应触发瘦身"

    index_text = _read(ctx, "INDEX.md")
    assert len(index_text.splitlines()) <= ctx.config.wiki.index_max_lines
    assert len(index_text.encode("utf-8")) <= ctx.config.wiki.index_max_bytes
    assert index_text.rstrip("\n").endswith(_INDEX_TAIL)

    # 注入段合计 ≤ l3_char_budget（L3 段 + 索引段，合计口径）
    cores = await ctx.tiers.retriever.core_profile()
    l3_chars = sum(len(c.display(200)) for c in cores)
    long_recs = await ctx.db.list_memories(
        tier="long", status="active,promoted", limit=500,
    )
    index_mem = [
        m for m in long_recs if m.layer in ("L1_atom", "L2_scenario")
    ]
    index_block, index_ids = ctx.mirror.render_index_block_with_ids(
        index_mem, ctx.config.wiki.index_max_lines, L3_INDEX_MARGIN,
    )
    assert index_ids
    assert l3_chars + len(index_block.encode("utf-8")) <= \
        ctx.config.review.l3_char_budget

    # 复跑 U3⑤ 的 slimming 断言（跨单元义务）
    demoted = {
        m.id for m in await ctx.db.list_memories(tier="long", limit=200)
        if m.layer == "L1_atom" and m.title in _L3_TITLES
    }
    assert demoted
    for m in await ctx.db.list_memories(tier="long", limit=200):
        if m.id in demoted:
            assert m.tier == "long" and m.status == "active", \
                "降级不动 tier/status（回放兼容）"
            assert not os.path.isfile(os.path.join(
                ctx.mirror.longterm_dir, f"L3_core-{m.title}.md",
            ))
            assert os.path.isfile(os.path.join(
                ctx.mirror.longterm_dir, f"L1_atom-{m.title}.md",
            ))
    for c in cores:
        loaded = await ctx.db.get_memory(c.id)
        assert any(
            l.kind == "related" and l.target_id in demoted
            for l in loaded.links
        ), "留下的 L3 含 related 指向被降级条目"


# ---------------------------------------------------------------------------
# ⑥ middleware e2e：prompt 含「记忆索引」段与 memory_search 指引
# ---------------------------------------------------------------------------

async def test_middleware_prompt_contains_index(tmp_path):
    ctx = RSIMemContext(RSIMemConfig(data_dir=str(tmp_path / "e2e_u4")))
    try:
        l2 = await ctx.tiers.longterm.add(
            title="磁盘处置场景", content="磁盘场景知识",
            layer="L2_scenario",
        )
        await ctx.tiers.longterm.add(
            title="磁盘清理原子", content="先清理再扩容",
            links=[MemoryLink(target_id=l2.id, kind="related")],
        )
        mem_mw = OpsMemoryMiddleware(ctx)
        agent = Agent(
            name="WikiAgent", system_prompt="你是运维助手。",
            model=DummyChatModel(), toolkit=Toolkit(),
            middlewares=[mem_mw],
        )
        session = agent.state.session_id
        mem_mw._current_query[session] = "磁盘"
        prompt = await mem_mw.on_system_prompt(agent, "基础提示词")
        assert "## 记忆索引（知道什么、去哪找）" in prompt
        assert _INDEX_TAIL in prompt, "memory_search 指引（wiki+搜索组合）"
        assert "磁盘处置场景" in prompt
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# ⑧ 归档一条后 rebuild → wiki 页与 INDEX 不再含它
# ---------------------------------------------------------------------------

async def test_archived_excluded_after_rebuild(ctx):
    await ctx.tiers.longterm.add(
        title="归档验证场景", content="场景", layer="L2_scenario",
    )
    await ctx.tiers.longterm.add(title="保留原子", content="保留内容")
    drop = await ctx.tiers.longterm.add(
        title="待归档原子", content="归档内容",
    )
    await ctx.tiers.rebuild_wiki()
    assert "待归档原子" in _read(ctx, "其他-misc.md")

    await ctx.tiers.longterm.archive(drop.id)
    await ctx.tiers.rebuild_wiki()
    all_text = "\n".join(_read_all(ctx).values())
    assert "待归档原子" not in all_text, "归档后 wiki 与 INDEX 均不再含"
    assert "保留原子" in all_text


# ---------------------------------------------------------------------------
# ⑩ 端到端：RSI_MEM_HOME 指仓库外临时目录跑 CLI bench（10 轮门）
# ---------------------------------------------------------------------------

def test_cli_bench_ten_rounds_outside_repo(tmp_path):
    """CLI bench 端到端（退出码 / 轮次 / 部署 / 非退化 / 预测样本）。

    能力边界：该门只守基因组参数不退化（bench 的 genome_quality 不读
    记忆内容）；记忆行为回归由本文件与 consolidate/connector 的 e2e
    断言把守。
    """
    env = dict(os.environ, RSI_MEM_HOME=str(tmp_path / "bench_home"))
    repo_root = str(Path(__file__).resolve().parents[2])
    proc = subprocess.run(
        [sys.executable, "-m", "rsi_mem.cli", "bench", "--rounds", "10"],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    out = proc.stdout
    payload = json.loads(out[out.index("{"):])
    summary = payload["summary"]
    assert summary["deploys"] >= 1, "至少一轮应部署改进"
    assert summary["non_degraded"] is True
    assert summary["prediction_samples"] >= 5, "预测-观测对应持续积累"

    report = json.loads(
        Path(payload["bench_path"]).read_text(encoding="utf-8"),
    )
    assert len(report["rows"]) == 10, "10 轮全完成"
    first, last = report["rows"][0], report["rows"][-1]
    assert last["quality_after"] >= first["quality_after"] - 0.02
