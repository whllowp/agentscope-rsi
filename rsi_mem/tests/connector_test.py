# -*- coding: utf-8 -*-
"""http_proxy 与 agent_builder / model_proxy 测试。"""
import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from rsi_mem.config import RSIMemConfig
from rsi_mem.connector import http_proxy as proxy_mod
from rsi_mem.connector.model_proxy import TraceProxyChatModel
from rsi_mem.context import RSIMemContext

from .conftest import DummyChatModel


class _FakeUpstream:
    """替代 httpx.AsyncClient 的桩：返回固定 OpenAI 响应。"""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def post(self, url, json=None, headers=None):
        self.requests.append({"url": url, "json": json})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant",
                                "content": "mock 上游回复"},
                }],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
            request=httpx.Request("POST", url),
        )


@pytest.fixture
async def proxy_env(tmp_path, monkeypatch):
    config = RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem"))
    config.proxy.upstream_base_url = "https://upstream.example.com"
    config.proxy.upstream_api_key = "sk-test"
    ctx = RSIMemContext(config)
    # 预置长期记忆供注入
    await ctx.tiers.longterm.add(
        title="k8s 集群档案",
        content="生产集群 prod-1，etcd 三节点。",
    )
    fake = _FakeUpstream()
    monkeypatch.setattr(proxy_mod, "_CTX", ctx)
    monkeypatch.setattr(proxy_mod, "_CLIENT", fake)
    yield ctx, fake
    await ctx.close()


async def test_http_proxy_injects_memory_and_records_trace(proxy_env):
    ctx, fake = proxy_env
    client = TestClient(proxy_mod.app)
    resp = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "生产集群是什么架构"},
        ],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "mock 上游回复"

    # 上游收到注入后的 system 消息
    sent = fake.requests[0]["json"]["messages"]
    assert sent[0]["role"] == "system"
    assert "集群" in sent[0]["content"], "记忆上下文应注入 system 消息"

    # 轨迹已记录（模型侧）；注入 ids 进入 used_memory_ids（U2 接线）
    traces = await ctx.db.list_traces(limit=5)
    assert len(traces) == 1
    assert traces[0].outcome.score > 0  # 有答复 → base+0.2
    assert traces[0].model_calls[0].input_tokens == 11
    assert traces[0].used_memory_ids, "代理轨迹应携带注入记忆 id"


async def test_http_proxy_wires_used_ids_and_consolidation(
    tmp_path, monkeypatch,
):
    """U2 代理可达性：注入 ids 与 used_memory_ids 一致 + 周期固化触发。

    纯代理部署下 L3 也能拿到驻留计数 → demote_l3 自动执行，验证 /
    瘦身防御与中间件模式同等可达。
    """
    config = RSIMemConfig(data_dir=str(tmp_path / "proxy_u2"))
    config.proxy.upstream_base_url = "https://upstream.example.com"
    config.proxy.upstream_api_key = "sk-test"
    config.review.every_n_turns = 1
    ctx = RSIMemContext(config)
    long_rec = await ctx.tiers.longterm.add(
        title="k8s 集群档案", content="生产集群 prod-1，etcd 三节点。",
    )
    l3 = await ctx.tiers.longterm.add(
        title="值班档案", content="值班表 2026Q4。", layer="L3_core",
    )
    fake = _FakeUpstream()
    monkeypatch.setattr(proxy_mod, "_CTX", ctx)
    monkeypatch.setattr(proxy_mod, "_CLIENT", fake)
    monkeypatch.setattr(proxy_mod, "_PROXY_TURNS", 0)
    try:
        # 注入前缀与 ids 一起返回（ids 即注入清单）
        prefix, ids = await proxy_mod._build_memory_prefix(
            "生产集群是什么架构",
        )
        assert prefix and long_rec.id in ids

        # 直驱 _record_trace：used_memory_ids 落库且与注入 ids 一致；
        # 后台固化任务跑在本测试的事件循环内，可等待
        for i in range(3):
            resp = await fake.post("u", json={"i": i})
            await proxy_mod._record_trace(
                ctx, f"生产集群 {i}", resp, time.perf_counter(), ids,
            )
            await asyncio.sleep(0.2)  # 等 review_session 后台任务
        traces = await ctx.db.list_traces(limit=10)
        assert len(traces) == 3
        assert all(
            long_rec.id in t.used_memory_ids for t in traces
        ), "代理轨迹 used_memory_ids 应与注入 ids 一致"

        # 每 every_n_turns=1 条轨迹触发固化：3 次审查 → L3 驻留满限
        demoted = await ctx.db.get_memory(l3.id)
        assert demoted.layer == "L1_atom", "驻留上限降级应经代理触达"
        assert "l3_reviews" not in demoted.source
    finally:
        await ctx.close()


async def test_http_proxy_compact_prefix_with_index(
    tmp_path, monkeypatch,
):
    """U4 代理 e2e：紧凑前缀共用渲染器（L3 过滤 + 索引段在预算内），
    ids（索引/L3 注入）落 used_memory_ids。"""
    from rsi_mem.memory.tiers import L3_INDEX_MARGIN

    config = RSIMemConfig(data_dir=str(tmp_path / "proxy_u4"))
    config.proxy.upstream_base_url = "https://upstream.example.com"
    config.proxy.upstream_api_key = "sk-test"
    ctx = RSIMemContext(config)
    l3 = await ctx.tiers.longterm.add(
        title="环境档案", content="生产集群 prod-1 概要。", layer="L3_core",
        source={"origin": "memory_note"},
    )
    await ctx.tiers.longterm.add(
        title="未验证档案", content="内容。", layer="L3_core",
    )
    l1 = await ctx.tiers.longterm.add(
        title="k8s 档案", content="etcd 三节点。",
    )
    fake = _FakeUpstream()
    monkeypatch.setattr(proxy_mod, "_CTX", ctx)
    monkeypatch.setattr(proxy_mod, "_CLIENT", fake)
    try:
        prefix, ids = await proxy_mod._build_memory_prefix("生产集群")
        assert "核心档案" in prefix and "环境档案" in prefix, \
            "L3（过滤后）注入"
        assert "未验证档案" not in prefix, "l3_eligible 过滤未验证档案"
        assert "## 记忆索引（知道什么、去哪找）" in prefix, "索引段注入"
        assert prefix.rstrip().endswith("更多用 memory_search 检索") or \
            "更多用 memory_search 检索" in prefix

        # 索引段在 U3 预留余量内（同一合计预算口径）
        start = prefix.index("## 记忆索引")
        end = prefix.find("\n\n## ")
        index_section = prefix[start:] if end < 0 else prefix[start:end]
        body = "\n".join(
            l for l in index_section.splitlines()
            if l != "更多用 memory_search 检索"
        )
        assert len(body.encode("utf-8")) <= L3_INDEX_MARGIN

        # ids 含索引/L3 注入 id
        assert l3.id in ids and l1.id in ids

        # 经 U2 接好的链路落 used_memory_ids
        resp = await fake.post("u", json={})
        await proxy_mod._record_trace(
            ctx, "生产集群", resp, time.perf_counter(), ids,
        )
        trace = (await ctx.db.list_traces(limit=1))[0]
        assert l3.id in trace.used_memory_ids
        assert l1.id in trace.used_memory_ids, "索引入选 id 进轨迹"
    finally:
        await ctx.close()


async def test_http_proxy_stream_trace_scores_usage(tmp_path, monkeypatch):
    """复核修复 1（流式）：SSE 完成按有答复计，使用型验证可达。

    此前流式请求 answer 恒空 → score=0.5<0.7，流式部署下 L3 永远
    拿不到使用型验证。
    """
    config = RSIMemConfig(data_dir=str(tmp_path / "proxy_stream"))
    config.proxy.upstream_base_url = "https://upstream.example.com"
    config.proxy.upstream_api_key = "sk-test"
    ctx = RSIMemContext(config)
    l1 = await ctx.tiers.longterm.add(
        title="k8s 档案", content="etcd 三节点。",
    )
    fake = _FakeUpstream()
    monkeypatch.setattr(proxy_mod, "_CTX", ctx)
    monkeypatch.setattr(proxy_mod, "_CLIENT", fake)
    try:
        await proxy_mod._record_stream_trace(
            ctx, "生产集群", time.perf_counter(), [l1.id],
        )
        trace = (await ctx.db.list_traces(limit=1))[0]
        assert l1.id in trace.used_memory_ids
        assert trace.outcome.score >= 0.7, "流式完成按有答复计"
    finally:
        await ctx.close()


def test_http_proxy_health(proxy_env):
    client = TestClient(proxy_mod.app)
    assert client.get("/healthz").json() == {"status": "ok"}


async def test_model_proxy_records_usage(tmp_path):
    inner = DummyChatModel()
    stats = []

    class Sink:
        def on_model_stat(self, stat):
            stats.append(stat)

    proxy = TraceProxyChatModel(inner, sink=Sink())
    from agentscope.message import Msg, TextBlock

    resp = await proxy([
        Msg(name="user", role="user",
            content=[TextBlock(type="text", text="你好")]),
    ])
    assert resp.usage.input_tokens == 8
    assert len(stats) == 1
    assert stats[0].input_tokens == 8 and stats[0].model == "dummy-chat"
    # provider 专属属性透传
    assert proxy.formatter is inner.formatter


async def test_build_ops_agent_with_mock(tmp_path):
    """build_ops_agent 全链路组装（mock 模型 + LocalWorkspace）。"""
    from agentscope.message import Msg, TextBlock
    from agentscope.workspace import LocalWorkspace

    from rsi_mem.agent_builder import MockChatModel, build_ops_agent

    config = RSIMemConfig(data_dir=str(tmp_path / "data"))
    async with LocalWorkspace(workdir=str(tmp_path / "ws")) as ws:
        agent, ctx = await build_ops_agent(
            model=MockChatModel(), workspace=ws, config=config,
        )
        assert agent.name == "OpsFriday"
        # 中间件已挂载：回复一轮走完整链路
        reply = await agent.reply(
            Msg(name="user", role="user",
                content=[TextBlock(type="text", text="检查一下磁盘")]),
        )
        assert reply is not None
        await asyncio.sleep(0.5)  # 等待写回
        traces = await ctx.db.list_traces(limit=5)
        assert len(traces) == 1
        shorts = await ctx.db.list_memories(tier="short", limit=10)
        assert shorts
        await ctx.close()
