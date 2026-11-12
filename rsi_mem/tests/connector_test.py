# -*- coding: utf-8 -*-
"""http_proxy 与 agent_builder / model_proxy 测试。"""
import asyncio

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

    # 轨迹已记录（模型侧）
    traces = await ctx.db.list_traces(limit=5)
    assert len(traces) == 1
    assert traces[0].outcome.score > 0  # 有答复 → base+0.2
    assert traces[0].model_calls[0].input_tokens == 11


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
