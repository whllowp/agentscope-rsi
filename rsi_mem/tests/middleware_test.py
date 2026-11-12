# -*- coding: utf-8 -*-
"""中间件端到端测试：真实 Agent + 离线假模型，验证整条链路。

覆盖：系统提示词注入（基因组片段/L3/肌肉反射）、回复后短期记忆写回、
轨迹录制、RSI 工具、memory 工具直调。
"""
import asyncio

import pytest

from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit

from rsi_mem.config import RSIMemConfig
from rsi_mem.connector import OpsMemoryMiddleware, RSIMiddleware
from rsi_mem.context import RSIMemContext

from .conftest import DummyChatModel


@pytest.fixture
async def setup(tmp_path):
    config = RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem"))
    config.review.every_n_turns = 2  # 快速触发周期审查
    ctx = RSIMemContext(config)
    mem_mw = OpsMemoryMiddleware(ctx)
    rsi_mw = RSIMiddleware(ctx)
    model = DummyChatModel()
    agent = Agent(
        name="OpsAgent",
        system_prompt="你是运维助手。",
        model=model,
        toolkit=Toolkit(),
        middlewares=[mem_mw, rsi_mw],
    )
    yield agent, ctx, mem_mw, rsi_mw, model
    await ctx.close()


async def _say(agent, text):
    return await agent.reply(
        Msg(name="user", role="user",
            content=[TextBlock(type="text", text=text)]),
    )


async def _drain_background(seconds=0.6):
    """等待 fire-and-forget 后台任务（写回/审查）完成。"""
    await asyncio.sleep(seconds)


async def test_middleware_pipeline_e2e(setup):
    agent, ctx, mem_mw, rsi_mw, model = setup
    session = agent.state.session_id

    # 预置：长期 L3 + 肌肉 SOP
    await ctx.tiers.longterm.add(
        title="环境档案",
        content="生产集群 prod-1，k8s 1.31。",
        layer="L3_core",
    )
    await ctx.tiers.muscle.note(
        title="磁盘告警SOP",
        content="1. df -h 核实\n2. 清理日志",
        triggers=["磁盘"],
    )

    # ---- 第一轮：注入 + 写回 + 轨迹 ----
    reply = await _say(agent, "线上磁盘快满了怎么处理")
    assert reply is not None and reply.get_text_content()
    await _drain_background()

    # 轨迹已录制（模型调用 + outcome）
    traces = await ctx.db.list_traces(limit=5)
    assert len(traces) == 1
    assert traces[0].session_id == session
    assert traces[0].outcome is not None
    assert traces[0].outcome.score > 0
    assert len(traces[0].model_calls) == 1
    assert traces[0].model_calls[0].input_tokens == 8

    # 短期记忆写回（规则兜底：问题+结论）
    shorts = await ctx.db.list_memories(
        tier="short", session_id=session, limit=10,
    )
    assert shorts, "回合结束应写回短期记忆"

    # 系统提示词注入验证：直接调用 on_system_prompt
    mem_mw._current_query[session] = "磁盘快满了"
    prompt = await mem_mw.on_system_prompt(agent, "基础提示词")
    assert "安全底线" in prompt  # 种子基因组锁定片段
    assert "核心档案" in prompt  # L3 注入
    assert "磁盘告警SOP" in prompt  # 肌肉反射命中

    # reasoning 注入（后台检索在单轮内可能来不及 → 直接验证组装函数）
    genome = await ctx.db.active_genome()
    block, ids = await ctx.tiers.reasoning_hint_block(
        query="磁盘快满了", session_id=session, genome=genome,
        exclude_sessions=(session,),
    )
    assert "记忆检索结果" in block or "短期记忆" in block


async def test_periodic_review_and_tools(setup):
    agent, ctx, mem_mw, rsi_mw, model = setup
    # 两轮对话（every_n_turns=2 → 第二轮后触发审查）
    await _say(agent, "帮我看下 redis 内存")
    await _say(agent, "再看看 redis 连接数")
    await _drain_background(1.0)

    # memory 工具可直调
    tools = await mem_mw.list_tools()
    names = {t.name for t in tools}
    assert names >= {
        "memory_search", "memory_note", "memory_confirm",
        "memory_link", "memory_status",
    }
    search = next(t for t in tools if t.name == "memory_search")
    chunk = await search("redis")
    assert chunk.content[0].text

    # rsi 工具
    rsi_tools = await rsi_mw.list_tools()
    assert {t.name for t in rsi_tools} >= {
        "rsi_status", "rsi_feedback", "rsi_evolve", "rsi_history",
    }
    feedback = next(t for t in rsi_tools if t.name == "rsi_feedback")
    chunk = await feedback(score=1, comment="解决了")
    assert "已记录反馈 +1" in chunk.content[0].text
    traces = await ctx.db.list_traces(limit=1)
    assert traces[0].outcome.feedback == 1
    assert traces[0].outcome.score > 0.7  # 反馈抬升评分

    # rsi_evolve：手动 dream（离线规则计划）
    evolve = next(t for t in rsi_tools if t.name == "rsi_evolve")
    chunk = await evolve()
    text = chunk.content[0].text
    assert ("已部署" in text) or ("未部署" in text)
    genomes = await ctx.db.list_genomes()
    history = next(t for t in rsi_tools if t.name == "rsi_history")
    chunk = await history()
    assert "v0" in chunk.content[0].text


async def test_trace_records_tool_calls_via_on_acting(setup):
    """直接驱动 on_acting hook 验证工具调用观测。"""
    agent, ctx, mem_mw, rsi_mw, model = setup
    from agentscope.message import ToolResultState
    from agentscope.tool import ToolChunk, ToolResponse

    session = agent.state.session_id
    from rsi_mem.rsi.trace import TraceRecorder

    recorder = TraceRecorder(
        ctx.db, ctx.config, session_id=session, query="q",
    )
    rsi_mw._recorders[session] = recorder

    async def fake_handler(**kwargs):
        yield ToolChunk(content=[
            TextBlock(type="text", text="partial"),
        ])
        yield ToolResponse(content=[
            TextBlock(type="text", text="done"),
        ], state=ToolResultState.SUCCESS)

    tool_call = type("TC", (), {"name": "bash"})()
    consumed = []
    async for chunk in rsi_mw.on_acting(
        agent, {"tool_call": tool_call}, fake_handler,
    ):
        consumed.append(chunk)
    assert len(consumed) == 2
    assert recorder.record.tool_calls[0].name == "bash"
    assert recorder.record.tool_calls[0].state == "ok"
    assert "done" in recorder.record.tool_calls[0].input_digest

    # 错误态映射
    async def failing_handler(**kwargs):
        yield ToolResponse(
            content=[], state=ToolResultState.ERROR,
        )

    async for _ in rsi_mw.on_acting(
        agent, {"tool_call": tool_call}, failing_handler,
    ):
        pass
    assert recorder.record.tool_calls[1].state == "error"
