# -*- coding: utf-8 -*-
"""Agent 组装入口：把 rsi_mem 的全部连接器挂到一个 Agent 上。

镜像 ``examples/console/main.py`` 的组装方式，但完全不动 agentscope
源码；额外把**肌肉记忆镜像目录**作为技能加载器挂进 Toolkit —— SOP
固化为肌肉记忆后，下一轮启动即成为可加载技能。
"""
from __future__ import annotations

import os
from typing import Literal

from agentscope.agent import Agent
from agentscope.credential import (
    CredentialBase,
    DashScopeCredential,
    OpenAICredential,
)
from agentscope.formatter import FormatterBase
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase, ChatResponse, FinishedReason
from agentscope.model import ChatUsage
from agentscope.skill import LocalSkillLoader
from agentscope.tool import Toolkit
from agentscope.workspace import LocalWorkspace

from .config import RSIMemConfig
from .connector import OpsMemoryMiddleware, RSIMiddleware
from .context import RSIMemContext

DEFAULT_SYSTEM_PROMPT = (
    "你是一名资深运维工程师（SRE）助手。使用提供的工具完成巡检、"
    "诊断与处置建议；对生产变更保持谨慎，先核实再动手。"
)


class MockChatModel(ChatModelBase):
    """离线假模型（demo ``--backend mock`` 用）。

    读取注入上下文中的记忆区块并复述，验证整条链路（注入→提取→
    检索→轨迹→RSI）离线可跑。
    """

    def __init__(self) -> None:
        super().__init__(
            credential=CredentialBase(),
            model="rsi-mem-mock",
            parameters=self.Parameters(),
            stream=False,
        )

        class _EchoFormatter(FormatterBase):
            async def format(self, msgs):
                return [
                    {"role": m.role,
                     "content": m.get_text_content() or ""}
                    for m in msgs
                ]

        self.formatter = _EchoFormatter()

    async def _call_api(self, model, messages, tools=None, tool_choice=None,
                        **kwargs) -> ChatResponse:
        memory_hits = sum(
            1 for m in messages
            if "记忆" in (m.get_text_content() or "")
        )
        text = (
            f"(mock) 收到 {len(messages)} 条消息；"
            f"检测到 {memory_hits} 处记忆注入。"
            "建议：查看磁盘与日志后决定处置方案。"
        )
        return ChatResponse(
            content=[TextBlock(type="text", text=text)],
            is_last=True,
            finished_reason=FinishedReason.COMPLETED,
            usage=ChatUsage(input_tokens=16, output_tokens=8, time=0.01),
        )


def build_chat_model(
    backend: Literal["mock", "dashscope", "openai"] = "dashscope",
    model: str | None = None,
    base_url: str | None = None,
) -> ChatModelBase:
    """按后端构造 ChatModel（零源码修改，全部走官方模型类）。"""
    if backend == "mock":
        return MockChatModel()
    if backend == "dashscope":
        from agentscope.model import DashScopeChatModel

        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        return DashScopeChatModel(
            credential=DashScopeCredential(api_key=api_key),
            model=model or "qwen3.7-max",
            stream=True,
        )
    if backend == "openai":
        from agentscope.model import OpenAIChatModel

        api_key = os.environ.get("OPENAI_API_KEY", "")
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        cred_kwargs: dict = {"api_key": api_key}
        if url:
            cred_kwargs["base_url"] = url
        return OpenAIChatModel(
            credential=OpenAICredential(**cred_kwargs),
            model=model or "gpt-4o",
            stream=True,
        )
    raise ValueError(f"unknown backend: {backend}")


async def build_ops_agent(
    *,
    model: ChatModelBase,
    workspace: LocalWorkspace,
    config: RSIMemConfig | None = None,
    context: RSIMemContext | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    name: str = "OpsFriday",
) -> tuple[Agent, RSIMemContext]:
    """在**已打开**的 workspace 上组装带三层记忆 + RSI 的运维 Agent。

    调用方负责 workspace 生命周期::

        async with LocalWorkspace(workdir=...) as ws:
            agent, ctx = await build_ops_agent(model=..., workspace=ws)
            await launch_console(agent)

    Returns:
        ``(agent, context)`` —— context 供应用退出时 ``close()``。
    """
    ctx = context or RSIMemContext(config)
    muscle_loader = LocalSkillLoader(
        os.path.join(ctx.config.data_dir, "mirror", "muscle"),
        scan_subdir=True,
    )
    agent = Agent(
        name=name,
        system_prompt=(
            system_prompt + "\n\n" + await workspace.get_instructions()
        ),
        model=model,
        toolkit=Toolkit(
            tools=await workspace.list_tools(),
            skills_or_loaders=await workspace.list_skills() + [
                muscle_loader,
            ],
        ),
        middlewares=[
            OpsMemoryMiddleware(ctx),
            RSIMiddleware(ctx),
        ],
        offloader=workspace,
    )
    return agent, ctx
