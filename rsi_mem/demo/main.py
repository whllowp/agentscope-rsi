# -*- coding: utf-8 -*-
"""运维 Agent 演示入口（镜像 ``examples/console/main.py``，零源码修改）。

离线冒烟（不需要任何 API key）::

    python -m rsi_mem.demo.main --backend mock

真实模型::

    export DASHSCOPE_API_KEY=sk-...
    python -m rsi_mem.demo.main --backend dashscope --model qwen3.7-max

    # 或任意 OpenAI 兼容服务
    export OPENAI_API_KEY=... OPENAI_BASE_URL=https://.../v1
    python -m rsi_mem.demo.main --backend openai --model gpt-4o
"""
import argparse
import asyncio
import os

from agentscope.console import launch_console
from agentscope.workspace import LocalWorkspace

from rsi_mem.agent_builder import build_chat_model
from rsi_mem.config import RSIMemConfig


async def main() -> None:
    """演示入口：组装 OpsAgent 并启动终端交互。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["mock", "dashscope", "openai"],
        default="mock",
        help="模型后端（mock 离线可跑）",
    )
    parser.add_argument("--model", default=None, help="模型名")
    parser.add_argument(
        "--base-url", default=None, help="OpenAI 兼容上游地址",
    )
    parser.add_argument(
        "--verbosity",
        choices=["quiet", "default", "debug"],
        default="default",
    )
    parser.add_argument(
        "--workdir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "workspace",
        ),
        help="Agent 工作区目录",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="rsi_mem 数据目录（默认 ./<workdir>/.rsi_mem）",
    )
    args = parser.parse_args()

    from rsi_mem.agent_builder import build_ops_agent

    config = RSIMemConfig(
        data_dir=args.data_dir
        or os.path.join(os.path.abspath(args.workdir), ".rsi_mem"),
    )
    model = build_chat_model(
        args.backend, model=args.model, base_url=args.base_url,
    )

    async with LocalWorkspace(workdir=args.workdir) as workspace:
        agent, ctx = await build_ops_agent(
            model=model, workspace=workspace, config=config,
        )
        try:
            await launch_console(agent, verbosity=args.verbosity)
        finally:
            await ctx.close()


if __name__ == "__main__":
    asyncio.run(main())
