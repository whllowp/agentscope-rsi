# -*- coding: utf-8 -*-
"""app service（FastAPI）接入工厂。

AgentScope 的 ``create_app(extra_agent_middlewares=...)`` 钩子接受一个
工厂：按 (user_id, agent_id, session_id, workspace...) 构造每个 Agent
的中间件列表。这里返回共享同一个 :class:`RSIMemContext` 的两个中间件
（一个 context 一个数据库连接，跨 Agent 并发安全）。

用法（自己的服务入口，不动 agentscope 源码）::

    from agentscope.app import create_app
    from rsi_mem.service_factory import build_middleware_factory, ctx

    app = create_app(..., extra_agent_middlewares=build_middleware_factory())
    # 退出时 await ctx.close()
"""
from __future__ import annotations

from typing import Any, Callable

from .config import RSIMemConfig
from .connector import OpsMemoryMiddleware, RSIMiddleware
from .context import RSIMemContext

_ctx: RSIMemContext | None = None


def get_service_context(config: RSIMemConfig | None = None) -> RSIMemContext:
    """进程级单例 context（服务模式共享一份记忆库与 RSI 引擎）。"""
    global _ctx
    if _ctx is None:
        _ctx = RSIMemContext(config)
    return _ctx


async def close_service_context() -> None:
    global _ctx
    if _ctx is not None:
        await _ctx.close()
        _ctx = None


def build_middleware_factory(
    config: RSIMemConfig | None = None,
) -> Callable[..., Any]:
    """构造 ``extra_agent_middlewares`` 兼容的中间件工厂。"""

    ctx = get_service_context(config)

    def factory(*_args: Any, **_kwargs: Any):
        return [OpsMemoryMiddleware(ctx), RSIMiddleware(ctx)]

    return factory
