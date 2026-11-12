# -*- coding: utf-8 -*-
"""模型层代理：``TraceProxyChatModel`` 包装任意 ChatModelBase。

用途：无法挂中间件的部署（app service 里由存储配置直接实例化模型、
或只想观测模型调用成本/延迟）—— 在模型对象外再包一层，__call__ 级
透传并记录用量。格式化（formatter）等能力直接沿用内层模型。
"""
from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Callable

from agentscope.message import Msg
from agentscope.model import ChatModelBase, ChatResponse

from ..rsi.trace import ModelCallStat


class _UsageSink:
    """收集模型调用统计的最小接口（RSIMiddleware 的 recorder 与
    HTTP 代理各自的实现都满足它）。"""

    def on_model_stat(self, stat: ModelCallStat) -> None:
        raise NotImplementedError


class TraceProxyChatModel(ChatModelBase):
    """装饰任意内层 ChatModel 的观测代理。

    Example::

        inner = DashScopeChatModel(credential=..., model=...)
        proxy = TraceProxyChatModel(inner, sink=my_sink)
        agent = Agent(..., model=proxy)

    ``__call__`` 完全透传内层模型（含其重试/流式累积），外层不再
    包一层重试；``_call_api`` 不会被基类调用到（``__call__`` 已覆写）。
    """

    def __init__(
        self,
        inner: ChatModelBase,
        sink: _UsageSink | None = None,
    ) -> None:
        super().__init__(
            credential=inner.credential,
            model=inner.model,
            parameters=inner.parameters,
            stream=inner.stream,
            max_retries=inner.max_retries,
            retry_delay=inner.retry_delay,
            context_size=inner.context_size,
        )
        self.inner = inner
        self.sink = sink
        # 沿用内层的 formatter / provider 专属能力
        self.formatter = getattr(inner, "formatter", None)

    async def _call_api(self, model, messages, tools=None, tool_choice=None,
                        **kwargs) -> ChatResponse:
        # __call__ 被覆写后不会走到这里；保守起见直接委托内层。
        return await self.inner._call_api(  # pylint: disable=protected-access
            model, messages, tools=tools, tool_choice=tool_choice, **kwargs,
        )

    async def __call__(
        self,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        started = time.perf_counter()
        result = await self.inner(
            messages, tools=tools, tool_choice=tool_choice, **kwargs,
        )
        self._maybe_sink(getattr(result, "usage", None), started)

        if hasattr(result, "__aiter__"):
            async def _wrapped() -> AsyncGenerator[ChatResponse, None]:
                async for r in result:
                    if getattr(r, "is_last", False):
                        self._maybe_sink(
                            getattr(r, "usage", None), started,
                        )
                    yield r

            return _wrapped()
        return result

    def _maybe_sink(self, usage, started: float) -> None:
        if self.sink is None:
            return
        self.sink.on_model_stat(ModelCallStat(
            model=self.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_s=round(time.perf_counter() - started, 3),
        ))

    def __getattr__(self, name: str) -> Any:
        """其余属性（provider 专属字段等）转给内层模型。"""
        if name.startswith("_") or name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)
