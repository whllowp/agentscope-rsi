# -*- coding: utf-8 -*-
"""RSI 中间件 —— 轨迹录制与自改进工具的连接器。

挂载方式（零源码修改）::

    agent = Agent(..., middlewares=[
        OpsMemoryMiddleware(ctx),   # 共享同一个 RSIMemContext
        RSIMiddleware(ctx),
    ])

Hook 行为（全部"观测不干预"，模型/工具调用原样透传）：

- ``on_reply``：开轨迹 → 结算 outcome 评分（最终答复 / 超迭代 / 中断）
  → 记录本轮注入记忆 id（来自记忆中间件）→ 计数触发后台 dream；
- ``on_acting``：记录工具调用（名称 / 摘要 / 状态 / 耗时）；
- ``on_model_call``：记录模型调用 token 与延迟（流式与非流式两种
  返回形态都处理）；
- ``on_check_permission``：记录权限决策（安全目标信号）；
- ``list_tools``：rsi_status / rsi_feedback / rsi_evolve / rsi_history。
"""
from __future__ import annotations

import time
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    TYPE_CHECKING,
)

from agentscope._logging import logger
from agentscope.message import Msg
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatModelBase, ChatResponse

from ..context import RSIMemContext
from ..rsi.tools import build_rsi_tools
from ..rsi.trace import ModelCallStat, TraceRecorder
from .memory_middleware import _extract_user_text

if TYPE_CHECKING:
    from agentscope.agent import Agent
    from agentscope.permission import PermissionDecision
    from agentscope.tool import ToolBase


def _session_of(agent: "Agent"):
    return getattr(getattr(agent, "state", None), "session_id", None)


class RSIMiddleware(MiddlewareBase):
    """RSI 轨迹录制 + 自改进工具中间件。"""

    def __init__(self, context: RSIMemContext | None = None) -> None:
        self.ctx = context or RSIMemContext()
        self._recorders: dict[Any, TraceRecorder] = {}
        self._last_msgs: dict[Any, Msg] = {}

    # ------------------------------------------------------------------
    # on_reply：轨迹生命周期
    # ------------------------------------------------------------------
    async def on_reply(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        session = _session_of(agent)
        query = _extract_user_text(input_kwargs.get("inputs")) or ""
        recorder = TraceRecorder(
            self.ctx.db,
            self.ctx.config,
            session_id=session,
            query=query,
            events=getattr(self.ctx, "events", None),
        )
        self._recorders[session] = recorder
        finished_reason = ""
        try:
            async for item in next_handler(**input_kwargs):
                if isinstance(item, Msg):
                    self._last_msgs[session] = item
                    reason = getattr(item, "finished_reason", None)
                    if reason is not None:
                        finished_reason = str(
                            getattr(reason, "value", reason),
                        )
                yield item
        finally:
            self._recorders.pop(session, None)
            last = self._last_msgs.pop(session, None)
            try:
                digest = (
                    last.get_text_content() if last is not None else ""
                ) or ""
                await recorder.finish(
                    digest=digest,
                    has_answer=bool(digest.strip()),
                    max_iters_exceeded=(
                        finished_reason == "exceed_max_iters"
                    ),
                    interrupted=(finished_reason == "interrupted"),
                    injected_memory_ids=self.ctx.engine.pop_injected(
                        session,
                    ),
                )
                if self.ctx.config.rsi.enabled:
                    self.ctx.engine.on_trace_finished()
            except Exception as e:  # noqa: BLE001
                logger.warning("trace finish failed: %s", e)

    # ------------------------------------------------------------------
    # on_acting：工具调用观测
    # ------------------------------------------------------------------
    async def on_acting(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        session = _session_of(agent)
        recorder = self._recorders.get(session)
        tool_call = input_kwargs.get("tool_call")
        name = getattr(tool_call, "name", "") or ""
        if recorder is None:
            async for chunk in next_handler(**input_kwargs):
                yield chunk
            return

        started = time.perf_counter()
        state = "ok"
        digest = ""
        try:
            async for chunk in next_handler(**input_kwargs):
                state = self._chunk_state(chunk, state)
                digest = self._chunk_digest(chunk, digest)
                yield chunk
        finally:
            recorder.add_tool_call(
                name=name,
                input_digest=digest,
                state=state,
                latency_s=time.perf_counter() - started,
            )

    @staticmethod
    def _chunk_state(chunk: Any, current: str) -> str:
        state = getattr(chunk, "state", None)
        if state is None:
            return current
        s = str(getattr(state, "value", state))
        if s == "success":
            return "ok" if current == "ok" else current
        if s in ("error", "denied", "interrupted"):
            mapping = {"denied": "error", "interrupted": "interrupted"}
            return mapping.get(s, "error")
        return current

    @staticmethod
    def _chunk_digest(chunk: Any, current: str) -> str:
        """从最终响应提取简短摘要。"""
        content = getattr(chunk, "content", None)
        if not content:
            return current
        texts = [
            getattr(b, "text", "") for b in content
            if getattr(b, "type", "") == "text"
        ]
        merged = " ".join(t for t in texts if t)
        return (current + " " + merged).strip()[:200] if merged else current

    # ------------------------------------------------------------------
    # on_model_call：token/延迟观测（流式与非流式）
    # ------------------------------------------------------------------
    async def on_model_call(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[
            ..., Awaitable["ChatResponse | AsyncGenerator[ChatResponse]"],
        ],
    ):
        session = _session_of(agent)
        recorder = self._recorders.get(session)
        model: ChatModelBase | None = input_kwargs.get("current_model")
        started = time.perf_counter()

        def _stat(usage) -> ModelCallStat:
            return ModelCallStat(
                model=getattr(model, "model", "") or "",
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                latency_s=round(time.perf_counter() - started, 3),
            )

        result = await next_handler(**input_kwargs)
        if recorder is None:
            return result
        if isinstance(result, ChatResponse):
            recorder.add_model_call(_stat(result.usage))
            return result
        if hasattr(result, "__aiter__"):
            async def _wrapped():
                async for r in result:
                    if getattr(r, "is_last", False) and getattr(
                        r, "usage", None,
                    ):
                        recorder.add_model_call(_stat(r.usage))
                    yield r

            return _wrapped()
        return result

    # ------------------------------------------------------------------
    # on_check_permission：安全信号
    # ------------------------------------------------------------------
    async def on_check_permission(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., Awaitable[Any]],
    ) -> Any:
        session = _session_of(agent)
        recorder = self._recorders.get(session)
        decision = await next_handler(**input_kwargs)
        if recorder is not None and decision is not None:
            tool = input_kwargs.get("tool")
            behavior = getattr(
                getattr(decision, "behavior", None), "value", None,
            ) or str(getattr(decision, "behavior", ""))
            recorder.add_permission(
                getattr(tool, "name", "") or "", behavior,
            )
        return decision

    # ------------------------------------------------------------------
    async def list_tools(self) -> "list[ToolBase]":
        if not self.ctx.config.rsi.enabled:
            return []
        return build_rsi_tools(self.ctx)

    async def get_middleware_key(self) -> str:
        return "rsi_engine"

    async def close(self) -> None:
        await self.ctx.close()
