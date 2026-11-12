# -*- coding: utf-8 -*-
"""运维三层记忆中间件 —— ``rsi_mem`` 的主连接器之一。

挂载方式（零源码修改）::

    agent = Agent(..., middlewares=[OpsMemoryMiddleware(ctx), ...])

Hook 行为（对齐 ReMeMiddleware 的成熟模式）：

- ``on_reply``：回合开始启动后台混合检索（static_control/both）；
  回合结束把**本轮增量**交给提取器写入短期记忆（自动建立→长期
  链接），并计数触发周期固化审查（dsh 的"回合内记忆审查"）；
- ``on_reasoning``：检索完成后以 HintBlock 注入"近期短期记忆 +
  检索结果（含链接展开）"；
- ``on_system_prompt``：注入基因组 always 片段 + L3 核心档案 +
  命中的肌肉记忆反射（dsh 的多轨注入）；
- ``list_tools``：memory_search / memory_note / memory_confirm /
  memory_link / memory_status。
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Callable, TYPE_CHECKING

from agentscope._logging import logger
from agentscope.message import AssistantMsg, HintBlock, Msg
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatModelBase

from ..context import RSIMemContext
from ..memory.tools import build_memory_tools

if TYPE_CHECKING:
    from agentscope.agent import Agent
    from agentscope.tool import ToolBase

_MEMORY_MSG_NAME = "ops_memory"

_TOOL_INSTRUCTIONS = (
    "## 三层运维记忆\n"
    "你有跨会话的三层记忆：肌肉记忆（SOP/条件反射，命中会自动注入）、"
    "短期记忆（本会话临时）、长期记忆（环境知识/历史处置）。\n"
    "- 涉及历史经验、环境事实、既有 SOP 时先调用 `memory_search`；\n"
    "- 用户要求『记一下/沉淀为SOP』时调用 `memory_note`（需确认后生效）；\n"
    "- 处理结果好坏可用 `rsi_feedback` 反馈，驱动自我改进。"
)


def _extract_user_text(inputs) -> str | None:
    """从 on_reply inputs 提取用户文本（与 ReMe 同款语义）。"""
    if inputs is None:
        return None
    from agentscope.event import ExternalExecutionResultEvent, \
        UserConfirmResultEvent

    if isinstance(
        inputs, (ExternalExecutionResultEvent, UserConfirmResultEvent),
    ):
        return None
    msgs = inputs if isinstance(inputs, list) else [inputs]
    texts = []
    for m in msgs:
        if isinstance(m, Msg) and m.role == "user":
            text = m.get_text_content()
            if text:
                texts.append(text)
    return "\n".join(texts) if texts else None


class OpsMemoryMiddleware(MiddlewareBase):
    """运维三层记忆中间件（AgentScope 官方中间件扩展点）。"""

    def __init__(self, context: RSIMemContext | None = None) -> None:
        self.ctx = context or RSIMemContext()
        # 会话级运行态：当前查询（供 on_system_prompt 反射匹配）、
        # 后台检索任务、轮次计数、写回任务
        self._current_query: dict[object, str] = {}
        self._retrieval_tasks: dict[object, asyncio.Task] = {}
        self._turn_counts: dict[object, int] = {}

    # ------------------------------------------------------------------
    def _session_of(self, agent: "Agent"):
        return getattr(getattr(agent, "state", None), "session_id", None)

    def _agent_model(self, agent: "Agent") -> ChatModelBase | None:
        model = getattr(agent, "model", None)
        return model if isinstance(model, ChatModelBase) else None

    # ------------------------------------------------------------------
    # on_reply
    # ------------------------------------------------------------------
    async def on_reply(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        session = self._session_of(agent)
        query = _extract_user_text(input_kwargs.get("inputs"))

        stale = self._retrieval_tasks.pop(session, None)
        if stale is not None and not stale.done():
            stale.cancel()
        if query:
            self._current_query[session] = query
        mode = self.ctx.config.long_term.mode
        if mode != "agent_control" and query:
            self._retrieval_tasks[session] = asyncio.create_task(
                self._retrieve_hint(query, session),
            )

        pre_ids = {
            m.id for m in agent.state.context if isinstance(m, Msg)
        }
        try:
            async for item in next_handler(**input_kwargs):
                yield item
        finally:
            task = self._retrieval_tasks.pop(session, None)
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            # —— 回合结束：写回 + 周期审查（后台执行，不阻塞回复）——
            increment = [
                m for m in agent.state.context
                if isinstance(m, Msg) and m.id not in pre_ids
                and getattr(m, "name", None) != _MEMORY_MSG_NAME
            ]
            model = self._agent_model(agent)
            asyncio.create_task(self._turn_postwork(
                increment, session, model,
            ))

    async def _retrieve_hint(self, query: str, session) -> tuple[str, list[str]]:
        """后台组装 reasoning 注入文本与实际注入的记忆 id 列表。"""
        try:
            genome = await self.ctx.db.active_genome()
            block, ids = await self.ctx.tiers.reasoning_hint_block(
                query=query,
                session_id=session,
                genome=genome,
                exclude_sessions=(session,) if session else (),
            )
            return block, ids
        except Exception as e:  # noqa: BLE001
            logger.warning("memory retrieval failed: %s", e)
            return "", []

    async def _turn_postwork(
        self,
        increment: list[Msg],
        session,
        model: ChatModelBase | None,
    ) -> None:
        """回合后处理：提取短期记忆；每 N 轮触发固化审查。"""
        try:
            has_exchange = any(
                m.role == "assistant" and m.get_text_content()
                for m in increment
            )
            if has_exchange:
                await self.ctx.extractor.extract_turn(
                    increment, session_id=session, model=model,
                )
            turns = self._turn_counts.get(session, 0) + 1
            self._turn_counts[session] = turns
            if turns % self.ctx.config.review.every_n_turns == 0:
                traces = await self.ctx.db.list_traces(limit=60)
                await self.ctx.consolidator.review_session(
                    session, traces, model=model,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("memory turn postwork failed: %s", e)

    # ------------------------------------------------------------------
    # on_reasoning
    # ------------------------------------------------------------------
    async def on_reasoning(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        session = self._session_of(agent)
        task = self._retrieval_tasks.get(session)
        if task is not None and task.done():
            self._retrieval_tasks.pop(session, None)
            try:
                hint, injected_ids = task.result()
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                hint, injected_ids = "", []
            if hint:
                # 暂存本轮注入的记忆 id，RSIMiddleware 在轨迹结算时取走
                self.ctx.engine.note_injected(session, injected_ids)
                agent.state.context.append(
                    AssistantMsg(
                        name=_MEMORY_MSG_NAME,
                        content=[HintBlock(hint=hint)],
                    ),
                )
        async for event in next_handler(**input_kwargs):
            yield event

    # ------------------------------------------------------------------
    # on_system_prompt
    # ------------------------------------------------------------------
    async def on_system_prompt(
        self,
        agent: "Agent",
        current_prompt: str,
    ) -> str:
        session = self._session_of(agent)
        query = self._current_query.get(session, "")
        try:
            genome = await self.ctx.db.active_genome()
            block = await self.ctx.tiers.system_prompt_block(
                query=query or (current_prompt[:200]),
                genome=genome,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("system prompt memory block failed: %s", e)
            block = ""
        parts = [current_prompt]
        if block:
            parts.append(block)
        if self.ctx.config.long_term.mode != "static_control":
            parts.append(_TOOL_INSTRUCTIONS)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    async def list_tools(self) -> "list[ToolBase]":
        if self.ctx.config.long_term.mode == "static_control":
            return []
        return build_memory_tools(self.ctx.tiers)

    async def get_middleware_key(self) -> str:
        return "ops_memory"

    async def close(self) -> None:
        await self.ctx.close()
