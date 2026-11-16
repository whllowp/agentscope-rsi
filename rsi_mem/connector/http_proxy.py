# -*- coding: utf-8 -*-
"""OpenAI 兼容 HTTP 代理（TencentDB MemoryProxy 风格的零代码接入面）。

任意 OpenAI 客户端（包括 AgentScope 的 ``OpenAIChatModel``、app
service、curl）把 ``base_url`` 指向本服务即可获得：

- **记忆注入**：请求的 system 消息前拼入三层记忆上下文（基因组片段 +
  L3 核心档案 + 命中肌肉反射 + 混合检索 top-k）；
- **轨迹采集**：把每轮请求/用量记为一条 RSI 轨迹（模型侧；看不到
  工具执行，完整轨迹请用中间件模式）。

启动::

    uvicorn rsi_mem.connector.http_proxy:app --port 8130

    # 客户端侧
    OpenAICredential(base_url="http://127.0.0.1:8130/v1", api_key="x")

上游真实服务用 ``RSI_MEM_PROXY_UPSTREAM_URL`` /
``RSI_MEM_PROXY_UPSTREAM_KEY`` 环境变量或 :class:`ProxyConfig` 指定。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from agentscope._logging import logger
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..context import RSIMemContext
from ..rsi.trace import ModelCallStat, TraceRecorder, score_outcome
from ..storage import OutcomeRecord

app = FastAPI(title="rsi-mem-proxy", version="0.1.0")

_CTX: RSIMemContext | None = None
_CLIENT: httpx.AsyncClient | None = None
_PROXY_TURNS = 0
"""代理侧回合计数：每 every_n_turns 条轨迹触发一次固化审查
（镜像 memory_middleware 的周期审查，纯代理部署可达同等防御）。"""


def init_proxy(context: RSIMemContext | None = None) -> RSIMemContext:
    """初始化代理（默认路由处理器启动时懒加载调用）。"""
    global _CTX, _CLIENT
    if _CTX is None:
        _CTX = context or RSIMemContext()
    if _CLIENT is None:
        _CLIENT = httpx.AsyncClient(timeout=300.0)
    return _CTX


async def shutdown_proxy() -> None:
    global _CTX, _CLIENT
    if _CLIENT is not None:
        await _CLIENT.aclose()
        _CLIENT = None
    if _CTX is not None:
        await _CTX.close()
        _CTX = None


# ---------------------------------------------------------------------------
# 记忆上下文注入
# ---------------------------------------------------------------------------

async def _build_memory_prefix(query: str) -> tuple[str, list[str]]:
    """组装注入到 system 消息前部的记忆上下文（U4 紧凑路径）。

    与 ``system_prompt_block`` 共用同一渲染路径（基因组片段 + L3 超取
    过滤 + 顶部小索引 + 命中反射，同一 ``l3_char_budget`` 合计预算）。

    Returns:
        ``(prefix, 注入的记忆 id 列表)`` —— ids 经
        ``_record_trace → _save_trace(used_memory_ids)`` 落进轨迹，
        是代理模式下使用型验证（trace_verified）与污染防御的数据
        来源；失败走空串降级。
    """
    ctx = init_proxy()
    try:
        genome = await ctx.db.active_genome()
        prefix, ids = await ctx.tiers.compact_context_block(
            query=query, genome=genome,
        )
        return prefix, list(ids)
    except Exception as e:  # noqa: BLE001
        logger.warning("proxy memory prefix failed: %s", e)
        return "", []


def _extract_query(body: dict[str, Any]) -> str:
    """从 OpenAI 请求体提取查询文本（最后一条用户消息）。"""
    messages = body.get("messages") or []
    texts = [
        m.get("content", "") for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
        and isinstance(m.get("content"), str)
    ]
    return texts[-1] if texts else ""


def _inject_system(body: dict[str, Any], prefix: str) -> dict[str, Any]:
    """把 prefix 注入 system 消息（无则新建）。"""
    messages = body.get("messages") or []
    if messages and isinstance(messages[0], dict) \
            and messages[0].get("role") == "system":
        content = messages[0].get("content", "")
        messages[0]["content"] = f"{prefix}\n\n{content}"
    else:
        messages.insert(0, {"role": "system", "content": prefix})
    body["messages"] = messages
    return body


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> dict[str, str]:
    init_proxy()
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    ctx = init_proxy()
    base, key = ctx.config.effective_proxy_upstream()
    if not base:
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": "upstream not configured "
                           "(RSI_MEM_PROXY_UPSTREAM_URL)",
                "type": "proxy_error",
            }},
        )

    body = await request.json()
    query = _extract_query(body)
    started = time.perf_counter()

    prefix = ""
    injected_ids: list[str] = []
    if ctx.config.proxy.inject_into_system and query:
        prefix, injected_ids = await _build_memory_prefix(query)
        if prefix:
            body = _inject_system(body, prefix)

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    upstream = f"{base}/v1/chat/completions"
    stream = bool(body.get("stream"))

    if not stream:
        resp = await _CLIENT.post(upstream, json=body, headers=headers)
        await _record_trace(ctx, query, resp, started, injected_ids)
        return JSONResponse(
            status_code=resp.status_code,
            content=resp.json(),
        )

    async def _relay():
        async with _CLIENT.stream(
            "POST", upstream, json=body, headers=headers,
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk
        await _record_stream_trace(ctx, query, started, injected_ids)

    return StreamingResponse(
        _relay(),
        media_type="text/event-stream",
    )


@app.get("/v1/models")
async def models():
    ctx = init_proxy()
    base, key = ctx.config.effective_proxy_upstream()
    if not base:
        return {"object": "list", "data": []}
    resp = await _CLIENT.get(
        f"{base}/v1/models", headers=(
            {"Authorization": f"Bearer {key}"} if key else {}
        ),
    )
    return JSONResponse(status_code=resp.status_code, content=resp.json())


# ---------------------------------------------------------------------------
# 轨迹采集
# ---------------------------------------------------------------------------

async def _record_trace(
    ctx: RSIMemContext,
    query: str,
    resp: httpx.Response,
    started: float,
    used_memory_ids: list[str] | None = None,
) -> None:
    try:
        usage = {}
        answer = ""
        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage") or {}
            choices = data.get("choices") or []
            if choices:
                answer = (
                    choices[0].get("message", {}).get("content") or ""
                )
        await _save_trace(ctx, query, usage, answer, started,
                          used_memory_ids)
    except Exception as e:  # noqa: BLE001
        logger.warning("proxy trace record failed: %s", e)


async def _record_stream_trace(
    ctx: RSIMemContext,
    query: str,
    started: float,
    used_memory_ids: list[str] | None = None,
) -> None:
    """流式请求没有用量信息，记延迟（best-effort）。

    流式完成（SSE 已回放给客户端）按「有答复」计 —— 否则流式部署
    score 恒 0.5 < 0.7，使用型验证（trace_verified）永远不可达。
    """
    try:
        await _save_trace(
            ctx, query, {}, "", started, used_memory_ids,
            has_answer=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("proxy stream trace failed: %s", e)


async def _save_trace(
    ctx: RSIMemContext,
    query: str,
    usage: dict[str, Any],
    answer: str,
    started: float,
    used_memory_ids: list[str] | None = None,
    has_answer: bool | None = None,
) -> None:
    global _PROXY_TURNS
    recorder = TraceRecorder(
        ctx.db, ctx.config, session_id="http-proxy", query=query,
    )
    recorder.add_model_call(ModelCallStat(
        model=str(usage.get("model", "")),
        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
        latency_s=round(time.perf_counter() - started, 3),
    ))
    if used_memory_ids:
        # 注入即使用：记入 used_memory_ids，供 U2 使用型验证
        # （trace_verified）与污染防御回溯
        recorder.note_used_memories(used_memory_ids)
    score = score_outcome(
        has_answer=(
            bool(answer.strip()) if has_answer is None else has_answer
        ),
        max_iters_exceeded=False,
        tool_error_ratio=0.0,
        tool_ok_count=0,
        interrupted=False,
    )
    recorder.record.digest = answer[:500]
    recorder.record.outcome = OutcomeRecord(
        score=round(score, 3),
        signals={"source": "http_proxy"},
    )
    await recorder.upsert()
    if ctx.config.rsi.enabled:
        ctx.engine.on_trace_finished()
    # —— 周期固化审查（镜像 memory_middleware 的计数触发）——
    _PROXY_TURNS += 1
    if _PROXY_TURNS % ctx.config.review.every_n_turns == 0:

        async def _consolidate() -> None:
            try:
                traces = await ctx.db.list_traces(limit=60)
                await ctx.consolidator.review_session(None, traces)
            except Exception as e:  # noqa: BLE001
                logger.warning("proxy consolidation failed: %s", e)

        asyncio.create_task(_consolidate())
