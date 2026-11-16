# -*- coding: utf-8 -*-
"""回合结束提取器：把一轮对话增量蒸馏为短期记忆候选。

有后台模型时用 LLM 提取（结构化 JSON）；离线 / 无模型 / 解析失败时
退化为规则提取（用户问题 + 最终答复摘要），保证链路永远可用。
"""
from __future__ import annotations

from typing import Any

import json_repair
from agentscope._logging import logger
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase

from ..config import RSIMemConfig
from ..storage import MemoryRecord
from ..telemetry import LangfuseBridge, NOOP_BRIDGE
from .tiers import TierManager

_EXTRACT_SYSTEM = (
    "你是运维 Agent 的记忆提取器。从下面这轮对话中提取值得短期记住的"
    "信息（事实、事件、结论、待办、环境状态变化）。要求：\n"
    "1. 只提取对未来轮次有用的信息，忽略客套与过程细节；只记当前上下文"
    "之外仍成立的信息，每轮可重读的临时状态不记（确需记录时标 "
    "volatile=true）；\n"
    "2. 每条给 title（≤20字）、content（≤200字）、kind"
    "（fact/incident/decision/observation/todo 之一）、confidence(0-1)；\n"
    "3. 每条附 salience 显著性标注（记忆即预测）：kind ∈ strategy/plan/"
    "tested_rule/env_fact/state/preference，volatile 标记易变状态，"
    "rationale 一句话说明为何值得记，predicted_use 描述预测未来什么场景"
    "会用到它，predicted_value(0-1) 是该使用预测的价值估计——写入前先"
    "自问「这条记忆将来会被检索或注入吗」，预计不上的给低分或不记；\n"
    "4. 只输出 JSON 数组，如 [{\"title\":..., \"content\":...,"
    " \"kind\":..., \"confidence\":0.8, \"salience\": {\"kind\":...,"
    " \"volatile\":false, \"rationale\":..., \"predicted_use\":...,"
    " \"predicted_value\":0.7}}]；无值得记录的内容输出 []。"
)

_MAX_TURNS_CHARS = 6000
_MAX_CANDIDATES = 5
_DEDUP_RECENT_N = 20
"""规则兜底去重窗口：近 N 条短期记忆内同内容增量不重复写入。"""


def _dedup_key(content: str) -> str:
    """内容归一化去重键：strip + casefold + 空白折叠。

    刻意不含 title —— 规则兜底的标题恒为「本轮问题/本轮结论」，按键去
    重会把不同内容误杀成重复。
    """
    return " ".join(content.split()).casefold()


def _build_source(cand: dict, trace_id: str | None) -> dict:
    """构造溯源 dict：origin/trace_id + LLM 的 salience 原样透传。

    脏 salience（非 dict / 字段类型错）也透传 —— 记录照写，读取侧
    :func:`rsi_mem.memory.salience.tag_of` 返 None 容忍。
    """
    source: dict = {"origin": "turn_extract", "trace_id": trace_id}
    salience = cand.get("salience")
    if salience is not None:
        source["salience"] = salience
    return source


def _render_increment(increment: list[Msg]) -> str:
    lines = []
    for m in increment:
        text = m.get_text_content() or ""
        if not text:
            continue
        who = m.role
        name = getattr(m, "name", "")
        lines.append(f"[{who}:{name}] {text[:600]}")
    return "\n".join(lines)[:_MAX_TURNS_CHARS]


def _first_text(resp: Any) -> str:
    """从 ChatResponse 提取纯文本。"""
    for block in getattr(resp, "content", []):
        if getattr(block, "type", "") == "text" and block.text:
            return block.text
    return ""


class TurnExtractor:
    """回合增量的记忆提取。"""

    def __init__(
        self,
        config: RSIMemConfig,
        tiers: TierManager,
        *,
        langfuse: LangfuseBridge | None = None,
    ) -> None:
        self._config = config
        self._tiers = tiers
        self._langfuse = langfuse or NOOP_BRIDGE

    async def extract_turn(
        self,
        increment: list[Msg],
        *,
        session_id: str | None,
        trace_id: str | None = None,
        model: ChatModelBase | None = None,
    ) -> list[MemoryRecord]:
        """提取并写入短期记忆（自动长期链接 + 显著性标注 + 兜底去重）。"""
        convo = _render_increment(increment)
        if not convo.strip():
            return []
        candidates = await self._extract_candidates(convo, model)
        # 规则兜底去重：近 N 条短期记忆内同内容（归一化后）不重复写入
        seen_keys = {
            _dedup_key(m.content)
            for m in await self._tiers.shortterm.recent(
                session_id, _DEDUP_RECENT_N,
            )
        }
        records: list[MemoryRecord] = []
        for cand in candidates[:_MAX_CANDIDATES]:
            content = str(cand.get("content", ""))[:2000]
            key = _dedup_key(content)
            if key and key in seen_keys:
                continue
            seen_keys.add(key)
            try:
                rec = await self._tiers.shortterm.record(
                    session_id=session_id,
                    title=str(cand.get("title", ""))[:200],
                    content=content,
                    kind=str(cand.get("kind", "observation")),
                    confidence=float(cand.get("confidence", 0.5)),
                    source=_build_source(cand, trace_id),
                )
                records.append(rec)
            except Exception as e:  # noqa: BLE001
                logger.warning("short-term record failed: %s", e)
        # —— L0 原始证据：整轮增量的溯源快照（有产出才写）——
        # 这是 L0 层的生产写入路径，也是 U3 L0 回收插槽的输入面；
        # 未被链接引用且超量时由回收提议归档，生命周期闭环。
        if records:
            try:
                await self._tiers.longterm.add_raw_evidence(
                    text=convo[:2000],
                    source={
                        "origin": "turn_extract",
                        "trace_id": trace_id,
                        "session_id": session_id,
                    },
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("l0 evidence write failed: %s", e)
        return records

    async def _extract_candidates(
        self, convo: str, model: ChatModelBase | None,
    ) -> list[dict]:
        model = model or self._config.backend_model
        if model is None:
            return self._rule_fallback(convo)
        try:
            with self._langfuse.observe("memory_extract"):
                resp = await model(
                    messages=[
                        Msg(
                            name="system",
                            role="system",
                            content=[TextBlock(type="text",
                                               text=_EXTRACT_SYSTEM)],
                        ),
                        Msg(
                            name="user",
                            role="user",
                            content=[TextBlock(type="text", text=convo)],
                        ),
                    ],
                )
            parsed = json_repair.loads(_first_text(resp))
            if isinstance(parsed, list):
                return [c for c in parsed if isinstance(c, dict)]
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM extract failed (%s); rule fallback.", e)
        return self._rule_fallback(convo)

    @staticmethod
    def _rule_fallback(convo: str) -> list[dict]:
        """规则兜底：用户问题 + 首条助手答复各记一条 observation。

        兜底产物默认 kind=state / volatile=True —— 每轮可重读的会话状态，
        固化优先级低；去重交给 :meth:`extract_turn`（本方法是纯函数、
        无 DB 访问）。
        """
        out: list[dict] = []
        user_lines = [
            ln for ln in convo.splitlines() if ln.startswith("[user:")
        ]
        assistant_lines = [
            ln for ln in convo.splitlines()
            if ln.startswith("[assistant:") and "[memory" not in ln
        ]
        if user_lines:
            out.append({
                "title": "本轮问题",
                "content": user_lines[-1][:200],
                "kind": "observation",
                "confidence": 0.6,
                "salience": {
                    "kind": "state",
                    "volatile": True,
                    "rationale": "规则兜底：当前轮用户问题",
                    "predicted_use": "同话题后续轮次检索",
                    "predicted_value": 0.3,
                },
            })
        if assistant_lines:
            out.append({
                "title": "本轮结论",
                "content": assistant_lines[-1][:200],
                "kind": "observation",
                "confidence": 0.6,
                "salience": {
                    "kind": "state",
                    "volatile": True,
                    "rationale": "规则兜底：当前轮答复结论",
                    "predicted_use": "同话题后续轮次检索",
                    "predicted_value": 0.35,
                },
            })
        return out
