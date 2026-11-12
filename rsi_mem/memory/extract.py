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
from .tiers import TierManager

_EXTRACT_SYSTEM = (
    "你是运维 Agent 的记忆提取器。从下面这轮对话中提取值得短期记住的"
    "信息（事实、事件、结论、待办、环境状态变化）。要求：\n"
    "1. 只提取对未来轮次有用的信息，忽略客套与过程细节；\n"
    "2. 每条给 title（≤20字）、content（≤200字）、kind"
    "（fact/incident/decision/observation/todo 之一）、confidence(0-1)；\n"
    "3. 只输出 JSON 数组，如 [{\"title\":..., \"content\":...,"
    " \"kind\":..., \"confidence\":0.8}]；无值得记录的内容输出 []。"
)

_MAX_TURNS_CHARS = 6000
_MAX_CANDIDATES = 5


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

    def __init__(self, config: RSIMemConfig, tiers: TierManager) -> None:
        self._config = config
        self._tiers = tiers

    async def extract_turn(
        self,
        increment: list[Msg],
        *,
        session_id: str | None,
        trace_id: str | None = None,
        model: ChatModelBase | None = None,
    ) -> list[MemoryRecord]:
        """提取并写入短期记忆（含自动长期链接）。"""
        convo = _render_increment(increment)
        if not convo.strip():
            return []
        candidates = await self._extract_candidates(convo, model)
        records: list[MemoryRecord] = []
        for cand in candidates[:_MAX_CANDIDATES]:
            try:
                rec = await self._tiers.shortterm.record(
                    session_id=session_id,
                    title=str(cand.get("title", ""))[:200],
                    content=str(cand.get("content", ""))[:2000],
                    kind=str(cand.get("kind", "observation")),
                    confidence=float(cand.get("confidence", 0.5)),
                    source={
                        "origin": "turn_extract",
                        "trace_id": trace_id,
                    },
                )
                records.append(rec)
            except Exception as e:  # noqa: BLE001
                logger.warning("short-term record failed: %s", e)
        return records

    async def _extract_candidates(
        self, convo: str, model: ChatModelBase | None,
    ) -> list[dict]:
        model = model or self._config.backend_model
        if model is None:
            return self._rule_fallback(convo)
        try:
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
        """规则兜底：用户问题 + 首条助手答复各记一条 observation。"""
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
            })
        if assistant_lines:
            out.append({
                "title": "本轮结论",
                "content": assistant_lines[-1][:200],
                "kind": "observation",
                "confidence": 0.6,
            })
        return out
