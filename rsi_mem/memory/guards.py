# -*- coding: utf-8 -*-
"""记忆污染防御（U2）：注入层验证分级 + 矛盾检测。

「写入有预测，注入有验证」—— 污染记忆的杀伤面在注入层：

- :func:`is_injectable` —— ``refuted`` 的记忆即使 status 仍为 active
  也不得注入（retrieval 的双保险）；
- :func:`l3_eligible` —— L3 核心档案准入**不依赖 confidence**：
  必须 trace_verified 或用户手写（``origin ∈ memory_note/user``），
  驻留上限计数另见 :mod:`rsi_mem.memory.consolidate`；
- :func:`find_contradictions` —— 相似标题（difflib ≥ 0.8）且关键
  事实互斥的记忆对；backend_model 存在时 LLM 复核，复核不通过不报。

验证态 :class:`VerificationState` 存 ``source["verification"]``；
读写对脏数据一律容忍（返默认/None，绝不抛异常）。
"""
from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from typing import Any

import json_repair
from agentscope._logging import logger
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase

from ..storage import (
    MemoryRecord,
    RSIMemDatabase,
    TraceRecord,
    VerificationState,
)

__all__ = [
    "VerificationState",
    "broken_link_reason",
    "find_contradictions",
    "flag_contradiction_pair",
    "is_injectable",
    "l3_eligible",
    "llm_recheck",
    "mark_stale",
    "set_verification",
    "verification_of",
]

_USER_ORIGINS = ("memory_note", "user")
"""用户手写来源：L3 准入的白名单（不依赖 confidence）。"""

_TITLE_SIMILARITY = 0.8
"""矛盾检测的标题相似度阈值（difflib SequenceMatcher）。"""

_FACT_RE = re.compile(
    r"([\w\u4e00-\u9fff%-]{1,24})\s*(?:[：:=]|是)\s*([^\s，。；,;]{1,40})",
)
"""关键事实抽取：「属性：值」/「属性=值」/「属性是值」。"""

_RECHECK_SYSTEM = (
    "你是记忆库的核查员。对照【轨迹】的提问与最终答复，判断【记忆】"
    "内容是否被该轨迹证实（信息一致、无编造、无过度概括）。"
    "只输出 JSON：{\"verified\": true/false}。"
)

_CONTRA_SYSTEM = (
    "你是记忆库的矛盾核查员。对照下面两条运维记忆，判断它们对同一"
    "关键事实的描述是否互相矛盾。只输出 JSON："
    "{\"contradiction\": true/false}。拿不准给 false。"
)


def _coerce_verification(payload: Any) -> VerificationState:
    """脏数据容忍：非 dict / 字段类型错 → 默认 unverified。"""
    if not isinstance(payload, dict):
        return VerificationState()
    try:
        return VerificationState(**payload)
    except Exception:  # noqa: BLE001 — pydantic ValidationError 等
        return VerificationState()


def verification_of(rec: MemoryRecord) -> VerificationState:
    """读回记录的验证态；缺失/脏数据返回默认 ``unverified``。"""
    return _coerce_verification((rec.source or {}).get("verification"))


def set_verification(
    rec: MemoryRecord,
    state: str,
    evidence: list[str] | str | None = None,
) -> MemoryRecord:
    """就地写入验证态（整体覆盖 evidence），返回同一记录。

    非法 state 字面量静默放弃（不抛），保持原状。
    """
    evid = (
        [str(e) for e in evidence] if isinstance(evidence, list)
        else ([evidence] if evidence else [])
    )
    try:
        vs = VerificationState(
            state=state,  # type: ignore[arg-type]
            evidence=evid,
            checked_at=time.time(),
        )
    except Exception:  # noqa: BLE001
        return rec
    rec.source.setdefault("verification", {})
    rec.source["verification"] = vs.model_dump()
    return rec


def mark_stale(
    rec: MemoryRecord, note: str = "",
) -> MemoryRecord:
    """漂移再验证：trace_verified 重置为 unverified，保留 evidence 历史。"""
    vs = verification_of(rec)
    evidence = list(vs.evidence)
    if note:
        evidence.append(f"stale:{note}")
    return set_verification(rec, "unverified", evidence)


def is_injectable(rec: MemoryRecord) -> bool:
    """注入资格：status ∈ active/promoted 且未被证伪（refuted）。"""
    return (
        rec.status in ("active", "promoted")
        and verification_of(rec).state != "refuted"
    )


def l3_eligible(rec: MemoryRecord) -> bool:
    """L3 核心档案准入：可注入，且 trace_verified 或用户手写。

    刻意**不看 confidence** —— 高置信不等于已验证（污染防御的锚点）。
    """
    if not is_injectable(rec):
        return False
    if verification_of(rec).state == "trace_verified":
        return True
    return str((rec.source or {}).get("origin") or "") in _USER_ORIGINS


def broken_link_reason(
    target: MemoryRecord | None, now: float,
) -> str | None:
    """断链判定（U3 插槽 3 与 repair_links 执行共用，防口径漂移）。

    断链 = target 缺失 / status ∉ (active,promoted) / 短期已过期未清理。

    Returns:
        断链原因（"missing" / "status:<s>" / "expired"）；健康链接
        返回 ``None``。
    """
    if target is None:
        return "missing"
    if target.status not in ("active", "promoted"):
        return f"status:{target.status}"
    if (
        target.tier == "short"
        and target.expires_at is not None
        and target.expires_at < now
    ):
        return "expired"
    return None


# ---------------------------------------------------------------------------
# 矛盾检测
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _key_facts(content: str) -> dict[str, str]:
    """抽取「属性 → 值」关键事实（归一化键值）。"""
    facts: dict[str, str] = {}
    for m in _FACT_RE.finditer(content or ""):
        key, value = _norm(m.group(1)), _norm(m.group(2))
        if key and value:
            facts.setdefault(key, value)
    return facts


def _fact_conflict(a: str, b: str) -> str | None:
    """同一属性、不同取值 → 互斥；返回冲突描述，否则 None。"""
    fa, fb = _key_facts(a), _key_facts(b)
    for key in fa.keys() & fb.keys():
        if fa[key] != fb[key]:
            return f"{key}: {fa[key]} ≠ {fb[key]}"
    return None


def _first_text(resp: Any) -> str:
    for block in getattr(resp, "content", []):
        if getattr(block, "type", "") == "text" and block.text:
            return block.text
    return ""


async def _llm_confirms(
    model: ChatModelBase,
    a: MemoryRecord,
    b: MemoryRecord,
    detail: str,
) -> bool:
    """LLM 复核矛盾对；失败（异常/解析不出）按未确认处理。"""
    user = (
        f"【记忆A】标题：{a.title[:80]}\n内容：{a.content[:400]}\n"
        f"【记忆B】标题：{b.title[:80]}\n内容：{b.content[:400]}\n"
        f"【规则检出的冲突】{detail}"
    )
    try:
        resp = await model(messages=[
            Msg(name="system", role="system",
                content=[TextBlock(type="text", text=_CONTRA_SYSTEM)]),
            Msg(name="user", role="user",
                content=[TextBlock(type="text", text=user)]),
        ])
        parsed = json_repair.loads(_first_text(resp))
        return (
            isinstance(parsed, dict)
            and bool(parsed.get("contradiction"))
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("llm contradiction recheck failed: %s", e)
        return False


async def find_contradictions(
    db: RSIMemDatabase,
    memories: list[MemoryRecord],
    *,
    model: ChatModelBase | None = None,
) -> list[tuple[MemoryRecord, MemoryRecord, str]]:
    """找出「标题相似 ≥0.8 且关键事实互斥」的记忆对。

    Args:
        db: 预留（当前矛盾对已在调用方读出，检测本身纯内存）。
        memories: 候选记忆集（调用方圈定注入面，如 L3 档案）。
        model: backend_model 存在时做 LLM 复核；复核不通过/失败不报。

    Returns:
        ``(a, b, 冲突描述)`` 列表。
    """
    out: list[tuple[MemoryRecord, MemoryRecord, str]] = []
    for i, a in enumerate(memories):
        for b in memories[i + 1:]:
            if SequenceMatcher(
                None, (a.title or "").casefold(), (b.title or "").casefold(),
            ).ratio() < _TITLE_SIMILARITY:
                continue
            detail = _fact_conflict(a.content, b.content)
            if detail is None:
                continue
            if model is not None and not await _llm_confirms(
                model, a, b, detail,
            ):
                continue
            out.append((a, b, detail))
    return out


def flag_contradiction_pair(
    a: MemoryRecord, b: MemoryRecord, detail: str = "",
) -> MemoryRecord:
    """给一对矛盾记忆互插旗标（幂等：同一对只插一次）。

    已 trace_verified 的一侧降级为 unverified 待再验证 —— 矛盾证据
    动摇了「已被成功使用」的信心。持久化由调用方负责。
    """
    now = time.time()
    for one, other in ((a, b), (b, a)):
        flags = one.source.get("contradiction_flags")
        if not isinstance(flags, list):
            flags = []
        if not any(
            isinstance(f, dict) and f.get("with") == other.id
            for f in flags
        ):
            flags.append({
                "with": other.id, "detail": detail[:200], "at": now,
            })
        one.source["contradiction_flags"] = flags
        if verification_of(one).state == "trace_verified":
            mark_stale(one, f"contradiction with {other.id[:8]}")
    return a


# ---------------------------------------------------------------------------
# LLM 复核（trace_verified 授予路径 a）
# ---------------------------------------------------------------------------

async def llm_recheck(
    model: ChatModelBase,
    rec: MemoryRecord,
    trace: TraceRecord,
) -> bool:
    """LLM 对照创建轨迹（query/digest）复核记忆内容是否被证实。"""
    user = (
        f"【轨迹】提问：{trace.query[:300]}\n"
        f"最终答复：{trace.digest[:500]}\n"
        f"【记忆】标题：{rec.title[:80]}\n内容：{rec.content[:500]}"
    )
    try:
        resp = await model(messages=[
            Msg(name="system", role="system",
                content=[TextBlock(type="text", text=_RECHECK_SYSTEM)]),
            Msg(name="user", role="user",
                content=[TextBlock(type="text", text=user)]),
        ])
        parsed = json_repair.loads(_first_text(resp))
        return isinstance(parsed, dict) and bool(parsed.get("verified"))
    except Exception as e:  # noqa: BLE001
        logger.warning("llm recheck failed: %s", e)
        return False
