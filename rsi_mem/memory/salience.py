# -*- coding: utf-8 -*-
"""显著性标注与「记忆即预测」的结算（U1）。

写入路径给每条短期记忆带 :class:`SalienceTag`（存
``MemoryRecord.source["salience"]``，零 schema 变更）；预测的回填只发生在
**真实使用点** —— reasoning 注入 / 系统提示注入（L3 核心档案 + 命中反射）/
agent 主动 ``memory_search`` / 晋升 —— 内部检索（建链 / 回放 / 合成轨迹）
一律不回填。过期未用的结局不写库（行将被删、且写 predictions 表会污染
世界模型 MAE），改走 ``salience_expired`` 遥测事件。

容错契约（settle 错误隔离）：

- ``tag_of`` / ``coerce_tag`` 对脏数据永远返回 ``None`` 而不抛异常；
- :func:`settle_predictions` 内部整体 try/except —— 三个 settle 挂点的
  调用方外层都是「失败即整块置空」（``memory_middleware._retrieve_hint`` /
  ``on_system_prompt`` / ``http_proxy._build_memory_prefix``），settle 抛
  异常会吞掉当轮全部记忆注入，因此异常只 ``logger.warning`` 跳过回填，
  绝不向上抛。
"""
from __future__ import annotations

from typing import Any, Iterable

from pydantic import ValidationError

from agentscope._logging import logger

from ..storage import MemoryRecord, RSIMemDatabase, SalienceTag

__all__ = [
    "SalienceTag",
    "annotate",
    "coerce_tag",
    "dream_priority",
    "is_settled",
    "settle_predictions",
    "tag_of",
]

_OUTCOMES = frozenset({"hit", "promoted", "expired", "ignored"})
"""合法结局字面量（防外部传脏值把标注写成不可读形态）。"""

_KIND_WEIGHT: dict[str, float] = {
    "strategy": 1.0,
    "tested_rule": 0.85,
    "env_fact": 0.7,
    "preference": 0.6,
    "plan": 0.45,
    "state": 0.25,
}
"""种类 → 固化价值权重（dream 排序用；volatile 再乘衰减系数）。"""


def coerce_tag(payload: Any) -> SalienceTag | None:
    """把任意 payload（LLM 输出 / DB 读回）宽容转成 SalienceTag。

    非 dict、字段类型错或字面量越界都返回 ``None``（缺失字段取默认）。
    """
    if not isinstance(payload, dict):
        return None
    try:
        return SalienceTag(**payload)
    except ValidationError:
        return None


def tag_of(record: MemoryRecord) -> SalienceTag | None:
    """读回记录的显著性标注；``source["salience"]`` 非 dict 或字段类型
    错（脏数据 / 旧记录）返回 ``None``，绝不抛异常。"""
    return coerce_tag((record.source or {}).get("salience"))


def annotate(record: MemoryRecord, **fields: Any) -> MemoryRecord:
    """就地写入 / 部分更新记录的显著性标注，返回同一记录。

    已有合法标注作底、``fields`` 覆盖指定字段后整体重新校验；校验失败
    静默放弃（保持原状）不抛异常。
    """
    base = tag_of(record) or SalienceTag()
    try:
        merged = SalienceTag(**{**base.model_dump(), **fields})
    except ValidationError:
        return record
    record.source.setdefault("salience", {})
    record.source["salience"] = merged.model_dump()
    return record


def is_settled(record: MemoryRecord) -> bool:
    """预测是否已结算（无标注视为未结算）。"""
    tag = tag_of(record)
    return tag is not None and tag.settled


def dream_priority(tag: SalienceTag) -> float:
    """固化优先级（0..1）：种类权重 × 预测价值，volatile 显著降权。

    供固化审查排序与过期遥测参考 —— 「记忆即预测」的预测面得分。
    """
    value = min(1.0, max(0.0, tag.predicted_value))
    priority = _KIND_WEIGHT.get(tag.kind, 0.25) * value
    if tag.volatile:
        priority *= 0.3
    return round(priority, 4)


async def settle_predictions(
    db: RSIMemDatabase,
    memory_ids: Iterable[str],
    outcome: str = "hit",
) -> None:
    """真实使用点的预测回填（幂等、错误隔离）。

    - 幂等：``settled=False`` 才写，结算后不再覆盖；
    - 无标注不结算：``tag_of`` 为 ``None``（从未做过预测）的记录跳过
      —— 与 :func:`is_settled` 的「无标注视为未结算」契约一致，不凭空
      创建 settled=True 的默认标注污染预测面数据；
    - 批量：逐条 get + 单条 UPDATE；
    - 错误隔离：整体 try/except，任何异常仅 ``logger.warning`` 跳过回填，
      **绝不向上抛** —— 否则会吞掉调用方当轮的全部记忆注入。
    """
    try:
        if outcome not in _OUTCOMES:
            outcome = "hit"
        for mid in {i for i in memory_ids if i}:
            rec = await db.get_memory(mid)
            if rec is None:
                continue
            tag = tag_of(rec)
            if tag is None or tag.settled:
                continue  # 无标注视为未结算：不结算、不造标注
            annotate(rec, settled=True, settled_outcome=outcome)
            await db.update_memory(rec)
    except Exception as e:  # noqa: BLE001 — settle 失败绝不阻断注入
        logger.warning("salience settle failed: %s", e)
