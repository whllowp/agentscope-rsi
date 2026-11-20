# -*- coding: utf-8 -*-
"""执行级验证门（U2）：部署前的最后一道硬门。

validator 保证结构合法、pair gate 复验组装态，但都**不执行**——
参数收紧到杀死召回、锁定片段被禁用退出渲染面这类「校验过、执行
退化」的候选仍可能部署。本门在**隔离栈**（临时目录全新
DB/Mirror/TierManager，不建 RSIMemContext，无环无污染）上跑一次
真实注入与检索，对胜者基因组做三类硬断言：

- **必注入**：黄金场景 query 渲染 system prompt block，父代每个
  locked 片段 text 逐字在场（防 enabled/slot 旁路——配合 validator
  全字段锁双保险）；
- **必召回（kw 探针）**：探针词用场景关键词而非整 query。实证
  （隔离栈 + 默认 top_k=5/threshold=0）：磁盘/证书两场景整 query
  0 命中——FTS trigram 对『线上磁盘使用率』vs『磁盘处置要点』零
  3-gram 交集，LIKE 整词兜底同 miss；kw 探针 5/5 命中。**探针
  语义 = 「检索参数不杀关键词召回」，非「真实 query 召回质量」**
  （后者是质量评估的事，不归部署门管）。缺一（任一场景 miss）
  整轮否决；
- **必守预算（绝对臂）**：渲染总字符 ≤ ``governance.max_inject_chars``
  （默认 12000）。不设相对臂：隔离栈上父代渲染仅 ~160 chars，
  ×1.1 的相对阈值低于典型 logic_evolve 片段长度，等价于禁止一切
  内容增量候选。

O(1) 只验胜者；v0 获胜（保持现状）分支天然豁免——门只拦「要上线
的新东西」，不拦「什么都不改」。
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Any

from ..config import RSIMemConfig
from ..memory.tiers import TierManager
from ..storage import (
    GenomeSpec,
    MirrorStore,
    RSIMemDatabase,
)
from .golden import GOLDEN_KNOWLEDGE, GOLDEN_SCENARIOS


@dataclass
class ExecutionGateResult:
    """执行门结果（进 dream 报告 ``execution_gate`` 键）。"""

    passed: bool
    failed_checks: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


async def run_execution_gate(
    winner: GenomeSpec,
    parent: GenomeSpec,
    config: RSIMemConfig,
    *,
    max_inject_chars: int | None = None,
) -> ExecutionGateResult:
    """在临时目录的隔离栈上对 ``winner`` 跑三类硬断言。

    Args:
        winner: 本轮选择胜出、待部署的候选基因组（已过 pair gate）。
        parent: 当前基因组（locked 片段以父代为准）。
        config: 调用方配置（隔离栈自身用独立的默认配置，黄金知识
            固定，不受调用方记忆库影响）。
        max_inject_chars: 注入预算——**引擎路径恒传构造期快照**
            （``RSIEngine._max_inject_chars``）：插件可在 ``start(ctx)``
            里整字段重赋值 ``config.governance`` 放宽预算（pydantic
            只拦属性突变），运行期读活配置即被旁路。``None`` 时回退
            读 ``config.governance``，仅供无对抗场景的测试直调。
    """
    failed: list[str] = []
    details: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="rsi-exec-gate-") as tmp:
        gate_cfg = RSIMemConfig(data_dir=tmp)
        db = RSIMemDatabase(gate_cfg.db_path)
        mirror = MirrorStore(gate_cfg.data_dir)
        tiers = TierManager(db, mirror, gate_cfg)
        try:
            # 黄金知识逐条入隔离栈长期层（title 与 bench 环境同构）
            kw_ids: dict[str, str] = {}
            for kw, text in GOLDEN_KNOWLEDGE.items():
                rec = await tiers.longterm.add(
                    title=f"{kw}处置要点", content=text,
                )
                kw_ids[kw] = rec.id

            locked_texts = [
                f.text for f in parent.prompt_fragments if f.locked
            ]
            details["locked_fragments"] = len(locked_texts)

            # ---- (a) 必注入 + (c) 绝对预算臂（同一批渲染）----
            missing_locked: list[str] = []
            max_block_chars = 0
            for sc in GOLDEN_SCENARIOS:
                block, _ids = await tiers.system_prompt_block_with_ids(
                    query=sc["query"], genome=winner,
                )
                max_block_chars = max(max_block_chars, len(block))
                for frag_i, text in enumerate(locked_texts):
                    if text not in block:
                        missing_locked.append(f"{sc['kw']}:frag{frag_i}")
            details["max_block_chars"] = max_block_chars
            if missing_locked:
                failed.append("locked_injection")
                details["missing_locked"] = missing_locked
            budget = (
                max_inject_chars
                if max_inject_chars is not None
                else config.governance.max_inject_chars
            )
            if max_block_chars > budget:
                failed.append("inject_budget")
                details["inject_budget"] = budget

            # ---- (b) 必召回（kw 探针，胜者检索参数）----
            missed_kws: list[str] = []
            for sc in GOLDEN_SCENARIOS:
                results = await tiers.retriever.search(
                    sc["kw"],
                    tiers=("long", "short"),
                    top_k=winner.retrieval.top_k,
                    score_threshold=winner.retrieval.score_threshold,
                    char_budget=winner.retrieval.char_budget,
                )
                hit_ids = {r.record.id for r in results}
                if kw_ids[sc["kw"]] not in hit_ids:
                    missed_kws.append(sc["kw"])
            if missed_kws:
                failed.extend(f"kw_recall:{kw}" for kw in missed_kws)
                details["missed_kws"] = missed_kws
        finally:
            await db.close()
    return ExecutionGateResult(
        passed=not failed,
        failed_checks=failed,
        details=details,
    )
