# -*- coding: utf-8 -*-
"""L1-L5 分级自评（U4）：对齐公开 RSI 能级榜单判据的诚实快照。

判据（``assess`` 纯函数，输入 ``collect`` 收集的探针）：

- ``propose_autonomous``：dream 循环自主产候选（CAT_DREAM 事件 ≥1
  且最近报告候选数 >1）；
- ``implement_autonomous``：自主部署存在（CAT_DREAM end deployed
  或基因组版本前移）；
- ``verify_autonomous``：部署前自主验证（报告 replays 非空——回放
  评估真的跑了）；
- ``verify_executed``：部署后执行级验证存在——**双源**：CAT_DREAM
  事件历史（phase=end 且 payload.execution_gate.passed is True）∪
  基因组 ``scores["execution_gate"]``（U2 部署时写入）。旧事件无键
  静默跳过；末轮未部署不影响（历史即证据）；
- ``imp_mutable``：基因组是否覆盖引擎超参（当前 False——
  GenomeSpec 不含治理/引擎参数，如实静态检查）；
- ``domain_generality``：覆盖的问题域数（黄金场景全部属运维单域
  → 1）。

级别映射（诚实输出，不夸大）：三项自主（propose/implement/verify）
全真 → L3；``imp_mutable=False`` 卡在 L4 之外、单域卡在 L5 之外
——故当前满分快照是 ``L3 + failing=[imp_mutable, domain_generality]``。
全部证据为 E0（确定性静态/事件检查），自标 ``evidence="E0"``。
"""
from __future__ import annotations

import json
from typing import Any

from ..storage import RSIMemDatabase
from ..telemetry import CAT_DREAM, LocalEventLog
from .golden import GOLDEN_SCENARIOS


async def collect(
    db: RSIMemDatabase,
    events: LocalEventLog | None,
    config: Any,
) -> dict[str, Any]:
    """收集自评探针（ genomes / 事件历史 / 最近 dream 报告 / meta）。"""
    genomes = await db.list_genomes()
    dream_events = (
        events.read(category=CAT_DREAM, limit=200)
        if events is not None and events.enabled else []
    )
    end_payloads = [
        e["payload"] for e in dream_events
        if e["payload"].get("phase") == "end"
    ]
    raw_report = await db.get_meta("last_dream_report")
    try:
        last_report = (
            json.loads(raw_report) if raw_report
            else None
        ) or {}
    except ValueError:
        last_report = {}
    return {
        "dream_event_count": len(dream_events),
        # 历史峰值候选数（事件历史优先，旧事件无键回退最近报告）：
        # 「是否曾自主提出多候选」看历史而非末轮
        "last_candidates": max(
            [int(p.get("candidates") or 0) for p in end_payloads]
            + [int(last_report.get("candidates") or 0)],
        ),
        "deployed_seen": bool(
            genomes and genomes[0].version >= 1,
        ) or any(p.get("deployed") for p in end_payloads),
        "replays_count": len(last_report.get("replays") or []),
        "exec_gate_event": any(
            isinstance(p.get("execution_gate"), dict)
            and p["execution_gate"].get("passed") is True
            for p in end_payloads
        ),
        "exec_gate_genome": any(
            g.scores.get("execution_gate") is True
            for g in genomes
        ),
        # 静态判据（诚实）：基因组不覆盖引擎超参；黄金场景单域
        "imp_mutable": False,
        "domain_generality": 1,  # GOLDEN_SCENARIOS 全部属运维单域
        "golden_scenarios": len(GOLDEN_SCENARIOS),
    }


def assess(probes: dict[str, Any]) -> dict[str, Any]:
    """纯函数：探针 → {level, criteria, failing, evidence}。"""
    criteria: dict[str, bool] = {
        "propose_autonomous": bool(
            probes.get("dream_event_count", 0) >= 1
            and probes.get("last_candidates", 0) > 1,
        ),
        "implement_autonomous": bool(probes.get("deployed_seen")),
        "verify_autonomous": bool(probes.get("replays_count", 0) >= 1),
        "verify_executed": bool(
            probes.get("exec_gate_event")
            or probes.get("exec_gate_genome"),
        ),
        "imp_mutable": bool(probes.get("imp_mutable", False)),
        "domain_generality": int(
            probes.get("domain_generality", 1),
        ) > 1,
    }
    failing = [k for k, ok in criteria.items() if not ok]
    autonomous = all(
        criteria[k] for k in (
            "propose_autonomous", "implement_autonomous",
            "verify_autonomous",
        )
    )
    if autonomous:
        level = "L3"
        # L4/L5 被 imp_mutable / domain_generality 卡住（在 failing 里
        # 如实列出，不虚标）
    elif criteria["propose_autonomous"] or criteria["implement_autonomous"]:
        level = "L2"
    else:
        level = "L1"
    return {
        "level": level,
        "criteria": criteria,
        "failing": failing,
        "evidence": "E0",
    }
