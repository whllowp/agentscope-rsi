# -*- coding: utf-8 -*-
"""10 轮离线迭代基准：合成运维环境 × RSI 引擎。

模拟"论文式"的 update-and-acceptance 循环（RSI-Anything 的 cycle k）：

每轮（共 ``rounds``=10）：

1. **在线**：合成环境按当前基因组生成 ``traces_per_round`` 条轨迹
   （outcome 由环境模型对基因组的确定性评分 + 可复现噪声给出）；
2. **离线 dream**：引擎跑完整 诊断→插件计划→校验→预测→回放→
   非退化选择 流程；
3. **度量**：记录环境质量分（E2）、估计成本、预测器校准（MAE/
   样本数）、保守模式状态、部署版本。

产物：逐轮 rows + summary，写 ``<data_dir>/logs/bench_report.json``。
断言式结论（见 ``rsi_mem/tests/rsi_anything_test.py`` 的
``test_bench_ten_rounds_non_degrading``）：10 轮全部完成、末轮环境质量
不低于首轮（容差 0.02）、预测样本随轮次积累。
"""
from __future__ import annotations

import json
import random
import time
from typing import Any

from ..context import RSIMemContext
from ..rsi.golden import GOLDEN_KNOWLEDGE as _KNOWLEDGE
from ..rsi.golden import GOLDEN_SCENARIOS as SCENARIOS
from ..storage import (
    ModelCallStat,
    OutcomeRecord,
    ToolCallNode,
    TraceRecord,
)
# U2：场景与黄金知识原值冻结到 rsi/golden.py（leaf 模块，执行门共用，
# 单一事实源）；此处别名保持 bench 内符号兼容。


class OpsEnv:
    """合成运维环境（E2 证据源）：基因组 → 确定性质量分 + 可复现噪声。"""

    def __init__(self, ctx: RSIMemContext, seed: int = 42) -> None:
        self._ctx = ctx
        self._rng = random.Random(seed)

    async def seed_knowledge(self) -> None:
        """预置各场景的长期知识（让检索/覆盖在环境中起作用）。"""
        for kw, text in _KNOWLEDGE.items():
            await self._ctx.tiers.longterm.add(
                title=f"{kw}处置要点", content=text,
            )

    async def genome_quality(self, genome) -> float:
        """环境对基因组的确定性评分（0..1 区间内的相对指标）。"""
        score = 0.35
        # —— SOP 覆盖：激活肌肉记忆命中场景关键词 ——
        sops = [
            await self._ctx.db.get_memory(sid)
            for sid in genome.active_sops
        ]
        sops = [s for s in sops if s is not None]
        if sops:
            matched = 0
            for sc in SCENARIOS:
                query_l = sc["query"].casefold()
                hit = any(
                    t.casefold() in query_l
                    for s in sops for t in s.triggers
                )
                if hit:
                    matched += 1
            score += 0.12 * matched / len(SCENARIOS)
        # —— 检索参数合理性 ——
        if 3 <= genome.retrieval.top_k <= 6:
            score += 0.05
        if genome.retrieval.char_budget >= 1200:
            score += 0.03
        if genome.retrieval.score_threshold <= 0.2:
            score += 0.02
        return min(1.0, score)

    @staticmethod
    def est_cost(genome) -> float:
        frag = sum(len(f.text) for f in genome.prompt_fragments if f.enabled)
        return 2500.0 + genome.retrieval.char_budget * 1.5 + frag * 0.5

    async def emit_traces(
        self, n: int, round_idx: int,
    ) -> list[TraceRecord]:
        """按当前基因组生成 n 条合成轨迹。

        早期轮（1-3）错误密集（驱动"高错误率→蒸馏 SOP"）；token 成本
        与基因组注入预算联动（驱动"成本高→收紧检索"）。
        """
        genome = await self._ctx.db.active_genome()
        base = await self.genome_quality(genome)
        est_input = 1800 + int(genome.retrieval.char_budget)
        out: list[TraceRecord] = []
        for i in range(n):
            sc = SCENARIOS[(round_idx + i) % len(SCENARIOS)]
            score = max(0.0, min(1.0, base + self._rng.uniform(-0.03, 0.03)))
            t = TraceRecord(
                session_id=f"bench-{round_idx}",
                query=sc["query"],
                digest=f"已处置：{sc['kw']}",
            )
            t.model_calls = [
                ModelCallStat(
                    model="bench-model",
                    input_tokens=est_input,
                    output_tokens=400,
                    latency_s=1.2,
                ),
            ]
            t.tool_calls = [
                ToolCallNode(name="bash", input_digest="df -h/check",
                             state="ok"),
                ToolCallNode(name="read_file", input_digest="logs",
                             state="ok"),
            ]
            if round_idx <= 2 and i % 2 == 0:
                # 早期轮一半轨迹带 2 个错误调用 → 聚合错误率 > 0.25
                t.tool_calls.extend([
                    ToolCallNode(name="restart_service",
                                 input_digest="盲目重启",
                                 state="error"),
                    ToolCallNode(name="restart_service",
                                 input_digest="再次盲目重启",
                                 state="error"),
                ])
            t.outcome = OutcomeRecord(
                score=round(score, 3),
                signals={"source": "bench", "round": round_idx},
            )
            # 记录该查询真实可召回的记忆（回放回归检查的锚点）
            hits = await self._ctx.tiers.retriever.search(
                sc["query"], tiers=("long", "short"), top_k=3,
            )
            t.used_memory_ids = [h.record.id for h in hits]
            await self._ctx.db.upsert_trace(t)
            out.append(t)
        return out


async def run_bench(
    ctx: RSIMemContext,
    *,
    rounds: int = 10,
    traces_per_round: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """跑 ``rounds`` 轮 benchmark，返回报告并落盘。

    **治理契约（U4）**：``ctx`` 传入时治理断路器参数已在**构造期**
    定型——bench 需要宽松治理（构造期传入 ``governance=
    GovernanceConfig(max_deploys_per_hour=10**6, min_deploy_interval_s
    =0.0, consecutive_deploy_limit=10**6)``，见 cli bench 分支与
    bench 测试）。默认严格治理（6 次/小时 + 300s 冷却）会使 deploys
    被限流，此时 ``rows[0]["governor_state"]`` 可见原因而非静默
    ``deploys=0``。签名不变；构造后覆盖只能走整字段重赋值旁路，
    不支持也不承诺。
    """
    env = OpsEnv(ctx, seed=42)
    await env.seed_knowledge()

    rows: list[dict[str, Any]] = []
    for r in range(1, rounds + 1):
        genome_before = await ctx.db.active_genome()
        q_before = await env.genome_quality(genome_before)
        await env.emit_traces(traces_per_round, r)
        dream = await ctx.engine.dream(trigger=f"bench:{r}")
        genome_after = await ctx.db.active_genome()
        q_after = await env.genome_quality(genome_after)
        row = {
            "round": r,
            "version_before": genome_before.version,
            "version_after": genome_after.version,
            "deployed": bool(dream.get("deployed")),
            "operator": dream.get("operator"),
            "quality_before": round(q_before, 4),
            "quality_after": round(q_after, 4),
            "est_cost_after": round(env.est_cost(genome_after), 1),
            "predictor_mae": dream["predictor"]["mae"],
            "predictor_samples": dream["predictor"]["samples"],
            "conservative_mode": dream["conservative_mode"],
            "dropped_by_gate": dream["dropped_by_gate"],
            "candidates": dream["candidates"],
            # U4：ρ 扫描前沿面积与选择器（取顶层键，不靠 replays 猜行）
            "pareto_auc": (
                dream.get("pareto_auc_winner")
                if dream.get("deployed")
                else dream.get("pareto_auc_v0")
            ),
            "selector": dream.get("selector"),
            # U4 治理/执行门观测
            "execution_gate": (
                (dream.get("execution_gate") or {}).get("passed")
            ),
            "governor": dream.get("governor_decision"),
        }
        if r == 1:
            # 首轮快照治理器状态（调用方忘传宽松治理时原因可见）
            row["governor_state"] = ctx.engine._governor.snapshot()
        rows.append(row)

    first, last = rows[0], rows[-1]
    predictions = await ctx.db.load_predictions()
    report = {
        "rounds": rounds,
        "traces_per_round": traces_per_round,
        "seed": seed,
        "generated_at": time.time(),
        "rows": rows,
        "summary": {
            "quality_first": first["quality_after"],
            "quality_last": last["quality_after"],
            "quality_delta": round(
                last["quality_after"] - first["quality_after"], 4,
            ),
            "deploys": sum(1 for x in rows if x["deployed"]),
            "final_version": last["version_after"],
            "prediction_samples": len(predictions),
            "non_degraded": (
                last["quality_after"] >= first["quality_after"] - 0.02
            ),
            # U4 增量字段（观测先行：不做 AUC 阈值断言——环境奖励
            # char_budget≥1200，质量寻求部署会合理压低 AUC，阈值待
            # bench_report 积累后下轮再定）
            "pareto_auc_first": first.get("pareto_auc"),
            "pareto_auc_last": last.get("pareto_auc"),
            "pareto_auc_delta": round(
                (last.get("pareto_auc") or 0.0)
                - (first.get("pareto_auc") or 0.0), 4,
            ),
            "balanced_deploys": sum(
                1 for x in rows
                if x["deployed"] and x.get("selector") == "balanced"
            ),
            # U4 治理/执行门统计（只增不改：旧键名与语义原样）
            "execution_gate_passes": sum(
                1 for x in rows if x["execution_gate"] is True
            ),
            "governor_denies": sum(
                1 for x in rows
                if (x.get("governor") or {}).get("allowed") is False
            ),
        },
    }
    out_path = f"{ctx.config.data_dir}/logs/bench_report.json"
    ctx.events.emit_sync(
        "bench",
        {
            "rounds": rounds,
            "summary": report["summary"],
            "path": out_path,
        },
    )
    try:
        import os

        os.makedirs(ctx.config.data_dir + "/logs", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["path"] = out_path
    except OSError as e:  # noqa: BLE001
        report["path_error"] = str(e)
    return report


def format_report_table(report: dict[str, Any]) -> str:
    """把基准报告渲染成可读表格（CLI 打印用）。"""
    header = (
        f"{'轮':>3} {'v前':>4} {'v后':>4} {'部署':>4} {'算子':<18} "
        f"{'质量':>6} {'成本':>7} {'AUC':>6} {'MAE':>6} {'样本':>4} {'保守':>4}"
    )
    lines = [header, "-" * len(header)]
    for r in report["rows"]:
        mae = r["predictor_mae"]
        mae_str = f"{mae:.3f}" if mae is not None else "-"
        auc = r.get("pareto_auc")
        auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else "-"
        lines.append(
            f"{r['round']:>3} {r['version_before']:>4} "
            f"{r['version_after']:>4} "
            f"{'是' if r['deployed'] else '否':>4} "
            f"{(r['operator'] or '-'):<18} "
            f"{r['quality_after']:>6} {r['est_cost_after']:>7} "
            f"{auc_str:>6} "
            f"{mae_str:>6} "
            f"{r['predictor_samples']:>4} "
            f"{'是' if r['conservative_mode'] else '否':>4}",
        )
    s = report["summary"]
    auc_d = s.get("pareto_auc_delta")
    auc_d_str = f"{auc_d:+.4f}" if isinstance(auc_d, (int, float)) else "-"
    lines.append("-" * len(header))
    lines.append(
        f"总结: 首轮质量 {s['quality_first']} → 末轮质量 {s['quality_last']} "
        f"(Δ={s['quality_delta']:+.4f}), 部署 {s['deploys']}/{report['rounds']} 轮 "
        f"(balanced {s.get('balanced_deploys', '-')}), "
        f"AUC {s.get('pareto_auc_first', '-')} → {s.get('pareto_auc_last', '-')} "
        f"(Δ={auc_d_str}), "
        f"预测样本 {s['prediction_samples']}, "
        f"非退化 {'通过' if s['non_degraded'] else '未通过'}",
    )
    return "\n".join(lines)
