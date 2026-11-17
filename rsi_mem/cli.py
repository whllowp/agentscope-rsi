# -*- coding: utf-8 -*-
"""rsi_mem CLI：基于本地日志 + SQLite 的快速发现与运维。

::

    python -m rsi_mem.cli doctor [--data-dir DIR]   # 体检 + 异常发现
    python -m rsi_mem.cli report [--data-dir DIR]   # dream 报告/基因组史
    python -m rsi_mem.cli bench [--rounds 10] [--data-dir DIR]  # 10轮基准

langfuse（若启用）的事件与本地 JSONL 内容一致；doctor 只读本地，
秒级出结果。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from .bench.rounds import format_report_table, run_bench
from .config import RSIMemConfig
from .context import RSIMemContext
from .rsi.evidence import evidence_legend
from .telemetry import CAT_DREAM, CAT_TRACE


def _load_context(data_dir: str) -> RSIMemContext:
    return RSIMemContext(
        RSIMemConfig(data_dir=data_dir or os.environ.get("RSI_MEM_HOME",
                                                         "./.rsi_mem")),
    )


# ---------------------------------------------------------------------------
# doctor：快速发现
# ---------------------------------------------------------------------------

async def doctor(ctx: RSIMemContext) -> dict[str, Any]:
    """扫描本地日志 + 数据库，产出发现（findings）与摘要。"""
    findings: list[dict[str, Any]] = []

    def add(sev: str, msg: str, detail: Any = None) -> None:
        findings.append({"severity": sev, "message": msg, "detail": detail})

    stats = await ctx.db.stats()
    traces = await ctx.db.list_traces(limit=40)

    # —— 轨迹质量漂移（E3）：近 20 vs 更早 20 ——
    recent = [t for t in traces if t.outcome][:20]
    older = [t for t in traces if t.outcome][20:40]
    if recent:
        mean_recent = sum(t.outcome.score for t in recent) / len(recent)
        if older:
            mean_older = sum(t.outcome.score for t in older) / len(older)
            if mean_recent < mean_older - 0.1:
                add(
                    "warn",
                    f"轨迹质量下滑：近20条均值 {mean_recent:.3f} < "
                    f"更早20条 {mean_older:.3f}（E3）",
                    {"recent": round(mean_recent, 3),
                     "older": round(mean_older, 3)},
                )
        if mean_recent < 0.4:
            add("warn", f"近期轨迹质量偏低（{mean_recent:.3f}）",
                None)
        neg = sum(
            1 for t in recent
            if t.outcome.feedback is not None and t.outcome.feedback < 0
        )
        if neg >= max(2, len(recent) // 5):
            add("warn", f"近 20 条轨迹中 {neg} 条负反馈（E3）", None)

    # —— 工具错误聚集（E0）——
    if traces:
        err: dict[str, int] = {}
        total: dict[str, int] = {}
        for t in traces:
            for c in t.tool_calls:
                total[c.name] = total.get(c.name, 0) + 1
                if c.state == "error":
                    err[c.name] = err.get(c.name, 0) + 1
        for name, e in sorted(err.items(), key=lambda x: -x[1])[:3]:
            ratio = e / total.get(name, 1)
            if ratio > 0.3:
                add(
                    "warn",
                    f"工具 {name} 错误率 {ratio:.0%}（{e}/{total[name]}，E0）",
                    None,
                )

    # —— 预测器健康（E1 世界模型校准）——
    preds = await ctx.db.load_predictions(realized_only=True)
    if len(preds) >= ctx.config.rsi.predictor_min_samples:
        mae = sum(
            abs(p["predicted"].get("quality_delta", 0.0)
                - p["realized"].get("quality_delta", 0.0))
            for p in preds
        ) / len(preds)
        if mae > ctx.config.rsi.predictor_mae_threshold:
            add(
                "warn",
                f"世界模型预测误差偏高：MAE={mae:.3f}"
                f"（阈值 {ctx.config.rsi.predictor_mae_threshold}，E1）"
                f"—— 引擎将进入保守模式直至重新校准",
                {"samples": len(preds), "mae": round(mae, 3)},
            )

    # —— 记忆库健康 ——
    if stats.get("pending_reviews", 0) >= 10:
        add(
            "info",
            f"待确认提议积压 {stats['pending_reviews']} 条"
            f"（memory_confirm 批量处理）",
            None,
        )
    if stats.get("muscle", 0) >= 20:
        add("info", f"肌肉记忆 {stats['muscle']} 条，检查长期未命中的"
                    f"降级提议", None)

    # —— 事件日志侧 ——
    dream_events = ctx.events.read(category=CAT_DREAM, limit=100)
    deploy_fails = [
        e for e in dream_events
        if e["payload"].get("phase") == "end"
        and e["payload"].get("deployed") is False
    ]
    if len(deploy_fails) >= 5 and stats.get("traces", 0) > 30:
        add(
            "info",
            f"最近 {len(deploy_fails)} 次 dream 均未部署"
            f"（可能已到局部最优，或候选被校验/保留门拦截）",
            None,
        )

    # —— 子问题覆盖（U4：U2 MOEA/D 回归哨兵，E2）——
    # meta 'rsi_moead_incumbents' 全同 version 且多子问题 → λ≥1 的
    # 邻域替换从未产生独立 incumbent（per-λ 机制退化为单 λ）
    if len(ctx.config.rsi.subproblems) > 1:
        try:
            pool = json.loads(
                await ctx.db.get_meta("rsi_moead_incumbents") or "{}",
            )
            if not isinstance(pool, dict):
                pool = {}
        except ValueError:
            pool = {}
        versions: set[Any] = set()
        for item in pool.values():
            if not (isinstance(item, dict) and item.get("spec_json")):
                continue
            try:
                versions.add(
                    json.loads(item["spec_json"]).get("version"),
                )
            except ValueError:
                continue
        if (
            len(pool) >= len(ctx.config.rsi.subproblems)
            and len(versions) == 1
        ):
            add(
                "warn",
                "第二子问题从未激活：全部 per-λ incumbent 同版本"
                f"（v{next(iter(versions))}），λ≥1 邻域替换未生效"
                "（U2 回归哨兵，E2）",
                {"versions": sorted(x for x in versions if x is not None)},
            )

    # —— Pareto 退化（U4，P7：读事件日志，E2）——
    # 末 5 次 dream end 的 pareto_auc_winner 持续下降且总降幅 >0.1
    # 才告警；事件日志缺失 / 无该字段（旧版本事件）时静默降级不误报
    auc_series = [
        e["payload"]["pareto_auc_winner"]
        for e in dream_events
        if e["payload"].get("phase") == "end"
        and isinstance(
            e["payload"].get("pareto_auc_winner"), (int, float),
        )
    ]
    if len(auc_series) >= 5:
        tail = auc_series[-5:]
        if (
            all(tail[i] > tail[i + 1] for i in range(len(tail) - 1))
            and tail[0] - tail[-1] > 0.1
        ):
            add(
                "warn",
                f"Pareto 前沿面积连续下滑：末 5 次梦 {tail[0]:.3f} → "
                f"{tail[-1]:.3f}（E2，观测先行——阈值待积累后校准）",
                {"series": [round(x, 4) for x in tail]},
            )

    if not findings:
        add("ok", "未发现异常", None)

    return {
        "data_dir": ctx.config.data_dir,
        "stats": stats,
        "langfuse_active": ctx.langfuse.active,
        "events_log": (
            ctx.events.path if ctx.events.enabled else "(disabled)"
        ),
        "trace_events": len(ctx.events.read(category=CAT_TRACE, limit=500)),
        "findings": findings,
        "evidence_legend": evidence_legend(),
    }


# ---------------------------------------------------------------------------
# report：dream 报告 + 基因组史
# ---------------------------------------------------------------------------

async def report(ctx: RSIMemContext) -> dict[str, Any]:
    last = await ctx.db.get_meta("last_dream_report")
    genomes = await ctx.db.list_genomes()
    predictions = await ctx.db.load_predictions()
    return {
        "genome_history": [
            {
                "version": g.version,
                "operator": g.operator,
                "scores": g.scores,
                "notes": g.notes[:80],
            }
            for g in genomes[:10]
        ],
        "predictions": {
            "total": len(predictions),
            "realized": sum(1 for p in predictions if p["realized"]),
        },
        "last_dream_report": (
            json.loads(last) if last else None
        ),
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rsi_mem", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="体检 + 异常快速发现")
    p_doctor.add_argument("--data-dir", default="")

    p_report = sub.add_parser("report", help="dream 报告与基因组历史")
    p_report.add_argument("--data-dir", default="")

    p_bench = sub.add_parser("bench", help="离线迭代基准（默认 10 轮）")
    p_bench.add_argument("--rounds", type=int, default=10)
    p_bench.add_argument("--traces", type=int, default=4)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--data-dir", default="")

    args = parser.parse_args(argv)

    async def _run() -> dict[str, Any]:
        ctx = _load_context(args.data_dir)
        try:
            if args.command == "doctor":
                return await doctor(ctx)
            if args.command == "report":
                return await report(ctx)
            result = await run_bench(
                ctx,
                rounds=args.rounds,
                traces_per_round=args.traces,
                seed=args.seed,
            )
            print(format_report_table(result))
            return {"bench_path": result.get("path"),
                    "summary": result["summary"]}
        finally:
            await ctx.close()

    print(json.dumps(asyncio.run(_run()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
