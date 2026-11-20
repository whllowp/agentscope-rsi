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
import hashlib
import json
import os
from collections import Counter
from typing import Any

from .bench.rounds import format_report_table, run_bench
from .config import GovernanceConfig, RSIMemConfig
from .context import RSIMemContext
from .rsi.evidence import evidence_legend
from .telemetry import CAT_DREAM, CAT_ERROR, CAT_TRACE


def _load_context(
    data_dir: str,
    *,
    governance: GovernanceConfig | None = None,
) -> RSIMemContext:
    kwargs: dict[str, Any] = {
        "data_dir": data_dir or os.environ.get(
            "RSI_MEM_HOME", "./.rsi_mem",
        ),
    }
    if governance is not None:
        kwargs["governance"] = governance
    return RSIMemContext(RSIMemConfig(**kwargs))


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

    # —— 子问题哨兵（U1 证据制修订）：restore 失败 = 机制失效证据 →
    # warn；全同 version 只按 wins（替换胜出累计）分 info 语气，「候选
    # 从未胜过 v0」不再被误报为故障——那可能是支配解合法收敛 ----
    if len(ctx.config.rsi.subproblems) > 1:
        try:
            pool = json.loads(
                await ctx.db.get_meta("rsi_moead_incumbents") or "{}",
            )
            if not isinstance(pool, dict):
                pool = {}
        except ValueError:
            pool = {}
        items: list[dict[str, Any]] = []
        for i in range(len(ctx.config.rsi.subproblems)):
            item = pool.get(str(i))
            if isinstance(item, dict) and item.get("spec_json"):
                items.append(item)
        versions: set[Any] = set()
        for item in items:
            try:
                versions.add(
                    json.loads(item["spec_json"]).get("version"),
                )
            except ValueError:
                continue
        failed_restores = sorted({
            str(item.get("last_restore"))
            for item in items
            if item.get("last_restore") == "corrupt"
            or str(item.get("last_restore")).startswith("dropped:")
        })
        if failed_restores:
            # 唯一 warn 分支：恢复失败（校验拒绝/存储损坏）有真实机制
            # 失效证据，与「是否替换过」无关（E2）
            add(
                "warn",
                "incumbent 恢复失败（校验拒绝/存储损坏）= 机制失效证据"
                f"：{failed_restores}——跨轮 per-λ 记忆未能在本轮重评，"
                "检查 incumbent spec 引用的 SOP 是否被归档（U1 证据制"
                "哨兵，E2）",
                {"failed_restores": failed_restores},
            )
        elif (
            len(items) >= len(ctx.config.rsi.subproblems)
            and len(versions) == 1
        ):
            if all(
                "wins" in item and "last_restore" in item
                for item in items
            ):
                wins = [int(item.get("wins") or 0) for item in items]
                if all(w > 0 for w in wins):
                    add(
                        "info",
                        "全 λ 均有替换胜出记录（wins>0）：支配解合法收敛"
                        f"（全同 v{next(iter(versions))} 非故障）",
                        {"wins": wins},
                    )
                else:
                    add(
                        "info",
                        "λ≥1 子问题休眠：无候选在该子问题胜出（非故障，"
                        "候选从未胜过 v0 不再误报）",
                        {"wins": wins},
                    )
            else:
                # 旧格式池（U1 前写入）：无 wins/last_restore 键，无法
                # 评证 → 降级 info（向后兼容，不再 warn）
                add(
                    "info",
                    "incumbent 池为旧格式（缺 wins/last_restore 键）："
                    "证据制哨兵降级，跑一轮 dream 后即有证据（非故障）",
                    None,
                )
        # 版本有异（per-λ 分化正常）→ 不报

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
    # (auc, traces_evaluated) 对（复核修订）：恒定 AUC 只有在**有新
    # 轨迹流入**（窗口内 traces_evaluated 有变化）时才是「前沿失明」
    # 信号——空闲系统零新轨迹，回放是确定性函数，逐轮完全相同属
    # 健康，不报。旧事件无 traces_evaluated 键 → 静默降级不报。
    auc_pairs = [
        (
            e["payload"].get("pareto_auc_winner"),
            e["payload"].get("traces_evaluated"),
        )
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
        # U3：恒定序列（含恒 0）——此前 n_effective 缺失使全平轨迹把
        # AUC 稀释成恒 0，严格递减哨兵永盲；replay 端已修，此处补哨兵
        pairs_tail = auc_pairs[-5:]
        data_flowing = (
            all(p[1] is not None for p in pairs_tail)
            and len({p[1] for p in pairs_tail}) > 1
        )
        if all(x == tail[0] for x in tail) and data_flowing:
            add(
                "warn",
                f"Pareto 前沿面积持续无变化（末 5 次梦恒 {tail[0]:.4f}，"
                "含恒 0）：前沿失明或分布坍缩（E2）",
                {"series": [round(x, 4) for x in tail]},
            )

    # —— U3 自证回路哨兵 ——
    genomes = await ctx.db.list_genomes()  # 新版在前
    # a) 预测器自参考：近 20 条 realized 无一条 production（全部
    #    replay 回填 = 世界模型只在评价自己）。
    #    冷启动豁免（复核修订）：部署后真实轨迹不足
    #    min_traces_after_deploy，或还没有一次 dream 在这些轨迹之后
    #    运行（E3 回填未到期）时不报——那时没有 production 行是
    #    预期而非故障。
    realized_rows = await ctx.db.load_predictions(realized_only=True)
    recent_sources = [
        (p["realized"] or {}).get("source") or "replay"
        for p in realized_rows[:20]
    ]
    if recent_sources and "production" not in recent_sources:
        anchor = genomes[0] if genomes else None
        post_times = sorted(
            (
                (t.finished_at or t.started_at)
                for t in traces
                if t.outcome is not None and anchor is not None
                and (t.finished_at or t.started_at) >= anchor.created_at
            ),
            reverse=True,
        )
        min_after = ctx.config.governance.min_traces_after_deploy
        last_dream_ts = max(
            (e.get("ts") or 0.0) for e in dream_events
        ) if dream_events else 0.0
        backfill_due = (
            anchor is not None
            and anchor.version >= 1  # 从未部署过的库不可能有 E3 行
            and len(post_times) >= min_after
            and last_dream_ts >= post_times[min_after - 1]
        )
        if backfill_due:
            add(
                "warn",
                "预测器自参考校准：近 "
                f"{len(recent_sources)} 条 realized 全部来自回放（无 "
                "production/E3 行）——世界模型在评价自己，MAE 不可信",
                {"sources": recent_sources[:10]},
            )
    # b) 评审未激活：近 5 次 dream judge_delta_mean 恒 0（旧事件无键
    #    静默降级，不误报）。复核修订：从未配置过 judge 的部署恒 0
    #    属常态 → info；曾经非零（judge 激活过）后沉默 → warn。
    judge_series = [
        e["payload"]["judge_delta_mean"]
        for e in dream_events
        if e["payload"].get("phase") == "end"
        and "judge_delta_mean" in e["payload"]
    ]
    if len(judge_series) >= 5 and all(v == 0 for v in judge_series[-5:]):
        ever_active = any(v != 0 for v in judge_series)
        add(
            "warn" if ever_active else "info",
            "评审未激活，纯内循环：近 5 次 dream judge_delta_mean 恒 0"
            "（"
            + (
                "此前曾有非零评审——judge 沉默是回归"
                if ever_active
                else "该部署从未配置 judge，质量信号只剩自产回放"
            )
            + "）",
            {"series": judge_series[-5:]},
        )

    # —— U3 多样性哨兵：基因组史（genomes 已在自证回路段加载）——
    recent_ops = [g.operator for g in genomes[:10] if g.operator != "seed"]
    if len(recent_ops) >= 5:
        top_op, top_n = Counter(recent_ops).most_common(1)[0]
        if top_n / len(recent_ops) > 0.8:
            add(
                "warn",
                f"算子集中度坍缩：近 {len(recent_ops)} 版 {top_n} 版为 "
                f"{top_op}（{top_n / len(recent_ops):.0%}）——探索面 "
                "收窄（engine 将自动注入 explore 扰动）",
                {"operator": top_op, "ratio": round(
                    top_n / len(recent_ops), 4,
                )},
            )

    def _content_hash(g) -> str | None:
        return hashlib.sha256(
            g.model_dump_json(exclude={
                "version", "created_at", "parent_version",
                "operator", "notes", "scores",
            }).encode("utf-8"),
        ).hexdigest()[:16]

    dup_pairs = [
        (genomes[i + 1].version, genomes[i].version)
        for i in range(len(genomes) - 1)
        if _content_hash(genomes[i]) == _content_hash(genomes[i + 1])
    ]
    if dup_pairs:
        add(
            "warn",
            f"相邻版本内容重复（去元数据哈希相同，空演化漂移）："
            f"{dup_pairs[:5]}",
            {"pairs": dup_pairs[:10]},
        )

    # —— U3 帧锁定人工锚：外部审计包（供人定期复核）——
    audit_rows = [
        {
            "version": g.version,
            "operator": g.operator,
            "render_hash": g.scores.get("render_hash"),
        }
        for g in reversed(genomes[:6])  # 旧 → 新
    ]
    hash_changed = [
        audit_rows[i]["version"]
        for i in range(1, len(audit_rows))
        if audit_rows[i]["render_hash"]
        and audit_rows[i - 1]["render_hash"]
        and audit_rows[i]["render_hash"]
        != audit_rows[i - 1]["render_hash"]
    ]
    slope_vals = [
        g.scores.get("balanced_min")
        for g in reversed(genomes[:6])
        if isinstance(g.scores.get("balanced_min"), (int, float))
    ]
    if len(audit_rows) >= 2:  # 有部署史才有 diff 可审计（种子库跳过）
        add(
            "info",
            "外部审计包：近 "
            f"{len(audit_rows)} 版 render_hash/operator/部署斜率快照"
            "（帧锁定人工锚，定期复核渲染面演化是否符合预期）",
            {
                "rows": audit_rows,
                "render_hash_changed_versions": hash_changed,
                "deploy_slope": (
                    round(slope_vals[-1] - slope_vals[0], 4)
                    if len(slope_vals) >= 2 else None
                ),
            },
        )

    # —— U3 分布监测：外部信号消失（全自产）——
    window = [t for t in traces if t.outcome is not None][:40]
    if window:
        prod_ratio = sum(
            1 for t in window
            if ((t.outcome.signals or {}).get("source") or "production")
            == "production"
        ) / len(window)
        if prod_ratio <= ctx.config.rsi.external_data_min_ratio:
            add(
                "info",
                "外部信号消失风险：近期轨迹窗口 production 占比 "
                f"{prod_ratio:.0%}（全自产——回放/评审都在评价自己产"
                "生的数据）",
                {"production_ratio": round(prod_ratio, 4),
                 "window": len(window)},
            )

    # —— U3 遥测断链 ——
    trace_events = ctx.events.read(category=CAT_TRACE, limit=500)
    if dream_events and not trace_events:
        add(
            "info",
            "结算遥测断链：有 dream 事件但零 trace 事件（轨迹结算未落"
            "遥测，评分漂移/错误聚集哨兵失明）",
            None,
        )
    recent_all = ctx.events.read(limit=20)
    tail_errors = 0
    for e in reversed(recent_all):
        if e["category"] == "error":
            tail_errors += 1
        else:
            break
    if tail_errors >= 3:
        add(
            "warn",
            f"事件流尾部连续 {tail_errors} 条 CAT_ERROR（dream 失败/"
            "校验拒绝/保守丢弃连发——检查最近 dream 报告）",
            None,
        )

    # —— U4 治理器状态（fail-closed 可观测；从未演化的空库不报）——
    governor = ctx.engine._governor
    governor.refresh()
    gov_snap = governor.snapshot()
    if (
        gov_snap["degraded"]
        or gov_snap["deployment_pause"]
        or gov_snap["pinned_version"] is not None
        or os.path.isfile(governor.state_path)
    ):
        if gov_snap["degraded"]:
            add(
                "warn",
                "governor 状态不可读或被删除（fail-closed）：部署一律"
                f"拒绝，修复 {governor.state_path} 后自动恢复",
                gov_snap,
            )
        else:
            add(
                "info",
                "治理器：pin="
                f"{gov_snap['pinned_version']}，pause="
                f"{'是' if gov_snap['deployment_pause'] else '否'}"
                f"（{gov_snap['paused_reason'] or '-'}），近 1 小时部署 "
                f"{gov_snap['deploys_last_hour']} 次",
                gov_snap,
            )

    # —— U4 L1-L5 分级自评（有 dream 活动才评）——
    if dream_events:
        from .rsi.level import assess, collect as collect_probes

        assessment = assess(
            await collect_probes(ctx.db, ctx.events, ctx.config),
        )
        add(
            "info",
            f"自评等级 {assessment['level']}（未达判据: "
            f"{', '.join(assessment['failing']) or '无'}；"
            f"{assessment['evidence']} 静态检查）",
            assessment["criteria"],
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
        "trace_events": len(trace_events),
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
    # —— U4 变更审计段：逐版本指纹 + 近期拦截（事件按 trigger 聚合）——
    end_events = [
        e["payload"]
        for e in ctx.events.read(category=CAT_DREAM, limit=100)
        if e["payload"].get("phase") == "end"
    ]
    denies = [
        {
            "trigger": p.get("trigger"),
            "reason": p.get("reason"),
            "governor_decision": p.get("governor_decision"),
        }
        for p in end_events
        if p.get("deployed") is False and p.get("reason")
    ][:10]
    audit_errors = [
        {
            "scope": e["payload"].get("scope"),
            "operator": e["payload"].get("operator"),
            "stage": e["payload"].get("stage"),
            "reason": str(e["payload"].get("reason"))[:120],
        }
        for e in ctx.events.read(category=CAT_ERROR, limit=20)
        if e["payload"].get("kind") == "audit"
    ][:10]
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
        "change_audit": {
            "rows": [
                {
                    "version": g.version,
                    "operator": g.operator,
                    "render_hash": g.scores.get("render_hash"),
                    "deployed": bool(g.scores.get("execution_gate")),
                }
                for g in genomes[:10]
            ],
            "denied_rounds": denies,
            "audit_errors": audit_errors,
        },
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
        if args.command == "bench":
            # U4 治理契约：bench 是人运行的运维面，构造期传入宽松治理
            # （默认严格断路器会把 10 轮连发部署限流成静默 deploys=0）
            ctx = _load_context(
                args.data_dir,
                governance=GovernanceConfig(
                    max_deploys_per_hour=10 ** 6,
                    min_deploy_interval_s=0.0,
                    consecutive_deploy_limit=10 ** 6,
                ),
            )
        else:
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
