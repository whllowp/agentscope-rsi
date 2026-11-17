# -*- coding: utf-8 -*-
"""RSIEngine：online → dream → select → deploy 循环。

三篇论文的融合：

- **Dream-RSI §3**（轨迹录制 → 回放模拟器 → 候选 argmax 选择）：
  候选恒含当前基因组 v0，同一评分函数下 Tchebycheff 距离最小者获胜
  ⇒ 数学上保证非退化；
- **Harmony Harness**（多目标 MOEA/D、diagnose→hypothesize→operator、
  三段校验 + 一次修复 + 回滚）；
- **RSI-Anything**（策略-世界模型对）：新增
  * :class:`OutcomePredictor` —— 候选执行前预测三目标增量，
    预测-观测对在线校准（predictions 表）；
  * **可靠性保留门** —— 预测样本足够且 MAE 超阈时进入保守模式
    （仅放行 retrieval_evolve 小步候选）；
  * **E0-E3 分级证据** —— 诊断与报告信号全部标注证据层级；
  * **插件化** —— 计划生成收敛到 :class:`PluginRegistry`，内置
    改进器也只是插件，第三方经配置或 entry points 注入且无法绕过
    校验/非退化门。

对外接口：``maybe_dream``（计数触发）、``dream``（手动，供 rsi_evolve）、
``iterate``（连跑 N 轮，基准用）、``rollback``。
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any

from agentscope._logging import logger
from agentscope.model import ChatModelBase

from ..config import RSIMemConfig
from ..memory.tiers import TierManager
from ..storage import GenomeSpec, RSIMemDatabase
from ..telemetry import (
    CAT_DREAM,
    CAT_PREDICTION,
    LangfuseBridge,
    LocalEventLog,
    NOOP_BRIDGE,
)
from .evidence import annotate_diagnosis, evidence_legend, tag_evidence
from .improver import Improver
from .objectives import IdealPoint, ObjectiveVector, tprime
from .operators import MutationOutcome, apply_plan
from .plugins import PluginRegistry, build_default_registry
from .predictor import (
    CandidateFeatures,
    OutcomePredictor,
    backfill_realized,
    is_conservative_operator,
    realized_dict,
    record_prediction,
)
from .replay import ReplayEvaluator, ReplayReport
from .validator import GenomeValidator

_REPORT_EVIDENCE = {
    "coverage": 2,
    "est_tokens_mean": 2,
    "judge_delta": 1,
    "objectives": 2,
    "predictor": 1,
    "pareto_auc": 2,
    "per_lambda_incumbents": 2,
    "uncertainty": 1,
}


class RSIEngine:
    """RSI 演化引擎（每 ``RSIMemContext`` 一个）。"""

    def __init__(
        self,
        config: RSIMemConfig,
        db: RSIMemDatabase,
        tiers: TierManager,
        *,
        registry: PluginRegistry | None = None,
        predictor: OutcomePredictor | None = None,
        events: LocalEventLog | None = None,
        langfuse: LangfuseBridge | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._tiers = tiers
        self._validator = GenomeValidator(db)
        self._replay = ReplayEvaluator(config, db, tiers.retriever)
        self._improver = Improver(config, db)
        self.registry = registry or build_default_registry(
            config, self._improver,
        )
        self.predictor = predictor or OutcomePredictor(db)
        self._events = events
        self._langfuse = langfuse or NOOP_BRIDGE
        self._dream_lock = asyncio.Lock()
        self._dreaming = False
        self._traces_since_dream = 0
        # 后台 dream 任务的强引用（U1 债务④）：事件循环只持弱引用，
        # 不存引用会被 GC 中途回收；done 回调即移除
        self._bg_tasks: set[asyncio.Task] = set()
        self.last_report: dict[str, Any] | None = None
        # 记忆中间件暂存的本轮注入记忆 id（按会话），轨迹结算时取走
        self._injected: dict[Any, list[str]] = {}
        # RSI 中间件在轨迹结算后上报的 trace id（按会话），记忆写回取走
        self._last_trace: dict[Any, str] = {}

    # ------------------------------------------------------------------
    def note_injected(self, session: Any, memory_ids: list[str]) -> None:
        """记忆中间件记录某会话本轮注入的记忆 id（多注入面累积）。

        系统提示面（L3/索引/反射）与 reasoning hint 面在同一次 reply
        里都会调用 —— 累积合并（保序去重），轨迹结算时一次性取走。
        """
        if not memory_ids:
            return
        current = self._injected.setdefault(session, [])
        for mid in memory_ids:
            if mid not in current:
                current.append(mid)

    def pop_injected(self, session: Any) -> list[str]:
        """RSI 中间件取走该会话暂存的注入 id（供轨迹回归检查）。"""
        return self._injected.pop(session, [])

    def note_trace_finished(
        self, session: Any, trace_id: str | None,
    ) -> None:
        """RSI 中间件在轨迹结算（finish 落库）后上报 trace id。

        记忆中间件回合结束后经 :meth:`last_trace_id` 取走，补传给
        提取器写 ``source.trace_id``（此前恒为 None）。洋葱序保证 RSI
        （内层）的 finally 先于 OpsMemory（外层）执行；单挂 OpsMemory
        时取到 None，安全降级。
        """
        if trace_id:
            self._last_trace[session] = trace_id

    def last_trace_id(self, session: Any) -> str | None:
        """取走该会话最近一条已结算轨迹 id（pop 语义）。"""
        return self._last_trace.pop(session, None)

    # ------------------------------------------------------------------
    # 在线侧入口
    # ------------------------------------------------------------------
    def on_trace_finished(self) -> None:
        """轨迹完成后调用：计数并在达到阈值时调度后台 dream。"""
        self._traces_since_dream += 1
        if (
            self._config.rsi.enabled
            and self._traces_since_dream
            >= self._config.rsi.dream_every_n_traces
        ):
            task = asyncio.create_task(self._guarded_dream(auto=True))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

    async def _guarded_dream(self, *, auto: bool) -> None:
        """带锁保护的 dream；已在跑则跳过。"""
        if self._dreaming or self._dream_lock.locked():
            return
        self._dreaming = True
        try:
            await self.dream(trigger="auto" if auto else "manual")
        except Exception as e:  # noqa: BLE001
            logger.warning("RSI dream failed: %s", e)
        finally:
            self._dreaming = False

    async def iterate(
        self,
        rounds: int,
        *,
        judge_model: ChatModelBase | None = None,
    ) -> list[dict[str, Any]]:
        """连跑 ``rounds`` 轮 dream（基准/演示用），返回逐轮报告。"""
        reports = []
        for i in range(rounds):
            report = await self.dream(
                trigger=f"iterate:{i + 1}/{rounds}",
                judge_model=judge_model,
            )
            reports.append(report)
        return reports

    # ------------------------------------------------------------------
    # dream 主流程
    # ------------------------------------------------------------------
    async def dream(
        self,
        *,
        trigger: str = "manual",
        judge_model: ChatModelBase | None = None,
    ) -> dict[str, Any]:
        """执行一轮完整改进循环，返回报告。"""
        async with self._dream_lock:
            started = time.time()
            self._traces_since_dream = 0
            # 插件懒启动（首次 dream；幂等）
            await self.registry.start_all(
                getattr(self, "plugin_context", None),
            )
            await self.predictor._ensure_loaded()
            traces = await self._db.list_traces(limit=100)
            current = await self._db.active_genome()
            await self._emit(
                CAT_DREAM,
                {"phase": "start", "trigger": trigger,
                 "traces": len(traces),
                 "genome_version": current.version},
            )

            # ---- 可靠性保留门（RSI-Anything）----
            reliable = self.predictor.reliable(
                min_samples=self._config.rsi.predictor_min_samples,
                mae_threshold=self._config.rsi.predictor_mae_threshold,
            )
            conservative = not reliable

            # ---- 1. 诊断（证据完整性门 + E0-E3 标注）----
            diagnosis = annotate_diagnosis(
                await self._improver.diagnose(traces),
            )

            # ---- 2. 候选集：v0 恒在 + 插件计划变异 ----
            entries: list[
                tuple[GenomeSpec, list[MutationOutcome], str]
            ] = [(current, [], "v0")]
            dropped_by_gate = 0
            if diagnosis is not None:
                plans = await self.registry.propose_all(
                    diagnosis, current, traces, model=judge_model,
                )
                budget = self._config.rsi.candidates - 1
                for plugin_name, plan in plans[:budget]:
                    if conservative and not is_conservative_operator(
                        plan.operator,
                    ):
                        dropped_by_gate += 1
                        continue
                    outcome = await apply_plan(
                        current, plan, self._tiers,
                    )
                    if outcome is None:
                        continue
                    report = await self._validator.validate(
                        outcome.genome, current,
                    )
                    if not report.passed:
                        repaired = await self._validator.repair(
                            outcome.genome, current,
                        )
                        if repaired is None:
                            logger.info(
                                "candidate rejected by validator: %s",
                                report,
                            )
                            for sop in outcome.sop_candidates:
                                await self._tiers.muscle.archive(sop.id)
                            continue
                        outcome.genome = repaired
                    entries.append(
                        (outcome.genome, [outcome], plugin_name),
                    )

            # ---- 2b. MOEA/D incumbent 恢复（U2）：跨轮 per-λ 记忆 ----
            # meta 'rsi_moead_incumbents' 只存 {λ 索引: {spec_json,
            # objectives, dream_seq}}（【P1】不存 t 值——t′ 每轮在当前
            # ideal 下重算，杜绝旧值过门破非退化）。逐 λ 校验：失败
            # （含引用 SOP 已被上轮归档）丢弃该 λ incumbent 并重置为
            # v0；通过且 ≠ current 的作为 entries 追加项（origin=
            # 'incumbent:<i>'，不占 candidates 预算；outcomes 为空 ⇒
            # 无 sop_candidates，落选归档循环对其无操作）进本轮重评。
            dream_seq = await self._db.bump_meta("dream_seq")
            lambdas = self._config.rsi.subproblems
            inc_entry: list[int] = [0] * len(lambdas)
            inc_restore: list[str] = ["fresh"] * len(lambdas)
            try:
                stored_pool = json.loads(
                    await self._db.get_meta("rsi_moead_incumbents") or "{}",
                )
                if not isinstance(stored_pool, dict):
                    stored_pool = {}
            except ValueError:
                stored_pool = {}
            seen_spec: dict[str, int] = {current.model_dump_json(): 0}
            for i in range(len(lambdas)):
                item = stored_pool.get(str(i))
                if not isinstance(item, dict):
                    continue
                try:
                    spec = GenomeSpec.model_validate_json(
                        item.get("spec_json") or "",
                    )
                except Exception:  # noqa: BLE001 — 脏存储容忍
                    inc_restore[i] = "corrupt"
                    continue
                v_report = await self._validator.validate(spec, current)
                if not v_report.passed:
                    # 丢弃并重置为 v0（v0 恒在 ⇒ 非退化不受跨轮影响）
                    inc_restore[i] = f"dropped:{v_report.stage}"
                    continue
                key = spec.model_dump_json()
                if key in seen_spec:
                    # 存档即当前版（上轮已部署）或与其他 λ 同 spec：
                    # 复用既有条目（v0 或首个重评条目）的本轮新鲜评估
                    inc_entry[i] = seen_spec[key]
                    inc_restore[i] = (
                        "is_current" if seen_spec[key] == 0 else "shared"
                    )
                    continue
                entries.append((spec, [], f"incumbent:{i}"))
                seen_spec[key] = len(entries) - 1
                inc_entry[i] = len(entries) - 1
                inc_restore[i] = "reentered"

            # ---- 3. 预测（世界模型，E1）+ 回放评分（E2）----
            # U3-3b：排序目标的 quality 分量换混合
            # q′ = (1−pareto_alpha)·quality + pareto_alpha·pareto_auc
            # （ρ 扫描 Eq.1 前沿面积）。**ideal 与 t′ 必须同一目标空
            # 间**：ideal 也按混合目标维护，否则质量项 |z−q′| 恒大、
            # 掩蔽其他维度。对全候选（含 v0）一致计算，非退化论证不
            # 受影响；alpha=0 回退旧行为（纯 quality）。
            alpha = self._config.rsi.pareto_alpha

            def _mix(rep: ReplayReport) -> ObjectiveVector:
                obj = rep.objective
                if alpha <= 0.0:
                    return obj
                return replace(
                    obj,
                    quality=(1.0 - alpha) * obj.quality
                    + alpha * rep.pareto_auc,
                )

            ideal = IdealPoint()
            reports: list[ReplayReport] = []
            scored: list[
                tuple[
                    float,
                    GenomeSpec,
                    list[MutationOutcome],
                    ReplayReport,
                ]
            ] = []
            pending_predictions: list[
                tuple[str, CandidateFeatures, str]
            ] = []  # (prediction_id, features, candidate_op)
            candidate_predictions: dict[int, dict] = {}
            v0_objective: ObjectiveVector | None = None
            for genome, outcomes, origin in entries:
                v_report = await self._validator.validate(genome, current)
                replay_report = await self._replay.evaluate(
                    genome,
                    current,
                    traces,
                    validation_passed=v_report.passed,
                    judge_model=judge_model,
                    diagnosis_summary=diagnosis,
                )
                replay_report.is_v0 = origin == "v0"  # U3-3b 基准位标注
                reports.append(replay_report)
                ideal.update(_mix(replay_report))
                if origin == "v0":
                    v0_objective = replay_report.objective
                # —— 预测（仅变异候选）——
                if origin != "v0":
                    features = CandidateFeatures.extract(genome, current)
                    prediction = self.predictor.predict(features)
                    pid = await record_prediction(
                        self._db,
                        genome_version=None,
                        operator=genome.operator,
                        features=features,
                        prediction=prediction,
                    )
                    pending_predictions.append(
                        (pid, features, genome.operator),
                    )
                    candidate_predictions[id(genome)] = {
                        "operator": genome.operator,
                        "origin_plugin": origin,
                        "predicted": prediction.as_dict(),
                        "realized": None,  # 校准阶段回填
                    }
                scored.append((0.0, genome, outcomes, replay_report))

            # ---- 4a. 每候选全 λ 各一个 t′（U2 唯一子问题度量）----
            # u 取 candidate_predictions 的预测不确定性；v0 与无预测
            # 条目 u=0（已验证真实值不惩罚）。scored 首位同步为 λ₀ t′。
            lambda_u = self._config.rsi.uncertainty_lambda

            def _entry_objective(idx: int) -> ObjectiveVector:
                return _mix(scored[idx][3])

            entry_u: list[float] = []
            entry_tprime: list[list[float]] = []
            for idx, (genome, _, origin) in enumerate(entries):
                replay_report = scored[idx][3]
                if origin == "v0":
                    u = 0.0
                else:
                    u = float(
                        (candidate_predictions.get(id(genome)) or {})
                        .get("predicted", {})
                        .get("uncertainty")
                        or 0.0,
                    )
                entry_u.append(u)
                entry_tprime.append([
                    tprime(
                        _entry_objective(idx), w, ideal, u, lambda_u,
                    )
                    for w in lambdas
                ])
                scored[idx] = (
                    entry_tprime[idx][0], genome, scored[idx][2],
                    replay_report,
                )

            # ---- 3b. 预测-观测校准（E2 回放作为观测）----
            assert v0_objective is not None
            for (pid, features, op), (
                t0_penalized, genome, _, replay_report,
            ) in zip(
                pending_predictions,
                [s for s in scored if s[1] is not current],
            ):
                err = self.predictor.update(
                    features, replay_report.objective, v0_objective,
                )
                await backfill_realized(
                    self._db,
                    prediction_id=pid,
                    realized=realized_dict(
                        replay_report.objective, v0_objective,
                    ),
                )
                if id(genome) in candidate_predictions:
                    candidate_predictions[id(genome)]["realized"] = (
                        realized_dict(replay_report.objective, v0_objective)
                    )
                    candidate_predictions[id(genome)]["abs_err"] = round(
                        err, 4,
                    )
                # U4：uncertainty（E1）与 penalized（惩罚后的 λ₀ t′，
                # E2 口径）入事件载荷
                u_pred = float(
                    (candidate_predictions.get(id(genome)) or {})
                    .get("predicted", {})
                    .get("uncertainty")
                    or 0.0,
                )
                await self._emit(
                    CAT_PREDICTION,
                    {
                        "operator": op,
                        "predicted_abs_err": round(err, 4),
                        "uncertainty": round(u_pred, 4),
                        "penalized": round(t0_penalized, 6),
                        "mae": self.predictor.mae(),
                        "samples": self.predictor.sample_count(),
                    },
                )

            # ---- 4b. 邻域替换（U2 MOEA/D）：改进即换人，nr 封顶 ----
            # λ_i 的比较基准 = 恢复后的 incumbent 在**本轮重评**的 t′
            # （当前 ideal 下重算，非存值）；N=2 时邻域退化为全体 λ。
            # 按改进幅度降序贪心应用，单候选至多替换 replace_cap 个。
            inc_holder: list[int] = list(inc_entry)
            improvements: list[tuple[float, int, int]] = []  # (imp, λ, c)
            for ci in range(len(entries)):
                for li in range(len(lambdas)):
                    imp = (
                        entry_tprime[inc_holder[li]][li]
                        - entry_tprime[ci][li]
                    )
                    if imp > 1e-12:
                        improvements.append((imp, li, ci))
            improvements.sort(key=lambda x: (-x[0], x[1], x[2]))
            replace_count = [0] * len(entries)
            replacements: list[dict[str, Any]] = []
            for imp, li, ci in improvements:
                if replace_count[ci] >= self._config.rsi.replace_cap:
                    continue  # nr 封顶
                # 惰性重验：槽位可能已被更大改进换人
                imp_now = (
                    entry_tprime[inc_holder[li]][li]
                    - entry_tprime[ci][li]
                )
                if imp_now <= 1e-12:
                    continue
                inc_holder[li] = ci
                replace_count[ci] += 1
                replacements.append({
                    "lambda": li,
                    "origin": entries[ci][2],
                    "improvement": round(imp_now, 6),
                })

            # ---- 4c. 部署：质量门内最平衡（U2）----
            # 门 G = {c : t′(c,λ₀) ≤ t′(v0,λ₀)}（v0 恒在 ⇒ 非退化；
            # 门内全部是本轮回放数字，无跨轮口径错配）；
            # H† = argmax_G min(quality, cost_eff, safety)；
            # tie-break 链：min(f̂) 平 → λ₀-t′ 小者 → entries 序靠前者
            # （v0 恒为首位 ⇒ 末级平手天然偏 v0；version 不参与——
            # model_copy 保留父 version，同轮恒平无区分度）。
            v0_idx = 0
            v0_score = entry_tprime[v0_idx][0]
            gate = [
                idx for idx in range(len(entries))
                if entry_tprime[idx][0] <= v0_score + 1e-12
            ]

            def _balance_key(idx: int) -> tuple[float, float, int]:
                # U3-3b：平衡键同样用 q′ 混合目标（与 t′ 同一视野）
                obj = _entry_objective(idx)
                return (
                    min(obj.as_tuple()),
                    -entry_tprime[idx][0],
                    -idx,
                )

            winner_idx = max(gate, key=_balance_key)
            best_score = entry_tprime[winner_idx][0]
            winner, winner_outcomes, _winner_origin = entries[winner_idx]
            best_report = scored[winner_idx][3]

            report: dict[str, Any]
            deployed = False
            if winner is current:
                # v0 在门内最平衡 → 保持现状，落败候选 SOP 归档
                # （incumbent 重评条目无 sop_candidates，循环对其无操作）
                for _, genome, outcomes, _ in scored:
                    for outcome in outcomes:
                        for sop in outcome.sop_candidates:
                            await self._tiers.muscle.archive(sop.id)
                report = {
                    "trigger": trigger,
                    "deployed": False,
                    "reason": "无候选优于当前基因组（非退化保障生效）",
                    "candidates": len(scored),
                    "v0_score": round(v0_score, 4),
                }
            else:
                # ---- 保留门（pair gate）：部署前最终校验组装态 ----
                version = await self._db.active_version() + 1
                winner.version = version
                winner.parent_version = current.version
                winner.created_at = time.time()
                pair_report = await self._validator.validate(
                    winner, current,
                )
                if not pair_report.passed:
                    for outcome in winner_outcomes:
                        for sop in outcome.sop_candidates:
                            await self._tiers.muscle.archive(sop.id)
                    report = {
                        "trigger": trigger,
                        "deployed": False,
                        "reason": (
                            "保留门拒绝（部署前校验失败）: "
                            + str(pair_report)
                        ),
                        "candidates": len(scored),
                        "v0_score": round(v0_score, 4),
                    }
                else:
                    winner.scores = {
                        # 键名兼容：primary_tchebycheff 由 rsi_history
                        # 工具消费；值 = t′(λ₀)（λu=0 时与旧口径相同）
                        "primary_tchebycheff": round(best_score, 4),
                        "v0_tchebycheff": round(v0_score, 4),
                        "objectives": best_report.objective.as_tuple(),
                        "per_lambda_tchebycheff": [
                            round(t, 4) for t in entry_tprime[winner_idx]
                        ],
                        "balanced_min": round(
                            min(best_report.objective.as_tuple()), 4,
                        ),
                        "selector": "balanced",
                    }
                    await self._db.save_genome(winner)
                    # 获胜候选 SOP → active；其余候选 SOP → 归档
                    winning_ids = {
                        o.id
                        for o in _flat_sops(winner_outcomes)
                    }
                    for _, genome, outcomes, _ in scored:
                        for outcome in outcomes:
                            for sop in outcome.sop_candidates:
                                if sop.id in winning_ids:
                                    sop.status = "active"
                                    await self._db.update_memory(sop)
                                    self._tiers.muscle._mirror.write_muscle(
                                        sop,
                                    )
                                else:
                                    await self._tiers.muscle.archive(
                                        sop.id,
                                    )
                    deployed = True
                    report = {
                        "trigger": trigger,
                        "deployed": True,
                        "version": version,
                        "operator": winner.operator,
                        "notes": winner.notes,
                        "candidates": len(scored),
                        "v0_score": round(v0_score, 4),
                        "winner_score": round(best_score, 4),
                    }

            # ---- incumbent 池写回（U2）：被替换者换人、幸存者刷新
            #      objectives 为本轮值（【P1】下轮恢复即新鲜）----
            pool_out: dict[str, dict[str, Any]] = {}
            for i in range(len(lambdas)):
                holder = inc_holder[i]
                pool_out[str(i)] = {
                    "spec_json": entries[holder][0].model_dump_json(),
                    "objectives": [
                        round(x, 6)
                        for x in scored[holder][3].objective.as_tuple()
                    ],
                    "dream_seq": dream_seq,
                }
            await self._db.set_meta(
                "rsi_moead_incumbents",
                json.dumps(pool_out, ensure_ascii=False),
            )

            # ---- 报告增补：预测器/保守模式/证据/插件/候选预测 ----
            report.update(
                {
                    "conservative_mode": conservative,
                    "dropped_by_gate": dropped_by_gate,
                    "predictor": self.predictor.stats_summary(),
                    "plugins": self.registry.names(),
                    "evidence": {
                        "signals": _REPORT_EVIDENCE,
                        "legend": evidence_legend(),
                        "extra": self.registry.evidence_tags(),
                    },
                    "candidate_predictions": list(
                        candidate_predictions.values(),
                    ),
                    "replays": [self._brief(r) for r in reports],
                    # U3-3b：ρ 扫描前沿面积顶层键（供 U4 直接消费，
                    # 不靠 replays 猜行）
                    "pareto_auc_v0": round(scored[0][3].pareto_auc, 4),
                    "pareto_auc_winner": round(best_report.pareto_auc, 4),
                    "selector": "balanced",
                    "moead": {
                        "dream_seq": dream_seq,
                        "lambdas": lambdas,
                        "uncertainty_lambda": lambda_u,
                        "replace_cap": self._config.rsi.replace_cap,
                        "gate_size": len(gate),
                        "replacements": replacements,
                        "incumbents": [
                            {
                                "lambda": i,
                                "restore": inc_restore[i],
                                "holder": entries[inc_holder[i]][2],
                                "objectives": pool_out[str(i)]["objectives"],
                            }
                            for i in range(len(lambdas))
                        ],
                    },
                },
            )
            report["duration_s"] = round(time.time() - started, 2)
            await self._db.set_meta("last_dream_at", str(time.time()))
            await self._db.set_meta(
                "last_dream_report",
                json.dumps(report, ensure_ascii=False),
            )
            self.last_report = report
            # U4：per-λ incumbent 概要（E2）——事件与 langfuse 同名载荷
            per_lambda_incumbents = [
                {
                    "lambda": i,
                    "version": entries[inc_holder[i]][0].version,
                    "tprime": round(entry_tprime[inc_holder[i]][i], 6),
                    "dream_seq": dream_seq,
                }
                for i in range(len(lambdas))
            ]
            await self._emit(
                CAT_DREAM,
                {
                    "phase": "end",
                    "deployed": deployed,
                    "version": report.get("version"),
                    "operator": report.get("operator"),
                    "predictor_mae": report["predictor"]["mae"],
                    "conservative_mode": conservative,
                    "trigger": trigger,
                    # U4：U2/U3 新信号入观测面（沿用 CAT_DREAM 类别）
                    "pareto_auc_winner": report["pareto_auc_winner"],
                    "pareto_auc_v0": report["pareto_auc_v0"],
                    "selector": "balanced",
                    "per_lambda_incumbents": per_lambda_incumbents,
                },
            )
            self._langfuse.track("rsi_mem:dream", {
                **{
                    k: report.get(k)
                    for k in (
                        "deployed", "version", "operator", "candidates",
                        "v0_score", "winner_score", "trigger",
                        "pareto_auc_winner", "pareto_auc_v0",
                    )
                },
                "selector": "balanced",
                "per_lambda_incumbents": per_lambda_incumbents,
            })
            return report

    async def _emit(
        self, category: str, payload: dict[str, Any],
    ) -> None:
        if self._events is not None:
            await self._events.emit(category, payload)

    @staticmethod
    def _brief(r: ReplayReport) -> dict[str, Any]:
        return {
            "operator": r.candidate_op,
            "objectives": r.objective.as_tuple(),
            "coverage": r.coverage,
            "est_tokens_mean": r.est_tokens_mean,
            "judge_delta": r.judge_delta,
            "judge_reason": r.judge_reason,
            "pareto_auc": r.pareto_auc,
            "is_v0": r.is_v0,
        }

    # ------------------------------------------------------------------
    # 回滚
    # ------------------------------------------------------------------
    async def rollback(self) -> GenomeSpec | None:
        """回滚到上一版本（追加式版本历史：复制上一版为新版本）。"""
        versions = [g.version for g in await self._db.list_genomes()]
        if len(versions) < 2:
            return None
        prev = await self._db.load_genome(versions[1])
        if prev is None:
            return None
        prev.version = versions[0] + 1
        prev.parent_version = versions[0]
        prev.operator = "rollback"
        prev.created_at = time.time()
        await self._db.save_genome(prev)
        await self._emit(
            CAT_DREAM, {"phase": "rollback", "to_version": prev.version},
        )
        return prev


def _flat_sops(outcomes: list[MutationOutcome]):
    """获胜 outcomes 里的全部候选 SOP 记录。"""
    return [s for o in outcomes for s in o.sop_candidates]
