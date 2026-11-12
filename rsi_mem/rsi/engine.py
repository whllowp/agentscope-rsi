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
from .objectives import IdealPoint, ObjectiveVector, tchebycheff
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
        self.last_report: dict[str, Any] | None = None
        # 记忆中间件暂存的本轮注入记忆 id（按会话），轨迹结算时取走
        self._injected: dict[Any, list[str]] = {}

    # ------------------------------------------------------------------
    def note_injected(self, session: Any, memory_ids: list[str]) -> None:
        """记忆中间件记录某会话本轮注入的记忆 id。"""
        if memory_ids:
            self._injected[session] = memory_ids

    def pop_injected(self, session: Any) -> list[str]:
        """RSI 中间件取走该会话暂存的注入 id（供轨迹回归检查）。"""
        return self._injected.pop(session, [])

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
            asyncio.create_task(self._guarded_dream(auto=True))

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

            # ---- 3. 预测（世界模型，E1）+ 回放评分（E2）----
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
            primary_w = self._config.rsi.subproblems[0]
            v0_objective: ObjectiveVector | None = None
            v0_tchebycheff: float | None = None
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
                reports.append(replay_report)
                ideal.update(replay_report.objective)
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

            for i, (_, genome, _, replay_report) in enumerate(scored):
                t = tchebycheff(
                    replay_report.objective, primary_w, ideal,
                )
                scored[i] = (t, genome, scored[i][2], replay_report)
                if genome is current:
                    v0_tchebycheff = t

            # ---- 3b. 预测-观测校准（E2 回放作为观测）----
            assert v0_objective is not None
            for (pid, features, op), (_, genome, _, replay_report) in zip(
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
                await self._emit(
                    CAT_PREDICTION,
                    {
                        "operator": op,
                        "predicted_abs_err": round(err, 4),
                        "mae": self.predictor.mae(),
                        "samples": self.predictor.sample_count(),
                    },
                )

            # ---- 4. 选择：Tchebycheff 距离最小者获胜（v0 在集合内 ⇒
            #    获胜者距离 ≤ v0 距离，非退化保障）----
            scored.sort(key=lambda x: x[0])
            best_score, winner, winner_outcomes, best_report = scored[0]
            assert v0_tchebycheff is not None
            v0_score = v0_tchebycheff

            report: dict[str, Any]
            deployed = False
            if winner is current:
                # 没有候选胜过 v0 → 保持现状，落败 SOP 归档
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
                        "primary_tchebycheff": round(best_score, 4),
                        "v0_tchebycheff": round(v0_score, 4),
                        "objectives": best_report.objective.as_tuple(),
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
                },
            )
            report["duration_s"] = round(time.time() - started, 2)
            await self._db.set_meta("last_dream_at", str(time.time()))
            await self._db.set_meta(
                "last_dream_report",
                json.dumps(report, ensure_ascii=False),
            )
            self.last_report = report
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
                },
            )
            self._langfuse.track("rsi_mem:dream", {
                k: report.get(k)
                for k in (
                    "deployed", "version", "operator", "candidates",
                    "v0_score", "winner_score", "trigger",
                )
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
