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
import hashlib
import json
import os
import random
import time
from collections import Counter
from dataclasses import replace
from typing import Any

from agentscope._logging import logger
from agentscope.model import ChatModelBase

from ..config import RSIMemConfig
from ..governor import Governor, GovernorDecision
from ..memory.tiers import TierManager
from ..storage import GenomeSpec, RSIMemDatabase, TraceRecord
from ..telemetry import (
    CAT_DREAM,
    CAT_ERROR,
    CAT_PREDICTION,
    LangfuseBridge,
    LocalEventLog,
    NOOP_BRIDGE,
)
from .evidence import annotate_diagnosis, evidence_legend, tag_evidence
from .execution_gate import run_execution_gate
from .improver import Improver
from .level import assess, collect
from .objectives import IdealPoint, ObjectiveVector, tprime
from .operators import MutationOutcome, OperatorPlan, apply_plan
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
    "execution_gate": 2,
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
        # U4 治理断路器：**构造期**建立——参数自 config.governance 值
        # 拷贝快照（此后整字段重赋值 ctx.config.governance 不影响本实
        # 例，插件无法松动断路器）；状态在 data_dir/governor.json（非
        # 插件可写的 meta 表）。context 构造链（context.py）早于一切
        # plugin.start（首个 dream 才懒启动），故快照先于任何插件代码。
        self._governor = Governor(
            config.governance,
            state_path=os.path.join(config.data_dir, "governor.json"),
        )
        # 执行门注入预算同规则快照：max_inject_chars 也属治理参数，
        # 插件在 start(ctx) 里整字段重赋值 config.governance 放行
        # （pydantic 只拦属性突变），门若运行期读活配置即被旁路——
        # 构造期取值后恒以快照传给 run_execution_gate。
        self._max_inject_chars = config.governance.max_inject_chars
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
            # U3：后台 dream 失败必须可观测（此前仅 logger.warning，
            # events.jsonl 里静默消失）
            await self._emit(
                CAT_ERROR,
                {
                    "scope": "dream",
                    "trigger": "auto" if auto else "manual",
                    "error": f"{type(e).__name__}: {e}",
                },
                level="warn",
            )
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
    async def _backfill_e3(
        self, traces: list[TraceRecord], current: GenomeSpec,
    ) -> None:
        """U3 E3 回填：用部署后真实轨迹结算 winner 的 pending 预测。

        自证回路观测面的另一半：部署时记 pending（genome_version 锚
        定），此处用**真实外部信号**（production 轨迹）回填
        ``realized``（source="production"；同轮回放回填为 "replay"）。
        realized = mean(post-deploy score) − mean(pre-deploy window
        score)，cost/safety 置 0（轨迹分数只承载质量信号）。部署后
        样本 <5 或无部署前窗口 → 保持 pending（下轮再试）。

        production 行不进 predictor 在线统计（``_ensure_loaded``
        加载层过滤，跨重启成立）。
        """
        if current.version == 0:
            return  # 种子版：无「部署」可言
        try:
            preds = await self._db.load_predictions(realized_only=False)
        except Exception:  # noqa: BLE001 — 兼容空库
            return
        pending = [
            p for p in preds
            if p["genome_version"] == current.version
            and p["realized"] is None
        ]
        if not pending:
            return

        def _at(t: TraceRecord) -> float:
            return t.finished_at or t.started_at

        post = [
            t for t in traces
            if t.outcome is not None and _at(t) >= current.created_at
        ]
        pre = [
            t for t in traces
            if t.outcome is not None and _at(t) < current.created_at
        ]
        if len(post) < _E3_MIN_POST_SAMPLES or not pre:
            return  # 样本不足：保持 pending
        mean_post = sum(t.outcome.score for t in post) / len(post)
        mean_pre = sum(t.outcome.score for t in pre) / len(pre)
        realized = {
            "quality_delta": round(mean_post - mean_pre, 4),
            "cost_eff_delta": 0.0,
            "safety_delta": 0.0,
            "source": "production",
        }
        for p in pending:
            await backfill_realized(
                self._db, prediction_id=p["id"], realized=realized,
            )

    async def _governed_quality_check(
        self, traces: list[TraceRecord], current: GenomeSpec,
    ) -> GenomeSpec:
        """U4 治理质量规则：部署后真实质量回归 → 自动回滚 + 暂停。

        近 ``quality_rollback_window`` 条部署后带结局轨迹均值较部署前
        窗口均值下滑 > ``quality_rollback_drop``（且部署后样本 ≥
        ``min_traces_after_deploy``）→ 调 :meth:`rollback`（其内部有
        pin/复验防线，被拒则不回滚不暂停）并置 deployment_pause。
        回滚成功则返回新 current，本轮 dream 按回滚后基线继续。
        """
        if current.version == 0 or current.operator in ("seed",
                                                        "rollback"):
            return current

        def _at(t: TraceRecord) -> float:
            return t.finished_at or t.started_at

        post = [
            t for t in traces
            if t.outcome is not None and _at(t) >= current.created_at
        ]
        pre = [
            t for t in traces
            if t.outcome is not None and _at(t) < current.created_at
        ]
        if not pre:
            return current
        decision = self._governor.check_quality_regression(
            [t.outcome.score for t in post],
            sum(t.outcome.score for t in pre) / len(pre),
        )
        if decision.allowed:
            return current
        rolled = await self.rollback()
        if rolled is None:
            logger.warning(
                "governor quality regression detected but rollback "
                "refused: %s", decision.reason,
            )
            return current
        self._governor.note_rollback(time.time(), rolled.version)
        self._governor.pause_deployments(
            f"quality regression on v{current.version}: "
            f"{decision.reason}",
        )
        await self._emit(
            CAT_ERROR,
            {
                "scope": "governor",
                "kind": "quality_rollback",
                "from_version": current.version,
                "to_version": rolled.version,
                "reason": decision.reason,
            },
            level="warn",
        )
        return rolled
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
            # ---- 0. E3 回填（U3 自证回路观测面）：部署后真实轨迹结算 ----
            await self._backfill_e3(traces, current)
            # ---- 0b. 治理质量规则（U4）：部署后质量回归 → 自动回滚+暂停
            #      （bench 的 non_degraded 逻辑搬进运行时）----
            current = await self._governed_quality_check(traces, current)
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
                        # U3：保守模式丢弃此前只计数——现在落 CAT_ERROR
                        # 级 audit（可观测的截断行为）
                        await self._emit(
                            CAT_ERROR,
                            {
                                "scope": "conservative_gate",
                                "kind": "audit",
                                "operator": plan.operator,
                                "plugin": plugin_name,
                                "reason": "预测器不可靠期间丢弃非保守算子",
                            },
                            level="warn",
                        )
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
                            # U3：校验拒绝此前仅 logger.info——落 audit 事件
                            await self._emit(
                                CAT_ERROR,
                                {
                                    "scope": "validator",
                                    "kind": "audit",
                                    "operator": outcome.genome.operator,
                                    "stage": report.stage,
                                    "reason": "; ".join(
                                        report.problems,
                                    )[:200],
                                },
                                level="warn",
                            )
                            for sop in outcome.sop_candidates:
                                await self._tiers.muscle.archive(sop.id)
                            continue
                        outcome.genome = repaired
                    entries.append(
                        (outcome.genome, [outcome], plugin_name),
                    )

            # ---- 2b. MOEA/D incumbent 恢复（U2）：跨轮 per-λ 记忆 ----
            # meta 'rsi_moead_incumbents' 存 {λ 索引: {spec_json,
            # objectives, dream_seq, wins, last_restore}}（【P1】不存
            # t 值——t′ 每轮在当前 ideal 下重算，杜绝旧值过门破非退化；
            # wins = 该 λ 历史邻域替换胜出累计、last_restore = 最近一次
            # 恢复结局，二者是 doctor 证据制哨兵的数据源——restore 失败
            # （dropped:*/corrupt）才是机制失效证据，wins 只区分 info
            # 语气）。逐 λ 校验：失败（含引用 SOP 已被上轮归档）丢弃该
            # λ incumbent 并重置为 v0；通过且 ≠ current 的作为 entries
            # 追加项（origin='incumbent:<i>'，不占 candidates 预算；
            # outcomes 为空 ⇒ 无 sop_candidates，落选归档循环对其无
            # 操作）进本轮重评。
            dream_seq = await self._db.bump_meta("dream_seq")
            lambdas = self._config.rsi.subproblems
            inc_entry: list[int] = [0] * len(lambdas)
            inc_restore: list[str] = ["fresh"] * len(lambdas)
            inc_wins: list[int] = [0] * len(lambdas)
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
                stored_wins = item.get("wins")
                if isinstance(stored_wins, int) and stored_wins > 0:
                    inc_wins[i] = stored_wins  # 持久累计，跨轮不回退
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

            # ---- 2c. 探索注入（U3 多样性坍缩防御）----
            # 连续同算子部署 ≥K → 注入一个有界随机 retrieval_evolve
            # 扰动候选：random.Random(dream_seq) 种子可复现，参数在
            # RetrievalParams 边界内小步抖动，走下方全部校验/回放/
            # 双门（与任何候选同权竞争，v0 保护不变），origin="explore"。
            history = await self._db.list_genomes()  # 新版在前
            consecutive_same_operator = 0  # 头部同算子连跑（探索触发）
            for g in history:
                if g.operator != history[0].operator or g.operator == "seed":
                    break
                consecutive_same_operator += 1
            consecutive_deploys = 0  # U4：连续部署（无回滚隔断）轮数
            for g in history:
                if g.operator in ("seed", "rollback"):
                    break
                consecutive_deploys += 1
            if (
                consecutive_same_operator
                >= self._config.rsi.explore_after_same_operator
            ):
                changes = _explore_retrieval_changes(
                    random.Random(dream_seq), current,
                )
                explore_plan = OperatorPlan(
                    operator="retrieval_evolve",
                    rationale=(
                        f"U3 explore: 连续 {consecutive_same_operator} 轮"
                        f" {history[0].operator} 部署，注入多样性扰动"
                    ),
                    retrieval_changes=changes,
                )
                explore_outcome = await apply_plan(
                    current, explore_plan, self._tiers,
                )
                if explore_outcome is not None:
                    e_report = await self._validator.validate(
                        explore_outcome.genome, current,
                    )
                    if e_report.passed:
                        entries.append(
                            (explore_outcome.genome, [], "explore"),
                        )

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

            def _mix_e2(rep: ReplayReport) -> ObjectiveVector:
                # U2 双门的 E2 视野：质量分量换 quality_e2（剥 E1
                # judge_delta），其余维度与 _mix 同式——排序/门用同一
                # 混合口径，仅质量证据层级不同
                obj = rep.objective
                if alpha <= 0.0:
                    return replace(obj, quality=rep.quality_e2)
                return replace(
                    obj,
                    quality=(1.0 - alpha) * rep.quality_e2
                    + alpha * rep.pareto_auc,
                )

            ideal = IdealPoint()
            ideal_e2 = IdealPoint()
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
                ideal_e2.update(_mix_e2(replay_report))
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

            def _entry_objective_e2(idx: int) -> ObjectiveVector:
                return _mix_e2(scored[idx][3])

            entry_u: list[float] = []
            entry_tprime: list[list[float]] = []
            entry_tprime_e2: list[list[float]] = []
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
                # U2 双门：同一 t′ 式、E2 视野（quality_e2）的并行
                # 度量列——理想点与子问题度量都不含 E1 分量
                entry_tprime_e2.append([
                    tprime(
                        _entry_objective_e2(idx), w, ideal_e2, u, lambda_u,
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
                # U3 source 分类学：同轮回放回填标 "replay"（E2 自产
                # 证据），与 E3 生产回填（"production"）区分
                replay_realized = {
                    **realized_dict(
                        replay_report.objective, v0_objective,
                    ),
                    "source": "replay",
                }
                await backfill_realized(
                    self._db,
                    prediction_id=pid,
                    realized=replay_realized,
                )
                if id(genome) in candidate_predictions:
                    candidate_predictions[id(genome)]["realized"] = (
                        replay_realized
                    )
                    candidate_predictions[id(genome)]["abs_err"] = round(
                        err, 4,
                    )
                # U4：uncertainty（E1）与 penalized（惩罚后的 λ₀ t′，
                # E2 口径）入事件载荷；U3 增 realized_source
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
                        "realized_source": "replay",
                    },
                )

            # ---- 4b. 邻域替换（U2 MOEA/D）：改进即换人，nr 封顶 ----
            # λ_i 的比较基准 = 恢复后的 incumbent 在**本轮重评**的 t′
            # （当前 ideal 下重算，非存值）；N=2 时邻域退化为全体 λ。
            # 按改进幅度降序贪心应用，单候选至多替换 replace_cap 个。
            # wins[li] 在每次替换成功时 +1（持久累计，恢复自旧池）——
            # 替换活性证据，供 doctor 证据制哨兵区分「合法收敛」与
            # 「休眠」，不再拿「候选从未胜过 v0」当故障。
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
                # 惰性重验：槽位可能已经被更大改进换人
                imp_now = (
                    entry_tprime[inc_holder[li]][li]
                    - entry_tprime[ci][li]
                )
                if imp_now <= 1e-12:
                    continue
                inc_holder[li] = ci
                inc_wins[li] += 1
                replace_count[ci] += 1
                replacements.append({
                    "lambda": li,
                    "origin": entries[ci][2],
                    "improvement": round(imp_now, 6),
                })

            # ---- 4c. 部署：质量门内最平衡（U2）----
            # 门 G = {c : t′(c,λ₀) ≤ t′(v0,λ₀)}（v0 恒在 ⇒ 非退化；
            # 门内全部是本轮回放数字，无跨轮口径错配）；
            # U2 双门 G′ = G ∩ {c : t′_e2(c,λ₀) ≤ t′_e2(v0,λ₀)+ε}——
            # E2 视野（剥 E1 judge_delta 的 quality_e2）下也不劣于 v0
            # 才准部署，仅 E1 证据支持的质量优势不得越过 E2/E0 门。
            # v0 在两门恒属（≤ 自身）⇒ 非退化不变；judge 缺席时
            # judge_delta≡0 ⇒ t′_e2 ≡ t′，双门退化单门（行为不变）。
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
            v0_score_e2 = entry_tprime_e2[v0_idx][0]
            gate_e2 = {
                idx for idx in range(len(entries))
                if entry_tprime_e2[idx][0] <= v0_score_e2 + 1e-12
            }
            gate = [idx for idx in gate if idx in gate_e2]

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
            # U2 incumbent 驱逐标记：部署后置拒绝（pair gate/执行门/
            # governor 任一拒绝且 winner 非 current）时置为被拒 winner，
            # 池写回前把持有它的 λ 槽位重置回 v0
            rejected_winner: GenomeSpec | None = None
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
                    rejected_winner = winner
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
                    # ---- 执行门（U2）：隔离栈真实注入/检索/预算硬断言，
                    #      失败走 pair-gate 拒绝同款路径 ----
                    exec_result = await run_execution_gate(
                        winner, current, self._config,
                        max_inject_chars=self._max_inject_chars,
                    )
                    if not exec_result.passed:
                        for outcome in winner_outcomes:
                            for sop in outcome.sop_candidates:
                                await self._tiers.muscle.archive(sop.id)
                        rejected_winner = winner
                        report = {
                            "trigger": trigger,
                            "deployed": False,
                            "reason": (
                                "执行门拒绝（执行级验证失败）: "
                                + "; ".join(exec_result.failed_checks)
                            ),
                            "candidates": len(scored),
                            "v0_score": round(v0_score, 4),
                            "execution_gate": {
                                "passed": False,
                                "failed_checks": list(
                                    exec_result.failed_checks,
                                ),
                            },
                        }
                    else:
                        # ---- 治理断路器（U4）：执行门通过后、落库前 ----
                        gov = self._governor.check_deploy(
                            time.time(), trigger, consecutive_deploys,
                        )
                        if not gov.allowed:
                            # 拒绝走 pair-gate 同款路径（SOP 归档/
                            # incumbent 驱逐/v0 保持）
                            for outcome in winner_outcomes:
                                for sop in outcome.sop_candidates:
                                    await self._tiers.muscle.archive(
                                        sop.id,
                                    )
                            rejected_winner = winner
                            report = {
                                "trigger": trigger,
                                "deployed": False,
                                "reason": f"governor: {gov.reason}",
                                "candidates": len(scored),
                                "v0_score": round(v0_score, 4),
                                "execution_gate": {
                                    "passed": True,
                                    "failed_checks": [],
                                },
                                "governor_decision": {
                                    "allowed": False,
                                    "reason": gov.reason,
                                },
                            }
                        else:
                            winner.scores = {
                                # 键名兼容：primary_tchebycheff 由
                                # rsi_history 工具消费；值 = t′(λ₀）
                                # （λu=0 时与旧口径相同）
                                "primary_tchebycheff": round(best_score, 4),
                                "v0_tchebycheff": round(v0_score, 4),
                                "objectives": (
                                    best_report.objective.as_tuple()
                                ),
                                "per_lambda_tchebycheff": [
                                    round(t, 4)
                                    for t in entry_tprime[winner_idx]
                                ],
                                "balanced_min": round(
                                    min(
                                        best_report.objective.as_tuple(),
                                    ), 4,
                                ),
                                "selector": "balanced",
                                # U2：执行门通过打标（供 U4
                                # verify_executed 探针扫基因组历史）
                                "execution_gate": True,
                                # U3 帧锁定审计锚：渲染面指纹
                                "render_hash": _render_hash(winner),
                            }
                            await self._db.save_genome(winner)
                            # U4：部署记账（速率/冷却窗口）
                            self._governor.record_deploy(
                                time.time(), version,
                            )
                            # U3 E3 回填链：部署即记 pending 预测
                            # （genome_version 锚定），下轮 dream 用部署后
                            # 真实轨迹回填 realized.source="production"
                            e3_features = CandidateFeatures.extract(
                                winner, current,
                            )
                            await record_prediction(
                                self._db,
                                genome_version=version,
                                operator=winner.operator,
                                features=e3_features,
                                prediction=self.predictor.predict(
                                    e3_features,
                                ),
                            )
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
                                "execution_gate": {
                                    "passed": True,
                                    "failed_checks": [],
                                },
                                "governor_decision": {
                                    "allowed": True,
                                    "reason": gov.reason,
                                },
                            }

            # ---- U2 incumbent 驱逐：部署后置拒绝的 winner 不得经池重入
            #      （否则下轮恢复→重评重赢→再拒，死循环）；被逐槽位
            #      重置 v0——v0 恒为槽位底线，非退化不受影响 ----
            evicted: list[int] = []
            if rejected_winner is not None:
                for i in range(len(lambdas)):
                    if entries[inc_holder[i]][0] is rejected_winner:
                        inc_holder[i] = 0
                        evicted.append(i)

            # ---- incumbent 池写回（U2）：被替换者换人、幸存者刷新
            #      objectives 为本轮值（【P1】下轮恢复即新鲜）；
            #      wins/last_restore 随池持久化（doctor 证据制哨兵源）----
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
                    "wins": inc_wins[i],
                    "last_restore": inc_restore[i],
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
                    # U3 多样性观测面：候选 origin 分布 + 连续同算子部署数
                    "diversity": {
                        "origins": dict(Counter(
                            origin for _, _, origin in entries
                        )),
                        "consecutive_same_operator": (
                            consecutive_same_operator
                        ),
                    },
                    # U3 分布监测透传（诊断侧计算，报告侧不重复实现）
                    "distribution": (
                        (diagnosis or {}).get("distribution")
                    ),
                    # U4：L1-L5 分级自评（诚实快照，E0）
                    "level_assessment": assess(await collect(
                        self._db, self._events, self._config,
                    )),
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
                        "evicted": evicted,
                        "incumbents": [
                            {
                                "lambda": i,
                                "restore": inc_restore[i],
                                "wins": inc_wins[i],
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
            # U3：本轮回放 judge_delta 非空样本均值（无 judge 模型时
            # 0.0）——「评审未激活」哨兵的数据源
            jd_samples = [
                r.judge_delta for r in reports if r.judge_delta != 0.0
            ]
            judge_delta_mean = (
                round(sum(jd_samples) / len(jd_samples), 4)
                if jd_samples else 0.0
            )
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
                    # U2：执行门结论入观测面（v0 获胜未跑门 → None）
                    "execution_gate": report.get("execution_gate"),
                    # U3：judge 信号入观测面（自证回路的评审侧证据）
                    "judge_delta_mean": judge_delta_mean,
                    # U4：治理判决与自评等级（未尝试部署 → None）
                    "governor_decision": report.get("governor_decision"),
                    "level": report["level_assessment"]["level"],
                    "candidates": report.get("candidates"),
                    "traces_evaluated": len(traces),
                    # 审计聚合用：未部署轮的拒因（部署轮为 None）
                    "reason": report.get("reason"),
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
        *, level: str = "info",
    ) -> None:
        if self._events is not None:
            await self._events.emit(category, payload, level=level)

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
        """回滚到上一版本（追加式版本历史：复制上一版为新版本）。

        U4 内检（任何调用方同一防线——rsi_evolve 工具对插件可达、
        governor 自动回滚、测试直调）：

        - ``pinned_version`` 非空 → 拒（None + CAT_ERROR 事件）；
        - ``validate(prev, current)`` 复验失败（如 incumbent 污染的
          dangling SOP）→ 拒 + CAT_ERROR，不落库；
        - deployment_pause / governor degraded 不拦（回滚是恢复动作）。
        """
        pin = self._governor.pinned_version
        if pin is not None:
            await self._emit(
                CAT_ERROR,
                {
                    "scope": "governor",
                    "kind": "pin",
                    "action": "rollback_denied",
                    "pinned_version": pin,
                },
                level="warn",
            )
            return None
        versions = [g.version for g in await self._db.list_genomes()]
        if len(versions) < 2:
            return None
        prev = await self._db.load_genome(versions[1])
        if prev is None:
            return None
        current = await self._db.load_genome(versions[0])
        v_report = await self._validator.validate(prev, current)
        if not v_report.passed:
            await self._emit(
                CAT_ERROR,
                {
                    "scope": "governor",
                    "kind": "revalidation",
                    "action": "rollback_denied",
                    "stage": v_report.stage,
                    "reason": "; ".join(v_report.problems)[:200],
                },
                level="warn",
            )
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


# ----------------------------------------------------------------------
# U3 失败模式防御：探索扰动 / render_hash / E3 回填常量
# ----------------------------------------------------------------------
_E3_MIN_POST_SAMPLES = 5
"""E3 回填所需的最少部署后带结局轨迹数（不足则保持 pending）。"""


def _explore_retrieval_changes(
    rng: random.Random, current: GenomeSpec,
) -> dict[str, Any]:
    """有界随机 retrieval 扰动（纯函数，种子可复现）。

    参数在 ``RetrievalParams`` 边界内围绕当前值小步抖动（top_k ±2 /
    char_budget ±400 / threshold ±0.05），至少改 top_k 一项，保证候选
    与父代结构可区分（过「空演化」校验）。
    """
    cur = current.retrieval
    k_choices = [
        k for k in (
            cur.top_k - 2, cur.top_k - 1, cur.top_k + 1, cur.top_k + 2,
        ) if 1 <= k <= 20 and k != cur.top_k
    ]
    changes: dict[str, Any] = {"top_k": rng.choice(k_choices)}
    if rng.random() < 0.7:
        b_choices = [
            b for b in (
                cur.char_budget - 400, cur.char_budget - 200,
                cur.char_budget + 200, cur.char_budget + 400,
            ) if 200 <= b <= 8000 and b != cur.char_budget
        ]
        if b_choices:
            changes["char_budget"] = rng.choice(b_choices)
    if rng.random() < 0.3:
        t_choices = [
            t for t in (
                round(cur.score_threshold - 0.05, 4),
                round(cur.score_threshold + 0.05, 4),
            ) if 0.0 <= t <= 1.0 and t != cur.score_threshold
        ]
        if t_choices:
            changes["score_threshold"] = rng.choice(t_choices)
    return changes


def _render_hash(genome: GenomeSpec) -> str:
    """渲染面指纹：sha256(fragment_text()) 前 16 hex（帧锁定审计锚）。"""
    return hashlib.sha256(
        genome.fragment_text().encode("utf-8"),
    ).hexdigest()[:16]
