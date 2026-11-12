# -*- coding: utf-8 -*-
"""RSIEngine：online → dream → select → deploy 循环。

Dream-RSI §3 的三阶段在声明式基因组上的落地：

1. **在线探索**：``RSIMiddleware`` 录制轨迹（``maybe_dream`` 计数）；
2. **离线 dreaming**：每 N 条轨迹后台触发一次 —— 诊断（证据门）→
   假设 → 变异出 M 个候选（**恒含当前基因组 v0**）→ 三段校验（一次
   修复机会，仍败则弃）→ 回放模拟器上按各子问题权重做 Tchebycheff
   评分；
3. **选择与部署**：主子问题 argmax ⇒ **数学上保证选中者不差于 v0**
   （v0 在候选集内、评分函数同一）——Dream-RSI 的非退化性质。
   部署写入新版本并激活获胜候选 SOP；落败候选 SOP 归档。

对外接口：``maybe_dream``（轨迹计数触发）、``dream``（手动/强制，供
``rsi_evolve`` 工具）、``rollback``（回到上一版本）。
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
from .improver import Improver
from .objectives import IdealPoint, tchebycheff
from .operators import MutationOutcome, apply_plan
from .replay import ReplayEvaluator, ReplayReport
from .validator import GenomeValidator


class RSIEngine:
    """RSI 演化引擎（每 ``RSIMemContext`` 一个）。"""

    def __init__(
        self,
        config: RSIMemConfig,
        db: RSIMemDatabase,
        tiers: TierManager,
    ) -> None:
        self._config = config
        self._db = db
        self._tiers = tiers
        self._validator = GenomeValidator(db)
        self._replay = ReplayEvaluator(config, db, tiers.retriever)
        self._improver = Improver(config, db)
        self._dream_lock = asyncio.Lock()
        self._dreaming = False
        self._traces_since_dream = 0
        self.last_report: dict[str, Any] | None = None
        # 记忆中间件暂存的本轮注入记忆 id（按会话），轨迹结算时取走
        self._injected: dict[Any, list[str]] = {}

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
            and self._traces_since_dream >= self._config.rsi.dream_every_n_traces
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
            traces = await self._db.list_traces(limit=100)
            current = await self._db.active_genome()

            # 1. 诊断（证据完整性门）
            diagnosis = await self._improver.diagnose(traces)

            # 2. 候选集：v0 恒在 + 变异候选
            entries: list[tuple[GenomeSpec, list[MutationOutcome]]] = [
                (current, []),
            ]
            if diagnosis is not None:
                plans = await self._improver.hypothesize(
                    diagnosis, model=judge_model,
                )
                budget = self._config.rsi.candidates - 1
                for plan in plans[:budget]:
                    outcome = await apply_plan(current, plan, self._tiers)
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
                    entries.append((outcome.genome, [outcome]))

            # 3. 回放评分（各子问题 Tchebycheff）
            ideal = IdealPoint()
            reports: list[ReplayReport] = []
            scored: list[tuple[float, GenomeSpec, list[MutationOutcome],
                               ReplayReport]] = []
            primary_w = self._config.rsi.subproblems[0]
            for genome, outcomes in entries:
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
                scored.append((0.0, genome, outcomes, replay_report))
            for i, (_, genome, _, replay_report) in enumerate(scored):
                t = tchebycheff(
                    replay_report.objective, primary_w, ideal,
                )
                scored[i] = (t, genome, scored[i][2], replay_report)

            # 4. 选择：Tchebycheff 距离最小者获胜（v0 在集合内 ⇒
            #    获胜者距离 ≤ v0 距离，非退化保障）
            scored.sort(key=lambda x: x[0])
            best_score, winner, winner_outcomes, best_report = scored[0]
            v0_report = next(
                r for (_, g, _, r) in scored if g is current
            )
            v0_score = next(
                s for (s, g, _, _) in scored if g is current
            )

            report: dict[str, Any]
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
                    "replays": [self._brief(r) for r in reports],
                }
            else:
                version = await self._db.active_version() + 1
                winner.version = version
                winner.parent_version = current.version
                winner.created_at = time.time()
                winner.scores = {
                    "primary_tchebycheff": round(best_score, 4),
                    "v0_tchebycheff": round(v0_score, 4),
                    "objectives": best_report.objective.as_tuple(),
                }
                await self._db.save_genome(winner)
                # 获胜候选 SOP → active；其余候选 SOP → 归档
                winning_ids = {o.id for o in _flat_sops(winner_outcomes)}
                for _, genome, outcomes, _ in scored:
                    for outcome in outcomes:
                        for sop in outcome.sop_candidates:
                            if sop.id in winning_ids:
                                sop.status = "active"
                                await self._db.update_memory(sop)
                                self._tiers.muscle._mirror.write_muscle(sop)
                            else:
                                await self._tiers.muscle.archive(sop.id)
                report = {
                    "trigger": trigger,
                    "deployed": True,
                    "version": version,
                    "operator": winner.operator,
                    "notes": winner.notes,
                    "candidates": len(scored),
                    "v0_score": round(v0_score, 4),
                    "winner_score": round(best_score, 4),
                    "replays": [self._brief(r) for r in reports],
                }

            report["duration_s"] = round(time.time() - started, 2)
            await self._db.set_meta("last_dream_at", str(time.time()))
            await self._db.set_meta(
                "last_dream_report",
                json.dumps(report, ensure_ascii=False),
            )
            self.last_report = report
            return report

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
        """回滚到上一版本（保留历史，只把 active 指针前移不了——
        SQLite 里 active=max(version)，故回滚实现为"复制上一版本为
        新版本"，保持只追加的版本历史）。"""
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
        return prev


def _flat_sops(outcomes: list[MutationOutcome]):
    """获胜 outcomes 里的全部候选 SOP 记录。"""
    return [s for o in outcomes for s in o.sop_candidates]
