# -*- coding: utf-8 -*-
"""治理断路器（U4）：RSI 演化的速率/冷却/连续上限/质量回归防线。

**不可演化声明**：本模块是治理组件，刻意站在 RSI 可改对象
（GenomeSpec / 插件注册 / 记忆库 / meta 表）**之外**——

- 参数在 :class:`RSIEngine` 构造期从 ``config.governance`` **值拷贝
  快照**（frozen config 本身拦属性突变；整字段重赋值 ``ctx.config.
  governance = ...`` 被 pydantic 放行，但不影响已快照的参数——
  插件事后改配置无法松动断路器）；
- 状态持久化在 ``<data_dir>/governor.json``（原子写 tmp+os.replace）
  而非 meta 表——database ``set_meta`` 是 ctx 插件可写的面；
- 人工干预只有两条路：手工编辑 governor.json，或环境变量
  ``RSI_MEM_GOVERNOR_PIN``（Governor 构造时读取）。

**fail-closed**：状态文件解析失败/非 dict，**或本实例已见过/写过的
状态文件消失（被删）** → degraded，``check_deploy`` 一律拒绝（reason
含 "fail-closed"），不重置、不 fail-open；rollback（恢复动作）不受
断路器拦截（engine.rollback 另有 pin/复验防线）。损坏期间计数不可
恢复；每次检查前重读文件，修复后从修复文件的现状继续。删除的
fail-closed 保证覆盖本进程生命周期——进程重启后「删除后重启」与
「首装」从 data_dir 内部无法区分（无外部可信存储），如实声明。"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .config import GovernanceConfig

_STATE_FILENAME = "governor.json"


@dataclass(frozen=True)
class GovernorDecision:
    """一次治理判决。"""

    allowed: bool
    reason: str


def _default_state() -> dict[str, Any]:
    return {
        "deploys": [],  # [[ts, version], ...]
        "pinned_version": None,
        "deployment_pause": False,
        "paused_reason": "",
        "rollbacks": [],  # [[ts, version], ...]（审计）
    }


class Governor:
    """部署断路器：构造期参数快照 + 独立状态文件。

    每次判决/记账前重读状态文件（损坏即 degraded fail-closed，修复
    即恢复）。全部方法同步、无 IO 等待语义（文件小，直接读写）。
    """

    def __init__(
        self, cfg: GovernanceConfig, state_path: str,
    ) -> None:
        # —— 构造期值拷贝快照（此后 config 重赋值不影响本实例）——
        self._max_per_hour = cfg.max_deploys_per_hour
        self._min_interval = float(cfg.min_deploy_interval_s)
        self._consecutive_limit = cfg.consecutive_deploy_limit
        self.rollback_window = cfg.quality_rollback_window
        self.rollback_drop = cfg.quality_rollback_drop
        self.min_traces_after_deploy = cfg.min_traces_after_deploy
        self._state_path = state_path
        pin = os.environ.get("RSI_MEM_GOVERNOR_PIN", "").strip()
        self._env_pin = int(pin) if pin.isdigit() else None
        self._degraded = False
        self._had_state = False  # 本实例是否见过/写过状态文件
        self._state: dict[str, Any] = _default_state()
        self.refresh()

    # ------------------------------------------------------------------
    # 状态读写（原子 / fail-closed）
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """重读状态文件：损坏/非 dict → degraded；**已见过/写过的
        文件消失（被删）同样 degraded**——删除不是首装，是绕过断路器
        的路径（fail-closed，与损坏同判）。首装（从未写过）才允许
        默认空态。进程重启后无法区分「删除后重启」与「首装」（无
        data_dir 之外的可信存储），如实声明：删除的 fail-closed 保
        证覆盖本进程生命周期。
        """
        try:
            with open(self._state_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("state is not a dict")
        except FileNotFoundError:
            if self._had_state:
                self._state = _default_state()
                self._degraded = True  # 曾在的文件被删 → fail-closed
            else:
                self._state = _default_state()
                self._degraded = False
            return
        except (OSError, ValueError):
            # fail-closed：保持默认内存态并标记 degraded；不写文件
            # （不覆盖现场）、不清计数（计数在文件修复前不可知）
            self._state = _default_state()
            self._degraded = True
            return
        self._state = loaded
        self._had_state = True
        self._degraded = False

    def _save(self) -> None:
        """原子写（tmp + os.replace）；degraded 期不写（防毁现场）。"""
        if self._degraded:
            return
        directory = os.path.dirname(self._state_path) or "."
        os.makedirs(directory, exist_ok=True)
        tmp = self._state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False)
        os.replace(tmp, self._state_path)
        self._had_state = True  # 此后该文件消失即 fail-closed

    # ------------------------------------------------------------------
    # 判决
    # ------------------------------------------------------------------
    def check_deploy(
        self,
        now: float,
        trigger: str = "",
        consecutive_deploys: int = 0,
    ) -> GovernorDecision:
        """部署前四查：degraded / 暂停 / 钉住 / 速率+冷却+连续上限。"""
        self.refresh()
        if self._degraded:
            return GovernorDecision(
                False,
                "governor state unreadable (fail-closed)",
            )
        if self._state.get("deployment_pause"):
            return GovernorDecision(
                False,
                f"deployment paused: {self._state.get('paused_reason') or 'manual'}",
            )
        pin = self.pinned_version
        if pin is not None:
            return GovernorDecision(
                False, f"version pinned at v{pin}",
            )
        window = [
            d for d in self._state.get("deploys", [])
            if now - float(d[0]) < 3600.0
        ]
        if len(window) >= self._max_per_hour:
            return GovernorDecision(
                False,
                f"rate limit: {len(window)}/{self._max_per_hour} "
                "deploys in the last hour",
            )
        deploys = self._state.get("deploys") or []
        if deploys:
            last_ts = float(deploys[-1][0])
            if now - last_ts < self._min_interval:
                return GovernorDecision(
                    False,
                    f"cooldown: {self._min_interval - (now - last_ts):.0f}s"
                    " since last deploy",
                )
        if consecutive_deploys >= self._consecutive_limit:
            self._state["deployment_pause"] = True
            self._state["paused_reason"] = (
                f"consecutive deploys limit ({consecutive_deploys})"
            )
            self._save()
            return GovernorDecision(
                False,
                f"consecutive deploy limit {consecutive_deploys} reached;"
                " deployment paused (manual governor.json edit to clear)",
            )
        return GovernorDecision(True, "ok")

    def check_quality_regression(
        self, post_scores: list[float], pre_mean: float,
    ) -> GovernorDecision:
        """质量回归规则（dream 开始时由 engine 调用）。

        近 ``quality_rollback_window`` 条部署后真实轨迹均值较部署前
        窗口均值下滑 > ``quality_rollback_drop``，且部署后样本 ≥
        ``min_traces_after_deploy`` → 判退化（engine 据此自动回滚并
        暂停部署）。只读参数快照与入参，不读状态文件（degraded 不
        影响本规则——回滚是恢复动作，fail-closed 只拦部署）。
        """
        scores = post_scores[-self.rollback_window:]
        if len(scores) < self.min_traces_after_deploy:
            return GovernorDecision(True, "insufficient post samples")
        mean_post = sum(scores) / len(scores)
        drop = pre_mean - mean_post
        if drop > self.rollback_drop:
            return GovernorDecision(
                False,
                f"post-deploy mean {mean_post:.3f} dropped {drop:.3f}"
                f" > {self.rollback_drop} vs pre {pre_mean:.3f}",
            )
        return GovernorDecision(True, "ok")

    # ------------------------------------------------------------------
    # 记账（degraded 期丢弃——fail-closed 不落不可信账）
    # ------------------------------------------------------------------
    def record_deploy(self, now: float, version: int) -> None:
        self.refresh()
        if self._degraded:
            return
        deploys = self._state.setdefault("deploys", [])
        deploys.append([round(now, 3), version])
        # 只保留近 2 小时（速率窗口 ×2）的记录；最新一条永远保留
        # （冷却锚点），即使已出窗
        kept = [d for d in deploys if now - float(d[0]) < 7200.0]
        if not kept or kept[-1] is not deploys[-1]:
            kept.append(deploys[-1])
        self._state["deploys"] = kept
        self._save()

    def note_rollback(self, now: float, version: int | None = None) -> None:
        """记录一次回滚（审计；不清 pause——暂停只能人工清）。"""
        self.refresh()
        if self._degraded:
            return
        self._state.setdefault("rollbacks", []).append(
            [round(now, 3), version],
        )
        self._save()

    def pause_deployments(self, reason: str) -> None:
        """置部署暂停（质量回归自动回滚后由 engine 调用）。"""
        self.refresh()
        if self._degraded:
            return
        self._state["deployment_pause"] = True
        self._state["paused_reason"] = reason[:200]
        self._save()

    # ------------------------------------------------------------------
    # 观测面（doctor / bench）
    # ------------------------------------------------------------------
    @property
    def pinned_version(self) -> int | None:
        """env pin（构造期读取）优先，其次状态文件。"""
        return self._env_pin if self._env_pin is not None \
            else self._state.get("pinned_version")

    @property
    def deployment_pause(self) -> bool:
        return bool(self._state.get("deployment_pause"))

    @property
    def paused_reason(self) -> str:
        return str(self._state.get("paused_reason") or "")

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def state_path(self) -> str:
        return self._state_path

    def deploy_count_last_hour(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        return sum(
            1 for d in self._state.get("deploys", [])
            if now - float(d[0]) < 3600.0
        )

    def snapshot(self) -> dict[str, Any]:
        """给 doctor / bench 的只读概要。"""
        return {
            "pinned_version": self.pinned_version,
            "deployment_pause": self.deployment_pause,
            "paused_reason": self.paused_reason,
            "deploys_last_hour": self.deploy_count_last_hour(),
            "degraded": self._degraded,
        }
