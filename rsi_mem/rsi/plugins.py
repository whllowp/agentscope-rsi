# -*- coding: utf-8 -*-
"""RSI 插件体系（RSI-Anything 的 domain-adapter 思路）。

引擎的"改进智能"全部收敛为插件贡献物，内置实现也只是一个插件
（``builtin-improver``）。第三方可通过两种方式注入：

1. **配置装载**：``RSIMemConfig(rsi=["mypkg.rsi:MyPlugin"])``；
2. **entry points 发现**：任何 pip 包声明 ``[project.entry-points."rsi_mem.plugins"]``
   即被自动装载（本包不安装第三方，只做发现）。

插件可以贡献：

- ``propose_plans``：额外的候选算子计划（诊断 → 假设 → 计划）；
- ``evidence_tags``：额外的证据层级标注；
- ``start/stop``：生命周期（预热连接、后台线程等）。

所有贡献物经过与内置路径**完全相同**的三段校验、回放评估与非退化
选择 —— 插件不可能绕过安全门。
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agentscope._logging import logger
from agentscope.model import ChatModelBase

from ..config import RSIMemConfig
from .evidence import EvidenceLevel
from .operators import OperatorPlan

if TYPE_CHECKING:
    from ..context import RSIMemContext
    from ..storage import GenomeSpec, TraceRecord

ENTRY_POINT_GROUP = "rsi_mem.plugins"


class RSIPlugin(ABC):
    """RSI 插件协议：全部钩子可选，只实现需要的。"""

    name: str = "unnamed-plugin"
    description: str = ""

    async def start(self, ctx: "RSIMemContext") -> None:
        """生命周期：context 组装完成后调用一次。"""

    async def stop(self) -> None:
        """生命周期：context 关闭时调用。"""

    async def propose_plans(
        self,
        diagnosis: dict[str, Any] | None,
        current: "GenomeSpec",
        traces: list["TraceRecord"],
        *,
        model: ChatModelBase | None = None,
    ) -> list[OperatorPlan]:
        """贡献候选算子计划（空列表 = 本轮不贡献）。"""
        return []

    def evidence_tags(self) -> dict[str, EvidenceLevel]:
        """插件信号的证据层级标注。"""
        return {}


class BuiltinImproverPlugin(RSIPlugin):
    """内置改进器插件：包装 :class:`Improver`（LLM 假设 + 规则兜底）。

    保持与历史行为一致 —— 差别只是它现在与其他插件走同一条装配线。
    """

    name = "builtin-improver"
    description = "LLM 假设 + 规则兜底的默认改进器"

    def __init__(self, improver: Any) -> None:
        self._improver = improver

    async def propose_plans(
        self,
        diagnosis: dict[str, Any] | None,
        current: "GenomeSpec",
        traces: list["TraceRecord"],
        *,
        model: ChatModelBase | None = None,
    ) -> list[OperatorPlan]:
        if diagnosis is None:
            return []
        return await self._improver.hypothesize(diagnosis, model=model)


class PluginRegistry:
    """插件注册表：显式注册 + 配置装载 + entry points 发现。"""

    def __init__(self) -> None:
        self._plugins: dict[str, RSIPlugin] = {}
        self._started = False

    # ------------------------------------------------------------------
    def add(self, plugin: RSIPlugin) -> None:
        """注册一个插件（重名覆盖，后注册优先）。"""
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> RSIPlugin | None:
        return self._plugins.get(name)

    def names(self) -> list[str]:
        return list(self._plugins)

    def __len__(self) -> int:
        return len(self._plugins)

    # ------------------------------------------------------------------
    @staticmethod
    def _load_spec(spec: str) -> RSIPlugin | None:
        """加载 "module.path:AttrName" 形式的插件。"""
        try:
            module_path, _, attr = spec.partition(":")
            if not attr:
                raise ValueError(
                    "plugin spec must be 'module.path:AttrName'",
                )
            module = importlib.import_module(module_path)
            obj = getattr(module, attr)
            plugin = obj() if isinstance(obj, type) else obj
            if not isinstance(plugin, RSIPlugin):
                raise TypeError(f"{spec!r} is not an RSIPlugin")
            return plugin
        except Exception as e:  # noqa: BLE001 — 坏插件不拖垮引擎
            logger.warning("plugin load failed for %r: %s", spec, e)
            return None

    def load_from_config(self, specs: list[str]) -> list[str]:
        """按配置装载，返回成功装载的名字。"""
        loaded = []
        for spec in specs:
            plugin = self._load_spec(spec)
            if plugin is not None:
                self.add(plugin)
                loaded.append(plugin.name)
        return loaded

    def load_entry_points(self) -> list[str]:
        """发现已安装包声明的 ``rsi_mem.plugins`` entry points。"""
        loaded = []
        try:
            from importlib.metadata import entry_points

            eps = entry_points()
            selected = (
                eps.select(group=ENTRY_POINT_GROUP)
                if hasattr(eps, "select") else
                eps.get(ENTRY_POINT_GROUP, [])  # type: ignore[union-attr]
            )
            for ep in selected:
                try:
                    plugin = ep.load()
                    plugin = plugin() if isinstance(plugin, type) else plugin
                    if isinstance(plugin, RSIPlugin):
                        self.add(plugin)
                        loaded.append(plugin.name)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "entry point %s load failed: %s", ep.name, e,
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning("entry point discovery failed: %s", e)
        return loaded

    # ------------------------------------------------------------------
    async def start_all(self, ctx: "RSIMemContext") -> None:
        if self._started:
            return
        for plugin in self._plugins.values():
            try:
                await plugin.start(ctx)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "plugin %s start failed: %s", plugin.name, e,
                )
        self._started = True

    async def stop_all(self) -> None:
        for plugin in self._plugins.values():
            try:
                await plugin.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "plugin %s stop failed: %s", plugin.name, e,
                )
        self._started = False

    # ------------------------------------------------------------------
    async def propose_all(
        self,
        diagnosis: dict[str, Any] | None,
        current: "GenomeSpec",
        traces: list["TraceRecord"],
        *,
        model: ChatModelBase | None = None,
    ) -> list[tuple[str, OperatorPlan]]:
        """汇总所有插件的计划（按注册序，保留插件名溯源）。"""
        out: list[tuple[str, OperatorPlan]] = []
        for plugin in self._plugins.values():
            try:
                plans = await plugin.propose_plans(
                    diagnosis, current, traces, model=model,
                )
            except Exception as e:  # noqa: BLE001 — 插件失败不拖垮 dream
                logger.warning(
                    "plugin %s propose failed: %s", plugin.name, e,
                )
                continue
            for plan in plans or []:
                if isinstance(plan, OperatorPlan):
                    out.append((plugin.name, plan))
        return out

    def evidence_tags(self) -> dict[str, int]:
        merged: dict[str, int] = {}
        for plugin in self._plugins.values():
            try:
                for k, v in plugin.evidence_tags().items():
                    merged[f"plugin:{plugin.name}:{k}"] = int(v)
            except Exception:  # noqa: BLE001
                continue
        return merged


def build_default_registry(
    config: RSIMemConfig, improver: Any,
) -> PluginRegistry:
    """构造默认注册表：内置插件 + 配置插件 + entry points。"""
    registry = PluginRegistry()
    registry.add(BuiltinImproverPlugin(improver))
    registry.load_from_config(config.rsi.plugins)
    registry.load_entry_points()
    return registry
