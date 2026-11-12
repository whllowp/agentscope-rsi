# -*- coding: utf-8 -*-
"""RSIMemContext：一个数据目录对应的一整套共享组件。

两个中间件（OpsMemoryMiddleware / RSIMiddleware）、HTTP 代理、
tools 都从同一个 context 取组件 —— 这保证它们看到同一份三层记忆、
同一个 RSI 引擎与同一个数据库连接。组件懒初始化、幂等。
"""
from __future__ import annotations

import os
from typing import Any

from .config import RSIMemConfig
from .memory import TierManager
from .memory.consolidate import Consolidator
from .memory.extract import TurnExtractor
from .rsi.engine import RSIEngine
from .storage import MirrorStore, RSIMemDatabase
from .telemetry import (
    CAT_PLUGIN,
    LangfuseBridge,
    LocalEventLog,
    NOOP_BRIDGE,
)


class RSIMemContext:
    """组装并持有 ``rsi_mem`` 的全部运行时组件。"""

    def __init__(self, config: RSIMemConfig | None = None) -> None:
        self.config = config or RSIMemConfig()
        os.makedirs(self.config.data_dir, exist_ok=True)
        self.db = RSIMemDatabase(self.config.db_path)
        self.mirror = MirrorStore(self.config.data_dir)
        # —— 遥测：本地 JSONL（默认开） + langfuse（可选，自动探测）——
        self.events = LocalEventLog(
            self.config.events_path,
            enabled=self.config.telemetry.local_log,
        )
        self.langfuse = (
            LangfuseBridge(enabled=self.config.telemetry.langfuse)
            if self.config.telemetry.langfuse else NOOP_BRIDGE
        )
        self.tiers = TierManager(
            self.db, self.mirror, self.config,
        )
        self.extractor = TurnExtractor(
            self.config, self.tiers, langfuse=self.langfuse,
        )
        self.consolidator = Consolidator(self.config, self.tiers)
        self.engine = RSIEngine(
            self.config,
            self.db,
            self.tiers,
            events=self.events,
            langfuse=self.langfuse,
        )
        # 插件可通过 start(ctx) 拿到共享 context（首个 dream 时启动）
        self.engine.plugin_context = self  # type: ignore[attr-defined]

    async def start(self) -> None:
        """可选预热：启动全部 RSI 插件。"""
        loaded = self.engine.registry.names()
        await self.engine.registry.start_all(self)
        if loaded:
            await self.events.emit(
                CAT_PLUGIN, {"plugins": loaded, "phase": "started"},
            )

    async def close(self) -> None:
        """显式收尾（应用退出时调用）。"""
        try:
            await self.engine.registry.stop_all()
        finally:
            self.langfuse.flush()
            await self.db.close()

    async def snapshot(self) -> dict[str, Any]:
        """状态一览（rsi_status 用）。"""
        stats = await self.db.stats()
        genome = await self.db.active_genome()
        last_dream = await self.db.get_meta("last_dream_at")
        return {
            "config": self.config.describe(),
            "stats": stats,
            "genome": {
                "version": genome.version,
                "operator": genome.operator,
                "active_sops": len(genome.active_sops),
                "fragments": len(genome.prompt_fragments),
                "retrieval": genome.retrieval.model_dump(),
            },
            "last_dream_at": last_dream,
        }
