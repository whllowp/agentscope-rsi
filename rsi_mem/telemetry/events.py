# -*- coding: utf-8 -*-
"""本地遥测：结构化 JSONL 事件日志（快速发现的数据底座）。

所有 RSI/记忆关键事件落 ``<data_dir>/logs/events.jsonl``，每行一个
JSON 对象：``{ts, level, category, payload}``。``rsi_mem.cli doctor``
基于它做快速诊断；langfuse 桥接（``langfuse_bridge``）把同一批事件
异步投递到 langfuse 面板。

写盘策略：追加写 + 每事件独立 open/close（事件量低，换取崩溃安全与
零锁竞争）；失败只告警不抛出 —— 遥测永不阻塞主链路。
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from agentscope._logging import logger

EventCategory = str  # 自由类别，常用见下方常量

CAT_TRACE = "trace"
CAT_DREAM = "dream"
CAT_PREDICTION = "prediction"
CAT_MEMORY = "memory"
CAT_PLUGIN = "plugin"
CAT_REVIEW = "review"
CAT_ERROR = "error"


class LocalEventLog:
    """append-only JSONL 事件日志（并发安全、尽力而为）。"""

    def __init__(self, path: str, enabled: bool = True) -> None:
        self._path = path
        self._enabled = enabled
        if enabled:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    @property
    def path(self) -> str:
        return self._path

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _append_sync(
        self, level: str, category: str, payload: dict[str, Any],
    ) -> None:
        record = {
            "ts": round(time.time(), 3),
            "level": level,
            "category": category,
            "payload": payload,
        }
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(record, ensure_ascii=False,
                               default=str) + "\n",
                )
        except OSError as e:  # 遥测失败不阻塞主链路
            logger.warning("event log write failed: %s", e)

    async def emit(
        self,
        category: str,
        payload: dict[str, Any] | None = None,
        *,
        level: str = "info",
    ) -> None:
        """追加一条事件（异步，内部走线程池）。"""
        if not self._enabled:
            return
        await asyncio.to_thread(
            self._append_sync, level, category, payload or {},
        )

    def emit_sync(
        self,
        category: str,
        payload: dict[str, Any] | None = None,
        *,
        level: str = "info",
    ) -> None:
        """同步追加（已在线程池/后台任务里时用）。"""
        if not self._enabled:
            return
        self._append_sync(level, category, payload or {})

    # ------------------------------------------------------------------
    # 读取侧（doctor / 快速发现）
    # ------------------------------------------------------------------
    def read(
        self,
        *,
        category: str | None = None,
        level: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """读取最近 ``limit`` 条匹配事件（时间正序）。"""
        if not self._enabled or not os.path.isfile(self._path):
            return []
        out: list[dict[str, Any]] = []
        try:
            with open(self._path, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        for line in lines[-limit * 3:]:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if category and rec.get("category") != category:
                continue
            if level and rec.get("level") != level:
                continue
            out.append(rec)
            if len(out) > limit:
                out.pop(0)
        return out
