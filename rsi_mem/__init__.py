# -*- coding: utf-8 -*-
"""rsi_mem：AgentScope 的零源码修改 RSI + 运维三层记忆扩展。

通过连接器（AgentScope 中间件 / ChatModelBase 代理 / OpenAI 兼容 HTTP
代理）在不改动 ``agentscope`` 源码的前提下引入：

- 三层运维记忆：肌肉记忆（SOP/条件反射）、短期记忆（会话临时，TTL）、
  长期记忆（L0-L3 分层知识），短期记忆自动建立指向长期记忆的索引链接；
- RSI（递归自我改进）：轨迹录制 → 回放模拟器评估候选基因组 →
  多目标（质量/成本/安全）非退化选择，演化对象为声明式基因组
  （SOP 集 / 提示词片段 / 检索参数）。

公共入口见 :func:`build_ops_agent` 与
:class:`rsi_mem.connector.OpsMemoryMiddleware` /
:class:`rsi_mem.connector.RSIMiddleware`。
"""
from .config import RSIMemConfig
from .context import RSIMemContext
from .storage import (
    GenomeSpec,
    MemoryRecord,
    PendingReview,
    TraceRecord,
)

__version__ = "0.1.0"

__all__ = [
    "RSIMemConfig",
    "RSIMemContext",
    "GenomeSpec",
    "MemoryRecord",
    "PendingReview",
    "TraceRecord",
    "__version__",
]


def __getattr__(name: str):  # 懒加载重依赖入口，避免 import 主包即拉起
    if name in {"OpsMemoryMiddleware", "RSIMiddleware"}:
        from . import connector

        return getattr(connector, name)
    if name == "build_ops_agent":
        from . import agent_builder

        return getattr(agent_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
