# -*- coding: utf-8 -*-
"""连接器层：中间件、模型代理、HTTP 代理与服务工厂。

零源码修改的三个接入面：

1. **进程内中间件**（推荐）：``OpsMemoryMiddleware`` + ``RSIMiddleware``
   挂到 ``Agent(middlewares=[...])``；
2. **模型层代理**：``TraceProxyChatModel`` 包装任意 ChatModelBase；
3. **HTTP 代理**：OpenAI 兼容服务（MemoryProxy 风格），任意客户端把
   ``base_url`` 指过来即可获得记忆注入与轨迹采集。
"""
from .memory_middleware import OpsMemoryMiddleware
from .rsi_middleware import RSIMiddleware

__all__ = [
    "OpsMemoryMiddleware",
    "RSIMiddleware",
]
