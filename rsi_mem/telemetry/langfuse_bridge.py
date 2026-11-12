# -*- coding: utf-8 -*-
"""langfuse 可选桥接：把 rsi_mem 事件与 LLM 调用投递到 langfuse 面板。

设计约束：**langfuse 是可选依赖** —— 未安装或未配置密钥时自动退化为
no-op（本地 JSONL 仍然完整记录）。API 版本差异（v2/v3）通过特性探测
兼容，任何一次投递失败只告警、不影响主链路。

启用方式（环境变量，与 langfuse SDK 约定一致）::

    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_HOST=https://cloud.langfuse.com  # 自托管则改地址
"""
from __future__ import annotations

import time
from typing import Any, Protocol

from agentscope._logging import logger

try:  # langfuse 可选
    import langfuse as _langfuse_mod
except ImportError:  # pragma: no cover - 环境相关
    _langfuse_mod = None


class _EventSink(Protocol):
    """langfuse 客户端需要满足的最小接口（v2/v3 探测后包装）。"""

    def emit_event(self, name: str, **kw: Any) -> Any: ...


class LangfuseBridge:
    """事件 → langfuse 的尽力而为桥。

    不直接依赖 langfuse 具体版本 API：优先 ``get_client()``（v3），
    再退 ``Langfuse()``（v2）；两者都拿不到时保持禁用。测试可通过
    ``inject_client`` 注入假客户端验证投递路径。
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        client: Any | None = None,
    ) -> None:
        self._enabled = enabled
        self._client: Any | None = client
        self._warned = False
        if self._client is None and enabled:
            self._client = self._autodiscover()

    # ------------------------------------------------------------------
    def _autodiscover(self) -> Any | None:
        if _langfuse_mod is None:
            return None
        try:
            if hasattr(_langfuse_mod, "get_client"):
                client = _langfuse_mod.get_client()
                if getattr(client, "auth_check", lambda: False)():
                    return client
                return None  # 密钥未配置
            if hasattr(_langfuse_mod, "Langfuse"):
                client = _langfuse_mod.Langfuse()
                if getattr(client, "auth_check", lambda: False)():
                    return client
        except Exception as e:  # noqa: BLE001
            logger.warning("langfuse autodiscover failed: %s", e)
        return None

    @property
    def active(self) -> bool:
        return self._enabled and self._client is not None

    def inject_client(self, client: Any) -> None:
        """测试注入口（也是自定义 langfuse 客户端的通道）。"""
        self._client = client
        self._enabled = client is not None

    # ------------------------------------------------------------------
    def track(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """投递一条自定义事件（尽力而为）。"""
        if not self.active:
            return
        try:
            c = self._client
            emitted = False
            for method in ("emit_event", "event"):
                fn = getattr(c, method, None)
                if callable(fn):
                    if method == "emit_event":
                        fn(name, payload=payload or {})
                    else:  # v2 风格 event(name=...)
                        fn(name=name, input=payload or {})
                    emitted = True
                    break
            if not emitted and hasattr(c, "trace"):
                c.trace(name=name, metadata=payload or {})
        except Exception as e:  # noqa: BLE001
            if not self._warned:
                logger.warning(
                    "langfuse track failed (%s); further errors suppressed.", e,
                )
                self._warned = True

    def flush(self) -> None:
        """退出前冲刷（best-effort）。"""
        if not self.active:
            return
        try:
            fn = getattr(self._client, "flush", None)
            if callable(fn):
                fn()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # LLM 调用观测（记忆提取 / RSI 评审 / 假设生成）
    # ------------------------------------------------------------------
    class _Obs:
        def __init__(self, bridge: "LangfuseBridge", name: str) -> None:
            self._bridge = bridge
            self._name = name
            self._t0 = 0.0

        def __enter__(self) -> "LangfuseBridge._Obs":
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            duration_ms = round((time.perf_counter() - self._t0) * 1000, 1)
            self._bridge.track(
                "rsi_mem:llm_job",
                {
                    "job": self._name,
                    "duration_ms": duration_ms,
                    "error": exc_type.__name__ if exc_type else None,
                },
            )

    def observe(self, name: str) -> "LangfuseBridge._Obs":
        """上下文管理器：观测一次后台 LLM 任务。

        Example::

            with bridge.observe("memory_extract"):
                await model(...)
        """
        return LangfuseBridge._Obs(self, name)


NOOP_BRIDGE = LangfuseBridge(enabled=False)
"""未启用 langfuse 时的共享 no-op 实例。"""
