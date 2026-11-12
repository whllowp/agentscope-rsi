# -*- coding: utf-8 -*-
"""``rsi_mem`` 的集中配置。

一个 :class:`RSIMemConfig` 唯一确定数据目录与三层记忆 / 固化 / RSI /
代理的行为参数；``OpsMemoryMiddleware``、``RSIMiddleware``、HTTP 代理
共享同一个配置对象（以及它派生的同一个数据库实例）。
"""
from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentscope.embedding import EmbeddingModelBase
from agentscope.model import ChatModelBase

MemoryMode = Literal["static_control", "agent_control", "both"]


class ShortTermConfig(BaseModel):
    """短期记忆参数（会话临时层）。"""

    ttl_hours: float = Field(default=24.0, ge=0.1)
    max_per_session: int = Field(default=200, ge=10)
    inject_recent: int = Field(default=6, ge=0)
    """每轮注入的近期短期记忆条数（受基因组 ``short_inject_recent``
    约束，取两者较小值）。"""


class LongTermConfig(BaseModel):
    """长期记忆参数（大量知识低频层）。"""

    mode: MemoryMode = "both"
    top_k: int = Field(default=5, ge=1, le=20)
    char_budget: int = Field(default=2000, ge=200, le=8000)


class MuscleConfig(BaseModel):
    """肌肉记忆参数（SOP / 条件反射层）。"""

    max_reflex_bytes: int = Field(default=2048, ge=256, le=8192)
    max_hits: int = Field(default=3, ge=1, le=10)
    """单轮最多注入的命中反射条数。"""


class ReviewConfig(BaseModel):
    """固化审查参数（dsh-memory-evolve 的周期审查 + 分级确认）。"""

    every_n_turns: int = Field(default=10, ge=1)
    auto_accept_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    """置信度不低于该值且非安全敏感的提议可自动执行。"""
    min_turns_for_sop: int = Field(default=3, ge=1)
    """同一模式至少出现几次工具轨迹才提议 SOP。"""


class RSIConfig(BaseModel):
    """RSI 引擎参数（Dream-RSI × Harmony Harness × RSI-Anything）。"""

    enabled: bool = True
    trace_outcome_base: float = 0.5
    dream_every_n_traces: int = Field(default=20, ge=2)
    """每积累 N 条完整轨迹触发一次后台 dream（离线改进评估）。"""
    candidates: int = Field(default=4, ge=2, le=8)
    """每轮 dream 的候选数（含当前基因组 v0，保证非退化）。"""
    subproblems: list[list[float]] = Field(
        default_factory=lambda: [
            [0.7, 0.2, 0.1],  # 质量优先
            [0.2, 0.7, 0.1],  # 成本优先
        ],
    )
    """多目标权重向量（质量/成本效率/安全）——Harmony 的 MOEA/D 子问题。"""
    regression_traces: int = Field(default=5, ge=1, le=20)
    """回放评估时抽查的近期成功轨迹条数（回归保障）。"""
    plugins: list[str] = Field(default_factory=list)
    """RSI 插件（"module:Attr" 形式的导入路径）。除内置插件外，
    额外通过这里或 ``rsi_mem.plugins`` entry points 装载。"""
    predictor_min_samples: int = Field(default=3, ge=1)
    """预测器可靠性门的最小样本数（RSI-Anything 的 retention gate：
    样本不足或预测误差未超阈时不收紧，保持探索）。"""
    predictor_mae_threshold: float = Field(default=0.25, ge=0.05, le=1.0)
    """预测平均绝对误差超过该值且样本足够 → 进入保守模式
    （只允许 retrieval_evolve 候选），直至预测重新校准。"""


class TelemetryConfig(BaseModel):
    """遥测参数（本地 JSONL + langfuse）。"""

    local_log: bool = True
    langfuse: bool = True
    """langfuse 桥开关；实际生效还需安装 langfuse 并配置
    ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``。"""


class ProxyConfig(BaseModel):
    """HTTP 代理（OpenAI 兼容，MemoryProxy 风格）参数。"""

    upstream_base_url: str = ""
    upstream_api_key: str = ""
    """上游真实 OpenAI 兼容服务；空则读环境变量
    ``RSI_MEM_PROXY_UPSTREAM_URL`` / ``RSI_MEM_PROXY_UPSTREAM_KEY``。"""
    inject_into_system: bool = True
    """把检索到的记忆拼进 system 消息（False 则不注入，仅记录轨迹）。"""


class RSIMemConfig(BaseModel):
    """顶层配置。"""

    data_dir: str = Field(
        default_factory=lambda: os.environ.get(
            "RSI_MEM_HOME",
            os.path.join(os.getcwd(), ".rsi_mem"),
        ),
    )
    short_term: ShortTermConfig = Field(default_factory=ShortTermConfig)
    long_term: LongTermConfig = Field(default_factory=LongTermConfig)
    muscle: MuscleConfig = Field(default_factory=MuscleConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    rsi: RSIConfig = Field(default_factory=RSIConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    backend_model: ChatModelBase | None = None
    """记忆提取 / RSI 评审等后台任务专用模型；``None`` 时复用 Agent
    当前模型（hook 里的 ``current_model``）。"""
    embedding_model: EmbeddingModelBase | None = None
    """可选向量模型；配置后启用 BM25+向量+RRF 混合检索。"""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "memory.db")

    @property
    def events_path(self) -> str:
        return os.path.join(self.data_dir, "logs", "events.jsonl")

    def effective_proxy_upstream(self) -> tuple[str, str]:
        """解析上游代理地址与密钥（配置优先，环境变量兜底）。"""
        base = self.proxy.upstream_base_url or os.environ.get(
            "RSI_MEM_PROXY_UPSTREAM_URL", "",
        )
        key = self.proxy.upstream_api_key or os.environ.get(
            "RSI_MEM_PROXY_UPSTREAM_KEY", "",
        )
        return base.rstrip("/"), key

    def describe(self) -> dict[str, Any]:
        """导出可序列化摘要（rsi_status 工具展示用）。"""
        return {
            "data_dir": self.data_dir,
            "short_term": self.short_term.model_dump(),
            "long_term": self.long_term.model_dump(),
            "muscle": self.muscle.model_dump(),
            "review": self.review.model_dump(),
            "rsi": self.rsi.model_dump(),
            "telemetry": self.telemetry.model_dump(),
            "has_backend_model": self.backend_model is not None,
            "has_embedding_model": self.embedding_model is not None,
        }
