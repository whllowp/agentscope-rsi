# -*- coding: utf-8 -*-
"""黄金数据（U2 执行级验证门的冻结基准）。

从 ``bench/rounds.py`` 的合成运维场景**原值冻结**而来。独立成 leaf
模块（不 import rsi_mem 任何其他模块）的原因：执行门
``rsi/execution_gate.py`` 需要这份数据，但它不能反向 import
``bench.rounds``——bench.rounds 顶部 import context，而
context → engine → execution_gate，会成环。bench 反向引用本模块
保持符号兼容（单一事实源，见 ``test_golden_single_source``）。
"""
from __future__ import annotations

GOLDEN_SCENARIOS: list[dict[str, str]] = [
    {"query": "线上磁盘使用率 95% 告警怎么处理", "kw": "磁盘"},
    {"query": "redis 内存告警不断触发", "kw": "redis"},
    {"query": "k8s pod 反复 crashloop", "kw": "crashloop"},
    {"query": "证书还有 3 天到期", "kw": "证书"},
    {"query": "数据库连接池耗尽", "kw": "连接池"},
]
"""五个运维场景（query=真实形态查询，kw=场景关键词探针）。"""

GOLDEN_KNOWLEDGE: dict[str, str] = {
    "磁盘": "磁盘满先 du -sh 定位大文件，清日志前确认轮转策略。",
    "redis": "redis 内存高先查 maxmemory 与淘汰策略，再看待淘汰键。",
    "crashloop": "crashloop 先 describe pod 看 events，再查日志退出码。",
    "证书": "证书到期用 openssl x509 -noout -enddate 核对，提前 7 天换。",
    "连接池": "连接池耗尽先看慢查询与泄漏，再调 pool_max 小步放大。",
}
"""每场景关键词对应的一条长期知识（执行门隔离栈的必召回标的）。"""
