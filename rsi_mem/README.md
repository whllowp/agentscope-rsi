# rsi_mem — AgentScope 的 RSI + 运维三层记忆扩展（零源码修改）

`rsi_mem` 是 [AgentScope 2.0](../src/agentscope) 的外部扩展包：**不改动
`src/agentscope` 任何一行代码**，通过官方扩展点（中间件 / ChatModelBase
包装器 / OpenAI 兼容 HTTP 代理）为运维 Agent 增加两类能力：

1. **RSI（Recursive Self-Improvement，递归自我改进）**——融合
   [Dream-RSI (arXiv 2609.14858 §3)](https://arxiv.org/html/2609.14858v2)
   与 [Harmony Harness (arXiv 2610.15150)](https://arxiv.org/html/2610.15150v1)；
2. **三层运维记忆**——肌肉记忆 / 短期记忆 / 长期记忆，短期自动建立指向
   长期的索引链接；参考 dsh-memory-evolve、TencentDB-Agent-Memory、ReMe
   （仓库内置 `ReMeMiddleware` 的成熟模式）。

```
┌────────────────────────── Agent（agentscope，未改动）──────────────────────────┐
│  on_reply ─ on_reasoning ─ on_acting ─ on_model_call ─ on_system_prompt        │
│      ▲              ▲            ▲                ▲               ▲            │
└──────┼──────────────┼────────────┼────────────────┼───────────────┼────────────┘
       │   OpsMemoryMiddleware     │         RSIMiddleware         │
       │  · 回合结束提取短期记忆    │  · 轨迹录制（发现树）          │
       │  · 周期固化审查(dsh)       │  · 工具/模型/权限观测          │
       │  · system prompt 多轨注入 │  · rsi_status/feedback/evolve │
       │  · memory_* 工具          │                               │
       └────────────┬─────────────┴───────────────┬───────────────┘
                    ▼                             ▼
          ┌─────────────────── RSIMemContext（共享）───────────────────┐
          │  TierManager        RSIEngine         SQLite + FTS5       │
          │  ├ muscle  肌肉记忆  ├ 轨迹→诊断(证据门)  ├ memories(+fts)  │
          │  ├ short   短期记忆  ├ 假设→算子计划     ├ traces           │
          │  └ long    长期L0-3  ├ 三段校验+修复     ├ genomes          │
          │  混合检索 bm25+向量   ├ 回放模拟器        └ pending_reviews  │
          │  +RRF+链接展开       └ 非退化选择→部署                        │
          └────────────────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 环境：Python 3.11+，已 pip install -e agentscope-rsi
# 零新增硬依赖（SQLite/FTS5 走标准库；fastapi/uvicorn 仅 HTTP 代理需要）

# 离线冒烟（无需 API key）
python -m rsi_mem.demo.main --backend mock

# 真实模型
export DASHSCOPE_API_KEY=sk-...
python -m rsi_mem.demo.main --backend dashscope --model qwen3.7-max
# 或任意 OpenAI 兼容上游
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://.../v1
python -m rsi_mem.demo.main --backend openai --model gpt-4o

# 测试（34 个，全部离线）
python -m pytest rsi_mem/tests -q
```

## 三种接入方式（都不改 agentscope 源码）

### 1. 中间件（推荐，功能最全）

```python
from rsi_mem import RSIMemConfig
from rsi_mem.connector import OpsMemoryMiddleware, RSIMiddleware
from rsi_mem.context import RSIMemContext
from rsi_mem.agent_builder import build_ops_agent

ctx = RSIMemContext(RSIMemConfig(data_dir="./.rsi_mem"))
async with LocalWorkspace(workdir="./workspace") as ws:
    agent, ctx = await build_ops_agent(
        model=my_chat_model, workspace=ws, context=ctx,
    )
    # = Agent(..., middlewares=[OpsMemoryMiddleware(ctx), RSIMiddleware(ctx)])
```

自己组装时把 `Toolkit(tools=..., skills_or_loaders=...)` 里加上
`await mw.list_tools()` 的结果即可（见 `agent_builder.py`）。

### 2. HTTP 代理（零代码，任意 OpenAI 客户端）

```bash
export RSI_MEM_PROXY_UPSTREAM_URL=https://api.deepseek.com
export RSI_MEM_PROXY_UPSTREAM_KEY=sk-...
uvicorn rsi_mem.connector.http_proxy:app --port 8130
```

```python
# 客户端只需把 base_url 指向代理
OpenAICredential(base_url="http://127.0.0.1:8130/v1", api_key="any")
```

代理把三层记忆上下文注入 system 消息并采集模型侧轨迹
（TencentDB MemoryProxy 模式）。局限：看不到工具执行层，完整 RSI
轨迹请用方式 1。

### 3. 模型层包装（观测任意 ChatModelBase 的用量/延迟）

```python
from rsi_mem.connector.model_proxy import TraceProxyChatModel
agent = Agent(..., model=TraceProxyChatModel(inner_model, sink=my_sink))
```

### 附：app service 模式

```python
from agentscope.app import create_app
from rsi_mem.service_factory import build_middleware_factory

app = create_app(..., extra_agent_middlewares=build_middleware_factory())
```

## 三层记忆设计

| 层 | 定位 | 写入 | 注入 | 生命周期 |
|---|---|---|---|---|
| **肌肉记忆** (`muscle`) | SOP / 条件反射，长期高效 | `memory_note` 提议、周期审查从重复工具序列蒸馏、RSI `skill_evolve` | 触发器（关键词/`re:`正则/`tool:`工具名）规则匹配命中即注入，**零 LLM 调用**；同时镜像为可加载 `SKILL.md` 技能 | `hit_count` 衰减，长期不命中提议降级为长期 L2 |
| **短期记忆** (`short`) | 会话临时 | 每轮结束 LLM/规则提取 | 最近 N 条 + 检索联动 | TTL（默认 24h）+ 每会话 LRU |
| **长期记忆** (`long`) | 大量知识低频 | 晋升 / LLM 提议 / L0 证据落库 | 检索按需（`memory_search` 或自动 top-k）；L3 核心档案每轮注入 | 去重合并提议、归档 |

- **短期 → 长期索引链接**（核心需求）：短期条目写入时自动混合检索长期
  层，命中即存 `links=[{target_id, kind:"related"}]`；检索命中短期条目时
  连带浮出其链接的长期条目（`via=link`）；晋升后保留 `provenance` 链。
- **L0-L3 分层**（TencentDB-Agent-Memory）：`L0_raw` 原始证据 →
  `L1_atom` 原子事实 → `L2_scenario` 场景块 → `L3_core` 核心档案。
- **分级确认**（dsh「AI 只提议，你确认」）：高置信低风险提议自动执行；
  SOP 与安全敏感内容（含 `rm -rf`/`systemctl restart` 等模式）一律进
  `pending_reviews`，由 `memory_confirm` 工具或用户确认。
- **检索**：FTS5 trigram bm25（中文友好）+ 可选向量（配置
  `embedding_model`）+ RRF 融合 + 阈值/条数/字符预算，参数由基因组控制。

## RSI 设计（论文机制 → 本实现）

| 论文机制 | 实现 |
|---|---|
| Dream-RSI：发现树在线录制 | `RSIMiddleware` 把每轮 reply 录成 `TraceRecord`（工具调用/模型成本/权限/结局评分/注入过的记忆 id） |
| Dream-RSI：历史 = 回放模拟器 | `rsi/replay.py` 在录制轨迹上做反事实评估：检索重放（确定性）、成本重估（注入字符差→token 差）、**回归检查**（成功轨迹用过的记忆，候选参数下必须仍召回） |
| Dream-RSI：候选恒含 v0、argmax 选择 → 非退化 | `RSIEngine.dream` 候选集始终含当前基因组，同一评分函数下 Tchebycheff 距离最小者获胜 ⇒ **选中者永不差于现状**（有单测断言该性质） |
| Harmony：MOEA/D 多目标 + Tchebycheff | `rsi/objectives.py`：质量（结局评分×回归覆盖+LLM评审增量）/ 成本效率 / 安全（权限拒绝率×校验）；多权重子问题（默认质量优先/成本优先） |
| Harmony：diagnose→hypothesize→operator | `rsi/improver.py` 诊断聚合（错误工具 Top、最好/最差轨迹）→ LLM 假设或规则兜底 → `operators.py` 四算子：`skill_evolve` / `logic_evolve` / `retrieval_evolve` / `generalize` |
| Harmony：证据完整性门 | 诊断引用的轨迹 id 全部对库校验，伪造引用直接丢弃 |
| Harmony：三段校验 + 一次修复 + 回滚 | `rsi/validator.py`：结构段（schema/上限）→ 冒烟段（SOP 引用存在、序列化往返）→ 完整性段（**锁定安全片段逐字保留**、拒绝空演化）；`rsi_evolve(rollback=True)` 一键回滚 |
| Harmony：gen/env 作用域 | `generalize` 算子做会话 SOP → 全局反射晋升；记忆条目带 `scope` 字段 |
| **RSI-Anything：世界模型** | `rsi/predictor.py` 候选执行**前**预测三目标增量（特征=候选 vs 父代结构差）；预测-观测对写 `predictions` 表并按算子在线校准（MAE/不确定性随样本下降） |
| **RSI-Anything：保留门** | ① 可靠性门：预测样本足够且 MAE 超阈 → 保守模式（仅放行 `retrieval_evolve` 小步候选）直至重新校准；② pair gate：胜者部署前再做一次组装态终校验 |
| **RSI-Anything：E0-E3 分级证据** | `rsi/evidence.py` —— E0 规则检查 / E1 模型预测 / E2 模拟回放 / E3 真实测量，诊断与 dream 报告逐信号标注（doctor 按证据强度排序发现） |
| **RSI-Anything：域适配器插件化** | `rsi/plugins.py` —— 计划生成收敛到 `PluginRegistry`，内置改进器也只是插件；第三方经 `rsi.plugins=["pkg:Plugin"]` 或 entry points `rsi_mem.plugins` 注入，**无法绕过校验与非退化门** |

**演化对象是声明式基因组**（`GenomeSpec`：提示词片段 + 检索参数 + 激活
SOP 集），不是任意代码——这是"不改动源码"约束下的安全演化边界。RSI
产生的 SOP 候选以 `status=candidate` 落库（不注入、不生效），仅在候选
基因组部署获胜时才转 `active`，落败即归档。

触发节奏：每 `dream_every_n_traces`（默认 20）条轨迹后台自动 dream；
`rsi_evolve` 工具随时手动触发/回滚；`rsi_feedback` 工具（👍/👎）为质量
目标提供真实用户信号。

## 数据目录

```
<data_dir>/                # 默认 <workdir>/.rsi_mem，或 RSI_MEM_HOME
├── memory.db              # SQLite（WAL + FTS5）：记忆/轨迹/基因组/待确认
└── mirror/
    ├── muscle/<name>/SKILL.md   # 肌肉记忆镜像（兼容 LocalSkillLoader，
    │                            #  demo 已自动挂为技能目录）
    └── longterm/<id>.md         # 长期记忆人类可读镜像
```

## Agent 可用的新工具

| 工具 | 作用 |
|---|---|
| `memory_search` | 跨三层检索；短期命中自动展开链接的长期条目 |
| `memory_note` | 提议写入持久记忆（分层/分 kind；走确认策略） |
| `memory_confirm` | 列出/确认/拒绝待确认提议 |
| `memory_link` | 手工补建短期→长期索引链接 |
| `memory_status` | 三层记忆统计 |
| `rsi_status` | 基因组版本 / dream 报告 / 统计 |
| `rsi_feedback` | 对处置结果打分（驱动自我改进） |
| `rsi_evolve` | 手动触发 dream 或回滚上一版基因组 |
| `rsi_history` | 基因组演化历史 |

## 配置速览（`RSIMemConfig`）

```python
RSIMemConfig(
    data_dir="./.rsi_mem",          # 或环境变量 RSI_MEM_HOME
    short_term=dict(ttl_hours=24, max_per_session=200, inject_recent=6),
    long_term=dict(mode="both", top_k=5, char_budget=2000),
    muscle=dict(max_reflex_bytes=2048, max_hits=3),
    review=dict(every_n_turns=10, auto_accept_confidence=0.85),
    rsi=dict(enabled=True, dream_every_n_traces=20, candidates=4),
    backend_model=None,             # 后台 LLM（None=复用 Agent 主模型）
    embedding_model=None,           # 可选向量检索
)
```

## 已知边界

- HTTP 代理模式看不到工具执行，RSI 轨迹只有模型侧（用中间件模式补全）。
- 后台 LLM（提取/评审/假设）默认复用 Agent 主模型；离线/失败自动降级
  为规则兜底，链路永不因后台模型缺失而中断。
- SQLite 单文件 + 线程锁：单进程多 Agent 共享安全；跨进程请每进程独立
  数据目录或仅读。

## 测试

```bash
python -m pytest rsi_mem/tests -q   # 47 个用例，全部离线
```

覆盖：FTS 中文检索、TTL/LRU、链接展开、晋升溯源、分级确认、SOP 蒸馏、
锁定片段防护、空演化拒绝、悬挂引用修复、回放回归检出、**非退化保障**、
回滚、中间件 e2e（真实 Agent + 假模型）、HTTP 代理注入与轨迹、mock 模型
全链路组装，以及本轮新增：**插件注册表（含坏插件容错）**、**预测器
先验→校准→可靠性门**、**保守模式拦截高风险计划**、**预测持久化回填**、
**本地事件日志与 langfuse 桥（含假客户端）**、**诊断证据分级**、
**10 轮基准非退化断言**、**CLI doctor/report**。
