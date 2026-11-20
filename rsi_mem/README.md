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

# 测试（161 个，全部离线）
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

## 记忆组件设计（本轮优化：对应 Anthropic 记忆组件）

在三层骨架之上，本轮按 Anthropic agent 记忆组件框架补齐五组机制
（详细设计与数据流见 [docs/memory-components-design.md](docs/memory-components-design.md)）：

| Anthropic 组件 | 本实现 | 主要接缝 |
|---|---|---|
| **显著性**（什么值得记） | 写入侧 `SalienceTag` 标注（种类/易变性/预测使用/价值），LLM 提取硬约束 + 规则兜底 volatile，存 `source["salience"]` 零 schema 变更 | `memory/salience.py`、`memory/extract.py` |
| **记忆即预测**（写入即预测、使用即检验） | 只在真实使用点（系统提示/推理注入/主动检索/晋升）回填 `settled_outcome`，内部检索不回填；过期走 `salience_expired` 遥测不写库 | `memory/tiers.py`（5 个结算挂点） |
| **Dreaming 固化**（空闲期巩固） | `Consolidator` 周期审查管线：晋升/SOP 蒸馏/轨迹校验/去重/断链修复/瘦身/肌肉降级/L0 回收，维护+污染提议合计 12 条预算 | `memory/consolidate.py`（中间件与代理两条触发链路） |
| **污染防御**（注入层验证） | `VerificationState` 三级（unverified/trace_verified/refuted）：负反馈先于一切 score 判定、使用型验证 + LLM 复核、漂移再验证、L3 驻留上限、矛盾检测互插旗标；`l3_eligible` 准入**不看 confidence**；refuted 记忆检索双保险拦截 | `memory/guards.py`、`memory/consolidate.py` |
| **记忆索引（Wiki）** | 长期记忆镜像为主题页 + `INDEX.md`；上下文顶部注入「知道什么、去哪找」小索引（行数/字节双预算），细节按需 `memory_search` | `storage/mirror.py`、`config.py`（`WikiConfig`） |
| 注入层瘦身（索引化注入的预算面） | L3 段 + 索引段合计 ≤ `l3_char_budget`=1200（共享 `L3_INDEX_MARGIN`=400）；超限按 (trace_verified, hit_count, dream_priority) 排序保留头部，其余降级 L1 并留 related 链接可回捞 | `memory/consolidate.py`、`memory/tiers.py` |

关键不变式：注入 id 经 `note_injected` 落进轨迹 `used_memory_ids`
（中间件与代理两条链路），是使用型验证、负反馈隔离、驻留/漂移判定
与预测结算的共同信号源；所有标注读写对脏数据容忍、settle/重建/遥测
失败只告警不阻断主链路。

## RSI 设计（论文机制 → 本实现）

v3 引擎深化（MOEA/D 邻域替换与平衡部署、不确定性惩罚选择、ρ 扫描
Pareto 评分、前缀盲评防泄漏、债务清偿）的逐机制数学定义与接缝文件见
[docs/rsi-engine-v3-design.md](docs/rsi-engine-v3-design.md)；本轮
「可信自改进」升级（执行反馈级验证 / 失败模式防御三件套 / 治理断路器
与 L1-L5 自评）见 [docs/trusted-rsi-design.md](docs/trusted-rsi-design.md)。

| 论文机制 | 实现 |
|---|---|
| Dream-RSI：发现树在线录制 | `RSIMiddleware` 把每轮 reply 录成 `TraceRecord`（工具调用含 `step_index`/`(n,k*)` 信号/模型成本/权限/结局评分/注入过的记忆 id） |
| Dream-RSI：历史 = 回放模拟器 | `rsi/replay.py` 在录制轨迹上做反事实评估：检索重放（确定性）、成本重估（注入字符差→token 差）、**回归检查**（成功轨迹用过的记忆，候选参数下必须仍召回） |
| Dream-RSI：候选恒含 v0 → 非退化 | `RSIEngine.dream` 候选集始终含当前基因组（v0 的预测不确定性 u≡0），部署经 λ₀ 质量门 G={c: t′(c,λ₀)≤t′(v0,λ₀)}，门内最平衡者获胜（tie-break 链末级偏 v0）⇒ **选中者永不差于现状**（有单测断言该性质） |
| Harmony：MOEA/D 多目标 + Tchebycheff | `rsi/objectives.py`：质量（结局评分×回归覆盖+LLM评审增量）/ 成本效率 / 安全（权限拒绝率×校验）；多权重子问题（默认质量优先 λ₀/成本优先 λ₁） |
| Harmony：MOEA/D 邻域替换与 per-λ incumbent | 每个 λ 维护独立 incumbent（跨轮持久化于 meta，恢复即重验、失败回退 v0）；改进即换人（降序贪心 + 惰性重验），单候选替换数 `replace_cap` 封顶（nr）；部署 = 质量门内 **H†=argmax min(quality′,cost,safety)** 最平衡选择 |
| Harmony：diagnose→hypothesize→operator | `rsi/improver.py` 诊断聚合（错误工具 Top、最好/最差轨迹）→ LLM 假设或规则兜底 → `operators.py` 四算子：`skill_evolve` / `logic_evolve` / `retrieval_evolve` / `generalize` |
| Harmony：证据完整性门 | 诊断引用的轨迹 id 全部对库校验，伪造引用直接丢弃 |
| Harmony：三段校验 + 一次修复 + 回滚 | `rsi/validator.py`：结构段（schema/上限）→ 冒烟段（SOP 引用存在、序列化往返）→ 完整性段（**锁定安全片段逐字保留**、拒绝空演化）；`rsi_evolve(rollback=True)` 一键回滚 |
| Harmony：gen/env 作用域 | `generalize` 算子做会话 SOP → 全局反射晋升；记忆条目带 `scope` 字段 |
| RSI-Anything：世界模型 | `rsi/predictor.py` 候选执行**前**预测三目标增量（特征=候选 vs 父代结构差）；预测-观测对写 `predictions` 表并按算子在线校准（MAE/不确定性随样本下降） |
| RSI-Anything：不确定性惩罚选择（Eq.7） | 统一子问题度量 **t′ = tchebycheff + λu·u**（距离域转写；u=预测不确定性，有样本 1/(1+n)、零样本先验 0.8；v0 与已验证条目 u≡0 不惩罚）；λu=`uncertainty_lambda` 默认 0 关闭；λ₀ 质量门、邻域替换、报告分数三处同一本账 |
| **RSI-Anything：保留门** | ① 可靠性门：预测样本足够且 MAE 超阈 → 保守模式（仅放行 `retrieval_evolve` 小步候选）直至重新校准；② pair gate：胜者部署前再做一次组装态终校验 |
| **RSI-Anything：E0-E3 分级证据** | `rsi/evidence.py` —— E0 规则检查 / E1 模型预测 / E2 模拟回放 / E3 真实测量，诊断与 dream 报告逐信号标注（doctor 按证据强度排序发现） |
| **RSI-Anything：域适配器插件化** | `rsi/plugins.py` —— 计划生成收敛到 `PluginRegistry`，内置改进器也只是插件；第三方经 `rsi.plugins=["pkg:Plugin"]` 或 entry points `rsi_mem.plugins` 注入，**无法绕过校验与非退化门** |
| 论文 Eq.1：ρ 扫描 Pareto 评分 | 回放对每条轨迹按 `rho_grid`（默认 5 档）插值 top_k/char_budget，得归一化质量-成本点 (n,a)，非支配过滤后算前沿下面积 `pareto_auc`（−并行度惩罚 λ∥·k\*/n）；经 `pareto_alpha` 混入排序目标 q′=(1−α)·quality+α·AUC（ideal 同空间维护，α=0 回退旧行为） |
| 论文附录 D.2：prefix-only 防泄漏 | 评审（judge）证据为**前缀视图**：剥离结局分数（`mean_outcome`/`worst·best_traces[].score`），前置前 5 步工具序列与 (n,k\*) 数据；v0 第二轮起同被评审（同视野公平），评审每候选一次不随 ρ 放大；诊断全量数据仍留给策略侧 |
| **Trusted-RSI U2：执行反馈级验证门** | validator 保证结构合法、pair gate 复验组装态，但都不执行——`rsi/execution_gate.py` 在**隔离栈**（临时目录全新 DB/Mirror/Tiers，不建 context）上对胜者做三类硬断言：①**必注入**（5 个黄金场景 query 渲染 system prompt，父代锁定片段 text 逐字在场）；②**必召回**（kw 探针 + 胜者检索参数下黄金知识必须命中——实证整 query 对磁盘/证书场景 FTS 零 3-gram 交集，故用关键词探针；语义=「检索参数不杀关键词召回」）；③**必守预算**（渲染 ≤ `governance.max_inject_chars` 绝对臂，不设相对臂防误杀内容增量）。O(1) 只验胜者，v0 获胜豁免；被拒走 pair-gate 同款路径并驱逐 incumbent 槽位（防重入死循环）。黄金数据冻结在 `rsi/golden.py`（leaf 模块防环，bench 同源） |
| **Trusted-RSI U2：锁定片段全字段锁** | `rsi/validator.py` 完整性段按 `(text, enabled, slot, triggers)` 四元组逐字段比对——旧版只比 text，`enabled=False`/`slot=reflex` 可让锁定片段静默退出渲染面而校验照过；与执行门必注入断言双保险 |
| **Trusted-RSI U2：证据层级感知选择（E2 双门）** | `rsi/replay.py` 额外产出 `quality_e2`（剥掉 `0.15×judge_delta` 的 E1 分量，只留 base×coverage + SOP 触发覆盖）；部署门从 G 升级为 G∩G′（E1 口径与 E2 口径两道 λ₀ Tchebycheff 门都不得劣于 v0）——**仅 LLM 评审（E1）支持的质量优势不得越过 E2/E0 门**；judge 缺席时双门退化单门，行为不变；v0 两门恒属 ⇒ 非退化不变 |
| **Trusted-RSI U3：自证回路防御** | 预测回填行强制 `source` 分类学：同轮回放标 `replay`（E2 自产）、部署后真实轨迹标 `production`（E3 外部）；`rsi/predictor.py` 加载层过滤 production 行（世界模型不拿部署后分数自校准，跨重启成立）；doctor 三哨兵：预测器自参考（近 20 条 realized 无 production + 冷启动豁免）、评审未激活（judge_delta_mean 恒 0，曾激活后沉默才 warn）、外部信号消失（全自产 info） |
| **Trusted-RSI U3：多样性坍缩防御** | 连续同算子部署 ≥3 轮 → dream 注入有界随机 `retrieval_evolve` 扰动候选（origin="explore"，种子可复现、走全部校验/回放/双门，v0 保护不变）；doctor 哨兵：近 10 版同算子 >80% warn、相邻版本去元数据哈希相同（空演化漂移）warn |
| **Trusted-RSI U3：帧锁定审计** | 部署基因组 `scores["render_hash"]`=sha256(fragment_text())[:16]——渲染面指纹（内容增量才移动）；doctor「外部审计包」输出近 6 版 render_hash/operator/部署斜率供人定期复核 |
| **Trusted-RSI U4：治理断路器（独立于 RSI 可改对象）** | `governor.py` 刻意站在 RSI 可改对象之外：参数在 `RSIEngine` 构造期从 frozen `GovernanceConfig` **值拷贝快照**（整字段重赋值旁路无效，且 context 构造链早于一切 plugin.start）；状态在 `data_dir/governor.json` 原子写而非插件可写的 meta 表；人工干预只有手工编辑该文件或 `RSI_MEM_GOVERNOR_PIN`。`check_deploy` 四查：fail-closed / 暂停 / 钉住 / 速率(6/h)+冷却(300s)+连续上限(8→自动暂停)；质量回归规则（部署后 20 条均值下滑 >0.1 且样本 ≥5）→ 自动回滚+暂停+审计事件；状态损坏**或被删**一律 fail-closed 拒部署，修复后从现状续算。rollback 是恢复动作不被拦，但有内检（pin 拒 + 目标版复验拒 dangling SOP） |
| **Trusted-RSI U4：L1-L5 分级自评** | `rsi/level.py` 对齐公开 RSI 能级榜单判据的诚实快照：三项自主（提出/部署/回放验证）全真 → **L3**；`verify_executed` 双源取证（CAT_DREAM 事件 ∪ 基因组 scores 的 execution_gate 标，末轮未部署不抹历史）；`imp_mutable`（基因组不覆盖引擎超参）与 `domain_generality`（黄金场景单域）如实卡在 L4/L5 外落进 failing 列表——**当前满分快照 = L3 + failing=[imp_mutable, domain_generality]，evidence=E0**；dream 报告挂 `level_assessment` 段、doctor 输出 |

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
├── governor.json          # 治理断路器状态（U4：独立于 meta 表，原子写；
│                          #  部署/回滚账本 + pause/pin，人工干预唯一入口）
└── mirror/
    ├── muscle/<name>/SKILL.md   # 肌肉记忆镜像（兼容 LocalSkillLoader，
    │                            #  demo 已自动挂为技能目录）
    ├── longterm/<id>.md         # 长期记忆人类可读镜像
    └── wiki/                    # 长期记忆 wiki（U4）：主题页 + INDEX.md
                                 #  顶部小索引（周期全量重建，可随时删重建）
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
    wiki=dict(enabled=True, index_max_lines=60, index_max_bytes=8192),
    rsi=dict(enabled=True, dream_every_n_traces=20, candidates=4,
             subproblems=[[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]],  # MOEA/D λ
             replace_cap=2,           # 邻域替换 nr 封顶
             uncertainty_lambda=0.0,  # t′ = tchebycheff + λu·u（0 关闭）
             rho_grid=[0.0, 0.25, 0.5, 0.75, 1.0],  # ρ 预算扫描
             pareto_alpha=0.5),       # q′=(1−α)·quality+α·pareto_auc
    governance=dict(               # 治理断路器（frozen；构造期快照生效）
        max_inject_chars=12000,      # 执行门绝对注入预算
        max_deploys_per_hour=6, min_deploy_interval_s=300.0,
        consecutive_deploy_limit=8,  # 触顶自动暂停部署
        quality_rollback_window=20, quality_rollback_drop=0.1,
        min_traces_after_deploy=5),  # 质量回归判定最少样本
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
python -m pytest rsi_mem/tests -q   # 161 个用例，全部离线
```

覆盖：FTS 中文检索、TTL/LRU、链接展开、晋升溯源、分级确认、SOP 蒸馏、
锁定片段防护、空演化拒绝、悬挂引用修复、回放回归检出、**非退化保障**、
回滚、中间件 e2e（真实 Agent + 假模型）、HTTP 代理注入与轨迹、mock 模型
全链路组装，以及上轮新增：**插件注册表（含坏插件容错）**、**预测器
先验→校准→可靠性门**、**保守模式拦截高风险计划**、**预测持久化回填**、
**本地事件日志与 langfuse 桥（含假客户端）**、**诊断证据分级**、
**10 轮基准非退化断言**、**CLI doctor/report**；记忆组件轮新增：
**显著性标注与「只有注入才结算」**、**污染防御（负反馈先于 score、
l3 准入不看 confidence、矛盾互插旗标）**、**固化管线四插槽（去重/
断链修复/注入瘦身/L0 回收 + 预算封顶与并发幂等）**、**Wiki 索引
（归簇/防撞/双预算截断/幂等重建/归档排除）**；本轮引擎深化新增：
**MOEA/D 邻域替换与 per-λ incumbent（λ₁ 激活/nr 封顶/跨轮 SOP 归档
回归/存档目标值不进门/引擎重启恢复）**、**t′ 不确定性惩罚（λu>0 压
伪优候选、λu=0 对照照常部署、λu×α 双重压制）**、**ρ 扫描 Pareto
前沿面积（点集/被支配剔除/缩放不变/全零分池）**、**前缀盲评（结局
分数剥离、(n,k\*) 前置、评审调用不随 ρ 放大）**、**doctor 子问题覆盖
与 Pareto 退化哨兵**，以及 U1 债务①-⑥ 清偿（merge 锚点/断链预过滤/
空拆分语义/后台任务强引用与原子计数/游标扫描与精确查重/注入面截断
兜底）；本轮可信自改进新增：**执行级验证门（全字段锁/kw 探针真命中/
锁定片段必注入/绝对预算臂/隔离栈无泄漏/拒绝驱逐 incumbent/门绿端到端
部署/预算快照免疫重赋值/黄金数据单源）**、**E2 双门（quality_e2 剥
E1 分量）**、**失败模式防御三件套（E3 production/replay 回填分类与
跨重启过滤/冷启动豁免的自参考哨兵/评审哨兵/分布监测/探索注入确定性
与触发/算子集中度/render_hash 审计包）**、**治理断路器（速率/冷却/
连续上限暂停/质量回归自动回滚/pin/fail-closed 损坏与删除/构造期快照
免疫/回滚复验/bench 宽松治理契约）**、**L1-L5 自评（事件历史双源
verify_executed + L3 诚实输出 + 空库 L1）**。
