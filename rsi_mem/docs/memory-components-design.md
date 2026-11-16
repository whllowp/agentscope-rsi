# 记忆组件设计 —— 显著性 / 记忆即预测 / Dreaming 固化 / 污染防御 / Wiki 索引 / 注入瘦身

本文档描述 `rsi_mem` 本轮记忆优化（工作区改动，基线 commit `339c5e9`）
引入的五组机制，及其与 Anthropic agent 记忆组件框架的对应关系：
**显著性（salience）**、**记忆即预测（memory as prediction）**、
**dreaming 固化（周期巩固）**、**污染防御（contamination defenses）**、
**记忆索引（Wiki / 顶部小索引）**；注入层瘦身是索引化注入的预算运营面。
每节给出**接缝文件**（机制挂接的位置）与**数据流**。

所有行为描述均以下列源码为准（引用为 `文件:行`）：

| 机制 | 核心接缝文件 |
|---|---|
| U1 显著性 / 记忆即预测 | `memory/salience.py`、`storage/models.py`（`SalienceTag`）、`memory/extract.py`、`memory/shortterm.py`、`memory/tiers.py`（结算挂点） |
| Dreaming 固化管线 | `memory/consolidate.py`（`Consolidator`）、`memory/tiers.py`（`propose`/`_dispatch`）、`config.py`（`ReviewConfig`） |
| U2 污染防御 | `memory/guards.py`、`memory/consolidate.py::_verify_against_traces`、`memory/retrieval.py`（双保险）、`connector/memory_middleware.py` + `connector/http_proxy.py`（信号接缝）、`rsi/engine.py`（`note_injected`） |
| U4 Wiki 索引 | `storage/mirror.py`（归簇/渲染/重建）、`memory/tiers.py::_core_context_sections`（注入接缝）、`config.py`（`WikiConfig`） |
| 注入层瘦身 | `memory/consolidate.py::_propose_injection_slimming`、`memory/tiers.py`（`L3_INDEX_MARGIN`、`demote_l3`）、`memory/retrieval.py`（回捞通道） |

一个回合的总体数据流（五组机制的位置）：

```
用户 query
  │
  ├─ on_reply ─────────── 后台检索任务（reasoning hint 面）
  ├─ on_system_prompt ─── 基因组片段 + L3（l3_eligible 过滤）
  │                        + U4 顶部小索引（≤L3_INDEX_MARGIN 字节）
  │                        + 命中肌肉反射            ← 注入瘦身预算口径
  │      │ 注入 id → engine.note_injected（累积）     ← U2 信号接缝
  │      └─ 注入 id → settle_predictions("hit")       ← U1 结算
  ├─ on_reasoning ──────── 近期短期 + 混合检索（含链接展开）→ settle("hit")
  │                        回捞通道：demote_l3 的条目经 link 注入
  ▼
模型/工具执行 → RSI 轨迹（used_memory_ids 落库）→ trace_id 回传提取器
  │
  └─ 回合结束（on_reply finally）
       ├─ 提取器写短期记忆（salience 标注 + 默认标注兜底）+ L0 证据
       └─ 每 every_n_turns 轮 → Consolidator.review_session（dreaming）
             ├─ 晋升（settle "promoted"）
             ├─ SOP 蒸馏（trace_ids 锚点，防重复 INSERT）
             ├─ 插槽1 轨迹校验（U2：负反馈隔离/验证授予/驻留/漂移/矛盾）
             ├─ 插槽2 去重 / 插槽3 断链修复 / 插槽4 注入瘦身
             └─ maintenance()：TTL 清理（过期先遥测 salience_expired）
                           + rebuild_wiki（U4 全量重建）
```

---

## 1. 显著性与「记忆即预测」（U1）

### 1.1 组件语义

Anthropic 的两个组件在此合并为一条闭环：

- **显著性**：写入侧回答「什么值得记」——提取器对每条候选做种类
  （strategy / plan / tested_rule / env_fact / state / preference）、
  易变性（volatile）、预测使用场景与价值估计的标注；
- **记忆即预测**：每条写入都是一个可检验的预测（「这条记忆将来会被
  这样使用」）。之后只在**真实使用点**回填结局
  （hit / promoted / expired / ignored），用结算数据检验记忆系统的
  写入质量，而不是只看写入量。

### 1.2 数据模型

- `SalienceTag`（`storage/models.py:67`）：`kind` / `volatile` /
  `rationale` / `predicted_use` / `predicted_value(0..1)` / `settled` /
  `settled_outcome`。结局字面量 `SalienceOutcome` 见
  `storage/models.py:44`。
- 存放于 `MemoryRecord.source["salience"]`（JSON 列，**零 schema 变更**）；
  读取侧 `tag_of` / `coerce_tag`（`memory/salience.py:54`、
  `memory/salience.py:67`）对脏数据（非 dict、字段类型错、字面量越界）
  一律返回 `None`，绝不抛异常。

### 1.3 写入路径（预测的做出）

```
回合增量 → TurnExtractor.extract_turn（memory/extract.py:102）
  ├─ LLM 提取：系统提示硬约束 salience 字段
  │   （memory/extract.py:21-38：predicted_use 描述未来使用场景、
  │    predicted_value 为价值估计，"预计不上的给低分或不记"；
  │    每轮可重读的临时状态要求 volatile=true 或不记）
  ├─ 规则兜底 _rule_fallback（memory/extract.py:188）：
  │   产物默认 kind=state / volatile=True / predicted_value≤0.35
  ├─ 脏 salience 原样透传（memory/extract.py:55-65），读取侧容忍
  └─ ShortTermStore.record（memory/shortterm.py:37）
      └─ source 无 salience 时补默认 SalienceTag()
         （memory/shortterm.py:56-58）→ 每条短期记忆必然带标注
```

同轮的对话增量另写一条 L0 原始证据（`memory/extract.py:141-155`，
`add_raw_evidence`），只作溯源、不参与注入，生命周期由固化管线的
L0 回收插槽闭环（见 §2.3）。

### 1.4 结算路径（预测的检验）

`settle_predictions`（`memory/salience.py:107`）的契约：

- **只在真实使用点调用**。挂点共五处：
  1. 系统提示注入面 `system_prompt_block_with_ids`
     （`memory/tiers.py:156`，settle 在 `memory/tiers.py:175`）；
  2. 代理紧凑注入面 `compact_context_block`
     （`memory/tiers.py:178`，settle 在 `memory/tiers.py:194`）；
  3. reasoning 注入面 `reasoning_hint_block`
     （`memory/tiers.py:197`，settle 在 `memory/tiers.py:239`）；
  4. agent 主动检索 `search_for_tool`（`memory/tiers.py:242`，
     settle 在 `memory/tiers.py:255-257`）；
  5. 晋升 `ShortTermStore.promote` → `settled_outcome="promoted"`
     （`memory/shortterm.py:143-147`）。
- **内部检索一律不回填**：建链检索（`memory/shortterm.py:72-79`）、
  回放/合成轨迹走 `Retriever.search`，该方法本身不结算——结算只在
  上列挂点显式发生（`memory/salience.py:5-8` 的模块契约）。
- **幂等**：`settled=False` 才写，结算后不再覆盖
  （`memory/salience.py:130-133`）。
- **无标注不结算**：`tag_of` 为 `None` 的记录跳过，不凭空创建
  settled 标注污染预测面数据（`memory/salience.py:130-131`）。
- **错误隔离**：整体 try/except，异常仅 `logger.warning`
  （`memory/salience.py:134-135`）。三个注入面挂点的调用方外层都是
  「失败即整块置空」（`memory_middleware._retrieve_hint` /
  `on_system_prompt` / `http_proxy._build_memory_prefix`），settle
  抛异常会吞掉当轮全部记忆注入，因此绝不向上抛。

### 1.5 过期结局与固化优先级

- **过期不写库**：过期短期记忆行将被删、且写 predictions 表会污染
  世界模型 MAE，结局改走遥测——`maintenance()` 先取
  `db.expired_short_ids`（`storage/database.py:394-404`，只 SELECT
  不删）、读标注、emit `salience_expired` 事件（CAT_MEMORY，
  `memory/tiers.py:572-594`），再真正删除（`memory/tiers.py:529-550`）。
- **固化优先级** `dream_priority`（`memory/salience.py:95`）：
  种类权重（strategy=1.0 … state=0.25，`memory/salience.py:43-51`）
  × `predicted_value`，volatile 再乘 0.3。供瘦身插槽排序键
  （§5.2）与过期遥测参考。

### 1.6 接缝文件

| 接缝 | 位置 | 作用 |
|---|---|---|
| 提取提示约束 | `memory/extract.py:21-38` | LLM 必须输出 salience 标注 |
| 默认标注兜底 | `memory/shortterm.py:56-58` | 无标注补 `SalienceTag()` |
| 读写容忍 | `memory/salience.py:54-92` | 脏数据 → None，不抛 |
| 结算挂点 ×5 | `memory/tiers.py:175,194,239,255`、`memory/shortterm.py:145-147` | 真实使用点回填 |
| 过期遥测 | `memory/tiers.py:529-594` | `salience_expired` 事件 |
| 优先级 | `memory/salience.py:95-104` | 固化排序得分 |

---

## 2. Dreaming 固化管线

### 2.1 组件语义与触发

Anthropic 的 **dreaming**：agent 在空闲/回合边界做离线巩固，把会话
中的临时经验固化为长期结构。本实现即 `Consolidator.review_session`
（`memory/consolidate.py:49`），**周期触发、以后台任务执行、不阻塞
回复**，两条接入链路都能到达：

1. **中间件模式**：`OpsMemoryMiddleware._turn_postwork`
   （`connector/memory_middleware.py:174-180`）——每会话计数达到
   `review.every_n_turns`（默认 10，`config.py:50`）时，取近 60 条
   轨迹执行审查；
2. **HTTP 代理模式**：`http_proxy._save_trace`
   （`connector/http_proxy.py:281-292`）——代理侧回合计数每达
   `every_n_turns` 条轨迹，`asyncio.create_task` 后台审查（纯代理
   部署可达同等防御）。

审查以**轨迹为 ground truth**：判定素材是 `TraceRecord`（结局评分、
用户反馈、`used_memory_ids`），不是记忆自身的内容自证。

### 2.2 插槽结构与预算

`review_session` 的插槽与预算口径（`memory/consolidate.py:56-72`，
与 `config.ReviewConfig` docstring 同步）：

| 插槽 | 数量上限 | 内容 |
|---|---|---|
| 晋升提议 | `out[:5]`，不计数 | 短期 → 长期 L1 |
| SOP 蒸馏 | 单条（含查重），不计数 | 重复工具序列 → 肌肉记忆 |
| 维护+污染（合计） | `maintenance_budget`=12 封顶 | 插槽 1-4 + 既有维护 + L0 回收 |

所有提议走 `TierManager.propose` 的分级确认
（`memory/tiers.py:269`）：高置信（≥`auto_accept_confidence`=0.85）
且非安全敏感自动执行，其余入 `pending_reviews` 等人工确认。

### 2.3 各插槽语义

1. **晋升**（`memory/consolidate.py:112-136`）：会话内
   `confidence≥0.7 且 kind∈(fact/incident/decision)` 或
   `hit_count≥2` 的短期条目 → `promote_longterm` 提议；执行侧
   `ShortTermStore.promote`（`memory/shortterm.py:117`）翻转 tier、
   保留 `promotes` 溯源链并结算 `settled_outcome="promoted"`。
2. **SOP 蒸馏**（`memory/consolidate.py:141-199`）：同会话成功工具
   序列重复 ≥2 次（`_SOP_MIN_REPEAT`，`memory/consolidate.py:36`）
   → `create_muscle` 提议。**提案前查重**：同 title 的 active SOP
   已存在则跳过；执行侧 `create_muscle` 另有幂等守卫
   （`memory/tiers.py:330-338`）——修复无界增长点（每 N 轮重复
   INSERT 同一 SOP）。提议携带 `trace_ids`（蒸馏来源轨迹，
   `memory/consolidate.py:193-194`），写入
   `muscle.source.trace_ids`（`memory/tiers.py:343-345`）作为
   U2 污染防御的 SOP↔轨迹关联锚点（负反馈可回溯隔离）。
3. **插槽 1 · 轨迹校验（U2 污染防御）**：见 §3.4。
4. **插槽 2 · 去重**（`memory/consolidate.py:378-431`）：L1 层标题
   精确重复 + 内容 difflib 相似（阈值 0.85；long 层超
   `long_soft_cap`=500 时放宽 0.75，软信号促合并而非写入墙）→
   `merge_memories`（置信 0.8 走人工）。内容相似扫描为 O(n²)，
   限 `_DEDUP_SCAN_LIMIT`=300 条防大库膨胀（`memory/consolidate.py:38`）。
   merge 执行侧在归档前写 `source["merged_into"]` 并**即时重定向**
   指向 duplicate 的 related 链接（`memory/longterm.py:97-155`）。
5. **插槽 3 · 断链修复**（`memory/consolidate.py:436-490`）：扫描
   long+short 的链接，断链判定 `guards.broken_link_reason`
   （`memory/guards.py:145`：target 缺失 / status 失效 / 短期已过期）
   → `repair_links` 提议；执行侧逐链**重验**后重定向（target 带
   `merged_into` → primary）或删链，天然幂等
   （`memory/tiers.py:473-509`）。扫描依赖 `list_memories` 的 tier
   逗号拆分（`storage/database.py:334-338` 本轮补齐——此前传
   `tier="long,short"` 得空集、修复静默 no-op）。`stats()` 同步输出
   `broken_links` 计数（`storage/database.py:842-862`）。
6. **插槽 4 · 注入瘦身**：见 §5。
7. **既有维护**（`memory/consolidate.py:555-571`）：肌肉记忆
   `use_count≥10 且 hit_count==0` → 降级为长期 L2 场景知识；
8. **L0 回收**（`memory/consolidate.py:573-605`）：未被任何
   related/provenance 链接引用且总量超 `long_soft_cap/2` 的 L0_raw →
   `archive_memory`（置信 0.6 人工确认）——与 §1.3 的 L0 写入路径
   闭环。

long 层超 `long_soft_cap` 时去重/瘦身插槽提到维护队列队首
（`memory/consolidate.py:85-93`）。

审查末尾调用 `maintenance()`（`memory/tiers.py:529`）：TTL 清理
（先过期遥测再删）+ **wiki 全量重建**（§4.4）——两条链路都触发
review，故 `INDEX.md` 首建最迟第 `every_n_turns` 轮。

### 2.4 接缝文件

| 接缝 | 位置 | 作用 |
|---|---|---|
| 中间件触发 | `connector/memory_middleware.py:174-180` | 每 N 轮审查 |
| 代理触发 | `connector/http_proxy.py:281-292` | 每 N 条轨迹审查 |
| 插槽编排/预算 | `memory/consolidate.py:49-107` | 五插槽 + 12 条预算 |
| 分级确认 | `memory/tiers.py:269-301` | 自动/人工分流 |
| 动作执行 | `memory/tiers.py:315-511`（`_dispatch`） | 十种 review action |
| 参数 | `config.py:47-72`（`ReviewConfig`） | 预算/上限/软上限 |

---

## 3. 污染防御（U2）

### 3.1 组件语义

Anthropic 的**污染防御**：错误记忆一旦写入会持续影响后续所有回合，
杀伤面在**注入层**，因此防御围绕「写入有预测，注入有验证」展开——
验证态与置信度解耦，高置信不等于已验证。

### 3.2 验证分级与注入资格

- `VerificationState`（`storage/models.py:96`）：`state ∈
  {unverified, trace_verified, refuted}` + `evidence`（轨迹 id /
  llm_recheck id / 原因）+ `checked_at`，存
  `source["verification"]`（零 schema 变更）；读取侧
  `verification_of` 对脏数据返回默认 unverified
  （`memory/guards.py:73-85`）。
- `is_injectable`（`memory/guards.py:125`）：status ∈
  active/promoted **且** 未被证伪——`refuted` 的记忆即使 status 仍为
  active 也不得注入。
- `l3_eligible`（`memory/guards.py:133`）：L3 核心档案准入**不依赖
  confidence**，必须 trace_verified 或用户手写（`origin ∈
  {memory_note, user}`）。用户手写 L3 的生产创建路径是
  `memory_note` 工具的 `layer=L3_core` 参数（`memory/tools.py:187-195`，
  payload 带 `source.origin="memory_note"`）。

### 3.3 信号接缝：注入即使用

验证判定需要「哪些记忆被注入进了哪条轨迹」：

- **中间件主路径**：`on_system_prompt` 把系统提示注入面
  （L3 / 索引入选 / 命中反射）的 id 经 `engine.note_injected` 计入
  （`connector/memory_middleware.py:216-243`）；`on_reasoning` 对
  reasoning hint 面做同样的事（`connector/memory_middleware.py:187-211`）。
  `note_injected` 为**多注入面累积**（保序去重，
  `rsi/engine.py:105-116`），RSI 中间件在轨迹结算时
  `pop_injected` 取走写入 `used_memory_ids`。
- **代理路径**：`_build_memory_prefix` 返回
  `(prefix, 注入 id 列表)`（`connector/http_proxy.py:70-91`），经
  `_record_trace → _save_trace → recorder.note_used_memories` 落进
  轨迹（`connector/http_proxy.py:260-263`）；流式完成按「有答复」计
  （否则流式部署 score 恒 0.5，使用型验证永远不可达，
  `connector/http_proxy.py:221-236`）。
- **trace_id 回传**：RSI 中间件轨迹结算后
  `engine.note_trace_finished` 上报 trace id，记忆中间件回合结束时
  `last_trace_id` 取走、补传提取器写 `source.trace_id`
  （`rsi/engine.py:122-137`）——LLM 复核路径（§3.4a）依赖它定位
  创建轨迹；单挂 OpsMemory 时为 None，安全降级。

### 3.4 审查判定序（`_verify_against_traces`，`memory/consolidate.py:204-373`）

对注入面（L3 档案 + active SOP）逐条、按固定顺序判定：

1. **负反馈先于一切 score 判定**：关联轨迹（L3 被注入
   `used_memory_ids` / SOP 蒸馏来源 `trace_ids`）收到 👎
   （`feedback==-1`）即提议 `quarantine_memory`（置信 0.9，默认阈值
   下自动执行；用户手写 origin 强制人工），本轮不再做验证判定
   （`memory/consolidate.py:249-273`）；
2. **trace_verified 授予**（创建轨迹的回合级 score 不单独授予，
   `memory/consolidate.py:274-305`）：
   - (a) backend_model 对照**创建轨迹**的 query/digest 复核通过
     （`guards.llm_recheck`，`memory/guards.py:296`；失败按未确认
     处理）；
   - (b) **使用型**：`id ∈ t.used_memory_ids` 且 `t.id ≠
     source.trace_id`（排除创建回合自证）且（`feedback==1` 或
     `score≥0.7`）；
3. **漂移再验证**：trace_verified 条目连续
   `verify_refresh_reviews`（默认 10，`config.py:58`）轮审查无新
   命中（hit_count 未增加）→ `mark_stale` 重置 unverified（保留
   evidence 历史），要求重新挣得验证（`memory/consolidate.py:307-323`）；
4. **L3 驻留上限**：unverified 且 `hit_count==0` 的 L3 每轮
   `l3_reviews`+1，有命中清零；达 `l3_unverified_max_reviews`（默认
   3，`config.py:55`）提议 `demote_l3` 降级退出每轮注入面
   （`memory/consolidate.py:324-349`）；
5. **矛盾检测**：L3 面内标题相似（difflib ≥0.8）且关键事实
   （「属性：值/= /是」正则，`memory/guards.py:55-58`）互斥的记忆对
   → backend_model 存在时 LLM 复核（拿不准不报）→ `flag_contradiction`
   提议（`memory/guards.py:232-263`、`memory/consolidate.py:352-372`）。

### 3.5 提议与执行

- **手写保护**：`quarantine_memory` 与 `demote_l3` 的目标是用户手写
  （`origin ∈ user/memory_note`）时**强制人工确认**——手写核心档案
  只有主人能删/降级（`memory/tiers.py:269-301`）；其余污染防御提议
  置信 0.9 > 0.85 自动执行，目标不静默降级（注入累积 hit_count
  自证存留）。
- **quarantine 执行**（`memory/tiers.py:410-430`）：写 `refuted`
  验证态 + 归档退出一切注入池（muscle 同步清理 SKILL.md 镜像），
  幂等。
- **flag_contradiction 执行**（`memory/tiers.py:432-441` →
  `guards.flag_contradiction_pair`，`memory/guards.py:266-289`）：
  两侧互插 `contradiction_flags`（同一对只插一次）；已
  trace_verified 的一侧降级 unverified 待再验证——矛盾证据动摇了
  「已被成功使用」的信心。

### 3.6 检索双保险

`Retriever.search` 的直接命中（`memory/retrieval.py:148`）与链接
展开（`memory/retrieval.py:179-183`）都过 `is_injectable`；
`core_profile` 超取 30 条后按 `l3_eligible` 过滤、截断回 10 条并
计 hit_count（`memory/retrieval.py:203-217`）——手写 L3 靠注入命中
自证存留，否则驻留计数会反复提议降级。

### 3.7 接缝文件

| 接缝 | 位置 | 作用 |
|---|---|---|
| 验证态模型/读写 | `storage/models.py:96-108`、`memory/guards.py:73-122` | 分级 + 脏数据容忍 |
| 注入资格/准入 | `memory/guards.py:125-142` | refuted 拦截、L3 不看 confidence |
| 信号接缝 | `connector/memory_middleware.py:187-243`、`connector/http_proxy.py:70-91,260-263`、`rsi/engine.py:105-137` | 注入 id → 轨迹 |
| 判定序 | `memory/consolidate.py:204-373` | 负反馈/授予/漂移/驻留/矛盾 |
| 分级确认 + 手写保护 | `memory/tiers.py:269-301` | 自动 vs 人工 |
| 检索双保险 | `memory/retrieval.py:148,179-183,203-217` | 注入层最后防线 |

---

## 4. Wiki 索引（U4）

### 4.1 组件语义与形态

Anthropic 的**记忆索引**：上下文顶部注入一份简洁的「知道什么、去哪
找」索引，细节按需加载，而非把长期记忆全文塞进上下文。本实现把长期
记忆镜像为 wiki：

```
<data_dir>/mirror/wiki/
├── INDEX.md            # 顶部小索引（与注入段同一渲染器）
└── <主题>-<锚点id前8>.md  # 主题页：锚点摘要 + 成员条目 + 交叉引用
```

配置 `WikiConfig`（`config.py:124-137`）：`enabled`（默认 True）、
`index_max_lines`=60、`index_max_bytes`=8192。

### 4.2 归簇（`cluster_topics`，`storage/mirror.py:120-154`）

- `L2_scenario` 为主题**锚点**；`L1_atom` 若有指向锚点的 related
  链接则归入该主题；
- 未归簇的孤儿 L1 进「其他」页（`_MISC_TOPIC`）；
- 输出按主题名排序（稳定键——索引截断确定性的来源）。

主题页文件名 = `_safe_name(topic)`（48 字符截断）+ 锚点 id 前 8 位
后缀防同名互覆；「其他」页固定 `misc` 后缀
（`storage/mirror.py:156-165`）。

### 4.3 索引渲染与预算（`_index_lines` / `render_index_block_with_ids`）

`storage/mirror.py:167-229`：每个主题渲染一行
`- [主题](./页面)（N 条）：代表性标题`；**行数与 UTF-8 字节双上限**
（头行计入，尾行「更多用 memory_search 检索」提示不计入、永远保留）；
超限按主题名稳定键做前缀截断。`render_index_block_with_ids` 同时
返回入选记录 id——供注入面回填轨迹（U2 信号）与 settle（U1 结算）。
同一渲染器服务三条消费路径：中间件系统提示段、代理注入段、
`INDEX.md` 写盘（`storage/mirror.py:203-213`）。

### 4.4 重建（`rebuild_wiki`）

`MirrorStore.rebuild_wiki`（`storage/mirror.py:231-316`）：先清空
`mirror/wiki`（幂等、可随时删重建）→ 逐主题写页（成员条目摘要 +
页间相对链接交叉引用 `↔`）→ 渲染 `INDEX.md`。仅收调用方过滤后的
active/promoted 条目；断链有效性由插槽 3 repair 后重建保证。
触发点：`TierManager.rebuild_wiki`（`memory/tiers.py:552-570`）在
每次 `maintenance()` 末尾执行（失败只告警不阻塞维护，
`memory/tiers.py:546-549`）。

### 4.5 注入接缝（不依赖 rebuild）

`TierManager._core_context_sections`（`memory/tiers.py:108-127`）：
`wiki.enabled` 时从 DB 直渲染 L1/L2 索引段，字节上限即
`L3_INDEX_MARGIN`=400（`memory/tiers.py:47-50`），入选 id 并入
`used_ids`。**注入不依赖 rebuild**——`INDEX.md` 首建前注入侧仍可
从 DB 渲染（写穿副本语义）；rebuild 只维护镜像文件。索引段渲染在
L3 段之后、肌肉反射之前，与两注入口共用
`system_prompt_block_with_ids` / `compact_context_block`。

### 4.6 接缝文件

| 接缝 | 位置 | 作用 |
|---|---|---|
| 配置 | `config.py:124-137` | 开关 + 行数/字节上限 |
| 归簇/文件名防撞 | `storage/mirror.py:120-165` | L2 锚点 + related 归簇 |
| 索引渲染（三路共用） | `storage/mirror.py:167-229` | 双预算 + id 回传 |
| 全量重建 | `storage/mirror.py:231-316`、`memory/tiers.py:529-570` | 周期幂等重建 |
| 注入接缝 | `memory/tiers.py:108-127` | DB 直渲染 ≤400 字节 |

---

## 5. 注入层瘦身

### 5.1 合计预算口径

系统提示的记忆段受**合计字符预算**约束：

```
L3 段（display(200) 渲染）+ U4 索引段 ≤ l3_char_budget = 1200
                                          （config.py:68-72）
实现预留 L3_INDEX_MARGIN = 400 索引余量（memory/tiers.py:47-50）
  ⇒ 瘦身插槽对 L3 段生效的预算 = 1200 − 400 = 800
     （memory/consolidate.py:513）
```

`L3_INDEX_MARGIN` 是瘦身插槽与注入路径**共用**的常量，防止两处口径
漂移。短期/检索/反射段各有独立预算（`short_inject_recent` /
`char_budget` / `reflex_budget`，均在基因组 `RetrievalParams` 安全线
内），不在本口径内。

### 5.2 瘦身插槽（`_propose_injection_slimming`，`memory/consolidate.py:495-550`）

L3 全体 `l3_eligible` 条目渲染字符合计超预算时，按
`(trace_verified 降序, hit_count 降序, dream_priority 降序)` 排序
保留头部（首个必留，其后按序装满即止），其余提议 `demote_l3`
（置信 0.9；手写 origin 强制人工）。排序键的三个分量分别来自 U2 验证
态、注入命中计数、U1 预测面得分——三个机制在预算竞争上汇合。

### 5.3 降级执行与回捞通道

`demote_l3` 执行（`memory/tiers.py:443-471`）：

- `layer` 从 `L3_core` 翻转为 `L1_atom`，**tier/status 不动**——
  RSI 回放回归检查（`rsi/replay.py`）只认 tier+status，保证回放
  兼容；
- 镜像先删旧层文件再写新层（文件名含层前缀，
  `storage/mirror.py:318-330` 的调用约定）；
- 剩余 L3 条目追加 `related` 链接指向被降级条目——降级知识仍可从
  L3 注入面经检索层 link 通道展开回捞（`memory/retrieval.py:169-197`，
  `via=link`、得分 = 父命中分 × 0.8）。

因此瘦身不是删除而是**降密度**：每轮必注入面变小，条目退到
按需检索层。中间件与代理共用同一渲染路径
（`system_prompt_block_with_ids` / `compact_context_block` →
`_core_context_sections`），两条接入方式同等生效。

### 5.4 接缝文件

| 接缝 | 位置 | 作用 |
|---|---|---|
| 预算常量（共用） | `memory/tiers.py:47-50`、`config.py:68-72` | 合计口径锚点 |
| 瘦身插槽 | `memory/consolidate.py:495-550` | 排序保留头部 |
| 降级执行 | `memory/tiers.py:443-471` | 层翻转 + related 回捞 |
| 回捞通道 | `memory/retrieval.py:169-197` | link 展开注入 |
| 注入渲染 | `memory/tiers.py:78-140` | 中间件/代理共用 |

---

## 6. 容错与幂等契约（横切）

- **脏数据容忍**：`tag_of` / `coerce_tag` / `verification_of` 对
  非 dict、字段类型错、字面量越界一律返默认/None，绝不抛异常
  （`memory/salience.py:54-70`、`memory/guards.py:73-85`）；
- **错误隔离**：settle 异常只告警不阻断注入
  （`memory/salience.py:134-135`）；wiki 重建失败不阻塞维护
  （`memory/tiers.py:546-549`）；过期遥测失败不阻塞清理
  （`memory/tiers.py:592-594`）；提议单条失败跳过不中断整轮审查
  （`memory/consolidate.py:97-106`）；
- **幂等**：settle 结算后不覆盖；SOP 蒸馏提案前 + 执行前双查重；
  merge 重复执行不追加；repair_links 执行侧逐链重验；demote/quarantine
  对已处理目标直接返回；wiki 重建先清空。

## 7. 测试覆盖

全套离线测试（`rsi_mem/pytest.ini`，`asyncio_mode=auto`）：

```bash
cd rsi_mem && python -m pytest tests -q    # 94 passed（本轮实测）
```

本轮新增四个测试文件，逐机制断言：

- `tests/salience_test.py`——标注往返与脏数据容忍、**内部使用不结算
  只有注入结算**、无标注不结算、工具检索结算、settle 抛异常不阻断
  注入、规则兜底按内容去重；
- `tests/guards_test.py`——l3_eligible 判定矩阵（不看 confidence）、
  **负反馈先于 score**、创建轨迹不授予/使用轨迹授予 trace_verified、
  默认阈值自动执行 + 手写强制人工、SOP 污染隔离（含 SKILL.md 镜像
  清理）、L3 驻留计数、漂移重置、矛盾正/负例；
- `tests/consolidate_test.py`——merge 写 `merged_into` 并重定向、
  断链三边界（缺失/过期/重定向）、内容相似去重（含软上限放宽阈值）、
  **L3 合计预算瘦身**（降级后总量 ≤ 预算、tier/status 不变、镜像换层、
  降级条目可经检索回捞）、SOP 蒸馏查重、L0 回收、维护预算封顶、
  并发双审查幂等、demote 后系统提示变短且注入 id 可达；
- `tests/wiki_test.py`——重建幂等并清空陈旧页、文件名防撞、L2 锚点
  页 + 孤儿「其他」页 + 交叉引用、索引截断边界（尾行提示永在）、
  百条记忆的行数/字节上限与 L3+索引合计预算、中间件提示含索引段、
  归档后 wiki 与 INDEX 不再含该条目。
