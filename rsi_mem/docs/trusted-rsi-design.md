# 可信自改进（Trusted RSI）设计 —— 执行反馈级验证 / 失败模式防御三件套 / 治理断路器与 L1-L5 自评

本文档描述 `rsi_mem` 本轮「可信自改进」升级（工作区改动，基线 commit
`d248347`）引入的三组机制，编号沿用代码内注释的施工单元 **U2 / U3 / U4**：

- **U2 验证层级升级到执行反馈级**：黄金场景回归门（执行级验证门）+
  锁定片段全字段锁 + 证据层级感知选择（E2 双门）——此前 validator 保证
  *结构合法*、pair gate 复验*组装态*，但都不*执行*；本轮补上部署前在
  隔离栈上真实注入与检索的硬断言，并让仅 E1（LLM 评审）证据支持的
  质量优势不得越过 E2/E0 门；
- **U3 失败模式防御三件套**：自证回路防御 / 多样性坍缩防御 / 帧锁定
  审计——一个能自主改自己的系统，最危险的不是改错，而是**在自产数据
  上自我感觉良好**；
- **U4 治理器与 L1-L5 自评**：部署断路器（速率/冷却/连续上限/质量
  回归自动回滚）刻意站在 RSI 可改对象之外 + 治理审计 + 对齐公开 RSI
  能级榜单判据的诚实分级快照。

每节给出**接缝文件**与**判定逻辑**。公式与编号（E0-E3、Eq.7、λ₀）
沿用 `rsi/evidence.py` 与 [docs/rsi-engine-v3-design.md](rsi-engine-v3-design.md)
的口径。

| 机制 | 核心接缝文件 |
|---|---|
| 黄金场景回归门（执行级验证门） | `rsi/execution_gate.py`、`rsi/golden.py`（冻结基准）、`rsi/engine.py`（dream 部署段接线）、`config.py`（`GovernanceConfig.max_inject_chars`） |
| 锁定片段全字段锁 | `rsi/validator.py`（完整性段 `_lock_fields`） |
| 证据层级感知选择（E2 双门） | `rsi/replay.py`（`quality_e2`）、`rsi/engine.py`（`_mix_e2` / `gate_e2`） |
| 自证回路防御 | `rsi/engine.py`（`_backfill_e3`）、`rsi/predictor.py`（加载层过滤）、`rsi/trace.py`（source 缺省）、`cli.py`（自参考/评审哨兵） |
| 多样性坍缩防御 | `rsi/engine.py`（探索注入 + `_explore_retrieval_changes`）、`cli.py`（算子集中度/内容重复哨兵） |
| 帧锁定审计 | `rsi/engine.py`（`_render_hash`）、`cli.py`（外部审计包） |
| 治理断路器 | `governor.py`（独立模块）、`config.py`（`GovernanceConfig` frozen）、`rsi/engine.py`（构造期快照 + 部署段 + `rollback` 内检）、`context.py`（构造顺序） |
| L1-L5 分级自评 | `rsi/level.py`、`rsi/engine.py`（报告接线）、`cli.py`（doctor 输出） |

一轮 dream 里四道门与治理器的**顺序**（本设计的全部机制都在这条
管线上，缺一不可越）：

```
候选集（v0 恒在 + 插件计划 + incumbent 恢复 + explore 扰动）
  │  逐候选回放评估（E2）→ objective + quality_e2 + pareto_auc   ← §1.4
  │  双门 G ∩ G′（E1 口径门 + E2 口径门，v0 两门恒属）           ← §1.4
  │  门内最平衡 H† = argmax min(q′, cost, safety) → winner
  ▼
① validator 三段校验（含全字段锁，候选生成期 + pair gate 两处）  ← §1.3
② 执行门：隔离栈真实注入/检索/预算三类硬断言（O(1)，只验胜者）   ← §1.2
③ Governor.check_deploy：degraded/暂停/钉住/速率/冷却/连续上限   ← §3.2
  │  全过 → 落库（scores 打 execution_gate=True + render_hash），
  │         record_deploy 记账 + 部署即记 E3 pending 预测
  ▼
dream 开始时的两道前置闸（下轮）：
⓪ E3 回填（production realized）+ 治理质量规则（回归→自动回滚+暂停）← §2.1 §3.2
```

---

## 1. 验证层级升级到执行反馈级（U2）

### 1.1 层级模型：从「校验过」到「执行过」

项目原有的证据分级（`rsi/evidence.py`）标注的是**信号**的保真层级
（E0 规则 / E1 模型预测 / E2 模拟回放 / E3 真实测量）；本轮把同一
思想落到**验证手段**上，形成四级：

| 层级 | 手段 | 能保证 | 不能保证 |
|---|---|---|---|
| 结构合法 | validator 结构段+冒烟段（`rsi/validator.py:56-93`） | schema/上限/SOP 引用存在/可序列化 | 渲染面行为 |
| 组装态 | pair gate（部署前 `validate(winner, current)` 终校验，`rsi/engine.py:833-839`） | 锁定片段保留、非空演化 | 执行效果 |
| 执行反馈级 | **执行门**（隔离栈真实注入+检索+预算，`rsi/execution_gate.py:53`） | 锁定片段真渲染、关键词真召回、预算真守住 | 真实 query 召回质量 |
| 真实测量（E3） | 部署后真实轨迹回填 + 治理质量规则 | 上线后质量不回归（回归即回滚） | ——（事后信号） |

升级动机（`rsi/execution_gate.py:3-9`）：参数收紧到杀死召回、锁定
片段被禁用退出渲染面这类「校验过、执行退化」的候选，结构段与组装态
都拦不住——只有真的跑一次才知道。

### 1.2 黄金场景回归门（执行级验证门）

**接缝文件**：`rsi/execution_gate.py`（门本体）、`rsi/golden.py`
（冻结基准）、`rsi/engine.py:855-882`（部署段接线）、
`config.py:150-153`（预算参数）。

**黄金数据**（`rsi/golden.py:13-29`）：5 个运维场景
（磁盘 / redis / crashloop / 证书 / 连接池，每场景一条 query + 一个
关键词探针 kw）+ 每场景一条对应的长期知识，从 `bench/rounds.py` 的
合成运维场景**原值冻结**而来。独立成 leaf 模块（不 import rsi_mem
任何其他模块）的原因：bench.rounds 顶部 import context，而
context → engine → execution_gate，反向引用会成环；bench 反向引用
本模块保持单一事实源（`rsi/golden.py:4-9`，测试
`execution_gate_test.py:355-362` 断言双源相等）。

**判定逻辑**——在**隔离栈**（`tempfile.TemporaryDirectory` 全新
DB/Mirror/TierManager，不建 RSIMemContext，无环无污染，
`rsi/execution_gate.py:75-79`）上对胜者基因组做三类硬断言，任一
失败整轮否决：

1. **必注入**（`rsi/execution_gate.py:94-108`）：黄金场景 query 渲染
   system prompt block（`tiers.system_prompt_block_with_ids`），父代
   每个 locked 片段的 text 必须**逐字在场**。防 enabled/slot 旁路
   （与 §1.3 全字段锁双保险）；
2. **必召回（kw 探针）**（`rsi/execution_gate.py:118-133`）：探针词
   用场景**关键词**（如「磁盘」）而非整 query，用**胜者的检索参数**
   （top_k/threshold/char_budget）在隔离栈上真实检索，对应黄金知识
   id 必须出现在命中集。实证依据（`rsi/execution_gate.py:14-19`）：
   隔离栈 + 默认参数下，磁盘/证书两场景的**整 query** 0 命中——FTS
   trigram 对『线上磁盘使用率』vs『磁盘处置要点』零 3-gram 交集，
   LIKE 整词兜底同 miss；kw 探针 5/5 命中。探针语义明确声明为
   **「检索参数不杀关键词召回」**，不是「真实 query 召回质量」（后者
   是回放质量评估的事，不归部署门管）；
3. **必守预算（绝对臂）**（`rsi/execution_gate.py:109-116`）：五场景
   渲染 block 的最大字符数 ≤ `governance.max_inject_chars`（默认
   12000）。**不设相对臂**：隔离栈上父代渲染仅 ~160 chars，×1.1 的
   相对阈值低于典型 logic_evolve 片段长度，等价于禁止一切内容增量
   候选（`config.py:151-153`）。

**范围与豁免**：O(1) 只验胜者；v0 获胜（保持现状）分支天然豁免——
门只拦「要上线的新东西」，不拦「什么都不改」（`rsi/execution_gate.py:25-27`，
引擎侧 `rsi/engine.py:858-861` 只在 winner 非 current 时调门）。

**预算防旁路**：`max_inject_chars` 恒传引擎**构造期快照**
（`RSIEngine._max_inject_chars`，`rsi/engine.py:124`；门签名与理由见
`rsi/execution_gate.py:66-72`）。插件可在 `start(ctx)` 里整字段重赋值
`config.governance`（pydantic frozen 只拦属性突变，不拦整字段重赋值），
运行期读活配置即被旁路；构造期取值后恒以快照传参。测试
`execution_gate_test.py:315-348` 同时验证了两个方向：快照路径照拒、
被改宽的活配置路径确实会放行（证明快照是必要防线而非多余参数）。

**拒绝路径与 incumbent 驱逐**：门失败走 pair-gate 拒绝同款路径
（`rsi/engine.py:862-882`）——SOP 候选归档、active 版本不变、报告带
`execution_gate.{passed,failed_checks}`；并把持有被拒 winner 的
incumbent 槽位**驱逐**回 v0（`rsi/engine.py:996-1005`），否则下轮
恢复→重评重赢→再拒的死循环。测试 `execution_gate_test.py:198-258`
断言拒绝后版本不变、SOP 归档、`moead.evicted`、次轮不重入。

### 1.3 锁定片段全字段锁

**接缝文件**：`rsi/validator.py:98-124`（完整性段）。

**判定逻辑**：锁定片段按 `(text, enabled, slot, triggers)` 四元组
**逐字段**比对（`_lock_fields`，`rsi/validator.py:102`）——旧版只比
text，`enabled=False` 或 `slot="reflex"` 可让锁定片段**静默退出渲染
面**（`fragment_text` 只渲染 enabled 且 always 的片段）而校验照过，
等于旁路删除。四算子不触这些字段，合法候选不受影响。测试
`execution_gate_test.py:52-88`：四种篡改全部 `FAIL@integrity` 且
problems 指明字段名；新增普通片段候选照过（防误伤）。

这是「代码常量不在 RSI 可改对象内」原则的一例：校验规则本身不可被
基因组演化（`rsi/validator.py:14-19` 的治理不可演化声明）。

### 1.4 证据层级感知选择（E2 双门）

**接缝文件**：`rsi/replay.py:79-84, 353-362`（`quality_e2` 产出）、
`rsi/engine.py:569-580`（`_mix_e2`）、`rsi/engine.py:790-795`
（`gate_e2` 交集）。

**动机**：回放质量 `quality` 里混着 `0.15×judge_delta` 的 E1 分量
（LLM 评审增量）。一个候选可以只凭评审看好（E1 证据）在 λ₀ 质量门
里压过 v0——但 E1 是**预测**，不是模拟也不是测量。

**判定逻辑**：

1. 回放对每个候选额外产出 E2 视野的质量分量
   （`rsi/replay.py:353-362`）：

   ```
   quality_e2 = clamp(base_quality × coverage + 0.1 × sop_coverage, 0, 1)
   ```

   即把 quality 公式剥掉 `0.15×judge_delta`（E1）项，只留确定性
   E2/E0 分量（结局评分×回归覆盖 + SOP 触发覆盖加成）；
2. 引擎对混合目标同式并行维护 E2 列（`rsi/engine.py:569-580`：
   `quality=(1−α)·quality_e2+α·pareto_auc`），理想点 `ideal_e2`
   与子问题度量 `t′_e2` 都不含 E1 分量（`rsi/engine.py:644-675`）；
3. 部署门从单门升级为双门（`rsi/engine.py:786-795`）：

   ```
   G  = {c : t′(c,λ₀)   ≤ t′(v0,λ₀)  + ε}        （原质量门）
   G′ = {c : t′_e2(c,λ₀) ≤ t′_e2(v0,λ₀) + ε}      （E2 视野门）
   部署集 = G ∩ G′
   ```

   **语义**：仅 E1 证据支持的质量优势不得越过 E2/E0 门。

**非退化与兼容**：v0 在两门恒属（≤ 自身）⇒ 非退化论证不变；judge
缺席时 `judge_delta ≡ 0` ⇒ `t′_e2 ≡ t′`，双门退化为单门，现有测试
与 bench 行为不变（`rsi/engine.py:775-779`）。

---

## 2. 失败模式防御三件套（U3）

一个自主改自己的系统有三类「静默坏掉」的方式：**在自产数据上评价
自己**（自证回路）、**探索面收窄到只会一种改法**（多样性坍缩）、
**渲染面被悄悄改掉而无人知道**（帧漂移）。三件套逐一防御，测试集中在
`tests/failure_modes_test.py`。

### 2.1 自证回路防御

**接缝文件**：`rsi/engine.py:226-279`（`_backfill_e3`）、
`rsi/predictor.py:141-171`（加载层过滤）、`rsi/trace.py`（source 缺省）、
`rsi/improver.py:76-100`（分布监测）、`cli.py:275-339, 415-431`
（doctor 哨兵）。

**判定逻辑**分四层：

1. **source 分类学**（`rsi/engine.py:692-699, 270-275`）：预测回填行
   必须标注证据来源——同轮回放回填 `realized.source="replay"`（E2
   自产证据），部署后真实轨迹回填 `source="production"`（E3 外部
   信号）。TraceRecorder 主路径缺省 `production`（`rsi/trace.py`），
   bench 合成轨迹自带 `source="bench"`，无键直写轨迹计 production
   并入 `untagged` 计数（`rsi/improver.py:76-100`）；
2. **E3 回填链**（`rsi/engine.py:226-279`）：部署时记 pending 预测
   （`genome_version` 锚定，`rsi/engine.py:943-957`）；下一轮 dream
   用**真实外部轨迹**结算：`realized = mean(post-deploy score) −
   mean(pre-deploy window score)`，部署后样本 < `_E3_MIN_POST_SAMPLES`
   （=5，`rsi/engine.py:1241`）保持 pending；
3. **世界模型防自校准**（`rsi/predictor.py:141-157`）：`production`
   行**不进** predictor 在线统计（MAE/均值）——否则部署后真实轨迹
   分数会被当成世界模型的自校准样本，「自己评价自己」。过滤发生在
   **加载层**而非运行期内存态，跨重启/跨引擎实例同样成立（测试
   `failure_modes_test.py:103-131` 用同库新实例验证）；
4. **doctor 哨兵**（`cli.py`）：
   - **预测器自参考**（`cli.py:277-316`）：近 20 条 realized 全部
     来自回放（无 production 行）→ warn「世界模型在评价自己，MAE
     不可信」。带**冷启动豁免**：部署后真实轨迹不足
     `min_traces_after_deploy`、或这些轨迹之后还没跑过 dream（E3
     回填未到期）时不报——那时没有 production 行是预期而非故障
     （`cli.py:279-308` 的 `backfill_due` 判定）；
   - **评审未激活**（`cli.py:317-339`）：近 5 次 dream
     `judge_delta_mean` 恒 0——曾非零后沉默（judge 回归）→ warn；
     从未配置过 judge → info（常态不误报）。旧事件无键静默降级；
   - **外部信号消失**（`cli.py:415-431`）：近期轨迹窗口 production
     占比 ≤ `rsi.external_data_min_ratio`（默认 0，即全自产才报）→
     info，观测先行不设硬门。

### 2.2 多样性坍缩防御

**接缝文件**：`rsi/engine.py:507-548`（探索注入）、
`rsi/engine.py:1245-1279`（`_explore_retrieval_changes`）、
`cli.py:341-375`（哨兵）、`config.py:129-132`
（`explore_after_same_operator`）。

**判定逻辑**：

1. **探索注入**（engine 侧自动防御）：连续同算子部署 ≥
   `explore_after_same_operator`（默认 3，按基因组史头部统计，
   `rsi/engine.py:512-517`）→ 注入一个有界随机 `retrieval_evolve`
   扰动候选（`origin="explore"`）：`random.Random(dream_seq)` 种子
   可复现，参数在 `RetrievalParams` 边界内围绕当前值小步抖动
   （top_k ±2 / char_budget ±400 / threshold ±0.05，至少改 top_k
   一项保证过「空演化」校验），**走全部校验/回放/双门/执行门/治理
   门**，与任何候选同权竞争，v0 保护不变（`rsi/engine.py:507-548`）；
2. **doctor 算子集中度哨兵**（`cli.py:341-354`）：近 10 版（去 seed）
   某算子占比 > 80% → warn「探索面收窄」（提示 engine 会自动注入
   explore 扰动）；
3. **相邻版本内容重复检测**（`cli.py:356-375`）：去元数据 sha256
   相同的相邻版本对 → warn「空演化漂移」。

测试：`failure_modes_test.py:444-493`（扰动纯函数确定性/边界/可区分
+ 3 版连续史触发注入）、`failure_modes_test.py:500-555`（集中度
warn/混合不报）。

### 2.3 帧锁定审计

**接缝文件**：`rsi/engine.py:1282-1286`（`_render_hash`）、
`rsi/engine.py:936-937`（部署写入 scores）、`cli.py:377-413`
（外部审计包）。

**判定逻辑**：部署基因组的 `scores["render_hash"]` =
`sha256(fragment_text())[:16]`——渲染面指纹，retrieval/SOP 类部署不
改片段则指纹稳定，**内容增量（logic_evolve）才移动指纹**。doctor 输出
「外部审计包」发现项（info）：近 6 版 render_hash / operator / 部署
斜率（balanced_min 首尾差）快照 + `render_hash_changed_versions` 列表，
供人定期复核渲染面演化是否符合预期（`cli.py:377-413`）。测试
`failure_modes_test.py:605-644`。

---

## 3. 治理器与 L1-L5 自评（U4）

### 3.1 预算断路器：独立于 RSI 可改对象的设计

**接缝文件**：`governor.py`（独立顶层模块，不在 `rsi/` 包内）、
`config.py:139-166`（`GovernanceConfig`）、`rsi/engine.py:111-127`
（构造期快照）、`context.py:53-61`（构造顺序）。

**威胁模型**：RSI 的可改对象是基因组（GenomeSpec）、插件注册、记忆
库与 meta 表——治理器若从这些地方读参数或存状态，就处在了被治理者
的写权限之内。设计把每个接触面都切掉：

| RSI 可改对象 | 切断方式 | 位置 |
|---|---|---|
| `config.governance` 属性突变 | `GovernanceConfig` frozen，pydantic 拦 `ValidationError` | `config.py:148` |
| `config.governance` **整字段重赋值**（frozen 拦不住的旁路） | Governor 构造期**值拷贝快照**六个参数，此后重赋值不影响实例；执行门预算 `max_inject_chars` 同规则快照 | `governor.py:64-71`、`rsi/engine.py:111-124` |
| 插件在 dream 期改配置 | `RSIMemContext` 构造链（context → engine → Governor）**早于一切 plugin.start**（首个 dream 才懒启动），快照先于任何插件代码 | `rsi/engine.py:114-119`、测试 `governor_test.py:202-209` |
| meta 表（ctx 插件可写面） | 状态持久化在 `<data_dir>/governor.json`（原子写 tmp + `os.replace`），不落 meta | `governor.py:11-12, 113-123`、测试 `governor_test.py:313-333` |
| 基因组携带治理参数 | `GenomeSpec` 无 governance 字段（演化对象与治理对象物理隔离） | 测试 `governor_test.py:324-325` |
| data_dir 内删文件 | 已见过/写过的状态文件消失 → fail-closed（见 §3.2） | `governor.py:95-102` |

**人工干预只有两条路**（`governor.py:13-14`）：手工编辑
`governor.json`，或环境变量 `RSI_MEM_GOVERNOR_PIN`（Governor 构造期
读取，`governor.py:72-73`）。二者都在代码之外，RSI 路径够不到。

### 3.2 断路器判定逻辑

**接缝文件**：`governor.py:128-205`（判决）、`rsi/engine.py:884-911`
（部署段接线）、`rsi/engine.py:281-338`（质量规则）、`config.py:154-166`
（参数）。

**`check_deploy` 四查**（部署前，执行门通过后、落库前调用，
`rsi/engine.py:884-887`；每次判决前重读状态文件）：

1. **degraded**（fail-closed，见下）→ 拒，reason 含 "fail-closed"；
2. **deployment_pause** → 拒（连续上限触顶或质量回归自动置位，人工
   编辑 governor.json 才能清）；
3. **pinned_version** 非空（env pin 优先于状态文件，`governor.py:246-250`）
   → 拒「version pinned at vN」；
4. **速率 + 冷却 + 连续上限**：滚动 1 小时窗口部署数 ≥
   `max_deploys_per_hour`（默认 6）拒；距上次部署 <
   `min_deploy_interval_s`（默认 300s）拒；无回滚隔断的连续部署轮数
   ≥ `consecutive_deploy_limit`（默认 8）→ 拒**并置 deployment_pause**
   （`governor.py:170-180`）。引擎侧连续部署数按基因组史统计
   （`rsi/engine.py:518-522`，遇 seed/rollback 版本断开计数）。

拒绝走 pair-gate 同款路径（SOP 归档 / incumbent 驱逐 / v0 保持），
报告与 CAT_DREAM 事件带 `governor_decision`（`rsi/engine.py:896-911`）。
部署成功则 `record_deploy` 记账（只保留近 2 小时，最新一条永远保留
作冷却锚点，`governor.py:210-222`）。

**质量回归规则**（dream 开始时，`rsi/engine.py:358-360` 调
`_governed_quality_check`）：近 `quality_rollback_window`（默认 20）
条部署后带结局轨迹的均值，较部署前窗口均值下滑 >
`quality_rollback_drop`（默认 0.1），且部署后样本 ≥
`min_traces_after_deploy`（默认 5）→ 判退化（`governor.py:183-205`，
只读参数快照与入参，**不读状态文件**——degraded 不影响本规则）→
engine 自动回滚 + `note_rollback` 记账 + `pause_deployments` 置暂停 +
CAT_ERROR 审计事件（`rsi/engine.py:309-338`）。回滚本身被拒（pin/
复验）则不回滚不暂停，只告警。种子版与 rollback 版豁免本规则
（`rsi/engine.py:292-294`）。

**fail-closed 语义**（`governor.py:16-22, 82-111`）：状态文件解析
失败/非 dict，**或本实例已见过/写过的状态文件消失（被删）** →
degraded，`check_deploy` 一律拒绝，不重置、不 fail-open；degraded 期
不写文件（防毁现场）、记账丢弃（不落不可信账）、计数不可恢复；每次
检查前重读文件，**修复后从修复文件的现状继续**（不是重置为空——
修复文件里写 6 条部署就立即按 6 计数拒绝，测试
`governor_test.py:215-252`）。删除的 fail-closed 保证覆盖**本进程
生命周期**；进程重启后「删除后重启」与「首装」从 data_dir 内部无法
区分（无外部可信存储），如实声明（`governor.py:20-22`）——这是已知
边界，不是被宣称解决的。rollback（恢复动作）不受断路器拦截。

### 3.3 治理审计

**接缝文件**：`governor.py:210-283`（账本与观测面）、`cli.py:457-483`
（doctor）、`rsi/engine.py:1177-1230`（rollback 内检）、
`rsi/engine.py:1093-1151`（事件载荷）、`bench/rounds.py:196-204, 241-247`
（bench 列）。

审计面四条：

1. **账本**：`governor.json` 记 `deploys`（时间戳+版本，近 2h 窗口）、
   `rollbacks`（审计用，不清 pause——暂停只能人工清，`governor.py:224-232`）、
   `pinned_version` / `deployment_pause` / `paused_reason`；
2. **事件**：质量回归自动回滚、pin 拒回滚、复验拒回滚均落
   `CAT_ERROR`（scope=governor，`rsi/engine.py:327-337, 1189-1199,
   1210-1220`）；每次 dream 的 CAT_DREAM end 载荷带
   `governor_decision` 与自评 `level`（`rsi/engine.py:1131-1132`）；
3. **doctor**：governor.refresh() 后按快照输出——degraded → warn
   「fail-closed：部署一律拒绝，修复后自动恢复」；否则 info 汇报
   pin/pause/近 1 小时部署数（`cli.py:457-483`）；从未演化的空库
   （无状态文件且无异常）不报；
4. **回滚内检**（任何调用方同一防线——`rsi_evolve` 工具对插件可达、
   governor 自动回滚、测试直调，`rsi/engine.py:1179-1187`）：
   `pinned_version` 非空 → 拒（None + CAT_ERROR）；回滚目标 prev 经
   `validate(prev, current)` 复验失败（如 incumbent 污染的 dangling
   SOP）→ 拒 + CAT_ERROR，污染版本不得经回滚入库；deployment_pause /
   degraded 不拦（回滚是恢复动作）。

**bench 契约**（`bench/rounds.py:153-163`）：bench 需要宽松治理必须在
**构造期**传入 `GovernanceConfig(max_deploys_per_hour=10**6, ...)`——
治理参数在 ctx 构造时已快照定型，构造后覆盖只能走整字段重赋值旁路，
不支持也不承诺。默认严格治理（6 次/小时 + 300s 冷却）会使 deploys 被
限流，此时 `rows[0]["governor_state"]` 可见原因而非静默 `deploys=0`。
观测列只增不改：rows 带 `execution_gate` / `governor`（+首轮
`governor_state` 快照），summary 增 `execution_gate_passes` /
`governor_denies`（`bench/rounds.py:196-204, 241-247`；契约测试
`governor_test.py:374-414`）。

### 3.4 L1-L5 分级自评：判据与当前结果

**接缝文件**：`rsi/level.py`（collect + assess 纯函数）、
`rsi/engine.py:1047-1050`（dream 报告 `level_assessment` 段）、
`cli.py:485-498`（doctor 输出，有 dream 活动才评）。

**判据**（对齐公开 RSI 能级榜单口径，`assess` 纯函数消费 `collect`
收集的探针，`rsi/level.py:87-104`）：

| 判据 | 探针 | 判定 |
|---|---|---|
| `propose_autonomous` | CAT_DREAM 事件数、历史峰值候选数（事件历史优先，旧事件无键回退最近报告） | ≥1 次 dream 且候选数 >1 |
| `implement_autonomous` | 基因组版本前移 或 事件 payload.deployed | 任一为真 |
| `verify_autonomous` | 最近 dream 报告 replays 数 | ≥1（回放评估真的跑了） |
| `verify_executed` | **双源**：CAT_DREAM end 事件 `execution_gate.passed is True` ∪ 任一基因组 `scores["execution_gate"] is True` | 任一为真 |
| `imp_mutable` | 静态检查：基因组是否覆盖引擎超参 | 当前恒 False（GenomeSpec 不含治理/引擎参数） |
| `domain_generality` | 覆盖问题域数 | 黄金场景全部属运维单域 → 1 |

**级别映射**（诚实输出，不夸大，`rsi/level.py:106-119`）：三项自主
（propose/implement/verify）全真 → **L3**；仅 propose 或 implement →
L2；否则 L1。`imp_mutable=False` 卡在 L4 之外、单域卡在 L5 之外——
二者不虚标，如实落在 `failing` 列表里。

**当前自评结果**（`rsi/level.py:21-24` 声明 + 测试
`level_test.py:13-49` 验证）：跑通部署链路的库，满分快照是
**L3 + failing=[imp_mutable, domain_generality]**，全部证据为 E0
（确定性静态/事件检查），自标 `evidence="E0"`。`verify_executed` 的
双源设计保证**末轮未部署不抹掉历史执行门证据**（历史即证据）；空库
（无 dream）输出 L1、全部自主判据未达（`level_test.py:52-63`）。

---

## 4. 测试覆盖

本轮新增四个测试文件共 **33 个用例**（全部离线），全套
`python -m pytest rsi_mem/tests -q` 为 **161 个用例**（本轮实跑通过，
42.36s）：

| 文件 | 用例 | 覆盖 |
|---|---|---|
| `tests/execution_gate_test.py` | 9 | 全字段锁四路篡改、kw 探针真命中/threshold=1.0 全 miss、锁定片段禁用旁路、绝对预算臂（含正常 seed 照过）、隔离栈无泄漏（db 关闭/临时目录清理/黄金知识不进活库）、引擎拒绝+incumbent 驱逐+次轮不重入、门绿真实部署端到端（报告/事件/基因组 scores 三处打标）、预算快照免疫重赋值（含旁路对照）、黄金数据双源单源 |
| `tests/failure_modes_test.py` | 11 | E3 回填 production/replay 分类与 ±0.15 结算、跨重启加载层过滤、source 分类学（production 缺省/bench 标注/untagged/doctor 混合不报/全自产 info）、judge_delta_mean 载荷与评审哨兵（曾激活沉默 warn）、冷启动豁免（未到期不报/到期正例/回填链路失效模拟 warn）、pareto n_effective 与恒定序列哨兵（空闲不误报）、扰动纯函数确定性、探索注入触发、算子集中度 warn/混合不报、CAT_ERROR/CAT_REVIEW 接线、render_hash 与审计包 |
| `tests/governor_test.py` | 11 | 速率/冷却、连续上限暂停与手工恢复、质量回归自动回滚+后续部署被拦、pin（回滚拒+部署拒+事件）、构造期快照免疫重赋值（第 7 次照拒）、构造先于插件、状态损坏 fail-closed+修复续算、状态删除 fail-closed+首装豁免、frozen/基因组隔离/状态文件位置、回滚复验拒 dangling SOP、bench 契约（宽松治理构造期传入+旧键语义原样+新键只增） |
| `tests/level_test.py` | 2 | 事件历史双源 verify_executed + L3 诚实输出 + 报告挂段；空库 L1 |

测试舞台复用 `tests/rsi_test.py` 的共享助手（`_u2_ctx` /
`_seed_cost_stage` / `_cost_plan` / `_PlansPlugin` / `_token_trace`，
`tests/rsi_test.py:300-372`）：固定计划插件 + 成本收紧舞台保证候选
真的能走到部署步，从而测到的是完整四门管线而非单门直调。

## 5. 已声明的边界

以下边界在代码注释中如实声明、未宣称解决：

- **进程重启后的删除不可区分**（`governor.py:20-22`）：删除状态文件
  的 fail-closed 保证只覆盖本进程生命周期；重启后「删除后重启」与
  「首装」从 data_dir 内部无法区分（无外部可信存储）；
- **kw 探针语义**（`rsi/execution_gate.py:17-19`）：探针保证「检索
  参数不杀关键词召回」，不评「真实 query 召回质量」（后者归回放
  质量评估）；
- **执行门不评内容质量**：三类断言都是回归/预算性质（锁定片段在、
  关键词召回、预算内），候选内容好不好仍由回放双门与 E3 判定；
- **`imp_mutable` / `domain_generality` 是静态判据**（`rsi/level.py:80-82`）：
  当前实现基因组不覆盖引擎超参、黄金场景单域，判据值如实写死，不
  随运行状态浮动；
- **构造后整字段重赋值 `config.governance` 被 pydantic 放行**：这是
  声明的边界而非漏洞——防线是构造期快照（Governor 六参数 +
  `_max_inject_chars`）而非配置不可变（`config.py:144-146`）。
