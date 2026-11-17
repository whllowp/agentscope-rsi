# RSI 引擎 v3 设计 —— MOEA/D 邻域替换与平衡部署 / 不确定性惩罚选择 / ρ 扫描 Pareto 评分 / 前缀盲评 / 债务清偿

本文档描述 `rsi_mem` 本轮 RSI 引擎深化（工作区改动，基线 commit `c32b9de`）
引入的四组选择与评分机制，及其与三篇论文的对应关系：
**MOEA/D 邻域替换与最平衡部署**（Harmony Harness 的多目标分解）、
**不确定性惩罚选择**（世界模型预测不确定性进入子问题度量，Eq.7）、
**ρ 扫描 Pareto 评分**（预算-质量前沿面积 Eq.1）、**prefix-only 防泄漏**
（评审证据的前缀盲评约束，附录 D.2）；最后给出本轮**债务清偿清单**
（上轮记忆组件遗留的 U1 债务①-⑥ 与本轮施工单评审项 P1/P2/P5/P7/P10）。
每节给出**接缝文件**与**数学定义**。

三篇论文在引擎里的分工沿用 `rsi/engine.py:4-19` 的融合声明：

- **Dream-RSI §3**（arXiv 2609.14858）：轨迹录制 → 回放模拟器 → 候选
  argmax 选择（候选恒含 v0 ⇒ 非退化）；
- **Harmony Harness**（arXiv 2610.15150）：多目标 MOEA/D、
  diagnose→hypothesize→operator、三段校验 + 一次修复 + 回滚；
- **RSI-Anything**：策略-世界模型对 —— 候选执行前由预测器给出三目标
  增量与不确定性，预测-观测对在线校准。

公式编号（Eq.1 / Eq.7 / 附录 D.2）照引代码注释原文
（`rsi/objectives.py:73`、`rsi/objectives.py:112`、`rsi/replay.py:57`），
与施工单编号一致。

| 机制 | 核心接缝文件 |
|---|---|
| MOEA/D 邻域替换 + 平衡部署 | `rsi/engine.py`（`dream` 4b/4c 段、incumbent 池）、`rsi/objectives.py`（`tchebycheff`/`IdealPoint`）、`config.py`（`subproblems`/`replace_cap`） |
| 不确定性惩罚选择（Eq.7） | `rsi/objectives.py`（`tprime`）、`rsi/predictor.py`（`uncertainty`）、`rsi/engine.py`（4a 段）、`config.py`（`uncertainty_lambda`） |
| ρ 扫描 Pareto 评分（Eq.1） | `rsi/replay.py`（`evaluate` 双层扫描）、`rsi/objectives.py`（`pareto_frontier_auc`/`parallel_penalty`）、`rsi/engine.py`（`_mix`）、`config.py`（`rho_grid`/`pareto_alpha`/`parallel_lambda`） |
| prefix-only 防泄漏（附录 D.2） | `rsi/replay.py`（`_prefix_view`/`_JUDGE_SYSTEM`/`_judge`）、`rsi/trace.py`（`serial_rounds_k`/`step_index`）、`storage/models.py`（`ToolCallNode.step_index`） |
| 债务清偿 | `storage/database.py`、`connector/http_proxy.py`、`memory/longterm.py`、`memory/tiers.py`、`memory/consolidate.py`、`rsi/engine.py`、`cli.py`（逐项见 §5） |

一轮 dream 的选择管线（本四组机制的位置）：

```
轨迹集 traces ──► 诊断（证据完整性门，E0-E3 标注）
   │
   ├─ 候选集 entries：v0 恒在（首位）+ 插件计划变异（可靠性门/保守模式
   │   可先拦截）+ 跨轮 per-λ incumbent 恢复（逐 λ 重验，失败回退 v0）
   │
   ├─ 逐 entry 回放评估：judge 一次（前缀盲评证据）+ ρ∈rho_grid 扫描
   │   → 三目标 objective + pareto_auc + (n,a) 点集          ← §3 §4
   │   混合目标 q′=(1−α)·quality+α·pareto_auc，ideal 同空间维护
   │
   ├─ 逐 entry 预测（仅变异候选）：三目标增量 + 不确定性 u    ← §2
   │   预测-观测校准（回放 objective 为观测）→ predictions 表
   │
   ├─ 4a 每候选全 λ 各一个 t′ = tchebycheff + λu·u            ← §2
   ├─ 4b 邻域替换：imp = t′(holder,λ)−t′(c,λ) > 0 即换人，
   │   降序贪心 + 单候选 replace_cap 封顶 + 惰性重验          ← §1
   ├─ 4c 质量门 G = {c : t′(c,λ₀) ≤ t′(v0,λ₀)}，门内最平衡
   │   H† = argmax_G min(q′,cost,safety)（tie-break 链偏 v0） ← §1
   │
   ├─ incumbent 池写回（spec_json + objectives + dream_seq，不存 t 值）
   └─ 报告/事件/langfuse：per-λ 概要、pareto_auc、selector=balanced
```

---

## 1. MOEA/D 邻域替换与最平衡部署（U2）

### 1.1 论文机制

Harmony Harness 把自我改进建模为多目标优化：一组权重向量 λ 各定义一个
"子问题"，每个子问题维护自己的 incumbent（现任解），候选只在邻域内
比较、**改进即换人**（neighborhood replacement），且单候选替换数有
nr 封顶以防单点垄断。此前实现只有单一标量分数（λ₀ 视角），成本优先
候选永远赢不了质量优先的 v0，第二子问题形同虚设；本轮把选择器完整
升级为 per-λ incumbent + 邻域替换 + 门内最平衡部署。

### 1.2 数学定义

设三目标向量 f(c) = (q′, cost_eff, safety)（全部越大越好；q′ 为 §3
的混合质量），理想点 z\* 由本轮全部 entries 逐维取最大动态维护
（`rsi/objectives.py:31-42`，`rsi/engine.py:339-366`）。

**子问题度量（Tchebycheff 标量化）**：

```
t(c, λ) = max_i λ_i · |z*_i − f_i(c)|        （越小越好）
```

定义于 `rsi/objectives.py:45-61`。子问题集 λ 默认两个
（`config.py:84-95`）：λ₀=[0.7, 0.2, 0.1]（质量优先，兼任质量门与
部署口径）、λ₁=[0.2, 0.7, 0.1]（成本优先）。λ_i 权重和不必为 1。

**邻域替换**（`rsi/engine.py:469-502`）：设 `inc_holder[i]` 为本轮
恢复后 λ_i 的持有者条目（初始为恢复的 incumbent，无存档则 v0），
对每个候选 c 与子问题 λ_i 计算改进量：

```
imp(c, λ_i) = t′(inc_holder[i], λ_i) − t′(c, λ_i)
```

收集全部 `imp > 1e-12` 的三元组，按 imp 降序贪心应用：替换时先
**惰性重验**（槽位可能已被更大改进换人，`engine.py:489-495`），
单候选至多替换 `replace_cap`（默认 2，MOEA/D 的 nr）个 incumbent
（`engine.py:487-488`，`config.py:102-103`）。N=2 时邻域退化为全体
λ（`engine.py:470-471` 注释明示）。

**质量门 + 最平衡部署**（`rsi/engine.py:504-530`）：

```
G   = { c : t′(c, λ₀) ≤ t′(v0, λ₀) + 1e-12 }            （质量门）
H†  = argmax_{c∈G}  min_i f_i(c)                          （最平衡）
tie-break：min 平 → λ₀-t′ 小者 → entries 序靠前者
```

平衡键 `_balance_key` 定义于 `engine.py:518-525`（元组
`(min(f), −t′(c,λ₀), −idx)` 取 max）。entries 序 v0 恒为首位
（`engine.py:229-231`），故末级平手天然偏 v0；version 不参与比较
——`model_copy` 保留父 version，同轮恒平无区分度（`engine.py:509-511`
注释）。

**非退化论证**：v0 恒在 entries 且 u(v0)≡0（§2），故 v0 ∈ G 恒成立；
当无候选更平衡时 winner = v0 → 报告 `deployed=False`、落败候选的
SOP 全部归档（`engine.py:534-547`）。门内比较全部使用**本轮回放
数字**，无跨轮口径错配（`engine.py:506-508` 注释）。

### 1.3 跨轮 incumbent 池（持久化与【P1】【P2】）

per-λ incumbent 跨 dream 轮持久化于 meta 键 `rsi_moead_incumbents`，
值形如 `{λ_index: {spec_json, objectives, dream_seq}}`
（`rsi/engine.py:269-276`、写回 `engine.py:617-633`）：

- **【P1】不存 t 值**：t′ 每轮在当前 ideal 下对全部 entries（含恢复的
  incumbent）重算，存档 objectives 仅作展示，绝不进门比较——杜绝
  "旧值过门破非退化"（`engine.py:271-273`；测试
  `tests/rsi_test.py:509-546` 用伪装 (0.99,0.99,0.99) 的存档证明劣化
  incumbent 不进门）；
- **【P2】恢复即重验**：逐 λ 走完整 validator；失败（含引用 SOP 已被
  上轮归档）丢弃该 λ incumbent 并重置为 v0（`engine.py:290-304`），
  第二轮绝不出现"保留门拒绝"路径（测试 `tests/rsi_test.py:464-506`）；
- 与当前版同 spec 的恢复条目复用既有 entry（`is_current`/`shared`，
  `engine.py:306-314`）；恢复态枚举 fresh/corrupt/dropped:\<stage\>/
  is_current/shared/reentered 全部落报告 `moead.incumbents[].restore`
  （`engine.py:280-318`、`engine.py:663-671`）；
- incumbent 条目不占 `candidates` 预算、无 sop_candidates（outcomes
  为空），落选归档循环对其无操作（`engine.py:274-276`）。

部署分数键兼容：`genome.scores["primary_tchebycheff"]` = t′(λ₀)
（`rsi_history` 工具消费的旧键），另存 `per_lambda_tchebycheff`、
`balanced_min`、`selector="balanced"`（`engine.py:572-585`；测试
`tests/rsi_test.py:594-619`）。

### 1.4 接缝文件

| 接缝 | 位置 | 作用 |
|---|---|---|
| 子问题/参数 | `config.py:84-103` | `subproblems` 两个 λ、`replace_cap`=2 |
| Tchebycheff / 理想点 | `rsi/objectives.py:31-61` | 标量化与 z\* 维护 |
| 邻域替换 4b | `rsi/engine.py:469-502` | 改进即换人 + nr 封顶 + 惰性重验 |
| 门 + 平衡部署 4c | `rsi/engine.py:504-530` | G / H† / tie-break 链 |
| incumbent 池 | `rsi/engine.py:269-318`、`engine.py:617-633` | 恢复（重验）/ 写回（不存 t） |
| 版本分键 | `rsi/engine.py:572-585` | `primary_tchebycheff` 兼容 + 新键 |
| 观测面 | `rsi/engine.py:682-720` | dream end 事件 per-λ 概要（E2）、langfuse 同名载荷 |

测试：`tests/rsi_test.py:380-619` 共 7 例——λ₁ 激活与过门（380）、
不确定性惩罚压 v0（401）、nr 封顶各归其主（438）、跨 dream SOP 归档
回归（464）、ideal 重算不采信存档（509）、引擎重启恢复（557）、
报告键与工具兼容（594）。

---

## 2. 不确定性惩罚选择（Eq.7）

### 2.1 论文机制

RSI-Anything 的策略-世界模型对：候选动作执行**之前**由世界模型预测
结果，预测的不确定性应折价——"看起来更优但预测没把握"的候选不该
按面值采信。施工单 Eq.7 把它写进接受准则 `J′ = J − λu·u`（J 为奖励
域分数）。

### 2.2 数学定义

本库的选择度量在**距离域**（越小越好），Eq.7 的转写是惩罚项**相加**
（`rsi/objectives.py:64-84`）：

```
t′(c, λ, u) = t(c, λ) + λu · max(0, u)
```

即 J = −tchebycheff（奖励域）时 `J′ = J − λu·u` 与上式等价。注释
（`objectives.py:73-77`）明确论证：施工单原文的减号写在奖励域；若按
字面在距离域做减法，高不确定候选反而被奖励，"λu>0 压过 v0"的验收
语义不可能成立。

- **u 的来源**：变异候选取预测器输出的不确定性——该算子已有 n 个
  校准样本时 `u = 1/(1+n)`，零样本走先验 `u = 0.8`
  （`rsi/predictor.py:165-185`）；v0 与无预测条目（incumbent 重评）
  u ≡ 0——**已验证真实值不惩罚**（`rsi/engine.py:393-411`）；
- **λu**：`config.py:96-101`，默认 0（关闭，t′ 退化为纯 Tchebycheff）；
- **同一本账三处共用**：λ₀ 质量门、per-λ 邻域替换比较、报告键
  `v0_score`/`winner_score`（`objectives.py:78-80`），无口径分歧。

### 2.3 与可靠性门的分工

两者同属 RSI-Anything"保留门"家族但相互独立（`rsi/engine.py:216-221`
与 `engine.py:391-411`）：

- **可靠性门（保守模式）**：样本 ≥ `predictor_min_samples`(3) 且
  MAE > `predictor_mae_threshold`(0.25) → 只放行
  `retrieval_evolve` 小步候选（`rsi/predictor.py:225-233`、
  `predictor.py:248-250`、`config.py:109-114`）——作用于**候选生成**；
- **不确定性惩罚**：逐候选把 u 计入 t′——作用于**选择度量**；
  样本不足视为"未证实不可靠"，保持探索（`predictor.py:227-229`）。

### 2.4 接缝文件

| 接缝 | 位置 | 作用 |
|---|---|---|
| t′ 定义 | `rsi/objectives.py:64-84` | Eq.7 距离域转写（单元级） |
| u 计算 | `rsi/predictor.py:165-185` | 1/(1+n) / 先验 0.8 |
| 逐 entry 全 λ t′（4a） | `rsi/engine.py:391-422` | u 取自 candidate_predictions |
| λu 配置 | `config.py:96-101` | 默认 0 关闭 |
| 事件载荷 | `rsi/engine.py:449-467` | CAT_PREDICTION 带 uncertainty(E1)/penalized(E2)/mae/samples |

测试：`tests/rsi_test.py:82-96`（t′ 单元语义：λu=0 退化、λu>0 加惩、
u=0 任意 λu 不惩罚）；`tests/rsi_test.py:401-435`（λu=1 下高不确定
伪优候选被压过 v0 不部署、λ₁ 同被压制、u=0.8 断言；对照组 λu=0
同几何照常部署）；`tests/rsi_test.py:901-926`（λu×α 双重压制交互：
伪优与劣化候选都不部署）。

---

## 3. ρ 扫描 Pareto 评分（Eq.1）

### 3.1 论文机制

单一 (top_k, char_budget) 点位回答不了"预算花得值不值"：同一个候选
在满预算下质量高、在四分之一预算下可能质量几乎不变。Eq.1 用
**预算-质量前沿面积**衡量"任意预算水平下可达的最佳质量"，落在
Dream-RSI 的回放模拟器上做反事实扫描——记忆库与检索是纯函数，
ρ 插值一遍即可枚举整条前沿。

### 3.2 数学定义

**ρ 网格与参数插值**（`rsi/replay.py:264-278`）：ρ ∈ `rho_grid`
（默认 [0, 0.25, 0.5, 0.75, 1.0]，`config.py:115-119`），候选检索
参数按 ρ 线性插值，threshold 不插值：

```
top_k(ρ)      = max(1, round(lerp(1, top_k_cand, ρ)))
char_budget(ρ)= max(200, round(lerp(200, char_budget_cand, ρ)))
```

**质量-成本点**（逐 (trace_i, ρ)，`replay.py:280-308`）：

```
n_iρ = trace_i.total_tokens + (chars_c(ρ) − chars_p) / 2     （成本）
q_iρ = clip[0,1]( base_quality·(1 − 0.2·lost_iρ) + quality_floor )
a_iρ = q_iρ / S ,  S = max_j outcome.score_j（池内最高分）
       S ≤ 0 ⇒ a_iρ ≡ 0     （【P10】全零分池 guard，replay.py:233-235/304-307）
```

`lost_iρ` 是该轨迹 used 记忆在候选参数下的丢失率（回归覆盖锚点，
`replay.py:249-293`）；quality_floor = 0.1·sop_coverage +
0.15·judge_delta 与 ρ 无关（ρ 只经 lost 起作用，`replay.py:294-296`）。

**前沿面积**（`rsi/objectives.py:109-140`）：对每条轨迹的
{(n_iρ, a_iρ)} 点集做非支配过滤（n 小且 a 大者胜）→ n 升序 →

```
AUC_i = Σ_{j≥2} (ā_j − ā_{j−1}) · (B_i − n̄_j) / B_i ,
B_i = n_i,ρ=1     （ρ=1 候选满预算点为归一化基准，replay.py:309-311）
```

单点/空集返回 0（无步进可言）。**候选总分**（`replay.py:348-361`）：

```
pareto_auc = clip[0,1]( mean_i AUC_i − λ_∥ · mean_i (k*_i / n_calls_i) )
```

λ_∥ = `parallel_lambda` 默认 0——现网轨迹串行、k\*/n 区分度弱，
接口就位不硬凑（`config.py:123-125`）；k\* 见 §4.2。

**混合入排序目标**（`rsi/engine.py:320-337`）：前沿面积不另立山头，
按 `pareto_alpha`（默认 0.5，`config.py:120-122`）混进质量维：

```
q′ = (1−α)·quality + α·pareto_auc
```

对全候选（含 v0）一致计算；**ideal 与 t′ 必须同一目标空间**——
ideal 也按混合目标维护，否则质量项 |z−q′| 恒大、掩蔽其他维度
（`engine.py:323-326` 注释）；α=0 回退旧行为（纯 quality，
测试 `tests/rsi_test.py:885-898`）。`objective` 三分量仍取 ρ=1
单点产出（兼容既有分量与测试，`replay.py:326-341`）。

【P10】缩放不变性：judge 不跑且无 SOP 时 a 的分子分母同缩放，
全池 score 同乘常数 AUC 不变（测试 `tests/rsi_test.py:849-865`
显式化该条件并断言）。

### 3.3 接缝文件

| 接缝 | 位置 | 作用 |
|---|---|---|
| ρ 网格配置 | `config.py:115-125` | `rho_grid` / `pareto_alpha` / `parallel_lambda` |
| 双层扫描循环 | `rsi/replay.py:223-324` | 逐 (trace, ρ) 插值检索 → (n,a) 点 |
| Eq.1 面积 | `rsi/objectives.py:109-140` | 非支配过滤 + 阶梯面积 |
| 并行度惩罚 | `rsi/objectives.py:143-151` | λ_∥·mean(k\*/n) |
| 混合目标 | `rsi/engine.py:320-337` | q′ 与 ideal 同空间 |
| 报告顶层键 | `rsi/engine.py:653-654`、`bench/rounds.py:191-197` | `pareto_auc_v0`/`pareto_auc_winner`（不靠 replays 猜行） |

测试：`tests/rsi_test.py:775-792`（面积与惩罚单元：被支配点剔除、
单点/空集/零预算=0、惩罚线性）；`tests/rsi_test.py:795-883`（3 轨迹
ρ 扫描点集：5 个 ρ、n 单调、B=ρ=1 点、【P10】缩放不变与全零分池）；
`tests/rsi_test.py:885-926`（α=0 回退、λu×α 双重压制）。

---

## 4. prefix-only 防泄漏（附录 D.2）

### 4.1 论文机制

回放评审（LLM judge）若能看到轨迹**结局分数**，会锚定绝对分数给出
"这个轨迹本来就差/好"的评审，把结局信息泄漏进本应只反映候选变更
预期效果的 judge_delta——评审增量与结局分数脱钩。附录 D.2 的前缀
盲评约束：评审只允许依据轨迹**前缀**统计（执行前段可见的信息），
结局分数一律不出现在证据里。

### 4.2 定义与实现

**前缀视图 `_prefix_view`**（`rsi/replay.py:89-149`）对诊断 dict 做
浅拷贝后：

- **剥离**：`stats.mean_outcome` 与 `worst/best_traces[].score`
  （`replay.py:107-124`）——结局分数不进评审视野；
- **前置**：每条被引轨迹按 `step_index` 序取**前 5 步**
  （`_PREFIX_STEPS=5`，`replay.py:85-86`）工具调用的
  `(step, name, state)`（无任何分数），并附 `(n, k*)` 数据：
  `calls_n = len(tool_calls)`、`serial_rounds_k`
  （`replay.py:127-147`）；
- **k\***（串行轮数）：按 `at` 间隔 > 0.5s 聚类估计——完全并行的
  一批算一轮，完全串行的 n 次调用得 k\*=n（`rsi/trace.py:46-65`，
  `_SERIAL_GAP_S=0.5`），与 §3 并行度惩罚共用同一口径
  （`trace.py:53-56`）。

**录制侧支撑**：`TraceRecorder.add_tool_call` 自增填 `step_index`
（`rsi/trace.py:101`；`storage/models.py:162-166` 新增
`step_index`/`batch_id` 字段，旧库 JSON 无此键照常加载为 None）；
结算时 `outcome.signals` 落 `calls_n`/`serial_rounds_k`
（`trace.py:148-151`）。

**约束语进 system 提示**（`replay.py:50-58`）："评审只依据【证据】
中已揭示的\*\*前缀\*\*统计……禁止引用或推断结局绝对分数（附录 D.2
前缀盲评约束）"。

**两个边界**：

- **公平性【P5】**：v0 仅首轮（operator=='seed'）免评，第二轮起
  v0 同被 judge——公平性按"全部条目同视野同评审"论证
  （`replay.py:440-446`）；judge 每候选只调一次，**与 ρ 无关**
  （rho_grid 不放大评审调用，`replay.py:201-207`）。测试
  `tests/rsi_test.py:694-729`：首轮 1 次评审调用，第二轮 2 次，
  且全部条目证据同为前缀视图；
- **非对称裁剪**：前缀视图只喂评审侧；诊断全量数据仍留给策略侧
  `hypothesize`（improver 不动，`replay.py:95-97` 注释）——防泄漏
  针对的是"评审锚定结局"，不是"策略不许看结局"。

### 4.3 接缝文件

| 接缝 | 位置 | 作用 |
|---|---|---|
| 前缀视图 | `rsi/replay.py:89-149` | 剥离结局分 / 前置前缀序列与 (n,k\*) |
| 约束语 | `rsi/replay.py:50-58` | 附录 D.2 前缀盲评写进 judge system |
| 评审入口 | `rsi/replay.py:431-490` | evidence = prefix_view(diagnosis) |
| 录制序号 | `rsi/trace.py:101`、`storage/models.py:162-166` | step_index / batch_id |
| 串行轮数 | `rsi/trace.py:46-65` | k\*（与 §3 λ_∥ 共口径） |

测试：`tests/rsi_test.py:633-691`（证据无 `mean_outcome`/`"score"`、
含 prefix_tools/calls_n/serial_rounds_k、前缀步只有
step/name/state、judge_delta 仍解析、system 含约束语）；
`tests/rsi_test.py:732-769`（step_index 自增、旧 JSON 兼容、
间隔 <0.5s 得 k\*=1 / 间隔 2s 得 k\*=4）。

---

## 5. 债务清偿清单

本轮同时清偿两类债务：上轮记忆组件评审遗留的 **U1 债务①-⑥**（修复
随本轮工作区改动落地，基线 `c32b9de` 不含），与本轮施工单自身的评审项
**P1/P2/P5/P7/P10**（P1/P2/P5 已在 §1/§4 展开，此处汇总）。

### 5.1 U1 债务（上轮遗留，本轮清偿）

| 债务 | 内容 | 清偿方式（接缝） | 测试 |
|---|---|---|---|
| ① | merge 幂等锚点：已合并 duplicate 机械返回本次 primary，与既有链接指向不一致 | 幂等分支跟随 `merged_into` 锚点；锚点失联则改指本次 primary 走正常合并（`memory/longterm.py:110` 起的合并幂等分支） | `tests/memory_test.py`（`test_merge_idempotent_follows_anchor` / `test_merge_reanchors_when_anchor_lost`） |
| ② | 断链 stats 逐行 json.loads 全表（大库绝大多数行 `links='[]'`） | SQL 预过滤空链行 + 逐行 try/except，脏 JSON 计入 `broken_json` 不影响 `broken_links` 语义（`storage/database.py:934-940`） | `test_stats_prefilters_empty_links_and_skips_dirty_json` |
| ③ | `tier IN ()` 空拆分的方言依赖 | 空拆分显式 `AND 1=0` 返回空集，语义锁定不依赖 SQLite 方言（`storage/database.py:340-343`、`database.py:350` 两处） | `test_list_memories_empty_split_returns_empty` |
| ④ | 后台任务只持弱引用会被 GC 中途回收；进程内计数 get+set 有丢更新窗口 | 强引用集合 + done 回调移除（`rsi/engine.py:102-104`、`connector/http_proxy.py:41-47`）；`bump_meta` 单 locked 段原子自增（`storage/database.py:885-906`，dream_seq 与 proxy_turns 共用） | `test_bump_meta_atomic_under_concurrency`、`test_proxy_turn_counter_in_meta` |
| ⑤ | 维护扫描 `list_memories(limit=100)` 截断漏扫；存在性检查走截断列表漏判重复 | `(updated_at,id)` 复合游标走窗：主边界 `<=` 允许边界行重复入窗（幂等）、并列时间戳组按 id 走窗不跳行（`storage/database.py:366-380`、`database.py:396-408`、消费方 `memory/consolidate.py:110`）；`find_by_title` 精确查重（`database.py:431-439`，SOP 蒸馏 `consolidate.py:214`、SOP 落库 `memory/tiers.py:357`） | `test_link_repair_cursor_covers_beyond_single_limit`、`test_cursor_resets_after_tail_and_restarts_from_newest`、`test_sop_dedup_exact_query_beyond_list_limit`、`test_list_memories_updated_before_cursor_boundary`、`test_find_by_title` |
| ⑥ | 两次审查之间的注入面无预算兜底 | `_core_context_sections` 运行时按 `l3_char_budget − L3_INDEX_MARGIN` 截断（口径 `memory/tiers.py:89-93`、执行 `tiers.py:110`；库内瘦身提案之外的注入面兜底） | `tests/connector_test.py:282-341`（代理前缀 L3 字符预算单元） |

### 5.2 本轮施工单评审项

| 项 | 内容 | 落实位置 | 测试 |
|---|---|---|---|
| P1 | 跨轮 incumbent 不得采信存档目标值过门 | 池只存 spec_json/objectives/dream_seq，t′ 每轮当前 ideal 重算（`rsi/engine.py:269-276`、`engine.py:617-618`；详见 §1.3） | `tests/rsi_test.py:509-546` |
| P2 | 跨轮 incumbent 引用已归档 SOP 会在部署期触发"保留门拒绝" | 恢复即逐 λ 重验，失败回退 v0（`rsi/engine.py:290-304`；详见 §1.3） | `tests/rsi_test.py:464-506` |
| P5 | v0 首轮免评造成评审视野不对称；rho_grid 放大评审调用 | 第二轮起 v0 同被 judge；judge 与 ρ 无关每候选一次（`rsi/replay.py:201-207`、`replay.py:440-446`；详见 §4.2） | `tests/rsi_test.py:694-729` |
| P7 | Pareto 退化告警若直读 meta，旧报告缺字段易误报 | 改读事件日志末 5 次 dream end 的 `pareto_auc_winner`，持续下降且总降幅 >0.1 才告警；序列不足/字段缺失静默降级（`cli.py:170-192`） | `tests/rsi_anything_test.py:444-494` |
| P10 | AUC 缩放不变性依赖隐式条件；全零分池除零 | 条件显式化（judge 不跑 + 无 SOP）；S≤0 ⇒ a≡0（`rsi/replay.py:233-235`、`replay.py:304-307`；详见 §3.2） | `tests/rsi_test.py:849-883` |

### 5.3 随债务清偿新增的观测哨兵（U4）

- **子问题覆盖哨兵**：meta 池全同 version 且多子问题 → λ≥1 邻域替换
  从未产生独立 incumbent（per-λ 机制退化）→ warn（`cli.py:136-168`；
  测试 `tests/rsi_anything_test.py:383-441`）；
- **dream end 事件增量**：`pareto_auc_winner/v0`、`selector`、
  per-λ incumbent 概要（λ/version/tprime/dream_seq，E2）与
  CAT_PREDICTION 的 `uncertainty`(E1)/`penalized`(E2) 入观测面
  （`rsi/engine.py:449-467`、`engine.py:682-720`；测试
  `tests/rsi_anything_test.py:310-338`）；
- **基准增量字段**：逐轮 `pareto_auc`（部署取 winner、否则 v0 顶层键）
  与 summary 的 `pareto_auc_first/last/delta`、`balanced_deploys`
  （`bench/rounds.py:191-197`、`rounds.py:220-233`；AUC 阈值断言
  观测先行，待 bench_report 积累后下轮再定）。

---

## 6. 验证

全套离线测试（`rsi_mem/pytest.ini`，`asyncio_mode=auto`）：

```bash
cd rsi_mem && python -m pytest tests -q    # 124 passed（本轮实测）
```

本轮 rsi_test.py 由 10 例增至 25 例（§1-§4 逐机制断言）、
rsi_anything_test.py 由 13 例增至 16 例（U4 事件与哨兵），另有
storage/memory/consolidate/connector 的债务清偿用例（§5.1）。
非退化断言链：门内胜者 `winner_score ≤ v0_score + 1e-9`
（`tests/rsi_test.py:243-245`）在 λu>0、α>0、nr 封顶、跨轮恢复、
引擎重启五类场景下均成立（§1-§3 各测试），10 轮基准末轮环境质量
不低于首轮（容差 0.02，`bench/rounds.py:216-219`、
`tests/rsi_anything_test.py:264`）。
