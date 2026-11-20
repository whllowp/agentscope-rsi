# -*- coding: utf-8 -*-
"""U4 治理断路器测试：速率/冷却/连续上限/质量回滚/pin/fail-closed/
构造期快照免疫/回滚内检/bench 契约。"""
import json
import os
from pathlib import Path

import pytest

from rsi_mem.config import GovernanceConfig, RSIMemConfig
from rsi_mem.context import RSIMemContext
from rsi_mem.governor import Governor
from rsi_mem.storage import PromptFragment

from .rsi_test import (
    _PlansPlugin,
    _cost_plan,
    _seed_cost_stage,
    _token_trace,
    _u2_ctx,
)


@pytest.fixture
async def ctx(tmp_path):
    context = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem")),
    )
    yield context
    await context.close()


# ---------------------------------------------------------------------------
# ① 速率 / 冷却
# ---------------------------------------------------------------------------

def test_governor_rate_limit(tmp_path):
    """第 7 次/小时拒绝（默认上限 6）；冷却期内第二次拒。"""
    g = Governor(
        GovernanceConfig(
            max_deploys_per_hour=6, min_deploy_interval_s=0.0,
        ),
        str(tmp_path / "g.json"),
    )
    base = 1000.0
    for i in range(6):
        decision = g.check_deploy(base + i, "t", 0)
        assert decision.allowed, decision.reason
        g.record_deploy(base + i, i + 1)
    denied = g.check_deploy(base + 10, "t", 0)
    assert not denied.allowed and "rate" in denied.reason

    g2 = Governor(
        GovernanceConfig(
            max_deploys_per_hour=100, min_deploy_interval_s=300.0,
        ),
        str(tmp_path / "g2.json"),
    )
    assert g2.check_deploy(1000.0, "t", 0).allowed
    g2.record_deploy(1000.0, 1)
    cooldown = g2.check_deploy(1100.0, "t", 0)
    assert not cooldown.allowed and "cooldown" in cooldown.reason
    assert g2.check_deploy(1300.5, "t", 0).allowed, "出冷却期即放行"


# ---------------------------------------------------------------------------
# ② 连续上限 → 暂停；手工编辑恢复
# ---------------------------------------------------------------------------

def test_governor_consecutive_pause(tmp_path):
    path = Path(tmp_path / "g.json")
    g = Governor(
        GovernanceConfig(
            max_deploys_per_hour=100, min_deploy_interval_s=0.0,
            consecutive_deploy_limit=3,
        ),
        str(path),
    )
    assert g.check_deploy(1.0, "t", 2).allowed  # 2 < 3
    denied = g.check_deploy(2.0, "t", 3)
    assert not denied.allowed and "consecutive" in denied.reason
    assert g.deployment_pause is True, "触顶应置 deployment_pause"

    # 人工干预路径：手工编辑 governor.json 清 pause → 恢复
    state = json.loads(path.read_text(encoding="utf-8"))
    state["deployment_pause"] = False
    state["paused_reason"] = ""
    path.write_text(json.dumps(state), encoding="utf-8")
    assert g.check_deploy(3.0, "t", 0).allowed


# ---------------------------------------------------------------------------
# ③ 质量回归自动回滚 + 暂停
# ---------------------------------------------------------------------------

async def test_governor_quality_rollback(tmp_path):
    """部署后 20 条均值下滑 >0.1 → 下一轮 dream 自动回滚（active 前移
    operator=rollback）且后续部署被 pause 拦。"""
    ctx = _u2_ctx(tmp_path, "u4_qroll")
    try:
        await _seed_cost_stage(ctx)  # 4 条部署前轨迹（0.7）
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        r1 = await ctx.engine.dream(trigger="u4-q1")
        assert r1["deployed"] is True
        assert (await ctx.db.active_genome()).version == 1

        for i in range(20):  # 部署后真实轨迹均值 0.5（下滑 0.2 > 0.1）
            # 查询带「磁盘」：保持记忆可召回，恢复的 incumbent 候选
            # 在成本维仍严格占优 → 会真实走到部署步（被 pause 拦）
            await _token_trace(ctx, f"磁盘告警下滑 {i}", 0.5)
        r2 = await ctx.engine.dream(trigger="u4-q2")
        active = await ctx.db.active_genome()
        assert active.operator == "rollback", "质量回归应触发自动回滚"
        assert active.version == 2
        assert ctx.engine._governor.deployment_pause is True
        # 后续部署被 pause 拦（r2 内恢复的 incumbent 试图重部署）
        assert r2["deployed"] is False
        assert str(r2.get("reason", "")).startswith("governor:")
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# ④ pin：engine.rollback 内检 + 部署路径
# ---------------------------------------------------------------------------

async def test_governor_pin_via_engine_rollback(tmp_path, monkeypatch):
    """RSI_MEM_GOVERNOR_PIN 构造期读取：直接调 engine.rollback() 被拒
    （None + 事件 + active 不变）；部署路径同样被拒。"""
    monkeypatch.setenv("RSI_MEM_GOVERNOR_PIN", "1")
    ctx = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / "u4_pin")),
    )
    try:
        seed = await ctx.db.active_genome()
        v1 = seed.model_copy(deep=True)
        v1.version = 1
        v1.parent_version = 0
        v1.operator = "retrieval_evolve"
        v1.retrieval = v1.retrieval.model_copy(update={"top_k": 4})
        await ctx.db.save_genome(v1)

        result = await ctx.engine.rollback()
        assert result is None, "pin 期回滚必须拒"
        assert (await ctx.db.active_genome()).version == 1
        errors = ctx.events.read(category="error", limit=10)
        assert any(
            e["payload"].get("kind") == "pin" for e in errors
        ), "pin 拒绝应落 CAT_ERROR 事件"

        # 部署路径同样被拒（同一 Governor 实例）——成本舞台保证候选
        # 真的会赢到部署步（4 条轨迹 < min_traces_after_deploy=5，
        # 质量规则不会先触发）
        await _seed_cost_stage(ctx)
        ctx.engine.registry.add(
            _PlansPlugin("cost-tightener", [_cost_plan()]),
        )
        report = await ctx.engine.dream(trigger="u4-pin")
        assert report["deployed"] is False
        assert "pinned" in str(report.get("reason", ""))
        assert report["governor_decision"]["allowed"] is False
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# ⑤ 构造期快照免疫配置重赋值
# ---------------------------------------------------------------------------

async def test_governor_immune_to_config_reassignment(tmp_path):
    """ctx 构造后整字段重赋值 governance 被 pydantic 放行（边界如实），
    但 Governor 行为仍按构造期快照（第 7 次拒绝）。"""
    ctx = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / "u4_imm")),
    )
    try:
        # 边界证明：整字段重赋值是 ALLOWED 的（frozen 只拦属性突变）
        ctx.config.governance = GovernanceConfig(
            max_deploys_per_hour=999, min_deploy_interval_s=0.0,
        )
        assert ctx.config.governance.max_deploys_per_hour == 999

        g = ctx.engine._governor
        base = 100_000.0
        for i in range(6):  # 间隔 301s 跳过快照的 300s 冷却
            decision = g.check_deploy(base + i * 301.0, "t", 0)
            assert decision.allowed, decision.reason
            g.record_deploy(base + i * 301.0, i + 1)
        denied = g.check_deploy(base + 6 * 301.0, "t", 0)
        assert not denied.allowed and "rate" in denied.reason, \
            "快照仍按原 max=6 / interval=300（重赋值的 999/0 不生效）"
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# ⑥ 构造顺序：Governor 先于一切插件启动
# ---------------------------------------------------------------------------

async def test_governor_constructed_before_plugins(ctx):
    """engine 构造即有 Governor；此刻无任何 dream/插件启动痕迹。"""
    assert ctx.engine._governor is not None
    assert ctx.events.read(category="dream") == [], "尚无 dream"
    assert not os.path.exists(
        ctx.engine._governor.state_path,
    ), "构造期无状态写入（懒写）"


# ---------------------------------------------------------------------------
# ⑦ 状态损坏 fail-closed
# ---------------------------------------------------------------------------

async def test_governor_state_corrupt_fail_closed(tmp_path):
    """写坏 governor.json → check_deploy 拒（fail-closed）+ doctor warn；
    修复文件后从修复内容现状继续（不重置为空）。"""
    from rsi_mem.cli import doctor

    ctx = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / "u4_fc")),
    )
    try:
        g = ctx.engine._governor
        g.record_deploy(1.0, 1)
        with open(g.state_path, "w", encoding="utf-8") as f:
            f.write("{corrupt")

        denied = g.check_deploy(2.0, "t", 0)
        assert not denied.allowed
        assert "fail-closed" in denied.reason
        assert g.degraded is True
        result = await doctor(ctx)
        assert any(
            "fail-closed" in f["message"] and f["severity"] == "warn"
            for f in result["findings"]
        )

        # 修复：写回 6 条部署（默认上限 6）→ 立即按 6 计数拒绝
        with open(g.state_path, "w", encoding="utf-8") as f:
            json.dump({
                "deploys": [[1.0, i] for i in range(6)],
                "pinned_version": None,
                "deployment_pause": False,
                "paused_reason": "",
                "rollbacks": [],
            }, f)
        after = g.check_deploy(2.0, "t", 0)
        assert not after.allowed and "rate" in after.reason, \
            "恢复后从修复文件现状继续，而非重置"
    finally:
        await ctx.close()


# ---------------------------------------------------------------------------
# ⑦b（复核修订）状态文件被删 → fail-closed（删除不是首装）
# ---------------------------------------------------------------------------

async def test_governor_state_deleted_fail_closed(tmp_path):
    """已写过/见过的 governor.json 被删 → check_deploy 拒（fail-closed）、
    doctor warn；手工恢复文件后继续。首装（从未写过）不受影响。"""
    from rsi_mem.cli import doctor

    ctx = RSIMemContext(
        RSIMemConfig(data_dir=str(tmp_path / "u4_del")),
    )
    try:
        g = ctx.engine._governor
        g.record_deploy(1.0, 1)
        assert os.path.isfile(g.state_path)
        os.remove(g.state_path)  # 插件可写 data_dir 内的删除路径

        denied = g.check_deploy(2.0, "t", 0)
        assert not denied.allowed, "删除状态文件不得 fail-open"
        assert "fail-closed" in denied.reason
        assert g.degraded is True
        result = await doctor(ctx)
        assert any(
            "fail-closed" in f["message"] and f["severity"] == "warn"
            for f in result["findings"]
        )
        # 记账在 degraded 期被丢弃（不复活被删文件）
        g.record_deploy(3.0, 2)
        assert not os.path.isfile(g.state_path)

        # 手工恢复 → 从恢复内容继续
        with open(g.state_path, "w", encoding="utf-8") as f:
            json.dump({
                "deploys": [[1.0, i] for i in range(6)],
                "pinned_version": None,
                "deployment_pause": False,
                "paused_reason": "",
                "rollbacks": [],
            }, f)
        after = g.check_deploy(2.0, "t", 0)
        assert not after.allowed and "rate" in after.reason
    finally:
        await ctx.close()

    # 首装（从未写过状态文件）：缺失即默认空态，非 degraded
    fresh = Governor(
        GovernanceConfig(),
        str(tmp_path / "never_written.json"),
    )
    assert fresh.degraded is False
    assert fresh.check_deploy(1000.0, "t", 0).allowed


# ---------------------------------------------------------------------------
# ⑧ frozen / 基因组隔离 / 状态文件位置
# ---------------------------------------------------------------------------

async def test_governance_config_frozen(ctx):
    """属性突变抛 ValidationError；基因组不含 governance；治理状态在
    data_dir/governor.json 而非 meta 表。"""
    from pydantic import ValidationError

    from rsi_mem.storage import GenomeSpec

    cfg = GovernanceConfig()
    with pytest.raises(ValidationError):
        cfg.max_deploys_per_hour = 99

    assert "governance" not in GenomeSpec.seed().model_dump(), \
        "基因组是 RSI 可改对象，不得携带治理参数"

    ctx.engine._governor.record_deploy(123.0, 1)
    assert await ctx.db.get_meta("governor") is None, \
        "治理状态不得存放在插件可写的 meta 表"
    assert os.path.isfile(
        os.path.join(ctx.config.data_dir, "governor.json"),
    )


# ---------------------------------------------------------------------------
# ⑨ 回滚复验（防污染入库）
# ---------------------------------------------------------------------------

async def test_rollback_revalidates(ctx):
    """prev 带 dangling active_sop → rollback 拒（None + CAT_ERROR），
    污染版本不得经回滚入库。"""
    seed = await ctx.db.active_genome()
    v1 = seed.model_copy(deep=True)
    v1.version = 1
    v1.parent_version = 0
    v1.operator = "logic_evolve"
    v1.prompt_fragments.append(PromptFragment(
        id="x1", text="片段一", origin="rsi",
    ))
    v1.active_sops = ["no-such-sop"]
    await ctx.db.save_genome(v1)
    v2 = seed.model_copy(deep=True)
    v2.version = 2
    v2.parent_version = 1
    v2.operator = "logic_evolve"
    v2.prompt_fragments.append(PromptFragment(
        id="x2", text="片段二不一样", origin="rsi",
    ))
    await ctx.db.save_genome(v2)

    result = await ctx.engine.rollback()
    assert result is None
    assert (await ctx.db.active_genome()).version == 2
    errors = ctx.events.read(category="error", limit=10)
    assert any(
        e["payload"].get("kind") == "revalidation" for e in errors
    ), "复验拒绝应落 CAT_ERROR"


# ---------------------------------------------------------------------------
# ⑪ bench 契约：宽松治理构造期传入 + summary 兼容
# ---------------------------------------------------------------------------

async def test_bench_summary_compat(tmp_path):
    """宽松治理构造期传入 → 10 轮 bench：旧四键语义原样、新键只增、
    deploys≥1、rows 带 execution_gate/governor 列。"""
    from rsi_mem.bench.rounds import run_bench

    cfg = RSIMemConfig(
        data_dir=str(tmp_path / "bench_u4"),
        governance=GovernanceConfig(
            max_deploys_per_hour=10 ** 6,
            min_deploy_interval_s=0.0,
            consecutive_deploy_limit=10 ** 6,
        ),
    )
    context = RSIMemContext(cfg)
    try:
        report = await run_bench(context, rounds=10, traces_per_round=3)
        s = report["summary"]
        # 旧键：名与类型语义原样
        assert isinstance(s["quality_first"], float)
        assert isinstance(s["quality_last"], float)
        assert isinstance(s["deploys"], int) and s["deploys"] >= 1
        assert isinstance(s["prediction_samples"], int)
        assert s["non_degraded"] is True
        for key in (
            "quality_first", "quality_last", "quality_delta", "deploys",
            "final_version", "prediction_samples", "non_degraded",
        ):
            assert key in s
        # 新键只增不改
        assert "execution_gate_passes" in s and "governor_denies" in s
        assert s["execution_gate_passes"] >= 1, "部署轮执行门均应通过"
        assert s["governor_denies"] == 0, "宽松治理构造期传入 → 零拒绝"
        rows = report["rows"]
        assert len(rows) == 10
        assert all(
            "execution_gate" in r and "governor" in r for r in rows
        )
        assert "governor_state" in rows[0], "首轮记录治理器状态快照"
        assert json.loads(json.dumps(report))  # 可序列化往返
    finally:
        await context.close()
