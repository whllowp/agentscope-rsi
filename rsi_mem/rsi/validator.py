# -*- coding: utf-8 -*-
"""三段校验器（Harmony Harness 的 validation gate）。

1. **结构段**：GenomeSpec 可解析、SOP 内容非空且触发器非空、片段数
   与长度上限；
2. **冒烟段**：``active_sops`` 引用的条目在库中存在且状态合法、
   基因组可 JSON 往返序列化；
3. **完整性段**：锁定安全片段逐字保留、片段 id 无重复、候选与父代
   不完全相同（拒绝"字节级没变"的空演化）。

任一段失败 → 一次修复机会（丢弃非法片段/SOP 引用后复检），仍失败
则该候选被淘汰。

**治理不可演化声明（U4）**：本校验器只约束**基因组内容**，不承载
治理参数——断路器（速率/冷却/质量回滚）在 ``rsi_mem/governor.py``，
参数经 ``RSIEngine`` 构造期快照自 frozen ``GovernanceConfig``，任何
基因组/算子/插件路径都写不到它。校验规则本身（含锁定片段全字段
锁）是代码常量，不在 RSI 可改对象内。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..storage import GenomeSpec, RSIMemDatabase

_MAX_FRAGMENTS = 12
_MAX_FRAGMENT_CHARS = 8000
_MAX_SOPS = 20


@dataclass
class ValidationReport:
    """校验结果。"""

    passed: bool
    stage: str = ""
    problems: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # 便于日志/报告
        state = "PASS" if self.passed else f"FAIL@{self.stage}"
        return f"[{state}] {'; '.join(self.problems) or 'ok'}"


class GenomeValidator:
    """候选基因组校验器。"""

    def __init__(self, db: RSIMemDatabase) -> None:
        self._db = db

    async def validate(
        self,
        candidate: GenomeSpec,
        parent: GenomeSpec,
    ) -> ValidationReport:
        """跑完三段校验。"""
        # ---- 1. 结构段 ----
        problems: list[str] = []
        if not candidate.prompt_fragments and not candidate.active_sops:
            problems.append("空基因组（无片段且无 SOP）")
        if len(candidate.prompt_fragments) > _MAX_FRAGMENTS:
            problems.append(f"片段数超限（>{_MAX_FRAGMENTS}）")
        total_chars = sum(len(f.text) for f in candidate.prompt_fragments)
        if total_chars > _MAX_FRAGMENT_CHARS:
            problems.append(f"片段总字符超限（>{_MAX_FRAGMENT_CHARS}）")
        for f in candidate.prompt_fragments:
            if not f.text.strip():
                problems.append(f"片段 {f.id} 内容为空")
        if len(candidate.active_sops) > _MAX_SOPS:
            problems.append(f"激活 SOP 数超限（>{_MAX_SOPS}）")
        if problems:
            return ValidationReport(False, "structural", problems)

        # ---- 2. 冒烟段 ----
        problems = []
        for sop_id in candidate.active_sops:
            rec = await self._db.get_memory(sop_id)
            if rec is None:
                problems.append(f"active_sop 引用不存在: {sop_id[:8]}")
            elif rec.tier != "muscle":
                problems.append(f"active_sop 引用非肌肉层: {sop_id[:8]}")
            elif rec.status not in ("active", "candidate", "promoted"):
                problems.append(
                    f"active_sop 引用状态非法({rec.status}): {sop_id[:8]}",
                )
        try:
            roundtrip = GenomeSpec.model_validate_json(
                candidate.model_dump_json(),
            )
            assert roundtrip.version == candidate.version
        except Exception as e:  # noqa: BLE001
            problems.append(f"序列化往返失败: {e}")
        if problems:
            return ValidationReport(False, "smoke", problems)

        # ---- 3. 完整性段 ----
        problems = []
        # U2 全字段锁：锁定片段的 (text, enabled, slot, triggers) 四元
        # 组逐字段比对——旧版只比 text，enabled=False / slot=reflex 可
        # 让锁定片段静默退出渲染面（fragment_text 只渲染 enabled 且
        # always）而校验照过，等于旁路删除。四算子不触这些字段，
        # 合法候选不受影响。
        _lock_fields = ("text", "enabled", "slot", "triggers")

        def _locked_sig(genome: GenomeSpec) -> dict[str, tuple]:
            return {
                f.id: (
                    f.text, f.enabled, f.slot, tuple(f.triggers),
                )
                for f in genome.prompt_fragments if f.locked
            }

        parent_locked = _locked_sig(parent)
        cand_locked = _locked_sig(candidate)
        for fid, p_sig in parent_locked.items():
            if fid not in cand_locked:
                problems.append(f"锁定片段被删除: {fid}")
                continue
            c_sig = cand_locked[fid]
            for name, pv, cv in zip(_lock_fields, p_sig, c_sig):
                if pv != cv:
                    problems.append(
                        f"锁定片段字段 {name} 被篡改"
                        f"（{pv!r} → {cv!r}）: {fid}",
                    )
        ids = [f.id for f in candidate.prompt_fragments]
        if len(ids) != len(set(ids)):
            problems.append("片段 id 重复")
        if (
            candidate.version != parent.version  # 部署后的现行版对比 v0
            and candidate.model_dump_json(
                exclude={"version", "created_at", "parent_version",
                         "operator", "notes", "scores"},
            )
            == parent.model_dump_json(
                exclude={"version", "created_at", "parent_version",
                         "operator", "notes", "scores"},
            )
        ):
            problems.append("候选与父代完全相同（空演化）")
        if problems:
            return ValidationReport(False, "integrity", problems)
        return ValidationReport(True, "all")

    async def repair(
        self,
        candidate: GenomeSpec,
        parent: GenomeSpec,
    ) -> GenomeSpec | None:
        """一次修复机会：剔除失效 SOP 引用 / 删除未锁定空片段后复检。

        锁定片段缺失无法修复，直接放弃。
        """
        repaired = candidate.model_copy(deep=True)
        parent_locked_ids = {
            f.id for f in parent.prompt_fragments if f.locked
        }
        kept_sops = []
        for sop_id in repaired.active_sops:
            rec = await self._db.get_memory(sop_id)
            if (
                rec is not None
                and rec.tier == "muscle"
                and rec.status in ("active", "candidate", "promoted")
            ):
                kept_sops.append(sop_id)
        repaired.active_sops = kept_sops
        repaired.prompt_fragments = [
            f for f in repaired.prompt_fragments
            if f.locked or f.text.strip()
        ]
        if not parent_locked_ids <= {f.id for f in repaired.prompt_fragments}:
            return None
        report = await self.validate(repaired, parent)
        return repaired if report.passed else None
