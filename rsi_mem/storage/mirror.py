# -*- coding: utf-8 -*-
"""Markdown 镜像：把肌肉记忆/长期记忆写成人类可读（且可加载）的文件。

- 肌肉记忆 → ``mirror/muscle/<name>/SKILL.md``，frontmatter 兼容
  ``agentscope.skill.LocalSkillLoader`` —— SOP 因此可以同时作为 Agent
  技能被加载（dsh-memory-evolve 的技能自固化思路）。
- 长期记忆 L1/L2 → ``mirror/longterm/<id>.md``，纯人类阅读用。

镜像是 DB 的写穿副本，可随时删除重建（``rebuild``）。
"""
from __future__ import annotations

import os
import re

import frontmatter

from agentscope._logging import logger

from .models import MemoryRecord


def _safe_name(text: str, fallback: str) -> str:
    """把标题压成文件系统安全的目录名。"""
    name = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text).strip("-")
    return name[:48] or fallback


class MirrorStore:
    """维护 ``<data_dir>/mirror`` 下的 markdown 镜像。"""

    def __init__(self, data_dir: str) -> None:
        self.muscle_dir = os.path.join(data_dir, "mirror", "muscle")
        self.longterm_dir = os.path.join(data_dir, "mirror", "longterm")
        os.makedirs(self.muscle_dir, exist_ok=True)
        os.makedirs(self.longterm_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 肌肉记忆 → SKILL.md
    # ------------------------------------------------------------------
    def write_muscle(self, m: MemoryRecord) -> str:
        """把肌肉记忆条目写成可加载技能，返回目录路径。"""
        title = m.title or m.id
        skill_dir = os.path.join(self.muscle_dir, _safe_name(title, m.id))
        os.makedirs(skill_dir, exist_ok=True)
        post = frontmatter.Post(
            m.content or "",
            name=_safe_name(title, m.id),
            description=(
                f"[rsi_mem muscle:{m.layer or 'sop'}] "
                f"{title[:120]}"
            ),
            triggers=", ".join(m.triggers),
            rsi_mem_id=m.id,
            rsi_mem_tier=m.tier,
            rsi_mem_status=m.status,
            rsi_mem_version=m.version,
        )
        path = os.path.join(skill_dir, "SKILL.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))
        return skill_dir

    def remove_muscle(self, memory_id: str, title: str) -> None:
        """按 id/标题清理镜像（条目归档或删除时调用，尽力而为）。"""
        for name in {
            _safe_name(title or "", memory_id),
            memory_id,
        }:
            target = os.path.join(self.muscle_dir, name)
            if os.path.isdir(target):
                sk = os.path.join(target, "SKILL.md")
                if os.path.isfile(sk):
                    os.remove(sk)
                try:
                    os.rmdir(target)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # 长期记忆 → 单文件
    # ------------------------------------------------------------------
    def write_longterm(self, m: MemoryRecord) -> str:
        name = _safe_name(f"{m.layer or 'L1'}-{m.title or m.id}", m.id)
        path = os.path.join(self.longterm_dir, f"{name}.md")
        header = "\n".join(
            [
                "---",
                f"id: {m.id}",
                f"layer: {m.layer}",
                f"kind: {m.kind}",
                f"status: {m.status}",
                f"confidence: {m.confidence:.2f}",
                "---",
            ],
        )
        body = (
            f"# {m.title or m.id}\n\n{m.content}\n"
            if m.title
            else f"{m.content}\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{header}\n\n{body}")
        return path

    def rebuild(self, memories: list[MemoryRecord]) -> None:
        """从 DB 记录全量重建镜像（短期记忆不镜像）。"""
        for m in memories:
            try:
                if m.tier == "muscle" and m.status in ("active", "candidate"):
                    self.write_muscle(m)
                elif m.tier == "long" and m.status in (
                    "active", "candidate", "promoted",
                ):
                    self.write_longterm(m)
            except OSError as e:
                logger.warning("mirror write failed for %s: %s", m.id, e)
