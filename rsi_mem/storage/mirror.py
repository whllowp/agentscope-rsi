# -*- coding: utf-8 -*-
"""Markdown 镜像：把肌肉记忆/长期记忆写成人类可读（且可加载）的文件。

- 肌肉记忆 → ``mirror/muscle/<name>/SKILL.md``，frontmatter 兼容
  ``agentscope.skill.LocalSkillLoader`` —— SOP 因此可以同时作为 Agent
  技能被加载（dsh-memory-evolve 的技能自固化思路）。
- 长期记忆 L1/L2 → ``mirror/longterm/<id>.md``，纯人类阅读用。
- 长期记忆 wiki（U4）→ ``mirror/wiki/``：页间互链的主题页 +
  ``INDEX.md`` 顶部小索引；索引渲染器
  :meth:`MirrorStore.render_index_block` 同时服务注入路径（注入不依赖
  rebuild —— INDEX.md 首建前仍可从 DB 直渲染）。

镜像是 DB 的写穿副本，可随时删除重建（``rebuild`` / ``rebuild_wiki``）。
"""
from __future__ import annotations

import os
import re
import shutil

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
        self.wiki_dir = os.path.join(data_dir, "mirror", "wiki")
        os.makedirs(self.muscle_dir, exist_ok=True)
        os.makedirs(self.longterm_dir, exist_ok=True)
        os.makedirs(self.wiki_dir, exist_ok=True)

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

    # ------------------------------------------------------------------
    # 长期记忆 wiki（U4）：主题页 + 顶部小索引
    # ------------------------------------------------------------------
    _INDEX_HEADER = "## 记忆索引（知道什么、去哪找）"
    _INDEX_TAIL = "更多用 memory_search 检索"
    _MISC_TOPIC = "其他"

    def cluster_topics(
        self, memories: list[MemoryRecord],
    ) -> list[tuple[str, MemoryRecord | None, list[MemoryRecord]]]:
        """按 L2 锚点 + related 链接归簇（孤儿进「其他」页）。

        Returns:
            ``(主题名, 锚点记录或 None, 成员记录)`` 列表，按主题名排序
            （稳定键 —— 索引截断的确定性来源）。
        """
        anchors = sorted(
            (m for m in memories if m.layer == "L2_scenario"),
            key=lambda m: (m.title or "", m.id),
        )
        atoms = [m for m in memories if m.layer == "L1_atom"]
        clusters: list[
            tuple[str, MemoryRecord | None, list[MemoryRecord]]
        ] = []
        assigned: set[str] = set()
        for anchor in anchors:
            members = [anchor]
            for atom in atoms:
                if atom.id in assigned:
                    continue
                if any(
                    l.kind == "related" and l.target_id == anchor.id
                    for l in atom.links
                ):
                    members.append(atom)
                    assigned.add(atom.id)
            clusters.append((anchor.title or anchor.id[:8], anchor, members))
        misc = [a for a in atoms if a.id not in assigned]
        if misc:
            clusters.append((self._MISC_TOPIC, None, misc))
        clusters.sort(key=lambda c: c[0])
        return clusters

    def _page_filename(
        self, topic: str, anchor: MemoryRecord | None,
    ) -> str:
        """主题页文件名 = ``_safe_name(topic) + 锚点 id 前 8 位``。

        ``_safe_name`` 的 48 字符截断会让不同主题同名互覆——锚点 id
        后缀保证防撞；「其他」页无锚点，用固定 ``misc`` 后缀。
        """
        suffix = anchor.id[:8] if anchor is not None else "misc"
        return f"{_safe_name(topic, suffix)}-{suffix}.md"

    def _index_lines(
        self,
        memories: list[MemoryRecord],
        max_lines: int,
        max_chars: int,
    ) -> tuple[list[str], list[str]]:
        """预算内渲染索引主题行，返回 ``(行, 入选记录 id)``。

        行数/字节双上限对头行 + 主题行生效；行截断按主题名排序的
        稳定键做前缀截断（确定性）；尾行提示由调用方追加、不计入。
        """
        clusters = self.cluster_topics(memories)
        if not clusters:
            return [], []
        lines: list[str] = []
        ids: list[str] = []
        used = len(self._INDEX_HEADER.encode("utf-8"))
        lines_budget = max(0, max_lines - 1)  # 头行占一行
        for topic, anchor, members in clusters:
            if len(lines) >= lines_budget:
                break
            heads = "、".join(
                (m.title or m.id[:8]) for m in members[:2]
            )
            line = (
                f"- [{topic}](./{self._page_filename(topic, anchor)})"
                f"（{len(members)} 条）：{heads}"
            )
            cost = len(line.encode("utf-8"))
            if used + cost > max_chars:
                break  # 前缀截断（稳定键确定性）
            used += cost
            lines.append(line)
            ids.extend(m.id for m in members)
        return lines, ids

    def render_index_block(
        self,
        memories: list[MemoryRecord],
        max_lines: int,
        max_chars: int,
    ) -> str:
        """渲染顶部小索引（中间件/代理/INDEX.md 共用同一渲染器）。"""
        block, _ids = self.render_index_block_with_ids(
            memories, max_lines, max_chars,
        )
        return block

    def render_index_block_with_ids(
        self,
        memories: list[MemoryRecord],
        max_lines: int,
        max_chars: int,
    ) -> tuple[str, list[str]]:
        """同 :meth:`render_index_block`，并返回入选记录 id（供 ids
        回填轨迹 / settle）。无簇时返回空串（调用方跳过该段）。"""
        lines, ids = self._index_lines(memories, max_lines, max_chars)
        if not lines:
            return "", []
        block = "\n".join(
            [self._INDEX_HEADER, *lines, self._INDEX_TAIL],
        )
        return block, ids

    def rebuild_wiki(
        self,
        memories: list[MemoryRecord],
        *,
        index_max_lines: int,
        index_max_bytes: int,
    ) -> int:
        """全量重建 wiki 镜像（先清空 ``mirror/wiki``，幂等）。

        页面结构：L2_scenario 为主题锚点页，L1_atom 按 related 链接
        归簇（孤儿进「其他」页）；条目摘要 + 相对链接 + 交叉引用；
        仅收 status ∈ active/promoted 的调用方过滤结果（断链有效性由
        U3 repair 后重建保证）。返回写入的主题页数。
        """
        # —— 先清空（幂等）——
        try:
            for name in os.listdir(self.wiki_dir):
                path = os.path.join(self.wiki_dir, name)
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        except OSError as e:  # 尽力而为，镜像失败不阻塞主链路
            logger.warning("wiki clean failed: %s", e)

        by_id = {m.id: m for m in memories}
        clusters = self.cluster_topics(memories)
        page_of: dict[str, str] = {}
        for topic, anchor, members in clusters:
            fname = self._page_filename(topic, anchor)
            for m in members:
                page_of[m.id] = fname

        written = 0
        for topic, anchor, members in clusters:
            fname = self._page_filename(topic, anchor)
            lines = [f"# {topic}", "", "[返回索引](./INDEX.md)", ""]
            if anchor is not None:
                lines.append(
                    f"- **{anchor.title or anchor.id[:8]}**："
                    f"{anchor.content[:200]}"
                )
            for m in members:
                if anchor is not None and m.id == anchor.id:
                    continue
                refs = []
                for link in m.links:
                    if link.kind != "related":
                        continue
                    target = by_id.get(link.target_id)
                    target_page = page_of.get(link.target_id)
                    if (
                        target is None
                        or target_page is None
                        or target_page == fname
                    ):
                        continue
                    refs.append(
                        f"[↔ {target.title or target.id[:8]}]"
                        f"(./{target_page})"
                    )
                cross = (" " + " ".join(refs)) if refs else ""
                lines.append(
                    f"- **{m.title or m.id[:8]}**：{m.content[:160]}{cross}"
                )
            path = os.path.join(self.wiki_dir, fname)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                written += 1
            except OSError as e:
                logger.warning("wiki page write failed: %s", e)

        index_text = self.render_index_block(
            memories, index_max_lines, index_max_bytes,
        )
        if index_text:
            try:
                with open(
                    os.path.join(self.wiki_dir, "INDEX.md"), "w",
                    encoding="utf-8",
                ) as f:
                    f.write(index_text + "\n")
            except OSError as e:
                logger.warning("wiki index write failed: %s", e)
        return written

    def remove_longterm(self, m: MemoryRecord) -> None:
        """按 write_longterm 同名规则删除镜像文件（降层/归档时调用）。

        必须在记录 layer 变更**之前**调用（文件名含旧层前缀）；
        尽力而为，失败只告警。
        """
        name = _safe_name(f"{m.layer or 'L1'}-{m.title or m.id}", m.id)
        path = os.path.join(self.longterm_dir, f"{name}.md")
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as e:
            logger.warning("mirror remove failed for %s: %s", m.id, e)

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
