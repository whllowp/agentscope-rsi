# -*- coding: utf-8 -*-
"""SQLite 存储层：三层记忆、RSI 轨迹、基因组版本与待确认队列。

零新增第三方依赖：标准库 ``sqlite3``（WAL 模式）+ ``asyncio.to_thread``
异步化。全文检索用 FTS5 ``trigram`` 分词器（中英文子串均可命中，
Python 自带 SQLite ≥ 3.34 支持）；探测失败时自动降级为 ``LIKE`` 扫描。

单个 ``RSIMemDatabase`` 实例可被多个中间件共享（连接由线程锁保护），
所有异步方法都是并发安全的。
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from typing import Any, Iterable, Sequence

from agentscope._logging import logger

from .models import (
    GenomeSpec,
    MemoryRecord,
    PendingReview,
    TraceRecord,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    layer TEXT,
    kind TEXT NOT NULL DEFAULT 'observation',
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    triggers TEXT NOT NULL DEFAULT '[]',
    scope TEXT NOT NULL DEFAULT 'global',
    session_id TEXT,
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    links TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT '{}',
    use_count INTEGER NOT NULL DEFAULT 0,
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    expires_at REAL,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_memories_tier_status ON memories(tier, status);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at);

CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    query TEXT NOT NULL DEFAULT '',
    digest TEXT NOT NULL DEFAULT '',
    started_at REAL NOT NULL,
    finished_at REAL,
    tool_calls TEXT NOT NULL DEFAULT '[]',
    model_calls TEXT NOT NULL DEFAULT '[]',
    permissions TEXT NOT NULL DEFAULT '[]',
    outcome TEXT,
    used_memory_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_traces_started ON traces(started_at);

CREATE TABLE IF NOT EXISTS genomes (
    version INTEGER PRIMARY KEY,
    spec TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_reviews (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    memory_id TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    reason TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL,
    resolved_at REAL,
    resolved_by TEXT
);

CREATE TABLE IF NOT EXISTS vectors (
    memory_id TEXT PRIMARY KEY,
    vec TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_FTS_TRIGRAM = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
    "title, content, content='memories', content_rowid='rowid', "
    "tokenize='trigram')"
)
_FTS_PLAIN = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
    "title, content, content='memories', content_rowid='rowid')"
)
_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, title, content)
  VALUES (new.rowid, new.title, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, title, content)
  VALUES ('delete', old.rowid, old.title, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, title, content)
  VALUES ('delete', old.rowid, old.title, old.content);
  INSERT INTO memories_fts(rowid, title, content)
  VALUES (new.rowid, new.title, new.content);
END;
"""


def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        tier=row["tier"],
        layer=row["layer"],
        kind=row["kind"],
        title=row["title"],
        content=row["content"],
        triggers=json.loads(row["triggers"]),
        scope=row["scope"],
        session_id=row["session_id"],
        confidence=row["confidence"],
        status=row["status"],
        links=json.loads(row["links"]),
        source=json.loads(row["source"]),
        use_count=row["use_count"],
        hit_count=row["hit_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        expires_at=row["expires_at"],
        version=row["version"],
    )


def _row_to_trace(row: sqlite3.Row) -> TraceRecord:
    from ..storage.models import (  # noqa: PLC0415
        ModelCallStat,
        OutcomeRecord,
        PermissionEvent,
        ToolCallNode,
    )

    return TraceRecord(
        id=row["id"],
        session_id=row["session_id"],
        query=row["query"],
        digest=row["digest"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        tool_calls=[
            ToolCallNode(**c) for c in json.loads(row["tool_calls"])
        ],
        model_calls=[
            ModelCallStat(**c) for c in json.loads(row["model_calls"])
        ],
        permissions=[
            PermissionEvent(**p) for p in json.loads(row["permissions"])
        ],
        outcome=(
            OutcomeRecord(**json.loads(row["outcome"]))
            if row["outcome"]
            else None
        ),
        used_memory_ids=json.loads(row["used_memory_ids"]),
    )


def _row_to_review(row: sqlite3.Row) -> PendingReview:
    return PendingReview(
        id=row["id"],
        action=row["action"],  # type: ignore[arg-type]
        memory_id=row["memory_id"],
        payload=json.loads(row["payload"]),
        reason=row["reason"],
        confidence=row["confidence"],
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
        resolved_by=row["resolved_by"],  # type: ignore[arg-type]
    )


class RSIMemDatabase:
    """``rsi_mem`` 的唯一持久化入口（一个数据目录一个实例）。

    所有公开方法都是 ``async``；内部经 ``asyncio.to_thread`` 操作单个
    SQLite 连接（WAL + 线程锁），因此一个实例可以被记忆中间件、RSI
    中间件、HTTP 代理并发共享。
    """

    def __init__(self, path: str) -> None:
        self._path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._fts_available = True
        self._init_schema()

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            try:
                self._conn.execute(_FTS_TRIGRAM)
                self._conn.executescript(_FTS_TRIGGERS)
            except sqlite3.OperationalError as e:
                # trigram 不可用（SQLite < 3.34）：退回 unicode61，再不行
                # 就整体放弃 FTS，检索层会用 LIKE 兜底。
                logger.warning(
                    "FTS5 trigram unavailable (%s); trying plain FTS5, "
                    "then LIKE fallback in retrieval.",
                    e,
                )
                try:
                    self._conn.execute(_FTS_PLAIN)
                    self._conn.executescript(_FTS_TRIGGERS)
                except sqlite3.OperationalError:
                    self._fts_available = False

    async def _run(self, fn, *args, **kwargs):
        """在线程池里执行同步数据库操作。"""
        def _wrapped() -> Any:
            with self._lock:
                return fn(*args, **kwargs)

        return await asyncio.to_thread(_wrapped)

    async def close(self) -> None:
        await self._run(self._conn.close)

    @property
    def path(self) -> str:
        return self._path

    # ------------------------------------------------------------------
    # 记忆 CRUD
    # ------------------------------------------------------------------
    def _insert_memory(self, m: MemoryRecord) -> MemoryRecord:
        self._conn.execute(
            "INSERT INTO memories (id, tier, layer, kind, title, content,"
            " triggers, scope, session_id, confidence, status, links,"
            " source, use_count, hit_count, created_at, updated_at,"
            " expires_at, version)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                m.id, m.tier, m.layer, m.kind, m.title, m.content,
                json.dumps(m.triggers, ensure_ascii=False),
                m.scope, m.session_id, m.confidence, m.status,
                json.dumps([l.model_dump() for l in m.links]),
                json.dumps(m.source, ensure_ascii=False),
                m.use_count, m.hit_count, m.created_at, m.updated_at,
                m.expires_at, m.version,
            ),
        )
        return m

    async def insert_memory(self, m: MemoryRecord) -> MemoryRecord:
        return await self._run(self._insert_memory, m)

    def _update_memory(self, m: MemoryRecord) -> None:
        self._conn.execute(
            "UPDATE memories SET tier=?, layer=?, kind=?, title=?,"
            " content=?, triggers=?, scope=?, session_id=?, confidence=?,"
            " status=?, links=?, source=?, use_count=?, hit_count=?,"
            " updated_at=?, expires_at=?, version=? WHERE id=?",
            (
                m.tier, m.layer, m.kind, m.title, m.content,
                json.dumps(m.triggers, ensure_ascii=False),
                m.scope, m.session_id, m.confidence, m.status,
                json.dumps([l.model_dump() for l in m.links]),
                json.dumps(m.source, ensure_ascii=False),
                m.use_count, m.hit_count, m.updated_at, m.expires_at,
                m.version, m.id,
            ),
        )

    async def update_memory(self, m: MemoryRecord) -> None:
        await self._run(self._update_memory, m)

    def _get_memory(self, memory_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=?", (memory_id,),
        ).fetchone()
        return _row_to_memory(row) if row else None

    async def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return await self._run(self._get_memory, memory_id)

    def _list_memories_impl(
        self,
        tier: str | None,
        status: str | None,
        session_id: str | None,
        layer: str | None,
        scope: str | None,
        limit: int,
    ) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list[Any] = []
        if tier is not None:
            sql += " AND tier=?"
            params.append(tier)
        if status is not None:
            states = [s.strip() for s in status.split(",") if s.strip()]
            sql += " AND status IN (%s)" % ",".join("?" * len(states))
            params.extend(states)
        if session_id is not None:
            sql += " AND session_id=?"
            params.append(session_id)
        if layer is not None:
            sql += " AND layer=?"
            params.append(layer)
        if scope is not None:
            sql += " AND scope=?"
            params.append(scope)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_memory(r) for r in rows]

    async def list_memories(
        self,
        *,
        tier: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
        layer: str | None = None,
        scope: str | None = None,
        limit: int = 200,
    ) -> list[MemoryRecord]:
        return await self._run(
            self._list_memories_impl,
            tier, status, session_id, layer, scope, limit,
        )

    def _delete_memory(self, memory_id: str) -> None:
        self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self._conn.execute(
            "DELETE FROM vectors WHERE memory_id=?", (memory_id,),
        )
        # FTS 触发器随 DELETE 自动清理全文索引。

    async def delete_memory(self, memory_id: str) -> None:
        await self._run(self._delete_memory, memory_id)

    def _expire_short(self, now: float) -> int:
        cur = self._conn.execute(
            "DELETE FROM memories WHERE tier='short' AND expires_at"
            " IS NOT NULL AND expires_at < ?",
            (now,),
        )
        return cur.rowcount or 0

    async def expire_short_term(self, now: float) -> int:
        """清理过期短期记忆，返回删除条数。"""
        return await self._run(self._expire_short, now)

    def _touch_hits(self, memory_ids: Sequence[str]) -> None:
        for mid in memory_ids:
            self._conn.execute(
                "UPDATE memories SET hit_count=hit_count+1,"
                " use_count=use_count+1 WHERE id=?",
                (mid,),
            )

    async def touch_hits(self, memory_ids: Iterable[str]) -> None:
        ids = list(memory_ids)
        if ids:
            await self._run(self._touch_hits, ids)

    # ------------------------------------------------------------------
    # 全文检索（FTS5，bm25 排序；查询过短/FTS 缺失时 LIKE 兜底）
    # ------------------------------------------------------------------
    @staticmethod
    def _match_expr(query: str) -> str | None:
        """把自由文本转成安全的 FTS MATCH 表达式；过短返回 None。

        中文词转成滑窗 trigram 后 OR（提升部分匹配召回）；拉丁词按
        整词短语匹配。
        """
        def _is_cjk(ch: str) -> bool:
            return "\u4e00" <= ch <= "\u9fff"

        terms = [t for t in query.split() if t]
        quoted: list[str] = []
        for term in terms:
            if len(term) < 3:
                continue  # 交给 LIKE 兜底
            if any(_is_cjk(c) for c in term):
                grams = [
                    term[i:i + 3] for i in range(len(term) - 2)
                ]
                quoted.extend(
                    '"%s"' % g.replace('"', '""') for g in grams
                )
            else:
                quoted.append('"%s"' % term.replace('"', '""'))
        if not quoted:
            return None
        return " OR ".join(quoted)

    def _fts_search(
        self,
        match_expr: str,
        tiers: Sequence[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        """返回 (memory_id, score) 列表，score ∈ (0,1]，越大越相关。"""
        sql = (
            "SELECT m.id, bm25(memories_fts) AS rank FROM memories m"
            " JOIN memories_fts ON memories_fts.rowid = m.rowid"
            " WHERE memories_fts MATCH ?"
        )
        params: list[Any] = [match_expr]
        if tiers:
            sql += " AND m.tier IN (%s)" % ",".join("?" * len(tiers))
            params.extend(tiers)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            (r["id"], 1.0 / (1.0 + max(0.0, r["rank"])))
            for r in rows
        ]

    def _like_search(
        self,
        query: str,
        tiers: Sequence[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        """LIKE 兜底：任一分词子串命中即召回，按命中数粗排。"""
        terms = [t for t in query.split() if t][:5]
        if not terms:
            return []
        rows = self._conn.execute(
            "SELECT id, title, content, tier FROM memories"
            " WHERE status != 'archived'"
        ).fetchall()
        scored: list[tuple[str, float]] = []
        for r in rows:
            if tiers and r["tier"] not in tiers:
                continue
            text = (r["title"] or "") + "\n" + (r["content"] or "")
            text_l = text.casefold()
            hits = sum(
                1 for t in terms if t.casefold() in text_l
            )
            if hits:
                scored.append((r["id"], 0.3 * hits / len(terms)))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    async def search_text(
        self,
        query: str,
        *,
        tiers: Sequence[str] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """混合文本检索：优先 FTS5 bm25，空结果/异常时 LIKE 兜底。"""
        expr = self._match_expr(query)
        if expr is not None and self._fts_available:
            try:
                hits = await self._run(self._fts_search, expr, tiers, limit)
                if hits:
                    return hits
            except sqlite3.OperationalError as e:
                logger.warning("FTS search failed (%s); LIKE fallback.", e)
        return await self._run(self._like_search, query, tiers, limit)

    # ------------------------------------------------------------------
    # 向量表（可选向量检索）
    # ------------------------------------------------------------------
    def _upsert_vector(self, memory_id: str, vec: Sequence[float]) -> None:
        self._conn.execute(
            "INSERT INTO vectors (memory_id, vec) VALUES (?,?)"
            " ON CONFLICT(memory_id) DO UPDATE SET vec=excluded.vec",
            (memory_id, json.dumps(list(vec))),
        )

    async def upsert_vector(
        self, memory_id: str, vec: Sequence[float],
    ) -> None:
        await self._run(self._upsert_vector, memory_id, vec)

    def _load_vectors(
        self, tiers: Sequence[str] | None,
    ) -> list[tuple[str, list[float]]]:
        sql = (
            "SELECT v.memory_id, v.vec FROM vectors v"
            " JOIN memories m ON m.id = v.memory_id"
            " WHERE m.status IN ('active','candidate')"
        )
        params: list[Any] = []
        if tiers:
            sql += " AND m.tier IN (%s)" % ",".join("?" * len(tiers))
            params.extend(tiers)
        out = []
        for r in self._conn.execute(sql, params).fetchall():
            out.append((r["memory_id"], json.loads(r["vec"])))
        return out

    async def load_vectors(
        self, tiers: Sequence[str] | None = None,
    ) -> list[tuple[str, list[float]]]:
        return await self._run(self._load_vectors, tiers)

    # ------------------------------------------------------------------
    # 轨迹
    # ------------------------------------------------------------------
    def _insert_trace(self, t: TraceRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO traces (id, session_id, query, digest,"
            " started_at, finished_at, tool_calls, model_calls,"
            " permissions, outcome, used_memory_ids)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                t.id, t.session_id, t.query, t.digest, t.started_at,
                t.finished_at,
                json.dumps([c.model_dump() for c in t.tool_calls]),
                json.dumps([c.model_dump() for c in t.model_calls]),
                json.dumps([p.model_dump() for p in t.permissions]),
                (
                    json.dumps(t.outcome.model_dump())
                    if t.outcome else None
                ),
                json.dumps(t.used_memory_ids),
            ),
        )

    async def upsert_trace(self, t: TraceRecord) -> None:
        await self._run(self._insert_trace, t)

    def _list_traces(self, limit: int, since: float | None) -> list:
        sql = "SELECT * FROM traces"
        params: list[Any] = []
        if since is not None:
            sql += " WHERE started_at >= ?"
            params.append(since)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return [_row_to_trace(r) for r in self._conn.execute(sql, params)]

    async def list_traces(
        self, limit: int = 50, since: float | None = None,
    ) -> list[TraceRecord]:
        return await self._run(self._list_traces, limit, since)

    async def count_traces_since(self, since: float) -> int:
        def _count() -> int:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM traces WHERE started_at >= ?",
                (since,),
            ).fetchone()
            return int(row["c"])

        return await self._run(_count)

    # ------------------------------------------------------------------
    # 基因组
    # ------------------------------------------------------------------
    def _save_genome(self, g: GenomeSpec) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO genomes (version, spec, created_at)"
            " VALUES (?,?,?)",
            (g.version, g.model_dump_json(), g.created_at),
        )

    async def save_genome(self, g: GenomeSpec) -> None:
        await self._run(self._save_genome, g)

    def _load_genome(self, version: int) -> GenomeSpec | None:
        row = self._conn.execute(
            "SELECT spec FROM genomes WHERE version=?", (version,),
        ).fetchone()
        return GenomeSpec.model_validate_json(row["spec"]) if row else None

    async def load_genome(self, version: int) -> GenomeSpec | None:
        return await self._run(self._load_genome, version)

    def _active_version(self) -> int:
        row = self._conn.execute(
            "SELECT MAX(version) AS v FROM genomes",
        ).fetchone()
        return int(row["v"]) if row and row["v"] is not None else -1

    async def active_version(self) -> int:
        return await self._run(self._active_version)

    async def active_genome(self) -> GenomeSpec:
        """当前激活基因组；空库时落一份种子并返回。"""
        version = await self.active_version()
        if version < 0:
            seed = GenomeSpec.seed()
            await self.save_genome(seed)
            return seed
        g = await self.load_genome(version)
        assert g is not None
        return g

    def _list_genomes(self) -> list[GenomeSpec]:
        rows = self._conn.execute(
            "SELECT spec FROM genomes ORDER BY version DESC LIMIT 50",
        ).fetchall()
        return [GenomeSpec.model_validate_json(r["spec"]) for r in rows]

    async def list_genomes(self) -> list[GenomeSpec]:
        return await self._run(self._list_genomes)

    # ------------------------------------------------------------------
    # 待确认队列
    # ------------------------------------------------------------------
    def _enqueue_review(self, r: PendingReview) -> PendingReview:
        self._conn.execute(
            "INSERT INTO pending_reviews (id, action, memory_id, payload,"
            " reason, confidence, created_at, resolved_at, resolved_by)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                r.id, r.action, r.memory_id,
                json.dumps(r.payload, ensure_ascii=False),
                r.reason, r.confidence, r.created_at,
                r.resolved_at, r.resolved_by,
            ),
        )
        return r

    async def enqueue_review(self, r: PendingReview) -> PendingReview:
        return await self._run(self._enqueue_review, r)

    def _pending_reviews(self, limit: int) -> list[PendingReview]:
        rows = self._conn.execute(
            "SELECT * FROM pending_reviews WHERE resolved_at IS NULL"
            " ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_review(r) for r in rows]

    async def pending_reviews(self, limit: int = 20) -> list[PendingReview]:
        return await self._run(self._pending_reviews, limit)

    def _get_review(self, review_id: str) -> PendingReview | None:
        row = self._conn.execute(
            "SELECT * FROM pending_reviews WHERE id=?", (review_id,),
        ).fetchone()
        return _row_to_review(row) if row else None

    async def get_review(self, review_id: str) -> PendingReview | None:
        return await self._run(self._get_review, review_id)

    def _resolve_review(
        self, review_id: str, by: str, now: float,
    ) -> None:
        self._conn.execute(
            "UPDATE pending_reviews SET resolved_at=?, resolved_by=?"
            " WHERE id=?",
            (now, by, review_id),
        )

    async def resolve_review(
        self, review_id: str, by: str, now: float,
    ) -> None:
        await self._run(self._resolve_review, review_id, by, now)

    # ------------------------------------------------------------------
    # meta kv
    # ------------------------------------------------------------------
    async def get_meta(self, key: str) -> str | None:
        def _get() -> str | None:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key=?", (key,),
            ).fetchone()
            return row["value"] if row else None

        return await self._run(_get)

    async def set_meta(self, key: str, value: str) -> None:
        def _set() -> None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

        await self._run(_set)

    async def stats(self) -> dict[str, Any]:
        """给 rsi_status / memory_status 用的一览统计。"""
        def _stats() -> dict[str, Any]:
            out: dict[str, Any] = {}
            for tier in ("muscle", "short", "long"):
                row = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM memories WHERE tier=?"
                    " AND status IN ('active','candidate')",
                    (tier,),
                ).fetchone()
                out[tier] = int(row["c"])
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM traces",
            ).fetchone()
            out["traces"] = int(row["c"])
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM pending_reviews"
                " WHERE resolved_at IS NULL",
            ).fetchone()
            out["pending_reviews"] = int(row["c"])
            row = self._conn.execute(
                "SELECT MAX(version) AS v FROM genomes",
            ).fetchone()
            out["genome_version"] = (
                int(row["v"]) if row and row["v"] is not None else -1
            )
            return out

        return await self._run(_stats)
