from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import DATA_DIR, DB_PATH
from .parse import Prompt

SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid        TEXT UNIQUE NOT NULL,
    session_id  TEXT NOT NULL,
    project     TEXT NOT NULL,
    cwd         TEXT NOT NULL,
    git_branch  TEXT,
    version     TEXT,
    timestamp   TEXT NOT NULL,
    content     TEXT NOT NULL,
    char_length INTEGER NOT NULL,
    parent_uuid TEXT,
    source_file TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prompts_session ON prompts(session_id);
CREATE INDEX IF NOT EXISTS idx_prompts_project ON prompts(project);
CREATE INDEX IF NOT EXISTS idx_prompts_timestamp ON prompts(timestamp);

CREATE TABLE IF NOT EXISTS embeddings (
    prompt_id   INTEGER PRIMARY KEY REFERENCES prompts(id),
    model_name  TEXT NOT NULL,
    vector      BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    prompt_id   INTEGER PRIMARY KEY REFERENCES prompts(id),
    cluster_id  INTEGER NOT NULL,
    run_id      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clusters_cluster ON clusters(cluster_id);

CREATE TABLE IF NOT EXISTS ingested_files (
    file_path    TEXT PRIMARY KEY,
    file_mtime   REAL NOT NULL,
    ingested_at  TEXT NOT NULL,
    prompt_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_labels (
    cluster_id  INTEGER PRIMARY KEY,
    run_id      TEXT NOT NULL,
    label       TEXT NOT NULL,
    top_terms   TEXT
);
"""


class PromptDB:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.init_schema()

    def init_schema(self):
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    # -- Ingestion tracking --

    def is_file_ingested(self, file_path: str, mtime: float) -> bool:
        row = self.conn.execute(
            "SELECT file_mtime FROM ingested_files WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        return row is not None and row["file_mtime"] == mtime

    def ingest_prompts(self, prompts: list[Prompt], source_file: str, mtime: float):
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        with self.conn:
            for p in prompts:
                try:
                    self.conn.execute(
                        """INSERT OR IGNORE INTO prompts
                           (uuid, session_id, project, cwd, git_branch, version,
                            timestamp, content, char_length, parent_uuid, source_file)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            p.uuid, p.session_id, p.project, p.cwd,
                            p.git_branch, p.version, p.timestamp,
                            p.content, p.char_length, p.parent_uuid, p.source_file,
                        ),
                    )
                    count += self.conn.execute("SELECT changes()").fetchone()[0]
                except sqlite3.IntegrityError:
                    pass
            self.conn.execute(
                """INSERT OR REPLACE INTO ingested_files
                   (file_path, file_mtime, ingested_at, prompt_count)
                   VALUES (?, ?, ?, ?)""",
                (source_file, mtime, now, count),
            )

    # -- Queries --

    def get_all_prompts(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM prompts ORDER BY timestamp"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_prompt_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]

    def get_prompt_by_id(self, prompt_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM prompts WHERE id = ?", (prompt_id,)
        ).fetchone()
        return dict(row) if row else None

    def search_text(self, query: str, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM prompts WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_history(
        self, project: str | None = None, limit: int = 50, since: str | None = None
    ) -> list[dict]:
        sql = "SELECT * FROM prompts WHERE 1=1"
        params: list = []
        if project:
            sql += " AND project LIKE ?"
            params.append(f"%{project}%")
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        c = self.conn
        return {
            "prompts": c.execute("SELECT COUNT(*) FROM prompts").fetchone()[0],
            "projects": c.execute("SELECT COUNT(DISTINCT project) FROM prompts").fetchone()[0],
            "sessions": c.execute("SELECT COUNT(DISTINCT session_id) FROM prompts").fetchone()[0],
            "clusters": c.execute(
                "SELECT COUNT(DISTINCT cluster_id) FROM clusters WHERE cluster_id != -1"
            ).fetchone()[0],
            "unclustered": c.execute(
                "SELECT COUNT(*) FROM clusters WHERE cluster_id = -1"
            ).fetchone()[0],
            "min_date": c.execute("SELECT MIN(timestamp) FROM prompts").fetchone()[0],
            "max_date": c.execute("SELECT MAX(timestamp) FROM prompts").fetchone()[0],
        }

    # -- Embeddings --

    def get_prompts_without_embeddings(self, model_name: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT p.* FROM prompts p
               LEFT JOIN embeddings e ON p.id = e.prompt_id AND e.model_name = ?
               WHERE e.prompt_id IS NULL
               ORDER BY p.id""",
            (model_name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def store_embeddings(
        self, prompt_ids: list[int], vectors: np.ndarray, model_name: str
    ):
        with self.conn:
            for pid, vec in zip(prompt_ids, vectors):
                self.conn.execute(
                    "INSERT OR REPLACE INTO embeddings (prompt_id, model_name, vector) VALUES (?, ?, ?)",
                    (pid, model_name, vec.astype(np.float32).tobytes()),
                )

    def get_all_embeddings(self, model_name: str) -> tuple[list[int], np.ndarray | None]:
        rows = self.conn.execute(
            "SELECT prompt_id, vector FROM embeddings WHERE model_name = ? ORDER BY prompt_id",
            (model_name,),
        ).fetchall()
        if not rows:
            return [], None
        ids = [r["prompt_id"] for r in rows]
        dim = len(rows[0]["vector"]) // 4  # float32 = 4 bytes
        vectors = np.array(
            [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
        )
        return ids, vectors

    # -- Clusters --

    def store_clusters(self, prompt_ids: list[int], labels: list[int], run_id: str):
        with self.conn:
            self.conn.execute("DELETE FROM clusters")
            for pid, label in zip(prompt_ids, labels):
                self.conn.execute(
                    "INSERT INTO clusters (prompt_id, cluster_id, run_id) VALUES (?, ?, ?)",
                    (pid, int(label), run_id),
                )

    def store_cluster_labels(self, labels: dict[int, str], run_id: str):
        with self.conn:
            self.conn.execute("DELETE FROM cluster_labels")
            for cid, label in labels.items():
                self.conn.execute(
                    "INSERT INTO cluster_labels (cluster_id, run_id, label) VALUES (?, ?, ?)",
                    (cid, run_id, label),
                )

    def get_cluster_summary(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT c.cluster_id, COUNT(*) as count,
                      COALESCE(cl.label, '') as label
               FROM clusters c
               LEFT JOIN cluster_labels cl ON c.cluster_id = cl.cluster_id
               WHERE c.cluster_id != -1
               GROUP BY c.cluster_id
               ORDER BY count DESC""",
        ).fetchall()
        return [dict(r) for r in rows]

    def get_cluster_prompts(self, cluster_id: int) -> list[dict]:
        rows = self.conn.execute(
            """SELECT p.* FROM prompts p
               JOIN clusters c ON p.id = c.prompt_id
               WHERE c.cluster_id = ?
               ORDER BY p.timestamp DESC""",
            (cluster_id,),
        ).fetchall()
        return [dict(r) for r in rows]
