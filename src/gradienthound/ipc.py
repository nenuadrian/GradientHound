from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path


class IPCChannel:
    """SQLite-backed IPC between training capture and dashboard consumers.

    Replaces the previous file-based JSONL implementation with a single
    SQLite database.  Uses WAL mode so the training process can write
    while the dashboard reads concurrently.

    Public API is unchanged -- every ``read_*`` / ``append_*`` / ``write_*``
    method keeps its original signature.  Read methods gain optional
    keyword-only filters (``step_min``, ``step_max``, ``model``, ``last_n``)
    that push filtering into SQL for better performance on large runs.
    """

    _DB_NAME = "gradienthound.db"

    def __init__(self, directory: Path | str | None = None) -> None:
        if directory is None:
            self._dir = Path(tempfile.mkdtemp(prefix="gradienthound_"))
            self._owns_dir = True
        else:
            self._dir = Path(directory)
            self._dir.mkdir(parents=True, exist_ok=True)
            self._owns_dir = False

        self._db_path = self._dir / self._DB_NAME
        self._conn: sqlite3.Connection | None = None
        self._conn_lock = threading.Lock()
        self._ensure_schema()

    # ── Connection management ────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            with self._conn_lock:
                if self._conn is None:  # double-checked locking
                    conn = sqlite3.connect(
                        str(self._db_path),
                        check_same_thread=False,
                        timeout=10,
                    )
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA busy_timeout=5000")
                    self._conn = conn
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                channel   TEXT NOT NULL,
                step      INTEGER,
                model     TEXT,
                layer     TEXT,
                timestamp REAL,
                data      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_channel
                ON events(channel);
            CREATE INDEX IF NOT EXISTS idx_events_channel_step
                ON events(channel, step);
            CREATE INDEX IF NOT EXISTS idx_events_channel_model
                ON events(channel, model);
            CREATE INDEX IF NOT EXISTS idx_events_channel_model_step
                ON events(channel, model, step);

            CREATE TABLE IF NOT EXISTS requests (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS responses (
                request_id TEXT PRIMARY KEY,
                data       TEXT NOT NULL
            );
        """)
        conn.commit()

    @property
    def directory(self) -> Path:
        return self._dir

    # ── Key-value helpers (metadata, models, optimizers) ─────────────

    def _kv_write(self, key: str, data: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
            (key, json.dumps(data)),
        )
        conn.commit()

    def _kv_read(self, key: str) -> dict:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key = ?", (key,),
        ).fetchone()
        if row is None:
            return {}
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return {}

    # ── Event append / read ──────────────────────────────────────────

    def _append_events(self, channel: str, entries: list[dict]) -> None:
        if not entries:
            return
        conn = self._get_conn()
        rows = []
        for entry in entries:
            rows.append((
                channel,
                entry.get("step") or entry.get("_step"),
                entry.get("model"),
                entry.get("layer"),
                entry.get("_timestamp"),
                json.dumps(entry),
            ))
        conn.executemany(
            "INSERT INTO events (channel, step, model, layer, timestamp, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()

    def _read_events(
        self,
        channel: str,
        *,
        step_min: int | None = None,
        step_max: int | None = None,
        model: str | None = None,
        layer: str | None = None,
        last_n: int | None = None,
    ) -> list[dict]:
        """Read events for *channel* with optional SQL-level filters.

        Parameters
        ----------
        step_min, step_max : int, optional
            Inclusive step range.
        model : str, optional
            Filter to a single model name.
        layer : str, optional
            Filter to a single layer name.
        last_n : int, optional
            Return only the last *N* rows (by insertion order).
        """
        conn = self._get_conn()
        clauses = ["channel = ?"]
        params: list = [channel]

        if step_min is not None:
            clauses.append("step >= ?")
            params.append(step_min)
        if step_max is not None:
            clauses.append("step <= ?")
            params.append(step_max)
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        if layer is not None:
            clauses.append("layer = ?")
            params.append(layer)

        where = " AND ".join(clauses)

        if last_n is not None:
            # Sub-select to get last N rows, then re-order ascending.
            sql = (
                f"SELECT data FROM ("
                f"  SELECT data, id FROM events WHERE {where} "
                f"  ORDER BY id DESC LIMIT ?"
                f") ORDER BY id ASC"
            )
            params.append(last_n)
        else:
            sql = f"SELECT data FROM events WHERE {where} ORDER BY id ASC"

        rows = conn.execute(sql, params).fetchall()
        result = []
        for (data_json,) in rows:
            try:
                result.append(json.loads(data_json))
            except (json.JSONDecodeError, TypeError):
                continue
        return result

    def _max_step(self, channel: str, *, model: str | None = None) -> int | None:
        """Return the highest step value for *channel*, or None if empty."""
        conn = self._get_conn()
        if model is not None:
            row = conn.execute(
                "SELECT MAX(step) FROM events WHERE channel = ? AND model = ?",
                (channel, model),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT MAX(step) FROM events WHERE channel = ?",
                (channel,),
            ).fetchone()
        if row and row[0] is not None:
            return int(row[0])
        return None

    def _count_events(self, channel: str) -> int:
        """Return total number of events in *channel*."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE channel = ?", (channel,),
        ).fetchone()
        return row[0] if row else 0

    # ── Metadata ─────────────────────────────────────────────────────

    def write_metadata(self, metadata: dict) -> None:
        self._kv_write("metadata", metadata)

    def read_metadata(self) -> dict:
        return self._kv_read("metadata")

    # ── Models ───────────────────────────────────────────────────────

    def write_models(self, models: dict[str, dict]) -> None:
        self._kv_write("models", models)

    def read_models(self) -> dict[str, dict]:
        return self._kv_read("models")

    # ── Optimizers ───────────────────────────────────────────────────

    def write_optimizers(self, optimizers: dict[str, dict]) -> None:
        self._kv_write("optimizers", optimizers)

    def read_optimizers(self) -> dict[str, dict]:
        return self._kv_read("optimizers")

    # ── Metrics (wandb scalars) ──────────────────────────────────────

    def append_metrics(self, entry: dict) -> None:
        self._append_events("metrics", [entry])

    def read_metrics(
        self,
        *,
        step_min: int | None = None,
        step_max: int | None = None,
        last_n: int | None = None,
    ) -> list[dict]:
        return self._read_events(
            "metrics", step_min=step_min, step_max=step_max, last_n=last_n,
        )

    # ── Gradient stats ───────────────────────────────────────────────

    def append_gradient_stats(self, entries: list[dict]) -> None:
        self._append_events("gradient_stats", entries)

    def read_gradient_stats(
        self,
        *,
        step_min: int | None = None,
        step_max: int | None = None,
        model: str | None = None,
        layer: str | None = None,
        last_n: int | None = None,
    ) -> list[dict]:
        return self._read_events(
            "gradient_stats",
            step_min=step_min, step_max=step_max,
            model=model, layer=layer, last_n=last_n,
        )

    # ── Weight stats ─────────────────────────────────────────────────

    def append_weight_stats(self, entries: list[dict]) -> None:
        self._append_events("weight_stats", entries)

    def read_weight_stats(
        self,
        *,
        step_min: int | None = None,
        step_max: int | None = None,
        model: str | None = None,
        layer: str | None = None,
        last_n: int | None = None,
    ) -> list[dict]:
        return self._read_events(
            "weight_stats",
            step_min=step_min, step_max=step_max,
            model=model, layer=layer, last_n=last_n,
        )

    # ── Activation stats ─────────────────────────────────────────────

    def append_activation_stats(self, entries: list[dict]) -> None:
        self._append_events("activation_stats", entries)

    def read_activation_stats(
        self,
        *,
        step_min: int | None = None,
        step_max: int | None = None,
        model: str | None = None,
        layer: str | None = None,
        last_n: int | None = None,
    ) -> list[dict]:
        return self._read_events(
            "activation_stats",
            step_min=step_min, step_max=step_max,
            model=model, layer=layer, last_n=last_n,
        )

    # ── Predictions (value calibration) ──────────────────────────────

    def append_predictions(self, entries: list[dict]) -> None:
        self._append_events("predictions", entries)

    def read_predictions(
        self,
        *,
        step_min: int | None = None,
        step_max: int | None = None,
        last_n: int | None = None,
    ) -> list[dict]:
        return self._read_events(
            "predictions", step_min=step_min, step_max=step_max, last_n=last_n,
        )

    # ── Attention patterns ───────────────────────────────────────────

    def append_attention(self, entries: list[dict]) -> None:
        self._append_events("attention", entries)

    def read_attention(
        self,
        *,
        step_min: int | None = None,
        step_max: int | None = None,
        last_n: int | None = None,
    ) -> list[dict]:
        return self._read_events(
            "attention", step_min=step_min, step_max=step_max, last_n=last_n,
        )

    # ── Optimizer state stats ────────────────────────────────────────

    def append_optimizer_state(self, entries: list[dict]) -> None:
        self._append_events("optimizer_state", entries)

    def read_optimizer_state(
        self,
        *,
        step_min: int | None = None,
        step_max: int | None = None,
        last_n: int | None = None,
    ) -> list[dict]:
        return self._read_events(
            "optimizer_state", step_min=step_min, step_max=step_max, last_n=last_n,
        )

    # ── On-demand requests / responses ───────────────────────────────

    def write_request(self, request: dict) -> None:
        """Queue a computation request for the training process."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO requests (data) VALUES (?)",
            (json.dumps(request),),
        )
        conn.commit()

    def read_requests(self) -> list[dict]:
        """Read all pending requests."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT data FROM requests ORDER BY id ASC",
        ).fetchall()
        result = []
        for (data_json,) in rows:
            try:
                result.append(json.loads(data_json))
            except (json.JSONDecodeError, TypeError):
                continue
        return result

    def clear_requests(self) -> None:
        """Remove all pending requests."""
        conn = self._get_conn()
        conn.execute("DELETE FROM requests")
        conn.commit()

    def write_response(self, request_id: str, data: dict) -> None:
        """Write a computation response."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO responses (request_id, data) VALUES (?, ?)",
            (request_id, json.dumps(data)),
        )
        conn.commit()

    def read_response(self, request_id: str) -> dict | None:
        """Read a response if available.  Returns None if not yet computed."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data FROM responses WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    def clear_response(self, request_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM responses WHERE request_id = ?",
            (request_id,),
        )
        conn.commit()

    # ── Cleanup ──────────────────────────────────────────────────────

    def cleanup(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._owns_dir and self._dir.exists():
            shutil.rmtree(self._dir, ignore_errors=True)
