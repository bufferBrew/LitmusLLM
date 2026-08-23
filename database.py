"""SQLite persistence layer for LitmusLLM.

Deliberately raw `sqlite3` rather than an ORM: the schema is small, the
queries are simple, and it keeps the dependency list (and the mental model)
short. Every public function opens a short-lived connection, which is the
safe pattern when writes arrive from a mix of asyncio tasks and threads.

Concurrency notes:
  * WAL journalling lets the dashboard read while an eval run writes.
  * `busy_timeout` makes concurrent writers wait rather than raise
    "database is locked" -- writes here are tiny and infrequent.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from config import DB_PATH, ensure_dirs

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
# This extends the four tables from the original spec with the columns the UI
# actually needs (live progress, judge model, per-case ordering, error text).
# Everything added is additive -- the original column names are unchanged.

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name        TEXT    NOT NULL,   -- display name, e.g. 'llama3.1:latest'
    model_type        TEXT    NOT NULL,   -- 'local' | 'cloud'
    model_id          TEXT    NOT NULL,   -- routable id, e.g. 'cloud:anthropic/claude-opus-5'
    metrics           TEXT    NOT NULL,   -- JSON array of metric keys
    dataset_id        INTEGER REFERENCES datasets(id),
    judge_model       TEXT,               -- model id used as the LLM judge
    status            TEXT    NOT NULL,   -- queued|running|completed|stopped|failed
    progress_done     INTEGER NOT NULL DEFAULT 0,
    progress_total    INTEGER NOT NULL DEFAULT 0,
    progress_note     TEXT,               -- human-readable "what's happening now"
    error             TEXT,
    comparison_run_id INTEGER REFERENCES comparison_runs(id),
    runtime           TEXT,               -- which local runtime served the model
    quantization      TEXT,               -- as served, e.g. 'Q4_K_M' -- see ADDED_COLUMNS
    perf_json         TEXT,               -- JSON: TTFT/throughput/token/cost measurements
    vram_bytes        INTEGER,            -- accelerator memory held while the run was live
    started_at        TIMESTAMP,
    completed_at      TIMESTAMP
);

CREATE TABLE IF NOT EXISTS eval_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_run_id     INTEGER NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    case_index      INTEGER NOT NULL,     -- position in the dataset, for stable ordering
    test_case_input TEXT,
    actual_output   TEXT,                 -- what the model under test produced
    metric_name     TEXT    NOT NULL,
    score           REAL,                 -- NULL when skipped or errored
    reason          TEXT,                 -- the judge's explanation
    passed          BOOLEAN,
    threshold       REAL,
    status          TEXT    NOT NULL DEFAULT 'scored',  -- scored|skipped|error
    created_at      TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    csv_path    TEXT,                     -- NULL for the built-in dataset
    is_builtin  BOOLEAN NOT NULL DEFAULT 0,
    row_count   INTEGER NOT NULL DEFAULT 0,
    uploaded_at TIMESTAMP
);

-- Uploaded rows live in SQLite (not just on disk) so a dataset stays readable
-- even if the CSV file is moved or deleted.
CREATE TABLE IF NOT EXISTS dataset_rows (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    row_index       INTEGER NOT NULL,
    input           TEXT    NOT NULL,
    expected_output TEXT,
    context         TEXT,                 -- JSON array of context chunks
    tools_called    TEXT,                 -- JSON array of tool names (optional)
    expected_tools  TEXT                  -- JSON array of tool names (optional)
);

CREATE TABLE IF NOT EXISTS comparison_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT,
    dataset_id   INTEGER REFERENCES datasets(id),
    metrics      TEXT    NOT NULL,        -- JSON array of metric keys
    model_ids    TEXT    NOT NULL,        -- JSON array of model ids
    judge_model  TEXT,
    mode         TEXT    NOT NULL DEFAULT 'sequential',  -- sequential|parallel
    status       TEXT    NOT NULL,
    error        TEXT,
    created_at   TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comparison_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison_run_id INTEGER NOT NULL REFERENCES comparison_runs(id) ON DELETE CASCADE,
    eval_run_id       INTEGER REFERENCES eval_runs(id),
    model_name        TEXT    NOT NULL,
    metric_name       TEXT    NOT NULL,
    average_score     REAL,
    pass_rate         REAL,
    scored_cases      INTEGER NOT NULL DEFAULT 0,
    rank              INTEGER
);

-- Published benchmark figures measured by third parties, NOT by LitmusLLM.
-- Kept in its own table (and its own UI panel) precisely so it can never be
-- confused with, or averaged into, a measured eval score.
CREATE TABLE IF NOT EXISTS benchmark_reference (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_label  TEXT    NOT NULL,
    variant      TEXT,                  -- reasoning-effort setting, if any
    vendor       TEXT,
    family       TEXT,                  -- Claude / Qwen / Llama / ...
    kind         TEXT    NOT NULL,      -- 'frontier' | 'open_weight'
    index_name   TEXT    NOT NULL,      -- which index this score belongs to
    score        REAL    NOT NULL,      -- 0-100, higher is better
    source_name  TEXT,
    source_url   TEXT,
    retrieved_at TEXT,                  -- when the figure was read off the source
    notes        TEXT,
    is_seed      BOOLEAN NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_bench_family ON benchmark_reference(family);
CREATE INDEX IF NOT EXISTS idx_results_run   ON eval_results(eval_run_id);
CREATE INDEX IF NOT EXISTS idx_rows_dataset  ON dataset_rows(dataset_id, row_index);
CREATE INDEX IF NOT EXISTS idx_runs_cmp      ON eval_runs(comparison_run_id);
CREATE INDEX IF NOT EXISTS idx_cmp_results   ON comparison_results(comparison_run_id);
"""


# Columns added after the schema first shipped. `CREATE TABLE IF NOT EXISTS`
# is a no-op on an existing table, so a database created by an earlier version
# would silently lack these and every INSERT naming them would fail. Applying
# them as idempotent ALTERs keeps existing runs -- and the history that makes
# the dashboard worth having -- rather than asking anyone to delete the file.
#
# Additive only, by design. A column here is safe to apply to a live database;
# anything that rewrites or drops data does not belong in this list.
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "eval_runs": {
        "runtime": "TEXT",
        "quantization": "TEXT",
        "perf_json": "TEXT",
        "vram_bytes": "INTEGER",
    },
}


def _apply_added_columns(conn: sqlite3.Connection) -> None:
    for table, columns in ADDED_COLUMNS.items():
        try:
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        except sqlite3.Error:
            continue          # table absent entirely; the schema script owns that
        if not existing:
            continue
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                log.info("Added column %s.%s to the existing database.", table, name)


def utcnow() -> str:
    """Timestamps are stored as UTC ISO-8601 strings -- sortable and portable."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Schema bootstrap and self-healing
# --------------------------------------------------------------------------
# The schema is normally created once, at startup. But the database is a
# single file that can vanish underneath a running process -- a stray `rm`, a
# cleanup script, a remounted Docker volume. `sqlite3.connect` then silently
# recreates an *empty* file, so every later query dies with "no such table"
# and the whole app 500s until somebody thinks to restart it.
#
# Rather than trust that startup ran, each connection cheaply confirms it is
# talking to the database we actually initialised. We remember the file's
# identity (device + inode); when a connection opens a file we do not
# recognise, we rebuild the schema and re-run the seed hooks before handing
# the connection over. Steady-state cost is one `os.stat` per connection,
# which is cheaper than the query that would otherwise have failed.

_schema_lock = threading.RLock()
_known_db: tuple[int, int] | None = None   # (st_dev, st_ino) we last set up
_repairing = False                         # guards against hook re-entry

RepairHook = Callable[[], Any]
_repair_hooks: list[RepairHook] = []


def register_repair_hook(hook: RepairHook) -> None:
    """Register a callable that re-seeds data whenever the schema is built.

    Hooks run on first boot and again after any repair, so they must be
    idempotent. Rebuilding tables gets the app answering again; the hooks are
    what make it *useful* again -- without them a healed database would come
    back with no built-in dataset and an empty reference page.
    """
    _repair_hooks.append(hook)


def forget_schema() -> None:
    """Drop our cached identity so the next connection re-bootstraps."""
    global _known_db
    with _schema_lock:
        _known_db = None


def _db_identity() -> tuple[int, int] | None:
    """Device+inode of the database file, or None if it is not there."""
    try:
        st = os.stat(DB_PATH)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the schema if `conn` points at a database we don't know."""
    global _known_db, _repairing

    identity = _db_identity()
    if identity is not None and identity == _known_db:
        return                      # fast path: the file we already set up

    with _schema_lock:
        if _repairing:
            return                  # a hook's own connection; schema is fine
        identity = _db_identity()
        if identity is not None and identity == _known_db:
            return                  # another thread repaired while we waited

        replaced = _known_db is not None
        conn.executescript(SCHEMA)
        _apply_added_columns(conn)
        conn.commit()
        _known_db = identity        # set before hooks run, so they don't recurse

        if replaced:
            log.warning(
                "Database file %s was replaced or removed while the app was "
                "running -- schema rebuilt in place. Anything stored in the old "
                "file is gone.", DB_PATH,
            )

        _repairing = True
        try:
            for hook in _repair_hooks:
                try:
                    hook()
                except Exception:   # a broken seeder must not block the repair
                    log.exception(
                        "Repair hook %s failed; continuing.",
                        getattr(hook, "__name__", hook),
                    )
        finally:
            _repairing = False


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Yield a configured connection and commit (or roll back) on exit."""
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    _ensure_schema(conn)
    try:
        yield conn
        conn.commit()
    except sqlite3.OperationalError as exc:
        conn.rollback()
        # The inode check below catches a *replaced* file, but not one that is
        # still there with its tables dropped or truncated in place. Treat a
        # "no such table" as proof our cached identity is stale so the next
        # connection rebuilds. This request still fails; the next one heals.
        if "no such table" in str(exc).lower():
            forget_schema()
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the schema. Idempotent -- safe on every boot.

    Opening a connection is all it takes: `get_conn` bootstraps the schema
    for any database file it does not already recognise, and a brand-new
    (or freshly emptied) file is exactly that.
    """
    with get_conn() as conn:
        conn.execute("SELECT 1")


def reap_interrupted_runs() -> int:
    """Mark runs left 'running' by a crash/restart as failed.

    Called on startup: an in-flight asyncio task cannot survive a process
    restart, so any row still claiming to be running is a lie.
    """
    with get_conn() as conn:
        cur = conn.execute(
            """UPDATE eval_runs
                  SET status='failed',
                      error=COALESCE(error, 'Interrupted -- the app restarted mid-run.'),
                      completed_at=?
                WHERE status IN ('running','queued')""",
            (utcnow(),),
        )
        conn.execute(
            """UPDATE comparison_runs
                  SET status='failed',
                      error=COALESCE(error, 'Interrupted -- the app restarted mid-run.'),
                      completed_at=?
                WHERE status IN ('running','queued')""",
            (utcnow(),),
        )
        return cur.rowcount


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

def create_dataset(
    name: str,
    rows: list[dict[str, Any]],
    csv_path: str | None = None,
    is_builtin: bool = False,
) -> int:
    """Insert a dataset plus its rows, returning the new dataset id."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO datasets (name, csv_path, is_builtin, row_count, uploaded_at)"
            " VALUES (?,?,?,?,?)",
            (name, csv_path, int(is_builtin), len(rows), utcnow()),
        )
        dataset_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO dataset_rows"
            " (dataset_id, row_index, input, expected_output, context, tools_called, expected_tools)"
            " VALUES (?,?,?,?,?,?,?)",
            [
                (
                    dataset_id,
                    i,
                    row["input"],
                    row.get("expected_output") or None,
                    json.dumps(row.get("context") or []),
                    json.dumps(row.get("tools_called") or []),
                    json.dumps(row.get("expected_tools") or []),
                )
                for i, row in enumerate(rows)
            ],
        )
        return dataset_id


def list_datasets() -> list[dict[str, Any]]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM datasets ORDER BY is_builtin DESC, id DESC"
        )]


def get_dataset(dataset_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return dict(row) if row else None


def get_dataset_rows(dataset_id: int) -> list[dict[str, Any]]:
    """Return dataset rows with JSON columns already decoded."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM dataset_rows WHERE dataset_id=? ORDER BY row_index",
            (dataset_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for key in ("context", "tools_called", "expected_tools"):
            try:
                d[key] = json.loads(d[key]) if d[key] else []
            except (TypeError, json.JSONDecodeError):
                d[key] = []
        out.append(d)
    return out


def delete_dataset(dataset_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM datasets WHERE id=? AND is_builtin=0", (dataset_id,))


def find_builtin_dataset() -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM datasets WHERE is_builtin=1 ORDER BY id LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


# --------------------------------------------------------------------------
# Eval runs
# --------------------------------------------------------------------------

def create_eval_run(
    *,
    model_name: str,
    model_type: str,
    model_id: str,
    metrics: list[str],
    dataset_id: int,
    judge_model: str | None,
    total: int,
    comparison_run_id: int | None = None,
    runtime: str | None = None,
    quantization: str | None = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO eval_runs
                 (model_name, model_type, model_id, metrics, dataset_id, judge_model,
                  status, progress_done, progress_total, progress_note,
                  comparison_run_id, runtime, quantization, started_at)
               VALUES (?,?,?,?,?,?,'queued',0,?,?,?,?,?,?)""",
            (
                model_name, model_type, model_id, json.dumps(metrics), dataset_id,
                judge_model, total, "Queued", comparison_run_id,
                runtime, quantization, utcnow(),
            ),
        )
        return int(cur.lastrowid)


def update_eval_run(run_id: int, **fields: Any) -> None:
    """Patch arbitrary columns on an eval run. Unknown keys are rejected loudly."""
    allowed = {
        "status", "progress_done", "progress_total", "progress_note",
        "error", "completed_at", "judge_model",
        "runtime", "quantization", "perf_json", "vram_bytes",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"update_eval_run: unknown column(s) {sorted(unknown)}")
    if not fields:
        return
    assignments = ", ".join(f"{k}=?" for k in fields)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE eval_runs SET {assignments} WHERE id=?",
            (*fields.values(), run_id),
        )


def record_result(
    *,
    eval_run_id: int,
    case_index: int,
    test_case_input: str,
    actual_output: str | None,
    metric_name: str,
    score: float | None,
    reason: str | None,
    passed: bool | None,
    threshold: float | None,
    status: str = "scored",
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO eval_results
                 (eval_run_id, case_index, test_case_input, actual_output, metric_name,
                  score, reason, passed, threshold, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                eval_run_id, case_index, test_case_input, actual_output, metric_name,
                score, reason,
                None if passed is None else int(passed),
                threshold, status, utcnow(),
            ),
        )


def get_eval_run(run_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT r.*, d.name AS dataset_name
                 FROM eval_runs r LEFT JOIN datasets d ON d.id = r.dataset_id
                WHERE r.id=?""",
            (run_id,),
        ).fetchone()
    if not row:
        return None
    run = dict(row)
    run["metrics"] = json.loads(run["metrics"])
    return run


def list_eval_runs(limit: int = 200, standalone_only: bool = False) -> list[dict[str, Any]]:
    """List runs newest-first, each annotated with its per-metric averages."""
    where = "WHERE r.comparison_run_id IS NULL" if standalone_only else ""
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT r.*, d.name AS dataset_name
                  FROM eval_runs r LEFT JOIN datasets d ON d.id = r.dataset_id
                  {where}
                 ORDER BY r.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        runs = []
        for row in rows:
            run = dict(row)
            run["metrics"] = json.loads(run["metrics"])
            run["summary"] = _summarise(conn, run["id"])
            runs.append(run)
    return runs


def _summarise(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    """Per-metric average score and pass rate for one run (scored cases only)."""
    rows = conn.execute(
        """SELECT metric_name,
                  AVG(score)                                   AS average_score,
                  AVG(CASE WHEN passed THEN 1.0 ELSE 0.0 END)  AS pass_rate,
                  COUNT(*)                                     AS scored_cases
             FROM eval_results
            WHERE eval_run_id=? AND status='scored'
            GROUP BY metric_name
            ORDER BY metric_name""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_run_summary(run_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        return _summarise(conn, run_id)


def get_run_results(run_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM eval_results WHERE eval_run_id=? ORDER BY case_index, metric_name",
            (run_id,),
        )]


def delete_eval_run(run_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM eval_results WHERE eval_run_id=?", (run_id,))
        conn.execute("DELETE FROM eval_runs WHERE id=?", (run_id,))


# --------------------------------------------------------------------------
# Comparison runs
# --------------------------------------------------------------------------

def create_comparison_run(
    *,
    name: str,
    dataset_id: int,
    metrics: list[str],
    model_ids: list[str],
    judge_model: str | None,
    mode: str,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO comparison_runs
                 (name, dataset_id, metrics, model_ids, judge_model, mode, status, created_at)
               VALUES (?,?,?,?,?,?,'queued',?)""",
            (name, dataset_id, json.dumps(metrics), json.dumps(model_ids),
             judge_model, mode, utcnow()),
        )
        return int(cur.lastrowid)


def update_comparison_run(comparison_id: int, **fields: Any) -> None:
    allowed = {"status", "error", "completed_at", "name"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"update_comparison_run: unknown column(s) {sorted(unknown)}")
    if not fields:
        return
    assignments = ", ".join(f"{k}=?" for k in fields)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE comparison_runs SET {assignments} WHERE id=?",
            (*fields.values(), comparison_id),
        )


def get_comparison_run(comparison_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT c.*, d.name AS dataset_name
                 FROM comparison_runs c LEFT JOIN datasets d ON d.id = c.dataset_id
                WHERE c.id=?""",
            (comparison_id,),
        ).fetchone()
    if not row:
        return None
    cmp_run = dict(row)
    cmp_run["metrics"] = json.loads(cmp_run["metrics"])
    cmp_run["model_ids"] = json.loads(cmp_run["model_ids"])
    return cmp_run


def list_comparison_runs(limit: int = 100) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT c.*, d.name AS dataset_name,
                      (SELECT COUNT(*) FROM eval_runs e WHERE e.comparison_run_id = c.id)
                        AS run_count
                 FROM comparison_runs c LEFT JOIN datasets d ON d.id = c.dataset_id
                ORDER BY c.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["metrics"] = json.loads(d["metrics"])
        d["model_ids"] = json.loads(d["model_ids"])
        out.append(d)
    return out


def get_comparison_child_runs(comparison_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_runs WHERE comparison_run_id=? ORDER BY id",
            (comparison_id,),
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["metrics"] = json.loads(d["metrics"])
        out.append(d)
    return out


def replace_comparison_results(comparison_id: int, results: list[dict[str, Any]]) -> None:
    """Overwrite the aggregate table for a comparison.

    Called after every child run finishes so the /compare page can show
    partial standings while the rest of the models are still working.
    """
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM comparison_results WHERE comparison_run_id=?", (comparison_id,)
        )
        conn.executemany(
            """INSERT INTO comparison_results
                 (comparison_run_id, eval_run_id, model_name, metric_name,
                  average_score, pass_rate, scored_cases, rank)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (
                    comparison_id, r.get("eval_run_id"), r["model_name"], r["metric_name"],
                    r.get("average_score"), r.get("pass_rate"),
                    r.get("scored_cases", 0), r.get("rank"),
                )
                for r in results
            ],
        )


def get_comparison_results(comparison_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT * FROM comparison_results
                WHERE comparison_run_id=?
                ORDER BY metric_name, rank""",
            (comparison_id,),
        )]


def delete_comparison_run(comparison_id: int) -> None:
    with get_conn() as conn:
        child_ids = [r["id"] for r in conn.execute(
            "SELECT id FROM eval_runs WHERE comparison_run_id=?", (comparison_id,)
        )]
        for cid in child_ids:
            conn.execute("DELETE FROM eval_results WHERE eval_run_id=?", (cid,))
        conn.execute("DELETE FROM eval_runs WHERE comparison_run_id=?", (comparison_id,))
        conn.execute("DELETE FROM comparison_results WHERE comparison_run_id=?", (comparison_id,))
        conn.execute("DELETE FROM comparison_runs WHERE id=?", (comparison_id,))


# --------------------------------------------------------------------------
# Benchmark reference data (third-party published figures)
# --------------------------------------------------------------------------

def insert_benchmark_rows(rows: list[dict[str, Any]]) -> int:
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO benchmark_reference
                 (model_label, variant, vendor, family, kind, index_name, score,
                  source_name, source_url, retrieved_at, notes, is_seed)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    r["model_label"], r.get("variant") or None, r.get("vendor"),
                    r.get("family"), r["kind"], r["index_name"], float(r["score"]),
                    r.get("source_name"), r.get("source_url"), r.get("retrieved_at"),
                    r.get("notes"), int(bool(r.get("is_seed"))),
                )
                for r in rows
            ],
        )
        return len(rows)


def count_benchmark_rows(seed_only: bool = False) -> int:
    query = "SELECT COUNT(*) FROM benchmark_reference"
    if seed_only:
        query += " WHERE is_seed=1"
    with get_conn() as conn:
        return int(conn.execute(query).fetchone()[0])


def list_benchmark_rows(family: str | None = None) -> list[dict[str, Any]]:
    """Reference rows, highest score first."""
    query = "SELECT * FROM benchmark_reference"
    params: tuple[Any, ...] = ()
    if family:
        query += " WHERE family=?"
        params = (family,)
    query += " ORDER BY score DESC, model_label"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(query, params)]


def delete_benchmark_row(row_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM benchmark_reference WHERE id=?", (row_id,))
