# lrg_test_tool/metrics/runtime_store.py

import sqlite3
from pathlib import Path
from datetime import date, timedelta
from typing import List, Optional, Dict

from .schemas_runtime import LrgRuntimeSample, LrgRuntimeQueryResult

# DB location: .../exa-assistant/lrg_test_tool/metrics/lrg_metrics.db
METRICS_DIR = Path(__file__).resolve().parent
DB_PATH = METRICS_DIR / "lrg_metrics.db"
# DB_PATH = METRICS_DIR / "lrg_metrics.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema() -> None:
    """
    Create the lrg_runtime_history table if it doesn't exist.
    """
    conn = get_conn()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS lrg_runtime_history (
        lrg_id      TEXT NOT NULL,
        ts          TEXT NOT NULL,      -- 'YYYY-MM-DD'
        runtime_sec REAL NOT NULL,
        suite       TEXT,
        PRIMARY KEY (lrg_id, ts)
    );
    """)
    conn.commit()
    conn.close()


def upsert_samples(samples: List[LrgRuntimeSample], keep_days: int = 10) -> None:
    """
    Insert or update runtime samples for LRGs, and prune to last `keep_days`
    days globally (by ts).

    Call this once per day after you compute runtimes for each LRG.
    """
    if not samples:
        return

    ensure_schema()
    conn = get_conn()
    cur = conn.cursor()

    rows = [(s.lrg_id, s.ts, s.runtime_sec, s.suite) for s in samples]

    cur.executemany("""
        INSERT INTO lrg_runtime_history (lrg_id, ts, runtime_sec, suite)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(lrg_id, ts) DO UPDATE SET
            runtime_sec = excluded.runtime_sec,
            suite       = excluded.suite;
    """, rows)

    # Global prune: remove rows older than `keep_days` days from today.
    cutoff_date = (date.today() - timedelta(days=keep_days)).isoformat()
    cur.execute("""
        DELETE FROM lrg_runtime_history
        WHERE ts < ?
    """, (cutoff_date,))

    conn.commit()
    conn.close()


def get_lrg_runtimes(
    lrg_id: Optional[str],
    days: int = 30,
    as_of: Optional[date] = None,
) -> List[LrgRuntimeQueryResult]:
    """
    Fetch runtime history for one LRG or all LRGs.

    NOTE:
    - Retention (last N days) is already enforced by the populate_runtimes_from_api.py script.
    - We no longer filter by timestamp here; we just read whatever is present.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if lrg_id:
        rows = cur.execute(
            """
            SELECT lrg_id, label, runtime_hours
            FROM lrg_runtime_history
            WHERE lrg_id = ?
            ORDER BY label
            """,
            (lrg_id,),
        ).fetchall()
    else:
        rows = cur.execute(
            """
            SELECT lrg_id, label, runtime_hours
            FROM lrg_runtime_history
            ORDER BY lrg_id, label
            """
        ).fetchall()

    conn.close()

    by_lrg: Dict[str, List[LrgRuntimeSample]] = {}

    for r in rows:
        lid = r["lrg_id"]
        label = r["label"]
        rh = float(r["runtime_hours"])

        sample = LrgRuntimeSample(
            lrg_id=lid,
            ts=label,               # <- MUST be a string, not None
            label=label,
            runtime_hours=rh,
            runtime_sec=rh * 3600.0,
        )
        by_lrg.setdefault(lid, []).append(sample)

    results: List[LrgRuntimeQueryResult] = []
    for lid, samples in by_lrg.items():
        results.append(
            LrgRuntimeQueryResult(
                lrg_id=lid,
                samples=samples,
            )
        )
    return results


# NEW: small, clean helper for MCP tools / writer.py
def get_runtimes_for_lrg(
    lrg_id: str,
    days: int = 30,
) -> Optional[LrgRuntimeQueryResult]:
    """
    Convenience wrapper: return the LrgRuntimeQueryResult for a single LRG,
    or None if no rows exist.

    `days` is kept for interface symmetry, but retention is already enforced
    when populating the DB.
    """
    results = get_lrg_runtimes(lrg_id=lrg_id, days=days)
    if not results:
        return None
    # There is only one LRG in the result for a specific lrg_id
    return results[0]