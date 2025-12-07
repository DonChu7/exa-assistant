# query_backend.py
#
# Backend for the MCP tool: takes a QueryPayload and returns a ToolOutput
# using your existing profiles_fts.db and mapping JSON files.

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

from .schemas import QueryPayload, ToolOutput, Row, FacetEntry, ToolPage
from lrg_test_tool.brain.constants import normalize_setup
from lrg_test_tool.metrics.runtime_store import get_lrg_runtimes
from lrg_test_tool.metrics.schemas_runtime import LrgRuntimeQueryResult
from functools import lru_cache

# --------------------------------------------------------------------
# CONFIG: adjust paths to match your repo layout
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DB_PATH = ROOT_DIR / "profiles_fts.db"
TEST_TO_LRGS_PATH = ROOT_DIR / "test_to_lrgs.json"
LRG_MAP_PATH = ROOT_DIR / "lrg_map_with_suites.json"


# --------------------------------------------------------------------
# Load mappings: test -> [lrg_ids], lrg_id -> info (suites, runtime, etc.)
# --------------------------------------------------------------------
class Mappings:
    def __init__(
        self,
        test_to_lrgs: Dict[str, List[str]],
        lrg_info: Dict[str, Dict[str, Any]],
    ):
        self.test_to_lrgs = test_to_lrgs
        self.lrg_info = lrg_info

    @classmethod
    def load(cls) -> "Mappings":
        # test_to_lrgs.json: {"tsag4kcolumn.tsc": ["lrgsaexacldegs26", ...], ...}
        if TEST_TO_LRGS_PATH.exists():
            t2l = json.loads(TEST_TO_LRGS_PATH.read_text(encoding="utf-8"))
        else:
            t2l = {}

        # lrg_map_with_suites.json: list of
        #   {"lrgname": "...", "lrg_suite": [...], "runtime": <minutes>?, ...}
        lrg_info: Dict[str, Dict[str, Any]] = {}
        if LRG_MAP_PATH.exists():
            lrgs = json.loads(LRG_MAP_PATH.read_text(encoding="utf-8"))
            for L in lrgs:
                lrg_id = (L.get("lrgname") or "").strip()
                if not lrg_id:
                    continue

                suites = L.get("lrg_suite") or L.get("suites") or []
                if isinstance(suites, str):
                    suites = [suites]
                if not isinstance(suites, list):
                    suites = []
                suites = [
                    s.strip()
                    for s in suites
                    if isinstance(s, str) and s.strip()
                ]

                runtime_min = L.get("runtime")
                runtime_sec = (
                    runtime_min * 60
                    if isinstance(runtime_min, (int, float))
                    else None
                )

                lrg_info[lrg_id] = {
                    "suite": suites[0] if suites else None,  # primary
                    "suites": suites,                         # all suites
                    "runtime_sec": runtime_sec,
                    "raw": L,
                }

        return cls(test_to_lrgs=t2l, lrg_info=lrg_info)


# Global singleton mappings
_MAPPINGS: Optional[Mappings] = None


def get_mappings() -> Mappings:
    global _MAPPINGS
    if _MAPPINGS is None:
        _MAPPINGS = Mappings.load()
    return _MAPPINGS


# --------------------------------------------------------------------
# SQLite connection helper
# --------------------------------------------------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


# --------------------------------------------------------------------
# Flag filter (Python-side)
# --------------------------------------------------------------------
def flags_match(flags: Dict[str, Any], required_flags: List[str]) -> bool:
    """
    required_flags look like ["iscsi=true", "foo=bar"] or ["iscsi"].
    Case-insensitive keys. Values compared as strings.
    """
    if not required_flags:
        return True
    if not isinstance(flags, dict):
        return False

    norm_flags = {str(k).lower(): str(v) for k, v in flags.items()}

    for rf in required_flags:
        rf = rf.strip()
        if not rf:
            continue

        if "=" in rf:
            k, v = rf.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if norm_flags.get(k) != v:
                return False
        else:
            k = rf.strip().lower()
            if k not in norm_flags:
                return False

    return True


# --------------------------------------------------------------------
# FTS query normalization
# --------------------------------------------------------------------
def normalize_fts_query(text: str) -> str:
    """
    Defensive normalizer for FTS MATCH input:

    - Splits on whitespace.
    - For tokens containing '=', keep only the part before '='.
      (e.g. 'iscsi=true' -> 'iscsi')
    - For tokens ending with '.tsc', strip the suffix
      (e.g. 'tsagbug37140353.tsc' -> 'tsagbug37140353')
    - Drops empty fragments.
    """
    if not text:
        return ""
    tokens: List[str] = []
    for raw_tok in text.split():
        tok = raw_tok.strip()
        if not tok:
            continue

        # Drop value part for flag-style tokens
        if "=" in tok:
            tok = tok.split("=", 1)[0].strip()

        # Strip .tsc suffix to avoid FTS '.' syntax problems
        if tok.lower().endswith(".tsc"):
            tok = tok[:-4]

        if tok:
            tokens.append(tok)

    return " ".join(tokens)


# --------------------------------------------------------------------
# Runtime spike backend (mode == "runtime_spike")
# --------------------------------------------------------------------
def _run_runtime_spike_query(qp: QueryPayload) -> ToolOutput:
    """
    Special backend path for mode == 'runtime_spike'.

    Uses lrg_metrics.db via get_lrg_runtimes() to find LRGs that have
    a runtime spike recently vs baseline days.
    """
    mappings = get_mappings()

    # Tunables: you can tweak later or even move into qp.ops
    window_days = 2      # "recent" window
    baseline_days = 5    # past baseline
    min_ratio = 1.2      # recent >= 1.2x baseline => spike

    total_days = window_days + baseline_days

    history: List[LrgRuntimeQueryResult] = get_lrg_runtimes(
        lrg_id=None,
        days=total_days,
        as_of=None,
    )

    rows: List[Row] = []

    suite_filter = set(qp.filters.suite or [])

    for lr in history:
        lrg_id = lr.lrg_id
        samples = lr.samples
        if not samples:
            continue

        linfo = mappings.lrg_info.get(lrg_id, {})
        suite = linfo.get("suite")

        if suite_filter and suite not in suite_filter:
            continue

        # samples sorted by date ascending
        if len(samples) < (baseline_days + window_days):
            continue

        recent = samples[-window_days:]
        baseline = samples[:-window_days]
        if not baseline or not recent:
            continue

        # Convert samples to seconds, tolerating either runtime_sec or runtime_hours
        def sample_sec(s):
            if hasattr(s, "runtime_sec") and s.runtime_sec is not None:
                return float(s.runtime_sec)
            if hasattr(s, "runtime_hours") and s.runtime_hours is not None:
                return float(s.runtime_hours) * 3600.0
            return None

        baseline_secs = [sample_sec(s) for s in baseline]
        recent_secs = [sample_sec(s) for s in recent]

        baseline_secs = [x for x in baseline_secs if x is not None]
        recent_secs = [x for x in recent_secs if x is not None]

        if not baseline_secs or not recent_secs:
            continue

        baseline_avg_sec = sum(baseline_secs) / len(baseline_secs)
        recent_avg_sec = sum(recent_secs) / len(recent_secs)
        if baseline_avg_sec <= 0:
            continue

        ratio = recent_avg_sec / baseline_avg_sec
        if ratio < min_ratio:
            continue

        desc = (
            f"LRG {lrg_id} in suite {suite or '(unknown)'} shows a runtime spike: "
            f"baseline ~{baseline_avg_sec/3600.0:.2f}h over {len(baseline_secs)} runs, "
            f"recent ~{recent_avg_sec/3600.0:.2f}h over {len(recent_secs)} runs "
            f"(x{ratio:.2f} slower)."
        )

        row = Row(
            doc_id=f"LRG_RUNTIME::{lrg_id}",
            doc_type="LRG",
            test_id=None,
            test_name=None,
            lrg_id=lrg_id,
            lrg_name=lrg_id,
            suite=suite,
            setup=None,
            flags={},
            description=desc,
            runtime_sec=recent_avg_sec,  # already seconds
            score=ratio,
            preview=desc[:400],
            source="runtime",
            citation=lrg_id,
            extra={
                "baseline_hours": baseline_avg_sec / 3600.0,
                "recent_hours": recent_avg_sec / 3600.0,
                "ratio": ratio,
                "baseline_count": len(baseline_secs),
                "recent_count": len(recent_secs),
            },
        )
        rows.append(row)

    rows.sort(key=lambda r: r.score or 0, reverse=True)

    total = len(rows)
    start = qp.ops.offset
    end = min(start + qp.ops.limit, total)
    page_rows = rows[start:end]

    def make_facet_suite():
        counts: Dict[str, int] = {}
        for r in rows:
            if not r.suite:
                continue
            counts[r.suite] = counts.get(r.suite, 0) + 1
        return [
            FacetEntry(value=v, count=c)
            for v, c in sorted(counts.items(), key=lambda x: x[0])
        ]

    facets: Dict[str, List[FacetEntry]] = {"suite": make_facet_suite()}

    page = ToolPage(
        limit=qp.ops.limit,
        offset=qp.ops.offset,
        returned=len(page_rows),
        total=total,
    )

    return ToolOutput(
        ok=True,
        results=page_rows,
        facets=facets,
        page=page,
        meta={"backend": "runtime_spike"},
    )


# --------------------------------------------------------------------
# Every place we care about an LRG runtime, we call this.
# --------------------------------------------------------------------
@lru_cache(maxsize=2048)
def get_lrg_recent_runtime_sec(lrg_id: str, days: int = 7) -> Optional[float]:
    """
    Return the average runtime for this LRG over the last `days` days,
    using the metrics DB. Result is in SECONDS (float), or None if no data.
    """
    if not lrg_id:
        return None

    history = get_lrg_runtimes(
        lrg_id=lrg_id,
        days=days,
        as_of=None,
    )
    if not history:
        return None

    lr = history[0]
    samples = lr.samples or []
    if not samples:
        return None

    # Prefer runtime_sec if present, otherwise fall back to runtime_hours
    if hasattr(samples[0], "runtime_sec") and samples[0].runtime_sec is not None:
        avg_sec = sum(float(s.runtime_sec) for s in samples) / len(samples)
        return avg_sec
    elif hasattr(samples[0], "runtime_hours") and samples[0].runtime_hours is not None:
        avg_hours = sum(float(s.runtime_hours) for s in samples) / len(samples)
        return avg_hours * 3600.0

    return None


# --------------------------------------------------------------------
# Main search function: QueryPayload -> ToolOutput
# --------------------------------------------------------------------
def run_query(qp: QueryPayload) -> ToolOutput:
    """
    Core backend query:
      - Uses FTS if qp.text is present
      - Otherwise scans docs (but NOW with sane limits)
      - Applies filters (suite, lrg, setup, flags, doc_type) in Python
      - Uses mappings to fill suite/runtime for TEST/LRG docs
    """
    # 1) Special runtime mode
    if qp.mode == "runtime_spike":
        return _run_runtime_spike_query(qp)

    mappings = get_mappings()
    conn = get_conn()
    cur = conn.cursor()

    filters = qp.filters

    # Normalize doc_type filter
    allowed_doc_types = [
        dt.upper() for dt in (filters.doc_type or []) if dt
    ]

    # For recommend_lrg, we always want TEST docs as base
    if qp.mode == "recommend_lrg":
        allowed_doc_types = ["TEST"]

    # Normalize setup filters
    setup_filters = [
        normalize_setup(s) for s in (filters.setup or [])
    ]
    setup_filters = [s for s in setup_filters if s]

    # 2) Select base candidates (FTS or full scan)
    base_candidates: List[sqlite3.Row] = []

    if qp.mode == "recommend_lrg":
        # For LRG recommendation we ignore free-text completely.
        # We want ALL TEST docs, then we filter by setup/flags/suite in Python.
        sql = """
        SELECT d.rowid AS rowid, d.id, d.doc_type, d.text, d.meta_json,
               NULL AS score
        FROM docs d
        WHERE d.doc_type = 'TEST'
        """
        base_candidates = list(cur.execute(sql))

    elif qp.text:
        safe_q = normalize_fts_query(qp.text)

        sql = """
        SELECT d.rowid AS rowid, d.id, d.doc_type, d.text, d.meta_json,
               bm25(docs_fts) AS score
        FROM docs_fts
        JOIN docs d ON d.rowid = docs_fts.rowid
        WHERE docs_fts MATCH :q
        ORDER BY score
        LIMIT :k;
        """
        k = max(qp.ops.limit * 5, 50)
        params = {"q": safe_q, "k": k}
        base_candidates = list(cur.execute(sql, params))
    else:
        # NO TEXT: full scan, but narrowed by doc_type in SQL
        if allowed_doc_types == ["LRG"]:
            sql = """
            SELECT d.rowid AS rowid, d.id, d.doc_type, d.text, d.meta_json,
                   NULL AS score
            FROM docs d
            WHERE d.doc_type = 'LRG'
            """
        elif allowed_doc_types == ["TEST"]:
            sql = """
            SELECT d.rowid AS rowid, d.id, d.doc_type, d.text, d.meta_json,
                   NULL AS score
            FROM docs d
            WHERE d.doc_type = 'TEST'
            """
        else:
            sql = """
            SELECT d.rowid AS rowid, d.id, d.doc_type, d.text, d.meta_json,
                   NULL AS score
            FROM docs d
            """
        base_candidates = list(cur.execute(sql))

    # 3) Python-side filtering & Row building
    rows: List[Row] = []

    for r in base_candidates:
        doc_id = r["id"]
        doc_type = (r["doc_type"] or "").upper()
        text = r["text"] or ""
        score = float(r["score"]) if r["score"] is not None else None

        try:
            meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        except Exception:
            meta = {}

        # doc_type filter
        if allowed_doc_types and doc_type not in allowed_doc_types:
            continue

        flags = meta.get("flags") or {}
        raw_setup = meta.get("setup")
        setup = normalize_setup(raw_setup) if raw_setup else None

        lrg_id = None
        lrg_name = None
        primary_suite = None
        runtime_sec = None
        description = None

        # TEST docs
        if doc_type == "TEST":
            test_id = meta.get("id") or None
            test_name = test_id
            # Map test -> primary LRG (first one)
            lrg_ids = mappings.test_to_lrgs.get(test_id, []) if test_id else []
            if lrg_ids:
                lrg_id = lrg_ids[0]
                lrg_name = lrg_id
                linfo = mappings.lrg_info.get(lrg_id, {})
                suite = linfo.get("suite")
                # static runtime from JSON (if any)
                runtime_sec = linfo.get("runtime_sec")

                # override with recent metrics if available
                rt_recent = get_lrg_recent_runtime_sec(lrg_id)
                if rt_recent is not None:
                    runtime_sec = rt_recent

            desc_idx = text.find("Description:")
            if desc_idx >= 0:
                description = text[desc_idx + len("Description:"):].strip()
            else:
                description = text.strip()

        # LRG docs
        elif doc_type == "LRG":
            test_id = None
            test_name = None
            lrg_id = meta.get("id") or None
            lrg_name = lrg_id

            linfo = mappings.lrg_info.get(lrg_id, {})
            # use lrg_suite if present
            suite_list = meta.get("lrg_suite") or linfo.get("suites") or []
            if isinstance(suite_list, str):
                suite_list = [suite_list]
            suite = suite_list[0] if suite_list else None

            # runtime in minutes if present in meta; else from mappings
            runtime_min = meta.get("runtime")
            if isinstance(runtime_min, (int, float)):
                runtime_sec = runtime_min * 60
            else:
                runtime_sec = linfo.get("runtime_sec")

            # override with recent metrics if available
            rt_recent = get_lrg_recent_runtime_sec(lrg_id)
            if rt_recent is not None:
                runtime_sec = rt_recent

            description = text.strip()

        else:
            # Unknown doc_type
            continue

        # ---------------------------
        # Compute doc_suites (all suites) for filtering & facets
        # ---------------------------
        doc_suites: List[str] = []
        if doc_type == "TEST" and lrg_id:
            # For tests, derive from LRG metadata
            linfo = mappings.lrg_info.get(lrg_id, {})
            doc_suites = [
                s.strip()
                for s in (linfo.get("suites") or [])
                if isinstance(s, str) and s.strip()
            ]
        elif doc_type == "LRG":
            suite_list = meta.get("lrg_suite") or mappings.lrg_info.get(
                lrg_id or "", {}
            ).get("suites") or []
            if isinstance(suite_list, str):
                suite_list = [suite_list]
            if not isinstance(suite_list, list):
                suite_list = []
            doc_suites = [
                s.strip()
                for s in suite_list
                if isinstance(s, str) and s.strip()
            ]

        # refresh primary_suite from doc_suites if missing
        if not primary_suite and doc_suites:
            primary_suite = doc_suites[0]

        # suite filter: check *any* of the doc_suites
        if filters.suite:
            wanted = set(filters.suite)
            if not any(s in wanted for s in doc_suites):
                continue

        # LRG filter
        if filters.lrg:
            if not lrg_id or lrg_id not in filters.lrg:
                continue

        # setup filter
        if setup_filters:
            if not setup or setup not in setup_filters:
                continue

        # flags filter
        if filters.flag and not flags_match(flags, filters.flag):
            continue

        row = Row(
            doc_id=doc_id,
            doc_type=doc_type,
            test_id=meta.get("id") if doc_type == "TEST" else None,
            test_name=meta.get("id") if doc_type == "TEST" else None,
            lrg_id=lrg_id,
            lrg_name=lrg_name,
            suite=primary_suite,
            setup=setup,
            flags=flags,
            description=description,
            runtime_sec=runtime_sec,
            score=score,
            preview=text[:400],
            source="fts" if qp.text else "db",
            citation=doc_id,
            extra={
                "chunk_id": meta.get("chunk_id"),
                "suites": doc_suites,
            },
        )
        rows.append(row)

    # 2b. Enrich LRGs with runtime from metrics DB (if missing)
    lrg_ids_missing_runtime = {
        r.lrg_id
        for r in rows
        if r.lrg_id and (r.runtime_sec is None or r.runtime_sec <= 0)
    }

    print("[DEBUG] runtime enrichment: need runtime for",
          len(lrg_ids_missing_runtime), "LRGs")

    if lrg_ids_missing_runtime:
        history: List[LrgRuntimeQueryResult] = get_lrg_runtimes(
            lrg_id=None,
            days=365,
            as_of=None,
        )
        print("[DEBUG] runtime enrichment: history entries from DB =", len(history))

        # NOTE: you can flesh this out if you want to attach avg runtimes here.
        # For now this is just debugging: we at least inspect the schema once.
        for h in history:
            lid = h.lrg_id
            if lid not in lrg_ids_missing_runtime:
                continue
            if not h.samples:
                continue
            s0 = h.samples[-1]
            has_hours = hasattr(s0, "runtime_hours")
            has_seconds = hasattr(s0, "runtime_sec")
            print("[DEBUG] sample fields for", lid, "has_hours:", has_hours,
                  "has_sec:", has_seconds)
            break

    # 2c. For recommend_lrg: collapse TEST rows into one row per LRG
    if qp.mode == "recommend_lrg":
        by_lrg: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            if not r.lrg_id:
                continue

            # grab all suites for this row (from extra if present)
            row_suites = []
            if r.extra and isinstance(r.extra, dict):
                row_suites = r.extra.get("suites") or []
            if isinstance(row_suites, str):
                row_suites = [row_suites]
            row_suites = [s for s in row_suites if isinstance(s, str) and s.strip()]

            bucket = by_lrg.setdefault(
                r.lrg_id,
                {
                    "primary_suite": r.suite,
                    "suites": set(),          # all suites we see for this LRG
                    "runtime_values": [],
                    "tests": set(),
                },
            )

            bucket["suites"].update(row_suites)
            if r.runtime_sec is not None:
                bucket["runtime_values"].append(r.runtime_sec)
            if r.test_id:
                bucket["tests"].add(r.test_id)

        aggregated_rows: List[Row] = []
        for lrg_id, info in by_lrg.items():
            rt_vals = info["runtime_values"]
            avg_runtime_sec = sum(rt_vals) / len(rt_vals) if rt_vals else None
            tests_sorted = sorted(info["tests"])
            suites_list = sorted(info["suites"]) if info.get("suites") else []

            # choose which suite name to show:
            # - if user filtered by suite, prefer one of those
            # - otherwise fall back to the first known suite or the original primary_suite
            chosen_suite = None
            if filters.suite:
                wanted = set(filters.suite)
                chosen_suite = next(
                    (s for s in suites_list if s in wanted),
                    None,
                )
            if not chosen_suite:
                if suites_list:
                    chosen_suite = suites_list[0]
                else:
                    chosen_suite = info.get("primary_suite")

            description = (
                f"Recommended LRG {lrg_id} for this setup; "
                f"matching tests: {', '.join(tests_sorted[:5])}"
            ) if tests_sorted else f"Recommended LRG {lrg_id} for this setup."

            if avg_runtime_sec is not None:
                preview = (
                    f"LRG {lrg_id} (suite {chosen_suite}), "
                    f"avg runtime {avg_runtime_sec/3600.0:.2f}h"
                )
            else:
                preview = f"LRG {lrg_id} (suite {chosen_suite}), runtime unknown"

            aggregated_rows.append(
                Row(
                    doc_id=f"LRG_RECOMMEND::{lrg_id}",
                    doc_type="LRG",
                    test_id=None,
                    test_name=None,
                    lrg_id=lrg_id,
                    lrg_name=lrg_id,
                    suite=chosen_suite,
                    setup=None,
                    flags={},
                    description=description,
                    runtime_sec=avg_runtime_sec,
                    score=None,
                    preview=preview[:400],
                    source="recommend_lrg",
                    citation=lrg_id,
                    extra={
                        "matching_tests": tests_sorted,
                        "suites": suites_list,
                    },
                )
            )

        rows = aggregated_rows

    # ---------------------------
    # 3. Sort + pagination + facets
    # ---------------------------
    if qp.ops.sort:
        spec = qp.ops.sort[0]
        key = spec.by
        reverse = (spec.dir == "desc")

        # For LRG recommendations, if we have runtime info,
        # rank by runtime (shortest first) instead of pure score.
        if qp.mode == "recommend_lrg" and any(r.runtime_sec is not None for r in rows):
            key = "runtime"
            reverse = False  # shortest first

        def sort_key(r: Row):
            if key == "score":
                # lower bm25 is better; we treat missing score as very bad
                return r.score if r.score is not None else 1e9
            elif key == "runtime":
                # None => push to the end
                return r.runtime_sec if r.runtime_sec is not None else 1e12
            elif key == "name":
                return r.test_name or r.lrg_name or ""
            elif key == "lrg":
                return r.lrg_id or ""
            elif key == "suite":
                return r.suite or ""
            elif key == "setup":
                return r.setup or ""
            return 0

        rows.sort(
            key=sort_key,
            reverse=False if key in ("score", "runtime") else reverse,
        )

    total = len(rows)
    start = qp.ops.offset
    end = min(start + qp.ops.limit, total)
    page_rows = rows[start:end]

    # 5) Facets: use primary suite for now
    def make_facet(getter):
        counts: Dict[str, int] = {}
        for r in rows:
            v = getter(r)
            if not v:
                continue
            counts[v] = counts.get(v, 0) + 1
        return [
            FacetEntry(value=v, count=c)
            for v, c in sorted(counts.items(), key=lambda x: x[0])
        ]

    facets: Dict[str, List[FacetEntry]] = {
        "suite": make_facet(lambda r: r.suite),
        "setup": make_facet(lambda r: r.setup),
        "lrg": make_facet(lambda r: r.lrg_id),
    }

    page = ToolPage(
        limit=qp.ops.limit,
        offset=qp.ops.offset,
        returned=len(page_rows),
        total=total,
    )

    out = ToolOutput(
        ok=True,
        results=page_rows,
        facets=facets,
        page=page,
        meta={"backend": "sqlite_fts", "db_path": str(DB_PATH)},
    )

    conn.close()
    return out