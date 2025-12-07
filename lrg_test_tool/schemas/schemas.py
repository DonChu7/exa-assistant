"""
schemas.py

This file defines the **single JSON contract** for your test/LRG assistant.

- The router LLM produces a QueryPayload (what to search, filters, pagination).
- The MCP tool receives that payload and queries SQLite FTS + your JSON mappings.
- The MCP tool returns a ToolOutput (normalized rows + facets + paging).
- The writer LLM uses ToolOutput to generate the final human answer.

Keep this stable and life stays easy:
- No explosion of tools/functions.
- You can add more metadata later without breaking everything.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# -----------------------------
# Sorting / paging primitives
# -----------------------------

class SortSpec(BaseModel):
    """
    How to sort query results.

    by:
      - "score" : relevance score (bm25 etc.)
      - "name"  : test_name or lrg_name
      - "lrg"   : lrg_id
      - "suite" : suite name
      - "setup" : test setup
    dir:
      - "asc" or "desc"
    """
    by: Literal["score", "name", "lrg", "suite", "setup", "runtime"] = "score"
    dir: Literal["asc", "desc"] = "desc"


class QueryOps(BaseModel):
    """
    Operational knobs for a query:
      - limit / offset: paging
      - group_by: later if you want aggregation
      - sort: list of SortSpec (we mostly use the first one)
    """
    limit: int = 10
    offset: int = 0
    group_by: Literal["none", "suite", "lrg"] = "none"
    sort: List[SortSpec] = Field(
        default_factory=lambda: [SortSpec(by="score", dir="desc")]
    )


class QueryFilters(BaseModel):
    """
    High-level filters coming from the router LLM or caller.

    All of these are *lists of strings*; we AND them together in run_query:
      - suite: only results whose suite is in this list
      - lrg:   only results whose lrg_id is in this list
      - setup: only results whose setup is in this list
      - flag:  strings like "iscsi=true" or "SAGE_MIRROR_MODE=high"
      - doc_type: ["TEST"], ["LRG"], or both
    """
    suite: List[str] = Field(default_factory=list)
    lrg: List[str] = Field(default_factory=list)
    setup: List[str] = Field(default_factory=list)
    flag: List[str] = Field(default_factory=list)
    doc_type: List[str] = Field(default_factory=list)


class QueryPayload(BaseModel):
    """
    Top-level request into the backend.

    mode:
      - "search"        : normal search over tests/LRGs
      - "answer"        : concept Q&A (flags doc, hybrid setup doc, etc.)
      - "recipe"        : step-by-step instructions
      - "recommend_lrg" : use tests + runtimes to suggest LRG placement
      - "runtime_spike" : (future) detect runtime anomalies
    """
    mode: str = "search"
    text: Optional[str] = None
    filters: QueryFilters = Field(default_factory=QueryFilters)
    ops: QueryOps = Field(default_factory=QueryOps)


# -----------------------------
# Backend row model + output
# -----------------------------

class Row(BaseModel):
    """
    One row in the final result set the MCP tool returns.

    doc_type:
      - "TEST" : a test chunk / profile
      - "LRG"  : an LRG summary or runtime spike synthetic row
    """
    doc_id: str
    doc_type: str  # "TEST" or "LRG"

    # Test identifiers
    test_id: Optional[str] = None
    test_name: Optional[str] = None

    # LRG identifiers
    lrg_id: Optional[str] = None
    lrg_name: Optional[str] = None

    # Context
    suite: Optional[str] = None
    setup: Optional[str] = None
    flags: Dict[str, Any] = Field(default_factory=dict)

    description: Optional[str] = None
    runtime_sec: Optional[float] = None  # or recent_avg for spike rows
    score: Optional[float] = None        # bm25 score OR spike ratio etc.

    preview: str = ""                    # short text snippet
    source: str = "fts"                  # "fts", "db", "runtime", etc.
    citation: Optional[str] = None       # doc_id or some stable ref
    extra: Dict[str, Any] = Field(default_factory=dict)


class FacetEntry(BaseModel):
    value: str
    count: int


class ToolPage(BaseModel):
    limit: int
    offset: int
    returned: int
    total: int


class ToolOutput(BaseModel):
    """
    Final shape returned from run_query, and from the MCP tool.

    ok: false and error != None means backend failed.
    """
    ok: bool = True
    results: List[Row] = Field(default_factory=list)
    facets: Dict[str, List[FacetEntry]] = Field(default_factory=dict)
    page: ToolPage = Field(default_factory=lambda: ToolPage(limit=0, offset=0, returned=0, total=0))
    meta: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# -----------------------------
# Runtime models (for lrg_metrics)
# -----------------------------

class LrgRuntimeSample(BaseModel):
    """
    One runtime sample for an LRG (from metrics DB or API).

    label: build label (e.g. OSS_MAIN_LINUX.X64_250610.1)
    runtime_hours: float, runtime in hours
    """
    label: str
    runtime_hours: float


class LrgRuntimeQueryResult(BaseModel):
    """
    Aggregated runtime history for one LRG.

    samples should be sorted by time (ascending) when used.
    """
    lrg_id: str
    samples: List[LrgRuntimeSample] = Field(default_factory=list)