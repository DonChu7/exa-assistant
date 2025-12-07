#!/usr/bin/env python3
from __future__ import annotations
import os, shlex, subprocess, time
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from lrg_test_tool.schemas.schemas import QueryPayload, ToolOutput
from lrg_test_tool.brain.pipeline import RouterWriterPipeline
from lrg_test_tool.schemas.query_backend import run_query

app = FastMCP("rag-fts-mcp")

_PIPELINE: Optional[RouterWriterPipeline] = None


def _get_pipeline() -> RouterWriterPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = RouterWriterPipeline.from_env()
    return _PIPELINE

import logging
logging.getLogger("mcp").setLevel(logging.ERROR)

# =============================================================================
# Configuration (ENV defaults; callers can override per-call)
# =============================================================================

BASE = Path(__file__).parent / "lrg_test_tool"

ENV_RAG_INDEX = str((BASE / "rag_tests.faiss").resolve())
ENV_RAG_META  = str((BASE / "rag_tests.jsonl").resolve())
ENV_FTS_DB    = str((BASE / "profiles_fts_v2.db").resolve())
ENV_T2L       = str((BASE / "test_to_lrgs.json").resolve())
ENV_L2T       = str((BASE / "lrg_to_tests.json").resolve())
ENV_LRGS_JSON = str((BASE / "lrg_map_with_suites.json").resolve())


ENV_SCRIPTS_DIR = Path(os.getenv("RAG_FTS_SCRIPTS_DIR", str(Path(__file__).parent / "lrg_test_tool" / "scripts"))).resolve()

# Keep same python as venv
PY = sys.executable


def _resolve_default_model() -> str:
    """
    Resolve the best local or online path for the default embedding model.
    - Uses the latest snapshot in Hugging Face cache if available
    - Falls back to canonical model id if not cached
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Respect HF_HOME, otherwise default to ~/.cache/huggingface/hub
    hf_cache = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    model_base = hf_cache / f"models--{model_name.replace('/', '--')}" / "snapshots"

    if model_base.exists():
        snapshots = sorted(
            model_base.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if snapshots:
            return str(snapshots[0])  # use the newest snapshot path

    # fallback if not cached
    return model_name

# --- use it here ---
DEFAULT_MODEL = os.getenv("RAG_EMBED_MODEL", _resolve_default_model())


DEFAULT_POOL_MULT = 8
DEFAULT_RRF_K     = 60.0
DEFAULT_TOP_K     = 10
DEFAULT_TIMEOUT_S = 900


# put near your imports
import os, shlex, subprocess, time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Give first-call headroom; you can tune back to 300 once warm
DEFAULT_TIMEOUT_S = int(os.getenv("MCP_SUBPROC_TIMEOUT", "900"))

def _inherit_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()

    # keep output snappy; reduce thread thrash
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "FAISS_NUM_THREADS"):
        env.setdefault(k, "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    # IMPORTANT: do NOT set HF_HOME / TRANSFORMERS_CACHE / SENTENCE_TRANSFORMERS_HOME / TORCH_HOME here.
    # IMPORTANT: do NOT set TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE here.
    # We want the subprocess to see exactly what your manual shell sees.

    if extra:
        env.update(extra)
    return env

# =============================================================================
# Utilities
# =============================================================================
def _run_cmd(
    argv: List[str],
    cwd: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Runs a subprocess and returns a JSON-serializable result:
      {
        "cmd": "<escaped command string>",
        "rc": <int>,
        "stdout": "<str>",
        "stderr": "<str>",
        "elapsed_sec": <float>,
        "status": "ok" | "error" | "timeout" | "exception"
      }
    """
    t0 = time.time()
    cmd_str = " ".join(shlex.quote(a) for a in argv)
    debug = os.getenv("DEBUG_MCP_SUBPROC", "0") == "1"
    if debug:
        print(f"[{time.strftime('%H:%M:%S')}] spawn: {cmd_str}  cwd={cwd}", flush=True)

    try:
        p = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            # env=_inherit_env(env),
            # small QoL flags; safe defaults
            check=False,
        )
        elapsed = round(time.time() - t0, 3)
        if debug:
            print(f"[{time.strftime('%H:%M:%S')}] done rc={p.returncode} elapsed={elapsed}s", flush=True)
        return {
            "cmd": cmd_str,
            "rc": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "elapsed_sec": elapsed,
            "status": "ok" if p.returncode == 0 else "error",
        }

    except subprocess.TimeoutExpired as e:
        elapsed = round(time.time() - t0, 3)
        if debug:
            print(f"[{time.strftime('%H:%M:%S')}] TIMEOUT after {timeout}s (elapsed={elapsed}s)", flush=True)
        return {
            "cmd": cmd_str,
            "rc": 124,
            "stdout": (e.stdout or ""),
            "stderr": f"TIMEOUT after {timeout}s",
            "elapsed_sec": elapsed,
            "status": "timeout",
        }

    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        if debug:
            print(f"[{time.strftime('%H:%M:%S')}] EXCEPTION {e!r} (elapsed={elapsed}s)", flush=True)
        return {
            "cmd": cmd_str,
            "rc": 1,
            "stdout": "",
            "stderr": f"EXCEPTION: {e!r}",
            "elapsed_sec": elapsed,
            "status": "exception",
        }

def _path_info(p: str) -> Dict[str, Any]:
    path = Path(p)
    info: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        st = path.stat()
        info.update({
            "size": st.st_size,
            "mtime": st.st_mtime,
            "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
        })
    return info


# =============================================================================
# Tools
# =============================================================================

def _ensure_ok(backend_output: ToolOutput) -> Dict[str, Any]:
    data = backend_output.model_dump()
    if not backend_output.ok:
        return {"ok": False, "error": backend_output.error or "backend failure", "result": data}
    return data


@app.tool()
def query_tests(payload: dict) -> dict:
    """Run the structured backend with a pre-built QueryPayload."""
    # Validate and normalize with Pydantic
    qp = QueryPayload(**payload)

    # Run backend (SQLite FTS + mappings)
    return _ensure_ok(run_query(qp))


@app.tool()
def smart_search(question: str, plan: Optional[dict] = None, k: int = 10) -> dict:
    """Run router→backend→writer pipeline with optional plan override."""

    pipeline = _get_pipeline()

    if plan is not None:
        qp = QueryPayload(**plan)
    else:
        qp = pipeline.route(question)
        if k and qp.ops.limit != k:
            qp.ops.limit = k

    result = run_query(qp)
    result_dict = result.model_dump()

    if not result.ok:
        return {
            "ok": False,
            "error": result.error or "query backend failed",
            "plan": qp.model_dump(),
            "result": result_dict,
            "mode": qp.mode,
        }

    answer = pipeline.compose(question, result_dict)

    return {
        "ok": True,
        "answer": answer,
        "plan": qp.model_dump(),
        "result": result_dict,
        "mode": qp.mode,
    }


@app.tool()
def tool_manifest() -> Dict[str, Any]:
    return {
        "service": "rag-fts-mcp",
        "tools": [
            {
                "name": "smart_search",
                "description": "Auto router that plans, queries, and writes answers for LRG/test questions.",
                "intents": ["query", "ask", "diagnose", "recommend"],
            },
            {
                "name": "query_tests",
                "description": "Raw backend access for pre-built QueryPayloads.",
                "intents": ["advanced", "payload"],
            },
        ],
    }


# @app.tool()
# def health_check() -> dict:
#     """
#     Quick, machine-readable health report for RAG/FTS inputs and shared JSON maps.

#     What it returns:
#       - scripts_dir              : where the two python CLIs are expected
#       - rag.index / rag.meta     : presence, size, mtime for FAISS index & jsonl
#       - fts.db                   : presence, size, mtime for SQLite FTS db
#       - maps.t2l / l2t / lrgs_json: presence for mapping/metadata JSONs
#       - default knobs            : useful defaults the LLM can rely on

#     Example human queries this enables:
#       - “Are the indexes loaded?”
#       - “Which files am I missing before I run semantic search?”
#     """
#     return {
#         "scripts_dir": ENV_SCRIPTS_DIR,
#         "rag": {
#             "index": _path_info(ENV_RAG_INDEX),
#             "meta":  _path_info(ENV_RAG_META),
#             "defaults": {
#                 "model": DEFAULT_MODEL,
#                 "pool_mult": DEFAULT_POOL_MULT,
#                 "rrf_k": DEFAULT_RRF_K,
#                 "k": DEFAULT_TOP_K,
#             }
#         },
#         "fts": {
#             "db": _path_info(ENV_FTS_DB),
#             "defaults": {"k": DEFAULT_TOP_K},
#         },
#         "maps": {
#             "t2l":       _path_info(ENV_T2L),
#             "l2t":       _path_info(ENV_L2T),
#             "lrgs_json": _path_info(ENV_LRGS_JSON),
#         },
#     }


# @app.tool()
# def rag_query_tests(
#     q: str,
#     k: int = DEFAULT_TOP_K,
#     index: Optional[str] = None,
#     meta: Optional[str] = None,
#     t2l: Optional[str] = None,
#     lrgs_json: Optional[str] = None,
#     model: str = DEFAULT_MODEL,
#     pool_mult: int = DEFAULT_POOL_MULT,
#     rrf_k: float = DEFAULT_RRF_K,
#     timeout_sec: int = DEFAULT_TIMEOUT_S,
# ) -> dict:
#     """
#     Semantic search over TEST descriptions (FAISS ANN + MiniLM + BM25 + RRF fusion).

#     When to use:
#       - You have a natural-language request and want top-N relevant TESTs,
#         optionally annotated with LRG memberships (via --t2l/--lrgs-json).
#       - No hard constraints like exact setup/flags (use FTS for that).

#     Inputs:
#       q           : human query (e.g., “blockstore performance bottlenecks”)
#       k           : top-N results (default 10)
#       index/meta  : override RAG index paths (otherwise ENV defaults)
#       t2l/lrgs_json: optional maps to print LRGs/suites/runtime for each TEST
#       model       : embedding model id (MiniLM by default)
#       pool_mult   : how many FAISS candidates to re-rank (k * pool_mult)
#       rrf_k       : Reciprocal Rank Fusion smoothing constant
#       timeout_sec : process timeout

#     Returns: { cmd, rc, stdout, stderr, elapsed_sec, status }

#     Example human prompts → suggested calls:
#       - “Find 15 tests about blockstore write amplification”
#         → rag_query_tests(q="blockstore write amplification", k=15)

#       - “Show tests related to diskstate sandbox and tell me which LRGs they hit”
#         → rag_query_tests(q="diskstate sandbox", k=10, t2l=T2L, lrgs_json=LRGS_JSON)

#       - “I want broader recall; consider more candidates before re-ranking”
#         → rag_query_tests(q="rebalance & resync operations", k=10, pool_mult=16)
#     """
#     idx = index or ENV_RAG_INDEX
#     met = meta  or ENV_RAG_META
#     argv = [
#         PY, "rag_query_tests.py",
#         "--index", idx,
#         "--meta",  met,
#         "--q", q,
#         "--k", str(int(k)),
#         "--model", model,
#         "--pool-mult", str(int(pool_mult)),
#         "--rrf-k", str(float(rrf_k)),
#     ]
#     if t2l or ENV_T2L:
#         argv += ["--t2l", str(t2l or ENV_T2L)]
#     if lrgs_json or ENV_LRGS_JSON:
#         argv += ["--lrgs-json", str(lrgs_json or ENV_LRGS_JSON)]
#     return _run_cmd(argv, cwd=ENV_SCRIPTS_DIR, timeout=timeout_sec)


# @app.tool()
# def fts_query(
#     q: str,
#     k: int = DEFAULT_TOP_K,
#     db: Optional[str] = None,
#     t2l: Optional[str] = None,
#     l2t: Optional[str] = None,
#     lrgs_json: Optional[str] = None,
#     doc_type: Optional[str] = None,            # "TEST" | "LRG"
#     require_setup: Optional[str] = None,       # e.g. "xblockini"
#     require_flag: Optional[List[str]] = None,  # e.g. ["iscsi=true","sector=oltp"]
#     prefer_setup: Optional[str] = None,        # soft-boost
#     prefer_flag: Optional[List[str]] = None,   # soft-boost
#     require_suite: Optional[List[str]] = None, # e.g. ["perf","sanity"]
#     prefer_suite: Optional[List[str]] = None,  # soft-boost
#     pool: int = 0,                             
#     structured_only: bool = False,             
#     no_expand: bool = False,                   
#     k_lrgs: int = 50,
#     lrg_order: str = "count",                  # "runtime" | "count" | "name"
#     no_flag_values: bool = False,
#     show_desc: bool = False,
#     desc_chars: int = 800,
#     only_direct: bool = False,
#     debug: bool = False,
#     timeout_sec: int = DEFAULT_TIMEOUT_S,
# ) -> dict:
#     """
#     Constraint-focused retriever over the FTS DB — best when you know setups/flags/suites.

#     When to use:
#       - You want “exact” filtering (setup=…, flag K=V, suite membership).
#       - You need TEST-only or LRG-only results with strict conditions.

#     Key behaviors:
#       - `structured_only=True` skips FTS and filters every TEST by metadata only.
#       - `no_expand=True` avoids TEST↔LRG expansion via mappings (t2l/l2t).
#       - `require_suite=["perf"]` restricts aggregated LRGs to specific suites.

#     Inputs (common):
#       q              : human text (“xblockini iscsi”), or entities within free text
#       doc_type       : "TEST" or "LRG" (omit to let retriever pick both)
#       require_setup  : exact setup filter
#       require_flag   : list of “key=value” constraints (truthy checks allowed e.g. “iscsi=true”)
#       show_desc      : print test descriptions (useful for inspection)

#     Returns: { cmd, rc, stdout, stderr, elapsed_sec, status }

#     Example human prompts → suggested calls:
#       - “List tests in setup xblockini that require iscsi=true”
#         → fts_query(q="xblockini iscsi", doc_type="TEST",
#                     require_setup="xblockini", require_flag=["iscsi=true"],
#                     show_desc=True)

#       - “Show LRGs relevant to diskstate sandbox; prefer perf suite”
#         → fts_query(q="diskstate sandbox", doc_type="LRG",
#                     prefer_suite=["perf"], k_lrgs=30)

#       - “Filter by flags only, ignore FTS scoring”
#         → fts_query(q="anything", structured_only=True,
#                     require_flag=["sector=oltp","nvme=true"])
#     """
#     database = db or ENV_FTS_DB
#     argv = [
#         PY, "query_fts_refined.py",
#         "--db", database,
#         "--q", q,
#         "--k", str(int(k)),
#         "--k-lrgs", str(int(k_lrgs)),
#         "--lrg-order", lrg_order,
#     ]
#     if t2l or ENV_T2L:       argv += ["--t2l", str(t2l or ENV_T2L)]
#     if l2t or ENV_L2T:       argv += ["--l2t", str(l2t or ENV_L2T)]
#     if lrgs_json or ENV_LRGS_JSON:
#         argv += ["--lrgs-json", str(lrgs_json or ENV_LRGS_JSON)]

#     if doc_type:             argv += ["--doc-type", doc_type]
#     if require_setup:        argv += ["--require-setup", require_setup]
#     for rf in (require_flag or []): argv += ["--require-flag", rf]
#     if prefer_setup:         argv += ["--prefer-setup", prefer_setup]
#     for pf in (prefer_flag or []): argv += ["--prefer-flag", pf]
#     for rs in (require_suite or []): argv += ["--require-suite", rs]
#     for ps in (prefer_suite or []):  argv += ["--prefer-suite", ps]

#     if pool > 0:             argv += ["--pool", str(int(pool))]
#     if structured_only:      argv += ["--structured-only"]
#     if no_expand:            argv += ["--no-expand"]
#     if no_flag_values:       argv += ["--no-flag-values"]
#     if show_desc:            argv += ["--show-desc"]
#     if only_direct:          argv += ["--only-direct"]
#     if debug:                argv += ["--debug"]

#     return _run_cmd(argv, cwd=ENV_SCRIPTS_DIR, timeout=timeout_sec)

# ## NEEDS WORK!!!
# @app.tool()
# def smart_search(
#     natural_query: str,
#     mode: str = "auto",              # "auto" | "rag" | "fts"
#     prefer_tests: bool = True,       # for FTS path when constraints detected
#     k: int = DEFAULT_TOP_K,
#     timeout_sec: int = DEFAULT_TIMEOUT_S,
# ) -> dict:
#     """
#     Router over the two tools (still only uses rag_query_tests.py and query_fts_refined.py).

#     Mode selection:
#       - mode="rag": always semantic (RAG).
#       - mode="fts": always FTS.
#       - mode="auto": if query contains structured hints (“setup:”, “flag:”, “suite:”)
#                      → use FTS (doc_type=TEST if prefer_tests=True), otherwise RAG.

#     Inputs:
#       natural_query : raw human text
#       k             : top-k for either path
#       prefer_tests  : when auto→FTS, bias to TEST results

#     Example human prompts → behavior:
#       - “Find diskstate sandbox tests in setup:hybrid flag:sector=oltp”
#         → auto-detects constraints → FTS path with doc_type=TEST
#       - “Blockstore throughput regressions”
#         → no constraints → RAG path

#     Returns: { cmd, rc, stdout, stderr, elapsed_sec, status }
#     """
#     q = natural_query.strip().lower()
#     has_struct = any(tok in q for tok in ["setup:", "flag:", "suite:"])
#     if mode == "fts" or (mode == "auto" and has_struct):
#         return fts_query(
#             q=natural_query, k=k,
#             doc_type="TEST" if prefer_tests else None,
#             show_desc=False,
#             timeout_sec=timeout_sec,
#         )
#     else:
#         return rag_query_tests(q=natural_query, k=k, timeout_sec=timeout_sec)


# # To warm up the RAG as it takes time first time
# @app.tool()
# def rag_warmup() -> dict:
#     """Load embedding model once so later calls are fast."""
#     return rag_query_tests(q="warmup", k=1, pool_mult=1, timeout_sec=900)


# =============================================================================
# Run MCP server
# =============================================================================
if __name__ == "__main__":
    app.run()
