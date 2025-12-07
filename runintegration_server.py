#!/usr/bin/env python3
"""
MCP server for RunIntegration utilities.

Exposes tools:
- idle_envs() -> {"idle_envs": list|str}
- disabled_envs() -> {"disabled_envs": list}
- status(rack: str) -> {"status": str}
- pending_tests(txn: str) -> pending tests + host map
- disabled_txn_status(txn: str) -> disabled env statuses

Depends on your existing runintegration_agent.py in PYTHONPATH.
"""

import os
import json
import re
from typing import Any, Dict, List, Union

# pip install mcp
from mcp.server.fastmcp import FastMCP

from runintegration_agent import (
    get_idle_envs_concurrent,
    get_disabled_envs,
    check_runintegration_status,
    get_pending_tests,
    get_pending_tests_with_details,
    get_disabled_txn_status,
    get_disabled_txn_status_with_details,
)

# Optional: assert required files exist early (fail-fast & clearer errors)
RUNTABLE_PATH = "/net/10.32.19.91/export/exadata_images/ImageTests/daily_runs_1/OSS_MAIN/runtable"
CONNECT_FILE = "/net/10.32.19.91/export/exadata_images/ImageTests/.pxeqa_connect"

app = FastMCP("runintegration-mcp")

@app.tool()
def idle_envs(max_workers: int | None = None, ssh_timeout: int | None = None, per_host_limit: int | None = None):
    """
    List idle RunIntegration environments (fast concurrent scanner).
    Optional args:
      max_workers: overall concurrency (default 24)
      ssh_timeout: per-SSH connect timeout seconds (default 3)
      per_host_limit: cap concurrent checks per user@host (default None)
    """
    mw = max_workers or 24
    st = ssh_timeout or 3
    ph = per_host_limit
    result = get_idle_envs_concurrent(max_workers=mw, ssh_timeout=st, per_host_limit=ph)
    return {"idle_envs": result}

@app.tool()
def disabled_envs() -> Dict[str, List[str]]:
    """
    List disabled environments from runtable.

    Returns:
      {"disabled_envs": ["<FULL_RACK> : <DEPLOY_TYPE>", ...]}
    """
    result = get_disabled_envs() or []
    return {"disabled_envs": result}

def _normalize_rack(r: str) -> str:
    m = re.search(r'(sca[\w-]*?adm\d{2})', r, flags=re.IGNORECASE)
    return m.group(1) if m else r.strip()

@app.tool()
def status(rack: str) -> Dict[str, str]:
    """
    Get RunIntegration status for a given rack (e.g., 'scaqan07adm07').

    Args:
      rack: short rack name to search in runtable.

    Returns:
      {"status": "<human-readable status string>"}
    """
    if not isinstance(rack, str) or not rack.strip():
        return {"status": "Invalid 'rack' parameter."}
    base = _normalize_rack(rack)
    return {"status": check_runintegration_status(rack.strip())}


@app.tool()
def pending_tests(txn: str) -> Dict[str, Any]:
    """
    List pending tests for a transaction name across enabled environments.

    Returns host-to-test mapping and the pending test list.
    """
    txn_name = (txn or "").strip()
    if not txn_name:
        return {"error": "Missing required 'txn' parameter."}
    detailed_rows = None
    pending_csv = ""
    try:
        detailed_rows, pending_csv = get_pending_tests_with_details(txn_name)
    except Exception:
        detailed_rows = None

    if detailed_rows is None:
        try:
            result = get_pending_tests(txn_name)
        except Exception as exc:
            return {"error": f"Failed to fetch pending tests: {exc}", "txn": txn_name}

        if isinstance(result, tuple):
            host_map, pending_csv = result
            pending_list = [t.strip() for t in (pending_csv or "").split(",") if t.strip()]
            pending_details = [
                {"hostname": host, "test_name": test}
                for host, test in (host_map or {}).items()
                if test in pending_list
            ]
            return {
                "txn": txn_name,
                "pending_tests": pending_list,
                "host_to_test": host_map,
                "pending_tests_csv": pending_csv,
                "pending_count": len(pending_list),
                "pending_details": pending_details,
            }
        if isinstance(result, str):
            return {"txn": txn_name, "error": result}
        return {"txn": txn_name, "error": "Unexpected response from get_pending_tests."}

    # Detailed data path (duplicates preserved)
    pending_list = [t.strip() for t in (pending_csv or "").split(",") if t.strip()]
    host_map: Dict[str, str] = {}
    pending_details: List[Dict[str, str]] = []
    for row in detailed_rows or []:
        hostname = row.get("hostname")
        test_name = row.get("test_name")
        if hostname and test_name and hostname not in host_map:
            host_map[hostname] = test_name
        if test_name in pending_list:
            pending_details.append({
                "hostname": hostname,
                "test_name": test_name,
                "rack_description": row.get("rack_description"),
            })

    return {
        "txn": txn_name,
        "pending_tests": pending_list,
        "host_to_test": host_map,
        "pending_tests_csv": pending_csv,
        "pending_count": len(pending_list),
        "pending_details": pending_details,
    }


@app.tool()
def disabled_txn_status(txn: str) -> Dict[str, Any]:
    """
    Show disabled environment status (PASSED/FAILED/RUNNING/PENDING) for a transaction.
    """
    txn_name = (txn or "").strip()
    if not txn_name:
        return {"error": "Missing required 'txn' parameter."}
    details = None
    try:
        details = get_disabled_txn_status_with_details(txn_name)
    except Exception:
        details = None

    if details is None:
        try:
            status_map = get_disabled_txn_status(txn_name)
        except Exception as exc:
            return {"error": f"Failed to fetch disabled txn status: {exc}", "txn": txn_name}
        return {
            "txn": txn_name,
            "disabled_env_status": status_map,
            "count": len(status_map or {}),
        }

    status_map: Dict[str, Dict[str, str]] = {}
    for row in details or []:
        host = row.get("hostname")
        if host and host not in status_map:
            status_map[host] = {
                "test_name": row.get("test_name"),
                "status": row.get("status"),
            }
    return {
        "txn": txn_name,
        "disabled_env_status": status_map,
        "count": len(status_map or {}),
        "disabled_details": details,
    }

@app.tool()
def tool_manifest() -> Dict[str, Any]:
    return {
        "service": "runintegration-mcp",
        "tools": [
            {
                "name": "idle_envs",
                "description": "List idle environments in RunIntegration pool.",
                "intents": ["idle envs", "available envs", "free envs"],
                "patterns": [r"\bidle\b.*runintegration", r"\bavailable\b.*runintegration", r"\bfree\b.*runintegration", r"\bavailable\b.*runintegration"]
            },
            {
                "name": "disabled_envs",
                "description": "List disabled environments in RunIntegration pool.",
                "intents": ["disabled envs", "unavailable envs"],
                "patterns": [r"\bdisabled\b.*runintegration", r"\bunavailable\b.*runintegration"]
            },
            {
                "name": "status",
                "description": "Show RunIntegration job status for a rack (e.g., scaqap19adm01).",
                "intents": ["status", "what job", "running", "busy"],
                "patterns": [r"\b(what\s+job|status|running|busy)\b.*sca.*adm\d{2}"]
            },
            {
                "name": "pending_tests",
                "description": "List pending tests for a transaction across enabled envs.",
                "intents": ["pending tests", "txn pending"],
                "patterns": [r"\bpending\b.*txn", r"\bpending\b.*test"]
            },
            {
                "name": "disabled_txn_status",
                "description": "Show disabled env status (PASSED/FAILED/RUNNING/PENDING) for a transaction.",
                "intents": ["disabled txn status", "disabled tests"],
                "patterns": [r"\bdisabled\b.*txn", r"\bdisabled\b.*test"]
            }
        ]
    }


if __name__ == "__main__":
    # Starts an MCP stdio server (reads/writes JSON-RPC on stdin/stdout)
    app.run()
