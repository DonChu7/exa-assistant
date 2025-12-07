# from mcp_client import PersistentMCPClient
# lrg = PersistentMCPClient(["python", "/home/shratyag/ai_tool/exa-assistant/lrg_test_mcp_server.py"])
# # print(lrg.call_tool("lrg_test_health", {}))
# # print(lrg.call_tool("list_tests_for_lrg", {"lrg": "lrgsaexacldegs26"}))
# # print(lrg.call_tool("search_tests", {"query": "volume group", "k": 10}))
# print(lrg.call_tool("find_tests_with_setup_and_flag", {"setup": "xblockini", "flag": "iscsi=true"}))
# lrg.close()


#!/usr/bin/env python3
"""
End-to-end tests for the MCP server tools that ONLY wrap:
  - rag_query_tests.py
  - query_fts_refined.py

What this does:
  - Connects once, calls each tool with multiple parameter sets
  - Prints short summaries so you can eyeball the behavior
  - Includes example “human text” you might say to ChatGPT that maps to each call
"""

import json, time
from mcp_client import PersistentMCPClient

from lrg_test_tool.schemas.schemas import QueryPayload
from lrg_test_tool.brain.router import build_router_prompt
from lrg_test_tool.brain.writer import build_writer_prompt
from mcp_client import PersistentMCPClient
import json

SERVER = ["python", "/home/shratyag/ai_tool/exa-assistant/lrg_test_mcp_server.py"]

def show(title, resp):
    print(f"\n=== {title} ===")
    if isinstance(resp, dict):
        # print command + first 20 lines of stdout/stderr
        print("cmd:", resp.get("cmd"))
        print("rc:", resp.get("rc"), "status:", resp.get("status"), "elapsed:", resp.get("elapsed_sec"))
        out = (resp.get("stdout") or "").strip().splitlines()
        err = (resp.get("stderr") or "").strip().splitlines()
        if out:
            print("--- FULL STDOUT ---")
            print("\n".join(out))
        if err:
            print("--- FULL STDERR ---")
            print("\n".join(err))
    else:
        print(json.dumps(resp, indent=2))

def format_results(resp: dict) -> str:
    if not resp.get("ok"):
        return f"Tool error: {resp.get('error')}"

    rows = resp.get("results", [])
    page = resp.get("page", {})
    lines = []

    lines.append(
        f"Showing {page.get('returned', len(rows))} of {page.get('total', '?')} results:"
    )

    for i, r in enumerate(rows, start=1):
        test_id  = r.get("test_id") or r.get("doc_id")
        lrg_id   = r.get("lrg_id")
        suite    = r.get("suite")
        setup    = r.get("setup")
        flags    = r.get("flags") or {}
        desc     = (r.get("description") or "").split("\n", 1)[0]  # first line only

        flag_str = ", ".join(f"{k}={v}" for k, v in flags.items()) or "(none)"

        lines.append(
            f"{i}. {test_id}  "
            f"[LRG={lrg_id or '-'}, suite={suite or '-'}, setup={setup or '-'}]\n"
            f"   flags: {flag_str}\n"
            f"   desc: {desc}"
        )

    return "\n".join(lines)

def main():
    client = PersistentMCPClient(SERVER)

    # lrg = PersistentMCPClient(["python", "/home/shratyag/ai_tool/exa-assistant/lrg_test_mcp_server.py"])

    payload = {
        "mode": "search",
        "text": "volume group",
        "filters": {
            "suite": [],
            "lrg": [],
            "setup": [],
            "flag": [],
            "doc_type": ["TEST"]
        },
        "ops": {
            "limit": 5,
            "offset": 0,
            "group_by": "none",
            "sort": [{"by": "score", "dir": "desc"}]
        }
    }

    # ✅ NOTE: wrap in {"payload": ...}
    resp = client.call_tool("query_tests", {"payload": payload})
    print("query_tests response:")
    print(format_results(resp))


    # # 1) Health
    # # Human: “Check if my search indexes are wired up.”
    # show("health_check", client.call_tool("health_check", {}))

    # # 2) RAG — simple semantic query
    # # Human: “Find 8 tests about volume groups.”
    # show("rag_query_tests basic",
    #      client.call_tool("rag_query_tests", {"q": "Cell unreachable", "k": 8}))

    # # 3) RAG — include LRG context
    # # Human: “Show diskstate sandbox related tests and also list LRGs/suites.”
    # show("rag_query_tests with LRG context",
    #      client.call_tool("rag_query_tests", {
    #          "q": "blockstore tests",
    #          "k": 10,
    #          # rely on ENV for t2l/lrgs_json, or pass explicitly:
    #          # "t2l": "/ade/.../test_to_lrgs.json",
    #          # "lrgs_json": "/ade/.../lrg_map_with_suites.json",
    #      }))

    # # 4) FTS — TEST-only with exact constraints
    # # Human: “List tests in setup xblockini that have iscsi=true (print descriptions too).”
    # show("fts_query TEST exact constraints + desc",
    #      client.call_tool("fts_query", {
    #          "q": "xblockini iscsi",
    #          "doc_type": "TEST",
    #          "require_setup": "xblockini",
    #          "require_flag": ["iscsi=true"],
    #          "show_desc": True,
    #          "k": 10
    #      }))

    # # 5) FTS — LRG-only, prefer suite
    # # Human: “Which LRGs are relevant to rebalance or resync; prefer EXAC_CORE suite ordering.”
    # show("fts_query LRG prefer suite",
    #      client.call_tool("fts_query", {
    #          "q": "rebalance resync",
    #          "doc_type": "LRG",
    #          "prefer_suite": ["EXAC_CORE"],
    #          "k_lrgs": 30
    #      }))

    # # 7) FTS — only_direct to keep entity mentions tight
    # # Human: “Only keep tests that directly mention my tokens or mapped entities.”
    # show("fts_query only_direct",
    #      client.call_tool("fts_query", {
    #          "q": "tsagsandboxaltersp_da.tsc lrgsaexcsandboxaltersp10",
    #          "only_direct": True,
    #          "k": 15
    #      }))

    # # 8) SMART — auto routes to FTS due to constraints
    # # Human: “Find fencing tests in setup:tsagexastackup flag:sage_mirror_mode”
    # show("smart_search -> FTS auto",
    #      client.call_tool("smart_search", {
    #          "natural_query": "fencing tests setup:tsagexastackup flag:sage_mirror_mode”",
    #          "mode": "auto",
    #          "k": 12
    #      }))

    # # 9) SMART — auto routes to RAG (no constraints)
    # # Human: “Volume Thin clones”
    # show("smart_search -> RAG auto",
    #      client.call_tool("smart_search", {
    #          "natural_query": "Volume Thin clones",
    #          "mode": "auto",
    #          "k": 12
    #      }))

    # # 10) SMART — explicitly choose a mode
    # # Human: “Force FTS and return LRGs only for volume group tests.”
    # show("smart_search force fts (LRG)",
    #      client.call_tool("smart_search", {
    #          "natural_query": "volume group tests",
    #          "mode": "fts",
    #          "k": 15,
    #          "prefer_tests": False  # still TEST bias only affects FTS doc_type in auto path
    #      }))

    # ✅ Graceful shutdown: close transport and wait a moment
    client.close()
    time.sleep(1)  # give the server time to flush

if __name__ == "__main__":
    main()