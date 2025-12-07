#!/usr/bin/env python3
"""
Basic RealHW MCP server test stub.

This exercises a few RealHW tools, including the new 'move_job_to_top' capability.

Usage examples:
  - Map LRG to scheduler
      python exa-assistant/test/test_realhw.py --lrg lrgrhexaprovcluster

  - View scheduler status (pool/in-use/waitlist) by scheduler name
      python exa-assistant/test/test_realhw.py --sched nshqap04

  - View scheduler status resolved from LRG
      python exa-assistant/test/test_realhw.py --lrg lrgrhexaprovcluster --view

  - Prioritize a waitlisted job (move to top) for an LRG
      python exa-assistant/test/test_realhw.py --lrg lrgrhexaprovcluster --job 1234567 --prioritize

  - Simulate scheduler end time
      python exa-assistant/test/test_realhw.py --sched nshqap04 --simulate

Environment:
  - The test connects to the RealHW MCP server using:
        ["python", "/scratch/dongyzhu/exa-assistant/realhw_mcp_server.py"]
    Adjust the path if needed by setting REALHW_SERVER_CMD (JSON array) to override.

Notes:
  - 'move_job_to_top' will only succeed if the job is currently in the waitlist for the LRG's scheduler.
  - Most methods return JSON dicts with fields such as status/rc/stdout/stderr or normalized JSON payloads.
"""

import os
import json
import argparse
import time
from typing import List
from mcp_client import PersistentMCPClient


def get_server_cmd() -> List[str]:
    env = os.getenv("REALHW_SERVER_CMD")
    if env:
        try:
            arr = json.loads(env)
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return arr
        except Exception:
            pass
    return ["python", "/scratch/dongyzhu/exa-assistant/realhw_mcp_server.py"]


def show(title: str, resp):
    print(f"\n=== {title} ===")
    if isinstance(resp, dict):
        # When server returns wrapper with cmd/rc/stdout/stderr:
        cmd = resp.get("cmd")
        rc = resp.get("rc")
        status = resp.get("status")
        elapsed = resp.get("elapsed_sec")
        if cmd or rc is not None or status or elapsed is not None:
            print("cmd:", cmd)
            print("rc:", rc, "status:", status, "elapsed:", elapsed)
            out = (resp.get("stdout") or "").strip()
            err = (resp.get("stderr") or "").strip()
            if out:
                print("--- FULL STDOUT ---")
                print(out)
            if err:
                print("--- FULL STDERR ---")
                print(err)
        else:
            print(json.dumps(resp, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(resp, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description="RealHW MCP server smoke tests")
    ap.add_argument("--lrg", help="LRG name (e.g., lrgrhexaprovcluster)")
    ap.add_argument("--sched", help="Scheduler name (e.g., nshqap04)")
    ap.add_argument("--job", help="Farm job ID to prioritize (used with --prioritize)")
    ap.add_argument("--view", action="store_true", help="View scheduler status (derive from --lrg if --sched not provided)")
    ap.add_argument("--prioritize", action="store_true", help="Move a waitlisted job to the top for the LRG (--lrg and --job required)")
    ap.add_argument("--simulate", action="store_true", help="Run scheduler end-time simulation (--sched required)")
    args = ap.parse_args()

    server_cmd = get_server_cmd()
    print("Connecting to RealHW MCP server:", server_cmd)

    client = PersistentMCPClient(server_cmd)
    try:
        # Optionally map LRG→scheduler
        if args.lrg and not (args.view or args.prioritize):
            show("map_lrg_to_scheduler", client.call_tool("map_lrg_to_scheduler", {"lrg": args.lrg}))

        # View scheduler status
        if args.view:
            payload = {}
            if args.sched:
                payload["sched"] = args.sched.lower().strip()
            if args.lrg:
                payload["lrg"] = args.lrg.strip()
            if not payload:
                print("error: --view requires either --sched or --lrg")
            else:
                show("view_status_of_sched", client.call_tool("view_status_of_sched", payload))

        # Prioritize job for LRG
        if args.prioritize:
            if not args.lrg or not args.job:
                print("error: --prioritize requires both --lrg and --job")
            else:
                payload = {"lrg": args.lrg.strip(), "job_id": args.job.strip()}
                show("move_job_to_top", client.call_tool("move_job_to_top", payload))

        # Simulate scheduler end time
        if args.simulate:
            if not args.sched:
                print("error: --simulate requires --sched")
            else:
                show("simulate_sched_end_time", client.call_tool("simulate_sched_end_time", {"sched": args.sched.lower().strip()}))

        # Minimal happy path if no flags provided: show functional mapping list
        if not any([args.view, args.prioritize, args.simulate, args.lrg, args.sched]):
            show("get_functional_hardware_mapping", client.call_tool("get_functional_hardware_mapping", {}))

    finally:
        client.close()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
