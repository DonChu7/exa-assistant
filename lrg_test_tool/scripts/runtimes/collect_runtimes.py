#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect per-LRG runtimes by calling a command for each LRG in lrg_to_tests.json.
- Substitutes {lrg} in the command
- Accepts top-level array or {"items":[...]} response
- Takes last N runs (default 3), computes average .time (minutes)
- Parallel execution
- Writes a JSON map { lrg: avg_minutes } to --out

Usage:
  python3 collect_runtimes.py \
    --l2t /ade/shratyag_v8/tklocal/lrg_to_tests.json \
    --cmd 'curl -sS "https://apex.oraclecorp.com/pls/apex/lrg_times/MAIN/lrg/{lrg}"' \
    --out /ade/shratyag_v8/tklocal/runtimes/runtimes_avg_last3.json \
    --last 3 --workers 8
"""
import argparse, json, subprocess, shlex, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

def json_only(stdout: str) -> str:
    """Strip any non-JSON preamble by starting at first '[' or '{'."""
    s = stdout.lstrip()
    i = min([x for x in (s.find('['), s.find('{')) if x >= 0] or [-1])
    return s if i <= 0 else s[i:]

def pick_runs(data: Any) -> List[dict]:
    """Return the array of run dicts from either top-level array or {'items': [...]}."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    return []

def avg_last_minutes(runs: List[dict], last: int) -> Optional[int]:
    """Average .time of last N items (by natural order), rounded to int minutes."""
    if not runs:
        return None
    sel = runs[-last:] if len(runs) > last else runs
    times = [r.get("time") for r in sel if isinstance(r.get("time"), (int, float, str))]
    # coerce strings
    vals: List[float] = []
    for t in times:
        try:
            vals.append(float(t))
        except Exception:
            pass
    if not vals:
        return None
    return int(round(sum(vals)/len(vals)))

def run_cmd(cmd_tmpl: str, lrg: str, timeout: int = 30) -> Optional[int]:
    """Run command for one LRG and return average minutes or None."""
    cmd_str = cmd_tmpl.replace("{lrg}", lrg)
    try:
        proc = subprocess.run(shlex.split(cmd_str), capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return None
        raw = json_only(proc.stdout)
        data = json.loads(raw)
        runs = pick_runs(data)
        return avg_last_minutes(runs, last=3)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2t", required=True, help="lrg_to_tests.json")
    ap.add_argument("--cmd", required=True, help='Command template with {lrg}')
    ap.add_argument("--out", required=True, help="Output JSON map file")
    ap.add_argument("--last", type=int, default=3, help="How many last runs to average")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    # Load LRG names (keys)
    l2t = json.loads(Path(args.l2t).read_text(encoding="utf-8"))
    if not isinstance(l2t, dict):
        print("lrg_to_tests.json must be an object { lrg: [tests...] }", file=sys.stderr)
        sys.exit(2)
    lrgs = sorted(l2t.keys())

    out: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = { ex.submit(run_cmd, args.cmd, l, args.timeout): l for l in lrgs }
        for fut in as_completed(futs):
            l = futs[fut]
            avg = fut.result()
            if avg is not None:
                out[l] = avg

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with {len(out)}/{len(lrgs)} LRGs")

if __name__ == "__main__":
    main()



