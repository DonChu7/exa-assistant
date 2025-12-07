#!/usr/bin/env python3
import argparse, shlex, subprocess, sys
from pathlib import Path
import re

"""
suites.py \
  --suite EXAC_VOL \
  --cmd "farm suite -suite EXAC_VOL"

"""

LRG_RE = re.compile(r'^lrg[a-z0-9_]+$', re.IGNORECASE)

def main():
    ap = argparse.ArgumentParser(description="Run a suite command and write a clean LRG list to suite_<name>.txt")
    ap.add_argument("--suite", required=True, help="Suite name (e.g., perf, redund)")
    ap.add_argument("--cmd", required=True, help="Shell command that outputs LRGs (space/newline separated)")
    ap.add_argument("--out-dir", default="/home/shratyag/ai_tool/exa-assistant/lrg_test_tool/suites", help="Output dir for suite_*.txt")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"suite_{args.suite}.txt"

    # Run the command (use shell=True so pipelines/greps work)
    try:
        proc = subprocess.run(args.cmd, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {args.cmd}", file=sys.stderr)
        print(e.stderr or "", file=sys.stderr)
        sys.exit(1)

    # Split on any whitespace, keep only proper LRG tokens, dedupe, sort
    tokens = re.split(r'\s+', proc.stdout.strip())
    lrgs = sorted({t for t in tokens if LRG_RE.match(t)})

    if not lrgs:
        print("[WARN] No valid LRGs detected from command output.", file=sys.stderr)

    out_file.write_text("\n".join(lrgs) + ("\n" if lrgs else ""), encoding="utf-8")
    print(f"Wrote {len(lrgs)} LRGs → {out_file}")

if __name__ == "__main__":
    main()