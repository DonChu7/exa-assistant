#!/usr/bin/env python3
import subprocess
from pathlib import Path

# BASE = "/tmp/ai_tool_shray"
BASE = "/home/shratyag/ai_tool/exa-assistant/lrg_test_tool"
CONFIG = Path(f"{BASE}/scripts/suites/suite_cmds.txt")
SCRIPT = f"{BASE}/scripts/suites/suites.py"

SUITES_DIR = Path(f"{BASE}/suites")
SUITES_DIR.mkdir(parents=True, exist_ok=True)

for line in CONFIG.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    try:
        suite, cmd = line.split("|", 1)
    except ValueError:
        print(f"[WARN] Skipping malformed line: {line}")
        continue

    suite = suite.strip()
    cmd = cmd.strip()

    print(f"[+] Running suite: {suite}")
    subprocess.run([
        "python3", SCRIPT, "--suite", suite, "--cmd", cmd
    ], check=True)

print("\n✅ All suite text files generated successfully!")