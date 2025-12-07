#!/usr/bin/env python3
"""
Extract from many *.tsc files:

1) Headers:
   - NAME
   - DESCRIPTION
   - NOTES
   We parse them as separate blocks and then combine them into a single
   "description" field in the JSON (NAME + DESCRIPTION + NOTES), skipping
   default template values.

2) Setup and Flags BEFORE the first "log start ..." line:
   - setup: lines like  include <something>
     The setup is accepted ONLY if it matches a name from --setups-file
     (case-insensitive, whole-word or exact token). First valid match wins.
   - flags: lines like   let <flag> <value>
     Collected as a dictionary of {flag: value}, unique keys; later
     redefinitions overwrite.

3) Platform classification:
   - If test_name matches /^tsagrh/i -> platform = "REAL HARDWARE"
   - Else if setup in {srdbmsini, tsaginit, tsagnini} -> "EXADATA"
   - Else if setup in {xblockini, tsagexastackup, xrdbmsini} -> "EXASCALE"

4) Skips:
   - Ignore headers like RUNS_STANDALONE, DRIVER_ONLY, EXTERNAL_SETUPFILES,
     PROJECT_IDS, SECURITY_VULNERABILITY, AREAS, FEATURES_TESTED,
     MODES_NOT_SUPPORTED, MODIFIED, etc.
   - Ignore default template text for NAME/DESCRIPTION/NOTES:
       NAME default fragment:
         "<one-line expansion of the name>"
       DESCRIPTION default fragment:
         "<short description of component this file declares/defines>"
       NOTES default fragment:
         "<other useful comments, qualifications, etc.>"

Output JSON shape (array of objects):
[
  {
    "test_name": "tsagfoo.tsc",
    "setup": "xblockini" | null,
    "flags": { ... } | null,
    "description": "combined name/description/notes" | null,
    "platform": "EXASCALE" | "EXADATA" | "REAL HARDWARE" | null
  },
  ...
]

Usage examples:
  python3 extract_descriptions.py \
    /ade/shratyag_v7/oss/test/tsage/src \
    --glob "tsag*.tsc" \
    --setups-file setups_allowlist.txt \
    --skip-setups-file skip_setups.txt \
    --skip-flags-file skip_flags.txt \
    --encoding latin-1 \
    --out-json /home/shratyag/ai_tool/exa-assistant/lrg_test_tool/tests_extracted.json \
    --print-first

Usage:
  python3 extract_descriptions.py \
  /ade/shratyag_v7/oss/test/tsage/sosd \
  --glob "tsagrh*.sh" \
  --setups-file setups_allowlist.txt \
  --out-json /ade/shratyag_v7/tklocal/tests_extracted2.json \
  --skip-setups-file skip_setups.txt \
  --skip-flags-file skip_flags.txt \
  --print-first

python3 extract_descriptions.py \
  /ade/shratyag_v7/oss/test/tsage/sosd \
  --glob "tsagrh*.sh" \
  --setups-file setups_allowlist.txt \
  --skip-setups-file skip_setups.txt \
  --skip-flags-file  skip_flags.txt \
  --encoding latin-1 \
  --out-json /ade/shratyag_v7/tklocal/tests_extracted2.json \
  --print-first

python3 extract_descriptions.py \
  /ade/shratyag_v7/oss/test/tsage/src \
  --glob "tsag*.tsc" \
  --setups-file setups_allowlist.txt \
  --skip-setups-file skip_setups.txt \
  --skip-flags-file  skip_flags.txt \
  --encoding latin-1 \
  --out-json /ade/shratyag_v7/tklocal/tests_extracted1.json \
  --print-first

"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Set

# ----- Patterns -----

LOG_START_RE = re.compile(r"^\s*log\s+start\b", re.IGNORECASE)
INCLUDE_RE = re.compile(
    r"^\s*include\s*(?:<\s*([^>]+?)\s*>|([^\s].*?))\s*$",
    re.IGNORECASE,
)
LET_RE = re.compile(
    r"^\s*let\s+([A-Za-z0-9_.\-]+)\s+(.*?)\s*$",
    re.IGNORECASE,
)

# Header markers like:
#   #    NAME
#   #    DESCRIPTION
#   #    NOTES
HEADER_RE = re.compile(r"^\s*#\s*([A-Z_]+)\s*$")
HASH_CONTENT_RE = re.compile(r"^\s*#\s?(.*)$")

# Default template fragments to ignore
DEFAULT_NAME_FRAGMENT = "<one-line expansion of the name>"
DEFAULT_DESC_FRAGMENT = "<short description of component this file declares/defines>"
DEFAULT_NOTES_FRAGMENT = "<other useful comments, qualifications, etc.>"

# Headers we care about
HEADER_NAME = "NAME"
HEADER_DESC = "DESCRIPTION"
HEADER_NOTES = "NOTES"

# Headers we explicitly ignore (and their blocks) for description:
IGNORED_HEADERS = {
    "RUNS_STANDALONE",
    "DRIVER_ONLY",
    "EXTERNAL_SETUPFILES",
    "PROJECT_IDS",
    "SECURITY_VULNERABILITY",
    "AREAS",
    "FEATURES_TESTED",
    "MODES_NOT_SUPPORTED",
    "MODIFIED",
}


# ----- Helpers for list files -----

def _read_list_file(path_str: Optional[str]) -> List[str]:
    out: List[str] = []
    if not path_str:
        return out
    p = Path(path_str)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def read_set_file_as_lower_set(path_str: Optional[str]) -> Set[str]:
    return {s.lower() for s in _read_list_file(path_str)}


def read_setups_allowlist(path_str: str) -> List[str]:
    lst = _read_list_file(path_str)
    norm = []
    for s in lst:
        ss = s.strip().lower()
        if ss.endswith(".tsc"):
            ss = ss[:-4]
        norm.append(ss)
    return norm


# ----- Header extraction: NAME, DESCRIPTION, NOTES -----

def _clean_block(lines: List[str]) -> Optional[str]:
    """
    Clean a block of '# ...' lines:
      - strip leading '#'
      - trim leading/trailing blank lines
      - join with '\n'
    """
    cleaned: List[str] = []
    for ln in lines:
        m = HASH_CONTENT_RE.match(ln)
        if m:
            cleaned.append(m.group(1).rstrip())
        else:
            # non-comment line -> treat as end of block
            break

    # Trim empty top/bottom
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    if not cleaned:
        return None
    return "\n".join(cleaned).strip()


def extract_name_desc_notes(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse the header comment blocks:

      #    NAME
      #      <something>
      #
      #    DESCRIPTION
      #      <something>
      #
      #    NOTES
      #      <something>

    Returns (name_text, desc_text, notes_text), each possibly None.
    Default template-only values are turned into None.
    """
    name_lines: List[str] = []
    desc_lines: List[str] = []
    notes_lines: List[str] = []

    current: Optional[str] = None

    for ln in text.splitlines():
        m = HEADER_RE.match(ln)
        if m:
            header = m.group(1).upper()
            if header in {HEADER_NAME, HEADER_DESC, HEADER_NOTES}:
                current = header
            elif header in IGNORED_HEADERS:
                current = None
            else:
                current = None
            continue

        if current is None:
            continue

        # We are inside one of NAME/DESCRIPTION/NOTES blocks; only consume
        # lines that still look like comments ("# ...").
        if not ln.lstrip().startswith("#"):
            # Non-comment -> end of current header block
            current = None
            continue

        if current == HEADER_NAME:
            name_lines.append(ln)
        elif current == HEADER_DESC:
            desc_lines.append(ln)
        elif current == HEADER_NOTES:
            notes_lines.append(ln)

    name_text = _clean_block(name_lines)
    desc_text = _clean_block(desc_lines)
    notes_text = _clean_block(notes_lines)

    # Remove default template placeholders
    if name_text and DEFAULT_NAME_FRAGMENT in name_text:
        name_text = None
    if desc_text and DEFAULT_DESC_FRAGMENT in desc_text:
        desc_text = None
    if notes_text and DEFAULT_NOTES_FRAGMENT in notes_text:
        notes_text = None

    return name_text, desc_text, notes_text


# ----- Setup & flags extraction (unchanged logic) -----

def _normalize_include_target(raw: str) -> str:
    s = raw.strip().strip('"').strip("'")
    s = s.lower()
    if s.endswith(".tsc"):
        s = s[:-4]
    return s


def _find_setup_in_include(
    target: str,
    allowlist: List[str],
    skip_setups_lower: Set[str],
) -> Optional[str]:
    tgt = target.strip()
    if not tgt:
        return None
    tokens = re.split(r"[^A-Za-z0-9_.\-]+", tgt)
    token_set_lower = {t.lower() for t in tokens if t}

    # 1) Exact token match
    for setup in allowlist:
        low = setup.lower()
        if low in skip_setups_lower:
            continue
        if low in token_set_lower:
            return setup

    # 2) Fallback: substring with word-ish boundaries
    for setup in allowlist:
        low = setup.lower()
        if low in skip_setups_lower:
            continue
        if re.search(rf"\b{re.escape(low)}\b", tgt.lower()):
            return setup

    return None


def extract_setup_and_flags(
    text: str,
    setups_allowlist: List[str],
    skip_setups_lower: Set[str],
    skip_flags_lower: Set[str],
) -> Tuple[Optional[str], Dict[str, str]]:
    lines = text.splitlines()
    setup: Optional[str] = None
    flags: Dict[str, str] = {}

    for ln in lines:
        if LOG_START_RE.match(ln):
            break

        m_inc = INCLUDE_RE.match(ln)
        if m_inc and setup is None:
            target = m_inc.group(1) or m_inc.group(2) or ""
            target = _normalize_include_target(target)
            found = _find_setup_in_include(target, setups_allowlist, skip_setups_lower)
            if found:
                setup = found

        m_let = LET_RE.match(ln)
        if m_let:
            k = m_let.group(1).strip()
            v = m_let.group(2).strip()
            if k and k.lower() not in skip_flags_lower:
                flags[k] = v

    return setup, flags


# ----- Platform classification -----

EXADATA_SETUPS = {"srdbmsini", "tsaginit", "tsagnini"}
EXASCALE_SETUPS = {"xblockini", "tsagexastackup", "xrdbmsini"}


def classify_platform(test_name: str, setup: Optional[str]) -> Optional[str]:
    """
    - tsagrh*.* -> "REAL HARDWARE"
    - setup in EXADATA_SETUPS  -> "EXADATA"
    - setup in EXASCALE_SETUPS -> "EXASCALE"
    """
    base = test_name.lower()

    if base.startswith("tsagrh"):
        return "REAL HARDWARE"

    if setup:
        s = setup.lower()
        if s in EXADATA_SETUPS:
            return "EXADATA"
        if s in EXASCALE_SETUPS:
            return "EXASCALE"

    return None


# ----- Main -----

def main():
    ap = argparse.ArgumentParser(
        description="Extract setup, flags, and combined NAME/DESCRIPTION/NOTES from .tsc files."
    )
    ap.add_argument("root", type=str, help="Root folder to scan (recursively)")
    ap.add_argument("--glob", default="**/*.tsc", help='Glob pattern to include files')
    ap.add_argument("--encoding", default="utf-8", help="File encoding")
    ap.add_argument("--out-json", default="tests_extracted.json", help="Output JSON filename")
    ap.add_argument("--setups-file", required=True, help="File listing ALL valid setup names")
    ap.add_argument("--skip-setups-file", default=None, help="File listing setups to skip")
    ap.add_argument("--skip-flags-file", default=None, help="File listing flags to skip")
    ap.add_argument("--print-first", action="store_true", help="Print first few extracted results to console")
    args = ap.parse_args()

    root = Path(args.root)
    setups_allowlist = read_setups_allowlist(args.setups_file)
    if not setups_allowlist:
        raise SystemExit(f"No setups loaded from {args.setups_file}")

    skip_setups_lower = read_set_file_as_lower_set(args.skip_setups_file)
    skip_flags_lower = read_set_file_as_lower_set(args.skip_flags_file)

    results: List[Dict[str, object]] = []
    err_log_path = Path("extract_errors.log")
    printed = 0
    total = found = 0

    with err_log_path.open("w", encoding="utf-8") as ferr:
        for p in root.glob(args.glob):
            if not p.is_file():
                continue
            total += 1
            try:
                text = p.read_text(encoding=args.encoding, errors="ignore")
            except Exception as e:
                ferr.write(f"[READ_ERROR] {p}: {e}\n")
                continue

            # 1) Headers: NAME/DESCRIPTION/NOTES
            name_text, desc_text, notes_text = extract_name_desc_notes(text)

            parts: List[str] = []
            if name_text:
                parts.append(name_text)
            if desc_text:
                parts.append(desc_text)
            if notes_text:
                parts.append(notes_text)

            combined_desc = "\n\n".join(parts).strip() if parts else ""

            # 2) Setup & flags
            setup, flags = extract_setup_and_flags(
                text, setups_allowlist, skip_setups_lower, skip_flags_lower
            )

            # 3) Platform
            test_name = p.name
            platform = classify_platform(test_name, setup)

            # 4) Normalize empties to None / null
            desc_val = combined_desc if combined_desc.strip() else None
            setup_val = setup if setup else None
            flags_val = flags if flags else None
            platform_val = platform if platform else None

            record = {
                "test_name": test_name,
                "setup": setup_val,
                "flags": flags_val,
                "description": desc_val,
                "platform": platform_val,
            }
            results.append(record)
            found += 1

            if args.print_first and printed < 3:
                print(f"\n=== {test_name} ===")
                print(f"setup: {setup_val}")
                print(f"flags: {flags_val}")
                print(f"platform: {platform_val}")
                print("description:")
                print(desc_val)
                printed += 1

    Path(args.out_json).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Scanned: {total} files")
    print(f"Extracted: {found} records")
    print(f"Wrote JSON array: {args.out_json}")
    print(f"Errors/misses: see {err_log_path}")


if __name__ == "__main__":
    main()