#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_profiles_simple.py

Goal:
  Build a single profiles.json that contains BOTH:
    - TEST documents (with setup, flags, LRGs, suites, description)
    - LRG documents (with suites + test lists)

python3.11 build_profiles_simple.py \
  --tests ../../tests_extracted.json \
  --lrgs  ../../lrg_map_with_suites.json \
  --out   ../../profiles.json

python3.11 build_fts_index.py \
  --profiles ../profiles.json \
  --db       ../profiles_fts.db \
  --wipe

"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path} must be a JSON ARRAY of objects.")
    return data


def fmt_flags(flags: Any) -> str:
    """
    Render flags dict into a compact string for FTS text:
      { "a": "1", "b": "true" } -> "a=1, b=true"
    """
    if not isinstance(flags, dict) or not flags:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in flags.items())


def build_test_profiles(
    tests: List[Dict[str, Any]],
    lrg_array: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Build TEST profiles with attached LRG + suite info.
    """

    # Build reverse mapping: test_name -> [lrg_ids], [suites]
    test_to_lrgs: Dict[str, List[str]] = {}
    test_to_suites: Dict[str, List[str]] = {}

    for lrg_obj in lrg_array:
        lrg_id = (lrg_obj.get("lrgname") or "").strip()
        suites = lrg_obj.get("lrg_suite") or []
        tests_in_lrg = lrg_obj.get("tests") or []

        if isinstance(suites, str):
            suites = [suites]
        elif not isinstance(suites, list):
            suites = []

        suites = [s.strip() for s in suites if isinstance(s, str) and s.strip()]

        if not isinstance(tests_in_lrg, list):
            continue

        for tname in tests_in_lrg:
            tname = (tname or "").strip()
            if not tname:
                continue
            test_to_lrgs.setdefault(tname, []).append(lrg_id)
            # accumulate suites per test as well
            if suites:
                test_to_suites.setdefault(tname, []).extend(suites)

    # dedupe and sort suites for each test
    for tname, suites in test_to_suites.items():
        test_to_suites[tname] = sorted(set(suites))

    profiles: List[Dict[str, Any]] = []

    for t in tests:
        test_name = (t.get("test_name") or "").strip()
        if not test_name:
            continue

        setup = t.get("setup")
        flags = t.get("flags") or {}
        description = (t.get("description") or "").strip()
        platform = (t.get("platform") or "").strip() or None  # new field

        lrgs = sorted(set(test_to_lrgs.get(test_name, [])))
        suites = sorted(set(test_to_suites.get(test_name, [])))

        flags_str = fmt_flags(flags)
        lrgs_str = ", ".join(lrgs) if lrgs else "(none)"
        suites_str = ", ".join(suites) if suites else "(none)"
        setup_str = setup or "(none)"
        platform_str = platform or "(none)"

        text = (
            f"Test {test_name}. "
            f"setup={setup_str}. "
            f"flags: {flags_str}. "
            f"platform: {platform_str}.\n"
            f"LRGs: {lrgs_str}.\n"
            f"Suites: {suites_str}.\n"
            f"Description: {description}"
        )

        meta = {
            "doc_type": "TEST",
            "id": test_name,
            "setup": setup,
            "flags": flags,
            "lrgs": lrgs,
            "suites": suites,
            "platform": platform,  # new field in meta
        }

        profiles.append(
            {
                "id": f"TEST::{test_name}",
                "doc_type": "TEST",
                "text": text,
                "meta": meta,
            }
        )

    return profiles


def build_lrg_profiles(lrg_array: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build LRG profiles summarizing suites + contained tests.
    """
    profiles: List[Dict[str, Any]] = []

    for L in lrg_array:
        lrg_id = (L.get("lrgname") or "").strip()
        if not lrg_id:
            continue

        suites = L.get("lrg_suite") or []
        if isinstance(suites, str):
            suites = [suites]
        elif not isinstance(suites, list):
            suites = []
        suites = [s.strip() for s in suites if isinstance(s, str) and s.strip()]

        tests = L.get("tests") or []
        if not isinstance(tests, list):
            tests = []
        tests = [t.strip() for t in tests if isinstance(t, str) and t.strip()]

        suites_str = ", ".join(suites) if suites else "(none)"
        tests_str = ", ".join(tests) if tests else "(none)"

        text = (
            f"LRG {lrg_id}.\n"
            f"Suites: {suites_str}.\n"
            f"Contains tests: {tests_str}.\n"
            f"This LRG groups tests for regression and validation of Exadata/Exascale features."
        )

        meta = {
            "doc_type": "LRG",
            "id": lrg_id,
            "lrg_suite": suites,
            "tests": tests,
            "tests_count": len(tests),
        }

        profiles.append(
            {
                "id": f"LRG::{lrg_id}",
                "doc_type": "LRG",
                "text": text,
                "meta": meta,
            }
        )

    return profiles


def main():
    ap = argparse.ArgumentParser(description="Build simple TEST+LRG profiles for FTS.")
    ap.add_argument("--tests", required=True, help="Path to tests_extracted.json")
    ap.add_argument("--lrgs",  required=True, help="Path to lrg_map_with_suites.json")
    ap.add_argument("--out",   required=True, help="Output profiles.json path")
    args = ap.parse_args()

    tests_path = Path(args.tests)
    lrgs_path  = Path(args.lrgs)
    out_path   = Path(args.out)

    tests = load_array(tests_path)
    lrgs  = load_array(lrgs_path)

    test_profiles = build_test_profiles(tests, lrgs)
    lrg_profiles  = build_lrg_profiles(lrgs)

    profiles = test_profiles + lrg_profiles

    out_path.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Built {len(test_profiles)} TEST profiles and {len(lrg_profiles)} LRG profiles")
    print(f"Total profiles: {len(profiles)} -> {out_path}")


if __name__ == "__main__":
    main()