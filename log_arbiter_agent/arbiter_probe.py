#!/usr/bin/env python3
"""
Interactive helper to inspect a single incident and optionally ask the LLM arbiter.

Examples:
    python arbiter_probe.py --incident-id 3fbcedc6 \
        --incidents data/incidents.csv \
        --known-index data/.known_miniLM.pkl \
        --ignore-index data/.ignore_miniLM.pkl \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --llm --llm-model gpt-4o-search-preview
"""

import argparse
from typing import Dict, Any, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import load_pickle, normalize_log_window

from triage_st_llm import (
    call_llm_arbiter,
    decide_with_embeddings,
    fmt_examples,
    should_arbitrate,
)


def _load_incident(args: argparse.Namespace) -> Tuple[str, str]:
    if args.incident_text:
        return args.incident_text, args.component or ""

    if args.incident_file:
        with open(args.incident_file, "r", errors="ignore") as fh:
            return fh.read(), args.component or ""

    if args.incident_id:
        if not args.incidents:
            raise SystemExit("--incidents CSV is required when using --incident-id")
        df = pd.read_csv(args.incidents)
        if "log_window" not in df.columns:
            if "messages" in df.columns:
                df["log_window"] = df["messages"]
            else:
                raise SystemExit(
                    f"{args.incidents} must contain either 'log_window' or 'messages' column."
                )
        row = df.loc[df["incident_id"] == args.incident_id]
        if row.empty:
            raise SystemExit(f"Incident {args.incident_id} not found in {args.incidents}")
        rec = row.iloc[0]
        return str(rec["log_window"]), str(rec.get("component", ""))

    raise SystemExit("Provide one of --incident-text, --incident-file, or --incident-id.")


def _print_heading(title: str) -> None:
    print(f"\n=== {title} ===")


def main(args: argparse.Namespace) -> None:
    incident_text, component = _load_incident(args)
    norm_text = normalize_log_window(incident_text)

    kb_idx = load_pickle(args.known_index)
    ig_idx = load_pickle(args.ignore_index)

    if "embs" not in kb_idx or "embs" not in ig_idx:
        raise SystemExit("MiniLM indexes required (run build_memories_st.py).")

    model = SentenceTransformer(args.model)

    pseudo_row: Dict[str, Any] = dict(
        incident_id=args.incident_id or "<manual>",
        host="",
        component=component,
        log_window=incident_text,
    )
    result = decide_with_embeddings(
        pseudo_row, kb_idx, ig_idx, model, t_known=args.t_known, t_ignore=args.t_ignore
    )

    baseline_label = result["label"]
    kbest = result.get("kbest", []) or []
    ibest = result.get("ibest", []) or []
    k0 = result.get("k0_sim", 0.0)
    i0 = result.get("i0_sim", 0.0)

    _print_heading("Incident Text (raw)")
    print(incident_text.strip())

    _print_heading("Incident Text (normalized)")
    print(norm_text.strip())

    _print_heading("Retrieval Scores")
    print(f"Best known similarity : {k0:.4f}")
    print(f"Best ignore similarity: {i0:.4f}")
    print(f"Baseline label        : {baseline_label} ({result['reason']})")

    if kbest:
        _print_heading("Top Known Matches")
        print(fmt_examples(kb_idx["df"], kbest, "bug_id", "context_window"))
    if ibest:
        _print_heading("Top Ignore Matches")
        print(fmt_examples(ig_idx["df"], ibest, "pattern_id", "context_window"))

    if baseline_label in (0, 1):
        _print_heading("Decision")
        print(f"MiniLM already classified this incident as {baseline_label}.")
        return

    if not args.llm:
        _print_heading("Decision")
        print("MiniLM classified this incident as -1 (novel/ambiguous).")
        print("Re-run with --llm to ask the arbiter.")
        return

    reason_min_lines = args.reason_min_lines

    if not should_arbitrate(k0, i0, incident_text, args.t_known, args.t_ignore):
        _print_heading("Decision")
        print("Heuristics suggest skipping arbitration (very low similarity).")
        print("Use --force to override.")
        if not args.force:
            return

    label, reason, confidence, sources, extra_sources = call_llm_arbiter(
        incident_text=incident_text,
        known_snips=fmt_examples(kb_idx["df"], kbest, "bug_id", "context_window") if kbest else "",
        ignore_snips=fmt_examples(ig_idx["df"], ibest, "pattern_id", "context_window") if ibest else "",
        model=args.llm_model,
        reason_min_lines=reason_min_lines,
        timeout=args.llm_timeout,
        max_retries=args.llm_retries,
    )

    _print_heading("LLM Arbiter Result")
    print(f"LLM label : {label}")
    print(f"Reason    : {reason}")
    print(f"Confidence: {confidence:.3f}")
    if sources:
        print("Sources   : " + ", ".join(str(s) for s in sources))
    else:
        print("Sources   : (none)")
    if extra_sources:
        print("Additional: " + ", ".join(str(s) for s in extra_sources))
    else:
        print("Additional: (none)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a single incident with the LLM arbiter.")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--incident-text", help="Raw incident text (multi-line allowed).")
    src.add_argument("--incident-file", help="Path to a file containing the incident text.")
    parser.add_argument("--incident-id", help="Incident ID from the incidents CSV.")
    parser.add_argument("--incidents", help="Path to incidents CSV (needed with --incident-id).")
    parser.add_argument("--component", help="Component hint when providing raw text.", default="")

    parser.add_argument("--known-index", required=True, help="MiniLM known index pickle.")
    parser.add_argument("--ignore-index", required=True, help="MiniLM ignore index pickle.")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--t-known", type=float, default=0.72)
    parser.add_argument("--t-ignore", type=float, default=0.70)

    parser.add_argument("--llm", action="store_true", help="Ask the LLM arbiter for a decision.")
    parser.add_argument("--llm-model", default="gpt-4o-search-preview",
                        help="Search-enabled model (e.g., gpt-4o-search-preview, gpt-4o-mini-search-preview).")
    parser.add_argument("--reason-min-lines", type=int, default=10,
                        help="Minimum number of lines expected in the LLM reasoning output.")
    parser.add_argument("--llm-timeout", type=float, default=60.0, help="Timeout per LLM request (seconds).")
    parser.add_argument("--llm-retries", type=int, default=5, help="Retry count for the LLM request.")
    parser.add_argument("--force", action="store_true", help="Force LLM arbitration even if heuristics skip.")

    args = parser.parse_args()
    main(args)
