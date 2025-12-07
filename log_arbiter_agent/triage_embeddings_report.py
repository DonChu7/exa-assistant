#!/usr/bin/env python3
"""
Quick scan utility: run MiniLM-only retrieval against the incidents CSV and
emit CSV reports for ambiguous (-1) windows and (optionally) confident matches,
all without invoking the LLM arbiter. Deduplication happens on the ambiguous
report by default so repeated structures collapse.

Example:
    python triage_embeddings_report.py \
        --incidents new_data/image_error_sessions.csv \
        --known-index data/.known_miniLM.pkl \
        --ignore-index data/.ignore_miniLM.pkl \
        --out-csv new_data/minilm_ambiguous.csv
"""

import argparse
import os
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import load_pickle
from triage_st_llm import decide_with_embeddings


def _normalize_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist (log_window, host, dev_feedback)."""
    if "log_window" not in df.columns:
        if "messages" in df.columns:
            df["log_window"] = df["messages"]
        else:
            raise ValueError("Incident CSV must contain 'log_window' or 'messages' column.")
    if "host" not in df.columns:
        if "hostname" in df.columns:
            df["host"] = df["hostname"]
        else:
            df["host"] = ""
    if "dev_feedback" not in df.columns:
        df["dev_feedback"] = ""
    if "message_structure" not in df.columns:
        df["message_structure"] = ""
    return df


def scan_minilm(
    df: pd.DataFrame,
    kb_idx: Dict[str, Any],
    ig_idx: Dict[str, Any],
    encoder: SentenceTransformer,
    t_known: float,
    t_ignore: float,
    include_confident: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (ambiguous_rows, confident_rows) from MiniLM decisions."""
    ambiguous: List[Dict[str, Any]] = []
    confident: List[Dict[str, Any]] = []
    for _, rec in df.iterrows():
        decision = decide_with_embeddings(
            rec, kb_idx, ig_idx, encoder, t_known=t_known, t_ignore=t_ignore
        )
        label = decision["label"]
        entry = {
            "incident_id": rec.get("incident_id", ""),
            "host": rec.get("host", ""),
            "component": rec.get("component", ""),
            "log_window": rec.get("log_window", ""),
            "message_count": rec.get("message_count", ""),
            "message_structure": rec.get("message_structure", ""),
            "label": label if label != -2 else -1,
            "reason": decision.get("reason", ""),
            "k0_sim": decision.get("k0_sim", np.nan),
            "i0_sim": decision.get("i0_sim", np.nan),
            "dev_feedback": rec.get("dev_feedback", ""),
        }
        if label == -2:
            ambiguous.append(entry)
        elif include_confident:
            confident.append(entry)
    return ambiguous, confident


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List MiniLM ambiguous (-1) incidents without invoking the LLM arbiter."
    )
    parser.add_argument("--incidents", required=True, help="Sessionized incidents CSV.")
    parser.add_argument("--known-index", required=True, help="Pickle with known bug embeddings.")
    parser.add_argument("--ignore-index", required=True, help="Pickle with ignorable pattern embeddings.")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument("--t-known", type=float, default=0.72, help="Known-match similarity threshold.")
    parser.add_argument("--t-ignore", type=float, default=0.70, help="Ignore-match similarity threshold.")
    parser.add_argument(
        "--out-csv",
        default="minilm_ambiguous.csv",
        help="Output CSV path for ambiguous incidents.",
    )
    parser.add_argument(
        "--out-confident-csv",
        default="",
        help="Optional CSV path for confident (label 0/1) MiniLM matches.",
    )
    parser.add_argument(
        "--dedupe-on",
        default="component,message_structure",
        help="Comma-separated column names to deduplicate on (default: component,message_structure). "
             "Set to empty string to retain all rows.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.incidents)
    df = _normalize_incidents(df)

    kb_idx = load_pickle(args.known_index)
    ig_idx = load_pickle(args.ignore_index)

    encoder = SentenceTransformer(args.model)

    include_confident = bool(args.out_confident_csv)
    ambiguous_rows, confident_rows = scan_minilm(
        df,
        kb_idx,
        ig_idx,
        encoder,
        t_known=args.t_known,
        t_ignore=args.t_ignore,
        include_confident=include_confident,
    )

    out_path = os.path.abspath(args.out_csv)
    if not ambiguous_rows:
        pd.DataFrame(columns=[
            "incident_id", "host", "component", "log_window", "message_count",
            "message_structure", "label", "reason", "k0_sim", "i0_sim", "dev_feedback"
        ]).to_csv(out_path, index=False)
        print(f"No ambiguous incidents found. Wrote empty report to {out_path}")
        return

    result_df = pd.DataFrame(ambiguous_rows)
    dedupe_cols = [col.strip() for col in args.dedupe_on.split(",") if col.strip()]
    if dedupe_cols:
        existing_cols = [c for c in dedupe_cols if c in result_df.columns]
        if existing_cols:
            result_df = result_df.drop_duplicates(subset=existing_cols, keep="first")
    result_df.to_csv(out_path, index=False)
    print(f"Wrote {len(result_df)} ambiguous incidents to {out_path}")

    if include_confident:
        confident_path = os.path.abspath(args.out_confident_csv)
        pd.DataFrame(confident_rows).to_csv(confident_path, index=False)
        print(f"Wrote {len(confident_rows)} confident incidents to {confident_path}")


if __name__ == "__main__":
    main()
