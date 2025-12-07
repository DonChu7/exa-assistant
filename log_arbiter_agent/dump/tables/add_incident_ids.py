#!/usr/bin/env python3
"""
Utility to attach a deterministic incident_id to sessionized log rows using structural normalization
of component + messages, and optionally deduplicate them.

Default behavior:
  - Read tables/image_error_sessions.csv
  - Canonicalize each row's component/messages (masking numbers, IPs, etc.)
  - Add an incident_id derived from the canonical structure
  - Drop duplicate incident_ids (combining key metadata columns via concatenation of unique values)
  - Write tables/image_error_sessions_dedup.csv

Use --keep-duplicates to retain all rows while still adding the incident_id column.
"""

import argparse
import hashlib
import math
import re
from pathlib import Path

import pandas as pd


IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_PATTERN = re.compile(r"\b0x[0-9a-fA-F]+\b")
UUIDISH_PATTERN = re.compile(r"\b[0-9a-f]{8,}\b")
TIME_PATTERN = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
QUOTED_SINGLE_PATTERN = re.compile(r"'[^']*'")
QUOTED_DOUBLE_PATTERN = re.compile(r'"[^"]*"')
PID_PATTERN = re.compile(r"\[\d+\]$")


def compute_incident_id(canonical_component: str, canonical_messages: str) -> str:
    """Hash canonicalized component/message structure for stable IDs."""
    key = f"{canonical_component}||{canonical_messages}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def normalize_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value}".strip()
    return str(value).strip()


def canonicalize_component(value) -> str:
    text = normalize_field(value).lower()
    text = PID_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "<num>", text)
    return text.strip()


def ensure_component_in_messages(component, messages) -> str:
    comp = normalize_field(component)
    text = normalize_field(messages)
    if not comp or not text:
        return text
    comp_lower = comp.lower()
    result_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.lower().startswith(comp_lower):
            line = f"{comp}: {line}"
        result_lines.append(line)
    return "\n".join(result_lines)


def canonicalize_messages(value) -> str:
    text = normalize_field(value).lower()
    lines = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = IP_PATTERN.sub("<ip>", line)
        line = HEX_PATTERN.sub("<hex>", line)
        line = UUIDISH_PATTERN.sub("<hex>", line)
        line = QUOTED_SINGLE_PATTERN.sub("'<str>'", line)
        line = QUOTED_DOUBLE_PATTERN.sub('"<str>"', line)
        line = DATE_PATTERN.sub("<date>", line)
        line = TIME_PATTERN.sub("<time>", line)
        line = re.sub(r"\d+", "<num>", line)
        line = re.sub(r"\s+", " ", line)
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return "\n".join(lines)


def parse_concat_columns(raw: str) -> list:
    if not raw:
        return []
    return [col.strip() for col in raw.split(",") if col.strip()]


def concat_unique(series: pd.Series) -> str:
    seen = set()
    ordered = []
    for val in series:
        sval = normalize_field(val)
        if not sval:
            continue
        if sval not in seen:
            seen.add(sval)
            ordered.append(sval)
    return " | ".join(ordered)


def first_nonempty(series: pd.Series):
    for val in series:
        sval = normalize_field(val)
        if sval:
            return val if not isinstance(val, float) or not math.isnan(val) else ""
    return "" if series.empty else series.iloc[0]


def aggregate_duplicates(df: pd.DataFrame, structure_col: str, concat_cols: list) -> pd.DataFrame:
    columns = list(df.columns)
    records = []
    for incident_id, group in df.groupby("incident_id", sort=False):
        row = {"incident_id": incident_id}
        for col in columns:
            if col == "incident_id":
                continue
            if concat_cols and col in concat_cols:
                row[col] = concat_unique(group[col])
            elif col == "message_count":
                try:
                    row[col] = int(group[col].fillna(0).astype(int).sum())
                except Exception:
                    row[col] = group[col].apply(normalize_field).str.cat(sep=" | ")
            elif structure_col and col == structure_col:
                row[col] = first_nonempty(group[col])
            else:
                row[col] = first_nonempty(group[col])
        records.append(row)
    return pd.DataFrame(records, columns=columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add incident_id hash to sessionized logs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tables/image_error_sessions.csv"),
        help="CSV produced by collect_image_scan.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tables/image_error_sessions_dedup.csv"),
        help="Destination CSV with incident_id column.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Retain duplicate incident_ids instead of dropping them.",
    )
    parser.add_argument(
        "--structure-column",
        default="message_structure",
        help="Optional column name to store canonicalized message structure (use '' to skip).",
    )
    parser.add_argument(
        "--concat-columns",
        default="hostname,viewname,rackname,deployment_type,hardware",
        help="Comma-separated columns to concatenate when deduplicating. Ignored if --keep-duplicates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    structure_col = args.structure_column
    concat_cols = parse_concat_columns(args.concat_columns)
    if df.empty:
        df["incident_id"] = pd.Series(dtype=str)
        if structure_col:
            df[structure_col] = pd.Series(dtype=str)
    else:
        components_series = df.get("component")
        messages_series = df.get("messages")
        if components_series is None or messages_series is None:
            raise ValueError("Input CSV must contain 'component' and 'messages' columns.")
        df["messages"] = [
            ensure_component_in_messages(comp, msg)
            for comp, msg in zip(components_series, messages_series)
        ]
        structures = df["messages"].apply(canonicalize_messages)
        canonical_components = df["component"].apply(canonicalize_component)
        df["incident_id"] = [
            compute_incident_id(comp, struct)
            for comp, struct in zip(canonical_components, structures)
        ]
        if structure_col:
            df[structure_col] = structures
        if not args.keep_duplicates:
            df = aggregate_duplicates(df, structure_col, concat_cols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
