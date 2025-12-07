#!/usr/bin/env python3
"""
Aggregate `image_errors_scan_*.log` artifacts into summary CSV tables.

Outputs:
1. Sessionized log message table.
2. systemctl --failed summary table.
"""

import argparse
import csv
import json
import re
import math
import hashlib
import subprocess
from collections import defaultdict

from dataclasses import dataclass, field

# Optional BeautifulSoup for robust HTML parsing
BS4_AVAILABLE = True
try:
    from bs4 import BeautifulSoup
except Exception:
    BS4_AVAILABLE = False
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Base paths (label will be appended dynamically)
RESULTS_BASE = Path(
    "/net/dbdevfssmnt-shared01.dev3fss1phx.databasede3phx.oraclevcn.com/"
    "exadata_dev_image_oeda/integration/logs"
)
PAGE_HTML_BASE_URL = "http://100.70.110.36/integration_logs"
JSON_DECODER = json.JSONDecoder()
LOG_LINE_RE = re.compile(
    r"^(?P<month>\w{3})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<rest>.*)$"
)
FILENAME_RE = re.compile(
    r"^image_errors_scan_(?P<host>[^.]+)\.(?P<stamp>\d{14})\.log$"
)


@dataclass(order=True)
class LogRecord:
    timestamp: datetime
    component: str
    message: str = field(compare=False)
    raw_line: str = field(compare=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sessionize /var/log/messages records from image_errors_scan logs "
            "and summarize systemctl --failed output."
        )
    )
    parser.add_argument(
        "--label",
        default="OSS_MAIN_LINUX.X64_251105",
        help="Label/build identifier (e.g., OSS_MAIN_LINUX.X64_251105). Used to derive root path and page.html URL.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory containing per-view folders with log files. If not provided, derived from --label.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables"),
        help="Destination directory for generated CSV outputs.",
    )
    parser.add_argument(
        "--session-gap",
        type=float,
        default=2.0,
        help="Maximum gap (seconds) between events to keep them in the same session.",
    )
    parser.add_argument(
        "--messages-csv",
        default="image_error_sessions.csv",
        help="Filename for the sessionized messages CSV (relative to --output-dir).",
    )
    parser.add_argument(
        "--systemctl-csv",
        default="systemctl_failed.csv",
        help="Filename for the systemctl summary CSV (relative to --output-dir).",
    )
    parser.add_argument(
        "--page-html",
        type=Path,
        default=None,
        help="Optional HTML file containing view metadata. If not provided, fetched automatically from URL based on --label.",
    )
    parser.add_argument(
        "--page-html-url",
        default=None,
        help="Override the base URL for fetching page.html (default: derived from PAGE_HTML_BASE_URL and --label).",
    )
    parser.add_argument(
        "--skip-page-fetch",
        action="store_true",
        help="Skip automatic fetching of page.html (useful if network unavailable or file already exists).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (prints matched/unmatched views from page.html).",
    )
    parser.add_argument(
        "--structure-column",
        default="message_structure",
        help="Column name to store canonicalized message structure (set empty string to disable).",
    )
    parser.add_argument(
        "--concat-columns",
        default="hostname,viewname,rackname,deployment_type,hardware",
        help="Comma-separated columns to merge when deduplicating incident windows.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable deduplication (keep every session row even if structurally identical).",
    )
    return parser.parse_args()


def fetch_page_html(label: str, output_dir: Path, base_url: str = None) -> Optional[Path]:
    """Fetch page.html from the integration logs server for the given label.
    
    Returns the path to the downloaded page.html file, or None if fetch failed.
    """
    if base_url is None:
        base_url = PAGE_HTML_BASE_URL
    
    url = f"{base_url}/{label}/"
    page_html_path = output_dir / "page.html"
    
    print(f"[INFO] Fetching page.html from {url}")
    try:
        result = subprocess.run(
            ["curl", "-s", url],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        page_html_path.parent.mkdir(parents=True, exist_ok=True)
        page_html_path.write_text(result.stdout, encoding="utf-8")
        print(f"[INFO] Saved page.html to {page_html_path}")
        return page_html_path
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Failed to fetch page.html: {e}")
        return None
    except subprocess.TimeoutExpired:
        print(f"[WARN] Timeout while fetching page.html from {url}")
        return None
    except Exception as e:
        print(f"[WARN] Error fetching page.html: {e}")
        return None


def strip_tags(text: str) -> str:
    """Remove simple HTML tags from a small snippet."""
    return re.sub(r"<[^>]+>", "", text).strip()


def parse_page_html(page_path: Path) -> Dict[str, Dict[str, str]]:
    """Parse a `page.html` produced by the dashboard and return mapping:
    {viewname: {"rackname": ..., "deployment_type": ..., "hardware": ...}}
    Uses BeautifulSoup when available for robustness, otherwise falls back to regex.
    """
    mapping: Dict[str, Dict[str, str]] = {}
    if not page_path or not page_path.exists():
        return mapping
    raw = page_path.read_text(encoding="utf-8", errors="replace")
    if BS4_AVAILABLE:
        soup = BeautifulSoup(raw, "html.parser")
        table = soup.find("table", {"id": "results_status"}) or soup.find("table")
        if not table:
            return mapping
        for tr in table.find_all("tr"):
            # find td elements
            tds = tr.find_all("td")
            if len(tds) < 5:
                continue
            # viewname is the anchor text inside first td
            a = tds[0].find("a")
            viewname = a.get_text(strip=True) if a else tds[0].get_text(strip=True)
            rackname = tds[1].get_text(strip=True)
            deployment = tds[2].get_text(strip=True)
            hardware = tds[4].get_text(strip=True)
            if viewname:
                mapping[viewname] = {
                    "rackname": rackname,
                    "deployment_type": deployment,
                    "hardware": hardware,
                }
        return mapping

    # Fallback to regex-based parsing (previous behavior)
    rows = re.findall(r"<tr>(.*?)</tr>", raw, flags=re.S | re.I)
    for row in rows:
        # Skip header row containing 'View Name' etc.
        if re.search(r"View Name", row, flags=re.I):
            continue
        # Find all <td>...</td> cells
        cells = re.findall(r"<td>(.*?)</td>", row, flags=re.S | re.I)
        if len(cells) < 5:
            continue
        viewcell = strip_tags(cells[0])
        view_match = re.search(r">([^<]+)</a>$", cells[0].strip())
        if view_match:
            viewname = view_match.group(1).strip()
        else:
            viewname = viewcell.split()[:1][0] if viewcell else ""
        rackname = strip_tags(cells[1])
        deployment = strip_tags(cells[2])
        hardware = strip_tags(cells[4])
        if viewname:
            mapping[viewname] = {
                "rackname": rackname,
                "deployment_type": deployment,
                "hardware": hardware,
            }
    return mapping


def iter_json_blocks(raw_text: str) -> Tuple[List[object], str]:
    """Parse sequential JSON values from the beginning of raw_text."""
    idx = 0
    values: List[object] = []
    length = len(raw_text)
    while idx < length:
        while idx < length and raw_text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        if raw_text[idx] not in "[{":
            break
        value, end = JSON_DECODER.raw_decode(raw_text, idx)
        values.append(value)
        idx = end
    return values, raw_text[idx:]


def flatten_match_entries(values: Sequence[object]) -> List[Dict[str, object]]:
    """Collect dict entries from nested list/dict structures."""
    results: List[Dict[str, object]] = []
    for value in values:
        if isinstance(value, dict):
            results.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    results.append(item)
    return results


def extract_base_timestamp(file_name: str) -> datetime:
    match = FILENAME_RE.match(file_name)
    if not match:
        raise ValueError(f"Unrecognized log filename format: {file_name}")
    stamp = match.group("stamp")
    return datetime.strptime(stamp, "%Y%m%d%H%M%S")


def parse_log_line(
    line: str, base_year: int
) -> Tuple[datetime, str, str]:
    """Parse a single /var/log/messages line."""
    match = LOG_LINE_RE.match(line)
    if not match:
        raise ValueError(f"Unable to parse syslog line: {line}")
    dt = datetime.strptime(
        f"{match.group('month')} {match.group('day')} {base_year} {match.group('time')}",
        "%b %d %Y %H:%M:%S",
    )
    rest = match.group("rest")
    component = rest
    message = ""
    if ":" in rest:
        component, message = rest.split(":", 1)
        component = component.strip()
        message = message.strip()
    else:
        component = component.strip()
    component = re.sub(r"\[\d+\]$", "", component).strip()
    return dt, component, message


def sessionize_records(
    records: Sequence[LogRecord], gap_seconds: float
) -> List[List[LogRecord]]:
    """Group records into sessions by component and time gap."""
    grouped: Dict[str, List[LogRecord]] = defaultdict(list)
    for record in records:
        grouped[record.component].append(record)

    sessions: List[List[LogRecord]] = []
    gap = timedelta(seconds=gap_seconds)
    for component, items in grouped.items():
        items.sort()
        current_session: List[LogRecord] = []
        for item in items:
            if (
                current_session
                and item.timestamp - current_session[-1].timestamp > gap
            ):
                sessions.append(current_session)
                current_session = []
            current_session.append(item)
        if current_session:
            sessions.append(current_session)
    return sessions


def parse_systemctl_output(lines: Iterable[str]) -> List[Dict[str, str]]:
    """Parse `systemctl --failed` tabular output into structured rows."""
    rows: List[Dict[str, str]] = []
    collecting = False
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        if re.match(r"\d+\s+loaded units listed", line):
            # Summary line; nothing more to capture.
            break
        if line.lstrip().startswith("UNIT "):
            collecting = True
            continue
        if line.lstrip().startswith("LOAD "):
            # Explanation block; stop collecting unit rows.
            collecting = False
            continue
        if not collecting:
            continue
        normalized = line.lstrip()
        if normalized.startswith("●"):
            normalized = normalized[1:].lstrip()
        parts = normalized.split(None, 4)
        if len(parts) < 5:
            continue
        unit, load, active, sub, description = parts
        rows.append(
            {
                "service": unit,
                "load": load,
                "active": active,
                "sub": sub,
                "description": description,
            }
        )
    return rows


# ---------------------- Post-processing helpers ----------------------

IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_PATTERN = re.compile(r"\b0x[0-9a-fA-F]+\b")
UUIDISH_PATTERN = re.compile(r"\b[0-9a-f]{8,}\b")
TIME_PATTERN = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
QUOTED_SINGLE_PATTERN = re.compile(r"'[^']*'")
QUOTED_DOUBLE_PATTERN = re.compile(r'"[^"]*"')
PID_PATTERN = re.compile(r"\[\d+\]$")

DEFAULT_CONCAT_COLS = [
    "hostname",
    "viewname",
    "rackname",
    "deployment_type",
    "hardware",
]


def normalize_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value}".strip()
    return str(value).strip()


def ensure_component_in_messages(component: str, messages: str) -> str:
    comp = normalize_field(component)
    text = normalize_field(messages)
    if not comp or not text:
        return text
    comp_lower = comp.lower()
    ensured_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.lower().startswith(comp_lower):
            line = f"{comp}: {line}"
        ensured_lines.append(line)
    return "\n".join(ensured_lines)


def canonicalize_component(value: str) -> str:
    text = normalize_field(value).lower()
    text = PID_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "<num>", text)
    return text.strip()


def canonicalize_messages(value: str) -> str:
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


def compute_incident_id(canonical_component: str, canonical_messages: str) -> str:
    key = f"{canonical_component}||{canonical_messages}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def concat_unique(series: Sequence[str]) -> str:
    seen = set()
    ordered = []
    for value in series:
        sval = normalize_field(value)
        if not sval:
            continue
        if sval not in seen:
            seen.add(sval)
            ordered.append(sval)
    return " | ".join(ordered)


def first_nonempty(values: Sequence[str]) -> str:
    for value in values:
        sval = normalize_field(value)
        if sval:
            return sval
    return "" if not values else normalize_field(values[0])


def parse_concat_columns(raw: str) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _session_fieldnames(structure_col: str) -> List[str]:
    fields = [
        "incident_id",
        "hostname",
        "viewname",
        "rackname",
        "deployment_type",
        "hardware",
        "session_window",
        "label",
        "component",
    ]
    if structure_col:
        fields.append(structure_col)
    fields.extend(["message_count", "messages"])
    return fields


def enhance_sessions(
    rows: List[Dict[str, str]],
    structure_col: str = "message_structure",
    concat_cols: Sequence[str] = DEFAULT_CONCAT_COLS,
    dedupe: bool = True,
) -> List[Dict[str, str]]:
    if not rows:
        return rows

    prepared_rows: List[Dict[str, str]] = []
    for row in rows:
        comp = row.get("component", "")
        messages = row.get("messages", "")
        ensured = ensure_component_in_messages(comp, messages)
        row["messages"] = ensured
        canonical_component = canonicalize_component(comp)
        structure = canonicalize_messages(ensured)
        if structure_col:
            row[structure_col] = structure
        row["incident_id"] = compute_incident_id(canonical_component, structure)
        prepared_rows.append(row)

    if not dedupe:
        return prepared_rows

    combined: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in prepared_rows:
        combined[row["incident_id"]].append(row)

    result_rows: List[Dict[str, str]] = []
    for incident_id, group in combined.items():
        if len(group) == 1:
            result_rows.append(group[0])
            continue
        merged = {"incident_id": incident_id}
        all_keys = set().union(*(r.keys() for r in group))
        for key in all_keys:
            if key == "incident_id":
                continue
            values = [g.get(key, "") for g in group]
            if key == "message_count":
                try:
                    merged[key] = str(sum(int(normalize_field(v) or 0) for v in values))
                except ValueError:
                    merged[key] = concat_unique(values)
            elif concat_cols and key in concat_cols:
                merged[key] = concat_unique(values)
            elif structure_col and key == structure_col:
                merged[key] = first_nonempty(values)
            elif key == "messages":
                merged[key] = first_nonempty(values)
            else:
                merged[key] = first_nonempty(values)
        result_rows.append(merged)
    return result_rows


def derive_view_name(root: Path, log_path: Path) -> str:
    rel = log_path.relative_to(root)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    resolved_root = root.resolve()
    if resolved_root.name:
        return resolved_root.name
    return rel.parent.name or "root"


def process_log_file(
    log_path: Path,
    root: Path,
    label: str,
    session_gap: float,
    page_map: Dict[str, Dict[str, str]] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    raw_text = log_path.read_text(encoding="utf-8", errors="replace")
    json_values, remainder = iter_json_blocks(raw_text)
    entries = flatten_match_entries(json_values)

    base_timestamp = extract_base_timestamp(log_path.name)
    message_records: List[LogRecord] = []
    for entry in entries:
        if entry.get("File") != "/var/log/messages":
            continue
        matches = entry.get("MatchFound") or []
        for line in matches:
            try:
                dt, component, message = parse_log_line(line, base_timestamp.year)
            except ValueError:
                continue
            message_records.append(
                LogRecord(
                    timestamp=dt,
                    component=component,
                    message=message,
                    raw_line=line,
                )
            )

    hostname = FILENAME_RE.match(log_path.name).group("host")  # type: ignore[union-attr]
    view_name = derive_view_name(root, log_path)
    # If page_map provides a mapping for view_name, use the metadata
    meta = (page_map or {}).get(view_name, {})
    rackname = meta.get("rackname", "")
    deployment_type = meta.get("deployment_type", "")
    hardware = meta.get("hardware", "")
    sessions = sessionize_records(message_records, session_gap)

    session_rows: List[Dict[str, str]] = []
    for session in sessions:
        start = session[0].timestamp
        end = session[-1].timestamp
        if start == end:
            window = start.strftime("%Y-%m-%d %H:%M:%S")
        else:
            window = f"{start.strftime('%Y-%m-%d %H:%M:%S')} - {end.strftime('%Y-%m-%d %H:%M:%S')}"
        consolidated: List[str] = []
        for record in session:
            # include component together with message (e.g. "journal[233863]: Failed to open ...")
            if record.message:
                consolidated.append(f"{record.component}: {record.message}")
            else:
                consolidated.append(record.raw_line)
        messages = "\n".join(consolidated)
        row = {
            "hostname": hostname,
            "viewname": view_name,
            "session_window": window,
            "label": label,
            "component": session[0].component,
            "message_count": str(len(session)),
            "messages": messages,
            "rackname": rackname,
            "deployment_type": deployment_type,
            "hardware": hardware,
        }
        session_rows.append(row)

    systemctl_rows: List[Dict[str, str]] = []
    if remainder.strip():
        systemctl_rows = parse_systemctl_output(remainder.splitlines())
    for row in systemctl_rows:
        row.update(
            {
                "hostname": hostname,
                "viewname": view_name,
                "rackname": rackname,
                "deployment_type": deployment_type,
                "hardware": hardware,
            }
        )

    return session_rows, systemctl_rows


def collect_logs(
    root: Path,
    label: str,
    session_gap: float,
    page_html: Path = None,
    verbose: bool = False,
    structure_col: str = "message_structure",
    concat_cols: Sequence[str] = DEFAULT_CONCAT_COLS,
    dedupe: bool = True,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    message_rows: List[Dict[str, str]] = []
    systemctl_rows: List[Dict[str, str]] = []

    # parse optional page.html for view metadata if present
    page_map = parse_page_html(page_html) if page_html else {}

    for log_path in sorted(root.rglob("image_errors_scan_*.log")):
        if not log_path.is_file():
            continue
        sessions, systemctl = process_log_file(
            log_path, root, label, session_gap, page_map
        )
        message_rows.extend(sessions)
        systemctl_rows.extend(systemctl)

    message_rows = enhance_sessions(
        message_rows,
        structure_col=structure_col,
        concat_cols=concat_cols,
        dedupe=dedupe,
    )
    # If verbose, print which views from page_map were matched and which were not
    if verbose:
        found_views = set(r["viewname"] for r in message_rows + systemctl_rows if r.get("viewname"))
        mapped_views = set(page_map.keys())
        matched = sorted(list(found_views & mapped_views))
        unmatched = sorted(list(mapped_views - found_views))
        print(f"[INFO] page.html parsing method: {'bs4' if BS4_AVAILABLE else 'regex-fallback'}")
        print(f"[INFO] views matched in logs ({len(matched)}): {matched}")
        print(f"[INFO] views present in page.html but not found in logs ({len(unmatched)}): {unmatched}")
    return message_rows, systemctl_rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    
    # Derive root path from label if not explicitly provided
    if args.root is None:
        args.root = RESULTS_BASE / args.label
        print(f"[INFO] Using derived root path: {args.root}")
    
    # Determine page.html path and fetch if needed
    page_html_path = args.page_html
    if page_html_path is None and not args.skip_page_fetch:
        # Fetch page.html automatically
        page_html_path = fetch_page_html(
            args.label, 
            args.output_dir, 
            base_url=args.page_html_url
        )
    elif page_html_path and not page_html_path.exists() and not args.skip_page_fetch:
        print(f"[WARN] Specified page.html not found at {page_html_path}, attempting fetch...")
        page_html_path = fetch_page_html(
            args.label,
            args.output_dir,
            base_url=args.page_html_url
        )
    
    # If bs4 not available, and user requested verbose, suggest installing it
    if args.verbose and not BS4_AVAILABLE:
        print("[WARN] BeautifulSoup (bs4) not available; using regex fallback for HTML parsing.")
        print("[WARN] Install with: pip install beautifulsoup4")

    concat_cols = parse_concat_columns(args.concat_columns)
    dedupe = not args.no_dedupe

    # collect logs while passing page_html path and verbosity
    message_rows, systemctl_rows = collect_logs(
        args.root,
        args.label,
        args.session_gap,
        page_html_path,
        args.verbose,
        structure_col=args.structure_column,
        concat_cols=concat_cols,
        dedupe=dedupe,
    )

    messages_path = args.output_dir / args.messages_csv
    systemctl_path = args.output_dir / args.systemctl_csv

    write_csv(
        messages_path,
        _session_fieldnames(args.structure_column),
        message_rows,
    )
    write_csv(
        systemctl_path,
        (
            "hostname",
            "viewname",
            "rackname",
            "deployment_type",
            "hardware",
            "service",
            "load",
            "active",
            "sub",
            "description",
        ),
        systemctl_rows,
    )


if __name__ == "__main__":
    main()
