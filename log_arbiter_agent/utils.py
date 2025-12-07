import re, json, os, math, pickle, hashlib, time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Basic component and PID heuristics
COMPONENT_PATTERNS = [
    (re.compile(r'\b(kernel):', re.I), 'kernel'),
    (re.compile(r'\b(systemd(?:-.*)?)(?:\[|\:)', re.I), 'systemd'),
    (re.compile(r'\bNetworkManager\b', re.I), 'NetworkManager'),
    (re.compile(r'\bdbus-daemon\b', re.I), 'dbus-daemon'),
    (re.compile(r'\bmultipathd\b', re.I), 'multipathd'),
    (re.compile(r'\bexadata-(?:virtmon|qmpmon)\.service\b', re.I), 'exadata-service'),
    (re.compile(r'\bmlx5_core\b', re.I), 'mlx5_core'),
    (re.compile(r'\bRDS/IB\b', re.I), 'RDS/IB'),
    (re.compile(r'\bcell(sr|adm)\b', re.I), 'cellsrv'),
]

SYSLOG_COMPONENT_RE = re.compile(r'^([^\s:]+?)(?:\[\d+\])?:')
HOST_TOKEN_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9\.-]{0,}$')

PID_RE = re.compile(r'\[(\d+)\]')

def extract_component(line: str) -> str:
    stripped = line.lstrip(" '\"")

    # Parse syslog-style component after timestamp + host (typical case)
    working = stripped
    ts = parse_timestamp_prefix(stripped)
    if ts:
        working = stripped[len(ts):].lstrip()
    parts = working.split(None, 1)
    if len(parts) == 2:
        working = parts[1]
    elif parts:
        working = parts[0]
    else:
        working = ''
    working = working.lstrip()
    m = SYSLOG_COMPONENT_RE.match(working)
    if m:
        return m.group(1)

    for rx, name in COMPONENT_PATTERNS:
        if rx.search(stripped):
            return name

    if 'kernel:' in stripped:
        return 'kernel'
    return 'unknown'

def strip_host_from_line(line: str) -> str:
    stripped = line.lstrip(" '\"")
    ts = parse_timestamp_prefix(stripped)
    rest = stripped[len(ts):].lstrip() if ts else stripped
    if not rest:
        return stripped

    parts = rest.split(None, 1)
    if parts:
        token = parts[0]
        # Host tokens usually lack ':' and fit hostname charset; avoid eating components like 'kernel:'
        if HOST_TOKEN_RE.match(token) and not token.endswith(':'):
            rest = parts[1] if len(parts) == 2 else ''

    # Remove trailing quote artefacts that often come from CSV contexts
    rest = rest.rstrip('"') if rest else rest

    if not rest:
        # If stripping host left us empty, fall back to the original stripped line (sans trailing quotes)
        return stripped.rstrip('"')

    return rest

def normalize_log_window(text: str) -> str:
    lines = text.splitlines()
    normalized = [strip_host_from_line(line) for line in lines]
    return "\n".join(normalized)

def extract_pid(line: str) -> str:
    m = PID_RE.search(line)
    return m.group(1) if m else 'NA'

TS_RXES = [
    # "Jul 27 18:04:39"
    re.compile(r'^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}'),
]

def parse_timestamp_prefix(line: str) -> Optional[str]:
    for rx in TS_RXES:
        m = rx.match(line)
        if m:
            return m.group(0)
    return None

def time_to_seconds(ts: str) -> int:
    # Format: "Jul 27 18:04:39" -> seconds since midnight; month/day ignored for gap calc
    try:
        parts = ts.split()
        h, m, s = parts[-1].split(':')
        return int(h)*3600 + int(m)*60 + int(s)
    except Exception:
        return -1

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = (a*b).sum()
    da = np.linalg.norm(a); db = np.linalg.norm(b)
    if da == 0 or db == 0: return 0.0
    return float(num/(da*db))

def dump_pickle(obj, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
