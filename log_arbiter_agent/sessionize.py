#!/usr/bin/env python3
import argparse, csv, os, sys, re, uuid
import pandas as pd
from utils import extract_component, extract_pid, parse_timestamp_prefix, time_to_seconds

def sessionize(input_path: str, output_path: str, gap_seconds: int = 10, max_span_seconds: int = 30):
    """
    Reads filtered log lines (optionally with a trailing label), and builds incident windows using:
      - time gap > gap_seconds
      - OR component changes
    Output CSV columns: incident_id, host, component, pid, log_window, label
    """
    rows = []
    with open(input_path, 'r', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line: 
                continue
            # Expect format like: "Jul 27 18:04:39 host ...",label?
            label = ''
            if line.endswith(', 1') or line.endswith(', 0') or line.endswith(', -1'):
                # naive split on last comma
                try:
                    idx = line.rfind(',')
                    label = line[idx+1:].strip()
                    line = line[:idx].strip()
                except Exception:
                    pass

            clean_line = line.lstrip("\"'")

            # host extraction heuristic
            parts = line.split()
            host = ''
            if len(parts) > 3:
                host = parts[3] if parts[3].find(':') == -1 else parts[2]

            ts = parse_timestamp_prefix(clean_line) or ''
            tsec = time_to_seconds(ts)
            comp = extract_component(line)
            pid = extract_pid(line)

            rows.append(dict(ts=ts, tsec=tsec, host=host, component=comp, pid=pid, line=line, label=label))

    # Build windows
    incidents = []
    cur = None

    for r in rows:
        # start new?
        start_new = False
        if cur is None:
            start_new = True
        else:
            gap = (r['tsec'] - cur['last_tsec']) if (r['tsec'] >=0 and cur['last_tsec']>=0) else 0
            if (gap > gap_seconds or 
                r['component'] != cur['component']):
                start_new = True

        if start_new:
            if cur is not None:
                incidents.append(cur)
            cur = dict(
                incident_id=str(uuid.uuid4())[:8],
                host=r['host'],
                start_time=r['ts'],
                end_time=r['ts'],
                start_tsec=r['tsec'],
                last_tsec=r['tsec'],
                component=r['component'],
                pid=r['pid'],
                lines=[r['line']],
                labels=[r['label']] if r['label'] else []
            )
        else:
            cur['lines'].append(r['line'])
            cur['last_tsec'] = r['tsec']
            cur['end_time'] = r['ts']
            if r['label']:
                cur['labels'].append(r['label'])

    if cur is not None:
        incidents.append(cur)

    out_rows = []
    for inc in incidents:
        uniq = sorted(set(inc.get('labels', [])))
        label = ''
        if len(uniq) == 1:
            label = uniq[0]

        out_rows.append(
            dict(
                incident_id=inc['incident_id'],
                host=inc['host'],
                component=inc['component'],
                pid=inc['pid'],
                log_window='\n'.join(inc['lines']),
                label=label
            )
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(out_rows).to_csv(output_path, index=False)
    print(f"Wrote {len(out_rows)} incidents to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Filtered log file (one entry per line, may end with , <label>)")
    ap.add_argument("--output", required=True, help="Output CSV for incidents")
    ap.add_argument("--gap-seconds", type=int, default=10)
    ap.add_argument("--max-span-seconds", type=int, default=30)
    args = ap.parse_args()
    sessionize(args.input, args.output, args.gap_seconds, args.max_span_seconds)
