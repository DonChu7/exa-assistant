#!/usr/bin/env python3
import argparse, os, json, numpy as np, pandas as pd, re
from typing import Dict, Any
from utils import load_pickle, normalize_log_window
from sentence_transformers import SentenceTransformer

def cosine_sim(a: np.ndarray, b: np.ndarray):
    # a: (d,), b: (n,d)
    return np.dot(b, a)  # if both are normalized, this is cosine

def topk(vec: np.ndarray, mat: np.ndarray, k=3):
    sims = cosine_sim(vec, mat)
    idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]

def summarize_records(records, id_field: str):
    groups = {}
    for rec in records:
        evid_raw = rec.get('evidence', '') or ''
        evid_items = [e.strip() for e in evid_raw.split(';') if e.strip()]
        key = evid_items[0] if evid_items else ''
        if not key:
            key = 'UNKNOWN'
        entry = groups.setdefault(key, dict(reason=rec.get('reason', ''), records=[]))
        entry['records'].append(rec)

    summary = []
    for key, data in groups.items():
        recs = data['records']
        hosts = sorted({r.get('host', '') for r in recs if r.get('host', '')})
        components = sorted({r.get('component', '') for r in recs if r.get('component', '')})
        confidences = []
        for r in recs:
            try:
                confidences.append(float(r.get('confidence', 0)))
            except (TypeError, ValueError):
                pass
        summary.append(dict(
            **{id_field: key},
            reason=data.get('reason', ''),
            occurrences=len(recs),
            incident_ids=";".join(r.get('incident_id', '') for r in recs),
            hosts=";".join(hosts),
            components=";".join(components),
            max_confidence=max(confidences) if confidences else '',
            min_confidence=min(confidences) if confidences else ''
        ))
    return summary

def decide(row: Dict[str, Any], kb_idx, ig_idx, model, t_known=0.72, t_ignore=0.70):
    raw_text = row.get('log_window', '') or ''
    text = normalize_log_window(str(raw_text))
    vec = model.encode([text], normalize_embeddings=True)
    v = vec[0].astype(np.float32)

    kbest = topk(v, kb_idx['embs'], k=3)
    ibest = topk(v, ig_idx['embs'], k=3)

    # Regex rules for ignorable (if present)
    rule_hit = None
    if 'rule_regex' in ig_idx['df'].columns:
        for i, rec in ig_idx['df'].iterrows():
            rx = rec.get('rule_regex')
            if isinstance(rx, str) and rx:
                try:
                    if re.search(rx, raw_text, re.I):
                        rule_hit = dict(pattern_id=rec.get('pattern_id', f'IGN-{i}'), rule_regex=rx)
                        break
                except re.error:
                    pass

    # Decision: prefer strong ignore rule
    if rule_hit:
        return dict(label=0, confidence=0.95, reason=f"ignore-rule:{rule_hit['pattern_id']}", evidence=[rule_hit['rule_regex']])

    (k0_idx, k0_sim) = (kbest[0] if len(kbest) else (-1, 0.0))
    (i0_idx, i0_sim) = (ibest[0] if len(ibest) else (-1, 0.0))

    if k0_sim >= max(t_known, i0_sim + 0.05):
        bug = kb_idx['df'].iloc[k0_idx].to_dict()
        return dict(label=1, confidence=float(k0_sim), reason="known-emb-match", evidence=[bug.get('bug_id','')])

    if i0_sim >= t_ignore:
        ign = ig_idx['df'].iloc[i0_idx].to_dict()
        return dict(label=0, confidence=float(i0_sim), reason="ignore-emb-match", evidence=[ign.get('pattern_id','')])

    return dict(label=-1, confidence=float(1 - max(k0_sim, i0_sim)), reason="novel-or-ambiguous", evidence=[])

def main(incidents_path: str, known_idx_path: str, ignore_idx_path: str, outdir: str, model_name: str):
    kb_idx = load_pickle(known_idx_path)
    ig_idx = load_pickle(ignore_idx_path)

    # Ensure we use the same embedding model
    model = SentenceTransformer(model_name)

    df = pd.read_csv(incidents_path)
    os.makedirs(outdir, exist_ok=True)

    outs = {1: [], 0: [], -1: []}
    rows = []
    for _, r in df.iterrows():
        decision = decide(r, kb_idx, ig_idx, model)
        incident_fields = {}
        for key, value in r.items():
            if key == 'label':
                continue
            if pd.isna(value):
                incident_fields[key] = None
            else:
                incident_fields[key] = value

        evidence = ";".join(str(ev) for ev in decision['evidence'])
        rec = dict(
            **incident_fields,
            label=int(decision['label']),
            confidence=round(decision['confidence'], 4),
            reason=decision['reason'],
            evidence=evidence
        )
        rows.append(rec)
        outs[int(decision['label'])].append(rec)

    for lab, items in outs.items():
        label_path = os.path.join(outdir, f"label_{lab}")
        with open(label_path + ".jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        pd.DataFrame(items).to_csv(label_path + ".csv", index=False)

    pd.DataFrame(rows).to_csv(os.path.join(outdir, "triage_all.csv"), index=False)

    summaries = [
        (1, "bug_id"),
        (0, "pattern_id"),
    ]
    for lab, field in summaries:
        summary_rows = summarize_records(outs[lab], field)
        pd.DataFrame(summary_rows).to_csv(os.path.join(outdir, f"label_{lab}_summary.csv"), index=False)
    print(f"Done. Wrote {sum(len(v) for v in outs.values())} decisions into {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", required=True)
    ap.add_argument("--known-index", required=True)
    ap.add_argument("--ignore-index", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    main(args.incidents, args.known_index, args.ignore_index, args.outdir, args.model)
