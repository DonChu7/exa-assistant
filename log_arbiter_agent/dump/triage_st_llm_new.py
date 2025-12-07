#!/usr/bin/env python3
# triage_st_llm.py
import argparse, os, json, re, numpy as np, pandas as pd
from typing import Dict, Any, Tuple, List
from sentence_transformers import SentenceTransformer
from utils import load_pickle, normalize_log_window

def cosine_sim(a: np.ndarray, b: np.ndarray):
    return np.dot(b, a)

def topk(vec: np.ndarray, mat: np.ndarray, k=3):
    sims = cosine_sim(vec, mat)
    idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]

def should_arbitrate(k0: float, i0: float, text: str,
                     t_known=0.72, t_ignore=0.70,
                     near=0.05, min_suspect=0.55) -> bool:
    if (k0 < t_known and i0 < t_ignore) and max(k0, i0) >= min_suspect:
        return True
    if abs(k0 - i0) < near and max(k0, i0) >= min_suspect:
        return True
    if len(text) > 1000:
        return True
    if len(re.findall(r"(failed|error|exception|panic)", text, flags=re.I)) >= 6:
        return True
    return False

def call_llm_arbiter(incident_text: str,
                     known_snips: str,
                     ignore_snips: str,
                     provider: str,
                     model: str,
                     api_key_env: str,
                     timeout: float = 30.0) -> Tuple[int, str]:
    """
    Returns (label, reason) where label in {1,0,-1}
    provider: 'openai' supported out of the box.
    """
    prov = provider.lower().strip()
    if prov != "openai":
        raise RuntimeError("Only provider=openai is wired in this snippet.")

    import requests
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing {api_key_env} env var for LLM arbitration.")

    system_prompt = (
        "You are a log triage arbiter for Exadata/Exascale infrastructure. "
        "Decide if the INCIDENT is a reproduced/known bug (label 1), "
        "ignorable/noise (label 0), or needs triage (label -1). "
        "Use KNOWN and IGNORE snippets as weak guidance (they may be imperfect). "
        "Return STRICT JSON with keys: label (one of 1,0,-1) and reason (<=200 chars)."
    )
    user_prompt = f"""
INCIDENT:
```
{incident_text}
```

TOP KNOWN (examples):
```
{known_snips or "N/A"}
```

TOP IGNORE (examples):
```
{ignore_snips or "N/A"}
```

Choose one label: 1 (known), 0 (ignorable), -1 (needs triage).
Return JSON only, no extra text.
"""

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        label = int(data.get("label", -1))
        if label not in (1,0,-1): label = -1
        reason = str(data.get("reason", ""))[:300]
        return label, reason
    except Exception as e:
        return -1, f"arbiter_error:{e}"

def decide_with_embeddings(row: Dict[str, Any], kb_idx, ig_idx, model,
                           t_known=0.72, t_ignore=0.70):
    raw_text = row.get('log_window', '') or ''
    norm_text = normalize_log_window(str(raw_text))
    v = model.encode([norm_text], normalize_embeddings=True)[0].astype(np.float32)

    kbest = topk(v, kb_idx['embs'], k=3) if kb_idx['embs'].size else []
    ibest = topk(v, ig_idx['embs'], k=3) if ig_idx['embs'].size else []
    (k0_idx, k0_sim) = (kbest[0] if len(kbest) else (-1, 0.0))
    (i0_idx, i0_sim) = (ibest[0] if len(ibest) else (-1, 0.0))

    if k0_sim >= max(t_known, i0_sim + 0.05):
        bug = kb_idx['df'].iloc[k0_idx].to_dict()
        return dict(
            label=1,
            confidence=float(k0_sim),
            reason="known-emb-match",
            evidence=[bug.get('bug_id','')],
            raw_text=raw_text,
            norm_text=norm_text,
            kbest=kbest,
            ibest=ibest,
            k0_sim=float(k0_sim),
            i0_sim=float(i0_sim)
        )
    if i0_sim >= t_ignore:
        ign = ig_idx['df'].iloc[i0_idx].to_dict()
        return dict(
            label=0,
            confidence=float(i0_sim),
            reason="ignore-emb-match",
            evidence=[ign.get('pattern_id','')],
            raw_text=raw_text,
            norm_text=norm_text,
            kbest=kbest,
            ibest=ibest,
            k0_sim=float(k0_sim),
            i0_sim=float(i0_sim)
        )

    return dict(
        label=-2,
        confidence=float(1 - max(k0_sim, i0_sim)),
        reason="needs-arbitration",
        evidence=[],
        raw_text=raw_text,
        norm_text=norm_text,
        kbest=kbest,
        ibest=ibest,
        k0_sim=float(k0_sim),
        i0_sim=float(i0_sim)
    )

def fmt_examples(df: pd.DataFrame, pairs: List[Tuple[int,float]], id_col: str, ctx_col: str) -> str:
    out = []
    for i, sim in pairs[:2]:
        rec = df.iloc[i]
        out.append(f"[sim={sim:.3f}] {id_col}={rec.get(id_col, '')} ::\n{str(rec.get(ctx_col, ''))[:1500]}")
    return "\n\n".join(out)

def main(incidents_path: str, known_idx_path: str, ignore_idx_path: str,
         outdir: str, model_name: str,
         use_llm: bool, llm_provider: str, llm_model: str, llm_api_key_env: str,
         t_known: float = 0.72, t_ignore: float = 0.70):

    kb_idx = load_pickle(known_idx_path)
    ig_idx = load_pickle(ignore_idx_path)
    enc = SentenceTransformer(model_name)

    df = pd.read_csv(incidents_path)
    os.makedirs(outdir, exist_ok=True)

    outs = {1: [], 0: [], -1: []}
    rows = []
    review_rows: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        base = decide_with_embeddings(r, kb_idx, ig_idx, enc, t_known=t_known, t_ignore=t_ignore)
        baseline_label = base.get('label', -1)
        raw_text = base.get('raw_text', r.get('log_window', '') or '')
        evidence_list = base.get('evidence', []) or []
        kbest = base.get('kbest', []) or []
        ibest = base.get('ibest', []) or []
        k0 = float(base.get('k0_sim', 0.0))
        i0 = float(base.get('i0_sim', 0.0))

        if baseline_label == -2:
            baseline_label = -1
            baseline_reason = "novel-or-ambiguous"
            baseline_confidence = float(base.get('confidence', 0.0))
            evidence_list = []
        else:
            baseline_reason = base.get('reason', '')
            baseline_confidence = float(base.get('confidence', 0.0))

        evidence = ";".join(str(ev) for ev in evidence_list)

        llm_status = ''
        llm_label = ''
        llm_reason = ''

        if baseline_label == -1:
            top_bug_id = ''
            if kbest:
                try:
                    top_bug_id = str(kb_idx['df'].iloc[kbest[0][0]].get('bug_id', ''))
                except Exception:
                    top_bug_id = ''
            top_pattern_id = ''
            if ibest:
                try:
                    top_pattern_id = str(ig_idx['df'].iloc[ibest[0][0]].get('pattern_id', ''))
                except Exception:
                    top_pattern_id = ''

            known_snips = fmt_examples(kb_idx['df'], kbest, 'bug_id', 'context_window') if kbest else ""
            ignore_snips = fmt_examples(ig_idx['df'], ibest, 'pattern_id', 'context_window') if ibest else ""

            if use_llm:
                llm_status = "skipped"
                if should_arbitrate(k0, i0, raw_text, t_known=t_known, t_ignore=t_ignore):
                    arb_label, arb_reason = call_llm_arbiter(
                        incident_text=raw_text,
                        known_snips=known_snips,
                        ignore_snips=ignore_snips,
                        provider=llm_provider,
                        model=llm_model,
                        api_key_env=llm_api_key_env
                    )
                    if isinstance(arb_reason, str) and arb_reason.startswith("arbiter_error:"):
                        llm_status = "error"
                        llm_reason = arb_reason
                    else:
                        llm_status = "decided" if arb_label in (1, 0) else "novel"
                        llm_label = str(arb_label)
                        llm_reason = arb_reason
                else:
                    llm_reason = "skipped_by_threshold"
            else:
                llm_status = "disabled"
                llm_reason = "llm_disabled"

            review_rows.append(dict(
                incident_id=r.get('incident_id', ''),
                host=r.get('host', ''),
                component=r.get('component', ''),
                log_window=raw_text,
                baseline_confidence=round(baseline_confidence, 4),
                max_known_sim=round(k0, 4),
                max_ignore_sim=round(i0, 4),
                baseline_reason=baseline_reason,
                baseline_evidence=evidence,
                top_bug_id=top_bug_id,
                top_pattern_id=top_pattern_id,
                known_examples=known_snips,
                ignore_examples=ignore_snips,
                llm_status=llm_status,
                llm_label=llm_label,
                llm_reason=llm_reason
            ))
        else:
            llm_status = ''
            llm_label = ''
            llm_reason = ''

        rec = dict(
            incident_id=r.get('incident_id', ''),
            host=r.get('host',''),
            component=r.get('component',''),
            log_window=raw_text,
            label=int(baseline_label),
            confidence=round(baseline_confidence, 4),
            reason=baseline_reason,
            evidence=evidence,
            llm_status=llm_status,
            llm_label=llm_label,
            llm_reason=llm_reason
        )
        rows.append(rec)
        outs[int(baseline_label)].append(rec)

    for lab, items in outs.items():
        with open(os.path.join(outdir, f"label_{lab}.jsonl"), "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")

    pd.DataFrame(rows).to_csv(os.path.join(outdir, "triage_all.csv"), index=False)

    review_df = pd.DataFrame(review_rows, columns=[
        "incident_id",
        "host",
        "component",
        "log_window",
        "baseline_confidence",
        "max_known_sim",
        "max_ignore_sim",
        "baseline_reason",
        "baseline_evidence",
        "top_bug_id",
        "top_pattern_id",
        "known_examples",
        "ignore_examples",
        "llm_status",
        "llm_label",
        "llm_reason"
    ])
    review_df.to_csv(os.path.join(outdir, "arbiter_review.csv"), index=False)
    print(f"Done. Wrote {sum(len(v) for v in outs.values())} decisions into {outdir} (review queue: {len(review_rows)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", required=True)
    ap.add_argument("--known-index", required=True)
    ap.add_argument("--ignore-index", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--llm", action="store_true", help="Enable LLM arbitration for ambiguous cases")
    ap.add_argument("--llm-provider", default="openai")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--t-known", type=float, default=0.72)
    ap.add_argument("--t-ignore", type=float, default=0.70)
    args = ap.parse_args()
    main(args.incidents, args.known_index, args.ignore_index, args.outdir,
         args.model, args.llm, args.llm_provider, args.llm_model, args.llm_api_key_env,
         t_known=args.t_known, t_ignore=args.t_ignore)
