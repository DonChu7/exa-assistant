#!/usr/bin/env python3
"""
triage_st_llm_web.py
--------------------
Exadata/Exascale log triage with:
  • MiniLM embedding filter (fast path): known vs ignorable using cosine similarity
  • LLM arbitration for ambiguous (-1) cases using OpenAI models with built-in web retrieval
      - Default model: gpt-4o-mini-search   (supports web search)
      - Future upgrade: gpt-4.5-search      (drop-in via --llm-model gpt-4.5-search)

Outputs:
  • CSV file with all decisions at: <outdir>/triaged_results_web.csv
  • Per-label JSONL files at:       <outdir>/label_{1|0|-1}.jsonl

Environment:
  • OPENAI_API_KEY must be set

Usage example:
  python triage_st_llm_web.py \
    --incidents data/incidents.csv \
    --known-index data/.known_miniLM.pkl \
    --ignore-index data/.ignore_miniLM.pkl \
    --outdir data/outputs \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --t-known 0.75 \
    --t-ignore 0.73 \
    --llm-model gpt-4o-mini-search

Switching to gpt-4.5-search later:
  python triage_st_llm_web.py ... --llm-model gpt-4.5-search

Notes:
  • Only windows deemed ambiguous by embeddings (internal label = -2) are sent to the LLM arbiter.
  • The arbiter must return STRICT JSON:
        {"label": 1|0|-1, "confidence": 0.0..1.0, "reason": ">=10 lines", "sources": ["url1", ...]}
  • The script enforces >=10 lines reasoning (adds a note if shorter) and stores citations in `evidence`.
  • Minimal/no console printing; results are written to files.
"""

import argparse, os, json, re, time, hashlib, threading
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

# Embeddings (MiniLM) for similarity
from sentence_transformers import SentenceTransformer

# OpenAI SDK v2
from openai import OpenAI

# Local helper expected to exist (same as your previous repo)
# It should provide load_pickle() that loads {"df": DataFrame, "embs": np.ndarray}
from utils import load_pickle


# ------------------------- Embedding helpers -------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray):
    return np.dot(b, a)

def topk(vec: np.ndarray, mat: np.ndarray, k=3):
    if mat is None or getattr(mat, "size", 0) == 0:
        return []
    sims = cosine_sim(vec, mat)
    idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]


# ------------------------- Arbitration trigger -------------------------
def should_arbitrate(k0: float, i0: float, text: str,
                     t_known=0.72, t_ignore=0.70,
                     near=0.05) -> bool:
    # arbitrate if neither side is confident OR scores are too close
    if (k0 < t_known and i0 < t_ignore):
        return True
    if abs(k0 - i0) < near:
        return True
    if len(text) > 1000:
        return True
    if len(re.findall(r"(failed|error|exception|panic)", text, flags=re.I)) >= 6:
        return True
    return False


# ------------------------- Caching & Rate limiting -------------------------
_LLM_CACHE = {}
_CACHE_LOCK = threading.Lock()

def _cache_key(incident_text: str, known_snips: str, ignore_snips: str, model: str) -> str:
    h = hashlib.sha256()
    for s in (incident_text, known_snips, ignore_snips, model):
        h.update((s or "").encode("utf-8"))
    return h.hexdigest()

class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = max(0.01, float(rate_per_sec))
        self.capacity = max(1, int(burst))
        self.tokens = float(self.capacity)
        self.last = time.time()
        self.lock = threading.Lock()
    def take(self) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False
    def wait(self):
        while not self.take():
            time.sleep(0.05)

_LLM_LIMITER = None
_MAX_LLM_CALLS = None
_calls_made = 0
_calls_lock = threading.Lock()

def llm_calls_remaining() -> bool:
    global _calls_made
    if _MAX_LLM_CALLS is None:
        return True
    with _calls_lock:
        return _calls_made < _MAX_LLM_CALLS

def incr_llm_calls():
    global _calls_made
    with _calls_lock:
        _calls_made += 1


# ------------------------- LLM arbitration (web retrieval built-in) -------------------------
def call_llm_arbiter(incident_text: str,
                     known_snips: str,
                     ignore_snips: str,
                     model: str,
                     reason_min_lines: int = 10,
                     timeout: float = 60.0,
                     max_retries: int = 5) -> Tuple[int, str, float, List[str]]:
    """
    Returns (label, reason, confidence, sources) where label in {1,0,-1}.
    Uses OpenAI search-enabled models (e.g., gpt-4o-mini-search, gpt-4.5-search).
    """
    # in-memory cache (per run)
    key = _cache_key(incident_text, known_snips, ignore_snips, model)
    with _CACHE_LOCK:
        if key in _LLM_CACHE:
            return _LLM_CACHE[key]

    # hard cap
    if not llm_calls_remaining():
        return -1, "arbiter_skipped:max_llm_calls_reached", 0.0, []

    # rate limit
    if _LLM_LIMITER is not None:
        _LLM_LIMITER.wait()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return -1, "arbiter_error:missing_env:OPENAI_API_KEY", 0.0, []

    client = OpenAI()  # uses OPENAI_API_KEY from env

    system_prompt = (
        "You are a senior Exadata/Exascale log triage arbiter.\n\n"
        "Task: Decide if the INCIDENT is:\n"
        "- 1 = reproduced/known bug,\n"
        "- 0 = ignorable/noise,\n"
        "- -1 = needs triage (novel or not enough evidence).\n\n"
        "Use KNOWN and IGNORE snippets as weak guidance. If needed, use your built-in web search to consult "
        "reliable documentation (Linux/Red Hat/kernel.org/Oracle/driver docs). Cite any URLs used.\n\n"
        "RESPONSE POLICY:\n"
        f'- Output STRICT JSON: {{"label": 1|0|-1, "confidence": 0.0..1.0, "reason": "...", "sources": ["..."]}}\n'
        f'- "reason": a clear explanation of AT LEAST {reason_min_lines} separate lines (each line one concise point).\n'
        '- "confidence": probability-like confidence in your label (0.0 to 1.0).\n'
        '- "sources": list of URLs you actually used (can be empty if none).\n'
        '- Do NOT include extra keys or any text outside JSON.'
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

Return STRICT JSON only: {{"label": 1|0|-1, "confidence": 0.0..1.0, "reason": "<>= {reason_min_lines} lines>", "sources": ["..."]}}
"""

    backoff = 0.8
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                timeout=timeout
            )
            content = resp.choices[0].message.content
            data = json.loads(content)

            # parse
            label = int(data.get("label", -1))
            if label not in (1, 0, -1):
                label = -1
            try:
                confidence = float(data.get("confidence", 0.0))
                confidence = max(0.0, min(1.0, confidence))
            except Exception:
                confidence = 0.0
            reason = str(data.get("reason", ""))
            lines = [ln for ln in reason.splitlines() if ln.strip()] if isinstance(reason, str) else []
            if len(lines) < reason_min_lines:
                reason = (reason if isinstance(reason, str) else "") + "\n\n[Note: reason had fewer than required lines.]"
            sources = data.get("sources", [])
            if not isinstance(sources, list):
                sources = []

            res = (label, reason, confidence, sources)
            with _CACHE_LOCK:
                _LLM_CACHE[key] = res
            incr_llm_calls()
            return res

        except Exception as e:
            if attempt >= max_retries:
                return -1, f"arbiter_error:{type(e).__name__}:{e}", 0.0, []
            # exponential backoff + small jitter
            sleep_s = (2.0 * (backoff ** attempt)) + (0.2 * (attempt + 1))
            time.sleep(sleep_s)


# ------------------------- Embedding decision -------------------------
def decide_with_embeddings(row: Dict[str, Any], kb_idx, ig_idx, model,
                           t_known=0.72, t_ignore=0.70):
    text = row['log_window']
    v = model.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    kbest = topk(v, kb_idx.get('embs'), k=3)
    ibest = topk(v, ig_idx.get('embs'), k=3)
    (k0_idx, k0_sim) = (kbest[0] if len(kbest) else (-1, 0.0))
    (i0_idx, i0_sim) = (ibest[0] if len(ibest) else (-1, 0.0))

    # fast paths
    if k0_sim >= max(t_known, i0_sim + 0.05):
        bug = kb_idx['df'].iloc[k0_idx].to_dict() if k0_idx >= 0 else {}
        return dict(label=1, confidence=float(k0_sim), reason="known-emb-match", evidence=[bug.get('bug_id','')])
    if i0_sim >= t_ignore:
        ign = ig_idx['df'].iloc[i0_idx].to_dict() if i0_idx >= 0 else {}
        return dict(label=0, confidence=float(i0_sim), reason="ignore-emb-match", evidence=[ign.get('pattern_id','')])

    return dict(label=-2, confidence=1 - max(k0_sim, i0_sim), reason="needs-arbitration",
                kbest=kbest, ibest=ibest)

def fmt_examples(df: pd.DataFrame, pairs: List[Tuple[int,float]], id_col: str, ctx_col: str) -> str:
    out = []
    if df is None or df.empty:
        return ""
    for i, sim in pairs[:2]:
        rec = df.iloc[i]
        out.append(f"[sim={sim:.3f}] {id_col}={rec.get(id_col, '')} ::\n{str(rec.get(ctx_col, ''))[:1500]}")
    return "\n\n".join(out)


# ------------------------- Main -------------------------
def main(incidents_path: str, known_idx_path: str, ignore_idx_path: str,
         outdir: str, model_name: str,
         use_llm: bool, llm_model: str,
         t_known: float = 0.72, t_ignore: float = 0.70,
         llm_qps: float = 0.5, llm_burst: int = 2, max_llm_calls: int = 200,
         reason_min_lines: int = 10,
         out_csv_name: str = "triaged_results_web.csv"):

    global _LLM_LIMITER, _MAX_LLM_CALLS
    _LLM_LIMITER = TokenBucket(rate_per_sec=llm_qps, burst=llm_burst) if use_llm else None
    _MAX_LLM_CALLS = max_llm_calls if use_llm else None

    kb_idx = load_pickle(known_idx_path)
    ig_idx = load_pickle(ignore_idx_path)
    enc = SentenceTransformer(model_name)

    df = pd.read_csv(incidents_path)
    os.makedirs(outdir, exist_ok=True)

    outs = {1: [], 0: [], -1: []}
    rows = []

    for _, r in df.iterrows():
        base = decide_with_embeddings(r, kb_idx, ig_idx, enc, t_known=t_known, t_ignore=t_ignore)
        label = base['label']

        if label == -2 and use_llm:
            text = r['log_window']
            kbest = base['kbest']; ibest = base['ibest']
            k0 = kbest[0][1] if kbest else 0.0
            i0 = ibest[0][1] if ibest else 0.0

            if should_arbitrate(k0, i0, text, t_known=t_known, t_ignore=t_ignore):
                known_snips = fmt_examples(kb_idx.get('df'), kbest, 'bug_id', 'context_window') if kbest else ""
                ignore_snips = fmt_examples(ig_idx.get('df'), ibest, 'pattern_id', 'context_window') if ibest else ""

                arb_label, arb_reason, arb_conf, arb_sources = call_llm_arbiter(
                    incident_text=text,
                    known_snips=known_snips,
                    ignore_snips=ignore_snips,
                    model=llm_model,
                    reason_min_lines=reason_min_lines
                )
                if arb_label in (1,0,-1):
                    conf = arb_conf if arb_conf > 0 else max(k0, i0)
                    base = dict(label=arb_label, confidence=float(conf),
                                reason=f"llm-arbiter:\n{arb_reason}", evidence=arb_sources or [])
                else:
                    base = dict(label=-1, confidence=float(1 - max(k0,i0)),
                                reason=arbiter_reason or "arbiter-unavailable", evidence=[])
            else:
                base = dict(label=-1, confidence=float(1 - max(k0, i0)),
                            reason="novel-or-ambiguous", evidence=[])

        elif label == -2:
            base = dict(label=-1, confidence=base['confidence'], reason="novel-or-ambiguous", evidence=[])

        rec = dict(
            incident_id=r.get('incident_id', ''),
            host=r.get('host',''),
            component=r.get('component',''),
            start_time=r.get('start_time',''),
            end_time=r.get('end_time',''),
            label=int(base['label']),
            confidence=round(float(base['confidence']), 4),
            reason=base['reason'],
            evidence=";".join(map(str, base.get('evidence', [])))
        )
        rows.append(rec)
        outs[int(base['label'])].append(rec)

    # Write outputs
    for lab, items in outs.items():
        with open(os.path.join(outdir, f"label_{lab}.jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")

    pd.DataFrame(rows).to_csv(os.path.join(outdir, out_csv_name), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", required=True)
    ap.add_argument("--known-index", required=True)
    ap.add_argument("--ignore-index", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="SentenceTransformer encode model for embeddings")
    # LLM controls
    ap.add_argument("--llm-model", default="gpt-4o-mini-search",
                    help="Search-enabled model (e.g., gpt-4o-mini-search, gpt-4.5-search)")
    ap.add_argument("--no-llm", dest="llm", action="store_false",
                    help="Disable LLM arbitration (embedding-only).")
    ap.set_defaults(llm=True)
    ap.add_argument("--t-known", type=float, default=0.72)
    ap.add_argument("--t-ignore", type=float, default=0.70)
    ap.add_argument("--llm-qps", type=float, default=0.5, help="LLM calls per second (avg).")
    ap.add_argument("--llm-burst", type=int, default=2, help="Burst size for token bucket.")
    ap.add_argument("--max-llm-calls", type=int, default=200, help="Max LLM calls per run.")
    ap.add_argument("--reason-min-lines", type=int, default=10, help="Minimum lines in LLM reasoning.")
    ap.add_argument("--out-csv-name", default="triaged_results_web.csv")
    args = ap.parse_args()

    # configure limiter
    if args.llm:
        _LLM_LIMITER = TokenBucket(rate_per_sec=args.llm_qps, burst=args.llm_burst)
        _MAX_LLM_CALLS = args.max_llm_calls

    main(args.incidents, args.known_index, args.ignore_index, args.outdir,
         args.model, args.llm, args.llm_model,
         t_known=args.t_known, t_ignore=args.t_ignore,
         llm_qps=args.llm_qps, llm_burst=args.llm_burst, max_llm_calls=args.max_llm_calls,
         reason_min_lines=args.reason_min_lines, out_csv_name=args.out_csv_name)