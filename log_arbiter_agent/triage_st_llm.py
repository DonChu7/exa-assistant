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

import argparse
import csv
import os
import sys
import json
import re
import time
import hashlib
import threading
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

# Embeddings (MiniLM) for similarity
from sentence_transformers import SentenceTransformer

# OpenAI SDK v2
# from openai import OpenAI, NotFoundError, BadRequestError

# ensure TaLLLMAccess is available
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ta_llm_access import TaLLLMAccess

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


def _cache_key(incident_text: str, known_snips: str, ignore_snips: str, model: str, dev_feedback: str = "") -> str:
    h = hashlib.sha256()
    for s in (incident_text, known_snips, ignore_snips, model, dev_feedback):
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

def call_llm_arbiter(
    incident_text: str,
    known_snips: str,
    ignore_snips: str,
    model: str,
    dev_feedback: str = "",
    reason_min_lines: int = 10,
    timeout: float = 60.0,
    max_retries: int = 5,
) -> Tuple[int, str, float, List[str], List[str]]:
    """
    Returns (label, reason, confidence, sources, additional_sources) where label in {1,0,-1}.
    Grok4-compatible version that keeps the original triage_st_llm_web.py structure.
    """
    cache_key = _cache_key(incident_text, known_snips, ignore_snips, model, dev_feedback)
    with _CACHE_LOCK:
        if cache_key in _LLM_CACHE:
            return _LLM_CACHE[cache_key]

    if not llm_calls_remaining():
        return -1, "arbiter_skipped:max_llm_calls_reached", 0.0, [], []

    if _LLM_LIMITER is not None:
        _LLM_LIMITER.wait()

    # if not os.getenv("OPENAI_API_KEY"):
    #     return -1, "arbiter_error:missing_env:OPENAI_API_KEY", 0.0, [], []
    access = TaLLLMAccess()

    # client = OpenAI()
    # supports_web = any(tag in model for tag in ("search", "search-preview"))

    system_prompt = (
        "You are an expert Oracle Exadata/Exascale diagnostic arbiter specializing in analyzing /var/log/messages, "
        "systemd, kernel, and service-level logs from complex distributed systems.\n\n"

        "You are allowed to use real-time web knowledge and external documentation."
        "When you cite external information, include full URLs in 'sources'.\n"

        "### DEVELOPER FEEDBACK PRIORITY ###\n"
        "If developer feedback or prior investigation notes are provided, treat them as the primary signal. "
        "Interpret logs using that context first and only override it when strong, evidence-based reasoning justifies it. "
        "Never ignore explicit developer input.\n\n"

        "### OBJECTIVE ###\n"
        "Determine whether the provided INCIDENT window indicates a *critical issue* affecting Exadata/Exascale reliability, "
        "or if it’s an ignorable/non-impacting message. You must interpret the operational context — not just keywords — "
        "and reason about what sequence of events or failures could cause the log pattern.\n\n"

        "### CLASSIFICATION RULES ###\n"
        "Assign exactly one label:\n"
        "- 1 → **Known or reproducible bug / Critical fault** (component/service failure, reproducible or reported bug)\n"
        "- 0 → **Ignorable / benign noise** (non-impacting log, normal service shutdown, transient error, missing optional unit, etc.)\n"
        "- -1 → **Needs triage / ambiguous** (new, unclear, or potentially critical but not confirmed)\n\n"

        "### EXPECTED REASONING STYLE ###\n"
        "Write the 'reason' as a single, cohesive, technically precise diagnostic summary (minimum {reason_min_lines} sentences). "
        "It must begin with **'Severity Assessment:'**, clearly stating whether the issue is Critical, Non-Critical, or Unclear, "
        "followed by an integrated explanation covering:\n"
        "- The component or subsystem context,\n"
        "- The likely technical root cause or misconfiguration,\n"
        "- The potential impact on Exadata/Exascale reliability or availability,\n"
        "- How the behavior could be reproduced,\n"
        "- Recommended resolution or mitigation steps,\n"
        "- Supporting evidence or known precedents (e.g., kernel, systemd, RDMA, or Oracle contexts).\n"
        "Avoid numbered lists or bullets — the reasoning must read like a professional internal Oracle diagnostic summary.\n\n"

        "### SPECIAL INSTRUCTION — EXADATA COMPONENTS ###\n"
        "If the log pattern or error clearly involves an **Exadata-specific component or subsystem** "
        "(for example: cellrsrv, cellsrv, ms, ohasd, ibsrvr, rdma stack, kdump, exadisk, or similar), "
        "you must append at the end of your 'reason' this explicit statement:\n"
        "\"This incident involves an Exadata/Exascale-specific component and should be reviewed by development for confirmation or deeper triage.\"\n\n"

        "### WEB SEARCH GUIDANCE ###\n"
        "When using built-in web search:\n"
        "- Prefer authoritative sources such as Oracle Docs, Red Hat, Linux kernel mailing lists, and trusted forums (bbs.archlinux.org, serverfault.com, etc.).\n"
        "- Avoid casual blogs unless no vendor documentation is available.\n"
        "- Always provide full citations (URLs) in 'sources'.\n"
        "- Populate 'additional_sources' with up to three extra authoritative references describing similar patterns or mitigations "
        "(format: URL - short note).\n"
        "- Summarize key findings from multiple sources concisely before your conclusion.\n\n"

        "### OUTPUT FORMAT (STRICT JSON ONLY) ###\n"
        "Respond with a single JSON object in this exact structure:\n"
        "{\n"
        '  "label": 1 | 0 | -1,\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "reason": "<multi-sentence summary beginning with Severity Assessment:, ending with Dev feedback note if applicable>",\n'
        '  "sources": ["<URLs of docs or articles you referenced>"],\n'
        '  "additional_sources": ["<URL - Additional authoritative references describing similar log pattern>"]\n'
        "}\n\n"

        "### IMPORTANT CONSTRAINTS ###\n"
        "- Do NOT output anything outside of the JSON object.\n"
        "- Do NOT summarize or explain outside of JSON.\n"
        "- 'confidence' represents your probability that the label is correct, based on reasoning completeness and source reliability.\n"
        "- When uncertain, prefer '-1' with confidence < 0.6.\n"
        "- Maintain technical precision and concise engineering tone.\n\n"

        "Example of high-quality reasoning (shortened):\n"
        "Severity Assessment: Critical – The logs show ohasd.service repeatedly failing to start due to missing init.ohasd script, "
        "causing clusterware startup failure. This matches Oracle Bug 37735911. The failure prevents GI initialization and can be "
        "reproduced by removing /etc/init.d/init.ohasd. Resolution: reinstall GI or restore the missing service file. "
        "This incident involves an Exadata/Exascale-specific component and should be reviewed by development for confirmation or deeper triage."
    )

    user_prompt = f"""
INCIDENT:
```
{incident_text}
```

DEVELOPER FEEDBACK:
```
{dev_feedback or "None provided"}
```

Return JSON only: {{"label": 1|0|-1, "confidence": 0.0..1.0, "reason": "<>= {reason_min_lines} lines>", "sources": ["..."]}}
"""

    full_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    backoff = 0.8
    access = TaLLLMAccess()
    for attempt in range(max_retries + 1):
        try:
            # --- single API entry point ---
            full_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
            resp_text = access.ta_request(full_prompt, model)

            if not resp_text or not isinstance(resp_text, str):
                return -1, "arbiter_error:empty_response", 0.0, [], []

            content = resp_text.strip()
            if content.startswith("```"):
                content = re.sub(r"^```[a-zA-Z0-9]*\\s*", "", content)
                content = re.sub(r"\\s*```$", "", content)

            attempts = [content]
            brace_match = re.search(r"\\{.*\\}", content, re.S)
            if brace_match:
                attempts.append(brace_match.group(0))

            data = None
            for candidate in attempts:
                candidate = candidate.strip()
                if not candidate:
                    continue
                try:
                    data = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue
            if data is None:
                return -1, "arbiter_error:invalid_json", 0.0, [], []

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
                reason = (reason if isinstance(reason, str) else "") + "\\n\\n[Note: reason had fewer than required lines.]"
            sources = data.get("sources", [])
            if not isinstance(sources, list):
                sources = []
            additional_sources = data.get("additional_sources", [])
            if not isinstance(additional_sources, list):
                additional_sources = [additional_sources] if additional_sources else []
            additional_sources = additional_sources[:3]

            result = (label, reason, confidence, sources, additional_sources)
            with _CACHE_LOCK:
                _LLM_CACHE[cache_key] = result
            incr_llm_calls()
            return result

        except Exception as exc:
            if attempt >= max_retries:
                return -1, f"arbiter_error:{type(exc).__name__}:{exc}", 0.0, [], []
            sleep_seconds = (2.0 * (backoff ** attempt)) + (0.2 * (attempt + 1))
            time.sleep(sleep_seconds)


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
        return dict(label=1, confidence=float(k0_sim), reason="known-emb-match",
                    evidence=[bug.get('bug_id','')], additional_sources=[],
                    kbest=kbest, ibest=ibest, k0_sim=float(k0_sim), i0_sim=float(i0_sim))
    if i0_sim >= t_ignore:
        ign = ig_idx['df'].iloc[i0_idx].to_dict() if i0_idx >= 0 else {}
        return dict(label=0, confidence=float(i0_sim), reason="ignore-emb-match",
                    evidence=[ign.get('pattern_id','')], additional_sources=[],
                    kbest=kbest, ibest=ibest, k0_sim=float(k0_sim), i0_sim=float(i0_sim))

    return dict(label=-2, confidence=1 - max(k0_sim, i0_sim), reason="needs-arbitration",
                kbest=kbest, ibest=ibest, k0_sim=float(k0_sim), i0_sim=float(i0_sim),
                additional_sources=[])

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
    if "incident_id" not in df.columns:
        raise ValueError("Input incidents file must contain 'incident_id' column.")
    if "log_window" not in df.columns:
        if "messages" in df.columns:
            df["log_window"] = df["messages"]
        else:
            raise ValueError("Input incidents file must contain 'log_window' or 'messages' column.")
    if "host" not in df.columns:
        if "hostname" in df.columns:
            df["host"] = df["hostname"]
        else:
            df["host"] = ""
    if "dev_feedback" not in df.columns:
        df["dev_feedback"] = ""
    base_cols = {"incident_id", "host", "component", "log_window", "dev_feedback"}
    extra_cols = [c for c in df.columns if c not in base_cols]
    target_dir = os.path.join(outdir, "outputs_1")
    os.makedirs(target_dir, exist_ok=True)

    label_files = {
        1: open(os.path.join(target_dir, "label_1.jsonl"), "w", encoding="utf-8"),
        0: open(os.path.join(target_dir, "label_0.jsonl"), "w", encoding="utf-8"),
        -1: open(os.path.join(target_dir, "label_-1.jsonl"), "w", encoding="utf-8"),
    }

    csv_path = os.path.join(target_dir, out_csv_name)
    fieldnames = [
        "incident_id",
        "host",
        "component",
        "label",
        "confidence",
        "reason",
        "evidence",
        "additional_sources",
        "log_window",
        "dev_feedback",
    ] + extra_cols

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _, r in df.iterrows():
            llm_used = False
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

                    arb_label, arb_reason, arb_conf, arb_sources, arb_additional = call_llm_arbiter(
                        incident_text=text,
                        known_snips=known_snips,
                        ignore_snips=ignore_snips,
                        model=llm_model,
                        dev_feedback=str(r.get('dev_feedback', '')),
                        reason_min_lines=reason_min_lines
                    )
                    llm_used = True
                    if arb_label in (1,0,-1):
                        conf = arb_conf if arb_conf > 0 else max(k0, i0)
                        base = dict(label=arb_label, confidence=float(conf),
                                    reason=f"llm-arbiter:\n{arb_reason}", evidence=arb_sources or [],
                                    additional_sources=arb_additional or [])
                    else:
                        base = dict(label=-1, confidence=float(1 - max(k0,i0)),
                                    reason=arb_reason if 'arbiter_error' in arb_reason else "arbiter-unavailable",
                                    evidence=[], additional_sources=[])
                else:
                    base = dict(label=-1, confidence=float(1 - max(k0, i0)),
                                reason="novel-or-ambiguous", evidence=[], additional_sources=[])

            elif label == -2:
                base = dict(label=-1, confidence=base['confidence'], reason="novel-or-ambiguous",
                            evidence=[], additional_sources=[])

            rec = dict(
                incident_id=r.get('incident_id', ''),
                host=r.get('host',''),
                component=r.get('component',''),
                label=int(base['label']),
                confidence=round(float(base['confidence']), 4),
                reason=base['reason'],
                evidence=";".join(map(str, base.get('evidence', []))),
                additional_sources=";".join(map(str, base.get('additional_sources', []))),
                log_window=r.get('log_window', ''),
                dev_feedback=str(r.get('dev_feedback', ''))
            )
            for col in extra_cols:
                val = r.get(col, "")
                if isinstance(val, float) and np.isnan(val):
                    val = ""
                rec[col] = val

            if llm_used:
                writer.writerow(rec)

            final_label = int(base['label'])
            if (not llm_used) or final_label == -1:
                label_files[final_label].write(json.dumps(rec) + "\n")

    for fh in label_files.values():
        fh.close()


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
