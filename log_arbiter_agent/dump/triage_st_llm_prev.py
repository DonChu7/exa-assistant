#!/usr/bin/env python3
"""
MiniLM triage with optional LLM arbitration.

Pipeline:
 1. Run MiniLM retrieval to label obvious known/ignore incidents.
 2. Anything still classed as "-1" can optionally be escalated to an LLM.
 3. LLM suggestions are captured in arbiter_review.csv for human approval.

Extras:
 - Host/timestamp normalization before embedding (matches triage_st.py behaviour).
 - Token-bucket rate limiting and a hard cap on LLM calls.
 - In-run caching so repeated incidents don't trigger duplicate API calls.
 - Structured outputs (per-label JSONL/CSV, triage_all.csv, arbiter_review.csv).
- Prompt customisation via arbiter_prefs.yaml (reason length, domain hints).
 - Prompt customisation via arbiter_prefs.yaml (reason length, domain hints, optional web search via Google CSE).
"""

import argparse
import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils import load_pickle, normalize_log_window

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def cosine_sim(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    return np.dot(mat, vec)


def topk(vec: np.ndarray, mat: Optional[np.ndarray], k: int = 3) -> List[Tuple[int, float]]:
    if mat is None or getattr(mat, "size", 0) == 0:
        return []
    sims = cosine_sim(vec, mat)
    idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]


# ---------------------------------------------------------------------------
# Arbitration trigger heuristics
# ---------------------------------------------------------------------------

def should_arbitrate(
    k0: float,
    i0: float,
    text: str,
    t_known: float = 0.72,
    t_ignore: float = 0.70,
    near: float = 0.05,
) -> bool:
    """Decide whether the LLM should be asked to weigh in."""
    if k0 < t_known and i0 < t_ignore:
        return True
    if abs(k0 - i0) < near:
        return True
    if len(text) > 1000:
        return True
    if len(re.findall(r"(failed|error|exception|panic)", text, flags=re.I)) >= 6:
        return True
    return False


# ---------------------------------------------------------------------------
# Rate limiting, call caps, and simple in-memory caching
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter (approximately qps, with burst)."""

    def __init__(self, qps: float, burst: int) -> None:
        self.qps = max(0.0, float(qps))
        self.capacity = max(1, int(burst)) if burst and burst > 0 else 1
        self.tokens = float(self.capacity)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        if self.qps <= 0.0:
            return
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last
                self.last = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.qps)
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                deficit = 1.0 - self.tokens
            sleep_for = max(deficit / self.qps, 0.0)
            time.sleep(min(sleep_for, 1.0))


class CallTracker:
    """Track how many LLM calls have been issued in this run."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls if max_calls and max_calls > 0 else None
        self.count = 0
        self.lock = threading.Lock()

    def allow(self) -> bool:
        if self.max_calls is None:
            return True
        with self.lock:
            return self.count < self.max_calls

    def incr(self) -> None:
        with self.lock:
            self.count += 1


_CACHE: Dict[str, Tuple[int, str, Optional[float]]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_key(
    incident_text: str,
    known_snips: str,
    ignore_snips: str,
    provider: str,
    model: str,
    doc_hint: str,
    reason_min_lines: int,
) -> str:
    h = hashlib.sha256()
    for part in (
        incident_text,
        known_snips,
        ignore_snips,
        provider,
        model,
        doc_hint,
        str(reason_min_lines),
    ):
        h.update((part or "").encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Preferences / prompt helpers
# ---------------------------------------------------------------------------

def load_arbiter_prefs(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                return {}
            return data
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def resolve_domain_hints(component: str, prefs: Dict[str, Any]) -> List[str]:
    comp = (component or "").lower()
    component_domains = prefs.get("component_domains", {}) or {}
    if comp in component_domains:
        domains = component_domains.get(comp, [])
    else:
        domains = prefs.get("allowed_domains", [])
    return [str(d) for d in domains if d]


def build_doc_hint(component: str, prefs: Dict[str, Any]) -> str:
    domains = resolve_domain_hints(component, prefs)
    if not domains:
        return ""
    lines = "\n".join(f"- {d}" for d in domains)
    return (
        "Suggested reference domains for background research (do not fabricate access):\n"
        f"{lines}"
    )


# ---------------------------------------------------------------------------
# LLM arbitration
# ---------------------------------------------------------------------------

def call_llm_arbiter(
    incident_text: str,
    known_snips: str,
    ignore_snips: str,
    provider: str,
    model: str,
    api_key_env: str,
    limiter: Optional[RateLimiter],
    tracker: Optional[CallTracker],
    doc_hint: str,
    reason_min_lines: int,
    timeout: float = 45.0,
    max_retries: int = 3,
) -> Tuple[int, str, Optional[float]]:
    """Returns (label, reason, confidence); label in {1,0,-1}. Never raises."""

    cache_key = _cache_key(
        incident_text,
        known_snips,
        ignore_snips,
        provider,
        model,
        doc_hint,
        reason_min_lines,
    )
    with _CACHE_LOCK:
        if cache_key in _CACHE:
            return _CACHE[cache_key]

    if tracker and not tracker.allow():
        return -1, "arbiter_skipped:max_llm_calls", None

    if limiter:
        limiter.acquire()

    api_key = os.getenv(api_key_env or "")
    if not api_key:
        return -1, f"arbiter_error:missing_env:{api_key_env}", None

    prov = (provider or "").strip().lower()
    if prov != "openai":
        return -1, f"arbiter_error:unsupported_provider:{prov}", None

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
    system_prompt = (
        "You are a log triage arbiter for Exadata/Exascale infrastructure. "
        "Decide if the INCIDENT is a reproduced/known bug (label 1), "
        "ignorable/noise (label 0), or needs triage (label -1). "
        "Use KNOWN and IGNORE snippets as weak guidance (they may be imperfect). "
        "If you incorporate external knowledge, ground it using the suggested domain list. "
        "Return STRICT JSON with keys: "
        "label (one of 1,0,-1), confidence (0.0-1.0) describing your certainty, and reason. "
        f"Reason must contain at least {max(reason_min_lines,1)} sentences, each on its own line, "
        "covering observed symptoms, root-cause hypothesis, supporting evidence, and next actions. "
        "Do not include markdown or extra text outside the JSON."
    )
    if doc_hint:
        user_prompt += f"\n\nREFERENCE HINTS:\n{doc_hint}\n"
    # no external web context injected when web scraping disabled

    import requests

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    backoff = 1.0
    made_request = False
    last_status: Optional[int] = None
    last_body: str = ""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        resp = None
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            made_request = True
            if resp.status_code == 429:
                last_status = resp.status_code
                try:
                    last_body = resp.text
                except Exception:
                    last_body = ""
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff * (attempt + 1)
                time.sleep(min(wait, 10.0))
                continue
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(message)
            label = int(data.get("label", -1))
            if label not in (1, 0, -1):
                label = -1
            reason = str(data.get("reason", "")).strip()
            confidence_val = data.get("confidence", None)
            try:
                confidence_float = float(confidence_val) if confidence_val is not None else None
            except (TypeError, ValueError):
                confidence_float = None
            with _CACHE_LOCK:
                _CACHE[cache_key] = (label, reason, confidence_float)
            if tracker:
                tracker.incr()
            return label, reason, confidence_float
        except Exception as exc:
            if attempt >= max_retries:
                if tracker and made_request:
                    tracker.incr()
                body = ""
                try:
                    if resp is not None:
                        body = resp.text
                except Exception:
                    pass
                last_exc = exc
                reason = f"arbiter_error:{type(exc).__name__}:{exc}"
                if body:
                    reason += f":{body[:160]}"
                return -1, reason, None
            sleep_for = min(backoff * (attempt + 1), 10.0)
            time.sleep(sleep_for)
            last_exc = exc

    detail = "arbiter_error:unknown"
    if last_status is not None:
        detail = f"arbiter_error:http_{last_status}"
        if last_body:
            detail += f":{last_body[:160]}"
    elif last_exc is not None:
        detail = f"arbiter_error:{type(last_exc).__name__}:{last_exc}"
    return -1, detail, None


# ---------------------------------------------------------------------------
# Embedding decision
# ---------------------------------------------------------------------------

def decide_with_embeddings(
    row: Dict[str, Any],
    kb_idx: Dict[str, Any],
    ig_idx: Dict[str, Any],
    model: SentenceTransformer,
    t_known: float = 0.72,
    t_ignore: float = 0.70,
) -> Dict[str, Any]:
    raw_text = row.get("log_window", "") or ""
    norm_text = normalize_log_window(str(raw_text))
    vec = model.encode([norm_text], normalize_embeddings=True)[0].astype(np.float32)

    kbest = topk(vec, kb_idx.get("embs"))
    ibest = topk(vec, ig_idx.get("embs"))
    k0_idx, k0_sim = (kbest[0] if kbest else (-1, 0.0))
    i0_idx, i0_sim = (ibest[0] if ibest else (-1, 0.0))

    if k0_sim >= max(t_known, i0_sim + 0.05):
        bug = kb_idx["df"].iloc[k0_idx].to_dict() if k0_idx >= 0 else {}
        return dict(
            label=1,
            confidence=float(k0_sim),
            reason="known-emb-match",
            evidence=[bug.get("bug_id", "")],
            raw_text=raw_text,
            kbest=kbest,
            ibest=ibest,
            k0_sim=float(k0_sim),
            i0_sim=float(i0_sim),
        )

    if i0_sim >= t_ignore:
        ign = ig_idx["df"].iloc[i0_idx].to_dict() if i0_idx >= 0 else {}
        return dict(
            label=0,
            confidence=float(i0_sim),
            reason="ignore-emb-match",
            evidence=[ign.get("pattern_id", "")],
            raw_text=raw_text,
            kbest=kbest,
            ibest=ibest,
            k0_sim=float(k0_sim),
            i0_sim=float(i0_sim),
        )

    return dict(
        label=-2,
        confidence=float(1.0 - max(k0_sim, i0_sim)),
        reason="needs-arbitration",
        evidence=[],
        raw_text=raw_text,
        kbest=kbest,
        ibest=ibest,
        k0_sim=float(k0_sim),
        i0_sim=float(i0_sim),
    )


def fmt_examples(
    df: Optional[pd.DataFrame],
    pairs: List[Tuple[int, float]],
    id_col: str,
    ctx_col: str,
) -> str:
    if df is None or df.empty:
        return ""
    snippets: List[str] = []
    for idx, score in pairs[:2]:
        rec = df.iloc[idx]
        snippets.append(
            f"[sim={score:.3f}] {id_col}={rec.get(id_col, '')} ::\n{str(rec.get(ctx_col, ''))[:1500]}"
        )
    return "\n\n".join(snippets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    incidents_path: str,
    known_idx_path: str,
    ignore_idx_path: str,
    outdir: str,
    model_name: str,
    use_llm: bool,
    llm_provider: str,
    llm_model: str,
    llm_api_key_env: str,
    t_known: float = 0.72,
    t_ignore: float = 0.70,
    llm_qps: float = 0.0,
    llm_burst: int = 1,
    max_llm_calls: int = 0,
    arbiter_prefs_path: Optional[str] = "arbiter_prefs.yaml",
) -> None:

    kb_idx = load_pickle(known_idx_path)
    ig_idx = load_pickle(ignore_idx_path)

    if "embs" not in kb_idx or "embs" not in ig_idx:
        raise RuntimeError("MiniLM indexes (with 'embs') required. Run build_memories_st.py first.")

    model = SentenceTransformer(model_name)

    df = pd.read_csv(incidents_path)
    os.makedirs(outdir, exist_ok=True)

    limiter = RateLimiter(llm_qps, llm_burst) if use_llm and llm_qps > 0.0 else None
    tracker = CallTracker(max_llm_calls) if use_llm else None
    prefs = load_arbiter_prefs(arbiter_prefs_path if use_llm else None)
    if use_llm:
        raw_min_lines = prefs.get("reason_min_lines", 10)
        try:
            reason_min_lines = int(raw_min_lines)
        except (TypeError, ValueError):
            reason_min_lines = 10
        reason_min_lines = max(reason_min_lines, 1)
    else:
        reason_min_lines = 10

    outs: Dict[int, List[Dict[str, Any]]] = {1: [], 0: [], -1: []}
    triage_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        base = decide_with_embeddings(row, kb_idx, ig_idx, model, t_known=t_known, t_ignore=t_ignore)
        baseline_label = int(base.get("label", -1))
        raw_text = base.get("raw_text", row.get("log_window", "") or "")
        evidence_list = base.get("evidence", []) or []
        kbest = base.get("kbest", []) or []
        ibest = base.get("ibest", []) or []
        k0 = float(base.get("k0_sim", 0.0))
        i0 = float(base.get("i0_sim", 0.0))
        baseline_conf = float(base.get("confidence", 0.0))
        baseline_reason = base.get("reason", "")
        component_value_raw = row.get("component", "")
        if pd.isna(component_value_raw):
            component_value = ""
        else:
            component_value = str(component_value_raw)
        doc_hint = build_doc_hint(component_value, prefs) if use_llm else ""

        if baseline_label == -2:
            baseline_label = -1
            baseline_reason = "novel-or-ambiguous"
            evidence_list = []

        top_bug_id = ""
        if kbest:
            try:
                top_bug_id = str(kb_idx["df"].iloc[kbest[0][0]].get("bug_id", ""))
            except Exception:
                top_bug_id = ""
        top_pattern_id = ""
        if ibest:
            try:
                top_pattern_id = str(ig_idx["df"].iloc[ibest[0][0]].get("pattern_id", ""))
            except Exception:
                top_pattern_id = ""

        known_snips = fmt_examples(kb_idx.get("df"), kbest, "bug_id", "context_window") if kbest else ""
        ignore_snips = fmt_examples(ig_idx.get("df"), ibest, "pattern_id", "context_window") if ibest else ""

        llm_status = ""
        llm_label = ""
        llm_reason = ""
        llm_confidence: Optional[float] = None

        if baseline_label == -1:
            if use_llm:
                if tracker and not tracker.allow():
                    llm_status = "limit_exhausted"
                    llm_reason = "arbiter_skipped:max_llm_calls"
                elif not should_arbitrate(k0, i0, raw_text, t_known=t_known, t_ignore=t_ignore):
                    llm_status = "skipped"
                    llm_reason = "skipped_by_threshold"
                else:
                    label, reason, confidence_val = call_llm_arbiter(
                        incident_text=raw_text,
                        known_snips=known_snips,
                        ignore_snips=ignore_snips,
                        provider=llm_provider,
                        model=llm_model,
                        api_key_env=llm_api_key_env,
                        limiter=limiter,
                        tracker=tracker,
                        doc_hint=doc_hint,
                        reason_min_lines=reason_min_lines,
                    )
                    if isinstance(reason, str) and reason.startswith("arbiter_error"):
                        llm_status = "error"
                        llm_reason = reason
                    elif isinstance(reason, str) and reason.startswith("arbiter_skipped"):
                        llm_status = "limit_exhausted"
                        llm_reason = reason
                    else:
                        llm_status = "decided" if label in (0, 1) else "novel"
                        llm_label = str(label)
                        llm_reason = reason
                        llm_confidence = confidence_val
            else:
                llm_status = "disabled"
                llm_reason = "llm_disabled"

            review_rows.append(
                dict(
                    incident_id=row.get("incident_id", ""),
                    host=row.get("host", ""),
                    component=row.get("component", ""),
                    log_window=raw_text,
                    baseline_confidence=round(baseline_conf, 4),
                    max_known_sim=round(k0, 4),
                    max_ignore_sim=round(i0, 4),
                    baseline_reason=baseline_reason,
                    baseline_evidence=";".join(str(ev) for ev in evidence_list),
                    top_bug_id=top_bug_id,
                    top_pattern_id=top_pattern_id,
                    known_examples=known_snips,
                    ignore_examples=ignore_snips,
                    llm_status=llm_status,
                    llm_label=llm_label,
                    llm_reason=llm_reason,
                    llm_confidence=llm_confidence,
                    doc_hint=doc_hint,
                )
            )

        triage_record = dict(
            incident_id=row.get("incident_id", ""),
            host=row.get("host", ""),
            component=row.get("component", ""),
            log_window=raw_text,
            label=baseline_label,
            confidence=round(baseline_conf, 4),
            reason=baseline_reason,
            evidence=";".join(str(ev) for ev in evidence_list),
            llm_status=llm_status,
            llm_label=llm_label,
            llm_reason=llm_reason,
            llm_confidence=llm_confidence,
        )
        triage_rows.append(triage_record)
        outs[baseline_label].append(triage_record)

    # Per-label outputs (JSONL + CSV)
    for lab, items in outs.items():
        label_path = os.path.join(outdir, f"label_{lab}")
        with open(label_path + ".jsonl", "w") as fh:
            for rec in items:
                fh.write(json.dumps(rec) + "\n")
        pd.DataFrame(items).to_csv(label_path + ".csv", index=False)

    pd.DataFrame(triage_rows).to_csv(os.path.join(outdir, "triage_all.csv"), index=False)

    review_columns = [
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
        "llm_reason",
        "llm_confidence",
        "doc_hint",
    ]
    pd.DataFrame(review_rows, columns=review_columns).to_csv(
        os.path.join(outdir, "arbiter_review.csv"), index=False
    )

    if tracker:
        llm_calls_used = tracker.count
    else:
        llm_calls_used = len(
            [rec for rec in triage_rows if rec["llm_status"] in ("decided", "novel", "error")]
        )
    print(
        f"Done. Wrote {sum(len(v) for v in outs.values())} decisions into {outdir} "
        f"(review queue: {len(review_rows)}, llm_calls={llm_calls_used})"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", required=True)
    ap.add_argument("--known-index", required=True)
    ap.add_argument("--ignore-index", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--t-known", type=float, default=0.72)
    ap.add_argument("--t-ignore", type=float, default=0.70)
    ap.add_argument("--llm", action="store_true", help="Enable LLM arbitration for baseline -1 incidents")
    ap.add_argument("--llm-provider", default="openai")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--llm-qps", type=float, default=0.0, help="Approximate LLM calls per second (0 = unlimited)")
    ap.add_argument("--llm-burst", type=int, default=1, help="Token bucket burst size for rate limiting")
    ap.add_argument("--max-llm-calls", type=int, default=0, help="Hard cap on LLM calls (0 = unlimited)")
    ap.add_argument("--arbiter-prefs", default="arbiter_prefs.yaml", help="Path to arbiter preferences YAML.")
    args = ap.parse_args()

    main(
        args.incidents,
        args.known_index,
        args.ignore_index,
        args.outdir,
        args.model,
        args.llm,
        args.llm_provider,
        args.llm_model,
        args.llm_api_key_env,
        t_known=args.t_known,
        t_ignore=args.t_ignore,
        llm_qps=args.llm_qps,
        llm_burst=args.llm_burst,
        max_llm_calls=args.max_llm_calls,
        arbiter_prefs_path=args.arbiter_prefs,
    )
