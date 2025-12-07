#!/usr/bin/env python3
"""
triage_grok_llm.py
------------------
Arbitrate ambiguous (-1) incidents using Oracle TA LLM access with Grok-3 / Grok-4
workflows (via ta_llm_access.TaLLLMAccess). This script expects the ambiguous
CSV produced by triage_embeddings_report.py and asks the LLM to classify each
incident window into:
  • 1 → Known or reproducible bug / Critical fault
  • 0 → Ignorable / benign noise
  • -1 → Needs triage / ambiguous

Outputs:
  • CSV file with all LLM decisions: <outdir>/grok_outputs/triaged_results_grok.csv
  • Per-label JSONL files:           <outdir>/grok_outputs/label_{1|0|-1}.jsonl
  • Diff report (with --memory):     <outdir>/grok_outputs/diff_report_<label>.txt

Usage (basic):
  python triage_grok_llm.py \
    --ambiguous-csv tables/minilm_ambiguous.csv \
    --outdir tables \
    --ta-model grok3

Usage (with memory for differential analysis):
  python triage_grok_llm.py \
    --ambiguous-csv tables/minilm_ambiguous.csv \
    --outdir tables \
    --ta-model grok3 \
    --memory \
    --current-label OSS_MAIN_LINUX.X64_250108

Memory Features (--memory):
  • Tracks incidents across the last 5 labels automatically
  • Provides historical context to LLM (previous classifications, confidence)
  • Identifies NEW vs RECURRING incidents
  • Detects classification changes between labels
  • Generates differential analysis report showing trends
  • Output CSV includes: incident_status, first_seen_label, last_seen_label,
    occurrence_count, previous_classifications

Notes:
  • Uses Oracle TA server workflows defined in ta_llm_access.py:
      - grok3 → workflow_id: 1814
      - grok4 → workflow_id: 986
  • The TA workflow may have built-in real-time web search; the prompt requests
    citations and a JSON response. We attempt to parse JSON and fall back
    gracefully if parsing fails.
  • Memory is stored persistently in <outdir>/memory/label_memory.pkl
  • Historical context helps LLM make more informed decisions based on past patterns
"""

import argparse
import os
import sys
import csv
import hashlib
import json
import os
import pickle
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Ensure we can import the TA access client from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ta_llm_access import TaLLLMAccess


# ------------------------- Rate limiting & caching -------------------------
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

_LLM_CACHE: Dict[str, Tuple[int, str, float, List[str], List[str]]] = {}
_CACHE_LOCK = threading.Lock()

_calls_made = 0
_calls_lock = threading.Lock()
_MAX_CALLS: int | None = None
_LIMITER: TokenBucket | None = None


# ------------------------- Memory Management -------------------------
class LabelMemory:
    """Manages historical incident data across labels for differential analysis."""
    
    def __init__(self, memory_dir: str, max_labels: int = 5):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.max_labels = max_labels
        self.memory_file = self.memory_dir / "label_memory.pkl"
        
        # Structure: {pattern_key: deque([{label, timestamp, classification, reason, ...}], maxlen=max_labels)}
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_labels))
        self.label_order: deque = deque(maxlen=max_labels)  # Track chronological order of labels
        
        self._load()
    
    def _pattern_key(self, component: str, message_structure: str) -> str:
        """Create unique key for incident pattern."""
        return f"{component}||{message_structure}"
    
    def _load(self):
        """Load historical memory from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.history = data.get('history', defaultdict(lambda: deque(maxlen=self.max_labels)))
                    self.label_order = data.get('label_order', deque(maxlen=self.max_labels))
            except Exception as e:
                print(f"Warning: Could not load memory file: {e}")
    
    def _save(self):
        """Persist memory to disk."""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump({
                    'history': dict(self.history),
                    'label_order': self.label_order
                }, f)
        except Exception as e:
            print(f"Warning: Could not save memory file: {e}")
    
    def get_history(self, component: str, message_structure: str) -> List[Dict[str, Any]]:
        """Retrieve historical occurrences of this incident pattern."""
        key = self._pattern_key(component, message_structure)
        return list(self.history.get(key, []))
    
    def add_incident(self, label: str, component: str, message_structure: str, 
                     classification: int, reason: str, confidence: float, **extra):
        """Record a new incident classification."""
        key = self._pattern_key(component, message_structure)
        self.history[key].append({
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'classification': classification,
            'reason': reason,
            'confidence': confidence,
            **extra
        })
    
    def finalize_label(self, label: str):
        """Mark label as complete and save to disk."""
        if label not in self.label_order:
            self.label_order.append(label)
        self._save()
    
    def get_incident_status(self, component: str, message_structure: str, current_label: str) -> Dict[str, Any]:
        """Determine if incident is NEW, RECURRING, or get historical context."""
        history = self.get_history(component, message_structure)
        
        if not history:
            return {
                'status': 'NEW',
                'first_seen_label': current_label,
                'last_seen_label': current_label,
                'occurrence_count': 0,
                'previous_labels': [],
                'classification_history': []
            }
        
        # Filter out current label (if reprocessing)
        past_occurrences = [h for h in history if h['label'] != current_label]
        
        if not past_occurrences:
            return {
                'status': 'NEW',
                'first_seen_label': current_label,
                'last_seen_label': current_label,
                'occurrence_count': 0,
                'previous_labels': [],
                'classification_history': []
            }
        
        return {
            'status': 'RECURRING',
            'first_seen_label': past_occurrences[0]['label'],
            'last_seen_label': past_occurrences[-1]['label'],
            'occurrence_count': len(past_occurrences),
            'previous_labels': [h['label'] for h in past_occurrences],
            'classification_history': [
                f"{h['label']}:L{h['classification']}(conf={h['confidence']:.2f})" 
                for h in past_occurrences
            ]
        }
    
    def get_all_patterns_from_previous_labels(self) -> Dict[str, Dict]:
        """Get all patterns seen in previous labels (for RESOLVED detection)."""
        patterns = {}
        for key, hist in self.history.items():
            if hist:
                last = hist[-1]
                patterns[key] = last
        return patterns
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        total_patterns = len(self.history)
        labels_tracked = len(self.label_order)
        oldest_label = self.label_order[0] if self.label_order else None
        newest_label = self.label_order[-1] if self.label_order else None
        
        # Count incidents per label
        label_counts = defaultdict(int)
        for hist_entries in self.history.values():
            for entry in hist_entries:
                label_counts[entry['label']] += 1
        
        return {
            'total_patterns': total_patterns,
            'labels_tracked': labels_tracked,
            'oldest_label': oldest_label,
            'newest_label': newest_label,
            'label_counts': dict(label_counts)
        }


# ------------------------- Cache key & helpers -------------------------
def _cache_key(record: Dict[str, Any], ta_model: str, reason_min_lines: int) -> str:
    h = hashlib.sha256()
    parts = [
        str(record.get("log_window", "")),
        str(record.get("dev_feedback", "")),
        str(record.get("message_structure", "")),
        ta_model or "",
        str(reason_min_lines),
    ]
    for s in parts:
        h.update((s or "").encode("utf-8"))
    return h.hexdigest()


def remaining_calls() -> bool:
    global _calls_made
    if _MAX_CALLS is None:
        return True
    with _calls_lock:
        return _calls_made < _MAX_CALLS


def _incr_calls():
    global _calls_made
    with _calls_lock:
        _calls_made += 1


# ------------------------- Prompting -------------------------
SYSTEM_PROMPT = (
    "You are an expert Oracle Exadata/Exascale diagnostic arbiter specializing in analyzing /var/log/messages, "
    "systemd, kernel, storage, RDMA and service-level logs in complex distributed systems.\n\n"

    "INPUT FORMAT: You will receive each incident as a JSON object with these fields:\n"
    "  - incident_id: unique identifier for the incident\n"
    "  - component: subsystem or service involved (e.g., systemd, kernel, cellsrv, ms, ohasd)\n"
    "  - message_structure: normalized log pattern with placeholders (e.g., <ip>, <num>, <hex>)\n"
    "  - log_window: full raw log text for the incident window (multi-line messages)\n"
    "  - dev_feedback: developer notes or prior investigation results (may be empty)\n\n"

    "PRIORITIZE DEVELOPER FEEDBACK: If developer feedback is provided, treat it as the primary signal. "
    "Interpret logs using that context first and override it only with strong, evidence-based reasoning.\n\n"

    "OBJECTIVE: Determine whether the INCIDENT indicates a critical issue (component/service failure, known or reproducible bug), "
    "an ignorable/non-impacting message, or a case needing triage. Consider operational context and failure sequences, not just keywords. "
    "Use the message_structure to identify patterns and the log_window for specific context.\n\n"

    "CLASSIFICATION: Return exactly one label: 1 (critical/known), 0 (ignorable), or -1 (needs triage).\n\n"

    "REASONING STYLE: Provide a concise, technically precise internal diagnostic summary (>= {reason_min_lines} sentences). "
    "The first line MUST be: \"Severity Assessment: Critical\" OR \"Severity Assessment: Non-Critical\" OR \"Severity Assessment: Unclear\". "
    "Then, starting from the next line, provide an integrated explanation covering:\n"
    "- The component or subsystem context,\n"
    "- The likely technical root cause or misconfiguration,\n"
    "- The potential impact on Exadata/Exascale reliability or availability,\n"
    "- How the behavior could be reproduced,\n"
    "- Recommended resolution or mitigation steps,\n"
    "- Supporting evidence or known precedents (e.g., kernel, systemd, RDMA, or Oracle contexts).\n"
    "Avoid numbered lists or bullets in the body — the reasoning must read like a professional internal Oracle diagnostic summary.\n\n"

    "EXADATA COMPONENT NOTE: If an Exadata/Exascale-specific component (e.g., cellsrv, cellrsrv, ms, ohasd, ibsrvr, rdma stack, kdump, exadisk) is involved, "
    "append the sentence: \"This incident involves an Exadata/Exascale-specific component and should be reviewed by development for confirmation or deeper triage.\"\n\n"

    "WEB SEARCH GUIDANCE: When searching for information, prioritize authoritative sources based on component type:\n\n"
    "  1. EXADATA/EXASCALE COMPONENTS (exachkcfg, celld, cellsrv, cellrsrv, ms, ohasd, ibsrvr, exaportmon, exadata-initvf, rdmaip-link-monitor):\n"
    "     PRIMARY: https://support.oracle.com/ (My Oracle Support - search 'Exadata [component] [error]')\n"
    "     SECONDARY: https://docs.oracle.com/en/engineered-systems/exadata-database-machine/\n"
    "     TERTIARY: https://docs.oracle.com/en/database/oracle/oracle-database/ (Grid Infrastructure/RAC)\n"
    "     BLOGS: https://blogs.oracle.com/exadata/\n"
    "     COMMUNITY: https://community.oracle.com/\n\n"
    
    "  2. SYSTEMD COMPONENTS (systemd, systemd-logind, systemd-udevd, systemd-modules-load, systemd-networkd, systemd-vconsole-setup):\n"
    "     PRIMARY: https://docs.oracle.com/en/operating-systems/oracle-linux/ (Oracle Linux systemd configs)\n"
    "     SECONDARY: https://www.freedesktop.org/wiki/Software/systemd/ (Official systemd docs)\n"
    "     TERTIARY: https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/\n"
    "     MAN PAGES: https://man7.org/linux/man-pages/\n\n"
    
    "  3. STORAGE/MULTIPATH (multipathd, lvm, iscsid, blkdeactivate, device-mapper):\n"
    "     PRIMARY: https://docs.oracle.com/en/engineered-systems/exadata-database-machine/ (Storage section)\n"
    "     SECONDARY: https://access.redhat.com/documentation/ (Device Mapper Multipath)\n"
    "     TERTIARY: https://docs.oracle.com/en/operating-systems/oracle-linux/ (Storage chapter)\n"
    "     KERNEL: https://www.kernel.org/doc/html/latest/admin-guide/device-mapper/\n\n"
    
    "  4. KERNEL/HARDWARE (kernel, kdumpctl, mcelog, rngd, RDS/IB, RDMA):\n"
    "     PRIMARY: https://docs.oracle.com/en/operating-systems/uek/ (Unbreakable Enterprise Kernel)\n"
    "     SECONDARY: https://www.kernel.org/doc/html/latest/ (Official kernel docs)\n"
    "     RDMA/IB: https://www.kernel.org/doc/Documentation/infiniband/\n"
    "     TERTIARY: https://access.redhat.com/documentation/ (Kernel section)\n\n"
    
    "  5. NETWORKING (NetworkManager, nm-dispatcher, network, dbus-daemon):\n"
    "     PRIMARY: https://docs.oracle.com/en/engineered-systems/exadata-database-machine/ (Network fabric)\n"
    "     SECONDARY: https://docs.oracle.com/en/operating-systems/oracle-linux/ (Networking)\n"
    "     NM DOCS: https://networkmanager.dev/docs/\n"
    "     D-BUS: https://www.freedesktop.org/wiki/Software/dbus/\n\n"
    
    "  6. OTHER COMPONENTS (dracut, journal, Keepalived, etc.):\n"
    "     PRIMARY: https://docs.oracle.com/en/operating-systems/oracle-linux/\n"
    "     SECONDARY: https://access.redhat.com/documentation/\n\n"
    
    "  7. FALLBACK (if vendor documentation unavailable):\n"
    "     - serverfault.com, stackoverflow.com (use cautiously)\n\n"
    
    "CITATION REQUIREMENTS: Always provide full URLs in 'sources'. Include up to 3 additional authoritative references in 'additional_sources' with brief notes.\n\n"

    "STRICT OUTPUT: Respond with only this JSON object (no prose outside JSON):\n"
    "{{\n"
    "  \"label\": 1 | 0 | -1,\n"
    "  \"confidence\": <float between 0.0 and 1.0>,\n"
    "  \"reason\": \"<multi-sentence reasoning beginning with 'Severity Assessment:'>\",\n"
    "  \"sources\": [\"<url>\"],\n"
    "  \"additional_sources\": [\"<url - brief note>\"]\n"
    "}}\n"
)


def build_user_prompt(record: Dict[str, Any], reason_min_lines: int = 5, 
                      historical_context: Optional[str] = None) -> str:
    """Build user prompt with structured JSON object containing incident fields."""
    prompt_obj = {
        "incident_id": record.get("incident_id", ""),
        "component": record.get("component", ""),
        "message_structure": record.get("message_structure", ""),
        "log_window": record.get("log_window", ""),
        "dev_feedback": record.get("dev_feedback", ""),
    }
    
    prompt = f"""Incident JSON:
```json
{json.dumps(prompt_obj, ensure_ascii=False, indent=2)}
```
"""
    
    if historical_context:
        prompt += f"\n{historical_context}\n"
    
    prompt += f"""
Return JSON only: {{"label": 1|0|-1, "confidence": 0.0..1.0, "reason": ">= {reason_min_lines} sentences", "sources": ["..."], "additional_sources": ["..."]}}
"""
    return prompt


# ------------------------- TA LLM call -------------------------
def call_ta_llm(
    record: Dict[str, Any],
    ta: TaLLLMAccess,
    ta_model: str,
    reason_min_lines: int = 5,
    max_retries: int = 3,
    memory: Optional['LabelMemory'] = None,
    current_label: Optional[str] = None,
) -> Tuple[int, str, float, List[str], List[str]]:
    """Returns (label, reason, confidence, sources, additional_sources)."""
    cache_key = _cache_key(record, ta_model, reason_min_lines)
    with _CACHE_LOCK:
        if cache_key in _LLM_CACHE:
            return _LLM_CACHE[cache_key]

    if not remaining_calls():
        return -1, "arbiter_skipped:max_calls_reached", 0.0, [], []

    if _LIMITER is not None:
        _LIMITER.wait()

    # Build historical context if memory is enabled
    historical_context = None
    if memory and current_label:
        component = record.get("component", "")
        message_structure = record.get("message_structure", "")
        history = memory.get_history(component, message_structure)
        
        if history:
            ctx_lines = ["### HISTORICAL CONTEXT ###"]
            ctx_lines.append(f"This incident pattern has been seen {len(history)} time(s) in previous labels:")
            for h in history[-3:]:  # Show last 3 occurrences
                label_name = h['label']
                classification = {1: "Critical", 0: "Ignorable", -1: "Ambiguous"}.get(h['classification'], "Unknown")
                conf = h['confidence']
                ctx_lines.append(f"  • Label: {label_name} → Classification: {classification} (confidence: {conf:.2f})")
                # Truncate reason to first 200 chars
                reason_preview = h['reason'][:200].replace('\n', ' ')
                ctx_lines.append(f"    Reasoning: {reason_preview}...")
            ctx_lines.append("\nConsider this historical context when making your classification decision.")
            historical_context = "\n".join(ctx_lines)

    system_prompt = SYSTEM_PROMPT.format(reason_min_lines=reason_min_lines)
    user_prompt = build_user_prompt(record, reason_min_lines=reason_min_lines, 
                                    historical_context=historical_context)

    prompt = (
        "[SYSTEM]\n" + system_prompt + "\n\n" +
        "[USER]\n" + user_prompt + "\n"
    )

    last_err: str | None = None
    for attempt in range(max_retries + 1):
        try:
            text = ta.ta_request(prompt, ta_model)
            if not text:
                last_err = "empty_response"
                raise RuntimeError("empty_response")

            # Extract or coerce JSON
            content = text.strip()
            if content.startswith("```"):
                content = re.sub(r"^```[a-zA-Z0-9]*\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            brace = re.search(r"\{.*\}", content, re.S)
            candidate_texts = [content]
            if brace:
                candidate_texts.append(brace.group(0))

            data = None
            for cand in candidate_texts:
                try:
                    data = json.loads(cand)
                    break
                except json.JSONDecodeError:
                    continue

            if data is None:
                last_err = "invalid_json"
                raise RuntimeError("invalid_json")

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
                reason = (reason if isinstance(reason, str) else "") + f"\n\n[Note: reason had fewer than {reason_min_lines} required sentences.]"
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
            _incr_calls()
            return result

        except Exception:
            if attempt >= max_retries:
                return -1, f"arbiter_error:{last_err or 'unknown'}", 0.0, [], []
            # basic backoff
            time.sleep(1.0 + attempt * 0.7)


# ------------------------- Main flow -------------------------

def _generate_diff_report(stats: Dict[str, Any], output_dir: str, label: str):
    """Generate a summary report of differential analysis."""
    report_path = os.path.join(output_dir, f"diff_report_{label}.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"DIFFERENTIAL ANALYSIS REPORT\n")
        f.write(f"Label: {label}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 80 + "\n\n")
        
        f.write(f"SUMMARY STATISTICS\n")
        f.write(f"-" * 80 + "\n")
        f.write(f"Total incidents processed: {stats['total']}\n")
        f.write(f"NEW incidents: {stats['new']} ({stats['new']/max(stats['total'],1)*100:.1f}%)\n")
        f.write(f"RECURRING incidents: {stats['recurring']} ({stats['recurring']/max(stats['total'],1)*100:.1f}%)\n")
        f.write(f"\n")
        f.write(f"Critical (label=1) breakdown:\n")
        f.write(f"  - NEW critical issues: {stats['critical_new']}\n")
        f.write(f"  - RECURRING critical issues: {stats['critical_recurring']}\n")
        f.write(f"\n")
        
        if stats['classification_changes']:
            f.write(f"CLASSIFICATION CHANGES (from previous labels)\n")
            f.write(f"-" * 80 + "\n")
            for change in stats['classification_changes']:
                f.write(f"  • Component: {change['component']}\n")
                f.write(f"    Previous: {change['from']} (in {change['label']}) → Current: {change['to']}\n")
            f.write(f"\nTotal classification changes: {len(stats['classification_changes'])}\n")
        else:
            f.write(f"No classification changes detected.\n")
        
        f.write(f"\n" + "=" * 80 + "\n")
    
    print(f"\nDiff report generated: {report_path}")
    print(f"NEW: {stats['new']}, RECURRING: {stats['recurring']}, Changes: {len(stats['classification_changes'])}")


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if "log_window" not in df.columns:
        if "messages" in df.columns:
            df["log_window"] = df["messages"]
        else:
            raise ValueError("Input CSV must contain 'log_window' or 'messages'.")
    if "host" not in df.columns:
        if "hostname" in df.columns:
            df["host"] = df["hostname"]
        else:
            df["host"] = ""
    if "dev_feedback" not in df.columns:
        df["dev_feedback"] = ""
    if "incident_id" not in df.columns:
        # try to synthesize a stable id
        df["incident_id"] = [f"inc_{i}" for i in range(len(df))]
    if "component" not in df.columns:
        df["component"] = ""
    return df


def run(ambiguous_csv: str, outdir: str, ta_model: str,
    qps: float, burst: int, max_calls: int,
    max_retries: int,
    reason_min_lines: int = 5,
    out_csv_name: str = "triaged_results_grok.csv",
    enable_memory: bool = False,
    memory_dir: Optional[str] = None,
    current_label: Optional[str] = None) -> None:
    global _LIMITER, _MAX_CALLS
    _LIMITER = TokenBucket(rate_per_sec=qps, burst=burst)
    _MAX_CALLS = max_calls if max_calls and max_calls > 0 else None

    df = pd.read_csv(ambiguous_csv)
    df = _normalize_df(df)

    target_dir = os.path.join(outdir, "grok_outputs")
    os.makedirs(target_dir, exist_ok=True)
    
    # Initialize memory if enabled
    memory = None
    if enable_memory:
        if not memory_dir:
            memory_dir = os.path.join(outdir, "memory")
        if not current_label:
            current_label = datetime.now().strftime("label_%Y%m%d_%H%M%S")
        memory = LabelMemory(memory_dir)
        
        # Display memory statistics
        stats = memory.get_statistics()
        print("\n" + "="*80)
        print("MEMORY FEATURE ENABLED")
        print("="*80)
        print(f"Current label: {current_label}")
        print(f"Tracking capacity: Last {memory.max_labels} labels")
        print(f"Unique patterns in memory: {stats['total_patterns']}")
        print(f"Labels currently tracked: {stats['labels_tracked']}")
        if stats['oldest_label']:
            print(f"Oldest label: {stats['oldest_label']}")
            print(f"Newest label: {stats['newest_label']}")
        if stats['label_counts']:
            print("\nIncidents per label:")
            for label, count in sorted(stats['label_counts'].items()):
                print(f"  • {label}: {count} incidents")
        else:
            print("No historical data found (first run)")
        print("="*80 + "\n")

    # Open sinks
    label_files = {
        1: open(os.path.join(target_dir, "label_1.jsonl"), "w", encoding="utf-8"),
        0: open(os.path.join(target_dir, "label_0.jsonl"), "w", encoding="utf-8"),
        -1: open(os.path.join(target_dir, "label_-1.jsonl"), "w", encoding="utf-8"),
    }

    csv_path = os.path.join(target_dir, out_csv_name)
    
    # Add memory-related columns if enabled
    base_fieldnames = [
        "incident_id", "host", "component", "label", "confidence", "reason",
        "evidence", "additional_sources", "log_window", "dev_feedback",
    ]
    if enable_memory:
        base_fieldnames.extend([
            "incident_status", "first_seen_label", "last_seen_label", 
            "occurrence_count", "previous_classifications"
        ])
    
    fieldnames = base_fieldnames + [c for c in df.columns if c not in set(base_fieldnames)]

    ta = TaLLLMAccess()
    
    # Track statistics for diff report
    stats = {
        'new': 0, 'recurring': 0, 'total': 0,
        'critical_new': 0, 'critical_recurring': 0,
        'classification_changes': []
    }
    
    total_incidents = len(df)
    print(f"\nProcessing {total_incidents} incidents...")
    print(f"Output: {csv_path}\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()  # Flush header immediately

        for idx, r in df.iterrows():
            rec_d = r.to_dict()
            component = r.get("component", "")
            message_structure = r.get("message_structure", "")
            
            # Get historical status if memory enabled
            hist_status = None
            if memory:
                hist_status = memory.get_incident_status(component, message_structure, current_label)
            
            label, reason, conf, sources, add_sources = call_ta_llm(
                record=rec_d,
                ta=ta,
                ta_model=ta_model,
                reason_min_lines=reason_min_lines,
                max_retries=max_retries,
                memory=memory,
                current_label=current_label,
            )

            out_row: Dict[str, Any] = {
                "incident_id": r.get("incident_id", ""),
                "host": r.get("host", ""),
                "component": component,
                "label": int(label),
                "confidence": round(float(conf), 4),
                "reason": reason,
                "evidence": ";".join(map(str, sources or [])),
                "additional_sources": ";".join(map(str, add_sources or [])),
                "log_window": r.get("log_window", ""),
                "dev_feedback": str(r.get("dev_feedback", "")),
            }
            
            # Add memory-related fields
            if enable_memory and hist_status:
                out_row["incident_status"] = hist_status['status']
                out_row["first_seen_label"] = hist_status['first_seen_label']
                out_row["last_seen_label"] = hist_status['last_seen_label']
                out_row["occurrence_count"] = hist_status['occurrence_count']
                out_row["previous_classifications"] = ";".join(hist_status['classification_history'])
                
                # Update statistics
                stats['total'] += 1
                if hist_status['status'] == 'NEW':
                    stats['new'] += 1
                    if int(label) == 1:
                        stats['critical_new'] += 1
                else:
                    stats['recurring'] += 1
                    if int(label) == 1:
                        stats['critical_recurring'] += 1
                    # Check for classification changes
                    if hist_status['classification_history']:
                        last_class = hist_status['classification_history'][-1].split(':L')[1].split('(')[0]
                        if last_class != str(label):
                            stats['classification_changes'].append({
                                'component': component,
                                'from': last_class,
                                'to': str(label),
                                'label': hist_status['last_seen_label']
                            })
                
                # Record to memory
                memory.add_incident(
                    label=current_label,
                    component=component,
                    message_structure=message_structure,
                    classification=int(label),
                    reason=reason,
                    confidence=float(conf)
                )
            
            # propagate extra columns
            for col in fieldnames:
                if col in out_row:
                    continue
                val = r.get(col, "")
                if isinstance(val, float) and np.isnan(val):
                    val = ""
                out_row[col] = val

            writer.writerow(out_row)
            csvfile.flush()  # Flush after each write for real-time visibility
            label_files[int(label)].write(json.dumps(out_row) + "\n")
            
            # Print progress every 10 incidents or at milestones
            processed = idx + 1 if isinstance(idx, int) else len(df[:idx+1])
            if processed % 10 == 0 or processed == total_incidents:
                pct = (processed / total_incidents) * 100
                new_count = stats.get('new', 0)
                rec_count = stats.get('recurring', 0)
                if enable_memory:
                    print(f"Progress: {processed}/{total_incidents} ({pct:.1f}%) | "
                          f"NEW: {new_count}, RECURRING: {rec_count}")
                else:
                    print(f"Progress: {processed}/{total_incidents} ({pct:.1f}%)")

    for fh in label_files.values():
        fh.close()
    
    # Print completion summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Total incidents processed: {total_incidents}")
    if enable_memory:
        print(f"  NEW: {stats['new']} ({stats['new']/max(total_incidents,1)*100:.1f}%)")
        print(f"  RECURRING: {stats['recurring']} ({stats['recurring']/max(total_incidents,1)*100:.1f}%)")
        print(f"  Classification changes: {len(stats['classification_changes'])}")
    print(f"\nOutputs:")
    print(f"  Main CSV: {csv_path}")
    for lbl in [1, 0, -1]:
        jsonl_path = os.path.join(target_dir, f"label_{lbl}.jsonl")
        print(f"  Label {lbl} JSONL: {jsonl_path}")
    
    # Save memory and generate diff report
    if memory:
        memory.finalize_label(current_label)
        _generate_diff_report(stats, target_dir, current_label)
    
    print("="*80 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Arbitrate ambiguous incidents using Grok-3/Grok-4 via Oracle TA access")
    ap.add_argument("--ambiguous-csv", required=True, help="CSV produced by triage_embeddings_report.py (ambiguous rows)")
    ap.add_argument("--outdir", required=True, help="Output directory (CSV + JSONL will be placed under grok_outputs/)")
    ap.add_argument("--ta-model", default="grok3", choices=["grok3", "grok4"], help="TA workflow model to use")
    ap.add_argument("--qps", type=float, default=0.5, help="Average LLM calls per second")
    ap.add_argument("--burst", type=int, default=2, help="Burst size for rate limiter")
    ap.add_argument("--max-calls", type=int, default=200, help="Max TA calls for this run (0 = unlimited)")
    ap.add_argument("--max-retries", type=int, default=3, help="Max retries for TA LLM calls")
    ap.add_argument("--reason-min-lines", type=int, default=5, help="Minimum sentences required in LLM reasoning")
    ap.add_argument("--out-csv-name", default="triaged_results_grok.csv")
    
    # Memory options
    ap.add_argument("--memory", action="store_true", 
                    help="Enable historical memory tracking across labels for differential analysis")
    ap.add_argument("--memory-dir", type=str, default=None,
                    help="Directory to store historical memory (default: <outdir>/memory)")
    ap.add_argument("--current-label", type=str, default=None,
                    help="Label identifier for this run (e.g., OSS_MAIN_LINUX.X64_250108). "
                         "Auto-generated if not provided.")
    
    args = ap.parse_args()

    run(
        ambiguous_csv=args.ambiguous_csv,
        outdir=args.outdir,
        ta_model=args.ta_model,
        qps=args.qps,
        burst=args.burst,
        max_calls=args.max_calls,
        max_retries=args.max_retries,
        reason_min_lines=args.reason_min_lines,
        out_csv_name=args.out_csv_name,
        enable_memory=args.memory,
        memory_dir=args.memory_dir,
        current_label=args.current_label,
    )


if __name__ == "__main__":
    main()
