# Memory Feature for triage_grok_llm.py

## Overview

The `--memory` flag enables historical tracking of incidents across multiple OSS label runs, allowing the Grok LLM to identify NEW vs RECURRING issues and provide historical context for better decision-making.

## Recent Improvements (Nov 2025)

### 1. Progress Indicator

- Real-time progress updates every 10 incidents
- Shows: `Progress: 20/50 (40%) | NEW: 5, RECURRING: 15`
- Final completion summary with statistics

### 2. Immediate CSV Visibility

- `csvfile.flush()` after each row write
- Results visible in real-time without waiting for script completion
- Enables monitoring with `tail -f` or opening CSV during processing

### 3. Memory Statistics on Startup

- Displays detailed memory state when `--memory` is enabled
- Shows:
  - Total unique patterns tracked
  - Number of labels in memory
  - Oldest/newest labels
  - Incident counts per label
- Helps validate memory feature is working correctly

## Features

### 1. Historical Context in Prompts

When an incident pattern has been seen before, the LLM receives:

- Previous label(s) where it occurred
- How it was classified previously (Critical/Ignorable/Ambiguous)
- Confidence scores from past runs
- Reasoning snippets from previous classifications

### 2. Differential Classification

Each incident is tagged with:

- **NEW**: First time seeing this pattern
- **RECURRING**: Seen in previous labels (up to last 5)

### 3. Enhanced Output Columns

With `--memory`, the CSV includes:

- `incident_status`: NEW or RECURRING
- `first_seen_label`: When first detected
- `last_seen_label`: Most recent prior occurrence
- `occurrence_count`: How many times seen before
- `previous_classifications`: History of classifications (e.g., "label1:L1(conf=0.85);label2:L0(conf=0.92)")

### 4. Automatic Diff Report

Generates `diff_report_<label>.txt` showing:

- Count of NEW vs RECURRING incidents
- Critical issues breakdown
- Classification changes between labels
- Trends and patterns

## Usage

### Basic (First Run)

```bash
python triage_grok_llm.py \
  --ambiguous-csv tables/minilm_ambiguous.csv \
  --outdir tables \
  --ta-model grok3 \
  --memory \
  --current-label OSS_MAIN_LINUX.X64_250101
```

### Subsequent Runs (Tracks Last 5 Labels)

```bash
# Day 2
python triage_grok_llm.py \
  --ambiguous-csv tables/minilm_ambiguous.csv \
  --outdir tables \
  --ta-model grok3 \
  --memory \
  --current-label OSS_MAIN_LINUX.X64_250108

# Day 3
python triage_grok_llm.py \
  --ambiguous-csv tables/minilm_ambiguous.csv \
  --outdir tables \
  --ta-model grok3 \
  --memory \
  --current-label OSS_MAIN_LINUX.X64_250115
```

### Custom Memory Directory

```bash
python triage_grok_llm.py \
  --ambiguous-csv tables/minilm_ambiguous.csv \
  --outdir tables \
  --ta-model grok3 \
  --memory \
  --memory-dir /path/to/shared/memory \
  --current-label OSS_MAIN_LINUX.X64_250108
```

## How It Works

### Pattern Matching

Incidents are uniquely identified by:

- **Component** (e.g., systemd, kernel, cellsrv)
- **Message Structure** (normalized pattern with placeholders)

Same pattern across labels = RECURRING incident.

### Memory Storage

- Stored in: `<outdir>/memory/label_memory.pkl`
- Automatically maintains last 5 labels
- Thread-safe for concurrent access
- Persisted after each run

### Historical Context Example

When the LLM sees a RECURRING incident, it receives:

```
### HISTORICAL CONTEXT ###
This incident pattern has been seen 2 time(s) in previous labels:
  • Label: OSS_MAIN_LINUX.X64_250101 → Classification: Critical (confidence: 0.85)
    Reasoning: Severity Assessment: Critical – The ohasd.service failure indicates...
  • Label: OSS_MAIN_LINUX.X64_250108 → Classification: Critical (confidence: 0.92)
    Reasoning: Severity Assessment: Critical – Recurring ohasd failure, likely Bug 37735911...

Consider this historical context when making your classification decision.
```

## Output Examples

### CSV with Memory Fields

```csv
incident_id,component,label,incident_status,first_seen_label,last_seen_label,occurrence_count,previous_classifications
inc_001,systemd,1,NEW,OSS_MAIN_LINUX.X64_250115,OSS_MAIN_LINUX.X64_250115,0,
inc_002,kernel,1,RECURRING,OSS_MAIN_LINUX.X64_250101,OSS_MAIN_LINUX.X64_250108,2,"250101:L1(conf=0.85);250108:L1(conf=0.92)"
inc_003,cellsrv,0,RECURRING,OSS_MAIN_LINUX.X64_250101,OSS_MAIN_LINUX.X64_250101,1,"250101:L0(conf=0.95)"
```

### Diff Report Sample

```
================================================================================
DIFFERENTIAL ANALYSIS REPORT
Label: OSS_MAIN_LINUX.X64_250115
Generated: 2025-01-15 14:30:22
================================================================================

SUMMARY STATISTICS
--------------------------------------------------------------------------------
Total incidents processed: 45
NEW incidents: 12 (26.7%)
RECURRING incidents: 33 (73.3%)

Critical (label=1) breakdown:
  - NEW critical issues: 3
  - RECURRING critical issues: 8

CLASSIFICATION CHANGES (from previous labels)
--------------------------------------------------------------------------------
  • Component: systemd
    Previous: -1 (in OSS_MAIN_LINUX.X64_250108) → Current: 1
  • Component: kernel
    Previous: 0 (in OSS_MAIN_LINUX.X64_250108) → Current: 1

Total classification changes: 2

================================================================================
```

## Benefits

1. **Informed Decisions**: LLM sees historical context and past reasoning
2. **Trend Detection**: Identify escalating issues (0→1 or -1→1)
3. **Reduced False Positives**: RECURRING benign issues stay classified as ignorable
4. **Focus on NEW**: Quickly identify which issues need immediate attention
5. **Audit Trail**: Track how incident classifications evolve over time

## Notes

- Memory is **optional** - script works normally without `--memory`
- Automatically limits to last 5 labels (configurable in code via `max_labels=5`)
- Memory file is portable - can be shared across systems
- If `--current-label` is omitted, auto-generates timestamp-based label
- Memory does NOT affect caching within a single run (separate feature)

## Troubleshooting

### Memory file corrupted

Delete `<outdir>/memory/label_memory.pkl` and restart with `--memory`

### Want to clear old labels

Delete memory file or manually edit with pickle/Python

### Change max labels tracked

Edit `LabelMemory(memory_dir, max_labels=5)` in code (default: 5)
