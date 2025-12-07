# Log Analyzer Agent (Exadata/Exascale)

This is a **retrieval-augmented triage agent** for Exadata/Exascale logs.  
It takes **filtered logs**, builds **incident windows** (time-gap + component/PID boundaries),
matches them against **known bug signatures (label=1)** and **ignorable signatures (label=0)**,
and routes everything else to **label=-1** (needs triage).  
If an `OPENAI_API_KEY` is present, the agent will optionally ask an LLM to arbitrate ambiguous cases.

## Project layout

```
log_ai_agent/
├── README.md
├── requirements.txt
├── sessionize.py
├── build_memories.py
├── triage.py
├── utils.py
├── data/
│   ├── known_bugs.csv                # you fill
│   ├── ignorable_patterns.csv        # you fill
│   ├── incidents.csv                 # produced by sessionize.py
│   ├── outputs/
│   │   ├── label_1.jsonl
│   │   ├── label_0.jsonl
│   │   └── label_-1.jsonl
└── examples/
    ├── known_bugs.sample.csv
    └── ignorable_patterns.sample.csv
```

## Quick start

1. **Collect and sessionize logs from daily OSS builds**:

```bash
# For today's build (e.g., OSS_MAIN_LINUX.X64_251105 for Nov 5, 2025)
python collect_image_scan.py \
  --label OSS_MAIN_LINUX.X64_251105 \
  --output-dir tables/

# Script will automatically:
# - Derive root path: /net/.../integration/logs/OSS_MAIN_LINUX.X64_251105
# - Fetch page.html from: http://100.70.110.36/integration_logs/OSS_MAIN_LINUX.X64_251105/
# - Generate: tables/image_error_sessions.csv and tables/systemctl_failed.csv

# For a different date (e.g., Nov 4, 2025):
python collect_image_scan.py \
  --label OSS_MAIN_LINUX.X64_251104 \
  --output-dir tables/

# Optional flags:
# --root <path>              Override auto-derived root path
# --skip-page-fetch          Don't fetch page.html (use existing file)
# --page-html <path>         Use specific page.html file
# --verbose                  Show matched/unmatched views from page.html
```

2. **Create/Fill** the two seed CSVs (you can start from the samples in `examples/`):

   - `data/known_bugs.csv` with columns:
     - `bug_id` (str), `context_window` (str; 5–15 lines), `component` (optional), `templates` (optional), `description` (optional)
   - `data/ignorable_patterns.csv` with columns:
     - `pattern_id` (str), `context_window` (str), `rule_regex` (optional), `component` (optional), `notes` (optional)

3. **Install dependencies** (CPU-friendly):

```bash
cd /mnt/data/log_ai_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. **Build retrieval memories** (embeddings + rule cache):

```bash
python build_memories_st.py \
  --known data/known_bugs.csv \
  --ignore data/ignorable_patterns.csv \
  --model sentence-transformers/all-MiniLM-L6-v2
```

5. **Generate ambiguous incidents** (MiniLM fast pass):

```bash
python triage_embeddings_report.py \
  --incidents tables/image_error_sessions.csv \
  --known-index data/.known_miniLM.pkl \
  --ignore-index data/.ignore_miniLM.pkl \
  --out-csv tables/minilm_ambiguous.csv \
  --out-confident-csv tables/minilm_confident.csv
```

6. **Run Grok arbitration** on ambiguous cases:

```bash
python triage_grok_llm.py \
  --ambiguous-csv tables/minilm_ambiguous.csv \
  --outdir tables \
  --ta-model grok3
```

Outputs:

- `tables/grok_outputs/triaged_results_grok.csv` — all LLM decisions
- `tables/grok_outputs/label_1.jsonl`, `label_0.jsonl`, `label_-1.jsonl` — per-label artifacts

---

## Legacy workflow (if not using Grok)

3. **Sessionize your filtered log file** (component+PID+time-gap):

```bash
# Example input: /mnt/data/raw_log_1_labeled.log  (CSV-ish lines with optional trailing label)
python sessionize.py   --input /mnt/data/raw_log_1_labeled.log   --output data/incidents.csv   --gap-seconds 10
```

4. **Build retrieval memories** (embeddings + rule cache):

```bash
python build_memories.py   --known data/known_bugs.csv   --ignore data/ignorable_patterns.csv
```

5. **Run triage** (produce 1/0/-1 files):

```bash
# Optional: export OPENAI_API_KEY=xxxxxxxx  # enables LLM arbitration for ambiguous cases
python triage.py   --incidents data/incidents.csv   --known-index data/.known_index.pkl   --ignore-index data/.ignore_index.pkl   --outdir data/outputs
```

Outputs:

- `data/outputs/label_1.jsonl` (reproduced/known)
- `data/outputs/label_0.jsonl` (ignorable)
- `data/outputs/label_-1.jsonl` (needs triage; sorted by novelty/confidence)
- `data/outputs/triage_all.csv` (one row per incident with decision details)

## Notes

- **No GPU required.** Uses small sentence-transformer (MiniLM) by default; falls back to TF-IDF if embeddings are unavailable.
- You can run this _entirely on filtered logs_. Raw 100k-line/node logs are _optional_ (use them later to learn normal templates/frequencies).
- Add more rows to `known_bugs.csv` and `ignorable_patterns.csv` weekly as you adjudicate `-1` items. This is your continuous learning loop.

## Use MiniLM embeddings (more AI-like semantics)

If you want deeper semantic matching, switch from TF‑IDF to **MiniLM** embeddings.

### Build MiniLM indexes

```bash
python build_memories_st.py   --known data/known_bugs.csv   --ignore data/ignorable_patterns.csv   --model sentence-transformers/all-MiniLM-L6-v2
```

This creates:

- `data/.known_miniLM.pkl`
- `data/.ignore_miniLM.pkl`

### Triage with MiniLM

```bash
python triage_st.py   --incidents data/incidents.csv   --known-index data/.known_miniLM.pkl   --ignore-index data/.ignore_miniLM.pkl   --outdir data/outputs   --model sentence-transformers/all-MiniLM-L6-v2
```

Notes:

- CPU-only is fine. Encoding a few thousand incidents will complete in a reasonable time on a typical workstation.
- Thresholds inside `triage_st.py` default to `t_known=0.72`, `t_ignore=0.70`. Tune per your data.

## Grok-3/Grok-4 arbitration via Oracle TA

If you want to arbitrate ambiguous incidents (-1) using Oracle TA workflows that call Grok-3/Grok-4 (with real-time search), use:

1. Generate ambiguous-only CSV using embeddings fast path:

```bash
python triage_embeddings_report.py \
  --incidents tables/image_error_sessions.csv \
  --known-index data/.known_miniLM.pkl \
  --ignore-index data/.ignore_miniLM.pkl \
  --out-csv tables/minilm_ambiguous.csv \
  --out-confident-csv tables/minilm_confident.csv
```

2. Run Grok arbiter on ambiguous cases:

```bash
python triage_grok_llm.py \
  --ambiguous-csv tables/minilm_ambiguous.csv \
  --outdir tables \
  --ta-model grok3
```

Outputs:

- `tables/grok_outputs/triaged_results_grok.csv` — all LLM decisions
- `tables/grok_outputs/label_1.jsonl`, `label_0.jsonl`, `label_-1.jsonl` — per-label artifacts

Flags:

- `--ta-model` accepts `grok3` (default) or `grok4`.
- `--qps`, `--burst`, and `--max-calls` control rate limiting.
- `--reason-min-lines` enforces minimum reasoning length.
