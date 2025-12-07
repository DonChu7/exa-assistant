#!/usr/bin/env bash
set -euo pipefail

# ============
# CONFIG
# ============
ROOT="/ade/shratyag_v8/tklocal"        # workspace root
SCRIPTS="$ROOT/scripts"                         # directory where your *.py live
LOGS="$ROOT/logs"
RUNTIMES="$ROOT/scripts/runtimes"
SUITES_DIR="$ROOT/suites"
WORK="$ROOT/work"
mkdir -p "$LOGS" "$RUNTIMES" "$SUITES_DIR" "$WORK"

# Sources for extraction
TSC_SRC="/ade/shratyag_v8/oss/test/tsage/src"     # folder with *.tsc files
CTRL_FILE="/ade/shratyag_v8/oss/test/tsage/src/tsagexacldsuite.tsc"
CTRL_START_LINE=880
PLATFORM="Exascale"

# Optional allow/skip lists used by extract_descriptions.py
ALLOW_SETUPS="$SCRIPTS/setups_allowlist.txt"   # optional (will skip if missing)
SKIP_SETUPS="$SCRIPTS/skip_setups.txt"         # optional
SKIP_FLAGS="$SCRIPTS/skip_flags.txt"           # optional

# Runtimes API template used by collect_runtimes.py (if your script needs it)
# Update if your script pulls from a different source.
RUNTIME_URL_TEMPLATE='http://example/api?runs={lrg}'

# Outputs
TESTS_JSON="$ROOT/tests_filtered.json"
TESTS_EXTRACTED_JSON="$ROOT/tests_extracted.json"
LRG_MAP_JSON="$ROOT/lrg_map.json"
RUNTIMES_JSON="$ROOT/runtimes_avg_last5.json"
LRG_WITH_RT_JSON="$ROOT/lrg_map_with_runtimes.json"
LRG_WITH_SUITES_JSON="$ROOT/lrg_map_with_suites.json"
PROFILES_JSON="$ROOT/profiles.json"
FTS_DB="$ROOT/profiles_fts.db"
T2L_JSON="$ROOT/test_to_lrgs.json"
L2T_JSON="$ROOT/lrg_to_tests.json"
T2L_CLEAN_JSON="$ROOT/test_to_lrgs_clean.json"
PROFILES_REPORT="$ROOT/profiles_report.json"
TESTS_IN_LRGS_JSON="$ROOT/tests_filtered_in_lrgs.json"
RAG_INDEX="$ROOT/rag_tests.faiss"
RAG_INDEX_META="$ROOT/rag_tests.jsonl"

log(){ printf '[%s] %s\n' "$(date +'%F %T')" "$*"; }

# ============
# 0) sanity
# ============
command -v python3 >/dev/null || { echo "python3 not found"; exit 2; }
command -v jq >/dev/null || { echo "jq not found"; exit 2; }

# ============
# 1) Extract TEST metadata from *.tsc -> tests_filtered.json
# ============
log "1) extract_descriptions.py → $TESTS_EXTRACTED_JSON"
EXTRACT_ARGS=( "$TSC_SRC" --out-json "$TESTS_EXTRACTED_JSON" --glob "tsag*.tsc" --encoding latin-1 )
[[ -f "$ALLOW_SETUPS" ]] && EXTRACT_ARGS+=( --setups-file "$ALLOW_SETUPS" )
[[ -f "$SKIP_SETUPS"  ]] && EXTRACT_ARGS+=( --skip-setups-file "$SKIP_SETUPS" )
[[ -f "$SKIP_FLAGS"   ]] && EXTRACT_ARGS+=( --skip-flags-file "$SKIP_FLAGS" )
python3 "$SCRIPTS/extract_descriptions.py" "${EXTRACT_ARGS[@]}" \
  >"$LOGS/01_extract_descriptions.log" 2>&1

sleep 10

# ============
# 2) Parse LRG sections from control map -> lrg_map.json
# ============
log "2) parse_lrg_map.py → $LRG_MAP_JSON"
python3 "$SCRIPTS/parse_lrg_map.py" \
  "$CTRL_FILE" \
  --start-line "$CTRL_START_LINE" \
  --platform "$PLATFORM" \
  --runtime null \
  --out-json "$LRG_MAP_JSON" \
  --print-first \
  --debug-unmatched-if "$LOGS/unmatched_if.log" \
  >"$LOGS/02_parse_lrg_map.log" 2>&1

sleep 10

# ============
# 3) Build test↔lrg mappings
# ============
log "3) build_mappings.py → $T2L_JSON / $L2T_JSON"
python3 "$SCRIPTS/build_mappings.py" \
  "$TESTS_EXTRACTED_JSON" \
  "$LRG_MAP_JSON" \
  "$T2L_JSON" \
  "$L2T_JSON" \
  >"$LOGS/03_build_mappings.log" 2>&1

sleep 10

# (Optional) remove empty test→[] entries
log "3b) clean_empty_tests.py → $T2L_JSON"
python3 "$SCRIPTS/clean_empty_tests.py" \
  --in "$T2L_JSON" \
  --out "$T2L_JSON" \
  >"$LOGS/03b_clean_empty_tests.log" 2>&1

sleep 10

# ============
# 4) Collect runtimes (avg of last 3) for each LRG
#     – If your collect_runtimes.py uses different args, adjust here.
# ============
log "4) collect_runtimes.py → $RUNTIMES_JSON"
python3 "$SCRIPTS/runtimes/collect_runtimes.py" \
  --l2t "$L2T_JSON" \
  --cmd 'curl -sS "https://apex.oraclecorp.com/pls/apex/lrg_times/MAIN/lrg/{lrg}"' \
  --out "$RUNTIMES_JSON" \
  --last 3 --workers 8 \
  >"$LOGS/04_collect_runtimes.log" 2>&1

sleep 10

# ============
# 5) Merge runtimes into LRG map
# ============
log "5) merge_runtimes_into_lrg.py → $LRG_WITH_RT_JSON"
python3 "$SCRIPTS/runtimes/merge_runtimes_into_lrg.py" \
  --lrgs "$LRG_MAP_JSON" \
  --rt   "$RUNTIMES_JSON" \
  --out  "$LRG_WITH_RT_JSON" \
  --write-hours \
  >"$LOGS/05_merge_runtimes.log" 2>&1

sleep 10

# ============
# 6) Generate suite text files and tag LRGs with suites
#     – If suite_wrapper.py + suites.py build the TXT files, run them first.
#     – Otherwise ensure $SUITES_DIR/*.txt already exist.
# ============
if [[ -f "$SCRIPTS/suites/suite_wrapper.py" ]]; then
  log "6a) suite_wrapper.py → emits suite TXT files into $SUITES_DIR"
  python3 "$SCRIPTS/suites/suite_wrapper.py" 
    >"$LOGS/06a_suite_wrapper.log" 2>&1 || true
fi

sleep 10

# Attach suites to LRGs
log "6b) add_suites_to_lrgs.py → $LRG_WITH_SUITES_JSON"
python3 "$SCRIPTS/suites/add_suites_to_lrgs.py" \
    "$LRG_WITH_RT_JSON" \
    "$SUITES_DIR" \
    "$LRG_WITH_SUITES_JSON" \
  >"$LOGS/06b_add_suites.log" 2>&1

sleep 10

# ============
# 7) (Optional) Filter tests down to only those that appear in any LRG
#     – Keeps profiles lean and consistent with LRG inventory.
# ============
log "7) filter_tests_by_lrgs.py → $TESTS_IN_LRGS_JSON"
python3 "$SCRIPTS/filter_tests_by_lrgs.py" \
  --tests "$TESTS_EXTRACTED_JSON" \
  --map   "$T2L_JSON" \
  --out-dir   "$ROOT" \
  >"$LOGS/07_filter_tests_by_lrgs.log" 2>&1 || cp -f "$TESTS_JSON" "$TESTS_IN_LRGS_JSON"

sleep 10

# ============
# 8) Build profiles (TEST+LRG) → profiles.json
# ============
log "8) build_profiles_fast.py → $PROFILES_JSON"
python3 "$SCRIPTS/build_profiles_fast.py" \
  --tests "$TESTS_EXTRACTED_JSON" \
  --lrgs  "$LRG_WITH_SUITES_JSON" \
  --out   "$PROFILES_JSON" \
  --pretty \
  --verify --report-out "$PROFILES_REPORT" \
  --print-examples 0 \
  >"$LOGS/08_build_profiles.log" 2>&1

sleep 10

# ============
# 9) Build FTS index (fresh)
# ============
log "9) build_fts_index.py → $FTS_DB"
python3 "$SCRIPTS/build_fts_index.py" \
  --profiles "$PROFILES_JSON" \
  --db "$FTS_DB" \
  --wipe \
  >"$LOGS/09_build_fts_index.log" 2>&1
  
sleep 10

# ============
# 10) Build RAG Index (fresh)
# ============
log "10) build_rag_index.py → $FTS_DB"
python3 "$SCRIPTS/build_rag_index.py" \
  --profiles "$PROFILES_JSON" \
  --index  "$RAG_INDEX" \
  --meta "$RAG_INDEX_META" \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  >"$LOGS/10_build_rag_index.log" 2>&1

sleep 10

# ============
# 11) Smoke query
# ============
log "11) query_fts_refined.py (smoke)"
python3 "$SCRIPTS/query_fts_refined.py" \
  --db "$FTS_DB" \
  --q "suite:EXAC_VOL setup xblockini" \
  --t2l "$T2L_JSON" \
  --l2t "$L2T_JSON" \
  --lrgs-json "$LRG_WITH_SUITES_JSON" \
  --structured-only \
  --require-setup xblockini \
  --require-flag  iscsi=true \
  --pool 10000000 \
  --lrg-order runtime \
  --k 20000
  >"$LOGS/11_smoke_query.log" 2>&1 || true


# ============
# 12) Smoke query
# ============
log "12) rag_query_tests.py (smoke)"
python3 "$SCRIPTS/rag_query_tests.py" \
  --index  "$RAG_INDEX" \
  --meta "$RAG_INDEX_META" \
  --q "Blockstore tests" \
  --k 10 \
  --pool-mult 8 \
  --t2l "$T2L_JSON" \
  --lrgs-json "$LRG_WITH_SUITES_JSON" 
  >"$LOGS/12_rag_query_tests.log" 2>&1 || true

log "✅ Pipeline complete."