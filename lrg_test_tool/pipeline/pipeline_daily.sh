#!/usr/bin/env bash
set -euo pipefail

# ============
# CONFIG
# ============
ROOT="/ade/shratyag_v7/tklocal"        # workspace root
SCRIPTS="$ROOT/scripts"                         # directory where your *.py live
LOGS="$ROOT/logs"
RUNTIMES="$ROOT/scripts/runtimes"
SUITES_DIR="$ROOT/suites"
WORK="$ROOT/work"
mkdir -p "$LOGS" "$RUNTIMES" "$SUITES_DIR" "$WORK"

# Sources for extraction
TSC_SRC="/ade/shratyag_v7/oss/test/tsage/src"     # folder with *.tsc files
CTRL_FILE="/ade/shratyag_v7/oss/test/tsage/src/tsagexacldsuite.tsc"
CTRL_START_LINE=880
PLATFORM="Exascale"

# Optional allow/skip lists used by extract_descriptions.py
ALLOW_SETUPS="$SCRIPTS/setups_allowlist.txt"   # optional (will skip if missing)
SKIP_SETUPS="$SCRIPTS/skip_setups.txt"         # optional
SKIP_FLAGS="$SCRIPTS/skip_flags.txt"           # optional
TEST_DESCRIPTIONS="$SCRIPTS/description/test_descriptions.txt"

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


log(){ printf '[%s] %s\n' "$(date +'%F %T')" "$*"; }

# ============
# 0) sanity
# ============
command -v python3 >/dev/null || { echo "python3 not found"; exit 2; }
command -v jq >/dev/null || { echo "jq not found"; exit 2; }

# ============
# 1) Collect runtimes (avg of last 3) for each LRG
#     – If your collect_runtimes.py uses different args, adjust here.
# ============
log "1) collect_runtimes.py → $RUNTIMES_JSON"
python3 "$SCRIPTS/runtimes/collect_runtimes.py" \
  --l2t "$L2T_JSON" \
  --cmd 'curl -sS "https://apex.oraclecorp.com/pls/apex/lrg_times/MAIN/lrg/{lrg}"' \
  --out "$RUNTIMES_JSON" \
  --last 3 --workers 8 \
  >"$LOGS/04_collect_runtimes.log" 2>&1

sleep 10

# ============
# 2) Merge runtimes into LRG map
# ============
log "2) merge_runtimes_into_lrg.py → $LRG_WITH_RT_JSON"
python3 "$SCRIPTS/runtimes/merge_runtimes_into_lrg.py" \
  --lrgs "$LRG_MAP_JSON" \
  --rt   "$RUNTIMES_JSON" \
  --out  "$LRG_WITH_RT_JSON" \
  --write-hours \
  >"$LOGS/05_merge_runtimes.log" 2>&1

sleep 10

# ============
# 3) Merge runtimes into LRG map
# ============
log "3) merge_test_descriptions.py → $TESTS_JSON"
python3 "$SCRIPTS/descriptions/merge_test_descriptions.py" \
  $TESTS_JSON \
  $TEST_DESCRIPTIONS
  >"$LOGS/03_merge_test_descriptions.log" 2>&1

sleep 10

# ============
# 4) Build profiles (TEST+LRG) → profiles.json
# ============
log "4) build_profiles_fast.py → $PROFILES_JSON"
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
# 5) Build FTS index (fresh)
# ============
log "5) build_fts_index.py → $FTS_DB"
python3 "$SCRIPTS/build_fts_index.py" \
  --profiles "$PROFILES_JSON" \
  --db "$FTS_DB" \
  --wipe \
  >"$LOGS/09_build_fts_index.log" 2>&1
  
sleep 10

# ============
# 6) Smoke query
# ============
log "6) query_fts_refined.py (smoke)"
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
  >"$LOGS/10_smoke_query.log" 2>&1 || true

log "✅ Pipeline complete."