#!/usr/bin/env bash
# Run the three Redbench paper workloads (Redset provisioned, instance 186, database 8, May 2024) on ClickHouse,
# in every cache configuration, sequentially.
#
# Usage: run_paper_workloads.sh <paper_workloads_dir> <user_files_dir> <results_dir> [workload ...] [-- config ...]
#   <paper_workloads_dir>  contains {generation,matching,baseline_round_robin}/workload.csv (downloaded by Redbench from OSF)
#   <user_files_dir>       the server's user_files_path, containing imdb_original/ and imdb_augmented_x2/
#                          (created with export_duckdb_to_parquet.py)
#   <results_dir>          results are written to <results_dir>/<workload>/run_clickhouse_<config>.parquet
# Environment: PYTHON (default: python), CLICKHOUSE_HOST (default: 127.0.0.1), CLICKHOUSE_PORT (HTTP, default: 8123)
set -euo pipefail

if [[ $# -lt 3 ]]; then
    sed -n '2,10p' "$0"
    exit 1
fi
WORKLOADS_DIR=$(realpath "$1")
USER_FILES=$(realpath "$2")
RESULTS=$(realpath -m "$3")
shift 3
HERE=$(dirname "$(realpath "$0")")
PY=${PYTHON:-python}

WORKLOADS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do WORKLOADS+=("$1"); shift; done
[[ $# -gt 0 ]] && shift
CONFIGS=("$@")
[[ ${#WORKLOADS[@]} -eq 0 ]] && WORKLOADS=(generation matching baseline_round_robin)
[[ ${#CONFIGS[@]} -eq 0 ]] && CONFIGS=(baseline rescache predcache allcaches)

for wl in "${WORKLOADS[@]}"; do
    # The generation workload runs on the schema augmented x2 (42 tables), the others on the original IMDb schema.
    if [[ "$wl" == generation ]]; then db=imdb_augmented_x2; else db=imdb_original; fi
    out="$RESULTS/$wl"
    mkdir -p "$out"
    for cfg in "${CONFIGS[@]}"; do
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $wl $cfg"
        "$PY" "$HERE/execute_queries_clickhouse.py" \
            --workload "$WORKLOADS_DIR/$wl/workload.csv" \
            --parquet_dir "$USER_FILES/$db" --user_files_rel "$db" \
            --host "${CLICKHOUSE_HOST:-127.0.0.1}" --port "${CLICKHOUSE_PORT:-8123}" \
            --config "$cfg" --out_dir "$out" > "$out/$cfg.log" 2>&1
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) done  $wl $cfg: $(grep -E '^Wrote' "$out/$cfg.log")"
    done
done
