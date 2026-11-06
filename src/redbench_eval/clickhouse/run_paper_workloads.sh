#!/usr/bin/env bash
# Run the three Redbench paper workloads (Redset provisioned, instance 186, database 8, May 2024) on ClickHouse,
# in every cache configuration, sequentially.
#
# Usage: run_paper_workloads.sh <redbench_root> [workload ...] [-- config ...]
#   <redbench_root> contains output/paper_workloads/, chserver/user_files/imdb_{original,augmented_x2}/, src/.venv/
set -euo pipefail

ROOT=$(realpath "$1"); shift
HERE=$(dirname "$(realpath "$0")")
PY="$ROOT/src/.venv/bin/python"

WORKLOADS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do WORKLOADS+=("$1"); shift; done
[[ $# -gt 0 ]] && shift
CONFIGS=("$@")
[[ ${#WORKLOADS[@]} -eq 0 ]] && WORKLOADS=(generation matching baseline_round_robin)
[[ ${#CONFIGS[@]} -eq 0 ]] && CONFIGS=(baseline rescache predcache allcaches)

for wl in "${WORKLOADS[@]}"; do
    if [[ "$wl" == generation ]]; then db=imdb_augmented_x2; else db=imdb_original; fi
    out="$ROOT/results/paper/$wl"
    mkdir -p "$out"
    for cfg in "${CONFIGS[@]}"; do
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $wl $cfg"
        "$PY" "$HERE/execute_queries_clickhouse.py" \
            --workload "$ROOT/output/paper_workloads/$wl/workload.csv" \
            --parquet_dir "$ROOT/chserver/user_files/$db" --user_files_rel "$db" \
            --config "$cfg" --out_dir "$out" > "$out/$cfg.log" 2>&1
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) done  $wl $cfg: $(grep -E '^Wrote' "$out/$cfg.log")"
    done
done
