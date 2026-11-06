"""
Summarize ClickHouse runs of Redbench workloads.

Usage: analyze.py <results_dir>   where <results_dir>/<workload>/run_clickhouse_*.parquet
"""

import os
import sys

import pandas as pd

ORDER = ["clickhouse_baseline", "clickhouse_predcache", "clickhouse_rescache", "clickhouse_predcache_rescache"]


def staleness(df):
    """Hits and stale hits of a cache that keeps results until TTL (never invalidated on writes), with infinite TTL."""
    cached_at = {}  # sql -> position when the entry was created
    last_write = {}  # table -> position of last write
    hits = stale = 0
    for pos, r in enumerate(df.itertuples(index=False)):
        if r.query_type == "select":
            if r.query in cached_at:
                hits += 1
                if any(last_write.get(t, -1) > cached_at[r.query] for t in r.ch_read_tables):
                    stale += 1
            else:
                cached_at[r.query] = pos
        for t in r.ch_write_tables:
            last_write[t] = pos
    return hits, stale


def main(results_dir):
    rows = []
    for wl in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, wl)
        if not os.path.isdir(d):
            continue
        runs = {f[4:-8]: pd.read_parquet(os.path.join(d, f)) for f in os.listdir(d) if f.startswith("run_clickhouse") and f.endswith(".parquet")}
        if "clickhouse_baseline" not in runs:
            continue
        base = runs["clickhouse_baseline"].set_index("query_idx")
        hits, stale = staleness(runs["clickhouse_baseline"])
        print(f"{wl}: native query cache (no invalidation, infinite TTL) would serve {hits} hits, "
              f"{stale} of them stale ({100 * stale / max(hits, 1):.1f}%)")
        for name in [n for n in ORDER if n in runs]:
            r = runs[name].set_index("query_idx")
            common = base.index.intersection(r.index)
            b, c = base.loc[common], r.loc[common]
            sel = c["query_type"] == "select"
            rel = (c["runtime_sec"] / b["runtime_sec"]).where(b["runtime_sec"] > 0)
            rows.append({
                "workload": wl,
                "config": name.replace("clickhouse_", ""),
                "queries": len(c),
                "total_s": c["runtime_sec"].sum(),
                "select_s": c.loc[sel, "runtime_sec"].sum(),
                "dml_s": c.loc[~sel, "runtime_sec"].sum(),
                "speedup_total": b["runtime_sec"].sum() / c["runtime_sec"].sum(),
                "speedup_select": b.loc[sel, "runtime_sec"].sum() / c.loc[sel, "runtime_sec"].sum(),
                "mean_rel_runtime": rel.mean(),
                "median_rel_runtime": rel.median(),
                "qc_hits": int(c["qc_hits"].fillna(0).sum()),
                "qcc_hits": int(c["qcc_hits"].fillna(0).sum()),
                "timeouts": int((c["error"] == "timeout").sum()),
            })
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
