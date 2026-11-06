"""
Execute a Redbench workload (workload.csv) on ClickHouse, in the format expected by Redbench-Eval
(run_<system>_<config>.parquet next to the workload, one row per workload query, with `runtime_sec`).

Queries are executed sequentially on a single keep-alive HTTP connection, results are discarded on the
server side (FORMAT Null), mirroring the DuckDB executor of Redbench-Eval which uses EXPLAIN ANALYZE.

Each run starts from a freshly loaded database, because workloads contain DML.

Configurations:
  baseline   - no query cache, no query condition cache
  predcache  - query condition cache (predicate cache) on
  rescache   - query cache (result cache) on, with invalidation on writes emulated via cache tags
  allcaches  - both
"""

import argparse
import hashlib
import http.client
import json
import os
import re
import time
import urllib.parse
import uuid

import pandas as pd
from tqdm import tqdm

from translate import split_statements, translate_statement, write_table_of

CONFIGS = {
    "baseline": {"use_query_cache": 0, "use_query_condition_cache": 0},
    "predcache": {"use_query_cache": 0, "use_query_condition_cache": 1},
    "rescache": {"use_query_cache": 1, "use_query_condition_cache": 0},
    "allcaches": {"use_query_cache": 1, "use_query_condition_cache": 1},
}

# Run-name suffixes understood by Redbench-Eval's extract_info_from_run_name.
RUN_NAME = {
    "baseline": "clickhouse_baseline",
    "predcache": "clickhouse_predcache",
    "rescache": "clickhouse_rescache",
    "allcaches": "clickhouse_predcache_rescache",
}

QUERY_CACHE_SETTINGS = {
    # Emulate a result cache that keeps entries until invalidated (like Redshift's).
    "query_cache_ttl": 1_000_000_000,
    "query_cache_min_query_runs": 0,
    "query_cache_min_query_duration": 0,
}


class CH:
    def __init__(self, host, port, database):
        self.host, self.port, self.database = host, port, database
        self.conn = http.client.HTTPConnection(host, port, timeout=3600)

    def execute(self, sql, settings=None, query_id=None, database=True):
        params = dict(settings or {})
        if database:
            params["database"] = self.database
        if query_id:
            params["query_id"] = query_id
        url = "/?" + urllib.parse.urlencode(params)
        self.conn.request("POST", url, body=sql.encode())
        resp = self.conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"ClickHouse error {resp.status}: {body.decode(errors='replace')[:2000]}\nQuery: {sql[:2000]}")
        return body.decode()


def load_database(ch: CH, parquet_dir: str, user_files_rel: str):
    """Recreate the database from Parquet files exported from the DuckDB database used by Redbench."""
    ch.execute(f"DROP DATABASE IF EXISTS {ch.database} SYNC", database=False)
    ch.execute(f"CREATE DATABASE {ch.database}", database=False)
    for fn in sorted(os.listdir(parquet_dir)):
        if not fn.endswith(".parquet"):
            continue
        table = fn[: -len(".parquet")]
        path = f"{user_files_rel}/{fn}"
        desc = ch.execute(f"DESCRIBE file('{path}', Parquet) FORMAT JSONEachRow")
        cols = []
        for line in desc.splitlines():
            c = json.loads(line)
            t = c["type"]
            base = t[len("Nullable("):-1] if t.startswith("Nullable(") else t
            base = {"Int32": "Int32", "Int64": "Int64", "String": "String"}.get(base)
            if base is None:
                raise RuntimeError(f"Unexpected type {t} in {fn}")
            # Primary key columns are non-nullable; all other columns keep SQL NULL semantics.
            cols.append(f'"{c["name"]}" {base if c["name"] == "id" else f"Nullable({base})"}')
        ch.execute(
            f'CREATE TABLE "{table}" ({", ".join(cols)}) ENGINE = MergeTree ORDER BY "id" '
            f"SETTINGS enable_block_number_column = 1, enable_block_offset_column = 1"
        )
        ch.execute(f"INSERT INTO \"{table}\" SELECT * FROM file('{path}', Parquet)")


def run(args):
    ch = CH(args.host, args.port, args.database)
    df = pd.read_csv(args.workload)
    if args.limit:
        df = df.head(args.limit)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.workload))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"run_{RUN_NAME[args.config]}.parquet")

    t0 = time.time()
    load_database(ch, args.parquet_dir, args.user_files_rel)
    print(f"Loaded database in {time.time() - t0:.1f}s", flush=True)
    for q in ("SYSTEM DROP QUERY CACHE", "SYSTEM DROP QUERY CONDITION CACHE", "SYSTEM DROP MARK CACHE", "SYSTEM DROP UNCOMPRESSED CACHE"):
        ch.execute(q, database=False)

    settings = dict(CONFIGS[args.config])
    use_rescache = settings["use_query_cache"] == 1
    if use_rescache:
        settings.update(QUERY_CACHE_SETTINGS)
    settings["max_execution_time"] = args.timeout
    settings["lightweight_delete_mode"] = args.lightweight_delete_mode
    # Workload DML is replayed sequentially; batching single-row inserts would only add flush latency.
    settings["async_insert"] = 0

    run_id = uuid.uuid4().hex[:8]
    tables_of_select = {}  # translated sql -> list of tables read
    readers = {}  # table -> set of cache tags of cached SELECTs reading it
    results = []
    invalidations = 0

    for idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), mininterval=10)):
        sql = row.sql
        stmts = [translate_statement(s) for s in split_statements(sql)]
        stmts = [s for part in stmts for s in part]
        runtime = 0.0
        query_ids = []
        read_tables, write_tables = [], []
        error = None
        for s in stmts:
            is_select = s.lstrip().upper().startswith("SELECT")
            qid = f"rb_{run_id}_{idx}_{len(query_ids)}"
            stmt_settings = dict(settings)
            if is_select:
                body = s.rstrip().rstrip(";") + " FORMAT Null"
                if s not in tables_of_select:
                    # Untimed: the set of tables the query reads, for result-cache invalidation and staleness analysis.
                    tree = ch.execute(f"EXPLAIN QUERY TREE {s.rstrip().rstrip(';')}")
                    tables_of_select[s] = sorted({m.split(".", 1)[1] for m in re.findall(r"table_name: ([^,\s]+)", tree)})
                    if not tables_of_select[s]:
                        raise RuntimeError(f"No tables found in query tree of: {s[:500]}")
                read_tables += tables_of_select[s]
                if use_rescache:
                    tag = hashlib.sha1(s.encode()).hexdigest()[:16]
                    stmt_settings["query_cache_tag"] = tag
                    for t in tables_of_select[s]:
                        readers.setdefault(t, set()).add(tag)
            else:
                body = s
                stmt_settings.pop("use_query_cache", None)
                write_tables.append(write_table_of(s))
            start = time.perf_counter()
            try:
                ch.execute(body, stmt_settings, query_id=qid)
            except RuntimeError as e:
                if "TIMEOUT_EXCEEDED" in str(e):
                    error = "timeout"
                else:
                    raise
            runtime += time.perf_counter() - start
            query_ids.append(qid)
            if error:
                break
            if not is_select and use_rescache:
                # Result-cache invalidation (not part of the measured runtime).
                wt = write_table_of(s)
                for tag in readers.pop(wt, ()):
                    ch.execute(f"SYSTEM DROP QUERY CACHE TAG '{tag}'", database=False)
                    invalidations += 1

        results.append({
            "query_idx": idx,
            "query": " ; ".join(stmts),
            "runtime_sec": runtime,
            "error": error,
            "ch_query_ids": query_ids,
            "ch_read_tables": sorted(set(read_tables)),
            "ch_write_tables": sorted(set(write_tables)),
            **row._asdict(),
        })
        if (idx + 1) % 2000 == 0:
            pd.DataFrame(results).to_parquet(out_file, index=False)

    # Enrich with server-side metrics.
    ch.execute("SYSTEM FLUSH LOGS", database=False)
    ql = ch.execute(
        f"""SELECT query_id, query_duration_ms, read_rows, read_bytes, result_rows, memory_usage,
                   ProfileEvents['QueryCacheHits'] AS qc_hits, ProfileEvents['QueryConditionCacheHits'] AS qcc_hits,
                   ProfileEvents['UserTimeMicroseconds'] AS user_us
            FROM system.query_log WHERE type = 'QueryFinish' AND startsWith(query_id, 'rb_{run_id}_') FORMAT JSONEachRow""",
        database=False,
    )
    qlog = {}
    for line in ql.splitlines():
        r = json.loads(line)
        qlog[r.pop("query_id")] = r
    for r in results:
        agg = {}
        for qid in r["ch_query_ids"]:
            for k, v in qlog.get(qid, {}).items():
                agg[k] = agg.get(k, 0) + int(v)
        for k in ("query_duration_ms", "read_rows", "read_bytes", "result_rows", "memory_usage", "qc_hits", "qcc_hits", "user_us"):
            r[k] = agg.get(k)
    out = pd.DataFrame(results)
    out.to_parquet(out_file, index=False)
    print(f"Wrote {out_file}: {len(out)} queries, total runtime {out['runtime_sec'].sum():.1f}s, "
          f"timeouts {int((out['error'] == 'timeout').sum())}, cache invalidations {invalidations}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True, help="Path to workload.csv")
    p.add_argument("--parquet_dir", required=True, help="Directory with <table>.parquet (inside user_files)")
    p.add_argument("--user_files_rel", required=True, help="Path of parquet_dir relative to the server user_files_path")
    p.add_argument("--config", choices=list(CONFIGS), required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=38123)
    p.add_argument("--database", default="redbench")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--lightweight_delete_mode", default="lightweight_update_force")
    run(p.parse_args())
