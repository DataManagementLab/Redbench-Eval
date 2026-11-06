# ClickHouse executor for Redbench workloads

Runs Redbench workloads (`workload.csv`) on a ClickHouse server and writes `run_clickhouse_<config>.parquet`
files in the layout used by Redbench-Eval (`runtime_sec` per workload query, plus all workload columns),
so the existing plotting notebooks pick them up as system `Clickhouse`.

## Files

- `translate.py`: translates the PostgreSQL/DuckDB dialect emitted by Redbench into ClickHouse SQL.
  - `DELETE ... USING` and `UPDATE ... FROM` become `... WHERE id IN (SELECT ...)`.
  - `INSERT ... ON CONFLICT (id) DO NOTHING` becomes an anti-join (`AND <key> NOT IN (SELECT id FROM target)`).
  - CTAS with `ADD PRIMARY KEY` becomes `CREATE TABLE ... ENGINE = MergeTree ORDER BY id AS SELECT`.
  - `~` becomes `match`.
  - Backslashes in literals are escaped.
  - Multi-statement entries are split.
- `execute_queries_clickhouse.py`: loads a fresh database from Parquet files exported from the Redbench DuckDB
  database, then replays the workload sequentially on one keep-alive HTTP connection. Results are discarded on the
  server side (`FORMAT Null`), similar to the DuckDB executor, which uses `EXPLAIN ANALYZE`. Server-side metrics
  are collected from `system.query_log`.
- `validate_against_duckdb.py`: replays a workload prefix on DuckDB (original SQL) and on ClickHouse (translated SQL)
  from the same database, then compares the result row counts of every `SELECT` and the final state of every table.
- `analyze.py`: computes the speedup of every configuration over the baseline. It also estimates how many hits a
  result cache that is never invalidated on writes would serve stale.
- `run_paper_workloads.sh`: runs the three paper workloads in all configurations.

## Configurations

| run name | settings |
| --- | --- |
| `clickhouse_baseline` | `use_query_cache = 0`, `use_query_condition_cache = 0` |
| `clickhouse_predcache` | query condition cache (a predicate cache) on |
| `clickhouse_rescache` | query cache (result cache) on, never expires, invalidated on writes |
| `clickhouse_predcache_rescache` | both |

The ClickHouse query cache does not invalidate entries when the underlying tables change; entries live until
`query_cache_ttl`. To measure a result cache with the semantics of the systems in the paper, every cached `SELECT`
is tagged (`query_cache_tag`) and the tags of all cached queries reading a table are dropped
(`SYSTEM DROP QUERY CACHE TAG`) after each write to it. Invalidation time is not included in the measured runtime.

All configurations use `lightweight_delete_mode = 'lightweight_update_force'` and `async_insert = 0`. With the defaults
(`alter_update` mutations for lightweight deletes, asynchronous inserts that wait for a flush), the sequential
single-row DML of the matching workload is 28x slower (453.8 s vs 16.0 s for the first 1,000 queries). Tables use
`enable_block_number_column = 1, enable_block_offset_column = 1`, which lightweight updates require.

## Running

```bash
# Redbench: produces output/paper_workloads/* and output/tmp_generation/imdb/{db_original,db_augmented_x2}.duckdb
python src/redbench/run.py --redset_path data/full_serverless.parquet --output_dir output

# Export the DuckDB databases to Parquet under the server's user_files directory
python -c "import duckdb; c = duckdb.connect('output/tmp_generation/imdb/db_original.duckdb', read_only=True); \
  [c.execute(f\"COPY {t} TO 'user_files/imdb_original/{t}.parquet' (FORMAT parquet)\") for (t,) in c.execute('SELECT table_name FROM duckdb_tables()').fetchall()]"

./run_paper_workloads.sh <redbench_root>
python analyze.py <redbench_root>/results/paper
```

The server needs `query_log` enabled and a large `keep_alive_timeout` (validation keeps the connection idle while
DuckDB runs).
