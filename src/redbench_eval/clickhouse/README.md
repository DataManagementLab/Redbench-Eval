# ClickHouse executor for Redbench workloads

Runs Redbench workloads (`workload.csv`) on a ClickHouse server. It writes `run_clickhouse_<config>.parquet` files in
the layout the plotting notebooks of this repository expect: one row per workload query, with `runtime_sec` and all
workload columns. The notebooks pick them up as system `Clickhouse`.

## Files

| File | Purpose |
| --- | --- |
| `translate.py` | Translates the PostgreSQL/DuckDB dialect emitted by Redbench into ClickHouse SQL |
| `execute_queries_clickhouse.py` | Loads a fresh database from Parquet and replays a workload sequentially |
| `validate_against_duckdb.py` | Replays a workload prefix on DuckDB and on ClickHouse, then compares results and table state |
| `check_translation.py` | Checks that every translated statement of a workload parses |
| `analyze.py` | Speedup of every configuration over the baseline; staleness of a cache without invalidation |
| `export_duckdb_to_parquet.py` | Exports a Redbench DuckDB database to Parquet files for loading into ClickHouse |
| `run_paper_workloads.sh` | Runs the three paper workloads in all configurations |
| `clickhouse_server_config.xml` | Server settings (query cache size, keep-alive timeout) |

## SQL translation

| Redbench (PostgreSQL / DuckDB) | ClickHouse |
| --- | --- |
| `DELETE FROM t USING a, b WHERE c` | `DELETE FROM t WHERE id IN (SELECT t.id FROM t, a, b WHERE c)` |
| `UPDATE t SET x = v FROM a, b WHERE c` | `UPDATE t SET x = v WHERE id IN (SELECT t.id FROM t, a, b WHERE c)` |
| `INSERT INTO t (...) SELECT ... ON CONFLICT (id) DO NOTHING` | `INSERT INTO t (...) SELECT ... WHERE ... AND <id expr> NOT IN (SELECT id FROM t)` |
| `CREATE TABLE t AS SELECT ...; ALTER TABLE t ADD PRIMARY KEY (id)` | `CREATE TABLE t ENGINE = MergeTree ORDER BY id AS SELECT ...` |
| `col ~ 'regex'` | `match(col, 'regex')` |
| `'a\b'` (backslash is literal) | `'a\\b'` (backslash is an escape character) |
| `DELETE ...; INSERT ...;` in one workload entry | executed as separate statements, runtimes summed |

`SELECT DISTINCT ON`, double-quoted identifiers, and comparisons of integer columns with string literals work as is.
Tables are created as `MergeTree ORDER BY id`: the primary key `id` is non-nullable, all other columns are
`Nullable`, to keep SQL `NULL` semantics.

All 51,016 distinct statements of the three paper workloads parse. `validate_against_duckdb.py` replayed the first
3,000 queries of each workload on DuckDB, running the original SQL, and on ClickHouse, running the translated SQL,
from the same database. It compared the result row count of every `SELECT` and the row count and sum of ids of every
table at the end. The result was 0 mismatches for all three workloads.

## Configurations

| Run name | Settings |
| --- | --- |
| `clickhouse_baseline` | `use_query_cache = 0`, `use_query_condition_cache = 0` |
| `clickhouse_predcache` | query condition cache (predicate cache) on |
| `clickhouse_rescache` | query cache (result cache) on, never expires, invalidated on writes |
| `clickhouse_predcache_rescache` | both |

The ClickHouse query cache does not invalidate entries when the underlying tables change; entries live until
`query_cache_ttl`. To measure a result cache with the semantics of the systems in the paper, every cached `SELECT`
is tagged (`query_cache_tag`) with a hash of its text. After each write to a table, the tags of all cached queries
that read the table are dropped (`SYSTEM DROP QUERY CACHE TAG`). The tables a query reads come from
`EXPLAIN QUERY TREE`. Neither the invalidation nor the table lookup is included in the measured runtime.

Queries run sequentially on one keep-alive HTTP connection. Results are discarded on the server side
(`FORMAT Null`), as the DuckDB executor does with `EXPLAIN ANALYZE`. Each run starts from a freshly loaded database,
because the workloads contain DML. Server-side metrics (CPU time, rows read, cache hits) are added from
`system.query_log`.

### Non-default settings

- `lightweight_delete_mode = 'lightweight_update_force'`. The default `alter_update` runs every lightweight `DELETE`
  as a synchronous mutation: 60-400 ms per single-row delete, versus about 4 ms with patch parts.
- `async_insert = 0`. With asynchronous inserts (the default since ClickHouse 26.2), a client that waits for each
  insert pays at least `async_insert_busy_timeout_min_ms` (50 ms) per insert whenever inserts are less than 200 ms
  apart, and gains nothing from batching
  ([ClickHouse#129734](https://github.com/ClickHouse/ClickHouse/issues/129734)).
  Together with the previous setting, this takes the first 1,000 queries of the matching workload from 453.8 s to 16.0 s.
- `query_cache_compress_entries = 0`. With compression, every hit decompresses the full result: 0.056 s versus 0.014 s
  for a 3.5M-row result.
- A 32 GiB query cache, see `clickhouse_server_config.xml`.
- Tables use `enable_block_number_column = 1, enable_block_offset_column = 1`, which lightweight updates require.

## Running

Requires ClickHouse 25.8 or newer, where lightweight `UPDATE` is enabled by default (`enable_lightweight_update`). The commands below assume the default `user_files_path`
`/var/lib/clickhouse/user_files` and Redbench checked out next to this repository.

```bash
# 1. Server settings
sudo cp src/redbench_eval/clickhouse/clickhouse_server_config.xml /etc/clickhouse-server/config.d/redbench.xml
sudo systemctl restart clickhouse-server

# 2. Redbench: downloads the paper workloads (output/paper_workloads/) and the IMDb databases from OSF
(cd ../Redbench && python src/redbench/run.py --redset_path data/full_serverless.parquet --output_dir output)

# 3. Export the IMDb databases to Parquet, where the server can read them
U=/var/lib/clickhouse/user_files
python src/redbench_eval/clickhouse/export_duckdb_to_parquet.py ../Redbench/output/tmp_generation/imdb/db_original.duckdb $U/imdb_original
python src/redbench_eval/clickhouse/export_duckdb_to_parquet.py ../Redbench/output/tmp_generation/imdb/db_augmented_x2.duckdb $U/imdb_augmented_x2

# 4. Run all workloads and configurations, then summarize
src/redbench_eval/clickhouse/run_paper_workloads.sh ../Redbench/output/paper_workloads $U results
python src/redbench_eval/clickhouse/analyze.py results
```

To run a single workload:

```bash
python src/redbench_eval/clickhouse/execute_queries_clickhouse.py \
  --workload ../Redbench/output/paper_workloads/matching/workload.csv \
  --parquet_dir /var/lib/clickhouse/user_files/imdb_original --user_files_rel imdb_original \
  --config rescache --out_dir results/matching
```

Use `--limit N` to run only the first `N` queries.

## Results

ClickHouse 26.11 (aarch64, 96 cores), the three paper workloads (Redset provisioned, instance 186, database 8,
May 2024; 36,353 queries each), all queries completed with no timeouts. Speedups are relative to `clickhouse_baseline`.
CPU is server CPU time.

| Workload | Configuration | Result cache hits | CPU speedup | Wall-clock speedup | Paper, Systems A-D |
| --- | --- | ---: | ---: | ---: | ---: |
| round-robin | result cache | 0 | 0.99x | within noise | <= 1.08x |
| generation | result cache | 18,533 | 1.92x | 1.53x | 1.10-2.00x |
| matching | result cache | 24,159 | 23.7x | 9.2x | 2.12-2.98x |
| any | query condition cache | - | 1.00-1.02x | within noise | |

- Wall-clock speedups are corrected for host load: only the cache hits' runtimes are replaced in the baseline.
  The host was shared, and a repeated baseline run differed by up to 25% in total wall time but by 4% in CPU time.
- Matching: DML is only 5% of runtime, 97% of `SELECT`s hit the cache, and a hit takes about 1.3 ms instead of 264 ms.
- Round-robin: every repeated `SELECT` is preceded by a write to a table it reads, so a correct result cache gets
  no hits.
- The query condition cache is disabled for tables with patch parts. The point `DELETE`s of the matching and
  round-robin workloads leave patch parts on almost every table, so they get no condition cache hits.
  Generation gets hits, but its range predicates rarely exclude whole granules.
- A result cache without invalidation would serve stale results for 32% (generation), 92% (matching) and
  100% (round-robin) of its hits.
