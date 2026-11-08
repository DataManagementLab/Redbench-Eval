"""
Semantic check of the PostgreSQL -> ClickHouse translation: replay the first N queries of a workload on DuckDB
(original SQL, as in Redbench-Eval) and on ClickHouse (translated SQL), both starting from the same database,
compare the number of result rows of every SELECT and, at the end, row count and sum of ids of every table.
"""

import argparse
import json
import re
import shutil

import duckdb
import pandas as pd

from execute_queries_clickhouse import CH, load_database
from translate import split_statements, translate_statement

REGEX = r"([^\s\(]+)\s*~\s*'([^']*)'"  # Redbench-Eval get_extract_col_and_regex_regex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True)
    p.add_argument("--duckdb_file", required=True)
    p.add_argument("--duckdb_copy", required=True)
    p.add_argument("--parquet_dir", required=True)
    p.add_argument("--user_files_rel", required=True)
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--database", default="redbench_validate")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port")
    a = p.parse_args()

    df = pd.read_csv(a.workload).head(a.limit)
    shutil.copyfile(a.duckdb_file, a.duckdb_copy)
    dk = duckdb.connect(a.duckdb_copy)
    dk.execute("SET enable_progress_bar = false")
    ch = CH(a.host, a.port, a.database)
    load_database(ch, a.parquet_dir, a.user_files_rel)
    settings = {"lightweight_delete_mode": "lightweight_update_force", "async_insert": 0, "use_query_cache": 0}

    mismatches = 0
    selects = 0
    for idx, row in enumerate(df.itertuples(index=False)):
        sql = row.sql
        if " ~ " in sql:
            sql_dk = re.sub(REGEX, r"regexp_full_match(\1, '\2')", sql)
        else:
            sql_dk = sql
        is_select = sql.lstrip().upper().startswith("SELECT")
        if is_select:
            selects += 1
            n_dk = dk.execute(f"SELECT count(*) FROM ({sql_dk.strip().rstrip(';')})").fetchone()[0]
            (s,) = [t for st in split_statements(sql) for t in translate_statement(st)]
            n_ch = int(ch.execute(f"SELECT count() FROM ({s.rstrip(';')})", settings).strip())
            if n_dk != n_ch:
                mismatches += 1
                if mismatches <= 5:
                    print(f"MISMATCH #{idx}: duckdb={n_dk} clickhouse={n_ch}\n  {sql[:400]}", flush=True)
        else:
            dk.execute(sql_dk)
            for st in split_statements(sql):
                for t in translate_statement(st):
                    ch.execute(t, settings)

    table_mismatch = 0
    for (t,) in dk.execute("SELECT table_name FROM duckdb_tables() ORDER BY 1").fetchall():
        d = dk.execute(f'SELECT count(*), coalesce(sum(id), 0) FROM "{t}"').fetchone()
        c = json.loads(ch.execute(f'SELECT count() AS n, sum(id) AS s FROM "{t}" FORMAT JSONEachRow', settings))
        if (int(d[0]), int(d[1])) != (int(c["n"]), int(c["s"])):
            table_mismatch += 1
            print(f"TABLE MISMATCH {t}: duckdb={d} clickhouse={c}", flush=True)
    print(f"{a.workload}: {len(df)} queries, {selects} SELECTs, {mismatches} result-count mismatches, "
          f"{table_mismatch} table-state mismatches", flush=True)


if __name__ == "__main__":
    main()
