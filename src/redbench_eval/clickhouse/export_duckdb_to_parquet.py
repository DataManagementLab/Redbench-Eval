"""
Export every table of a Redbench DuckDB database to <out_dir>/<table>.parquet, for loading into ClickHouse.
<out_dir> must be inside the ClickHouse server's `user_files_path`, so that the server can read it with `file`.

Usage: export_duckdb_to_parquet.py <db_file> <out_dir>
  e.g. export_duckdb_to_parquet.py ../Redbench/output/tmp_generation/imdb/db_original.duckdb /var/lib/clickhouse/user_files/imdb_original
"""

import os
import sys

import duckdb


def main(db_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    conn = duckdb.connect(db_file, read_only=True)
    conn.execute("SET enable_progress_bar = false")
    tables = [t for (t,) in conn.execute("SELECT table_name FROM duckdb_tables() ORDER BY table_name").fetchall()]
    for t in tables:
        conn.execute(f"COPY \"{t}\" TO '{os.path.join(out_dir, t)}.parquet' (FORMAT parquet, COMPRESSION zstd)")
    print(f"Exported {len(tables)} tables to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
