import argparse
import json
import os
import re
import sys
import time
import tracemalloc
from typing import Dict, List

import duckdb
import pandas as pd
import psutil
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import get_extract_col_and_regex_regex


def run_benchmark(
    result_dir: str,
    dataset: str,
    redset_dataset: str,
    exp_hash: str,
    cluster_id: int,
    database_id: int,
    strategy: str,
    db_file: str,
    copy_csv_files_dir: str,
) -> List[Dict]:
    """
    Runs each query in the workload_file against the DuckDB database and logs runtime, peak memory, cpu-time, and EXPLAIN ANALYZE plan.
    Returns a list of dicts with the results for each query.
    """

    exp_dir = os.path.join(
        result_dir,
        "generated_workloads",
        dataset,
        redset_dataset,
        f"cluster_{cluster_id}",
        f"database_{database_id}",
        f"{strategy}_{exp_hash}",
    )

    assert os.path.isdir(exp_dir), (
        f"exp_dir {exp_dir} must be a directory containing the workload CSV file."
    )
    workload_file = os.path.join(exp_dir, "workload.csv")
    out_file = os.path.join(exp_dir, "run_duckdb.parquet")

    if os.path.exists(out_file):
        print(f"Output file {out_file} already exists. Skipping execution.", flush=True)
        return

    if not workload_file.endswith(".csv"):
        raise ValueError(
            "workload_file must be a .csv file containing a 'query' column."
        )
    df = pd.read_csv(workload_file)
    if "sql" not in df.columns:
        raise ValueError(f"CSV file must contain a 'sql' column. Found:{df.columns}")

    # check that db_file exists
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"Database file {db_file} does not exist.")

    # extract dirname from wl_file
    exp_name = f"{dataset}_cluster{cluster_id}_db{database_id}_{exp_hash}"

    # get the path to the directory of the DuckDB database file
    db_dir = os.path.dirname(os.path.abspath(db_file))
    db_filename = os.path.basename(db_file)
    assert db_filename.endswith(".duckdb"), "db_file must be a .duckdb file."
    new_db_filename = db_filename.replace(".duckdb", f"_{exp_name}.duckdb")

    # create a copy of the db file to avoid modifying the original
    # new_db_file = os.path.join(db_dir, new_db_filename)
    new_db_file = new_db_filename  # have on local disk to ensure faster access

    # copy the DuckDB database file to the new location
    import shutil

    print(f"Copying DuckDB database file from {db_file} to {new_db_file}", flush=True)
    shutil.copyfile(db_file, new_db_file)

    results = []
    conn = duckdb.connect(database=new_db_file, read_only=False)
    process = psutil.Process(os.getpid())

    for idx, row in enumerate(
        tqdm(df.itertuples(index=False), desc="Running queries", total=len(df))
    ):
        query = getattr(row, "sql")
        # rewrite query to csv dir
        if "<<csv_path_placeholder>>" in query:
            if not copy_csv_files_dir:
                raise ValueError(
                    "copy_csv_files_dir must be provided when using <<csv_path_placeholder>> in queries."
                )
            query = query.replace("<<csv_path_placeholder>>", copy_csv_files_dir)

        # Measure memory and cpu
        tracemalloc.start()
        cpu_start = process.cpu_times().user
        start_time = time.perf_counter()
        memory_before = process.memory_info().rss

        # Polling for psutil peak memory
        poll_interval = 0.01  # 10ms
        peak_rss = process.memory_info().rss
        query_running = True
        import threading

        def poll_memory():
            nonlocal peak_rss, query_running
            while query_running:
                rss = process.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss
                time.sleep(poll_interval)

        poll_thread = threading.Thread(target=poll_memory)
        poll_thread.start()

        def cleanup_query(query: str) -> str:
            if " ~ " in query:
                # Duckdb does not support ~ operator - convert to REGEXP_LIKE
                query = re.sub(
                    get_extract_col_and_regex_regex(),
                    r"regexp_full_match(\1, '\2')",
                    query,
                )

            return query

        query = cleanup_query(query)

        def create_explain_query(query: str) -> str:
            exp_an = "EXPLAIN (ANALYZE, FORMAT json)"
            if (
                strategy in ["matching", "baseline_round_robin"]
                and query.startswith("DELETE ")
                and "INSERT INTO " in query
            ):
                # this is a delete+insert query, we need to split it
                delete_query, insert_query = query.split("INSERT INTO ", 1)

                # add explain analyze
                explain_query = (
                    f"{exp_an} {delete_query} {exp_an} INSERT INTO {insert_query}"
                )
            else:
                explain_query = f"{exp_an} {query}"
            return explain_query

        # Run query
        try:
            explain_plan = conn.execute(create_explain_query(query)).fetchall()
        except Exception as e:
            # stop memory monitoring if query fails
            print(query, flush=True)
            query_running = False
            poll_thread.join()
            raise e

        # collect measurements
        end_time = time.perf_counter()
        cpu_end = process.cpu_times().user
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_after = process.memory_info().rss

        # stop memory monitoring
        query_running = False
        poll_thread.join()

        assert len(explain_plan) > 0 and len(explain_plan[0]) > 1, (
            f"Unexpected EXPLAIN output: {explain_plan} / {query}"
        )

        explain_plan = json.loads(explain_plan[0][1])

        # convert dataframe row to dict
        redset_row_dict = {col: getattr(row, col) for col in df.columns}

        results.append(
            {
                "query_idx": idx,
                "query": query,
                "runtime_sec": end_time - start_time,
                "peak_memory_bytes_tracemalloc": peak,
                "peak_memory_bytes_psutil": peak_rss,
                "memory_before_bytes": memory_before,
                "memory_after_bytes": memory_after,
                "cpu_time_sec": cpu_end - cpu_start,
                "explain_analyze": explain_plan,
                **redset_row_dict,
            }
        )

        # save after every 500 queries
        if (idx + 1) % 500 == 0:
            save_results(results, out_file)

    conn.close()

    save_results(results, out_file)

    # drop database - cleanup the copied file
    print(f"Removing copied DuckDB database file {new_db_file}", flush=True)
    os.remove(new_db_file)

    return results


def save_results(results: List[Dict], out_file: str):
    if out_file.endswith(".json"):
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
    elif out_file.endswith(".parquet"):
        df = pd.DataFrame(results)

        # apply json serialization to explain_analyze column
        if "explain_analyze" in df.columns:
            df["explain_analyze"] = df["explain_analyze"].apply(json.dumps)

        df.to_parquet(out_file, index=False)
    else:
        raise ValueError("Output file must be either .json or .parquet format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SQL benchmark on DuckDB and save results."
    )
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--redset_dataset", type=str, required=True)
    parser.add_argument("--exp_hash", type=str, required=True)
    parser.add_argument("--cluster_id", type=int, required=True)
    parser.add_argument("--database_id", type=int, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument(
        "--db_file", required=True, help="Path to DuckDB database file."
    )

    parser.add_argument(
        "--copy_csv_files_dir",
        required=False,
        default="",
        help="Directory for CSV files (needed to support copy queries).",
    )

    args = parser.parse_args()

    print(f"Running DuckDB benchmark with args: {args}", flush=True)
    run_benchmark(
        args.result_dir,
        args.dataset,
        args.redset_dataset,
        args.exp_hash,
        args.cluster_id,
        args.database_id,
        args.strategy,
        args.db_file,
        args.copy_csv_files_dir,
    )
