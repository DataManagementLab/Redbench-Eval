import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from redbench.utils.load_and_preprocess_redset import load_and_preprocess_redset

from redbench_eval.plots.run_baseline_rewriter import (
    create_other_baseline_versions,
    create_redset_baseline_version,
)


def assemble_cache_key(path_kwargs: Dict) -> str:
    redset_abbr = {
        "provisioned": "prov",
        "serverless": "serv",
    }

    cluster_id_str = path_kwargs["cluster_id"].replace("cluster_", "")
    db_id_str = path_kwargs["database_id"].replace("database_", "")

    workload = path_kwargs["workload"]
    if "baseline_round_robin" in workload:
        workload = workload.replace("baseline_round_robin", "rr")
    elif "generation" in workload:
        workload = workload.replace("generation", "gen")
    elif "matching" in workload:
        workload = workload.replace("matching", "match")

    return f"{path_kwargs['dataset']}_{redset_abbr.get(path_kwargs['redset_dataset'], path_kwargs['redset_dataset'])}_{cluster_id_str}_{db_id_str}_{workload}"


def load_redset_baseline(
    artifacts_dir: str, path_kwargs: Dict, overwrite_redset_path: str
) -> pd.DataFrame:
    key = assemble_cache_key(path_kwargs)
    redset_cache_path = os.path.join(
        os.path.dirname(__file__), "cache", f"{key}_redset.pkl"
    )

    # create dir if not exists
    os.makedirs(os.path.dirname(redset_cache_path), exist_ok=True)

    if os.path.exists(redset_cache_path):
        df = pickle.load(open(redset_cache_path, "rb"))
        return df

    # load generated workload
    config_path = artifacts_dir + "used_config.json"

    # load config and extract parameters
    with open(config_path, "r") as f:
        config = json.load(f)

    if overwrite_redset_path is not None:
        redset_dataset_path = overwrite_redset_path
    else:
        redset_dataset_path = config["redset_path"]

    con = load_and_preprocess_redset(
        start_date=config["start_date"],
        end_date=config["end_date"],
        database_id=config["database_id"],
        instance_id=config["cluster_id"],
        redset_path=redset_dataset_path,
        include_copy=config.get("include_copy", False),
        include_analyze=config.get("include_analyze", False),
        include_ctas=config.get("include_ctas", False),
        exclude_tables_never_read=config.get("redset_exclude_tables_never_read", False),
    )

    redset_entries = con.execute("SELECT * FROM redset_preprocessed;").df()

    print(f"Num entries retrieved from redset: {len(redset_entries)}")

    # Ensure arrival_timestamp is datetime
    redset_entries = ensure_datetime(redset_entries)

    # annotate additional columns
    annotate_redset_stats(redset_entries)

    # write to cache
    pickle.dump(redset_entries, open(redset_cache_path, "wb"))

    return redset_entries


def annotate_redset_stats(df: pd.DataFrame):
    # available args: ['instance_id', 'cluster_size', 'user_id', 'database_id', 'query_id','arrival_timestamp', 'compile_duration_ms', 'queue_duration_ms',   'execution_duration_ms', 'feature_fingerprint', 'was_aborted','was_cached', 'cache_source_query_id', 'query_type', 'num_permanent_tables_accessed', 'num_external_tables_accessed', 'num_system_tables_accessed', 'read_table_ids', 'write_table_ids',       'mbytes_scanned', 'mbytes_spilled', 'num_joins', 'num_scans', 'num_aggregations', 'query_hash']
    df["runtime_sec"] = df["execution_duration_ms"] / 1000
    df["bytes_read"] = df["mbytes_scanned"] * 1e6
    df["hit_result_cache"] = df["was_cached"].apply(lambda x: x == 1)


def q_error(est: float, act: float):
    return max(est / act, act / est)


# annotate num rows read (i.e. output cardinality of all scan nodes)
def annotate_duckdb_stats(measurement_results: pd.DataFrame):
    if "explain_analyze" not in measurement_results.columns:
        print(
            "No 'explain_analyze' column found in measurement results. Skipping annotation of num rows read."
        )
        return

    dd_result_size_list = []
    dd_rows_scanned_list = []

    for _, row in measurement_results.iterrows():
        plan = row["explain_analyze"]

        # Parse the JSON plan if it's a string
        if isinstance(plan, str):
            plan = json.loads(plan)

        # Helper to recursively collect leaf nodes
        def get_leaves(node):
            if not isinstance(node, dict):
                return []
            if "children" in node and node["children"]:
                leaves = []
                for child in node["children"]:
                    leaves.extend(get_leaves(child))
                return leaves
            else:
                return [node]

        leaves = get_leaves(plan)
        sum_result_set_size = 0
        sum_rows_scanned = 0
        for leaf in leaves:
            sum_result_set_size += leaf["result_set_size"]
            sum_rows_scanned += leaf.get("operator_rows_scanned", 0)

        dd_result_size_list.append(sum_result_set_size)
        dd_rows_scanned_list.append(sum_rows_scanned)

    measurement_results["dd_scan_result_size"] = dd_result_size_list
    measurement_results["dd_rows_scanned"] = dd_rows_scanned_list
    measurement_results["bytes_read"] = dd_result_size_list

    # get cardinality estimation error
    def get_root_node_for_card_est(plan):
        if "rows_returned" in plan:
            assert len(plan["children"]) == 1
            return get_root_node_for_card_est(plan["children"][0])

        node_type = plan["operator_name"]

        if node_type in [
            "EXPLAIN_ANALYZE",
            "INSERT",
            "PROJECTION",
            "DELETE",
            "UPDATE",
            "UNGROUPED_AGGREGATE",
            "ORDER_BY",
            "MERGE_INTO",
        ]:
            # skip this node
            assert len(plan["children"]) == 1
            return get_root_node_for_card_est(plan["children"][0])

        return plan

    def get_card_est_error(row) -> float:
        query_type = row["query_type"]

        plan_str = row["explain_analyze"]
        plan = json.loads(plan_str) if isinstance(plan_str, str) else plan_str
        card_root_node = get_root_node_for_card_est(plan)

        assert "extra_info" in card_root_node
        extra_info = card_root_node["extra_info"]

        assert "Estimated Cardinality" in extra_info, json.dumps(
            card_root_node, indent=2
        )
        est_card = float(extra_info["Estimated Cardinality"])
        act_card = float(card_root_node["operator_cardinality"])

        # sanity checks
        operator_type = card_root_node["operator_name"]
        if act_card == 0:
            act_card = 1.0  # avoid division by zero - treat as if 1 row was returned
            # print(f"Warning: Actual cardinality is 0 for operator {operator_type} in query type {query_type}. Setting to 1.0 to avoid division by zero.")

        if est_card == 0:
            est_card = 1.0  # avoid division by zero - treat as if 1 row was estimated
            # print(f"Warning: Estimated cardinality is 0 for operator {operator_type} in query type {query_type}. Setting to 1.0 to avoid division by zero.")
        assert est_card > 0 and act_card > 0, json.dumps(card_root_node, indent=2)

        return q_error(est_card, act_card)

    measurement_results["root_card_error"] = measurement_results.apply(
        get_card_est_error, axis=1
    )


# Ensure arrival_timestamp is datetime
def ensure_datetime(df):
    if not pd.api.types.is_datetime64_any_dtype(df["arrival_timestamp"]):
        df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])
    return df


def load_measurement_run(
    filename: str, artifacts_dir: str, path_kwargs: Dict, skip_cache: bool = False
):
    cache_path = os.path.join(
        os.path.dirname(__file__),
        "cache",
        f"{assemble_cache_key(path_kwargs)}_{path_kwargs['key']}.pkl",
    )
    if os.path.exists(cache_path) and not skip_cache:
        df = pickle.load(open(cache_path, "rb"))
        return df

    measurement_results_path = os.path.join(artifacts_dir, filename)
    assert os.path.exists(measurement_results_path), (
        f"Measurement results file {measurement_results_path} does not exist."
    )

    measurement_results = pd.read_parquet(measurement_results_path)
    print(
        f"Number of measurement results ({filename}): {len(measurement_results)}",
        flush=True,
    )

    # annotate additional columns
    if "_duckdb" in filename:
        annotate_duckdb_stats(measurement_results)
    else:
        print(
            f"No annotation function found for {filename}. Skipping annotation of additional stats."
        )

    # ensure arrival_timestamp is datetime
    measurement_results = ensure_datetime(measurement_results)

    if "baseline_round_robin" not in measurement_results_path:
        # make sure query_hash is there
        if "query_hash" not in measurement_results.columns:
            assert "exact_repetition_hash" in measurement_results.columns, (
                f"Either query_hash or exact_repetition_hash must be present in measurement results {filename}. {measurement_results.columns}"
            )
            measurement_results["query_hash"] = measurement_results[
                "exact_repetition_hash"
            ]

    # write to cache
    pickle.dump(measurement_results, open(cache_path, "wb"))

    return measurement_results


def assemble_runs_dict(
    result_dir: str,
    dataset: str,
    redset_dataset: str,
    cluster_id: str,
    database_id: str,
    workloads: List[str],
    filter_string: str = None,
    overwrite_redset_path: str = None,
):
    assert os.path.isdir(result_dir)
    results = defaultdict(dict)

    for workload in workloads:
        artifacts_dir = f"{result_dir}/{dataset}/{redset_dataset}/{cluster_id}/{database_id}/{workload}/"
        if not os.path.exists(artifacts_dir):
            print(f"Artifacts directory {artifacts_dir} does not exist, skipping")
            continue

        # assemble key for cache lookup

        df = load_redset_baseline(
            artifacts_dir,
            path_kwargs={
                "dataset": dataset,
                "redset_dataset": redset_dataset,
                "cluster_id": cluster_id,
                "database_id": database_id,
                "workload": workload,
            },
            overwrite_redset_path=overwrite_redset_path,
        )
        results[workload]["redset"] = df

        # search for files with run_.*.parquet
        files = os.listdir(artifacts_dir)
        parquet_files = [
            f for f in files if f.startswith("run_") and f.endswith(".parquet")
        ]
        print(f"Found {len(parquet_files)} parquet files in {artifacts_dir}")
        for f in parquet_files:
            if filter_string is not None and filter_string not in f:
                continue

            # strip the run_ prefix and .parquet suffix
            key = f[len("run_") : -len(".parquet")]

            df = load_measurement_run(
                f,
                artifacts_dir,
                path_kwargs={
                    "dataset": dataset,
                    "redset_dataset": redset_dataset,
                    "cluster_id": cluster_id,
                    "database_id": database_id,
                    "workload": workload,
                    "key": key,
                },
            )
            results[workload][key] = df

    return results


def filter_groupby_extract(
    df: pd.DataFrame,
    query_type: str,
    target_metric: str,
    start_ts: pd.Timestamp = None,
    end_ts: pd.Timestamp = None,
    return_cummulated: bool = False,
    return_count: bool = False,
    return_avg: bool = False,
    return_median: bool = False,
    return_max: bool = False,
    return_all_buckets: bool = True,
    aggregation_freq: str = "3h",
):
    # filter by query type if specified
    if query_type == "all-no-update":
        filtered = df[df["query_type"] != "update"]
    elif query_type == "read-only":
        filtered = df[df["query_type"].isin(["select", "analyze", "ctas"])]
    elif query_type == "writes":
        filtered = df[df["query_type"].isin(["insert", "update", "copy", "delete"])]
    elif query_type != "all":
        filtered = df[df["query_type"] == query_type]
    else:
        filtered = df

    # Determine time range for complete bucket coverage
    if start_ts is None:
        start_ts = df["arrival_timestamp"].min()
    if end_ts is None:
        end_ts = df["arrival_timestamp"].max()

    # Create complete time range with 3-hour buckets
    time_buckets = pd.date_range(
        start=start_ts.floor(aggregation_freq),
        end=end_ts.ceil(aggregation_freq),
        freq=aggregation_freq,
    )

    aggregated = filtered.groupby(
        filtered["arrival_timestamp"].dt.floor(aggregation_freq)
    )[target_metric]

    if not return_median:
        if return_count:
            aggregated = aggregated.count()
        elif return_avg:
            aggregated = aggregated.mean()
        elif return_max:
            aggregated = aggregated.max()
        else:
            aggregated = aggregated.sum()

        if return_all_buckets:
            # Reindex to include all time buckets, filling missing values with 0
            aggregated = aggregated.reindex(time_buckets, fill_value=0)

        if return_cummulated:
            # cummulate the values
            aggregated = aggregated.cumsum()

        return aggregated
    else:
        if return_cummulated:
            # Compute rolling median from start date for each bucket
            medians = []

            up_to_bucket_data = []
            for bucket in time_buckets:
                # Select all data up to and including current bucket
                if bucket in aggregated.groups:
                    tmp = aggregated.get_group(bucket).values
                    if not isinstance(tmp, bool) and len(tmp) > 0:
                        # remove all non numeric
                        tmp = [
                            x for x in tmp if not isinstance(x, bool) and pd.notna(x)
                        ]
                        up_to_bucket_data.append(tmp)

                # flatten the list of arrays
                up_to_bucket = [
                    item for sublist in up_to_bucket_data for item in sublist
                ]

                if len(up_to_bucket) == 0 or pd.isna(up_to_bucket).all():
                    medians.append(0)
                else:
                    medians.append(pd.Series(up_to_bucket).median())
            aggregated = pd.Series(medians, index=time_buckets)
        else:
            # compute median per bucket

            try:
                aggregated = aggregated.median()

                if return_all_buckets:
                    # Reindex to include all time buckets, filling missing values with 0
                    aggregated = aggregated.reindex(time_buckets, fill_value=0)
            except Exception as e:
                aggregated = None
                print(
                    f"Error computing median for target_metric {target_metric}: {e}. Returning zeros."
                )

        return aggregated


def extract_info_from_run_name(run_name: str):
    if run_name.endswith("(generation)"):
        gen_strategy = "generation"
        run_name = run_name[: -len("(generation)")].strip()
    elif run_name.endswith("(matching)"):
        gen_strategy = "matching"
        run_name = run_name[: -len("(matching)")].strip()
    elif run_name.endswith("(baseline_round_robin)") or run_name.endswith("(baseline)"):
        gen_strategy = "baseline_round_robin"
        run_name = run_name[: -len("(baseline_round_robin)")].strip()
    elif run_name.lower() == "redset":
        gen_strategy = ""
    else:
        raise ValueError(
            f"Run name {run_name} does not end with (generation), (matching) or (baseline_round_robin)"
        )

    run_name = run_name.replace("_noexplain", "").replace("_explain", "")

    scan_cache = "_scan_cache" in run_name
    subres_cache = "_subresult_cache" in run_name
    baseline = "_baseline" in run_name
    pred_cache = "_predcache" in run_name
    res_cache = "_rescache" in run_name
    only_select = "_onlyselect" in run_name

    if "_subresult_result_cache" in run_name:
        subres_cache = True
        res_cache = True

    run_name = (
        run_name.replace("_scan_cache", "")
        .replace("_subresult_cache", "")
        .replace("_subresult_result_cache", "")
        .replace("_all_caches", "")
        .replace("_baseline", "")
        .replace("_predcache", "")
        .replace("_rescache", "")
        .replace("_onlyselect", "")
        .strip()
    )

    system = run_name.strip().title()

    return (
        system,
        baseline,
        scan_cache,
        subres_cache,
        pred_cache,
        res_cache,
        gen_strategy,
        only_select,
    )


def annotate_relative_runtime(
    per_system_dict, prune_queries_not_there_in_all_configs: bool = True
):
    # this will also drop queries where baseline produced no runtime

    for gen_strategy in ["matching", "generation", "baseline_round_robin"]:
        # compute query runtimes relative to baseline
        for system, system_dict in per_system_dict.items():
            if "baseline" not in system_dict:
                print(f"Skipping {system} as it has no baseline")
                continue

            if gen_strategy not in system_dict["baseline"].keys():
                print(
                    f"Skipping {gen_strategy} for system: {system} as not present for baseline \n {list(system_dict['baseline'].keys())}"
                )
                continue

            assert isinstance(system_dict, dict), (
                f"system_dict for {system} is not a dict: {type(system_dict)}"
            )

            assert gen_strategy in system_dict["baseline"].keys(), (
                f"Baseline for {system} has no '{gen_strategy}'"
            )
            baseline = system_dict["baseline"][gen_strategy]

            # create a mapping from arrival_timestamp -> baseline runtime
            assert "runtime_sec" in baseline.columns, (
                f"Baseline for {system} has no runtime_sec column"
            )
            baseline_runtimes = baseline.set_index("arrival_timestamp")["runtime_sec"]

            print(
                f"Computing relative runtimes for {system} with {len(baseline_runtimes)}/{len(baseline)} baseline queries"
            )

            if prune_queries_not_there_in_all_configs:
                # get minimum set of queries present in all configs

                # make sure arrival time is unique
                assert baseline["arrival_timestamp"].is_unique, (
                    f"Arrival time is not unique for {system}/baseline"
                )
                common_arrival_times = set(baseline["arrival_timestamp"])
                for config, tmp in list(system_dict.items()):
                    if gen_strategy not in tmp:
                        continue

                    df = tmp[gen_strategy]
                    if config == "baseline":
                        continue

                    # make sure arrival time is unique
                    assert df["arrival_timestamp"].is_unique, (
                        f"Arrival time is not unique for {system}/{config}"
                    )

                    common_arrival_times = common_arrival_times.intersection(
                        set(df["arrival_timestamp"])
                    )

                print(
                    f"Num queries reduced from {len(baseline)} to {len(common_arrival_times)} with `prune_queries_not_there_in_all_configs`"
                )

            for config, tmp in list(system_dict.items()):
                if config == "baseline":
                    continue

                if gen_strategy not in tmp:
                    print(
                        f"Skipping {gen_strategy} for system: {system}, config: {config} as not present \n {list(tmp.keys())}"
                    )
                    continue

                df = tmp[gen_strategy].copy()

                # make sure arrival time is unique
                assert df["arrival_timestamp"].is_unique, (
                    f"Arrival time is not unique for {system}/{config}"
                )
                assert baseline["arrival_timestamp"].is_unique, (
                    f"Arrival time is not unique for {system}/baseline"
                )

                # map baseline runtime into current df; non-overlaps become NaN
                df["baseline_runtime_sec"] = df["arrival_timestamp"].map(
                    baseline_runtimes
                )

                # avoid division by zero: treat zero baseline runtime as missing
                df.loc[df["baseline_runtime_sec"] == 0, "baseline_runtime_sec"] = pd.NA

                # compute relative runtime where baseline exists
                df["relative_runtime"] = df["runtime_sec"] / df["baseline_runtime_sec"]

                # keep only overlapping queries (those with a baseline runtime)
                df = df[df["baseline_runtime_sec"].notna()].copy()

                per_system_dict[system][config][gen_strategy] = df

    return per_system_dict


def group_by_system(runs_dict):
    per_system_dict = defaultdict(lambda: defaultdict(dict))
    for run, df in runs_dict.items():
        (
            system,
            baseline,
            scan_cache,
            subres_cache,
            pred_cache,
            res_cache,
            gen_strategy,
            only_select,
        ) = extract_info_from_run_name(run)

        system = system.lower()
        if system == "redset":
            res_cache = True  # redset always uses result cache
        elif system == "???":
            res_cache = True  # ??? always uses result cache
        else:
            assert system in ["duckdb"], f"Unknown system: {system}"

        # assemble key
        key = []
        if baseline:
            key.append("baseline")
        if scan_cache:
            key.append("scan")
        if subres_cache:
            key.append("subresult")
        if pred_cache:
            key.append("pred")
        if res_cache:
            key.append("result")

        if len(key) == 0:
            key = ["baseline"]

        if system == "redset":
            per_system_dict[system]["_".join(key)]["matching"] = df.copy()
            per_system_dict[system]["_".join(key)]["generation"] = df.copy()
            per_system_dict[system]["_".join(key)]["baseline_round_robin"] = df.copy()
        else:
            per_system_dict[system]["_".join(key)][gen_strategy] = df.copy()

    create_redset_baseline_version(per_system_dict)
    create_other_baseline_versions(per_system_dict)

    return per_system_dict
