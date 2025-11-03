from typing import Dict, List

import numpy as np
import pandas as pd
from plots.load_data_helper import extract_info_from_run_name, filter_groupby_extract


def build_label_lookup(runs_dict: Dict):
    # Build label lookup
    label_lookup = {}
    for orig in runs_dict.keys():
        (
            system,
            baseline,
            scan_cache,
            subres_cache,
            pred_cache,
            res_cache,
            gen,
            only_select,
        ) = extract_info_from_run_name(orig)

        config = [
            c
            for enabled, c in [
                (scan_cache, "scan"),
                (subres_cache, "subresult"),
                (pred_cache, "pred"),
                (res_cache, "result"),
                (only_select, "onlyselect"),
            ]
            if enabled
        ]

        label = f"{system}"
        if len(config) > 0:
            label += f" (Cache: {', '.join(config)})"

        gen_suffix = {
            "matching": " - match",
            "generation": " - gen",
            "baseline_round_robin": " - rr",
        }
        if gen in gen_suffix:
            label += gen_suffix[gen]
        elif system.lower() != "redset":
            raise ValueError(f"Unknown system/gen combination in run name: {orig}")

        label_lookup[orig] = label

    return label_lookup


def exclude_first_occurence_of_read(df):
    """Drop first occurrence of each SQL query, keeping subsequent repeats."""
    df_sorted = df.sort_values("arrival_timestamp")
    # Drop only the first occurrence of each exact_repetition_hash
    # This keeps the 2nd, 3rd, 4th, etc. occurrences of each hash
    # Get the indices of rows to keep (all except first occurrence of each hash)
    first_indices = df_sorted.drop_duplicates(subset=["sql"], keep="first").index
    return df_sorted.drop(first_indices).sort_values("arrival_timestamp")


def filter_exp(exp_names: List[str], kwargs: Dict):
    filtered_exp_names = []
    for label in exp_names:
        if "duckdb" in label.lower() and not kwargs.get("show_duckdb", True):
            continue
        if "baseline_round_robin" in label.lower() and not kwargs.get(
            "show_baseline_rr", True
        ):
            continue

        if "generation" in label.lower() and not kwargs.get("show_generation", True):
            continue

        if "matching" in label.lower() and not kwargs.get("show_matching", True):
            continue

        if "keyword_filter" in kwargs and kwargs["keyword_filter"] not in label:
            continue

        filtered_exp_names.append(label)

    return filtered_exp_names


def extract_data(
    df: pd.DataFrame,
    metric: str,
    label: str,
    axes,
    i: int,
    color: str,
    filter_extract_kwargs: Dict,
    kwargs: Dict,
):
    start_ts = min(df["arrival_timestamp"])
    end_ts = max(df["arrival_timestamp"])

    date_cutoff = kwargs.get("date_cutoff", None)
    if date_cutoff is not None:
        df = df[df["arrival_timestamp"] < pd.to_datetime(date_cutoff)]

    if metric == "read_ratio_runtime":
        assert "runtime_sec" in df.columns, f"runtime_sec not in columns of {label}"
        ro = filter_groupby_extract(
            df,
            "read-only",
            "runtime_sec",
            start_ts=start_ts,
            end_ts=end_ts,
            **filter_extract_kwargs,
        )
        data = ro / filter_groupby_extract(
            df,
            "all",
            "runtime_sec",
            start_ts=start_ts,
            end_ts=end_ts,
            **filter_extract_kwargs,
        )
    elif metric == "runtime_sec_read":
        data = filter_groupby_extract(
            df,
            "read-only",
            "runtime_sec",
            start_ts=start_ts,
            end_ts=end_ts,
            **filter_extract_kwargs,
        )
    elif metric == "num_queries":
        data = filter_groupby_extract(
            df,
            "all",
            "runtime_sec",
            start_ts=start_ts,
            end_ts=end_ts,
            return_count=True,
            **filter_extract_kwargs,
        )
    elif metric == "read_time_no_first":
        df_filtered = exclude_first_occurence_of_read(df)
        data = filter_groupby_extract(
            df_filtered,
            "read-only",
            "runtime_sec",
            start_ts=start_ts,
            end_ts=end_ts,
            **filter_extract_kwargs,
        )
    elif metric == "repetition_rate":
        total_queries = filter_groupby_extract(
            df,
            "read-only",
            "arrival_timestamp",
            start_ts=start_ts,
            end_ts=end_ts,
            return_count=True,
            **filter_extract_kwargs,
        )

        df_repeating = exclude_first_occurence_of_read(df)

        print(
            f"{label}: {len(df)} total, {len(df_repeating)} after excluding first reads"
        )

        repeated_queries = filter_groupby_extract(
            df_repeating,
            "read-only",
            "arrival_timestamp",
            start_ts=start_ts,
            end_ts=end_ts,
            return_count=True,
            **filter_extract_kwargs,
        )

        data = repeated_queries / total_queries
    elif metric in [
        "hit_result_cache_rate",
        "hit_subresult_cache_rate",
        "hit_scan_cache_rate",
    ]:
        if metric == "hit_result_cache_rate":
            cache_metric = "hit_result_cache"
        elif metric == "hit_subresult_cache_rate":
            cache_metric = "hit_subresult_cache"
        elif metric == "hit_scan_cache_rate":
            cache_metric = "hit_scan_cache"
        else:
            raise ValueError(f"Unknown cache hit rate metric: {metric}")

        if cache_metric not in df.columns or "query_hash" not in df.columns:
            # print(f'Metric "{metric}" not found in {label} data.')
            axes[i].plot([], [], label=label + " (no data)", color=color)
            return
        data = filter_groupby_extract(
            df,
            "read-only",
            cache_metric,
            start_ts=start_ts,
            end_ts=end_ts,
            **filter_extract_kwargs,
        ) / filter_groupby_extract(
            df,
            "read-only",
            "arrival_timestamp",
            start_ts=start_ts,
            end_ts=end_ts,
            return_count=True,
            **filter_extract_kwargs,
        )
    elif metric == "root_card_error":
        if metric not in df.columns:
            # print(f'Metric "{metric}" not found in {label} data.')
            axes[i].plot([], [], label=label + " (no data)", color=color)
            return

        error1 = filter_groupby_extract(
            df,
            "all",
            metric,
            start_ts=start_ts,
            end_ts=end_ts,
            **filter_extract_kwargs,
        )

        ct = filter_groupby_extract(
            df,
            "all",
            "runtime_sec",
            start_ts=start_ts,
            end_ts=end_ts,
            return_count=True,
            **filter_extract_kwargs,
        )

        # compute rolling average with sum divided by count
        # set ct 0 to nan to avoid division by zero
        ct = ct.replace(0, np.nan)
        data = error1.div(ct)
    elif metric == "root_card_error_median":
        if "root_card_error" not in df.columns:
            # print(f'Metric "{metric}" not found in {label} data.')
            axes[i].plot([], [], label=label + " (no data)", color=color)
            return

        data = filter_groupby_extract(
            df,
            "all",
            "root_card_error",
            start_ts=start_ts,
            end_ts=end_ts,
            return_median=True,
            **filter_extract_kwargs,
        )

        # print(f'card med: {data}')
    elif metric in ["qc_cache_size", "query_cache_count"]:
        if metric not in df.columns:
            # print(f'Metric "{metric}" not found in {label} data.')
            axes[i].plot([], [], label=label + " (no data)", color=color)
            return
        data = filter_groupby_extract(
            df,
            "all",
            metric,
            start_ts=start_ts,
            end_ts=end_ts,
            return_max=True,
            **filter_extract_kwargs,
        )

    else:
        if metric not in df.columns:
            # print(f'Metric "{metric}" not found in {label} data.')
            axes[i].plot([], [], label=label + " (no data)", color=color)
            return
        data = filter_groupby_extract(
            df,
            "all",
            metric,
            start_ts=start_ts,
            end_ts=end_ts,
            **filter_extract_kwargs,
        )
    return data
