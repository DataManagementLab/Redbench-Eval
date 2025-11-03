from typing import Dict

import pandas as pd
from tqdm import tqdm


def create_redset_baseline_version(per_system_dict: Dict[str, Dict[str, pd.DataFrame]]):
    # create the redset baseline & resultcache versions
    tmp = per_system_dict["redset"]["result"]

    if isinstance(tmp, dict):
        todo_dict = tmp
    else:
        todo_dict = {None: tmp}

    for gen_method, redset_baseline in todo_dict.items():
        redset_res_cache = redset_baseline.copy()

        cols_to_reset = [
            "compile_duration_ms",
            "execution_duration_ms",
            "mbytes_scanned",
            "mbytes_spilled",
            "runtime_sec",
            "bytes_read",
        ]
        for col in cols_to_reset:
            # reset where condition is True (use .loc to assign with a boolean mask)
            mask = redset_res_cache["hit_result_cache"]
            redset_res_cache.loc[mask, col] = 0

        # set to not cached
        redset_baseline["hit_result_cache"] = False
        redset_baseline["was_cached"] = False

        if gen_method is None:
            per_system_dict["redset"]["baseline"] = redset_baseline
            per_system_dict["redset"]["result"] = redset_res_cache
        else:
            per_system_dict["redset"]["baseline"][gen_method] = redset_baseline
            per_system_dict["redset"]["result"][gen_method] = redset_res_cache


def create_other_baseline_versions(per_system_dict: Dict[str, Dict[str, pd.DataFrame]]):
    for system in ["SYSTEM NAMES PRUNED HERE!!!!!!!!!!!!!!"]:
        if system in per_system_dict and "result" in per_system_dict[system]:
            tmp = per_system_dict[system]["result"]

            # check if this is already the last level of indirection
            last_level_of_indirection = isinstance(tmp, pd.DataFrame)
            if last_level_of_indirection:
                todo_dict = {None: tmp}
            else:
                todo_dict = tmp

            for key, df in todo_dict.items():
                system_baseline = []
                sql_row_dict = {}

                cols_to_keep = [
                    "query_idx",
                    "query",
                    "arrival_timestamp",
                    "feature_fingerprint",
                    "exact_repetition_hash",
                    "sql",
                ]

                all_cols = df.columns

                for i, row in tqdm(
                    df.iterrows(),
                    total=len(df),
                    desc=f"Creating {system}/{key} baseline version",
                ):
                    row = row.copy()
                    if row["hit_result_cache"]:
                        reference_row = sql_row_dict[row["sql"]]

                        for col in all_cols:
                            if col not in cols_to_keep:
                                row[col] = reference_row[col]

                        # overwrite
                        row["hit_result_cache"] = False
                    else:
                        # update reference row
                        sql_row_dict[row["sql"]] = row

                    system_baseline.append(row)

                # create a full dataframe from rows
                system_baseline = pd.DataFrame(system_baseline)

                if key is None:
                    per_system_dict[system]["baseline"] = system_baseline
                else:
                    per_system_dict[system]["baseline"][key] = system_baseline
