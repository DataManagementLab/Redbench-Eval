from collections import defaultdict
from typing import Dict, List


def invalidate_table(result_cache_storage: Dict[str, List[str]], table: str):
    # invalidate all entries in result_cache_storage that reference the given table
    post_cache = defaultdict(list)

    invalidated = 0

    for entry, values in result_cache_storage.items():
        if table not in entry:
            post_cache[entry] = values
        else:
            invalidated += len(values)

    return post_cache, invalidated


def serialize_tables(tables):
    return ",".join(sorted(tables))


def check_result_cache_hit(result_cache_storage, tables, query):
    serialized_tables = serialize_tables(tables)
    return (
        serialized_tables in result_cache_storage
        and query in result_cache_storage[serialized_tables]
    )


def annotate_analytic_rescache_invalidations(df, is_redset=False):
    result_cache_storage = defaultdict(
        list
    )  # sorted list of tables (as str) --> list of queries

    invalidations = []
    cache_hits = []

    if is_redset:
        read_key = "read_table_ids"
        write_key = "write_table_ids"
    else:
        read_key = "join_tables"
        write_key = "write_table"

    for idx, row in df.iterrows():
        assert write_key in row, (
            f"Row {idx} does not have '{write_key}' column. Columns: {df.columns}"
        )
        if row[write_key] is not None:
            # this is a write
            result_cache_storage, invalidated_count = invalidate_table(
                result_cache_storage, row[write_key]
            )
            invalidations.append(invalidated_count)
            cache_hits.append(0)  # writes cannot be cache hits
        else:
            assert read_key in row, (
                f"Row {idx} does not have '{read_key}' column. Columns: {df.columns}"
            )
            # this is a read
            tables = row[read_key]
            tables = tables.split(",")

            assert isinstance(tables, list), (
                f"Expected list of tables, got {type(tables)} in row {idx} / {tables}"
            )

            if is_redset:
                sql = row["query_hash"]
            else:
                assert "sql" in row, (
                    f"Row {idx} does not have 'sql' column. Columns: {df.columns}"
                )
                sql = row["sql"]

            if check_result_cache_hit(result_cache_storage, tables, sql):
                # cache hit
                cache_hits.append(1)
            else:
                cache_hits.append(0)

                # add to result cache
                result_cache_storage[serialize_tables(tables)].append(sql)
            invalidations.append(0)  # reads do not cause invalidations

    df["analytic_rescache_invalidations"] = invalidations
    df["analytic_rescache_cache_hits"] = cache_hits

    return df
