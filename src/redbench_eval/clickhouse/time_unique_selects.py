import sys, time, pandas as pd
from translate import split_statements, translate_statement
from execute_queries_clickhouse import CH
wl, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(wl)
sel = df[df.query_type == "select"].groupby("sql").size().reset_index(name="reps")
ch = CH("127.0.0.1", 38123, "redbench")
rows = []
for r in sel.itertuples():
    (s,) = [t for st in split_statements(r.sql) for t in translate_statement(st)]
    t0 = time.perf_counter()
    err = None
    try:
        ch.execute(s.rstrip(";") + " FORMAT Null", {"use_query_cache": 0, "max_execution_time": 300})
    except RuntimeError as e:
        err = str(e)[:300]
    rows.append({"sql": r.sql, "reps": r.reps, "sec": time.perf_counter() - t0, "err": err})
    print(len(rows), f"{rows[-1]['sec']:.2f}", err or "", flush=True)
pd.DataFrame(rows).to_parquet(out)
