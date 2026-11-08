import sys, subprocess, pandas as pd
from translate import split_statements, translate_statement
CH = sys.argv[1]

def ok(batch):
    return subprocess.run([CH, "format", "-n", "--oneline"], input=";\n".join(batch).encode() + b";", capture_output=True).returncode == 0

def failures(batch):
    if ok(batch):
        return []
    if len(batch) == 1:
        return batch
    m = len(batch) // 2
    return failures(batch[:m]) + failures(batch[m:])

for wl in sys.argv[2:]:
    df = pd.read_csv(wl)
    stmts = {}
    for sql in df["sql"].unique():
        for s in split_statements(sql):
            for t in translate_statement(s):
                stmts[t] = s
    keys = list(stmts)
    bad = []
    for i in range(0, len(keys), 1000):
        bad += failures(keys[i:i + 1000])
    for t in bad[:3]:
        print("FAIL:", stmts[t][:300], "\n ->", t[:300], flush=True)
    print(wl, "unique statements:", len(keys), "parse failures:", len(bad), flush=True)
