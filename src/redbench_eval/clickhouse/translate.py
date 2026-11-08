"""
Translate the PostgreSQL/DuckDB dialect emitted by Redbench into ClickHouse SQL.

Handled constructs:
  - multiple statements in one workload entry (matching: "DELETE ...; INSERT ...;")
  - backslashes in string literals (literal in PostgreSQL standard strings, escape character in ClickHouse)
  - col ~ 'regex'                                 -> match(col, 'regex')
  - DELETE FROM t USING a, b WHERE c              -> DELETE FROM t WHERE id IN (SELECT t.id FROM t, a, b WHERE c)
  - UPDATE t SET x = v FROM a, b WHERE c          -> UPDATE t SET x = v WHERE id IN (SELECT t.id FROM t, a, b WHERE c)
  - INSERT INTO t (...) SELECT ... ON CONFLICT (k) DO NOTHING
                                                  -> INSERT INTO t (...) SELECT ... WHERE ... AND <k expr> NOT IN (SELECT k FROM t)
  - CREATE TABLE t AS SELECT ...; ALTER TABLE t ADD PRIMARY KEY (k)
                                                  -> CREATE TABLE t ENGINE = MergeTree ORDER BY k AS SELECT ...
"""

import re

PK = '"id"'


def _scan(s):
    """Yield (index, char, in_quote) over s, tracking single-quoted literals and double-quoted identifiers."""
    in_s = in_d = False
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if in_s:
            if c == "'":
                if i + 1 < n and s[i + 1] == "'":
                    yield i, c, True
                    yield i + 1, "'", True
                    i += 2
                    continue
                in_s = False
                yield i, c, True
            else:
                yield i, c, True
        elif in_d:
            if c == '"':
                in_d = False
            yield i, c, True
        else:
            if c == "'":
                in_s = True
                yield i, c, True
            elif c == '"':
                in_d = True
                yield i, c, True
            else:
                yield i, c, False
        i += 1


def split_statements(sql):
    out, start = [], 0
    for i, c, q in _scan(sql):
        if c == ";" and not q:
            out.append(sql[start:i])
            start = i + 1
    out.append(sql[start:])
    return [s.strip() for s in out if s.strip()]


def _escape_backslashes(s):
    res = []
    for i, c, q in _scan(s):
        res.append("\\\\" if c == "\\" and q else c)
    return "".join(res)


def _find_kw(s, kw, start=0):
    """Position of keyword kw (case-insensitive, word-bounded) outside quotes and at paren depth 0."""
    depth = 0
    kwu = kw.upper()
    L = len(kw)
    for i, c, q in _scan(s):
        if q:
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and i >= start and s[i:i + L].upper() == kwu:
            before = s[i - 1] if i > 0 else " "
            after = s[i + L] if i + L < len(s) else " "
            if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                return i
    return -1


def _split_top_commas(s):
    parts, depth, start = [], 0, 0
    for i, c, q in _scan(s):
        if q:
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return parts


REGEX_OP = re.compile(r"([^\s\(]+)\s*~\s*('(?:[^']|'')*')")


def _rewrite_regex(s):
    # Only applied outside literals in practice: Redbench literals do not contain " ~ '".
    return REGEX_OP.sub(r"match(\1, \2)", s)


def write_table_of(stmt):
    m = re.match(r'\s*(?:DELETE\s+FROM|UPDATE|INSERT\s+INTO|CREATE\s+TABLE)\s+("[^"]+"|\w+)', stmt, re.I)
    if not m:
        raise ValueError(f"Cannot determine write table of: {stmt[:300]}")
    return m.group(1).strip('"')


def _rewrite_delete_using(s):
    m = re.match(r'\s*DELETE\s+FROM\s+("[^"]+"|\w+)\s+', s, re.I)
    if not m:
        return s
    t = m.group(1)
    p_using = _find_kw(s, "USING", m.end())
    if p_using < 0:
        return s
    p_where = _find_kw(s, "WHERE", p_using)
    using = s[p_using + len("USING"):p_where if p_where >= 0 else len(s)].strip()
    cond = s[p_where + len("WHERE"):].strip() if p_where >= 0 else "1"
    return f"DELETE FROM {t} WHERE {PK} IN (SELECT {t}.{PK} FROM {t}, {using} WHERE {cond})"


def _rewrite_update_from(s):
    m = re.match(r'\s*UPDATE\s+("[^"]+"|\w+)\s+SET\s+', s, re.I)
    if not m:
        return s
    t = m.group(1)
    p_from = _find_kw(s, "FROM", m.end())
    if p_from < 0:
        return s
    p_where = _find_kw(s, "WHERE", p_from)
    set_clause = s[m.end():p_from].strip()
    frm = s[p_from + len("FROM"):p_where if p_where >= 0 else len(s)].strip()
    cond = s[p_where + len("WHERE"):].strip() if p_where >= 0 else "1"
    return f"UPDATE {t} SET {set_clause} WHERE {PK} IN (SELECT {t}.{PK} FROM {t}, {frm} WHERE {cond})"


def _rewrite_on_conflict(s):
    p = _find_kw(s, "ON CONFLICT")
    if p < 0:
        return s
    tail = s[p:]
    m = re.match(r"ON\s+CONFLICT\s*\(([^)]*)\)\s*DO\s+NOTHING\s*$", tail, re.I)
    if not m:
        raise ValueError(f"Unsupported ON CONFLICT clause: {tail[:200]}")
    keys = [k.strip() for k in m.group(1).split(",")]
    s = s[:p].rstrip()

    mi = re.match(r'\s*INSERT\s+INTO\s+("[^"]+"|\w+)\s*\(([^)]*)\)\s*', s, re.I)
    if not mi:
        raise ValueError(f"Unsupported INSERT with ON CONFLICT: {s[:300]}")
    target = mi.group(1)
    write_cols = [c.strip() for c in mi.group(2).split(",")]
    select = s[mi.end():]

    p_select = _find_kw(select, "SELECT")
    p_from = _find_kw(select, "FROM")
    head = select[p_select + len("SELECT"):p_from].strip()
    md = re.match(r"DISTINCT\s+ON\s*\(", head, re.I)
    if md:
        # skip the DISTINCT ON (...) list
        depth, j = 1, md.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(head[j], 0)
            j += 1
        head = head[j:]
    exprs = _split_top_commas(head)
    if len(exprs) != len(write_cols):
        raise ValueError(f"Cannot map select list to insert columns: {s[:300]}")
    key_exprs = [exprs[write_cols.index(k)] for k in keys]
    lhs = key_exprs[0] if len(key_exprs) == 1 else f"({', '.join(key_exprs)})"
    rhs_cols = ", ".join(keys)
    anti = f"{lhs} NOT IN (SELECT {rhs_cols} FROM {target})"

    p_where = _find_kw(select, "WHERE", p_from)
    # Clauses that may follow WHERE at top level.
    p_end = len(select)
    for kw in ("GROUP BY", "ORDER BY", "LIMIT"):
        q = _find_kw(select, kw, p_from)
        if q >= 0:
            p_end = min(p_end, q)
    if p_where >= 0:
        cond = select[p_where + len("WHERE"):p_end].strip()
        select = f"{select[:p_where]}WHERE ({cond}) AND {anti} {select[p_end:]}"
    else:
        select = f"{select[:p_end].rstrip()} WHERE {anti} {select[p_end:]}"
    return f"INSERT INTO {target} ({', '.join(write_cols)}) {select.strip()}"


def translate_statement(stmt):
    """Translate one statement; returns a list of ClickHouse statements."""
    s = _escape_backslashes(stmt.strip().rstrip(";").strip())
    if " ~ " in s or "~'" in s:
        s = _rewrite_regex(s)
    u = s.lstrip().upper()
    if u.startswith("ALTER TABLE") and "ADD PRIMARY KEY" in u:
        return []  # folded into CREATE TABLE ... ORDER BY below
    if u.startswith("CREATE TABLE") and _find_kw(s, "AS") >= 0:
        m = re.match(r'\s*CREATE\s+TABLE\s+("[^"]+"|\w+)\s+AS\s+', s, re.I)
        if m:
            return [f"CREATE TABLE {m.group(1)} ENGINE = MergeTree ORDER BY {PK} "
                    f"SETTINGS enable_block_number_column = 1, enable_block_offset_column = 1 "
                    f"AS {s[m.end():]}"]
    if u.startswith("DELETE"):
        return [_rewrite_delete_using(s)]
    if u.startswith("UPDATE"):
        return [_rewrite_update_from(s)]
    if u.startswith("INSERT"):
        return [_rewrite_on_conflict(s)]
    if u.startswith("COPY"):
        raise ValueError("COPY statements are not supported (include_copy is disabled in all paper workloads)")
    return [s]
