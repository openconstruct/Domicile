"""Drain a local SQLite chat database into transcript logs (generic mapper).

No schema is assumed. Flow for the wizard: tables() -> columns() -> user picks
text/timestamp/sender/conversation columns -> preview() shows sample rows ->
run(). The file is opened read-only (URI mode=ro); nothing is ever written to it.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import import_common as IC

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_ ]{0,63}")


def connect(path) -> sqlite3.Connection:
    p = Path(path).expanduser()
    if not p.is_file():
        raise IC.ImportError_(f"Not found: {p}")
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        con.execute("select count(*) from sqlite_master").fetchone()
    except sqlite3.DatabaseError:
        raise IC.ImportError_("That isn't a readable SQLite database.") from None
    return con


def tables(con) -> list:
    return [r[0] for r in con.execute(
        "select name from sqlite_master where type in ('table','view') "
        "and name not like 'sqlite_%' order by name")]


def columns(con, table: str) -> list:
    if table not in tables(con):
        raise IC.ImportError_(f"No table named '{table}'.")
    return [r[1] for r in con.execute(f'pragma table_info("{table}")')]


def _check(con, table, cols):
    have = columns(con, table)            # also validates the table name
    for c in cols:
        if c and c not in have:
            raise IC.ImportError_(f"Table '{table}' has no column '{c}'.")


def _ts(v) -> str:
    """Accept ISO text, unix seconds, ms, or microseconds."""
    if v is None or v == "":
        return ""
    if isinstance(v, (int, float)) or (isinstance(v, str) and re.fullmatch(r"\d{9,17}(\.\d+)?", v)):
        n = float(v)
        while n > 1e11:                   # ms / us -> s
            n /= 1000
        return datetime.fromtimestamp(n, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return str(v)[:19].replace("T", " ")


def _rows(con, table, text, ts, sender, convo, limit=None):
    sel = ", ".join(f'"{c}"' if c else "NULL" for c in (text, ts, sender, convo))
    q = f'select {sel} from "{table}" where "{text}" is not null'
    if limit:
        q += f" limit {int(limit)}"
    return con.execute(q).fetchall()


def to_messages(rows, default_sender="Unknown", default_title="untitled") -> list:
    out = []
    for body, ts, who, convo in rows:
        body = str(body or "").strip()
        ts = _ts(ts)
        if not body or not ts:
            continue
        out.append({"ts": ts, "who": str(who) if who not in (None, "") else default_sender,
                    "title": str(convo) if convo not in (None, "") else default_title,
                    "body": body})
    out.sort(key=lambda m: m["ts"])
    return out


def preview(con, table, text, ts, sender=None, convo=None, n=5) -> list:
    _check(con, table, [text, ts, sender, convo])
    return to_messages(_rows(con, table, text, ts, sender, convo, limit=n))


def run(path, table, text, ts, sender=None, convo=None, dest=None) -> dict:
    con = connect(path)
    try:
        _check(con, table, [text, ts, sender, convo])
        msgs = to_messages(_rows(con, table, text, ts, sender, convo))
    finally:
        con.close()
    if not msgs:
        raise IC.ImportError_("No usable rows (need non-empty text and a timestamp).")
    return IC.write_by_day("sql", msgs, dest)
