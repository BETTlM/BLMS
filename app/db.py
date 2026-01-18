from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Paths:
    root: Path
    artifacts_dir: Path
    db_path: Path
    schema_path: Path


def get_paths() -> Paths:
    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts"
    return Paths(
        root=root,
        artifacts_dir=artifacts_dir,
        db_path=artifacts_dir / "bank_loan.db",
        schema_path=root / "schema.sql",
    )


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    paths = get_paths()
    dbp = db_path or paths.db_path
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Streamlit can rerun scripts in different threads; allow cross-thread usage.
    conn = sqlite3.connect(dbp, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    paths = get_paths()
    schema_sql = paths.schema_path.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    # Only drops our 3 tables; keeps the DB file.
    conn.executescript(
        """
        PRAGMA foreign_keys = OFF;
        DROP TABLE IF EXISTS loans;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS loan_officers;
        PRAGMA foreign_keys = ON;
        """
    )
    conn.commit()
    init_db(conn)


def execute_many(
    conn: sqlite3.Connection,
    sql: str,
    rows: Iterable[tuple[Any, ...]],
) -> None:
    conn.executemany(sql, list(rows))
    conn.commit()


def fetch_df(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cur = conn.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]

