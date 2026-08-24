"""Postgres connection management (Neon free tier). Reads DATABASE_URL from
the environment. One engine per process; callers get a Connection via
get_connection() and control their own transaction boundaries.
"""

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine

load_dotenv()

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        database_url = os.environ["DATABASE_URL"]
        _engine = create_engine(database_url, pool_pre_ping=True)
    return _engine


@contextmanager
def get_connection():
    """Yields a connection inside a transaction. Commits on clean exit, rolls
    back on exception. Callers should not call conn.commit() themselves.
    """
    engine = get_engine()
    with engine.begin() as conn:
        yield conn


def run_migrations(schema_path: str = "src/db/schema.sql") -> None:
    """Applies schema.sql. Idempotent (CREATE TABLE IF NOT EXISTS), safe to
    call on every deploy.
    """
    with open(schema_path) as f:
        ddl = f.read()
    with get_connection() as conn:
        for statement in ddl.split(";"):
            statement = statement.strip()
            if statement:
                conn.exec_driver_sql(statement)
