import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine

from src.utils.config import REPO_ROOT

load_dotenv()

_engine: Engine | None = None


def normalize_database_url(database_url: str) -> str:
    for prefix in ("postgresql://", "postgres://"):
        if database_url.startswith(prefix):
            return "postgresql+psycopg://" + database_url[len(prefix):]
    return database_url


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(normalize_database_url(os.environ["DATABASE_URL"]), pool_pre_ping=True)
    return _engine


@contextmanager
def get_connection():
    with get_engine().begin() as conn:
        yield conn


def run_migrations(schema_path: str | Path | None = None) -> None:
    path = Path(schema_path) if schema_path else REPO_ROOT / "src" / "db" / "schema.sql"
    ddl = path.read_text()
    with get_connection() as conn:
        conn.connection.driver_connection.execute(ddl)
