import os
from typing import Dict, Optional, Tuple
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


_common_engines: Dict[str, Engine] = {}
_common_sessionmakers: Dict[str, sessionmaker] = {}
_common_meta: Dict[str, dict] = {}


def _env(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    return (val or "").strip()


def _make_url(host: str, port: int, dbname: str, user: str, password: str) -> str:
    sslmode = _env("COMMON_DB_SSLMODE", "require")
    u = quote_plus(user)
    p = quote_plus(password)
    h = host.strip()
    return f"postgresql+psycopg2://{u}:{p}@{h}:{int(port)}/{dbname}?sslmode={sslmode}"


def connect_common_db(
    key: str,
    *,
    dbname: Optional[str] = None,
    schema: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> dict:
    """
    Creates (or reuses) a SQLAlchemy engine/session for the Common DB.

    - All connection secrets come from ENV by default.
    - dbname/schema can be overridden (e.g., derived from JWT token claims).
    - 'key' identifies the connection context (we use the JWT token string).
    """
    if key in _common_engines and key in _common_sessionmakers and key in _common_meta:
        # already connected for this key
        return {"connected": True, "meta": _common_meta[key]}

    host = (host or _env("COMMON_DB_HOST")).strip()
    port = int(port or _env("COMMON_DB_PORT", "5432"))
    dbname = (dbname or _env("COMMON_DB_NAME")).strip()
    user = (user or _env("COMMON_DB_USER")).strip()
    password = password or _env("COMMON_DB_PASSWORD")

    if not host:
        raise ValueError("COMMON_DB_HOST is not set.")
    if not dbname:
        raise ValueError("COMMON_DB_NAME is not set (and no dbname override was provided).")
    if not user:
        raise ValueError("COMMON_DB_USER is not set.")
    if not password:
        raise ValueError("COMMON_DB_PASSWORD is not set.")

    url = _make_url(host, port, dbname, user, password)

    engine = create_engine(
        url,
        pool_pre_ping=True,
        future=True,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    # quick sanity check
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    _common_engines[key] = engine
    _common_sessionmakers[key] = SessionLocal
    _common_meta[key] = {
        "host": host,
        "port": port,
        "dbname": dbname,
        "user": user,
        "sslmode": _env("COMMON_DB_SSLMODE", "require"),
        "schema": schema,  # default schema from token (municipality)
    }

    return {"connected": True, "meta": _common_meta[key]}


def disconnect_common_db(key: str) -> bool:
    engine = _common_engines.pop(key, None)
    _common_sessionmakers.pop(key, None)
    _common_meta.pop(key, None)

    if engine is not None:
        engine.dispose()
        return True
    return False


def get_common_db_session(key: str) -> Optional[Tuple[Session, dict]]:
    SessionLocal = _common_sessionmakers.get(key)
    meta = _common_meta.get(key)
    if not SessionLocal or not meta:
        return None
    return SessionLocal(), meta


def get_common_db_meta(key: str) -> Optional[dict]:
    return _common_meta.get(key)
