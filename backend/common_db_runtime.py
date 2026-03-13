# common_db_runtime.py

import os
import contextvars
from typing import Dict, Optional, Any

from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from jose import jwt
from jose.exceptions import JWTError


# =========================
# Request-scoped context
# =========================
_request_ctx: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "request_ctx", default=None
)


def set_request_context(ctx: Optional[Dict[str, Any]]) -> None:
    _request_ctx.set(ctx)


def clear_request_context() -> None:
    _request_ctx.set(None)


def get_request_context() -> Optional[Dict[str, Any]]:
    return _request_ctx.get()


# =========================
# ENV helpers
# =========================
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _jwt_secret() -> str:
    return _env("JWT_SECRET") or _env("SECRET_KEY")


def _jwt_alg() -> str:
    return _env("JWT_ALGORITHM") or "HS256"


# =========================
# Engine cache (per db_name)
# =========================
_ENGINE_CACHE: Dict[str, Any] = {}


def _make_engine(db_name: str):
    host = _env("COMMON_DB_HOST")
    port = _env("COMMON_DB_PORT", "5432")
    user = _env("COMMON_DB_USER")
    password = _env("COMMON_DB_PASSWORD")
    sslmode = _env("COMMON_DB_SSLMODE", "require")

    if not host or not user:
        raise HTTPException(
            status_code=500,
            detail="COMMON_DB_HOST/COMMON_DB_USER not configured in .env",
        )

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}?sslmode={sslmode}"
    return create_engine(url, pool_pre_ping=True, future=True)


def _get_engine(db_name: str):
    if db_name not in _ENGINE_CACHE:
        _ENGINE_CACHE[db_name] = _make_engine(db_name)
    return _ENGINE_CACHE[db_name]


# =========================
# Token decode
# =========================
def _decode_token(token: str) -> Dict[str, Any]:
    secret = _jwt_secret()
    alg = _jwt_alg()

    if not secret:
        raise HTTPException(status_code=500, detail="JWT secret not configured (.e  )")

    try:
        return jwt.decode(token, secret, algorithms=[alg])
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Signature verification failed. ({str(e)})")


# =========================
# Validate db/schema exist
# =========================
def _validate_database_exists(db_name: str) -> bool:
    """
    Check if a database with the exact name exists in PostgreSQL.
    """
    try:
        host = _env("COMMON_DB_HOST")
        port = _env("COMMON_DB_PORT", "5432")
        user = _env("COMMON_DB_USER")
        password = _env("COMMON_DB_PASSWORD")
        sslmode = _env("COMMON_DB_SSLMODE", "require")

        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/postgres?sslmode={sslmode}"
        temp_engine = create_engine(url, pool_pre_ping=True)

        with temp_engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": db_name},
            )
            return result.fetchone() is not None
    except Exception as e:
        print(f"Warning: Could not validate database '{db_name}': {e}")
        return False


def _validate_schema_exists(db_name: str, schema_name: str) -> bool:
    """
    Check if a schema with the exact name exists in the given database.
    """
    try:
        engine = _get_engine(db_name)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM information_schema.schemata WHERE schema_name = :schema_name"),
                {"schema_name": schema_name},
            )
            return result.fetchone() is not None
    except Exception as e:
        print(f"Warning: Could not validate schema '{schema_name}' in '{db_name}': {e}")
        return False


# =========================
# Token → context resolution
# =========================
def resolve_common_context_from_token(
    token: str,
    db_override: Optional[str] = None,
    schema_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Decode a generic JWT and resolve database context.

    Expected token claims:
        - user             : username / identifier
        - province_access  : exact database name   (e.g. 'PH04034_Laguna')
        - municipal_access : exact schema name      (e.g. 'PH0403406_Calauan')

    Headers X-Target-DB / X-Target-Schema can override the token values.
    """
    payload = _decode_token(token)

    # --- extract user ---
    user = payload.get("user")

    # --- resolve database name ---
    db_name = db_override or payload.get("province_access")

    if not db_name:
        raise HTTPException(
            status_code=401,
            detail=f"Token missing 'province_access'. Token claims: {list(payload.keys())}",
        )

    if not _validate_database_exists(db_name):
        raise HTTPException(
            status_code=401,
            detail=f"Database '{db_name}' does not exist.",
        )

    # --- resolve schema name ---
    schema = schema_override or payload.get("municipal_access")

    if not schema:
        raise HTTPException(
            status_code=401,
            detail=f"Token missing 'municipal_access'. Token claims: {list(payload.keys())}",
        )

    if not _validate_schema_exists(db_name, schema):
        raise HTTPException(
            status_code=401,
            detail=f"Schema '{schema}' does not exist in database '{db_name}'.",
        )

    print(f"✅ Resolved: db={db_name}, schema={schema}, user={user}")
    return {"db": str(db_name), "schema": str(schema), "user": user}


# =========================
# Public API used by routes
# =========================
def get_request_session() -> Session:
    ctx = get_request_context()
    if not ctx:
        raise HTTPException(status_code=401, detail="No request context set (missing token?)")

    db_name = ctx.get("db")
    if not db_name:
        raise HTTPException(status_code=401, detail="No db in request context")

    engine = _get_engine(db_name)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return SessionLocal()


def connect_common_db(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stateless 'connect': validates we can open a session to ctx['db'].
    """
    set_request_context(ctx)
    try:
        s = get_request_session()
        try:
            s.execute(text("SELECT 1"))
            s.commit()
        finally:
            s.close()
        return {"connected": True, "context": ctx, "meta": get_common_db_meta()}
    finally:
        clear_request_context()


def disconnect_common_db() -> Dict[str, Any]:
    clear_request_context()
    return {"connected": False, "context": None, "meta": get_common_db_meta()}


def get_common_db_meta() -> Dict[str, Any]:
    return {
        "host": _env("COMMON_DB_HOST"),
        "port": _env("COMMON_DB_PORT", "5432"),
        "user": _env("COMMON_DB_USER"),
        "sslmode": _env("COMMON_DB_SSLMODE", "require"),
        "alg": _jwt_alg(),
    }