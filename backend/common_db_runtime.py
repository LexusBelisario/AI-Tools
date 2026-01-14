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
    return _env("GIS_JWT_SECRET") or _env("SECRET_KEY")


def _jwt_alg() -> str:
    return _env("GIS_JWT_ALG") or _env("JWT_ALGORITHM") or "HS256"


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
# Token decode + resolve db/schema
# =========================
def _decode_token(token: str) -> Dict[str, Any]:
    secret = _jwt_secret()
    alg = _jwt_alg()

    if not secret:
        raise HTTPException(status_code=500, detail="JWT secret not configured (.env)")

    try:
        return jwt.decode(token, secret, algorithms=[alg])
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Signature verification failed. ({str(e)})")


# Province code → name mapping (add all your provinces)
PROVINCE_NAMES = {
    "PH04021": "Cavite",
    "PH04034": "Laguna",
    # Add more as needed...
}

# Municipality code → name mapping (add all your municipalities)
MUNICIPALITY_NAMES = {
    "PH0402118": "Silang",
    # Add more as needed...
}


def resolve_common_context_from_token(
    token: str,
    db_override: Optional[str] = None,
    schema_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Decode token and extract db/schema.
    Headers can override token values.
    
    For your GIS token structure:
    - provincial_access (e.g., 'PH04021') → 'PH04021_Cavite'
    - municipal_access (e.g., 'PH0402118') → 'PH0402118_Silang'
    """
    payload = _decode_token(token)

    # If headers provide complete overrides, use them
    if db_override and schema_override:
        return {"db": str(db_override), "schema": str(schema_override)}

    # Get province code from token
    prov_code = payload.get("provincial_access")
    mun_code = payload.get("municipal_access")

    # Build database name: PH04021 → PH04021_Cavite
    if db_override:
        db_name = db_override
    elif prov_code:
        prov_name = PROVINCE_NAMES.get(prov_code, prov_code)
        db_name = f"{prov_code}_{prov_name}"
    else:
        db_name = None

    # Build schema name: PH0402118 → PH0402118_Silang
    if schema_override:
        schema = schema_override
    elif mun_code:
        mun_name = MUNICIPALITY_NAMES.get(mun_code, mun_code)
        schema = f"{mun_code}_{mun_name}"
    else:
        schema = None

    if not db_name or not schema:
        raise HTTPException(
            status_code=401,
            detail=f"Token missing db/schema (db={db_name}, schema={schema}). Token claims: {list(payload.keys())}",
        )

    return {"db": str(db_name), "schema": str(schema)}


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
        "jwt_secret_source": "GIS_JWT_SECRET" if _env("GIS_JWT_SECRET") else "SECRET_KEY",
    }