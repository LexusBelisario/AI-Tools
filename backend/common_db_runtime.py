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


def _lookup_db_by_province_code(prov_code: str) -> Optional[str]:
    """
    Query PostgreSQL to find database name matching province PSGC code.
    Returns the actual database name (e.g., 'PH04021_Cavite').
    """
    try:
        # Connect to postgres system DB to list all databases
        host = _env("COMMON_DB_HOST")
        port = _env("COMMON_DB_PORT", "5432")
        user = _env("COMMON_DB_USER")
        password = _env("COMMON_DB_PASSWORD")
        sslmode = _env("COMMON_DB_SSLMODE", "require")
        
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/postgres?sslmode={sslmode}"
        temp_engine = create_engine(url, pool_pre_ping=True)
        
        with temp_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT datname FROM pg_database WHERE datname LIKE :pattern"
            ), {"pattern": f"{prov_code}_%"})
            
            db_names = [row[0] for row in result]
            if db_names:
                return db_names[0]  # Return first match
        
        return None
    except Exception as e:
        print(f"Warning: Could not lookup database for {prov_code}: {e}")
        return None

def _list_db_candidates_by_province_code(prov_code: str) -> list[str]:
    """
    Returns all DB names matching province code.
    Example: PH04021 -> ['PH04021_Cavite', 'PH04021_OtherCopy']
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
                text("SELECT datname FROM pg_database WHERE datname LIKE :pattern ORDER BY datname"),
                {"pattern": f"{prov_code}_%"},
            )
            return [row[0] for row in result]
    except Exception as e:
        print(f"Warning: Could not list databases for {prov_code}: {e}")
        return []


def _lookup_schema_by_mun_code(db_name: str, mun_code: str) -> Optional[str]:
    """
    Query the database to find schema name matching municipal PSGC code.
    Returns the actual schema name (e.g., 'PH0402118_Silang').
    """
    try:
        engine = _get_engine(db_name)
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE :pattern"
            ), {"pattern": f"{mun_code}_%"})
            
            schema_names = [row[0] for row in result]
            if schema_names:
                return schema_names[0]  # Return first match
        
        return None
    except Exception as e:
        print(f"Warning: Could not lookup schema for {mun_code}: {e}")
        return None


def resolve_common_context_from_token(
    token: str,
    db_override: Optional[str] = None,
    schema_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Decode token and extract db/schema by matching PSGC codes.
    
    For your GIS token structure:
    - provincial_access (e.g., 'PH04021') → looks up 'PH04021_Cavite' from database list
    - municipal_access (e.g., 'PH0402118') → looks up 'PH0402118_Silang' from schemas
    
    This matches schemas by PSGC code prefix, just like your GIS app.
    """
    payload = _decode_token(token)

    # If headers provide complete overrides, use them
    if db_override and schema_override:
        return {"db": str(db_override), "schema": str(schema_override)}

    # Get codes from token
    prov_code = payload.get("provincial_access")
    mun_code = payload.get("municipal_access")

    if not prov_code or not mun_code:
        raise HTTPException(
            status_code=401,
            detail=f"Token missing provincial_access or municipal_access. Token claims: {list(payload.keys())}",
        )

    # Lookup actual database name by province code
    if db_override:
        db_name = db_override
    else:
        candidates = _list_db_candidates_by_province_code(prov_code)
        if not candidates:
            raise HTTPException(
                status_code=401,
                detail=f"No database found matching province code '{prov_code}'. Check COMMON_DB connection.",
            )
            
        chosen_db = None
        chosen_schema = None
        
        for cand_db in candidates:
            cand_schema = _lookup_schema_by_mun_code(cand_db, mun_code)
            if cand_schema:
                chosen_db = cand_db
                chosen_schema = cand_schema
                break
        
        if chosen_db:
            db_name = chosen_db
            # if schema_override not provided, use the found one
            if not schema_override:
                schema = chosen_schema
        else:
            # fallback: first candidate
            db_name = candidates[0]

    # Lookup actual schema name by municipal code
    if schema_override:
        schema = schema_override
    else:
        # if schema already set from candidate scan, keep it
        if not locals().get("schema"):
            schema = _lookup_schema_by_mun_code(db_name, mun_code)
        if not schema:
            raise HTTPException(
                status_code=401,
                detail=f"No schema found in database '{db_name}' matching municipal code '{mun_code}'.",
            )

    print(f"✅ Resolved: {prov_code} → {db_name}, {mun_code} → {schema}")
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