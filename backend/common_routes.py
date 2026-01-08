import os
from typing import Optional, Dict, Any

import jwt
from fastapi import APIRouter, Header, HTTPException

from common_db_runtime import (
    connect_common_db,
    disconnect_common_db,
    get_common_db_session,
    get_common_db_meta,
)

router = APIRouter(prefix="/common", tags=["common-db"])

JWT_ALG = os.getenv("JWT_ALGORITHM", "HS256")
SECRET_KEY = os.getenv("SECRET_KEY", "")


def _token_from_auth_header(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(status_code=401, detail="Invalid Authorization header.")

    return parts[1].strip()


def _decode_token(token: str) -> Dict[str, Any]:
    if not SECRET_KEY:
        raise HTTPException(status_code=500, detail="SECRET_KEY is not set in backend ENV.")

    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


def _extract_context(claims: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    We support a few possible claim names for compatibility.
    Recommended (standardize this on GIS/CAMA side):
      - prov_dbname: "PH04021_Cavite"
      - schema: "PH0402118_Silang"
    """
    prov_dbname = (
        claims.get("prov_dbname")
        or claims.get("prov_db")
        or claims.get("database")
        or claims.get("db_name")
    )

    schema = (
        claims.get("schema")
        or claims.get("municipal_schema")
        or claims.get("active_schema")
    )

    # if they store allowed schemas as list, you can pick the active one if provided
    if not schema and isinstance(claims.get("schemas"), list) and claims["schemas"]:
        schema = claims["schemas"][0]

    return {
        "prov_dbname": prov_dbname,
        "schema": schema,
    }


@router.get("/ping")
def ping():
    return {"ok": True, "message": "common routes are alive"}


@router.get("/context")
def context(authorization: Optional[str] = Header(default=None)):
    token = _token_from_auth_header(authorization)
    claims = _decode_token(token)
    ctx = _extract_context(claims)
    return {
        "ok": True,
        "context": ctx,
        "claims_preview": {
            # keep this small on purpose
            "sub": claims.get("sub"),
            "username": claims.get("username"),
            "user_type": claims.get("user_type"),
        },
    }


@router.get("/status")
def status(authorization: Optional[str] = Header(default=None)):
    token = _token_from_auth_header(authorization)
    meta = get_common_db_meta(token)
    return {
        "connected": meta is not None,
        "meta": meta,
    }


@router.post("/connect")
def connect(authorization: Optional[str] = Header(default=None)):
    token = _token_from_auth_header(authorization)
    claims = _decode_token(token)
    ctx = _extract_context(claims)

    # fallback if token doesn't have prov_dbname yet
    dbname = ctx["prov_dbname"] or os.getenv("COMMON_DB_NAME", "")

    try:
        result = connect_common_db(
            token,
            dbname=dbname,
            schema=ctx["schema"],
        )
        return {
            "ok": True,
            "connected": True,
            "context": ctx,
            "meta": result.get("meta"),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Common DB connection failed: {str(e)}")


@router.post("/disconnect")
def disconnect(authorization: Optional[str] = Header(default=None)):
    token = _token_from_auth_header(authorization)
    ok = disconnect_common_db(token)
    return {"disconnected": ok}
