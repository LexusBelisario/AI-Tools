# common_routes.py

from fastapi import APIRouter, Header, HTTPException

from common_db_runtime import (
    connect_common_db,
    disconnect_common_db,
    get_common_db_meta,
    resolve_common_context_from_token,
)

router = APIRouter(prefix="/common", tags=["common"])


def _extract_bearer_token(authorization: str) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization must be: Bearer <token>")

    return parts[1].strip()


@router.post("/connect")
def connect(
    authorization: str = Header(default=""),
    x_target_schema: str = Header(default="", alias="X-Target-Schema"),
    x_target_db: str = Header(default="", alias="X-Target-DB"),
):
    token = _extract_bearer_token(authorization)

    # ✅ pass header overrides BEFORE validation
    ctx = resolve_common_context_from_token(
        token,
        db_override=(x_target_db or None),
        schema_override=(x_target_schema or None),
    )

    return connect_common_db(ctx)


@router.get("/status")
def status(
    authorization: str = Header(default=""),
    x_target_schema: str = Header(default="", alias="X-Target-Schema"),
    x_target_db: str = Header(default="", alias="X-Target-DB"),
):
    if not authorization:
        return {"connected": False, "context": None, "meta": get_common_db_meta()}

    token = _extract_bearer_token(authorization)

    ctx = resolve_common_context_from_token(
        token,
        db_override=(x_target_db or None),
        schema_override=(x_target_schema or None),
    )

    return connect_common_db(ctx)


@router.post("/disconnect")
def disconnect():
    return disconnect_common_db()


@router.get("/meta")
def meta():
    return get_common_db_meta()
