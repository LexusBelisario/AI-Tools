# main.py

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from common_routes import router as common_router
from AITools.ai_tools_router import router as ai_tools_router

from common_db_runtime import (
    resolve_common_context_from_token,
    set_request_context,
    clear_request_context,
)

app = FastAPI()


def _cors_origins():
    """
    Docker/dev safe:
    - If may CORS_ORIGINS sa env, yun ang gagamitin (comma-separated).
    - Else fallback sa common local dev origins.
    """
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]

    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


# CORS
# Note: wag maglagay ng "*" dito kapag allow_credentials=True
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def attach_ctx(request: Request, call_next):
    """
    If may Authorization Bearer token, decode + store request context
    para si db.get_user_database_session() gumana.
    """
    auth = request.headers.get("Authorization", "")
    x_target_schema = request.headers.get("X-Target-Schema", "")

    try:
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
            try:
                ctx = resolve_common_context_from_token(token)
                if x_target_schema:
                    ctx["schema"] = x_target_schema
                set_request_context(ctx)
            except Exception:
                # wag hard-fail, kasi may endpoints na di need auth
                clear_request_context()

        response = await call_next(request)
        return response
    finally:
        clear_request_context()


app.include_router(common_router, prefix="/api")
app.include_router(ai_tools_router, prefix="/api")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
