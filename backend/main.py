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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def attach_ctx(request: Request, call_next):
    """
    If Authorization Bearer token exists, decode and store request context
    so db.get_user_database_session() can use it.
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
                # Don't hard-fail here; endpoint may not need auth.
                clear_request_context()

        response = await call_next(request)
        return response
    finally:
        clear_request_context()


# Routes (IMPORTANT: include these before mounting static)
app.include_router(common_router, prefix="/api")
app.include_router(ai_tools_router, prefix="/api")

# Static build (optional)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "dist")
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
