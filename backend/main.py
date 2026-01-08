from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from common_routes import router as common_router
import os
import uvicorn

from AITools.ai_tools_router import router as ai_tools_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(ai_tools_router, prefix="/api")
app.include_router(common_router, prefix="/api")


@app.get("/health")
def health():
    return {"message": "API is up"}

STATIC_DIR = os.path.join(os.getcwd(), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

INDEX_HTML = os.path.join(STATIC_DIR, "index.html")


@app.middleware("http")
async def react_fallback(request: Request, call_next):
    path = request.url.path

    if (
        not path.startswith("/api")
        and not path.startswith("/assets")
        and not path.startswith("/static")
        and not path.startswith("/docs")
        and not path.startswith("/redoc")
        and path != "/openapi.json"
        and "." not in path
    ):
        if os.path.exists(INDEX_HTML):
            return FileResponse(INDEX_HTML)

    return await call_next(request)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
