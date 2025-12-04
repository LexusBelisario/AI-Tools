from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import uvicorn

# === Import Routers ===
from auth.routes import router as auth_router
from admin.routes import router as admin_router 
from routes.geomdisplay import router as geom_router
from routes.schemas import router as schema_router
from routes.parcelinfo import router as parcel_router
from routes.edit import router as edit_router
from routes.orthophoto import router as orthophoto_router
from routes.consolidate import router as merge_router
from routes.subdivide import router as subdivide_router
from routes.thematic import router as thematic_router
from routes.tableinfo import router as tableinfo_router
from routes.landmarks import router as landmark_router
from routes.search import router as search_router
from routes.province import router as province_router
from routes.municipal import router as municipal_router
from routes.sync import router as sync_router
from routes.tmcr import router as tmcr_router
from routes.matchingreport import router as matchingreport_router
from AITools.ai_tools_router import router as ai_tools_router

# ⭐ NEW ROUTER (ADDED ONLY THIS)
from routes.taxmaplayout import router as taxmaplayout_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ==========================================================
# 🔌 Register Routers
# ==========================================================
app.include_router(auth_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(geom_router, prefix="/api")
app.include_router(schema_router, prefix="/api")
app.include_router(parcel_router, prefix="/api")
app.include_router(orthophoto_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(edit_router, prefix="/api")
app.include_router(merge_router, prefix="/api")
app.include_router(subdivide_router, prefix="/api")
app.include_router(landmark_router, prefix="/api")
app.include_router(thematic_router, prefix="/api")
app.include_router(tableinfo_router, prefix="/api")
app.include_router(province_router, prefix="/api")
app.include_router(municipal_router, prefix="/api")
app.include_router(sync_router, prefix="/api")
app.include_router(tmcr_router, prefix="/api")
app.include_router(matchingreport_router, prefix="/api")
app.include_router(ai_tools_router, prefix="/api")

# ⭐ ADDED THIS ONE ONLY
app.include_router(taxmaplayout_router, prefix="/api")


# ==========================================================
# ❤️ Health Check
# ==========================================================
@app.get("/health")
def health():
    return {"message": "API is up"}


# ==========================================================
# 📦 Serve React Static Files
# ==========================================================
STATIC_DIR = os.path.join(os.getcwd(), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

INDEX_HTML = os.path.join(STATIC_DIR, "index.html")


# ==========================================================
# 🧭 Fallback for React Router
# ==========================================================
@app.middleware("http")
async def react_fallback(request: Request, call_next):
    if (
        not request.url.path.startswith("/api")
        and not request.url.path.startswith("/assets")
        and not request.url.path.startswith("/static")
        and "." not in request.url.path
    ):
        if os.path.exists(INDEX_HTML):
            return FileResponse(INDEX_HTML)
    return await call_next(request)


# ==========================================================
# ▶️ Entry Point
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
