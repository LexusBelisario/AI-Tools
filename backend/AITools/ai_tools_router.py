from fastapi import APIRouter

# Modular routers
from AITools.ai_tools_fields import router as fields_router
from AITools.ai_tools_preview import router as preview_router
from AITools.ai_tools_run_model import router as run_model_router
from AITools.ai_tools_downloads import router as downloads_router
from AITools.ai_tools_models import router as models_router  # dagdag: models endpoints
from AITools.lr_train import router as lr_router
from AITools.xgb_train import router as xgb_router
from AITools.rf_train import router as rf_router
from AITools.slm_train import router as slm_router
from AITools.sdm_train import router as sdm_router
from AITools.hybrid_slm_rf_train import router as hybrid_router
from AITools.sdm_xgb_train import router as hybrid_sdm_xgb_router
from AITools.gwr_train import router as gwr_router

# Main router for AI Tools
router = APIRouter(prefix="/ai-tools")

# ------------------------------------------------------------
# Mount all AI tool submodules
# ------------------------------------------------------------
router.include_router(lr_router,              prefix="/train-lr")
router.include_router(rf_router,              prefix="/train-rf")
router.include_router(xgb_router,             prefix="/train-xgb")
router.include_router(slm_router,             prefix="/train-slm")
router.include_router(sdm_router,             prefix="/train-sdm")
router.include_router(gwr_router,             prefix="/train-gwr")
router.include_router(hybrid_router,          prefix="/train-hybrid-slm-rf")
router.include_router(hybrid_sdm_xgb_router,  prefix="/train-hybrid-sdm-xgb")

router.include_router(preview_router)      # /ai-tools/...
router.include_router(run_model_router)    # /ai-tools/...
router.include_router(downloads_router)    # /ai-tools/...
router.include_router(fields_router)       # /ai-tools/...

router.include_router(models_router)       # dagdag: /ai-tools/list-models, /get-model-blob, save-trained-model-local, etc.

# ------------------------------------------------------------
# (Optional) Health check endpoint
# ------------------------------------------------------------
@router.get("/status")
async def ai_tools_status():
    return {
        "status": "AI Tools Router Loaded",
        "modules": [
            "train",
            "fields",
            "preview",
            "run-saved-model",
            "downloads",
            "models",
            "train-lr",
            "train-rf",
            "train-xgb",
            "train-slm",
            "train-sdm",
            "train-gwr",
            "train-hybrid-slm-rf",
            "train-hybrid-sdm-xgb",
        ],
    }