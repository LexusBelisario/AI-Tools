from fastapi import APIRouter
from .lr_train import router as train_router
from .lr_run_model import router as run_router
from .lr_fields import router as fields_router
from .lr_preview import router as preview_router
from .lr_downloads import router as downloads_router

router = APIRouter(prefix="/linear-regression", tags=["AI Model Tools"])
router.include_router(train_router)
router.include_router(run_router)
router.include_router(fields_router)
router.include_router(preview_router)
router.include_router(downloads_router)