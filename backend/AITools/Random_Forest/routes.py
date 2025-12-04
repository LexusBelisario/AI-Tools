# rf_routes.py
from fastapi import APIRouter

from .rf_train import router as rf_train_router
from .rf_run_model import router as rf_run_router
from .rf_fields import router as rf_fields_router
from .rf_preview import router as rf_preview_router
from .rf_downloads import router as rf_downloads_router


router = APIRouter(
    prefix="/rf",
    tags=["Random Forest"],
)
router.include_router(rf_train_router)
router.include_router(rf_run_router)
router.include_router(rf_fields_router)
router.include_router(rf_preview_router)
router.include_router(rf_downloads_router)