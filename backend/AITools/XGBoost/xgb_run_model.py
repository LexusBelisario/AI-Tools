from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional

import tempfile
import os
import joblib
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from db import get_user_database_session
from AITools.ai_utils import (
    gdf_from_db_with_geometry,
    gdf_from_zip_or_parts,
    get_provincial_code_from_schema,
)

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)


@router.post("/run-saved-model")
async def run_saved_model_unified(
    model_file: UploadFile,
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
):
    """
    Unified XGBoost run-saved-model endpoint.

    Behavior aligned with Linear Regression's /run-saved-model:
    - Accepts DB mode (schema + table_name) OR file mode (shapefiles / zip_file).
    - Loads saved model bundle (model + scaler + features).
    - Runs predictions, exports shapefile ZIP + simple PDF summary.
    - Optionally saves to DB as <table_name>_PredictedRun when in DB mode.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # --- Load model bundle ---
            model_path = os.path.join(tmpdir, model_file.filename)
            with open(model_path, "wb") as f:
                f.write(await model_file.read())

            bundle = joblib.load(model_path)
            model = bundle["model"]
            scaler = bundle.get("scaler", None)
            # keep features in lowercase for matching
            features = [f.lower() for f in bundle.get("features", [])]
            print(f"✅ Loaded XGBoost model with features: {features}")

            # --- Detect input source ---
            if schema and table_name:
                # ========== DATABASE MODE ==========
                print(f"🗄️ Running model on database: {schema}.{table_name}")
                gdf = gdf_from_db_with_geometry(schema, table_name)
            else:
                # ========== FILE MODE ==========
                print("📂 Running model on shapefile/zip input")
                gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)

            # --- Prepare dataframe ---
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
            df.columns = [c.lower() for c in df.columns]

            # Ensure all required features are present
            missing = [f for f in features if f not in df.columns]
            if missing:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Missing features in input: {missing}"},
                )

            # Numeric conversion + fill NaNs
            X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

            # Predict using scaler if present
            if scaler is not None:
                X_transformed = scaler.transform(X)
                preds = model.predict(X_transformed)
            else:
                preds = model.predict(X)

            preds = np.array(preds).flatten()
            gdf["prediction"] = preds

            # --- Export results ---
            export_path = os.path.join(
                EXPORT_DIR, f"xgb_run_{np.random.randint(100000, 999999)}"
            )
            os.makedirs(export_path, exist_ok=True)

            # Shapefile + ZIP
            shp_dir = os.path.join(export_path, "predicted_shapefile")
            os.makedirs(shp_dir, exist_ok=True)
            shp_path = os.path.join(shp_dir, "predicted_output.shp")
            gdf.to_file(shp_path)

            zip_out = os.path.join(export_path, "predicted_output.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(shp_dir):
                    z.write(os.path.join(shp_dir, f), f)

            # Simple PDF summary (same style as LR run)
            pdf_path = os.path.join(export_path, "run_report.pdf")
            with PdfPages(pdf_path) as pp:
                fig, ax = plt.subplots(figsize=(7, 2))
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"XGBoost predictions completed\n{len(preds)} records processed.",
                    ha="center",
                    va="center",
                    fontsize=13,
                    weight="bold",
                )
                pp.savefig(fig)
                plt.close(fig)

            # --- Optional: save to DB table ---
            if schema and table_name:
                try:
                    provincial_code = get_provincial_code_from_schema(schema)
                    db_session = get_user_database_session(provincial_code)
                    engine = db_session.get_bind()
                    out_table = f"{table_name}_PredictedRun"
                    gdf.to_postgis(
                        name=out_table,
                        con=engine,
                        schema=schema,
                        if_exists="replace",
                        index=False,
                    )
                    print(f"✅ Saved XGBoost run output to {schema}.{out_table}")
                    engine.dispose()
                    db_session.close()
                except Exception as e:
                    print(f"⚠️ Could not save XGBoost predictions to DB: {e}")

            # --- Downloads response (same pattern as LR, but XGB base URL) ---
            base_url = "/api/xgb/download"
            return {
                "message": "✅ XGBoost model run completed successfully.",
                "downloads": {
                    "report": f"{base_url}?file={pdf_path}",
                    "shapefile": f"{base_url}?file={zip_out}",
                },
                "record_count": int(len(gdf)),
                "isRunMode": True,
            }

    except Exception as e:
        import traceback

        print("❌ XGB RUN ERROR:", e)
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
