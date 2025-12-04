# AITools/rf_run_model.py
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from AITools.ai_utils import (
    gdf_from_zip_or_parts,
    gdf_from_db_with_geometry,
    get_provincial_code_from_schema,
)
from db import get_user_database_session

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)


@router.post("/run-saved-model")
async def run_saved_random_forest(
    model_file: UploadFile,
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
):
    """
    Run a saved Random Forest model bundle (.pkl) on new data.

    - Can use DB table (schema + table_name) or uploaded shapefile/ZIP.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, model_file.filename)
            with open(model_path, "wb") as f:
                f.write(await model_file.read())

            bundle = joblib.load(model_path)
            model = bundle["model"]
            scaler = bundle.get("scaler", None)
            features = [f.lower() for f in bundle.get("features", [])]
            print(f"✅ RF run: loaded model with features: {features}")

            # --------- Load data ----------
            if schema and table_name:
                gdf = gdf_from_db_with_geometry(schema, table_name)
            else:
                gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)

            if gdf is None or gdf.empty:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Input data is empty or could not be read."},
                )

            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
            df.columns = [c.lower() for c in df.columns]

            missing = [f for f in features if f not in df.columns]
            if missing:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Missing features in input: {missing}"},
                )

            X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

            if scaler is not None:
                X_arr = scaler.transform(X)
            else:
                X_arr = X.values

            preds = np.array(model.predict(X_arr)).flatten()
            gdf["prediction"] = preds

            # --------- Export outputs ----------
            run_id = f"rf_run_{np.random.randint(100000, 999999)}"
            out_dir = os.path.join(EXPORT_DIR, run_id)
            os.makedirs(out_dir, exist_ok=True)

            # shapefile
            shp_dir = os.path.join(out_dir, "predicted_shapefile")
            os.makedirs(shp_dir, exist_ok=True)
            shp_path = os.path.join(shp_dir, "rf_predicted_run.shp")
            gdf.to_file(shp_path)

            zip_out = os.path.join(out_dir, "predicted_output.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(shp_dir):
                    z.write(os.path.join(shp_dir, f), f)

            # simple PDF
            pdf_path = os.path.join(out_dir, "run_report.pdf")
            with PdfPages(pdf_path) as pp:
                fig, ax = plt.subplots(figsize=(7, 2))
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"Random Forest predictions completed\n{len(preds)} records processed.",
                    ha="center",
                    va="center",
                    fontsize=13,
                    weight="bold",
                )
                pp.savefig(fig)
                plt.close(fig)

            # Optional: save back to DB as <table_name>_PredictedRun
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
                    print(f"✅ RF run saved to {schema}.{out_table}")
                    engine.dispose()
                    db_session.close()
                except Exception as e:
                    print(f"⚠️ RF run: could not save to DB: {e}")

            base_url = "/api/random-forest/download"
            return {
                "message": "✅ Random Forest model run completed successfully.",
                "downloads": {
                    "report": f"{base_url}?file={pdf_path}",
                    "shapefile": f"{base_url}?file={zip_out}",
                },
                "record_count": int(len(gdf)),
                "isRunMode": True,
            }

    except Exception as e:
        import traceback

        print("❌ RF RUN ERROR:", e)
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
