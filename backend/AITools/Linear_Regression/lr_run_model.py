from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import numpy as np
import tempfile, os, joblib, zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from db import get_user_database_session
from .lr_train import gdf_from_db_with_geometry, gdf_from_zip_or_parts, get_provincial_code_from_schema

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")

@router.post("/run-saved-model")
async def run_saved_model_unified(
    model_file: UploadFile,
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # --- Load model bundle ---
            model_path = os.path.join(tmpdir, model_file.filename)
            with open(model_path, "wb") as f:
                f.write(await model_file.read())

            bundle = joblib.load(model_path)
            model = bundle["model"]
            scaler = bundle.get("scaler", None)
            features = [f.lower() for f in bundle.get("features", [])]
            print(f"✅ Loaded model with features: {features}")

            # --- Detect input source ---
            if schema and table_name:
                # ========== DATABASE MODE ==========
                print(f"🗄️ Running model on database: {schema}.{table_name}")
                gdf = gdf_from_db_with_geometry(schema, table_name)
            else:
                # ========== FILE MODE ==========
                print(f"📂 Running model on shapefile/zip input")
                gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)

            # --- Prepare dataframe ---
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
            from .lr_train import extract_pin_column
            pin_series, pin_colname = extract_pin_column(df)

            df.columns = [c.lower() for c in df.columns]

            missing = [f for f in features if f not in df.columns]
            if missing:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Missing features in input: {missing}"},
                )

            X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

            # Compute predictions
            if scaler is not None:
                preds = model.predict(scaler.transform(X))
            else:
                preds = model.predict(X)

            # --- RESTORE PIN BEFORE SAVING ---
            if pin_series is not None:
                gdf["PIN"] = pin_series.values

            gdf["prediction"] = preds

            # --- Export results ---
            export_path = os.path.join(EXPORT_DIR, f"run_{np.random.randint(100000, 999999)}")
            os.makedirs(export_path, exist_ok=True)

            shp_dir = os.path.join(export_path, "predicted_shapefile")
            os.makedirs(shp_dir, exist_ok=True)
            shp_path = os.path.join(shp_dir, "predicted_output.shp")
            gdf.to_file(shp_path)

            zip_out = os.path.join(export_path, "predicted_output.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(shp_dir):
                    z.write(os.path.join(shp_dir, f), f)

            pdf_path = os.path.join(export_path, "run_report.pdf")
            with PdfPages(pdf_path) as pp:
                fig, ax = plt.subplots(figsize=(7, 2))
                ax.axis("off")
                ax.text(
                    0.5, 0.5,
                    f"Predictions completed\n{len(preds)} records processed.",
                    ha="center", va="center", fontsize=13, weight="bold",
                )
                pp.savefig(fig); plt.close(fig)

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
                        index=False
                    )
                    print(f"✅ Saved to {schema}.{out_table}")
                    engine.dispose()
                    db_session.close()
                except Exception as e:
                    print(f"⚠️ Could not save to DB: {e}")

            base_url = "/api/linear-regression/download"
            return {
                "message": "✅ Model run completed successfully.",
                "downloads": {
                    "report": f"{base_url}?file={pdf_path}",
                    "shapefile": f"{base_url}?file={zip_out}",
                },
                "record_count": len(gdf),
            }

    except Exception as e:
        import traceback
        print("❌ RUN ERROR:", e)
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})