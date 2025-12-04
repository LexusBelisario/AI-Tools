# ==========================================================
# ai_tools_run_model.py (Unified Run-Saved-Model Router)
# ==========================================================

from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional

import os
import tempfile
import joblib
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from AITools.ai_utils import (
    gdf_from_db_with_geometry,
    gdf_from_zip_or_parts,
    get_provincial_code_from_schema,
)
from db import get_user_database_session

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)


# ==========================================================
# 🔥 UNIFIED RUN-SAVED-MODEL
# ==========================================================
@router.post("/run-saved-model")
async def run_saved_model(
    model_type: str = Form(...),            # "lr" | "rf" | "xgb"
    model_file: UploadFile = Form(...),
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
):
    """
    Unified Run-Saved-Model endpoint for:
      • Linear Regression
      • Random Forest  
      • XGBoost

    Behavior:
      • Load saved model file (.pkl)
      • Load input (DB OR shapefile)
      • Predict
      • Export PDF + shapefile ZIP
      • Optionally save output to DB
    """

    try:
        # ------------------------------------------------------
        # 1. Save model file to temporary path
        # ------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, model_file.filename)
            with open(model_path, "wb") as f:
                f.write(await model_file.read())

            # Load bundle from joblib
            bundle = joblib.load(model_path)
            model = bundle["model"]
            scaler = bundle.get("scaler", None)
            features = [f.lower() for f in bundle.get("features", [])]
            target = bundle.get("dependent_var", None)

            print(f"🔍 Loaded model type: {model_type}")
            print(f"🔍 Features: {features}")
            print(f"🔍 Dependent: {target}")

            # ------------------------------------------------------
            # 2. Load input (DB or file)
            # ------------------------------------------------------
            if schema and table_name:
                print(f"🗄 Input: DB table {schema}.{table_name}")
                gdf = gdf_from_db_with_geometry(schema, table_name)
            else:
                print("📁 Input: Uploaded shapefile")
                gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)

            # DataFrame version without geometry
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
            df.columns = [c.lower() for c in df.columns]

            # Ensure required features exist
            missing = [f for f in features if f not in df.columns]
            if missing:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Missing required fields: {missing}"},
                )

            # Numeric conversion
            X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

            # ------------------------------------------------------
            # 3. Predict depending on model type
            # ------------------------------------------------------
            if scaler:
                X_transformed = scaler.transform(X)
                preds = model.predict(X_transformed)
            else:
                preds = model.predict(X)

            preds = np.array(preds).flatten()
            gdf["prediction"] = preds

            print(f"✔ Prediction completed: {len(preds)} rows")

            # ✅ Case-insensitive: Use the dependent variable from training as actual field
            actual_field = None
            actual_values = None
            actual_range = None
            residual_stats = None
            
            # Check if the dependent variable exists in the input data (case-insensitive)
            if target:
                target_lower = target.lower()
                
                # ✅ Find the field with case-insensitive matching
                actual_field_name = None
                for col in df.columns:
                    if col.lower() == target_lower:
                        actual_field_name = col  # Keep original case from data
                        break
                
                if actual_field_name:
                    actual_field = actual_field_name
                    # Convert to numeric and handle NaN properly
                    actual_values = pd.to_numeric(df[actual_field], errors='coerce').values
                    
                    # Filter to only valid pairs (non-NaN in both actual and predicted)
                    valid_mask = (~np.isnan(actual_values)) & (~np.isnan(preds))
                    valid_actuals = actual_values[valid_mask]
                    valid_preds_for_residual = preds[valid_mask]
                    
                    valid_count = len(valid_actuals)
                    
                    if valid_count > 0:
                        print(f"✅ Using dependent variable as actual field: {actual_field}")
                        print(f"   (Matched '{target}' with '{actual_field}' in data)")
                        print(f"   Valid values: {valid_count} / {len(actual_values)}")
                        
                        # Calculate actual range
                        actual_range = {
                            "min": float(np.min(valid_actuals)),
                            "max": float(np.max(valid_actuals))
                        }
                        
                        # Calculate residual statistics
                        residuals = valid_actuals - valid_preds_for_residual
                        residual_stats = {
                            "min": float(np.min(residuals)),
                            "max": float(np.max(residuals)),
                            "mean": float(np.mean(residuals)),
                            "mae": float(np.mean(np.abs(residuals))),
                            "rmse": float(np.sqrt(np.mean(residuals**2)))
                        }
                        print(f"✅ Residual stats calculated from {len(residuals)} valid pairs")
                        print(f"   MAE: {residual_stats['mae']:.2f}")
                        print(f"   RMSE: {residual_stats['rmse']:.2f}")
                    else:
                        print(f"⚠️ Field {actual_field} exists but has no valid values")
                        actual_field = None
                        actual_values = None
                else:
                    print(f"⚠️ Dependent variable '{target}' not found in input data (case-insensitive search)")
                    print(f"   Available columns: {df.columns.tolist()}")
            else:
                print(f"⚠️ No dependent variable stored in model bundle")
            
            # ✅ Calculate prediction range (handle NaN)
            valid_preds = preds[~np.isnan(preds)]
            if len(valid_preds) > 0:
                prediction_range = {
                    "min": float(np.min(valid_preds)),
                    "max": float(np.max(valid_preds))
                }
            else:
                prediction_range = {"min": 0.0, "max": 0.0}
            
            # ------------------------------------------------------
            # 4. Create export folder
            # ------------------------------------------------------
            export_id = f"run_{model_type}_{np.random.randint(100000, 999999)}"
            export_path = os.path.join(EXPORT_DIR, export_id)
            os.makedirs(export_path, exist_ok=True)

            # ------------------------------------------------------
            # 5. Export shapefile ZIP
            # ------------------------------------------------------
            shp_dir = os.path.join(export_path, "predicted_shapefile")
            os.makedirs(shp_dir, exist_ok=True)

            shp_path = os.path.join(shp_dir, "predicted_output.shp")
            gdf.to_file(shp_path)

            zip_out = os.path.join(export_path, "predicted_output.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(shp_dir):
                    z.write(os.path.join(shp_dir, f), f)

            # ------------------------------------------------------
            # 6. Export simple PDF summary
            # ------------------------------------------------------
            pdf_path = os.path.join(export_path, "run_report.pdf")

            with PdfPages(pdf_path) as pp:
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.axis("off")
                ax.text(
                    0.5,
                    0.6,
                    f"Run-Saved-Model: {model_type.upper()}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    weight="bold",
                )
                ax.text(
                    0.5,
                    0.35,
                    f"Total predictions: {len(preds)} records",
                    ha="center",
                    va="center",
                    fontsize=11,
                )
                pp.savefig(fig)
                plt.close(fig)

            # ------------------------------------------------------
            # 7. Save to DB (optional)
            # ------------------------------------------------------
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

                    print(f"✔ Output saved to DB: {schema}.{out_table}")

                    engine.dispose()
                    db_session.close()

                except Exception as db_err:
                    print(f"⚠ Could not save predictions to DB: {db_err}")

            # ------------------------------------------------------
            # 8. Response (uses unified downloads API)
            # ------------------------------------------------------
            base_url = "/api/ai-tools/download"

            response_data = {
                "message": "Model run successful",
                "model_type": model_type,
                "record_count": int(len(preds)),
                "isRunMode": True,
                "prediction_field": "prediction",
                "prediction_range": prediction_range,
                "downloads": {
                    "report": f"{base_url}?file={pdf_path}",
                    "shapefile": f"{base_url}?file={zip_out}",
                    "shapefile_raw": shp_path,
                },
            }

            # ✅ Add actual field info if available
            if actual_field:
                response_data["actual_field"] = actual_field
                response_data["actual_range"] = actual_range
                response_data["residual_stats"] = residual_stats

            return response_data
            
    except Exception as e:
        import traceback
        print("❌ RUN-SAVED-MODEL ERROR:", e)
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
