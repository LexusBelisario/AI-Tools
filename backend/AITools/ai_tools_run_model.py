from fastapi import APIRouter, UploadFile, Form, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import pickle

import os
import tempfile
import joblib
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import io
import re
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sqlalchemy import text

from AITools.ai_utils import (
    gdf_from_zip_or_parts,
)
from common_db_runtime import (
    resolve_common_context_from_token,
    set_request_context,
    clear_request_context,
    get_request_session,
)

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)

def _normalize_blob_to_bytes(blob):
    if blob is None:
        return None
    if isinstance(blob, memoryview):
        return blob.tobytes()
    if isinstance(blob, bytearray):
        return bytes(blob)
    return blob


def _load_from_blob(blob_bytes: bytes):
    # try joblib then pickle
    try:
        return joblib.load(io.BytesIO(blob_bytes))
    except Exception:
        return pickle.loads(blob_bytes)


def _parse_features(features_json):
    if not features_json:
        return []
    # jsonb from postgres usually already list, but minsan string
    if isinstance(features_json, list):
        return features_json
    if isinstance(features_json, str):
        try:
            return json.loads(features_json)
        except Exception:
            return []
    return []

def _extract_bearer_token(authorization: str) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization must be: Bearer <token>")
    return parts[1].strip()


def _validate_ident(name: str, kind: str) -> str:
    if not name:
        raise HTTPException(status_code=400, detail=f"Missing {kind}")
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        raise HTTPException(status_code=400, detail=f"Invalid {kind}: {name}")
    return name


# ==========================================================
# DB HELPERS
# ==========================================================
def _ensure_numeric_column(engine, schema: str, table_name: str, col_name: str):
    """Add a double precision column to the table if it doesn't exist yet."""
    q = """
   DO $$
   BEGIN
       IF NOT EXISTS (
           SELECT 1
           FROM information_schema.columns
           WHERE table_schema = :schema
             AND table_name   = :table_name
             AND column_name  = :col_name
        ) THEN
           EXECUTE format(
                'ALTER TABLE %I.%I ADD COLUMN %I double precision',
                :schema, :table_name, :col_name
            );
       END IF;
   END $$;
    """
    with engine.begin() as conn:
        conn.execute(text(q), {"schema": schema, "table_name": table_name, "col_name": col_name})


def _update_predictions_by_pin(
    engine,
    schema: str,
    table_name: str,
    pin_col_in_db: str,
    pred_col_in_db: str,
    pin_pred_df: pd.DataFrame,
) -> int:
    """
   Write prediction values back to the target table matched by PIN.
   Uses a temporary staging table to avoid row-by-row updates.
    """
    staging = f"__pred_staging_{np.random.randint(100000, 999999)}"

    df = pin_pred_df.copy()
    df.columns = ["pin", "prediction"]
    df["pin"] = df["pin"].astype(str)
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["pin"]).drop_duplicates(subset=["pin"], keep="first")

    df.to_sql(staging, con=engine, schema=schema, if_exists="replace", index=False, method="multi")

    # Quote identifiers safely
    ip   = engine.dialect.identifier_preparer
    qs   = ip.quote(schema)
    qt   = ip.quote(table_name)
    qpin = ip.quote(pin_col_in_db)
    qpred = ip.quote(pred_col_in_db)
    qst  = ip.quote(staging)

    update_sql = f"""
       UPDATE {qs}.{qt} t
          SET {qpred} = s.prediction
         FROM {qs}.{qst} s
        WHERE lower(t.{qpin}::text) = lower(s.pin::text)
    """
    drop_sql = f"DROP TABLE IF EXISTS {qs}.{qst}"

    rowcount = 0
    try:
        with engine.begin() as conn:
            res = conn.execute(text(update_sql))
            rowcount = int(res.rowcount or 0)
    finally:
        with engine.begin() as conn:
            conn.execute(text(drop_sql))

    return rowcount


def _load_gdf_from_common_db(db_session, schema: str, table_name: str) -> gpd.GeoDataFrame:
    """Load GeoDataFrame from Common Database table"""
    engine = db_session.get_bind()
    
    # First check if table exists
    exists = db_session.execute(
        text("""
           SELECT 1 FROM information_schema.tables
           WHERE table_schema = :schema AND table_name = :table_name
        """),
        {"schema": schema, "table_name": table_name}
    ).fetchone()
    
    if not exists:
        raise HTTPException(
            status_code=404,
            detail=f"Table '{schema}.{table_name}' not found in Common Database"
        )
    
    # Find geometry column
    geom_cols = db_session.execute(
        text("""
           SELECT column_name
           FROM information_schema.columns
           WHERE table_schema = :schema
             AND table_name = :table_name
             AND udt_name IN ('geometry', 'geography')
        """),
        {"schema": schema, "table_name": table_name}
    ).fetchall()
    
    geom_col = geom_cols[0][0] if geom_cols else None
    
    # Load data
    query = f'SELECT * FROM "{schema}"."{table_name}"'
    
    if geom_col:
        gdf = gpd.read_postgis(query, engine, geom_col=geom_col)
    else:
        # No geometry column, load as DataFrame then convert
        df = pd.read_sql(query, engine)
        gdf = gpd.GeoDataFrame(df)
    
    return gdf


# ==========================================================
# UNIFIED RUN-SAVED-MODEL (Using Common Database)
# ==========================================================
@router.post("/run-saved-model")
async def run_saved_model(
    model_source: str = Form(...),  # "upload" or "db"
    model_file: Optional[UploadFile] = None,
    model_id: Optional[int] = Form(None),
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    authorization: str = Header(default=""),
    x_target_schema: str = Header(default="", alias="X-Target-Schema"),
    x_target_db: str = Header(default="", alias="X-Target-DB"),
):
    """
   Unified Run-Saved-Model endpoint that:
      1. Loads model from UPLOAD or COMMON DATABASE
      2. Loads input data from DB or file
      3. Runs predictions
      4. Exports PDF + shapefile
      5. Optionally updates DB table by PIN
    """
    
    print("=" * 60)
    print(" RUN SAVED MODEL ENDPOINT")
    print(f" Model Source: {model_source}")
    print(f" Model ID: {model_id}")
    print(f" Schema: {schema}")
    print(f" Table: {table_name}")
    print(f" Has Authorization: {bool(authorization)}")
    print("=" * 60)
    
    token = _extract_bearer_token(authorization)
    
    ctx = resolve_common_context_from_token(
        token,
        db_override=(x_target_db or None),
        schema_override=(x_target_schema or schema),
    )
    
    print(f" Resolved context: {ctx}")
    
    set_request_context(ctx)
    db = None
    
    try:
        target_schema = _validate_ident(ctx["schema"], "schema")
        db = get_request_session()
        
        # ------------------------------------------------------
        # 1. Load Model (from upload or Common DB)
        # ------------------------------------------------------
        model_bundle = None
        model_type = None
        
        if model_source == "upload":
            if not model_file:
                raise HTTPException(status_code=400, detail="model_file required for upload source")
            
            print(" Loading model from uploaded file...")
            content = await model_file.read()
            model_bundle = joblib.load(io.BytesIO(content))
            
        elif model_source == "db":
            if not model_id:
                raise HTTPException(status_code=400, detail="model_id required for db source")
            
            print(f" Loading model ID {model_id} from Common Database...")
            
            row = db.execute(
                text(f'''
                   SELECT 
                        model_blob,
                        model_name,
                        model_type,
                        model_version,
                        dependent_var,
                        features
                   FROM "{target_schema}"."ai_trained_models"
                   WHERE id = :model_id
                '''),
                {"model_id": model_id}
            ).fetchone()

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model ID {model_id} not found in {target_schema}.ai_trained_models"
                )

            blob, name, mtype, version, dep_var, features_json = row

            if not blob:
                raise HTTPException(status_code=404, detail="Model blob is empty")

            blob_bytes = _normalize_blob_to_bytes(blob)
            print(f" BLOB size: {len(blob_bytes)} bytes")

            loaded_obj = _load_from_blob(blob_bytes)

            db_features = _parse_features(features_json)
            db_dep = dep_var

            # If raw estimator lang yung PKL, wrap it
            if isinstance(loaded_obj, dict) and ("model" in loaded_obj):
                model_bundle = loaded_obj
            else:
                model_bundle = {
                    "model": loaded_obj,
                    "scaler": None,
                    "features": db_features,
                    "dependent_var": db_dep,
                    "model_type": mtype,
                }

            # Make sure may fallback pa rin kahit dict pero missing keys
            if not model_bundle.get("features"):
                model_bundle["features"] = db_features

            if not model_bundle.get("dependent_var"):
                model_bundle["dependent_var"] = db_dep

            model_type = mtype or model_bundle.get("model_type", "unknown")

            print(f" Loaded model: {name} (v{version}) - {str(model_type).upper()}")

        else:
            raise HTTPException(status_code=400, detail="Invalid model_source. Use 'upload' or 'db'")
        
        # Extract model components
        model = model_bundle["model"]
        scaler = model_bundle.get("scaler", None)
        features = [f.lower() for f in model_bundle.get("features", [])]
        target = model_bundle.get("dependent_var", None)
        
        if not model_type:
            model_type = model_bundle.get("model_type", "unknown")
        
        print(f" Model type: {model_type}")
        print(f" Features: {features}")
        print(f" Dependent: {target}")

        # ------------------------------------------------------
        # 2. Load Input Data
        # DB mode: always predict from Training_Table, write back to LandParcel
        # File mode: predict from uploaded shapefile
        # ------------------------------------------------------
        db_mode = bool(schema and table_name)

        if db_mode:
            # Always use Training_Table as prediction source (has the trained features)
            training_table = "Training_Table"
            print(f" Loading prediction input from Common DB: {schema}.{training_table}")
            gdf = _load_gdf_from_common_db(db, schema, training_table)
            print(f" Loaded {len(gdf)} rows from {schema}.{training_table}")
        else:
            print(" Loading input from uploaded shapefile")
            gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)

        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        df.columns = [c.lower() for c in df.columns]

        print(f" DataFrame columns: {list(df.columns)[:10]}...")

        # ------------------------------------------------------
        # 2b. Ensure required features exist
        # ------------------------------------------------------
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f" Missing features: {missing}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Missing required fields in {table_name}: {missing}"},
            )

        X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        print(f" Feature matrix shape: {X.shape}")

        # ------------------------------------------------------
        # 3. Predict
        # ------------------------------------------------------
        if scaler is not None:
            X_transformed = scaler.transform(X)
            preds = model.predict(X_transformed)
        else:
            preds = model.predict(X)

        preds = np.array(preds).flatten()
        gdf["prediction"] = preds

        print(f" Prediction completed: {len(preds)} rows")
        print(f" Prediction range: {np.min(preds):.2f} - {np.max(preds):.2f}")

        # ------------------------------------------------------
        # 3b. Optional actual/residual stats
        # ------------------------------------------------------
        actual_field = None
        actual_values = None
        actual_range = None
        residual_stats = None

        if target:
            target_lower = str(target).lower()

            actual_field_name = None
            for col in df.columns:
                if col.lower() == target_lower:
                    actual_field_name = col
                    break

            if actual_field_name:
                actual_field = actual_field_name
                actual_values = pd.to_numeric(df[actual_field], errors="coerce").values

                valid_mask = (~np.isnan(actual_values)) & (~np.isnan(preds))
                valid_actuals = actual_values[valid_mask]
                valid_preds_for_residual = preds[valid_mask]

                if len(valid_actuals) > 0:
                    actual_range = {
                        "min": float(np.min(valid_actuals)),
                        "max": float(np.max(valid_actuals)),
                    }

                    residuals = valid_actuals - valid_preds_for_residual
                    residual_stats = {
                        "min": float(np.min(residuals)),
                        "max": float(np.max(residuals)),
                        "mean": float(np.mean(residuals)),
                        "mae": float(np.mean(np.abs(residuals))),
                        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
                    }
                    print(f" Residual MAE: {residual_stats['mae']:.2f}")
            else:
                print(f" Note: Dependent variable '{target}' not found in input data")

        valid_preds = preds[~np.isnan(preds)]
        if len(valid_preds) > 0:
            prediction_range = {
                "min": float(np.min(valid_preds)),
                "max": float(np.max(valid_preds)),
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
            for fname in os.listdir(shp_dir):
                z.write(os.path.join(shp_dir, fname), fname)

        print(f" Exported shapefile: {zip_out}")

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

        print(f" Exported PDF: {pdf_path}")

        # ------------------------------------------------------
        # 7. Write predictions back to LandParcel by PIN (DB mode only)
        # Prediction source: Training_Table
        # Write-back target: LandParcel
        # ------------------------------------------------------
        db_updated_rows = None
        landparcel_table = "LandParcel"
        if db_mode:
            try:
                engine = db.get_bind()

                # Find PIN column in Training_Table data (case-insensitive)
                pin_col_in_input = next(
                    (c for c in df.columns if c.lower() == "pin"), None
                )

                if not pin_col_in_input:
                    print(" PIN column not found in Training_Table - skipping DB write-back.")
                else:
                    # Resolve actual PIN column name in LandParcel (case-insensitive)
                    pin_cols = db.execute(
                        text("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = :schema
                              AND table_name   = :table_name
                        """),
                        {"schema": schema, "table_name": landparcel_table},
                    ).fetchall()
                    pin_col_in_db = next(
                        (r[0] for r in pin_cols if r[0].lower() == "pin"), None
                    )

                    if not pin_col_in_db:
                        print(f" PIN column not found in {landparcel_table} - skipping DB write-back.")
                    else:
                        pred_col_in_db = "prediction"
                        _ensure_numeric_column(engine, schema, landparcel_table, pred_col_in_db)

                        pin_pred_df = pd.DataFrame({
                            "pin":        df[pin_col_in_input].astype(str),
                            "prediction": preds,
                        })

                        db_updated_rows = _update_predictions_by_pin(
                            engine=engine,
                            schema=schema,
                            table_name=landparcel_table,
                            pin_col_in_db=pin_col_in_db,
                            pred_col_in_db=pred_col_in_db,
                            pin_pred_df=pin_pred_df,
                        )

                        print(f" Updated {db_updated_rows} rows in {schema}.{landparcel_table} (prediction column)")

            except Exception as db_err:
                print(f" DB write-back failed (non-fatal): {db_err}")

        # ------------------------------------------------------
        # 8. Response
        # ------------------------------------------------------
        base_url = "/api/ai-tools/download"

        response_data = {
            "message": "Model run successful",
            "model_type": model_type,
            "model_source": model_source,
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

        if db_mode:
            response_data["db_target"] = f"{schema}.{table_name}"
            response_data["db_updated_rows"] = db_updated_rows
            response_data["db_prediction_column"] = "prediction"

        if actual_field:
            response_data["actual_field"] = actual_field
            response_data["actual_range"] = actual_range
            response_data["residual_stats"] = residual_stats

        print("=" * 60)
        print(" RUN SAVED MODEL COMPLETE")
        print("=" * 60)

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("=" * 60)
        print(" RUN-SAVED-MODEL ERROR:", e)
        print(traceback.format_exc())
        print("=" * 60)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    
    finally:
        if db is not None:
            db.close()
        clear_request_context()