# rf_train.py
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile
import os
import json
import zipfile
from datetime import datetime, timezone, timedelta

PHT = timezone(timedelta(hours=8))  # Philippine Standard Time (UTC+8)
from AITools.rf_print_handler import export_rf_report_and_artifacts

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sqlalchemy import text
from db import get_user_database_session

from AITools.ai_utils import (
    GEOM_NAMES,
    get_provincial_code_from_schema,
    safe_to_float,
    df_from_db,
    gdf_from_db_with_geometry,
    gdf_from_zip_or_parts,
    compute_variable_distributions,
    extract_pin_column,
    upsert_pin_field,
    drop_duplicate_pin_fields,
)

import joblib

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)

def build_artifact_base_name(model_used: str, table_name: str = "") -> str:
    now = datetime.now(PHT)
    base = f"{model_used}_{now.strftime('%Y-%b-%d_%I-%M-%S%p')}"
    if table_name and table_name.strip():
        base = f"{base}_{table_name.strip()}"
    return base

def _wrap_download_urls(paths: Dict[str, Optional[str]], base_url: str) -> Dict[str, Optional[str]]:
    # same pattern used sa ibang trainers: /download?file=...
    out = {}
    for k, p in paths.items():
        out[k] = f"{base_url}?file={p}" if p else None
    return out


@router.post("/train")
async def train_rf_model(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    independent_vars: str = Form(...),
    dependent_var: str = Form(...),
    excluded_indices: Optional[str] = Form("[]"),
):
    """
    LR-style flow:
    1) detect input mode + load
    2) parse + validate fields
    3) numeric convert + dropna
    4) excluded rows handling
    5) split + scale (RF computation)
    6) train + evaluate
    7) export artifacts (pdf/plots/csv/shp zip/db)
    8) return response (same shape expectation)
    """
    try:
        # ===========================
        # 1) INPUT MODE DETECTION
        # ===========================
        file_gdf = None
        is_db_mode = False

        if schema and schema.strip() and table_name and table_name.strip():
            is_db_mode = True
            print(f"✅ RF DB mode: schema={schema}, table={table_name}")
            df_full = df_from_db(schema.strip(), table_name.strip())
        else:
            print("✅ RF File mode")
            gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
            file_gdf = gdf.copy()
            df_full = gdf.drop(columns=[c for c in gdf.columns if str(c).lower() in GEOM_NAMES], errors="ignore")

        if df_full is None or df_full.empty:
            return JSONResponse(status_code=400, content={"error": "No data loaded."})

        # ===========================
        # 2) PARSE VARIABLES
        # ===========================
        indep = json.loads(independent_vars) if isinstance(independent_vars, str) else independent_vars
        target = dependent_var

        if target not in df_full.columns:
            return JSONResponse(status_code=400, content={"error": f"Dependent variable '{target}' not found in data."})

        for col in indep:
            if col not in df_full.columns:
                return JSONResponse(status_code=400, content={"error": f"Independent variable '{col}' not found in data."})

        # ===========================
        # 3) STORE ORIGINAL INDICES (LR-style)
        # ===========================
        df_full = df_full.copy()
        df_full["__orig_index__"] = df_full.index
        print(f"📍 Stored original indices for {len(df_full)} rows")

        # ===========================
        # 4) PIN HANDLING (remove from features, keep for preview/csv)
        # ===========================
        pin_series, pin_colname = extract_pin_column(df_full)
        if pin_colname and pin_colname in df_full.columns:
            if pin_colname in indep:
                indep = [c for c in indep if c != pin_colname]
                print(f"   🔧 Removed PIN column '{pin_colname}' from training features")

        # ===========================
        # 5) NUMERIC CLEANUP
        # ===========================
        for col in indep + [target]:
            df_full[col] = df_full[col].map(safe_to_float)

        # ===========================
        # 6) DROP NANS (selected only)
        # ===========================
        df_model = df_full[indep + [target, "__orig_index__"]].copy()
        before = len(df_model)
        df_model = df_model.dropna(subset=indep + [target])
        after = len(df_model)
        print(f"🔢 RF dropped {before - after} rows with NaNs in selected columns.")

        if df_model.empty:
            return JSONResponse(status_code=400, content={"error": "No valid rows after cleaning."})

        # ===========================
        # 7) EXCLUDED ROWS (LR-style using __orig_index__)
        # ===========================
        try:
            excluded_list = json.loads(excluded_indices) if excluded_indices else []
            if not isinstance(excluded_list, list):
                excluded_list = []
        except Exception:
            excluded_list = []

        excluded_list = [
            int(i) for i in excluded_list
            if isinstance(i, int) or (isinstance(i, str) and i.isdigit())
        ]

        if excluded_list:
            mask = df_model["__orig_index__"].isin(excluded_list)
            excluded_count = int(mask.sum())
            df_model = df_model[~mask].copy()
            print(f"✅ RF: excluding {excluded_count} rows before training...")
        else:
            print("✅ RF: no excluded rows received.")

        if df_model.empty:
            return JSONResponse(status_code=400, content={"error": "All rows were excluded. Nothing to train."})

        print(f"📊 Final training dataset: {len(df_model)} rows")

        # ===========================
        # 8) SPLIT + SCALE (RF computation aligned to Tkinter)
        # ===========================
        X = df_model[indep].values
        y = df_model[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ===========================
        # 9) TRAIN RF (actual RF computation)
        # ===========================
        print("🚀 Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,   # same default as Tkinter reference
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        # 10) df_valid (valid rows, keep original indices)
        valid_indices = df_model["__orig_index__"].values
        df_valid = df_full.loc[valid_indices].copy()
        df_valid = df_valid[indep + [target]].copy()

        # 11) EXPORT ARTIFACTS (keep everything)
        artifact_base = build_artifact_base_name("RF", table_name or "")
        export_path = os.path.join(EXPORT_DIR, artifact_base)
        os.makedirs(export_path, exist_ok=True)
        print(f"📦 Creating export folder: {artifact_base}")

        try:
            metrics, png_paths, pdf_path = export_rf_report_and_artifacts(
                export_path=export_path,
                model=model,
                scaler=scaler,
                feature_names=indep,
                target=target,
                X_train=X_train_scaled,
                y_train=y_train,
                X_test=X_test_scaled,
                y_test=y_test,
                y_pred=y_pred,
                df_valid=df_valid,
                artifact_base=artifact_base,
            )
        except Exception as e:
            import traceback
            print(f"❌ RF report generation failed: {e}")
            print(traceback.format_exc())
            raise

        plots = png_paths

        # Save model
        model_path = os.path.join(export_path, f"{artifact_base}.pkl")
        try:
            joblib.dump(
                {
                    "model": model,
                    "scaler": scaler,
                    "features": indep,
                    "target": target,
                    "model_type": "rf",
                    "trained_at": datetime.now(PHT).isoformat(),
                },
                model_path,
                compress=3,
            )
            print(f"Saved model: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"❌ Failed to save RF model: {e}")
            raise

        # Export CSV
        df_export = df_valid.copy()
        pin_series_export, _ = extract_pin_column(df_full)
        if pin_series_export is not None:
            try:
                df_export["PIN"] = pin_series_export.iloc[df_export.index].values
            except Exception as e:
                print(f"⚠ PIN injection failed: {e}")
        X_valid = df_valid[indep].values
        if scaler is not None:
            X_valid = scaler.transform(X_valid)
        df_export["prediction"] = model.predict(X_valid)
        csv_cols = []
        if "PIN" in df_export.columns:
            csv_cols.append("PIN")
        csv_cols.extend(indep)
        csv_cols.extend([target, "prediction"])
        csv_path = os.path.join(export_path, f"{artifact_base}.csv")
        df_export[csv_cols].to_csv(csv_path, index=False)
        print(f"Exported CSV: {csv_path}")

        # Export shapefile
        zip_out = None
        try:
            valid_indices = df_valid.index.tolist()
            if is_db_mode:
                gdf_db = gdf_from_db_with_geometry(schema, table_name)
                valid_gdf = gdf_db.iloc[valid_indices].copy()
                if pin_series is not None:
                    upsert_pin_field(valid_gdf, pin_series.iloc[valid_indices].values)
                drop_duplicate_pin_fields(valid_gdf)
                valid_gdf["prediction"] = df_export["prediction"].values
            elif file_gdf is not None:
                valid_gdf = file_gdf.iloc[valid_indices].copy()
                if pin_series is not None:
                    try:
                        valid_gdf["PIN"] = pin_series.iloc[valid_indices].values
                    except Exception as e:
                        print(f"⚠️ Could not add PIN: {e}")
                valid_gdf["prediction"] = df_export["prediction"].values
            else:
                raise ValueError("No geometry source available")

            shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
            os.makedirs(shp_pred_dir, exist_ok=True)
            shp_pred_path = os.path.join(shp_pred_dir, "RandomForest_Predicted.shp")
            valid_gdf.to_file(shp_pred_path)
            zip_out = os.path.join(export_path, "RandomForest_Predicted.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for fname in os.listdir(shp_pred_dir):
                    z.write(os.path.join(shp_pred_dir, fname), fname)
            print(f"ZIP created: {zip_out}")
        except Exception as e:
            print(f"⚠️ Shapefile export error: {e}")

        downloads = {
            "model":    model_path,
            "report":   pdf_path,
            "cama_csv": csv_path,
            "shapefile": zip_out,
        }

        # 12) INTERACTIVE DATA (same payload concept)
        residuals = y_test - y_pred
        counts, bin_edges = np.histogram(residuals, bins=20)
        residual_bins = bin_edges.tolist()
        residual_counts = counts.tolist()

        # 13) VARIABLE DISTRIBUTIONS (keep)
        print("📊 Computing variable distributions for RF...")
        variable_distributions = compute_variable_distributions(
            df_model[indep].copy(),
            indep
        )
        print(f"✅ Computed distributions for {len(variable_distributions)} variables")

        # ===========================
        # 14) PREVIEW (keep)
        # ===========================
        print("📋 Creating training result preview...")
        preview_df = df_valid.copy()

        preds_valid = model.predict(scaler.transform(df_valid[indep].values))
        preview_df["prediction"] = preds_valid

        if pin_series is not None:
            try:
                preview_df["PIN"] = pin_series.iloc[preview_df.index].values
                print("   ✅ Added PIN to preview")
            except Exception as e:
                print(f"   ⚠️ Could not add PIN to preview: {e}")

        preview_cols = []
        if "PIN" in preview_df.columns:
            preview_cols.append("PIN")
        preview_cols.extend(indep)
        preview_cols.append(target)
        preview_cols.append("prediction")

        cama_preview = preview_df[preview_cols].head(100).to_dict("records")
        print(f"   ✅ Created preview with {len(cama_preview)} rows")

        # ===========================
        # 15) WRAP URLS (same style)
        # ===========================
        base_url = "/api/ai-tools/download"
        wrapped_plots = _wrap_download_urls(plots, base_url)
        wrapped_downloads = _wrap_download_urls(downloads, base_url)

        if "shapefile" in downloads and downloads["shapefile"]:
            shp_path = downloads["shapefile"]
            wrapped_downloads["geojson"] = f"/api/ai-tools/preview-geojson?file_path={shp_path}"

        # ===========================
        # 16) RETURN RESPONSE (LR-like feel)
        # ===========================
        return {
            "model_name": artifact_base,
            "model_id": artifact_base,
            "message": "Random Forest training completed successfully.",
            "dependent_var": target,
            "metrics": metrics,
            "features": indep,
            "importance": [
                {"feature": feat, "value": float(val)}
                for feat, val in zip(indep, model.feature_importances_)
            ] if hasattr(model, "feature_importances_") else [],
            "interactive_data": {
                "residuals": residuals.tolist(),
                "residual_bins": residual_bins,
                "residual_counts": residual_counts,
                "y_test": y_test.tolist(),
                "preds": y_pred.tolist(),
            },
            "variable_distributions": variable_distributions,
            "cama_preview": cama_preview,
            "plots": wrapped_plots,
            "downloads": wrapped_downloads,
            "is_db_mode": is_db_mode,
            "isRunMode": False,
            "record_count": int(len(df_model)),
        }

    except Exception as e:
        import traceback
        print(f"❌ RF TRAIN ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})