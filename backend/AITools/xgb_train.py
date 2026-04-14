# xgb_train.py
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Tuple, Dict, Any
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile, os, pickle, json, zipfile
from AITools.xgb_print_handler import export_xgb_report_and_artifacts
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timezone, timedelta

PHT = timezone(timedelta(hours=8))  # Philippine Standard Time (UTC+8)
from sqlalchemy import text
from db import get_user_database_session
from AITools.ai_utils import (
    GEOM_NAMES,
    get_provincial_code_from_schema,
    safe_to_float,
    df_from_db,
    gdf_from_db_with_geometry,
    gdf_from_zip_or_parts,
    extract_pin_column,
    compute_variable_distributions,
    upsert_pin_field, 
    drop_duplicate_pin_fields
)


router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)

def build_artifact_base_name(model_used: str, table_name: str = "") -> str:
    now = datetime.now(PHT)
    base = f"{model_used}_{now.strftime('%Y-%b-%d_%I-%M-%S%p')}"
    if table_name and table_name.strip():
        base = f"{base}_{table_name.strip()}"
    return base

def wrap_plot_urls(plots: Dict[str, Optional[str]], prefix: str) -> Dict[str, Optional[str]]:
    return {
        key: (f"{prefix}?file={path}" if path else None)
        for key, path in plots.items()
    }


@router.post("/train")
async def train_xgb_model(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    independent_vars: str = Form(...),
    dependent_var: str = Form(...),
    scaler_choice: str = Form("None"),
    excluded_indices: Optional[str] = Form("[]"),
):
    import asyncio
    from fastapi.responses import StreamingResponse

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def emit(msg: str):
        import json as _json
        loop.call_soon_threadsafe(queue.put_nowait, f"event: progress\ndata: {_json.dumps(msg)}\n\n")

    def run_training_sync():
        try:
            file_gdf = None
            is_db_mode = False

            if schema and schema.strip() and table_name and table_name.strip():
                is_db_mode = True
                emit("Loading data from database")
                df_full = df_from_db(schema.strip(), table_name.strip())
            else:
                emit("Loading shapefile")
                gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
                file_gdf = gdf.copy()
                df_full = gdf.drop(columns=[c for c in gdf.columns if c.lower() in GEOM_NAMES])

            if df_full.empty:
                return {"error": "No data loaded."}

            emit("Dropping excluded rows and NaN values")
            try:
                excluded = json.loads(excluded_indices or "[]")
                if excluded:
                    df_full = df_full.drop(df_full.index[excluded]).reset_index(drop=True)
            except Exception:
                pass

            df_full["__original_index__"] = df_full.index
            pin_series, pin_colname = extract_pin_column(df_full)
            if pin_colname and pin_colname in df_full.columns:
                df_full.drop(columns=[pin_colname], inplace=True)

            indep = json.loads(independent_vars) if isinstance(independent_vars, str) else independent_vars
            target = dependent_var

            if target not in df_full.columns:
                return {"error": f"'{target}' not found"}
            for col in indep:
                if col not in df_full.columns:
                    return {"error": f"'{col}' not found"}

            for col in indep + [target]:
                df_full[col] = df_full[col].map(safe_to_float)

            df_model = df_full[indep + [target] + ["__original_index__"]].copy()
            df_model = df_model.dropna(subset=indep + [target])

            if df_model.empty:
                return {"error": "No valid rows"}

            emit("Splitting into 80/20 train and test sets")
            X = df_model[indep].values
            y = df_model[target].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            emit("Fitting scaler on training data")
            scaler = None
            if scaler_choice == "Standard":
                scaler = StandardScaler()
            elif scaler_choice == "MinMax":
                scaler = MinMaxScaler()

            if scaler is not None:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            emit("Fitting XGBRegressor model")
            model = XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                objective="reg:squarederror",
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            emit("Computing R\u00b2, MSE, MAE, RMSE")
            df_valid = df_model.copy()
            artifact_base = build_artifact_base_name("XGB", table_name or "")
            export_path = os.path.join(EXPORT_DIR, artifact_base)
            os.makedirs(export_path, exist_ok=True)

            emit("Generating PDF report and plots")
            metrics, png_paths, pdf_path = export_xgb_report_and_artifacts(
                export_path=export_path, model=model, scaler=scaler,
                feature_names=indep, target=target,
                X_train=X_train_scaled, y_train=y_train,
                X_test=X_test_scaled, y_test=y_test, y_pred=y_pred,
                df_valid=df_valid, scaler_choice=scaler_choice, artifact_base=artifact_base,
            )

            emit("Saving model to .pkl file")
            model_path = os.path.join(export_path, f"{artifact_base}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(
                    {"model": model, "scaler": scaler, "features": indep, "target": target,
                     "model_type": "xgb", "trained_at": datetime.now(PHT).isoformat()},
                    f,
                )

            emit("Writing predictions to CSV")
            preds_valid = model.predict(scaler.transform(df_valid[indep]) if scaler else df_valid[indep].values)
            df_export = df_valid.copy()
            if pin_series is not None:
                try:
                    df_export["PIN"] = pin_series.iloc[df_export.index].values
                except Exception:
                    pass
            df_export["prediction"] = preds_valid
            csv_cols = (["PIN"] if "PIN" in df_export.columns else []) + indep + [target, "prediction"]
            csv_path = os.path.join(export_path, f"{artifact_base}.csv")
            df_export[csv_cols].to_csv(csv_path, index=False)

            emit("Building predicted shapefile and ZIP")
            zip_out = None
            try:
                original_indices = df_valid["__original_index__"].tolist() if "__original_index__" in df_valid.columns else df_valid.index.tolist()
                if is_db_mode:
                    gdf_db = gdf_from_db_with_geometry(schema, table_name)
                    valid_gdf = gdf_db.iloc[original_indices].copy()
                    if pin_series is not None:
                        upsert_pin_field(valid_gdf, pin_series.iloc[original_indices].values)
                    drop_duplicate_pin_fields(valid_gdf)
                    valid_gdf[target] = df_valid[target].values
                    valid_gdf["prediction"] = df_export["prediction"].values
                elif file_gdf is not None:
                    valid_gdf = file_gdf.iloc[original_indices].copy()
                    if pin_series is not None:
                        try:
                            valid_gdf["PIN"] = pin_series.iloc[original_indices].values
                        except Exception:
                            pass
                    valid_gdf[target] = df_valid[target].values
                    valid_gdf["prediction"] = df_export["prediction"].values
                else:
                    raise ValueError("No geometry source available")

                shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
                os.makedirs(shp_pred_dir, exist_ok=True)
                valid_gdf = valid_gdf.drop(columns=["__original_index__"], errors="ignore")
                valid_gdf.to_file(os.path.join(shp_pred_dir, "XGBoost_Predicted.shp"))
                zip_out = os.path.join(export_path, "XGBoost_Predicted.zip")
                with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                    for fname in os.listdir(shp_pred_dir):
                        z.write(os.path.join(shp_pred_dir, fname), fname)
            except Exception as e:
                print(f"Shapefile export error: {e}")

            residuals = y_test - y_pred
            counts, bin_edges = np.histogram(residuals, bins=20)
            variable_distributions = compute_variable_distributions(
                pd.DataFrame(X_train_scaled if scaler else X_train, columns=indep), indep
            )

            preview_df = df_valid.copy()
            preview_df["prediction"] = model.predict(scaler.transform(df_valid[indep]) if scaler else df_valid[indep].values)
            if pin_series is not None:
                try:
                    preview_df["PIN"] = pin_series.iloc[preview_df.index].values
                except Exception:
                    pass
            preview_cols = (["PIN"] if "PIN" in preview_df.columns else []) + indep + [target, "prediction"]
            cama_preview = preview_df[preview_cols].head(100).to_dict("records")

            base_url = "/api/ai-tools/download"
            wrapped_plots = {k: (f"{base_url}?file={v}" if v else None) for k, v in png_paths.items()}
            wrapped_downloads = {k: f"{base_url}?file={v}" for k, v in {"model": model_path, "report": pdf_path, "cama_csv": csv_path}.items() if v}
            if zip_out:
                wrapped_downloads["shapefile"] = f"{base_url}?file={zip_out}"
                wrapped_downloads["geojson"] = f"/api/ai-tools/preview-geojson?file_path={zip_out}"

            return {
                "model_name": artifact_base, "model_id": artifact_base,
                "message": "XGBoost training completed",
                "dependent_var": target, "metrics": metrics, "features": indep,
                "importance": [{"feature": f, "value": float(v)} for f, v in zip(indep, model.feature_importances_)] if hasattr(model, "feature_importances_") else [],
                "interactive_data": {"residuals": residuals.tolist(), "residual_bins": bin_edges.tolist(), "residual_counts": counts.tolist(), "y_test": y_test.tolist(), "preds": y_pred.tolist()},
                "variable_distributions": variable_distributions, "cama_preview": cama_preview,
                "plots": wrapped_plots, "downloads": wrapped_downloads,
                "is_db_mode": is_db_mode, "isRunMode": False, "record_count": int(len(df_model)),
            }

        except Exception as e:
            import traceback
            print(f"❌ XGB TRAIN ERROR: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    async def event_stream():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    async def producer():
        import asyncio, contextvars
        ctx = contextvars.copy_context()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, ctx.run, run_training_sync)
        import json as _json
        queue.put_nowait(f"event: result\ndata: {_json.dumps(result)}\n\n")
        queue.put_nowait(None)

    asyncio.create_task(producer())
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
        },
    )