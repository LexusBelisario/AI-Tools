from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Tuple, Dict, Any
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile, os, joblib, json, zipfile
from datetime import datetime, timezone, timedelta

PHT = timezone(timedelta(hours=8))  # Philippine Standard Time (UTC+8)
from scipy import stats
from AITools.lr_print_handler import export_full_report_and_artifacts
from sqlalchemy import text
from db import get_user_database_session
from AITools.ai_utils import (
    extract_pin_column,
    compute_variable_distributions,
    upsert_pin_field,
    drop_duplicate_pin_fields,
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


def get_provincial_code_from_schema(schema: str) -> str:
    if not schema:
        return ""
    return schema[:7] if len(schema) >= 7 else schema


GEOM_NAMES = {"geom", "geometry", "wkb_geometry", "the_geom"}


def safe_to_float(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            y = x.strip().replace(",", "")
            if y.lower() in ("", "none", "nan", "null"):
                return np.nan
            return float(y)
        return float(x)
    except Exception:
        return np.nan


def df_from_db(schema: str, table: str) -> pd.DataFrame:
    provincial_code = get_provincial_code_from_schema(schema)
    db_session = get_user_database_session(provincial_code)
    try:
        cols_rows = db_session.execute(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema=:s AND table_name=:t
                ORDER BY ordinal_position
            """),
            {"s": schema, "t": table},
        ).fetchall()
        colnames = [r[0] for r in cols_rows]
        keep = [c for c in colnames if c.lower() not in GEOM_NAMES]
        if not keep:
            keep_sql = "*"
        else:
            keep_sql = ", ".join(f'"{c}"' for c in keep)

        rows = db_session.execute(
            text(f'SELECT {keep_sql} FROM "{schema}"."{table}"')
        ).fetchall()

        if keep_sql == "*":
            df = pd.DataFrame(rows, columns=colnames)
            df = df[[c for c in df.columns if c.lower() not in GEOM_NAMES]]
        else:
            df = pd.DataFrame(rows, columns=keep)
        return df
    finally:
        db_session.close()


def gdf_from_db_with_geometry(schema: str, table: str) -> gpd.GeoDataFrame:
    provincial_code = get_provincial_code_from_schema(schema)
    db_session = get_user_database_session(provincial_code)
    engine = db_session.get_bind()

    try:
        geom_check = db_session.execute(
            text("""
                SELECT column_name, udt_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND table_name = :table
                  AND udt_name = 'geometry'
            """),
            {"schema": schema, "table": table}
        ).fetchone()

        if geom_check:
            print(f"{table} has geometry column: {geom_check[0]}")
            sql = f'SELECT * FROM "{schema}"."{table}"'
            gdf = gpd.read_postgis(sql, engine, geom_col=geom_check[0])
            return gdf

        print(f"{table} has no geometry. Searching for spatial tables in {schema}...")

        spatial_tables_rows = db_session.execute(
            text("""
                SELECT DISTINCT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND udt_name = 'geometry'
                  AND table_name != :table
                ORDER BY table_name
            """),
            {"schema": schema, "table": table}
        ).fetchall()

        if not spatial_tables_rows:
            raise ValueError(f"No spatial tables found in schema '{schema}' to join with {table}")

        spatial_tables = {row[0]: row[1] for row in spatial_tables_rows}
        print(f"Found spatial tables: {list(spatial_tables.keys())}")

        target_cols_rows = db_session.execute(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ordinal_position
            """),
            {"schema": schema, "table": table}
        ).fetchall()
        target_cols = {row[0].upper() for row in target_cols_rows}

        join_key_candidates = [
            "PIN", "ARPN", "ARP_PIN", "TD_NO", "PROPERTY_ID",
            "PARCEL_ID", "TAX_DEC_NO", "OID", "OBJECTID", "FID"
        ]

        for spatial_table, geom_col in spatial_tables.items():
            spatial_cols_rows = db_session.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :table
                """),
                {"schema": schema, "table": spatial_table}
            ).fetchall()
            spatial_cols = {row[0].upper() for row in spatial_cols_rows}

            common_keys = []
            for key in join_key_candidates:
                if key in target_cols and key in spatial_cols:
                    common_keys.append(key)

            if common_keys:
                join_key = common_keys[0]
                print(f"Found join key '{join_key}' between {table} and {spatial_table}")

                join_sql = f'''
                    SELECT
                        t.*,
                        s."{geom_col}" as geometry
                    FROM "{schema}"."{table}" t
                    INNER JOIN "{schema}"."{spatial_table}" s
                      ON t."{join_key}" = s."{join_key}"
                '''

                print(f"Joining {table} with {spatial_table} on {join_key}")
                gdf = gpd.read_postgis(join_sql, engine, geom_col="geometry")
                print(f"Successfully joined. Result has {len(gdf)} rows")
                return gdf

        raise ValueError(
            f"Could not find a suitable spatial table to join with '{table}'. "
            f"Available spatial tables: {list(spatial_tables.keys())}. "
            f"Common join keys not found."
        )

    finally:
        engine.dispose()
        db_session.close()


def gdf_from_zip_or_parts(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
) -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = None
        if zip_file is not None:
            zpath = os.path.join(tmpdir, zip_file.filename)
            with open(zpath, "wb") as f:
                f.write(zip_file.file.read())
            with zipfile.ZipFile(zpath, "r") as z:
                z.extractall(tmpdir)
            for root, _, files in os.walk(tmpdir):
                for fn in files:
                    if fn.lower().endswith(".shp"):
                        shp_path = os.path.join(root, fn)
                        break
                if shp_path:
                    break
        elif shapefiles:
            for uf in shapefiles:
                with open(os.path.join(tmpdir, uf.filename), "wb") as f:
                    f.write(uf.file.read())
            for fn in os.listdir(tmpdir):
                if fn.lower().endswith(".shp"):
                    shp_path = os.path.join(tmpdir, fn)
                    break
        if not shp_path:
            raise ValueError("No .shp file found.")
        gdf = gpd.read_file(shp_path)
        return gdf


@router.post("/train")
async def train_linear_regression(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    independent_vars: str = Form(...),
    dependent_var: str = Form(...),
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
                df_full = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

            emit("Dropping excluded rows and NaN values")
            try:
                excluded = json.loads(excluded_indices or "[]")
                if excluded:
                    df_full = df_full.drop(df_full.index[excluded]).reset_index(drop=True)
            except Exception as e:
                print(f"Could not parse excluded_indices: {e}")

            df_full["__original_index__"] = df_full.index

            if independent_vars.startswith("["):
                indep = json.loads(independent_vars)
            else:
                indep = [v.strip() for v in independent_vars.split(",")]
            indep = [v for v in indep if v]
            target = dependent_var.strip()

            lower_map = {c.lower(): c for c in df_full.columns}
            missing = [v for v in indep + [target] if v.lower() not in lower_map]
            if missing:
                return {"error": f"Missing variables: {missing}"}

            df_full.columns = [c.lower() for c in df_full.columns]
            pin_series, pin_colname = extract_pin_column(df_full)
            indep = [v.lower() for v in indep]
            target = target.lower()

            for col in indep + [target]:
                df_full[col] = df_full[col].map(safe_to_float)

            df_valid = df_full.dropna(subset=indep + [target])
            if df_valid.empty:
                return {"error": "No valid numeric data found."}

            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split

            emit("Splitting into 70/30 train and test sets")
            X = df_valid[indep]
            y = df_valid[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            emit("Fitting StandardScaler on training data")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            emit("Fitting LinearRegression model")
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            emit("Predicting on test set, computing residuals")
            preds = model.predict(X_test_scaled)
            residuals = y_test - preds

            emit("Computing R\u00b2, MSE, MAE, RMSE")

            artifact_base = build_artifact_base_name("LR", table_name or "")
            export_path = os.path.join(EXPORT_DIR, artifact_base)
            os.makedirs(export_path, exist_ok=True)

            emit("Saving model to .pkl file")
            model_path = os.path.join(export_path, f"{artifact_base}.pkl")
            joblib.dump(
                {
                    "model": model,
                    "scaler": scaler,
                    "features": [v.lower() for v in indep],
                    "dependent_var": target.lower(),
                    "model_type": "lr",
                    "trained_at": datetime.now(PHT).isoformat(),
                },
                model_path,
            )

            emit("Generating PDF report and plots")
            metrics, png_paths, t_tests, pdf_path = export_full_report_and_artifacts(
                export_path, model, scaler, indep, target,
                X_train_scaled, y_train, X_test_scaled, y_test, preds, residuals,
                X_train_unscaled=X_train, artifact_base=artifact_base,
            )

            emit("Writing predictions to CSV")
            preds_valid = model.predict(scaler.transform(df_valid[indep]))
            df_valid = df_valid.copy()
            df_valid["prediction"] = preds_valid
            safe_target_name = "actual_val" if len(target) > 10 else target
            csv_path = os.path.join(export_path, f"{artifact_base}.csv")
            csv_df = df_valid[indep + [target, "prediction"]].copy()
            if pin_series is not None:
                csv_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
            csv_df.to_csv(csv_path, index=False)

            emit("Building predicted shapefile and ZIP")
            zip_out = None
            try:
                original_indices = df_valid["__original_index__"].tolist()
                if is_db_mode:
                    gdf_db = gdf_from_db_with_geometry(schema, table_name)
                    valid_gdf = gdf_db.iloc[original_indices].copy()
                    columns_to_drop = [col for col in valid_gdf.columns if col.upper() in ("UNIT_VALUE", "MARKET_VAL")]
                    if columns_to_drop:
                        valid_gdf = valid_gdf.drop(columns=columns_to_drop, errors="ignore")
                    if pin_series is not None:
                        upsert_pin_field(valid_gdf, pin_series.iloc[original_indices].values, preferred_name="PIN")
                        drop_duplicate_pin_fields(valid_gdf, keep_name="PIN")
                    valid_gdf[safe_target_name] = df_valid[target].values
                    valid_gdf["prediction"] = df_valid["prediction"].values
                elif file_gdf is not None:
                    valid_gdf = file_gdf.iloc[original_indices].copy()
                    columns_to_drop = [col for col in valid_gdf.columns if col.upper() in ("UNIT_VALUE", "MARKET_VAL")]
                    if columns_to_drop:
                        valid_gdf = valid_gdf.drop(columns=columns_to_drop, errors="ignore")
                    if pin_series is not None:
                        upsert_pin_field(valid_gdf, pin_series.iloc[original_indices].values, preferred_name="PIN")
                        drop_duplicate_pin_fields(valid_gdf, keep_name="PIN")
                    valid_gdf[safe_target_name] = df_valid[target].values
                    valid_gdf["prediction"] = df_valid["prediction"].values
                else:
                    raise ValueError("No geometry source")

                shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
                os.makedirs(shp_pred_dir, exist_ok=True)
                valid_gdf = valid_gdf.drop(columns=["__original_index__"], errors="ignore")
                valid_gdf.to_file(os.path.join(shp_pred_dir, "predicted_output.shp"))
                zip_out = os.path.join(export_path, "predicted_output.zip")
                with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in os.listdir(shp_pred_dir):
                        z.write(os.path.join(shp_pred_dir, f), f)
            except Exception as e:
                print(f"Shapefile export error: {e}")

            counts, bins = np.histogram(residuals, bins=20)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            variable_distributions = compute_variable_distributions(df_valid, indep)

            preview_df = df_valid.copy()
            if pin_series is not None:
                preview_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
            preview_cols = (["PIN"] if pin_series is not None else []) + indep + [target, "prediction"]
            cama_preview = preview_df[preview_cols].head(100).to_dict("records")

            base_url = "/api/ai-tools/download"
            plots = {key: f"{base_url}?file={path}" for key, path in png_paths.items()}
            downloads = {
                "model": f"{base_url}?file={model_path}",
                "report": f"{base_url}?file={pdf_path}",
                "cama_csv": f"{base_url}?file={csv_path}",
            }
            if zip_out:
                downloads["shapefile"] = f"{base_url}?file={zip_out}"
                downloads["geojson"] = f"/api/ai-tools/preview-geojson?file_path={zip_out}"

            return {
                "model_name": artifact_base,
                "model_id": artifact_base,
                "dependent_var": safe_target_name,
                "original_dependent_var": target,
                "metrics": {k: float(v) for k, v in metrics.items()},
                "features": indep,
                "importance": [{"feature": feat, "value": float(val)} for feat, val in zip(indep, model.coef_)],
                "coefficients": {k: float(v) for k, v in zip(indep, model.coef_)},
                "intercept": float(model.intercept_),
                "t_test": t_tests,
                "interactive_data": {
                    "residuals": residuals.tolist(),
                    "residual_bins": bin_centers.tolist(),
                    "residual_counts": counts.tolist(),
                    "y_test": y_test.tolist(),
                    "preds": preds.tolist(),
                },
                "variable_distributions": variable_distributions,
                "cama_preview": cama_preview,
                "plots": plots,
                "downloads": downloads,
                "is_db_mode": is_db_mode,
                "message": "Model trained successfully.",
            }

        except Exception as e:
            import traceback
            print(f"TRAIN ERROR: {e}")
            print(traceback.format_exc())
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