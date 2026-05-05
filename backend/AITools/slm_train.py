# slm_train.py
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import geopandas as gpd
import pandas as pd
import numpy as np
import os, joblib, json, zipfile
from datetime import datetime, timezone, timedelta

PHT = timezone(timedelta(hours=8))

from AITools.slm_print_handler import export_slm_report_and_artifacts
from AITools.ai_utils import (
    extract_pin_column,
    compute_variable_distributions,
    upsert_pin_field,
    drop_duplicate_pin_fields,
    safe_to_float,
    df_from_db,
    gdf_from_db_with_geometry,
    gdf_from_zip_or_parts,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)


def build_artifact_base_name(model_used: str, table_name: str = "") -> str:
    now = datetime.now(PHT)
    base = f"{model_used}_{now.strftime('%Y-%b-%d_%I-%M-%S%p')}"
    if table_name and table_name.strip():
        base = f"{base}_{table_name.strip()}"
    return base


@router.post("/train")
async def train_slm_model(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    independent_vars: str = Form(...),
    dependent_var: str = Form(...),
    excluded_indices: Optional[str] = Form("[]"),
):
    try:
        file_gdf = None
        is_db_mode = False

        # ------------------------------------------------------------------
        # 1. Load data with geometry
        # ------------------------------------------------------------------
        if schema and schema.strip() and table_name and table_name.strip():
            is_db_mode = True
            print(f"📦 DB mode: schema={schema}, table={table_name}")
            gdf_full = gdf_from_db_with_geometry(schema.strip(), table_name.strip())
            df_full = pd.DataFrame(gdf_full.drop(columns="geometry", errors="ignore"))
            file_gdf = gdf_full.copy()
        else:
            print("📁 File mode detected")
            gdf_full = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
            file_gdf = gdf_full.copy()
            df_full = pd.DataFrame(gdf_full.drop(columns="geometry", errors="ignore"))

        print(f"   Loaded {len(df_full)} rows")

        # ------------------------------------------------------------------
        # 2. Apply exclusions
        # ------------------------------------------------------------------
        try:
            excluded = json.loads(excluded_indices or "[]")
            if excluded:
                print(f"🧹 Excluding {len(excluded)} rows...")
                df_full  = df_full.drop(df_full.index[excluded]).reset_index(drop=True)
                gdf_full = gdf_full.drop(gdf_full.index[excluded]).reset_index(drop=True)
        except Exception as e:
            print(f"⚠️ Could not parse excluded_indices: {e}")

        df_full["__original_index__"] = df_full.index

        # ------------------------------------------------------------------
        # 3. Parse fields
        # ------------------------------------------------------------------
        indep = json.loads(independent_vars) if independent_vars.startswith("[") \
            else [v.strip() for v in independent_vars.split(",")]
        indep  = [v for v in indep if v]
        target = dependent_var.strip()

        df_full.columns  = [c.lower() for c in df_full.columns]
        gdf_full.columns = [c.lower() if c != "geometry" else c for c in gdf_full.columns]

        pin_series, _ = extract_pin_column(df_full)
        indep  = [v.lower() for v in indep]
        target = target.lower()

        missing = [v for v in indep + [target] if v not in df_full.columns]
        if missing:
            return JSONResponse(status_code=400, content={"error": f"Missing variables: {missing}"})

        for col in indep + [target]:
            df_full[col] = df_full[col].map(safe_to_float)

        df_valid  = df_full.dropna(subset=indep + [target]).copy()
        if df_valid.empty:
            return JSONResponse(status_code=400, content={"error": "No valid numeric data found."})

        # Align geometry to valid rows
        gdf_valid = gdf_full.loc[df_valid.index].copy().reset_index(drop=True)
        df_valid  = df_valid.reset_index(drop=True)

        print(f"   Valid rows: {len(df_valid)}")

        # ------------------------------------------------------------------
        # 4. Build Queen contiguity spatial weights
        # ------------------------------------------------------------------
        print("🗺️ Building Queen contiguity spatial weights matrix...")
        try:
            from libpysal.weights import Queen
            w = Queen.from_dataframe(gdf_valid, silence_warnings=True)
            w.transform = "r"
            print(f"   {w.n} units, avg {w.mean_neighbors:.2f} neighbors")
        except Exception as we:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to build spatial weights: {str(we)}"}
            )

        # ------------------------------------------------------------------
        # 5. Train Spatial Lag Model
        # ------------------------------------------------------------------
        print("🚀 Training Spatial Lag Model (spreg.GM_Lag)...")
        try:
            from spreg import GM_Lag

            y_arr = df_valid[[target]].values.astype(np.float64)
            X_arr = df_valid[indep].values.astype(np.float64)

            print(f"   y shape: {y_arr.shape}, X shape: {X_arr.shape}")

            # Do NOT pass name_y / name_x — spreg uses them in f-string
            # formatting that crashes on numpy >= 2.0 with:
            # "unsupported format string passed to numpy.ndarray.__format__"
            model = GM_Lag(y_arr, X_arr, w=w)

            rho = float(np.asarray(model.rho).flat[0])
            print(f"   ρ (rho): {rho:.4f}")
        except Exception as me:
            import traceback as _tb
            print(f"❌ GM_Lag error: {type(me).__name__}: {me}")
            print(_tb.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": f"Spatial Lag Model training failed: {str(me)}"}
            )

        # ------------------------------------------------------------------
        # 6. Predictions & metrics
        # ------------------------------------------------------------------
        preds_full = model.predy.flatten()
        y_full     = model.y.flatten()
        residuals  = y_full - preds_full

        idx_all = np.arange(len(df_valid))
        idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=42)
        y_test     = y_full[idx_test]
        preds_test = preds_full[idx_test]

        r2        = float(r2_score(y_test, preds_test))
        rmse      = float(np.sqrt(mean_squared_error(y_test, preds_test)))
        mae       = float(mean_absolute_error(y_test, preds_test))
        mse       = float(mean_squared_error(y_test, preds_test))
        pseudo_r2 = float(model.pr2) if hasattr(model, "pr2") and model.pr2 is not None else r2
        rho       = float(np.asarray(model.rho).flat[0])

        # Moran's I on residuals
        moran_i = moran_p = None
        try:
            from esda.moran import Moran
            moran   = Moran(residuals, w, transformation="r", permutations=99)
            moran_i = float(moran.I)
            moran_p = float(moran.p_sim)
            print(f"   Moran's I: {moran_i:.4f} (p={moran_p:.4f})")
        except Exception as mi_err:
            print(f"⚠️ Moran's I failed: {mi_err}")

        metrics = {
            "r2": r2, "pseudo_r2": pseudo_r2,
            "mse": mse, "mae": mae, "rmse": rmse,
            "rho": rho, "moran_i": moran_i, "moran_p": moran_p,
        }

        # ------------------------------------------------------------------
        # 7. Coefficients table
        # ------------------------------------------------------------------
        betas      = model.betas.flatten()
        x_betas    = betas[1:-1] if len(betas) > len(indep) + 1 else betas[1:]
        std_errors = model.std_err.flatten() if hasattr(model, "std_err") and model.std_err is not None else np.zeros(len(betas))
        z_stats    = model.z_stat if hasattr(model, "z_stat") and model.z_stat is not None else []

        importance   = []
        coeff_table  = []
        for i, feat in enumerate(indep):
            beta_val = float(x_betas[i]) if i < len(x_betas) else 0.0
            se_val   = float(std_errors[i + 1]) if (i + 1) < len(std_errors) else 0.0
            z_val    = float(z_stats[i + 1][0]) if (i + 1) < len(z_stats) else 0.0
            p_val    = float(z_stats[i + 1][1]) if (i + 1) < len(z_stats) else 1.0
            importance.append({"feature": feat, "value": abs(beta_val)})
            coeff_table.append({
                "variable": feat, "coef": beta_val, "std_err": se_val,
                "z": z_val, "p": p_val, "significant": p_val < 0.05,
            })

        # ------------------------------------------------------------------
        # 8. Save model bundle
        # ------------------------------------------------------------------
        artifact_base = build_artifact_base_name("SLM", table_name or "")
        export_path   = os.path.join(EXPORT_DIR, artifact_base)
        os.makedirs(export_path, exist_ok=True)

        model_path = os.path.join(export_path, f"{artifact_base}.pkl")
        joblib.dump({
            "model": model, "w": w,
            "features": indep, "dependent_var": target,
            "model_type": "slm", "rho": rho,
            "trained_at": datetime.now(PHT).isoformat(),
        }, model_path)
        print(f"✅ Model saved: {os.path.basename(model_path)}")

        # ------------------------------------------------------------------
        # 9. PDF report + PNG charts
        # ------------------------------------------------------------------
        try:
            png_paths, pdf_path = export_slm_report_and_artifacts(
                export_path=export_path,
                artifact_base=artifact_base,
                model=model,
                indep=indep,
                target=target,
                y_full=y_full,
                preds_full=preds_full,
                residuals=residuals,
                metrics=metrics,
                coeff_table=coeff_table,
                rho=rho,
                moran_i=moran_i,
                moran_p=moran_p,
                df_valid=df_valid,
            )
        except Exception as rep_err:
            import traceback
            print(f"⚠️ Report generation failed: {rep_err}")
            traceback.print_exc()
            png_paths = {}
            pdf_path  = None

        # ------------------------------------------------------------------
        # 10. Shapefile output
        # ------------------------------------------------------------------
        safe_target_name = "actual_val" if len(target) > 10 else target
        zip_out = None
        try:
            gdf_out = gdf_valid.copy()
            gdf_out["prediction"]    = preds_full
            gdf_out[safe_target_name] = y_full
            gdf_out = gdf_out.drop(columns=["__original_index__"], errors="ignore")

            if pin_series is not None:
                try:
                    upsert_pin_field(gdf_out, pin_series.iloc[df_valid.index].values, preferred_name="PIN")
                    drop_duplicate_pin_fields(gdf_out, keep_name="PIN")
                except Exception:
                    pass

            shp_dir = os.path.join(export_path, "predicted_shapefile")
            os.makedirs(shp_dir, exist_ok=True)
            gdf_out.to_file(os.path.join(shp_dir, "predicted_output.shp"))

            zip_out = os.path.join(export_path, "predicted_output.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(shp_dir):
                    z.write(os.path.join(shp_dir, f), f)
            print(f"✅ Shapefile ZIP: {zip_out}")
        except Exception as shp_err:
            print(f"⚠️ Shapefile export failed: {shp_err}")

        # ------------------------------------------------------------------
        # 11. CSV + variable distributions + CAMA preview
        # ------------------------------------------------------------------
        csv_path = os.path.join(export_path, f"{artifact_base}.csv")
        csv_df   = df_valid[indep + [target]].copy()
        csv_df["prediction"] = preds_full
        if pin_series is not None:
            csv_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        csv_df.to_csv(csv_path, index=False)

        variable_distributions = compute_variable_distributions(df_valid, indep)

        preview_df = df_valid.copy()
        preview_df["prediction"] = preds_full
        if pin_series is not None:
            preview_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        preview_cols = (["PIN"] if pin_series is not None else []) + indep + [target, "prediction"]
        cama_preview = preview_df[preview_cols].head(100).to_dict("records")

        # ------------------------------------------------------------------
        # 12. Build response URLs
        # ------------------------------------------------------------------
        base_url  = "/api/ai-tools/download"
        plots     = {k: f"{base_url}?file={v}" for k, v in png_paths.items() if v}
        downloads = {
            "model":    f"{base_url}?file={model_path}",
            "cama_csv": f"{base_url}?file={csv_path}",
        }
        if pdf_path:
            downloads["report"] = f"{base_url}?file={pdf_path}"
        if zip_out:
            downloads["shapefile"] = f"{base_url}?file={zip_out}"
            downloads["geojson"]   = f"/api/ai-tools/preview-geojson?file_path={zip_out}"

        counts, bins = np.histogram(residuals, bins=20)
        bin_centers  = 0.5 * (bins[:-1] + bins[1:])

        return {
            "model_name":             artifact_base,
            "model_id":               artifact_base,
            "dependent_var":          safe_target_name,
            "original_dependent_var": target,
            "metrics":                {k: float(v) for k, v in metrics.items() if v is not None},
            "features":               indep,
            "importance":             importance,
            "coefficients":           coeff_table,
            "rho":                    rho,
            "moran_i":                moran_i,
            "moran_p":                moran_p,
            "interactive_data": {
                "residuals":       residuals.tolist(),
                "residual_bins":   bin_centers.tolist(),
                "residual_counts": counts.tolist(),
                "y_test":          y_test.tolist(),
                "preds":           preds_test.tolist(),
            },
            "variable_distributions": variable_distributions,
            "cama_preview":           cama_preview,
            "plots":                  plots,
            "downloads":              downloads,
            "is_db_mode":             is_db_mode,
            "isRunMode":              False,
            "record_count":           int(len(df_valid)),
        }

    except Exception as e:
        import traceback
        print(f"❌ SLM TRAIN ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})