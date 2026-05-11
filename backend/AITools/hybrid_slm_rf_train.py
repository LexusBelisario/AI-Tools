# hybrid_slm_rf_train.py
"""
Hybrid Spatial Lag Model + Random Forest
Stage 1: Fit SLM → get ŷ_SLM and ε_SLM
Stage 2: Fit RF on ε_SLM using original X features
Final:   ŷ_hybrid = ŷ_SLM + ε̂_RF

Returns two separate result objects:
  - slm_stage:    SLM-only results (artifact base: SLM_...)
  - hybrid_stage: Full hybrid results (artifact base: H_SLM_RF_...)
"""
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import geopandas as gpd
import pandas as pd
import numpy as np
import os, joblib, json, zipfile
from datetime import datetime, timezone, timedelta

PHT = timezone(timedelta(hours=8))

from AITools.hybrid_slm_rf_print_handler import export_hybrid_report_and_artifacts
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)


def build_artifact_base_names(table_name: str = "") -> tuple:
    """Returns (slm_base, hybrid_base) with consistent timestamp."""
    now = datetime.now(PHT)
    ts = now.strftime('%Y-%b-%d_%I-%M-%S%p')
    suffix = f"_{table_name.strip()}" if table_name and table_name.strip() else ""
    slm_base    = f"SLM_{ts}{suffix}"
    hybrid_base = f"H_SLM_RF_{ts}{suffix}"
    return slm_base, hybrid_base


@router.post("/train")
async def train_hybrid_model(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    independent_vars: str = Form(...),
    dependent_var: str = Form(...),
    excluded_indices: Optional[str] = Form("[]"),
):
    try:
        is_db_mode = False

        # ------------------------------------------------------------------
        # 1. Load data with geometry
        # ------------------------------------------------------------------
        if schema and schema.strip() and table_name and table_name.strip():
            is_db_mode = True
            print(f"📦 DB mode: schema={schema}, table={table_name}")
            gdf_full = gdf_from_db_with_geometry(schema.strip(), table_name.strip())
            df_full  = pd.DataFrame(gdf_full.drop(columns="geometry", errors="ignore"))
        else:
            print("📁 File mode detected")
            gdf_full = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
            df_full  = pd.DataFrame(gdf_full.drop(columns="geometry", errors="ignore"))

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
        # 5. Stage 1 — Spatial Lag Model
        # ------------------------------------------------------------------
        print("🚀 Stage 1: Fitting Spatial Lag Model...")
        try:
            from spreg import GM_Lag
            y_arr = df_valid[[target]].values.astype(np.float64)
            X_arr = df_valid[indep].values.astype(np.float64)
            slm   = GM_Lag(y_arr, X_arr, w=w)
            rho   = float(np.asarray(slm.rho).flat[0])
            print(f"   ρ (rho): {rho:.4f}")
        except Exception as me:
            import traceback as _tb
            print(f"❌ SLM error: {me}\n{_tb.format_exc()}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Stage 1 (SLM) failed: {str(me)}"}
            )

        pred_slm = slm.predy.flatten()
        y_full   = slm.y.flatten()
        eps_slm  = y_full - pred_slm          # residuals → RF target

        print(f"   SLM predictions ready. Residual mean: {eps_slm.mean():.4f}")

        # SLM coefficients
        betas      = slm.betas.flatten()
        x_betas    = betas[1:-1] if len(betas) > len(indep) + 1 else betas[1:]
        std_errors = slm.std_err.flatten() if hasattr(slm, "std_err") and slm.std_err is not None else np.zeros(len(betas))
        z_stats    = slm.z_stat if hasattr(slm, "z_stat") and slm.z_stat is not None else []

        slm_coefficients = []
        for i, feat in enumerate(indep):
            beta_val = float(x_betas[i]) if i < len(x_betas) else 0.0
            se_val   = float(std_errors[i + 1]) if (i + 1) < len(std_errors) else 0.0
            z_val    = float(z_stats[i + 1][0]) if (i + 1) < len(z_stats) else 0.0
            p_val    = float(z_stats[i + 1][1]) if (i + 1) < len(z_stats) else 1.0
            slm_coefficients.append({
                "variable": feat, "coef": beta_val, "std_err": se_val,
                "z": z_val, "p": p_val, "significant": p_val < 0.05,
            })

        # Moran's I on SLM residuals
        moran_i_slm = moran_p_slm = None
        try:
            from esda.moran import Moran
            moran_slm   = Moran(eps_slm, w, transformation="r", permutations=99)
            moran_i_slm = float(moran_slm.I)
            moran_p_slm = float(moran_slm.p_sim)
            print(f"   Moran's I on SLM residuals: {moran_i_slm:.4f} (p={moran_p_slm:.4f})")
        except Exception as mi_err:
            print(f"⚠️ Moran's I (SLM residuals) failed: {mi_err}")

        # ------------------------------------------------------------------
        # 6. Stage 2 — Random Forest on SLM residuals
        # ------------------------------------------------------------------
        print("🌲 Stage 2: Fitting Random Forest on SLM residuals...")
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(X_arr, eps_slm)

        pred_rf_correction = rf.predict(X_arr)
        print(f"   RF correction ready. Mean correction: {pred_rf_correction.mean():.4f}")

        # ------------------------------------------------------------------
        # 7. Final hybrid prediction
        # ------------------------------------------------------------------
        pred_hybrid   = pred_slm + pred_rf_correction
        residuals_hybrid = y_full - pred_hybrid

        # Feature importance from RF stage
        importance = sorted(
            [{"feature": f, "value": float(v)} for f, v in zip(indep, rf.feature_importances_)],
            key=lambda x: x["value"], reverse=True
        )

        # ------------------------------------------------------------------
        # 8. Metrics (train/test split on indices)
        # ------------------------------------------------------------------
        idx_all = np.arange(len(df_valid))
        idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=42)

        y_test             = y_full[idx_test]
        preds_hybrid_test  = pred_hybrid[idx_test]
        preds_slm_test     = pred_slm[idx_test]

        r2_hybrid   = float(r2_score(y_test, preds_hybrid_test))
        rmse_hybrid = float(np.sqrt(mean_squared_error(y_test, preds_hybrid_test)))
        mae_hybrid  = float(mean_absolute_error(y_test, preds_hybrid_test))
        mse_hybrid  = float(mean_squared_error(y_test, preds_hybrid_test))

        r2_slm   = float(r2_score(y_test, preds_slm_test))
        rmse_slm = float(np.sqrt(mean_squared_error(y_test, preds_slm_test)))

        pseudo_r2 = float(slm.pr2) if hasattr(slm, "pr2") and slm.pr2 is not None else r2_slm

        # Moran's I on hybrid residuals
        moran_i_hybrid = moran_p_hybrid = None
        try:
            from esda.moran import Moran
            moran_hyb      = Moran(residuals_hybrid, w, transformation="r", permutations=99)
            moran_i_hybrid = float(moran_hyb.I)
            moran_p_hybrid = float(moran_hyb.p_sim)
            print(f"   Moran's I on hybrid residuals: {moran_i_hybrid:.4f} (p={moran_p_hybrid:.4f})")
        except Exception as mi2:
            print(f"⚠️ Moran's I (hybrid residuals) failed: {mi2}")

        metrics = {
            # Hybrid (final)
            "r2":        r2_hybrid,
            "rmse":      rmse_hybrid,
            "mae":       mae_hybrid,
            "mse":       mse_hybrid,
            # Stage 1 (SLM)
            "r2_slm":    r2_slm,
            "rmse_slm":  rmse_slm,
            "pseudo_r2": pseudo_r2,
            "rho":       rho,
            "moran_i_slm":    moran_i_slm,
            "moran_p_slm":    moran_p_slm,
            # Stage 3 (Hybrid diagnostics)
            "moran_i_hybrid": moran_i_hybrid,
            "moran_p_hybrid": moran_p_hybrid,
        }
        print(f"   Hybrid R²={r2_hybrid:.4f}, RMSE={rmse_hybrid:.2f} | SLM R²={r2_slm:.4f}")

        # ------------------------------------------------------------------
        # 9. Save artifacts — two separate export paths
        # ------------------------------------------------------------------
        safe_target_name = "actual_val" if len(target) > 10 else target
        slm_base, hybrid_base = build_artifact_base_names(table_name or "")

        slm_export_path    = os.path.join(EXPORT_DIR, slm_base)
        hybrid_export_path = os.path.join(EXPORT_DIR, hybrid_base)
        os.makedirs(slm_export_path, exist_ok=True)
        os.makedirs(hybrid_export_path, exist_ok=True)

        # SLM-only model bundle
        slm_model_path = os.path.join(slm_export_path, f"{slm_base}.pkl")
        joblib.dump({
            "slm": slm,
            "features": indep, "dependent_var": target,
            "model_type": "slm", "rho": rho,
            "trained_at": datetime.now(PHT).isoformat(),
        }, slm_model_path)
        print(f"✅ SLM model bundle saved: {slm_base}")

        # Hybrid model bundle
        hybrid_model_path = os.path.join(hybrid_export_path, f"{hybrid_base}.pkl")
        joblib.dump({
            "slm": slm, "rf": rf,
            "features": indep, "dependent_var": target,
            "model_type": "hybrid_slm_rf", "rho": rho,
            "trained_at": datetime.now(PHT).isoformat(),
        }, hybrid_model_path)
        print(f"✅ Hybrid model bundle saved: {hybrid_base}")

        # ------------------------------------------------------------------
        # 10. PDF report + PNGs (hybrid report only — SLM stage uses slm_print_handler)
        # ------------------------------------------------------------------
        try:
            png_paths, pdf_path = export_hybrid_report_and_artifacts(
                export_path=hybrid_export_path,
                artifact_base=hybrid_base,
                indep=indep,
                target=target,
                y_full=y_full,
                pred_slm=pred_slm,
                pred_rf_correction=pred_rf_correction,
                pred_hybrid=pred_hybrid,
                residuals_hybrid=residuals_hybrid,
                metrics=metrics,
                slm_coefficients=slm_coefficients,
                importance=importance,
                rho=rho,
                moran_i_slm=moran_i_slm,
                moran_p_slm=moran_p_slm,
                moran_i_hybrid=moran_i_hybrid,
                moran_p_hybrid=moran_p_hybrid,
                df_valid=df_valid,
            )
        except Exception as rep_err:
            import traceback
            print(f"⚠️ Report generation failed: {rep_err}")
            traceback.print_exc()
            png_paths = {}
            pdf_path  = None

        # ------------------------------------------------------------------
        # 11a. Shapefile — SLM only (pred_slm)
        # ------------------------------------------------------------------
        slm_zip_out = None
        try:
            gdf_slm = gdf_valid.copy()
            gdf_slm[safe_target_name] = y_full
            gdf_slm["prediction"]     = pred_slm
            gdf_slm["slm_residual"]   = eps_slm
            gdf_slm = gdf_slm.drop(columns=["__original_index__"], errors="ignore")
            if pin_series is not None:
                try:
                    upsert_pin_field(gdf_slm, pin_series.iloc[df_valid.index].values, preferred_name="PIN")
                    drop_duplicate_pin_fields(gdf_slm, keep_name="PIN")
                except Exception:
                    pass
            slm_shp_dir = os.path.join(slm_export_path, "predicted_shapefile")
            os.makedirs(slm_shp_dir, exist_ok=True)
            gdf_slm.to_file(os.path.join(slm_shp_dir, "predicted_output.shp"))
            slm_zip_out = os.path.join(slm_export_path, "predicted_output.zip")
            with zipfile.ZipFile(slm_zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(slm_shp_dir):
                    z.write(os.path.join(slm_shp_dir, f), f)
            print(f"✅ SLM shapefile ZIP saved")
        except Exception as shp_err:
            print(f"⚠️ SLM shapefile export failed: {shp_err}")

        # ------------------------------------------------------------------
        # 11b. Shapefile — Hybrid (pred_slm + pred_slm_rf)
        # ------------------------------------------------------------------
        hybrid_zip_out = None
        try:
            gdf_hybrid = gdf_valid.copy()
            gdf_hybrid[safe_target_name]  = y_full
            gdf_hybrid["slm_pred"]        = pred_slm
            gdf_hybrid["slm_residual"]    = eps_slm
            gdf_hybrid["rf_correction"]   = pred_rf_correction
            gdf_hybrid["prediction"]      = pred_hybrid
            gdf_hybrid = gdf_hybrid.drop(columns=["__original_index__"], errors="ignore")
            if pin_series is not None:
                try:
                    upsert_pin_field(gdf_hybrid, pin_series.iloc[df_valid.index].values, preferred_name="PIN")
                    drop_duplicate_pin_fields(gdf_hybrid, keep_name="PIN")
                except Exception:
                    pass
            hybrid_shp_dir = os.path.join(hybrid_export_path, "predicted_shapefile")
            os.makedirs(hybrid_shp_dir, exist_ok=True)
            gdf_hybrid.to_file(os.path.join(hybrid_shp_dir, "predicted_output.shp"))
            hybrid_zip_out = os.path.join(hybrid_export_path, "predicted_output.zip")
            with zipfile.ZipFile(hybrid_zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(hybrid_shp_dir):
                    z.write(os.path.join(hybrid_shp_dir, f), f)
            print(f"✅ Hybrid shapefile ZIP saved")
        except Exception as shp_err:
            print(f"⚠️ Hybrid shapefile export failed: {shp_err}")

        # ------------------------------------------------------------------
        # 12a. CSV — SLM only
        # ------------------------------------------------------------------
        slm_csv_path = os.path.join(slm_export_path, f"{slm_base}.csv")
        slm_csv_df   = df_valid[indep + [target]].copy()
        slm_csv_df[safe_target_name] = y_full
        slm_csv_df["prediction"]     = pred_slm
        slm_csv_df["slm_residual"]   = eps_slm
        if pin_series is not None:
            slm_csv_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        slm_csv_df.to_csv(slm_csv_path, index=False)

        # ------------------------------------------------------------------
        # 12b. CSV — Hybrid
        # ------------------------------------------------------------------
        hybrid_csv_path = os.path.join(hybrid_export_path, f"{hybrid_base}.csv")
        hybrid_csv_df   = df_valid[indep + [target]].copy()
        hybrid_csv_df[safe_target_name] = y_full
        hybrid_csv_df["slm_pred"]       = pred_slm
        hybrid_csv_df["slm_residual"]   = eps_slm
        hybrid_csv_df["rf_correction"]  = pred_rf_correction
        hybrid_csv_df["prediction"]     = pred_hybrid
        if pin_series is not None:
            hybrid_csv_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        hybrid_csv_df.to_csv(hybrid_csv_path, index=False)

        # ------------------------------------------------------------------
        # 13. CAMA previews
        # ------------------------------------------------------------------
        variable_distributions = compute_variable_distributions(df_valid, indep)

        # SLM preview
        slm_preview_df = df_valid.copy()
        slm_preview_df[safe_target_name] = y_full
        slm_preview_df["prediction"]     = pred_slm
        slm_preview_df["slm_residual"]   = eps_slm
        if pin_series is not None:
            slm_preview_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        slm_preview_cols = (
            (["PIN"] if pin_series is not None else [])
            + indep + [safe_target_name, "prediction", "slm_residual"]
        )
        slm_cama_preview = slm_preview_df[slm_preview_cols].head(100).to_dict("records")

        # Hybrid preview
        hybrid_preview_df = df_valid.copy()
        hybrid_preview_df[safe_target_name]  = y_full
        hybrid_preview_df["slm_pred"]        = pred_slm
        hybrid_preview_df["slm_residual"]    = eps_slm
        hybrid_preview_df["rf_correction"]   = pred_rf_correction
        hybrid_preview_df["prediction"]      = pred_hybrid
        if pin_series is not None:
            hybrid_preview_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        hybrid_preview_cols = (
            (["PIN"] if pin_series is not None else [])
            + indep + [safe_target_name, "slm_pred", "slm_residual", "rf_correction", "prediction"]
        )
        hybrid_cama_preview = hybrid_preview_df[hybrid_preview_cols].head(100).to_dict("records")

        # ------------------------------------------------------------------
        # 14. Build response — two separate result objects
        # ------------------------------------------------------------------
        base_url = "/api/ai-tools/download"

        # SLM downloads
        slm_downloads = {
            "model":    f"{base_url}?file={slm_model_path}",
            "cama_csv": f"{base_url}?file={slm_csv_path}",
        }
        if slm_zip_out:
            slm_downloads["shapefile"] = f"{base_url}?file={slm_zip_out}"
            slm_downloads["geojson"]   = f"/api/ai-tools/preview-geojson?file_path={slm_zip_out}"

        # Hybrid downloads
        hybrid_plots = {k: f"{base_url}?file={v}" for k, v in png_paths.items() if v}
        hybrid_downloads = {
            "model":    f"{base_url}?file={hybrid_model_path}",
            "cama_csv": f"{base_url}?file={hybrid_csv_path}",
        }
        if pdf_path:
            hybrid_downloads["report"] = f"{base_url}?file={pdf_path}"
        if hybrid_zip_out:
            hybrid_downloads["shapefile"] = f"{base_url}?file={hybrid_zip_out}"
            hybrid_downloads["geojson"]   = f"/api/ai-tools/preview-geojson?file_path={hybrid_zip_out}"

        # Residual histograms
        slm_counts, slm_bins = np.histogram(eps_slm, bins=20)
        slm_bin_centers      = 0.5 * (slm_bins[:-1] + slm_bins[1:])

        hybrid_counts, hybrid_bins = np.histogram(residuals_hybrid, bins=20)
        hybrid_bin_centers         = 0.5 * (hybrid_bins[:-1] + hybrid_bins[1:])

        slm_metrics = {
            "r2":   float(r2_slm),
            "rmse": float(rmse_slm),
            "pseudo_r2": float(pseudo_r2),
            "moran_i_slm": moran_i_slm,
            "moran_p_slm": moran_p_slm,
        }

        slm_stage = {
            "model_name":             slm_base,
            "model_id":               slm_base,
            "dependent_var":          safe_target_name,
            "original_dependent_var": target,
            "metrics":                {k: float(v) for k, v in slm_metrics.items() if v is not None},
            "features":               indep,
            "importance":             [],
            "slm_coefficients":       slm_coefficients,
            "rho":                    rho,
            "moran_i_slm":            moran_i_slm,
            "moran_p_slm":            moran_p_slm,
            "interactive_data": {
                "residuals":       eps_slm.tolist(),
                "residual_bins":   slm_bin_centers.tolist(),
                "residual_counts": slm_counts.tolist(),
                "y_test":          y_test.tolist(),
                "preds":           preds_slm_test.tolist(),
            },
            "variable_distributions": variable_distributions,
            "cama_preview":           slm_cama_preview,
            "plots":                  {},
            "downloads":              slm_downloads,
            "is_db_mode":             is_db_mode,
            "isRunMode":              False,
            "record_count":           int(len(df_valid)),
        }

        hybrid_stage = {
            "model_name":             hybrid_base,
            "model_id":               hybrid_base,
            "dependent_var":          safe_target_name,
            "original_dependent_var": target,
            "metrics":                {k: float(v) for k, v in metrics.items() if v is not None},
            "features":               indep,
            "importance":             importance,
            "slm_coefficients":       slm_coefficients,
            "rho":                    rho,
            "moran_i_slm":            moran_i_slm,
            "moran_p_slm":            moran_p_slm,
            "moran_i_hybrid":         moran_i_hybrid,
            "moran_p_hybrid":         moran_p_hybrid,
            "interactive_data": {
                "residuals":       residuals_hybrid.tolist(),
                "residual_bins":   hybrid_bin_centers.tolist(),
                "residual_counts": hybrid_counts.tolist(),
                "y_test":          y_test.tolist(),
                "preds":           preds_hybrid_test.tolist(),
                "preds_slm":       preds_slm_test.tolist(),
            },
            "variable_distributions": variable_distributions,
            "cama_preview":           hybrid_cama_preview,
            "plots":                  hybrid_plots,
            "downloads":              hybrid_downloads,
            "is_db_mode":             is_db_mode,
            "isRunMode":              False,
            "record_count":           int(len(df_valid)),
        }

        return {
            "slm_stage":    slm_stage,
            "hybrid_stage": hybrid_stage,
        }

    except Exception as e:
        import traceback
        print(f"❌ HYBRID TRAIN ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})