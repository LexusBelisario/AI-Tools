# sdm_xgb_train.py
"""
Hybrid Spatial Durbin Model + XGBoost
Stage 1: Fit SDM → get ŷ_SDM and ε_SDM
Stage 2: Fit XGBoost on ε_SDM using original X features
Final:   ŷ_hybrid = ŷ_SDM + ε̂_XGB

Returns two separate result objects:
  - sdm_stage:    SDM-only results (artifact base: SDM_...)
  - hybrid_stage: Full hybrid results (artifact base: H_SDM_XG_...)
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

from AITools.sdm_xgb_print_handler import export_sdm_xgb_report_and_artifacts
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
import xgboost as xgb

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)


def build_artifact_base_names(table_name: str = "") -> tuple:
    """Returns (sdm_base, hybrid_base) with consistent timestamp."""
    now = datetime.now(PHT)
    ts = now.strftime('%Y-%b-%d_%I-%M-%S%p')
    suffix = f"_{table_name.strip()}" if table_name and table_name.strip() else ""
    sdm_base    = f"SDM_{ts}{suffix}"
    hybrid_base = f"H_SDM_XG_{ts}{suffix}"
    return sdm_base, hybrid_base


def compute_wx(X_arr: np.ndarray, w) -> np.ndarray:
    """
    Compute WX: spatially lagged predictor matrix.
    For each observation i, WX_i = sum_j w_ij * X_j (row-standardized neighbor average).
    """
    n, k = X_arr.shape
    WX = np.zeros_like(X_arr)
    for i in range(n):
        neighbors = w.neighbors.get(i, [])
        weights   = w.weights.get(i, [])
        if neighbors:
            for j, wij in zip(neighbors, weights):
                WX[i] += wij * X_arr[j]
    return WX


def compute_impacts(rho: float, betas: np.ndarray, thetas: np.ndarray, w, n: int) -> list:
    """
    LeSage-Pace impacts decomposition using the power-series trace approximation.
    Avoids materializing the full n×n matrix.

    S_k(W) = (I - ρW)⁻¹ (β_k·I + θ_k·W)

    avg direct   ≈ β_k · tr((I-ρW)⁻¹)/n + θ_k · tr(W(I-ρW)⁻¹)/n
    avg total    = (β_k + θ_k) / (1 - ρ)
    avg indirect = avg total - avg direct
    """
    from scipy.sparse import lil_matrix, eye as speye

    W_sparse = lil_matrix((n, n))
    for i in range(n):
        neighbors = w.neighbors.get(i, [])
        weights_i = w.weights.get(i, [])
        for j, wij in zip(neighbors, weights_i):
            W_sparse[i, j] = wij
    W_sparse = W_sparse.tocsr()

    T = 10
    Wt = speye(n, format="csr")
    tr_IrW_inv  = 0.0
    tr_WIrW_inv = 0.0
    for t in range(T + 1):
        tr_Wt = float(Wt.diagonal().sum())
        tr_IrW_inv += (rho ** t) * tr_Wt
        WtNext = W_sparse.dot(Wt)
        tr_WIrW_inv += (rho ** t) * float(WtNext.diagonal().sum())
        Wt = WtNext

    impacts = []
    for beta_k, theta_k in zip(betas, thetas):
        avg_direct   = (beta_k * tr_IrW_inv + theta_k * tr_WIrW_inv) / n
        avg_total    = (beta_k + theta_k) / (1.0 - rho) if abs(1.0 - rho) > 1e-9 else 0.0
        avg_indirect = avg_total - avg_direct
        impacts.append({
            "avg_direct":   float(avg_direct),
            "avg_indirect": float(avg_indirect),
            "avg_total":    float(avg_total),
        })
    return impacts


@router.post("/train")
async def train_hybrid_sdm_xgb_model(
    shapefiles:       Optional[List[UploadFile]] = None,
    zip_file:         Optional[UploadFile]       = None,
    schema:           Optional[str]              = Form(None),
    table_name:       Optional[str]              = Form(None),
    independent_vars: str                        = Form(...),
    dependent_var:    str                        = Form(...),
    excluded_indices: Optional[str]              = Form("[]"),
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
        # 5. Stage 1 — Spatial Durbin Model
        # ------------------------------------------------------------------
        print("🚀 Stage 1: Fitting Spatial Durbin Model...")
        try:
            from spreg import GM_Lag
            n   = len(df_valid)
            k   = len(indep)
            y_arr = df_valid[[target]].values.astype(np.float64)
            X_arr = df_valid[indep].values.astype(np.float64)

            # Build WX and augment X → [X | WX]
            WX_arr    = compute_wx(X_arr, w)
            X_aug     = np.hstack([X_arr, WX_arr])

            model = GM_Lag(y_arr, X_aug, w=w)
            rho   = float(np.asarray(model.rho).flat[0])
            print(f"   ρ (rho): {rho:.4f}")
        except Exception as me:
            import traceback as _tb
            print(f"❌ SDM error: {me}\n{_tb.format_exc()}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Stage 1 (SDM) failed: {str(me)}"}
            )

        pred_sdm = model.predy.flatten()
        y_full   = model.y.flatten()
        eps_sdm  = y_full - pred_sdm          # residuals → XGBoost target

        print(f"   SDM predictions ready. Residual mean: {eps_sdm.mean():.4f}")

        # ------------------------------------------------------------------
        # 6. Extract SDM coefficients (β and θ) + z-stats
        # ------------------------------------------------------------------
        betas_all  = model.betas.flatten()        # [intercept, β1..βk, θ1..θk, ρ]
        std_errors = model.std_err.flatten() if hasattr(model, "std_err") and model.std_err is not None else np.zeros(len(betas_all))
        z_stats    = model.z_stat if hasattr(model, "z_stat") and model.z_stat is not None else []

        # Slice: intercept at index 0, β at 1..k, θ at k+1..2k, ρ at end
        beta_vals  = betas_all[1:k + 1]
        theta_vals = betas_all[k + 1:2 * k + 1]

        coeff_table = []
        importance  = []
        for i, feat in enumerate(indep):
            b_val  = float(beta_vals[i])  if i < len(beta_vals)  else 0.0
            th_val = float(theta_vals[i]) if i < len(theta_vals) else 0.0

            se_b  = float(std_errors[i + 1])         if (i + 1)         < len(std_errors) else 0.0
            se_th = float(std_errors[i + k + 1])     if (i + k + 1)     < len(std_errors) else 0.0

            z_b   = float(z_stats[i + 1][0])         if (i + 1)         < len(z_stats) else 0.0
            p_b   = float(z_stats[i + 1][1])         if (i + 1)         < len(z_stats) else 1.0
            z_th  = float(z_stats[i + k + 1][0])     if (i + k + 1)     < len(z_stats) else 0.0
            p_th  = float(z_stats[i + k + 1][1])     if (i + k + 1)     < len(z_stats) else 1.0

            coeff_table.append({
                "variable":       feat,
                "beta":           b_val,
                "beta_se":        se_b,
                "beta_z":         z_b,
                "beta_p":         p_b,
                "beta_sig":       p_b < 0.05,
                "theta":          th_val,
                "theta_se":       se_th,
                "theta_z":        z_th,
                "theta_p":        p_th,
                "theta_sig":      p_th < 0.05,
                "spillover_type": (
                    "positive" if th_val > 0.05 else
                    "negative" if th_val < -0.05 else
                    "none"
                ),
            })
            importance.append({"feature": feat, "value": abs(b_val)})

        # ------------------------------------------------------------------
        # 7. LeSage-Pace impacts decomposition
        # ------------------------------------------------------------------
        print("📐 Computing LeSage-Pace impacts decomposition...")
        try:
            impacts_list = compute_impacts(rho, beta_vals, theta_vals, w, n)
            for i, row in enumerate(coeff_table):
                row.update(impacts_list[i])
            print("   Impacts computed ✅")
        except Exception as imp_err:
            import traceback as _tb
            print(f"⚠️ Impacts decomposition failed: {imp_err}\n{_tb.format_exc()}")
            for row in coeff_table:
                row.update({"avg_direct": None, "avg_indirect": None, "avg_total": None})

        # Moran's I on SDM residuals (gate before Stage 2)
        moran_i_sdm = moran_p_sdm = None
        try:
            from esda.moran import Moran
            moran_sdm   = Moran(eps_sdm, w, transformation="r", permutations=99)
            moran_i_sdm = float(moran_sdm.I)
            moran_p_sdm = float(moran_sdm.p_sim)
            print(f"   Moran's I on SDM residuals: {moran_i_sdm:.4f} (p={moran_p_sdm:.4f})")
        except Exception as mi_err:
            print(f"⚠️ Moran's I (SDM residuals) failed: {mi_err}")

        # ------------------------------------------------------------------
        # 8. Stage 2 — XGBoost on SDM residuals
        # ------------------------------------------------------------------
        print("⚡ Stage 2: Fitting XGBoost on SDM residuals...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_model.fit(X_arr, eps_sdm)

        pred_xgb_correction = xgb_model.predict(X_arr)
        print(f"   XGBoost correction ready. Mean correction: {pred_xgb_correction.mean():.4f}")

        # XGBoost feature importance (gain-based)
        xgb_importance = sorted(
            [{"feature": f, "value": float(v)} for f, v in zip(indep, xgb_model.feature_importances_)],
            key=lambda x: x["value"], reverse=True
        )

        # ------------------------------------------------------------------
        # 9. Final hybrid prediction
        # ------------------------------------------------------------------
        pred_hybrid      = pred_sdm + pred_xgb_correction
        residuals_hybrid = y_full - pred_hybrid

        # ------------------------------------------------------------------
        # 10. Metrics (train/test split)
        # ------------------------------------------------------------------
        idx_all = np.arange(len(df_valid))
        idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=42)

        y_test            = y_full[idx_test]
        preds_hybrid_test = pred_hybrid[idx_test]
        preds_sdm_test    = pred_sdm[idx_test]

        r2_hybrid   = float(r2_score(y_test, preds_hybrid_test))
        rmse_hybrid = float(np.sqrt(mean_squared_error(y_test, preds_hybrid_test)))
        mae_hybrid  = float(mean_absolute_error(y_test, preds_hybrid_test))
        mse_hybrid  = float(mean_squared_error(y_test, preds_hybrid_test))

        r2_sdm   = float(r2_score(y_test, preds_sdm_test))
        rmse_sdm = float(np.sqrt(mean_squared_error(y_test, preds_sdm_test)))

        pseudo_r2 = float(model.pr2) if hasattr(model, "pr2") and model.pr2 is not None else r2_sdm

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
            "r2":             r2_hybrid,
            "rmse":           rmse_hybrid,
            "mae":            mae_hybrid,
            "mse":            mse_hybrid,
            # Stage 1 (SDM)
            "r2_sdm":         r2_sdm,
            "rmse_sdm":       rmse_sdm,
            "pseudo_r2":      pseudo_r2,
            "rho":            rho,
            "moran_i_sdm":    moran_i_sdm,
            "moran_p_sdm":    moran_p_sdm,
            # Stage 2 (Hybrid diagnostics)
            "moran_i_hybrid": moran_i_hybrid,
            "moran_p_hybrid": moran_p_hybrid,
        }
        print(f"   Hybrid R²={r2_hybrid:.4f}, RMSE={rmse_hybrid:.2f} | SDM R²={r2_sdm:.4f}")

        # ------------------------------------------------------------------
        # 11. Save artifacts — two separate export paths
        # ------------------------------------------------------------------
        safe_target_name = "actual_val" if len(target) > 10 else target
        sdm_base, hybrid_base = build_artifact_base_names(table_name or "")

        sdm_export_path    = os.path.join(EXPORT_DIR, sdm_base)
        hybrid_export_path = os.path.join(EXPORT_DIR, hybrid_base)
        os.makedirs(sdm_export_path, exist_ok=True)
        os.makedirs(hybrid_export_path, exist_ok=True)

        # SDM-only model bundle
        sdm_model_path = os.path.join(sdm_export_path, f"{sdm_base}.pkl")
        joblib.dump({
            "model":         model,
            "w":             w,
            "features":      indep,
            "dependent_var": target,
            "model_type":    "sdm",
            "rho":           rho,
            "beta_vals":     beta_vals.tolist(),
            "theta_vals":    theta_vals.tolist(),
            "trained_at":    datetime.now(PHT).isoformat(),
        }, sdm_model_path)
        print(f"✅ SDM model bundle saved: {sdm_base}")

        # Hybrid model bundle
        hybrid_model_path = os.path.join(hybrid_export_path, f"{hybrid_base}.pkl")
        joblib.dump({
            "model":         model,
            "xgb":           xgb_model,
            "w":             w,
            "features":      indep,
            "dependent_var": target,
            "model_type":    "hybrid_sdm_xgb",
            "rho":           rho,
            "beta_vals":     beta_vals.tolist(),
            "theta_vals":    theta_vals.tolist(),
            "trained_at":    datetime.now(PHT).isoformat(),
        }, hybrid_model_path)
        print(f"✅ Hybrid model bundle saved: {hybrid_base}")

        # ------------------------------------------------------------------
        # 12. PDF report + PNGs
        # ------------------------------------------------------------------
        try:
            png_paths, pdf_path = export_sdm_xgb_report_and_artifacts(
                export_path          = hybrid_export_path,
                artifact_base        = hybrid_base,
                indep                = indep,
                target               = target,
                y_full               = y_full,
                pred_sdm             = pred_sdm,
                pred_xgb_correction  = pred_xgb_correction,
                pred_hybrid          = pred_hybrid,
                residuals_hybrid     = residuals_hybrid,
                metrics              = metrics,
                coeff_table          = coeff_table,
                xgb_importance       = xgb_importance,
                rho                  = rho,
                moran_i_sdm          = moran_i_sdm,
                moran_p_sdm          = moran_p_sdm,
                moran_i_hybrid       = moran_i_hybrid,
                moran_p_hybrid       = moran_p_hybrid,
                df_valid             = df_valid,
            )
        except Exception as rep_err:
            import traceback
            print(f"⚠️ Report generation failed: {rep_err}")
            traceback.print_exc()
            png_paths = {}
            pdf_path  = None

        # ------------------------------------------------------------------
        # 13a. Shapefile — SDM only
        # ------------------------------------------------------------------
        sdm_zip_out = None
        try:
            gdf_sdm = gdf_valid.copy()
            gdf_sdm[safe_target_name] = y_full
            gdf_sdm["pred_sdm"]       = pred_sdm
            gdf_sdm = gdf_sdm.drop(columns=["__original_index__"], errors="ignore")
            if pin_series is not None:
                try:
                    upsert_pin_field(gdf_sdm, pin_series.iloc[df_valid.index].values, preferred_name="PIN")
                    drop_duplicate_pin_fields(gdf_sdm, keep_name="PIN")
                except Exception:
                    pass
            sdm_shp_dir = os.path.join(sdm_export_path, "predicted_shapefile")
            os.makedirs(sdm_shp_dir, exist_ok=True)
            gdf_sdm.to_file(os.path.join(sdm_shp_dir, "predicted_output.shp"))
            sdm_zip_out = os.path.join(sdm_export_path, "predicted_output.zip")
            with zipfile.ZipFile(sdm_zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(sdm_shp_dir):
                    z.write(os.path.join(sdm_shp_dir, f), f)
            print(f"✅ SDM shapefile ZIP saved")
        except Exception as shp_err:
            print(f"⚠️ SDM shapefile export failed: {shp_err}")

        # ------------------------------------------------------------------
        # 13b. Shapefile — Hybrid (pred_sdm + pred_sdm_xg)
        # ------------------------------------------------------------------
        hybrid_zip_out = None
        try:
            gdf_hybrid = gdf_valid.copy()
            gdf_hybrid[safe_target_name] = y_full
            gdf_hybrid["pred_sdm"]       = pred_sdm
            gdf_hybrid["pred_sdm_xg"]    = pred_hybrid
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
        # 14a. CSV — SDM only
        # ------------------------------------------------------------------
        sdm_csv_path = os.path.join(sdm_export_path, f"{sdm_base}.csv")
        sdm_csv_df   = df_valid[indep + [target]].copy()
        sdm_csv_df[safe_target_name] = y_full
        sdm_csv_df["pred_sdm"]       = pred_sdm
        if pin_series is not None:
            sdm_csv_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        sdm_csv_df.to_csv(sdm_csv_path, index=False)

        # ------------------------------------------------------------------
        # 14b. CSV — Hybrid
        # ------------------------------------------------------------------
        hybrid_csv_path = os.path.join(hybrid_export_path, f"{hybrid_base}.csv")
        hybrid_csv_df   = df_valid[indep + [target]].copy()
        hybrid_csv_df[safe_target_name] = y_full
        hybrid_csv_df["pred_sdm"]       = pred_sdm
        hybrid_csv_df["pred_sdm_xg"]    = pred_hybrid
        if pin_series is not None:
            hybrid_csv_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        hybrid_csv_df.to_csv(hybrid_csv_path, index=False)

        # ------------------------------------------------------------------
        # 15. CAMA previews
        # ------------------------------------------------------------------
        variable_distributions = compute_variable_distributions(df_valid, indep)

        # SDM preview
        sdm_preview_df = df_valid.copy()
        sdm_preview_df[safe_target_name] = y_full
        sdm_preview_df["pred_sdm"]       = pred_sdm
        if pin_series is not None:
            sdm_preview_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        sdm_preview_cols = (
            (["PIN"] if pin_series is not None else [])
            + indep + [safe_target_name, "pred_sdm"]
        )
        sdm_cama_preview = sdm_preview_df[sdm_preview_cols].head(100).to_dict("records")

        # Hybrid preview
        hybrid_preview_df = df_valid.copy()
        hybrid_preview_df[safe_target_name] = y_full
        hybrid_preview_df["pred_sdm"]       = pred_sdm
        hybrid_preview_df["pred_sdm_xg"]    = pred_hybrid
        if pin_series is not None:
            hybrid_preview_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        hybrid_preview_cols = (
            (["PIN"] if pin_series is not None else [])
            + indep + [safe_target_name, "pred_sdm", "pred_sdm_xg"]
        )
        hybrid_cama_preview = hybrid_preview_df[hybrid_preview_cols].head(100).to_dict("records")

        # ------------------------------------------------------------------
        # 16. Build response — two separate result objects
        # ------------------------------------------------------------------
        base_url = "/api/ai-tools/download"

        # SDM downloads
        sdm_downloads = {
            "model":    f"{base_url}?file={sdm_model_path}",
            "cama_csv": f"{base_url}?file={sdm_csv_path}",
        }
        if sdm_zip_out:
            sdm_downloads["shapefile"] = f"{base_url}?file={sdm_zip_out}"
            sdm_downloads["geojson"]   = f"/api/ai-tools/preview-geojson?file_path={sdm_zip_out}"

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
        sdm_counts, sdm_bins     = np.histogram(eps_sdm, bins=20)
        sdm_bin_centers          = 0.5 * (sdm_bins[:-1] + sdm_bins[1:])

        hybrid_counts, hybrid_bins = np.histogram(residuals_hybrid, bins=20)
        hybrid_bin_centers         = 0.5 * (hybrid_bins[:-1] + hybrid_bins[1:])

        sdm_metrics = {
            "r2":          float(r2_sdm),
            "rmse":        float(rmse_sdm),
            "pseudo_r2":   float(pseudo_r2),
            "moran_i_sdm": moran_i_sdm,
            "moran_p_sdm": moran_p_sdm,
        }

        sdm_stage = {
            "model_name":             sdm_base,
            "model_id":               sdm_base,
            "dependent_var":          safe_target_name,
            "original_dependent_var": target,
            "metrics":                {k: float(v) for k, v in sdm_metrics.items() if v is not None},
            "features":               indep,
            "importance":             importance,
            "coefficients":           coeff_table,
            "rho":                    rho,
            "moran_i_sdm":            moran_i_sdm,
            "moran_p_sdm":            moran_p_sdm,
            "interactive_data": {
                "residuals":       eps_sdm.tolist(),
                "residual_bins":   sdm_bin_centers.tolist(),
                "residual_counts": sdm_counts.tolist(),
                "y_test":          y_test.tolist(),
                "preds":           preds_sdm_test.tolist(),
            },
            "variable_distributions": variable_distributions,
            "cama_preview":           sdm_cama_preview,
            "plots":                  {},
            "downloads":              sdm_downloads,
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
            "importance":             xgb_importance,
            "coefficients":           coeff_table,
            "rho":                    rho,
            "moran_i_sdm":            moran_i_sdm,
            "moran_p_sdm":            moran_p_sdm,
            "moran_i_hybrid":         moran_i_hybrid,
            "moran_p_hybrid":         moran_p_hybrid,
            "interactive_data": {
                "residuals":       residuals_hybrid.tolist(),
                "residual_bins":   hybrid_bin_centers.tolist(),
                "residual_counts": hybrid_counts.tolist(),
                "y_test":          y_test.tolist(),
                "preds":           preds_hybrid_test.tolist(),
                "preds_sdm":       preds_sdm_test.tolist(),
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
            "sdm_stage":    sdm_stage,
            "hybrid_stage": hybrid_stage,
        }

    except Exception as e:
        import traceback
        print(f"❌ HYBRID SDM+XGB TRAIN ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})