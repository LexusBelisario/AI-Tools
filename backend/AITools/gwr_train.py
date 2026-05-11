# gwr_train.py
"""
Geographically Weighted Regression (GWR)
y_i = β_0(u_i,v_i) + Σ β_k(u_i,v_i) x_ik + ε_i

Strategy:
  - Extract centroid coordinates (u, v) from geometry
  - Select optimal bandwidth via AICc minimization (adaptive bi-square kernel)
  - Fit GWR using mgwr library — produces n×k local coefficient surfaces
  - Report per-predictor summary stats (min, mean, max, IQR) of local β surfaces
  - Moran's I on GWR residuals as final diagnostic
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

from AITools.gwr_print_handler import export_gwr_report_and_artifacts
from AITools.ai_utils import (
    extract_pin_column,
    compute_variable_distributions,
    upsert_pin_field,
    drop_duplicate_pin_fields,
    safe_to_float,
    gdf_from_db_with_geometry,
    gdf_from_zip_or_parts,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)


def build_artifact_base_name(table_name: str = "") -> str:
    now = datetime.now(PHT)
    base = f"GWR_{now.strftime('%Y-%b-%d_%I-%M-%S%p')}"
    if table_name and table_name.strip():
        base = f"{base}_{table_name.strip()}"
    return base


def extract_centroids(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extract centroid (u, v) coordinates from GeoDataFrame geometry.
    Reprojects to a metric CRS if needed so distances are in metres.
    Returns shape (n, 2) array of [easting, northing] or [lon, lat].
    """
    gdf_proj = gdf.copy()

    # Try to reproject to a metric CRS for accurate distance computation
    try:
        if gdf_proj.crs is None:
            gdf_proj = gdf_proj.set_crs(epsg=4326)
        if gdf_proj.crs.is_geographic:
            # Reproject to UTM zone 51N (covers Philippines); adjust if needed
            gdf_proj = gdf_proj.to_crs(epsg=32651)
            print("   Reprojected to UTM Zone 51N for metric GWR distances")
    except Exception as crs_err:
        print(f"   ⚠️ CRS reproject failed: {crs_err} — using raw coordinates")

    centroids = gdf_proj.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    print(f"   Centroid range x: {coords[:,0].min():.2f}–{coords[:,0].max():.2f}")
    print(f"   Centroid range y: {coords[:,1].min():.2f}–{coords[:,1].max():.2f}")
    return coords


@router.post("/train")
async def train_gwr_model(
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
        # 4. Extract centroid coordinates
        # ------------------------------------------------------------------
        print("📍 Extracting centroid coordinates...")
        try:
            coords = extract_centroids(gdf_valid)
        except Exception as coord_err:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to extract centroids: {str(coord_err)}"}
            )

        # ------------------------------------------------------------------
        # 5. Prepare arrays — standardize X and y to prevent singular matrices
        # ------------------------------------------------------------------
        y_arr = df_valid[[target]].values.astype(np.float64)
        X_arr = df_valid[indep].values.astype(np.float64)
        n, k  = X_arr.shape

        # Standardize X (zero mean, unit variance) — critical for GWR stability
        # with large datasets where raw predictor scales cause ill-conditioned matrices
        from sklearn.preprocessing import StandardScaler
        X_scaler = StandardScaler()
        X_std    = X_scaler.fit_transform(X_arr)

        # Standardize y as well for numerical stability
        y_mean = float(y_arr.mean())
        y_std  = float(y_arr.std()) if y_arr.std() > 0 else 1.0
        y_std_arr = ((y_arr - y_mean) / y_std).astype(np.float64)

        print(f"   y shape: {y_arr.shape}, X shape: {X_std.shape}, coords shape: {coords.shape}")
        print(f"   X standardized: mean≈0, std≈1 per feature")

        # ------------------------------------------------------------------
        # 6. Bandwidth selection + GWR fit (singular-safe)
        # ------------------------------------------------------------------
        print("🚀 Fitting GWR with AICc bandwidth selection (adaptive bi-square)...")
        try:
            from mgwr.gwr import GWR
            from mgwr.sel_bw import Sel_BW
            import spglm.iwls as _iwls
            import scipy.linalg as _slinalg

            # --- Monkey-patch spglm to use lstsq instead of solve ---
            # linalg.solve raises LinAlgError on singular matrices during
            # bandwidth search; lstsq gracefully handles rank-deficient cases
            def _safe_compute_betas_gwr(y, X, wi):
                sw       = wi ** 0.5
                wX       = sw[:, None] * X
                wy       = sw[:, None] * y
                xtx      = wX.T @ wX
                xty      = wX.T @ wy
                # lstsq replaces solve — never raises on singular/ill-conditioned
                betas, _, _, _ = np.linalg.lstsq(xtx, xty, rcond=None)
                xtx_inv_xt = np.linalg.lstsq(xtx, wX.T, rcond=None)[0]
                return betas, xtx_inv_xt

            _iwls._compute_betas_gwr = _safe_compute_betas_gwr
            print("   ✅ spglm patched with lstsq (singular-safe)")

            # Min bandwidth = k+2 to ensure each local window is overdetermined
            min_bw = k + 2

            selector = Sel_BW(
                coords, y_std_arr, X_std,
                kernel="bisquare", fixed=False,
                n_jobs=1,
            )
            bw = selector.search(criterion="AICc")
            bw = max(int(bw), min_bw)
            print(f"   Optimal bandwidth (adaptive k-NN): {bw}")

            gwr_model   = GWR(
                coords, y_std_arr, X_std,
                bw=bw, kernel="bisquare", fixed=False,
                n_jobs=1,
            )
            gwr_results = gwr_model.fit()
            print("   GWR fit complete ✅")

        except Exception as gwr_err:
            import traceback as _tb
            print(f"❌ GWR error: {gwr_err}\n{_tb.format_exc()}")
            return JSONResponse(
                status_code=500,
                content={"error": f"GWR training failed: {str(gwr_err)}"}
            )

        # ------------------------------------------------------------------
        # 7. Extract local coefficients (n × k+1: intercept + k predictors)
        # ------------------------------------------------------------------
        # gwr_results.params shape: (n, k+1) — col 0 = intercept, cols 1..k = β_k
        local_params  = gwr_results.params                        # (n, k+1) — on standardized scale
        local_t       = gwr_results.tvalues                       # (n, k+1)
        local_r2      = gwr_results.localR2                       # (n,)

        # Back-transform predictions to original y scale
        preds_std  = gwr_results.predy.flatten()
        preds_full = preds_std * y_std + y_mean                   # inverse standardization
        residuals  = gwr_results.resid_response.flatten() * y_std # scale residuals back
        y_full     = y_arr.flatten()                              # original y

        print(f"   Local params shape: {local_params.shape}")
        print(f"   Local R² range: {local_r2.min():.4f} – {local_r2.max():.4f}")

        # ------------------------------------------------------------------
        # 8. Per-predictor local β summary (min, Q1, mean, Q3, max, IQR)
        # ------------------------------------------------------------------
        # Column 0 = intercept; columns 1..k = β for each predictor
        local_betas = local_params[:, 1:]   # (n, k)
        local_t_betas = local_t[:, 1:]      # (n, k)

        coeff_summary = []
        importance    = []
        for i, feat in enumerate(indep):
            beta_col = local_betas[:, i]
            t_col    = local_t_betas[:, i]
            q1, q3   = float(np.percentile(beta_col, 25)), float(np.percentile(beta_col, 75))
            pct_sig  = float(np.mean(np.abs(t_col) >= 1.96) * 100)

            coeff_summary.append({
                "variable":   feat,
                "beta_min":   float(beta_col.min()),
                "beta_q1":    q1,
                "beta_mean":  float(beta_col.mean()),
                "beta_median":float(np.median(beta_col)),
                "beta_q3":    q3,
                "beta_max":   float(beta_col.max()),
                "beta_iqr":   float(q3 - q1),
                "beta_std":   float(beta_col.std()),
                "pct_sig":    pct_sig,
                # Full local β surface for mapping
                "local_betas": beta_col.tolist(),
                "local_t":     t_col.tolist(),
            })
            importance.append({"feature": feat, "value": float(np.abs(beta_col).mean())})

        # Sort importance by mean absolute β
        importance = sorted(importance, key=lambda x: x["value"], reverse=True)

        # ------------------------------------------------------------------
        # 9. Global metrics (train/test split on predictions)
        # ------------------------------------------------------------------
        idx_all             = np.arange(n)
        idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=42)
        y_test              = y_full[idx_test]
        preds_test          = preds_full[idx_test]

        r2   = float(r2_score(y_test, preds_test))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds_test)))
        mae  = float(mean_absolute_error(y_test, preds_test))
        mse  = float(mean_squared_error(y_test, preds_test))

        # GWR-specific diagnostics
        aicc     = float(gwr_results.aicc)
        eff_df   = float(gwr_results.tr_S)          # effective degrees of freedom (trace of hat matrix)
        mean_r2  = float(local_r2.mean())

        print(f"   R²={r2:.4f}, RMSE={rmse:.2f}, AICc={aicc:.2f}, mean local R²={mean_r2:.4f}")

        # Moran's I on GWR residuals
        moran_i = moran_p = None
        try:
            from libpysal.weights import Queen
            from esda.moran import Moran
            w       = Queen.from_dataframe(gdf_valid, silence_warnings=True)
            w.transform = "r"
            moran_res = Moran(residuals, w, transformation="r", permutations=99)
            moran_i   = float(moran_res.I)
            moran_p   = float(moran_res.p_sim)
            print(f"   Moran's I (GWR residuals): {moran_i:.4f} (p={moran_p:.4f})")
        except Exception as mi_err:
            print(f"   ⚠️ Moran's I failed: {mi_err}")

        metrics = {
            "r2":      r2,
            "rmse":    rmse,
            "mae":     mae,
            "mse":     mse,
            "aicc":    aicc,
            "eff_df":  eff_df,
            "mean_r2": mean_r2,
            "bandwidth": float(bw),
            "moran_i": moran_i,
            "moran_p": moran_p,
        }

        # ------------------------------------------------------------------
        # 10. Save model bundle
        # ------------------------------------------------------------------
        artifact_base = build_artifact_base_name(table_name or "")
        export_path   = os.path.join(EXPORT_DIR, artifact_base)
        os.makedirs(export_path, exist_ok=True)

        model_path = os.path.join(export_path, f"{artifact_base}.pkl")
        joblib.dump({
            "gwr_results":    gwr_results,
            "coords":         coords,
            "features":       indep,
            "dependent_var":  target,
            "model_type":     "gwr",
            "bandwidth":      float(bw),
            "local_params":   local_params.tolist(),
            "local_r2":       local_r2.tolist(),
            "trained_at":     datetime.now(PHT).isoformat(),
        }, model_path)
        print(f"✅ GWR model bundle saved: {artifact_base}")

        # ------------------------------------------------------------------
        # 11. PDF report + PNGs
        # ------------------------------------------------------------------
        try:
            png_paths, pdf_path = export_gwr_report_and_artifacts(
                export_path    = export_path,
                artifact_base  = artifact_base,
                indep          = indep,
                target         = target,
                y_full         = y_full,
                preds_full     = preds_full,
                residuals      = residuals,
                local_r2       = local_r2,
                coeff_summary  = coeff_summary,
                metrics        = metrics,
                moran_i        = moran_i,
                moran_p        = moran_p,
                df_valid       = df_valid,
                coords         = coords,
            )
        except Exception as rep_err:
            import traceback
            print(f"⚠️ Report generation failed: {rep_err}")
            traceback.print_exc()
            png_paths = {}
            pdf_path  = None

        # ------------------------------------------------------------------
        # 12. Shapefile — includes pred_gwr + local_r2 columns
        # ------------------------------------------------------------------
        safe_target_name = "actual_val" if len(target) > 10 else target
        zip_out = None
        try:
            gdf_out = gdf_valid.copy()
            gdf_out[safe_target_name] = y_full
            gdf_out["pred_gwr"]       = preds_full
            gdf_out["local_r2"]       = local_r2
            gdf_out = gdf_out.drop(columns=["__original_index__"], errors="ignore")

            if pin_series is not None:
                try:
                    upsert_pin_field(gdf_out, pin_series.iloc[df_valid.index].values, preferred_name="PIN")
                    drop_duplicate_pin_fields(gdf_out, keep_name="PIN")
                except Exception:
                    pass

            # Also add local β columns per predictor (truncated name for shapefile)
            for i, feat in enumerate(indep):
                col_name = f"b_{feat[:6]}" if len(feat) > 6 else f"b_{feat}"
                gdf_out[col_name] = local_betas[:, i]

            shp_dir = os.path.join(export_path, "predicted_shapefile")
            os.makedirs(shp_dir, exist_ok=True)
            gdf_out.to_file(os.path.join(shp_dir, "predicted_output.shp"))

            zip_out = os.path.join(export_path, "predicted_output.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(shp_dir):
                    z.write(os.path.join(shp_dir, f), f)
            print(f"✅ GWR shapefile ZIP saved")
        except Exception as shp_err:
            print(f"⚠️ Shapefile export failed: {shp_err}")

        # ------------------------------------------------------------------
        # 13. CSV
        # ------------------------------------------------------------------
        csv_path = os.path.join(export_path, f"{artifact_base}.csv")
        csv_df   = df_valid[indep + [target]].copy()
        csv_df[safe_target_name] = y_full
        csv_df["pred_gwr"]       = preds_full
        csv_df["local_r2"]       = local_r2
        for i, feat in enumerate(indep):
            csv_df[f"beta_{feat}"] = local_betas[:, i]
        if pin_series is not None:
            csv_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        csv_df.to_csv(csv_path, index=False)

        # ------------------------------------------------------------------
        # 14. CAMA preview
        # ------------------------------------------------------------------
        variable_distributions = compute_variable_distributions(df_valid, indep)

        preview_df = df_valid.copy()
        preview_df[safe_target_name] = y_full
        preview_df["pred_gwr"]       = preds_full
        preview_df["local_r2"]       = local_r2
        if pin_series is not None:
            preview_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        preview_cols = (
            (["PIN"] if pin_series is not None else [])
            + indep + [safe_target_name, "pred_gwr", "local_r2"]
        )
        cama_preview = preview_df[preview_cols].head(100).to_dict("records")

        # ------------------------------------------------------------------
        # 15. Build response
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
            "coefficients":           coeff_summary,    # local β summary per predictor
            "bandwidth":              float(bw),
            "moran_i":                moran_i,
            "moran_p":                moran_p,
            "local_r2":               local_r2.tolist(),
            "interactive_data": {
                "residuals":       residuals.tolist(),
                "residual_bins":   bin_centers.tolist(),
                "residual_counts": counts.tolist(),
                "y_test":          y_test.tolist(),
                "preds":           preds_test.tolist(),
                "local_r2":        local_r2.tolist(),
            },
            "variable_distributions": variable_distributions,
            "cama_preview":           cama_preview,
            "plots":                  plots,
            "downloads":              downloads,
            "is_db_mode":             is_db_mode,
            "isRunMode":              False,
            "record_count":           int(n),
        }

    except Exception as e:
        import traceback
        print(f"❌ GWR TRAIN ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})