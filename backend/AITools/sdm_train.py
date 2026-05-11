# sdm_train.py
"""
Spatial Durbin Model (SDM)
y = ρWy + Xβ + WXθ + ε

Strategy:
  - Manually construct WX columns (neighbor-averaged X) using the spatial weights matrix
  - Augment X with WX and feed the full matrix into GM_Lag
  - The first k betas are β (own effects); the next k are θ (spillover)
  - Compute LeSage-Pace impacts decomposition: S_k(W) = (I-ρW)⁻¹(β_k·I + θ_k·W)
  - Report average direct, indirect, and total impacts per predictor
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

from AITools.sdm_print_handler import export_sdm_report_and_artifacts
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


# ---------------------------------------------------------------------------
# Utility: sanitize nan / inf so JSON serialization never crashes
# ---------------------------------------------------------------------------

def _sanitize(obj):
    """
    Recursively walk dicts, lists, and scalar floats.
    Replaces nan and +/-inf with None so FastAPI's json.dumps never sees them.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):   # nan or inf
            return None
        return obj
    # numpy scalars
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if val != val or val == float("inf") or val == float("-inf"):
            return None
        return val
    return obj


def _safe_float(val, fallback=0.0):
    """Convert a value to float, returning fallback if nan/inf/None."""
    try:
        v = float(val)
        if v != v or v == float("inf") or v == float("-inf"):
            return fallback
        return v
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_artifact_base_name(table_name: str = "") -> str:
    now = datetime.now(PHT)
    base = f"SDM_{now.strftime('%Y-%b-%d_%I-%M-%S%p')}"
    if table_name and table_name.strip():
        base = f"{base}_{table_name.strip()}"
    return base


def compute_wx(X_arr: np.ndarray, w) -> np.ndarray:
    """
    Compute WX: spatially lagged predictor matrix.
    For each observation i, WX_i = sum_j w_ij * X_j (row-standardized neighbor average).
    Uses the sparse weights representation from libpysal.
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

    avg direct   = tr(S_k) / n   ≈ β_k · tr((I-ρW)⁻¹)/n + θ_k · tr(W(I-ρW)⁻¹)/n
    avg total    = (β_k + θ_k) / (1 - ρ)  [scalar shortcut for row-standardized W]
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

    T = 10  # sufficient for |ρ| < 1
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
            "avg_direct":   _safe_float(avg_direct),
            "avg_indirect": _safe_float(avg_indirect),
            "avg_total":    _safe_float(avg_total),
        })
    return impacts


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/train")
async def train_sdm_model(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file:   Optional[UploadFile]       = None,
    schema:     Optional[str]              = Form(None),
    table_name: Optional[str]              = Form(None),
    independent_vars: str                  = Form(...),
    dependent_var:    str                  = Form(...),
    excluded_indices: Optional[str]        = Form("[]"),
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

        df_valid = df_full.dropna(subset=indep + [target]).copy()
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
        # 5. Construct WX (spatially lagged predictors)
        # ------------------------------------------------------------------
        print("🔗 Computing WX (spatially lagged predictors)...")
        X_arr = df_valid[indep].values.astype(np.float64)
        y_arr = df_valid[[target]].values.astype(np.float64)
        n, k  = X_arr.shape

        WX_arr = compute_wx(X_arr, w)
        print(f"   WX shape: {WX_arr.shape}")

        # Augmented X = [X | WX]  →  GM_Lag sees 2k predictors
        # betas[0]        = intercept
        # betas[1:k+1]    = β  (own effects)
        # betas[k+1:2k+1] = θ  (spillover)
        # model.rho       = ρ  (spatial lag coefficient)
        X_aug = np.hstack([X_arr, WX_arr])
        print(f"   Augmented X shape: {X_aug.shape}")

        # ------------------------------------------------------------------
        # 6. Train SDM via GM_Lag on augmented X
        # ------------------------------------------------------------------
        print("🚀 Training Spatial Durbin Model (GM_Lag on [X | WX])...")
        try:
            from spreg import GM_Lag
            model = GM_Lag(y_arr, X_aug, w=w)
            rho   = _safe_float(float(np.asarray(model.rho).flat[0]))
            print(f"   ρ (rho): {rho:.4f}")
        except Exception as me:
            import traceback as _tb
            print(f"❌ GM_Lag (SDM) error: {type(me).__name__}: {me}")
            print(_tb.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": f"SDM training failed: {str(me)}"}
            )

        # ------------------------------------------------------------------
        # 7. Extract β and θ from betas
        # ------------------------------------------------------------------
        betas_all  = model.betas.flatten()
        intercept  = _safe_float(betas_all[0])
        beta_vals  = betas_all[1 : k + 1]           # own-predictor effects
        theta_vals = betas_all[k + 1 : 2 * k + 1]  # spillover effects

        std_errors = (
            model.std_err.flatten()
            if hasattr(model, "std_err") and model.std_err is not None
            else np.zeros(len(betas_all))
        )
        z_stats = (
            model.z_stat
            if hasattr(model, "z_stat") and model.z_stat is not None
            else []
        )

        print(f"   β (own effects):      {beta_vals}")
        print(f"   θ (spillover effects): {theta_vals}")

        # ------------------------------------------------------------------
        # 8. Predictions & metrics
        # ------------------------------------------------------------------
        preds_full = model.predy.flatten()
        y_full     = model.y.flatten()
        residuals  = y_full - preds_full

        idx_all             = np.arange(len(df_valid))
        idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=42)
        y_test              = y_full[idx_test]
        preds_test          = preds_full[idx_test]

        r2        = _safe_float(r2_score(y_test, preds_test))
        rmse      = _safe_float(np.sqrt(mean_squared_error(y_test, preds_test)))
        mae       = _safe_float(mean_absolute_error(y_test, preds_test))
        mse       = _safe_float(mean_squared_error(y_test, preds_test))
        pseudo_r2 = (
            _safe_float(model.pr2)
            if hasattr(model, "pr2") and model.pr2 is not None
            else r2
        )

        moran_i = moran_p = None
        try:
            from esda.moran import Moran
            moran   = Moran(residuals, w, transformation="r", permutations=99)
            moran_i = _safe_float(moran.I)
            moran_p = _safe_float(moran.p_sim)
            print(f"   Moran's I: {moran_i:.4f} (p={moran_p:.4f})")
        except Exception as mi_err:
            print(f"⚠️ Moran's I failed: {mi_err}")

        metrics = {
            "r2":        r2,
            "pseudo_r2": pseudo_r2,
            "mse":       mse,
            "mae":       mae,
            "rmse":      rmse,
            "rho":       rho,
            "moran_i":   moran_i,
            "moran_p":   moran_p,
        }

        # ------------------------------------------------------------------
        # 9. Coefficient table (β + θ per predictor)
        # ------------------------------------------------------------------
        coeff_table = []
        importance  = []
        for i, feat in enumerate(indep):
            b_val  = _safe_float(beta_vals[i])  if i < len(beta_vals)  else 0.0
            th_val = _safe_float(theta_vals[i]) if i < len(theta_vals) else 0.0
            se_b   = _safe_float(std_errors[i + 1])     if (i + 1)     < len(std_errors) else 0.0
            se_th  = _safe_float(std_errors[i + k + 1]) if (i + k + 1) < len(std_errors) else 0.0
            z_b    = _safe_float(z_stats[i + 1][0])     if (i + 1)     < len(z_stats)    else 0.0
            p_b    = _safe_float(z_stats[i + 1][1], fallback=1.0) if (i + 1)     < len(z_stats) else 1.0
            z_th   = _safe_float(z_stats[i + k + 1][0]) if (i + k + 1) < len(z_stats)   else 0.0
            p_th   = _safe_float(z_stats[i + k + 1][1], fallback=1.0) if (i + k + 1) < len(z_stats) else 1.0

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
        # 10. Impacts decomposition (LeSage-Pace)
        # ------------------------------------------------------------------
        print("📐 Computing LeSage-Pace impacts decomposition...")
        try:
            impacts_list = compute_impacts(rho, beta_vals, theta_vals, w, n)
            for i, row in enumerate(coeff_table):
                row.update(impacts_list[i])
            print("   Impacts computed ✅")
        except Exception as imp_err:
            import traceback as _tb
            print(f"⚠️ Impacts decomposition failed: {imp_err}")
            print(_tb.format_exc())
            for row in coeff_table:
                row.update({"avg_direct": None, "avg_indirect": None, "avg_total": None})

        # ------------------------------------------------------------------
        # 11. Save model bundle
        # ------------------------------------------------------------------
        artifact_base = build_artifact_base_name(table_name or "")
        export_path   = os.path.join(EXPORT_DIR, artifact_base)
        os.makedirs(export_path, exist_ok=True)

        model_path = os.path.join(export_path, f"{artifact_base}.pkl")
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
        }, model_path)
        print(f"✅ Model saved: {os.path.basename(model_path)}")

        # ------------------------------------------------------------------
        # 12. PDF report + PNG charts
        # ------------------------------------------------------------------
        try:
            png_paths, pdf_path = export_sdm_report_and_artifacts(
                export_path   = export_path,
                artifact_base = artifact_base,
                model         = model,
                indep         = indep,
                target        = target,
                y_full        = y_full,
                preds_full    = preds_full,
                residuals     = residuals,
                metrics       = metrics,
                coeff_table   = coeff_table,
                rho           = rho,
                moran_i       = moran_i,
                moran_p       = moran_p,
                df_valid      = df_valid,
            )
        except Exception as rep_err:
            import traceback
            print(f"⚠️ Report generation failed: {rep_err}")
            traceback.print_exc()
            png_paths = {}
            pdf_path  = None

        # ------------------------------------------------------------------
        # 13. Shapefile output — includes pred_sdm column
        # ------------------------------------------------------------------
        safe_target_name = "actual_val" if len(target) > 10 else target
        zip_out = None
        try:
            gdf_out = gdf_valid.copy()
            gdf_out["pred_sdm"]       = preds_full
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
        # 14. CSV + variable distributions + CAMA preview
        # ------------------------------------------------------------------
        csv_path = os.path.join(export_path, f"{artifact_base}.csv")
        csv_df   = df_valid[indep + [target]].copy()
        csv_df["pred_sdm"] = preds_full
        if pin_series is not None:
            csv_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        csv_df.to_csv(csv_path, index=False)

        variable_distributions = compute_variable_distributions(df_valid, indep)

        preview_df = df_valid.copy()
        preview_df["pred_sdm"] = preds_full
        if pin_series is not None:
            preview_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)
        preview_cols = (["PIN"] if pin_series is not None else []) + indep + [target, "pred_sdm"]
        cama_preview = preview_df[preview_cols].head(100).to_dict("records")

        # ------------------------------------------------------------------
        # 15. Build response URLs
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

        # ------------------------------------------------------------------
        # 16. Sanitize entire response — replaces all nan/inf with None
        #     so json.dumps never raises "Out of range float values"
        # ------------------------------------------------------------------
        response = _sanitize({
            "model_name":             artifact_base,
            "model_id":               artifact_base,
            "dependent_var":          safe_target_name,
            "original_dependent_var": target,
            "metrics":                {k: v for k, v in metrics.items()},
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
        })

        return response

    except Exception as e:
        import traceback
        print(f"❌ SDM TRAIN ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})