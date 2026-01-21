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
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

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
    get_next_model_version,
    extract_pin_column,
    upsert_pin_field,
    drop_duplicate_pin_fields,
)

import joblib

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)


def _wrap_download_urls(paths: Dict[str, Optional[str]], base_url: str) -> Dict[str, Optional[str]]:
    # same pattern used sa ibang trainers: /download?file=...
    out = {}
    for k, p in paths.items():
        out[k] = f"{base_url}?file={p}" if p else None
    return out


def export_rf_report_and_artifacts(
    export_path: str,
    model: RandomForestRegressor,
    scaler: Optional[StandardScaler],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    df_full: pd.DataFrame,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    indep: List[str],
    target: str,
    excluded_indices: List[int],
    is_db_mode: bool,
    schema: Optional[str],
    table_name: Optional[str],
    file_gdf: Optional[gpd.GeoDataFrame] = None,
    model_version: int = 1,
) -> Dict[str, Any]:
    os.makedirs(export_path, exist_ok=True)
    plots_dir = os.path.join(export_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ===========================
    # 1) METRICS
    # ===========================
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R²": float(r2),
    }

    # ===========================
    # 2) PLOTS (PNG)
    # ===========================
    accent = "#1e88e5"
    residuals = y_test - y_pred

    # feature importance
    fi_path = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        try:
            fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.3)))
            order = np.argsort(importance)
            ax.barh(np.array(feature_names)[order], np.array(importance)[order], color=accent)
            ax.set_xlabel("Feature Importance")
            ax.set_title("Random Forest Feature Importance", color=accent, fontsize=13, weight="bold", pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            fi_path = os.path.join(plots_dir, "feature_importance.png")
            fig.savefig(fi_path, dpi=200)
            plt.close(fig)
        except Exception:
            fi_path = None

    # residual distribution
    rd_path = None
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, ax=ax, color=accent, edgecolor="black")
        ax.set_title("Residual Distribution (RF)", color=accent, fontsize=13, weight="bold", pad=10)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        rd_path = os.path.join(plots_dir, "residual_distribution.png")
        fig.savefig(rd_path, dpi=200)
        plt.close(fig)
    except Exception:
        rd_path = None

    # actual vs predicted
    avp_path = None
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color=accent, edgecolor="black", linewidth=0.5)
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="Perfect Prediction")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted (RF)", color=accent, fontsize=13, weight="bold", pad=10)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        avp_path = os.path.join(plots_dir, "actual_vs_predicted.png")
        fig.savefig(avp_path, dpi=200)
        plt.close(fig)
    except Exception:
        avp_path = None

    # residuals vs predicted
    rvp_path = None
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_pred, residuals, alpha=0.6, color="#e53935", edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title("Residuals vs Predicted (RF)", color="#e53935", fontsize=13, weight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        rvp_path = os.path.join(plots_dir, "residuals_vs_predicted.png")
        fig.savefig(rvp_path, dpi=200)
        plt.close(fig)
    except Exception:
        rvp_path = None

    # ===========================
    # 3) PDF REPORT (same idea as LR/XGB)
    # ===========================
    pdf_path = os.path.join(export_path, f"RF_Report_v{model_version}.pdf")
    with PdfPages(pdf_path) as pp:
        # metrics table page
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis("off")
        table = ax.table(
            cellText=[
                ["Model", "MSE", "MAE", "RMSE", "R²"],
                ["Random Forest", f"{mse:.2f}", f"{mae:.2f}", f"{rmse:.2f}", f"{r2:.2f}"],
            ],
            loc="center",
            cellLoc="center",
        )
        table.scale(1, 2)
        for (i, _j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor(accent)
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#f0f0f0")
        pp.savefig(fig, facecolor="white")
        plt.close(fig)

        # feature importance page (direct plot)
        if hasattr(model, "feature_importances_"):
            try:
                importance = model.feature_importances_
                fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.3)))
                order = np.argsort(importance)
                ax.barh(np.array(feature_names)[order], np.array(importance)[order], color=accent)
                ax.set_xlabel("Feature Importance")
                ax.set_title("Feature Importance (RF)", color=accent, fontsize=13, weight="bold", pad=10)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                pp.savefig(fig, facecolor="white")
                plt.close(fig)
            except Exception:
                pass

        # residual distribution page
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(residuals, kde=True, ax=ax, color=accent, edgecolor="black")
            ax.set_title("Residual Distribution (RF)", color=accent, fontsize=13, weight="bold", pad=10)
            ax.set_xlabel("Residual")
            ax.set_ylabel("Frequency")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            pp.savefig(fig, facecolor="white")
            plt.close(fig)
        except Exception:
            pass

        # actual vs predicted page
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_test, y_pred, alpha=0.6, color=accent, edgecolor="black", linewidth=0.5)
            min_val = min(np.min(y_test), np.min(y_pred))
            max_val = max(np.max(y_test), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted Scatter Plot (RF)", color=accent, fontsize=13, weight="bold", pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            pp.savefig(fig, facecolor="white")
            plt.close(fig)
        except Exception:
            pass

        # per-variable distribution pages (same idea sa LR)
        print("   📊 Adding variable distribution pages...")
        for col in indep:
            try:
                col_data = df_valid[col].dropna()
                if len(col_data) == 0:
                    continue

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(col_data, kde=True, ax=ax, color=accent, edgecolor="black", bins=30)
                ax.set_title(f"Distribution of {col}", color=accent, fontsize=13, weight="bold", pad=10)
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                mean_val = col_data.mean()
                median_val = col_data.median()
                std_val = col_data.std()
                stats_text = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}"
                ax.text(
                    0.98, 0.97, stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor=accent),
                    fontsize=9
                )

                plt.tight_layout()
                pp.savefig(fig, facecolor="white")
                plt.close(fig)
                print(f"      ✅ Added distribution for {col}")
            except Exception as e:
                print(f"      ⚠️ Could not create distribution for {col}: {e}")
                continue

    print(f"   ✅ PDF report saved: {pdf_path}")

    # ===========================
    # 4) SAVE MODEL (keep .pkl, compress para di sobrang laki)
    # ===========================
    model_path = os.path.join(export_path, f"RF_model_{model_version}.pkl")
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,                # important for universal runner
            "features": feature_names,
            "target": target,
            "version": model_version,
            "model_type": "rf",
            "trained_at": datetime.now().isoformat(),
        },
        model_path,
        compress=3,
    )
    print(f"💾 Saved model: RF_model_{model_version}.pkl")

    # ===========================
    # 5) PREDICTIONS FOR df_valid + CSV (keep this feature)
    # ===========================
    df_valid = df_valid.copy()

    # inject PIN if present
    pin_series, _ = extract_pin_column(df_full)
    if pin_series is not None:
        try:
            df_valid["PIN"] = pin_series.iloc[df_valid.index].values
        except Exception as e:
            print("⚠ PIN injection failed:", e)

    # predict for valid rows (use scaler if exists)
    X_valid = df_valid[indep].values
    if scaler is not None:
        X_valid = scaler.transform(X_valid)
    preds_valid = model.predict(X_valid)

    df_valid["prediction"] = preds_valid

    cols = []
    if "PIN" in df_valid.columns:
        cols.append("PIN")
    cols.extend(indep)
    cols.append(target)
    cols.append("prediction")

    csv_path = os.path.join(export_path, f"RF_Training_Result_v{model_version}.csv")
    df_valid[cols].to_csv(csv_path, index=False)
    print(f"✅ Exported cleaned CSV (with PIN if available): {csv_path}")

    # ===========================
    # 6) SHAPEFILE + ZIP EXPORT (keep this feature)
    # ===========================
    zip_out = None
    try:
        if is_db_mode:
            print("✅ Database mode: fetching geometry for export")
            gdf_db = gdf_from_db_with_geometry(schema, table_name)

            # IMPORTANT: df_valid indices are ORIGINAL row positions
            valid_indices = df_valid.index.tolist()
            valid_gdf = gdf_db.iloc[valid_indices].copy()

            # upsert PIN and cleanup duplicates
            if pin_series is not None:
                upsert_pin_field(valid_gdf, pin_series.iloc[valid_indices].values)
            drop_duplicate_pin_fields(valid_gdf)

            # add prediction
            valid_gdf["prediction"] = df_valid["prediction"].values

        else:
            if file_gdf is None:
                raise ValueError("File GeoDataFrame is required for file mode export.")

            print("✅ File mode: using uploaded geometry for export")
            valid_indices = df_valid.index.tolist()
            valid_gdf = file_gdf.iloc[valid_indices].copy()

            if pin_series is not None:
                try:
                    valid_gdf["PIN"] = pin_series.iloc[valid_indices].values
                except Exception as e:
                    print(f"   ⚠️ Could not add PIN: {e}")

            valid_gdf["prediction"] = df_valid["prediction"].values

        shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
        os.makedirs(shp_pred_dir, exist_ok=True)
        shp_pred_path = os.path.join(shp_pred_dir, "RandomForest_Predicted.shp")
        valid_gdf.to_file(shp_pred_path)

        zip_out = os.path.join(export_path, "RandomForest_Predicted.zip")
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in os.listdir(shp_pred_dir):
                z.write(os.path.join(shp_pred_dir, f), f)

        print(f"   ✅ Shapefile saved: {shp_pred_path}")
        print(f"   ✅ ZIP created: {zip_out}")

    except Exception as e:
        print(f"⚠️ Could not export shapefile/ZIP: {e}")

    # ===========================
    # 7) RETURN PATHS
    # ===========================
    plots = {
        "feature_importance": fi_path,
        "residual_distribution": rd_path,
        "actual_vs_predicted": avp_path,
        "residuals_vs_predicted": rvp_path,
    }

    downloads = {
        "model": model_path,
        "report": pdf_path,
        "cama_csv": csv_path,
        "shapefile": zip_out,
    }

    return {
        "metrics": metrics,
        "plots": plots,
        "downloads": downloads,
    }


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

        # ===========================
        # 10) df_valid (valid rows, keep original indices)
        # ===========================
        valid_indices = df_model["__orig_index__"].values
        df_valid = df_full.loc[valid_indices].copy()
        df_valid = df_valid[indep + [target]].copy()

        # ===========================
        # 11) EXPORT ARTIFACTS (keep everything)
        # ===========================
        model_version = get_next_model_version("rf")
        export_id = f"rf_{model_version}"
        export_path = os.path.join(EXPORT_DIR, export_id)
        os.makedirs(export_path, exist_ok=True)
        print(f"📦 Creating export folder: {export_id} (Version {model_version})")

        artifacts = export_rf_report_and_artifacts(
            export_path=export_path,
            model=model,
            scaler=scaler,
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            feature_names=indep,
            df_full=df_full,
            df_train=df_model,
            df_valid=df_valid,
            indep=indep,
            target=target,
            excluded_indices=excluded_list,
            is_db_mode=is_db_mode,
            schema=schema,
            table_name=table_name,
            file_gdf=file_gdf,
            model_version=model_version,
        )

        plots = artifacts["plots"]
        downloads = artifacts["downloads"]
        metrics = artifacts["metrics"]

        # ===========================
        # 12) INTERACTIVE DATA (same payload concept)
        # ===========================
        residuals = y_test - y_pred
        counts, bin_edges = np.histogram(residuals, bins=20)
        residual_bins = bin_edges.tolist()
        residual_counts = counts.tolist()

        # ===========================
        # 13) VARIABLE DISTRIBUTIONS (keep)
        # ===========================
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
            "model_version": model_version,
            "model_id": export_id,
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
