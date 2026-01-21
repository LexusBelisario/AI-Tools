# xgb_train.py
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Tuple, Dict, Any
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile, os, pickle, json, zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
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
    get_next_model_version,
    upsert_pin_field, 
    drop_duplicate_pin_fields
)


router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)

def wrap_plot_urls(plots: Dict[str, Optional[str]], prefix: str) -> Dict[str, Optional[str]]:
    return {
        key: (f"{prefix}?file={path}" if path else None)
        for key, path in plots.items()
    }


def plot_feature_importance(importance: np.ndarray, feature_names: List[str], ax=None):
    """Plot feature importances as a horizontal bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    else:
        fig = ax.figure

    sorted_idx = np.argsort(importance)
    sorted_importance = importance[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]

    ax.barh(range(len(sorted_importance)), sorted_importance)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    return fig, ax


def plot_residual_distribution(residuals: np.ndarray, ax=None):
    """Plot the distribution of residuals with a histogram + KDE."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Residual Distribution (XGBoost)")
    ax.set_xlabel("Residual")
    fig.tight_layout()
    return fig, ax


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, ax=None):
    """Scatter plot: actual vs predicted."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.scatter(y_test, y_pred, alpha=0.6)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (XGBoost)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_residuals_vs_predicted(y_pred: np.ndarray, residuals: np.ndarray, ax=None):
    """Scatter plot of residuals vs predicted."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted (XGBoost)")
    fig.tight_layout()
    return fig, ax


def export_xgb_report_and_artifacts(
    export_path: str,
    model: XGBRegressor,
    scaler,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    df_full: pd.DataFrame,
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

    # 1️⃣ Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
    }

    # 2️⃣ Feature importance
    importance = getattr(model, "feature_importances_", None)
    if importance is not None:
        fig_fi, ax_fi = plot_feature_importance(importance, feature_names)
        fi_path = os.path.join(plots_dir, "feature_importance.png")
        fig_fi.savefig(fi_path, dpi=200)
        plt.close(fig_fi)
    else:
        fi_path = None

    # 3️⃣ Residual plots
    residuals = y_test - y_pred

    fig_rd, ax_rd = plot_residual_distribution(residuals)
    rd_path = os.path.join(plots_dir, "residual_distribution.png")
    fig_rd.savefig(rd_path, dpi=200)
    plt.close(fig_rd)

    fig_avp, ax_avp = plot_actual_vs_predicted(y_test, y_pred)
    avp_path = os.path.join(plots_dir, "actual_vs_predicted.png")
    fig_avp.savefig(avp_path, dpi=200)
    plt.close(fig_avp)

    fig_rvp, ax_rvp = plot_residuals_vs_predicted(y_pred, residuals)
    rvp_path = os.path.join(plots_dir, "residuals_vs_predicted.png")
    fig_rvp.savefig(rvp_path, dpi=200)
    plt.close(fig_rvp)

    # 4️⃣ PDF report (STYLED LIKE LR)
    accent = "#1e88e5"
    pdf_path = os.path.join(export_path, f"XGB_Report_v{model_version}.pdf")
    
    with PdfPages(pdf_path) as pp:
        # ========== METRICS TABLE ==========
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis("off")

        table = ax.table(
            cellText=[
                ["Model", "MSE", "MAE", "RMSE", "R²"],
                ["XGBoost", f"{mse:.2f}", f"{mae:.2f}", f"{rmse:.2f}", f"{r2:.2f}"],
            ],
            loc="center",
            cellLoc="center",
        )
        table.scale(1, 2)

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor(accent)
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#f0f0f0")

        pp.savefig(fig, facecolor="white")
        plt.close(fig)

        # ========== FEATURE IMPORTANCE ==========
        if importance is not None:
            fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.3)))
            
            sorted_idx = np.argsort(importance)
            sorted_importance = importance[sorted_idx]
            sorted_features = [feature_names[i] for i in sorted_idx]
            
            ax.barh(range(len(sorted_importance)), sorted_importance, color=accent)
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel("Feature Importance", fontsize=11)
            ax.set_title("Feature Importance (XGBoost)", color=accent, fontsize=13, weight='bold', pad=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            pp.savefig(fig, facecolor="white")
            plt.close(fig)

        # ========== RESIDUAL DISTRIBUTION ==========
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, ax=ax, color=accent, edgecolor="black")
        ax.set_title("Residual Distribution (XGBoost)", color=accent, fontsize=13, weight='bold', pad=10)
        ax.set_xlabel("Residual", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        pp.savefig(fig, facecolor="white")
        plt.close(fig)

        # ========== ACTUAL VS PREDICTED ==========
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color=accent, edgecolor="black", linewidth=0.5)
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="Perfect Prediction")
        ax.set_xlabel("Actual Values", fontsize=11)
        ax.set_ylabel("Predicted Values", fontsize=11)
        ax.set_title("Actual vs Predicted Scatter Plot", color=accent, fontsize=13, weight='bold', pad=10)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        pp.savefig(fig, facecolor="white")
        plt.close(fig)

        # ========== RESIDUALS VS PREDICTED ==========
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_pred, residuals, alpha=0.6, color="#e53935", edgecolor="black", linewidth=0.5)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, label="Zero Line")
        ax.set_xlabel("Predicted Values", fontsize=11)
        ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=11)
        ax.set_title("Residuals vs Predicted Values", color="#e53935", fontsize=13, weight='bold', pad=10)
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        pp.savefig(fig, facecolor="white")
        plt.close(fig)

        # ========== VARIABLE DISTRIBUTIONS ==========
        print("   📊 Adding variable distribution pages...")
        for col in indep:
            try:
                col_data = df_valid[col].dropna()
                if len(col_data) == 0:
                    print(f"      ⚠️ No data for {col}, skipping")
                    continue
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(col_data, kde=True, ax=ax, color=accent, edgecolor="black", bins=30)
                ax.set_title(f"Distribution of {col}", color=accent, fontsize=13, weight='bold', pad=10)
                ax.set_xlabel(col, fontsize=11)
                ax.set_ylabel("Frequency", fontsize=11)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                mean_val = col_data.mean()
                median_val = col_data.median()
                std_val = col_data.std()
                stats_text = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}"
                ax.text(0.98, 0.97, stats_text,
                        transform=ax.transAxes,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=accent),
                        fontsize=9)
                
                plt.tight_layout()
                pp.savefig(fig, facecolor="white")
                plt.close(fig)
                print(f"      ✅ Added distribution for {col}")
                
            except Exception as e:
                print(f"      ⚠️ Could not create distribution for {col}: {e}")
                continue
    
    print(f"   ✅ PDF report saved: {pdf_path}")

    # 5️⃣ Save model
    model_path = os.path.join(export_path, f"XGB_model_{model_version}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(
            {
                "model": model,
                "scaler": scaler,
                "features": feature_names,
                "target": target,
                "version": model_version,
                "model_type": "xgb",
                "trained_at": datetime.now().isoformat(),
            },
            f,
        )
    print(f"💾 Saved model: XGB_model_{model_version}.pkl")

    # 6️⃣ Export CSV
    preds_valid = model.predict(
        scaler.transform(df_valid[indep]) if scaler else df_valid[indep].values
    )

    df_valid = df_valid.copy()

    pin_series, _ = extract_pin_column(df_full)

    if pin_series is not None:
        try:
            df_valid["PIN"] = pin_series.iloc[df_valid.index].values
        except Exception as e:
            print("⚠ PIN injection failed:", e)

    df_valid["prediction"] = preds_valid

    cols = []
    if "PIN" in df_valid.columns:
        cols.append("PIN")
    cols.extend(indep)
    cols.append(target)
    cols.append("prediction")

    csv_path = os.path.join(export_path, f"XGB_Training_Result_v{model_version}.csv")
    df_valid[cols].to_csv(csv_path, index=False)
    print(f"✅ Exported CSV: {csv_path}")

    # 7️⃣ Export shapefile
    zip_out = None
    try:
        if '__original_index__' in df_valid.columns:
            original_indices = df_valid['__original_index__'].tolist()
            print(f"🔍 Using original indices: {original_indices[:10]}...")
        else:
            original_indices = df_valid.index.tolist()
            print(f"⚠️ '__original_index__' not found")

        if is_db_mode:
            print("✅ Database mode: fetching geometry")
            gdf_db = gdf_from_db_with_geometry(schema, table_name)
            valid_gdf = gdf_db.iloc[original_indices].copy()
            
            if pin_series is not None:
                upsert_pin_field(valid_gdf, pin_series.iloc[original_indices].values)
            drop_duplicate_pin_fields(valid_gdf)
            
            valid_gdf[target] = df_valid[target].values
            valid_gdf["prediction"] = df_valid["prediction"].values
            
        elif file_gdf is not None:
            print("✅ File mode: using uploaded geometry")
            valid_gdf = file_gdf.iloc[original_indices].copy()
            
            if pin_series is not None:
                try:
                    valid_gdf["PIN"] = pin_series.iloc[original_indices].values
                except Exception as e:
                    print(f"   ⚠️ Could not add PIN: {e}")
            
            valid_gdf[target] = df_valid[target].values
            valid_gdf["prediction"] = df_valid["prediction"].values
        else:
            raise ValueError("No geometry source available")

        shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
        os.makedirs(shp_pred_dir, exist_ok=True)
        shp_pred_path = os.path.join(shp_pred_dir, "XGBoost_Predicted.shp")
        valid_gdf = valid_gdf.drop(columns=['__original_index__'], errors='ignore')
        valid_gdf.to_file(shp_pred_path)

        zip_out = os.path.join(export_path, "XGBoost_Predicted.zip")
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in os.listdir(shp_pred_dir):
                z.write(os.path.join(shp_pred_dir, f), f)

    except Exception as e:
        print(f"⚠️ Shapefile export error: {e}")

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

    try:
        file_gdf = None
        is_db_mode = False

        # 1️⃣ Load data (SAME AS LR)
        if schema and schema.strip() and table_name and table_name.strip():
            is_db_mode = True
            print(f"✅ XGB DB mode: schema={schema}, table={table_name}")
            df_full = df_from_db(schema.strip(), table_name.strip())
        else:
            print(f"✅ XGB File mode detected")
            gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
            file_gdf = gdf.copy()
            df_full = gdf.drop(columns=[c for c in gdf.columns if c.lower() in GEOM_NAMES])

        if df_full.empty:
            return JSONResponse(status_code=400, content={"error": "No data loaded."})

        total_rows_before = len(df_full)

        # 2️⃣ Apply exclusions FIRST (SAME AS LR)
        try:
            excluded = json.loads(excluded_indices or "[]")
            excluded_count = len(excluded)
            if excluded_count:
                print(f"🧹 Excluding {excluded_count} rows before training...")
                df_full = df_full.drop(df_full.index[excluded]).reset_index(drop=True)
            else:
                print("✅ No excluded rows received.")
        except Exception as e:
            print(f"⚠️ Could not parse excluded_indices: {e}")
            excluded_count = 0

        # 🔒 CRITICAL: Store original indices AFTER exclusions (SAME AS LR)
        df_full['__original_index__'] = df_full.index
        print(f"🔍 Stored original indices for {len(df_full)} rows after exclusions")

        pin_series, pin_colname = extract_pin_column(df_full)
        if pin_colname and pin_colname in df_full.columns:
            df_full.drop(columns=[pin_colname], inplace=True)
            print(f"   🔧 Removed PIN column '{pin_colname}'")

        # 3️⃣ Parse fields
        indep = json.loads(independent_vars) if isinstance(independent_vars, str) else independent_vars
        target = dependent_var

        if target not in df_full.columns:
            return JSONResponse(status_code=400, content={"error": f"'{target}' not found"})

        for col in indep:
            if col not in df_full.columns:
                return JSONResponse(status_code=400, content={"error": f"'{col}' not found"})

        # 4️⃣ Convert numeric
        for col in indep + [target]:
            df_full[col] = df_full[col].map(safe_to_float)

        # 5️⃣ Create working dataset (SAME AS LR)
        df_model = df_full[indep + [target] + ['__original_index__']].copy()
        before = len(df_model)
        df_model = df_model.dropna(subset=indep + [target])
        after = len(df_model)
        print(f"📢 XGB dropped {before - after} rows with NaNs")

        if df_model.empty:
            return JSONResponse(status_code=400, content={"error": "No valid rows"})

        print(f"📊 Final training dataset: {len(df_model)} rows")

        # 6️⃣ Prepare X, y
        X = df_model[indep].values
        y = df_model[target].values

        # 7️⃣ Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 8️⃣ Scaling (XGB-SPECIFIC FEATURE)
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

        # 9️⃣ Train XGBoost (XGB-SPECIFIC COMPUTATION)
        print("🚀 Training XGBoost model...")
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )
        model.fit(X_train_scaled, y_train)

        # 🔟 Predictions
        y_pred = model.predict(X_test_scaled)

        # 1️⃣1️⃣ Prepare df_valid
        df_valid = df_model.copy()

        # 1️⃣2️⃣ Export
        model_version = get_next_model_version("xgb")
        export_id = f"xgb_{model_version}"
        export_path = os.path.join(EXPORT_DIR, export_id)
        os.makedirs(export_path, exist_ok=True)
        print(f"📦 Creating export: {export_id} (v{model_version})")

        artifacts = export_xgb_report_and_artifacts(
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
            df_valid=df_valid,
            indep=indep,
            target=target,
            excluded_indices=[],
            is_db_mode=is_db_mode,
            schema=schema,
            table_name=table_name,
            file_gdf=file_gdf,
            model_version=model_version,
        )

        plots = artifacts["plots"]
        downloads = artifacts["downloads"]
        metrics = artifacts["metrics"]
        
        # Interactive data
        residuals = y_test - y_pred
        counts, bin_edges = np.histogram(residuals, bins=20)
        
        print("📊 Computing variable distributions...")
        variable_distributions = compute_variable_distributions(
            pd.DataFrame(X_train_scaled if scaler else X_train, columns=indep),
            indep
        )
        
        # Preview
        print("📋 Creating preview...")
        preview_df = df_valid.copy()
        preds_valid = model.predict(
            scaler.transform(df_valid[indep]) if scaler else df_valid[indep].values
        )
        preview_df["prediction"] = preds_valid
        
        preview_cols = []
        if pin_series is not None:
            try:
                preview_df["PIN"] = pin_series.iloc[preview_df.index].values
                preview_cols.append("PIN")
            except:
                pass
        
        preview_cols.extend(indep)
        preview_cols.append(target)
        preview_cols.append("prediction")
        
        cama_preview = preview_df[preview_cols].head(100).to_dict('records')

        # URLs
        base_url = "/api/ai-tools/download"
        wrapped_plots = {
            key: (f"{base_url}?file={path}" if path else None)
            for key, path in plots.items()
        }

        wrapped_downloads = {
            key: f"{base_url}?file={path}"
            for key, path in downloads.items()
            if path
        }

        if "shapefile" in downloads and downloads["shapefile"]:
            wrapped_downloads["geojson"] = f"/api/ai-tools/preview-geojson?file_path={downloads['shapefile']}"

        return {
            "model_version": model_version,
            "model_id": export_id,
            "message": "XGBoost training completed",
            "dependent_var": target,
            "metrics": metrics,
            "features": indep,
            "importance": [
                {"feature": feat, "value": float(val)}
                for feat, val in zip(indep, model.feature_importances_)
            ] if hasattr(model, "feature_importances_") else [],
            "interactive_data": {
                "residuals": residuals.tolist(),
                "residual_bins": bin_edges.tolist(),
                "residual_counts": counts.tolist(),
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
        print(f"❌ XGB TRAIN ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})