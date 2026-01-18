# rf_train.py
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile, os, pickle, json, zipfile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

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

def plot_rf_feature_importance(importance: np.ndarray, feature_names: List[str], ax=None):
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
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    return fig, ax


def plot_rf_residual_distribution(residuals: np.ndarray, ax=None):
    """Plot the distribution of residuals with a histogram + KDE."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Residual Distribution (Random Forest)")
    ax.set_xlabel("Residual")
    fig.tight_layout()
    return fig, ax


def plot_rf_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, ax=None):
    """Scatter plot: actual vs predicted."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.scatter(y_test, y_pred, alpha=0.6)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (Random Forest)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_rf_residuals_vs_predicted(y_pred: np.ndarray, residuals: np.ndarray, ax=None):
    """Scatter plot of residuals vs predicted."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted (Random Forest)")
    fig.tight_layout()
    return fig, ax


def export_rf_report_and_artifacts(
    export_path: str,
    model: RandomForestRegressor,
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
    """
    Export Random Forest model artifacts.
    """
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
        fig_fi, ax_fi = plot_rf_feature_importance(importance, feature_names)
        fi_path = os.path.join(plots_dir, "feature_importance.png")
        fig_fi.savefig(fi_path, dpi=200)
        plt.close(fig_fi)
    else:
        fi_path = None

    # 3️⃣ Residual plots
    residuals = y_test - y_pred

    fig_rd, ax_rd = plot_rf_residual_distribution(residuals)
    rd_path = os.path.join(plots_dir, "residual_distribution.png")
    fig_rd.savefig(rd_path, dpi=200)
    plt.close(fig_rd)

    fig_avp, ax_avp = plot_rf_actual_vs_predicted(y_test, y_pred)
    avp_path = os.path.join(plots_dir, "actual_vs_predicted.png")
    fig_avp.savefig(avp_path, dpi=200)
    plt.close(fig_avp)

    fig_rvp, ax_rvp = plot_rf_residuals_vs_predicted(y_pred, residuals)
    rvp_path = os.path.join(plots_dir, "residuals_vs_predicted.png")
    fig_rvp.savefig(rvp_path, dpi=200)
    plt.close(fig_rvp)

    # 4️⃣ PDF report (styled like LR - direct plotting!)
    accent = "#1e88e5"
    pdf_path = os.path.join(export_path, f"RF_Report_v{model_version}.pdf")
    
    with PdfPages(pdf_path) as pp:
        # ========== METRICS TABLE ==========
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

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor(accent)
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#f0f0f0")

        pp.savefig(fig, facecolor="white")
        plt.close(fig)

        # ========== FEATURE IMPORTANCE (DIRECT PLOT) ==========
        if importance is not None:
            fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.3)))
            
            sorted_idx = np.argsort(importance)
            sorted_importance = importance[sorted_idx]
            sorted_features = [feature_names[i] for i in sorted_idx]
            
            ax.barh(range(len(sorted_importance)), sorted_importance, color=accent)
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel("Feature Importance", fontsize=11)
            ax.set_title("Feature Importance", color=accent, fontsize=13, weight='bold', pad=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            pp.savefig(fig, facecolor="white")
            plt.close(fig)

        # ========== RESIDUAL DISTRIBUTION (DIRECT PLOT) ==========
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, ax=ax, color=accent, edgecolor="black")
        ax.set_title("Residual Distribution", color=accent, fontsize=13, weight='bold', pad=10)
        ax.set_xlabel("Residual", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        pp.savefig(fig, facecolor="white")
        plt.close(fig)

        # ========== ACTUAL VS PREDICTED (DIRECT PLOT) ==========
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

        # ========== RESIDUALS VS PREDICTED (DIRECT PLOT) ==========
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

        # ========== PER VARIABLE DISTRIBUTIONS (DIRECT PLOT) ==========
        print("   📊 Adding variable distribution pages...")
        for col in indep:
            try:
                # Use df_valid data (what was actually used for training)
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
                
                # Add stats box
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
    model_path = os.path.join(export_path, f"RF_model_{model_version}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": feature_names,
                "target": target,
                "version": model_version,  # ✅ Store version
                "model_type": "rf",
                "trained_at": datetime.now().isoformat(),
            },
            f,
        )
    print(f"💾 Saved model: RF_model_{model_version}.pkl")

    # 6️⃣ Export CSV with predictions
    preds_valid = model.predict(df_valid[indep].values)
    df_valid = df_valid.copy()

    # Extract PIN from full dataset
    pin_series, _ = extract_pin_column(df_full)

    # Add PIN to df_valid
    if pin_series is not None:
        try:
            df_valid["PIN"] = pin_series.iloc[df_valid.index].values
            print("   ✅ Added PIN to df_valid")
        except Exception as e:
            print(f"   ⚠️ PIN injection failed: {e}")

    # Add predictions
    df_valid["prediction"] = preds_valid

    # Build CSV columns
    cols = []
    if "PIN" in df_valid.columns:
        cols.append("PIN")
    cols.extend(indep)
    cols.append(target)
    cols.append("prediction")

    csv_path = os.path.join(export_path, f"RF_Training_Result_v{model_version}.csv")
    df_valid[cols].to_csv(csv_path, index=False)
    print(f"✅ Exported CSV: {csv_path}")

    # 7️⃣ Export shapefile/ZIP
    zip_out = None
    try:
        # 🔑 Get original indices from df_valid
        if '__original_index__' in df_valid.columns:
            original_indices = df_valid['__original_index__'].tolist()
            print(f"📍 Using original indices: {original_indices[:10]}... (showing first 10)")
        else:
            # Fallback to current indices
            original_indices = df_valid.index.tolist()
            print(f"⚠️ '__original_index__' not found, using current indices")

        if is_db_mode:
            print("✅ Database mode: fetching geometry")
            gdf_db = gdf_from_db_with_geometry(schema, table_name)
            
            # Use original indices
            valid_gdf = gdf_db.iloc[original_indices].copy()
            print(f"   📊 valid_gdf shape: {valid_gdf.shape}")
            
            # Add PIN
            if pin_series is not None:
                upsert_pin_field(valid_gdf, pin_series.iloc[original_indices].values)
            drop_duplicate_pin_fields(valid_gdf)
            
            # ✅ Add actual values
            valid_gdf[target] = df_valid[target].values
            print(f"   ✅ Added actual values ({target})")
            
            # Add predictions
            valid_gdf["prediction"] = df_valid["prediction"].values
            print("   ✅ Added predictions")

        elif file_gdf is not None:
            print("✅ File mode: using uploaded geometry")
            
            # Use original indices
            valid_gdf = file_gdf.iloc[original_indices].copy()
            print(f"   📊 valid_gdf shape: {valid_gdf.shape}")
            
            # Add PIN
            if pin_series is not None:
                try:
                    valid_gdf["PIN"] = pin_series.iloc[original_indices].values
                    print("   ✅ Added PIN")
                except Exception as e:
                    print(f"   ⚠️ Could not add PIN: {e}")
            
            # ✅ Add actual values
            valid_gdf[target] = df_valid[target].values
            print(f"   ✅ Added actual values ({target})")
            
            # Add predictions
            valid_gdf["prediction"] = df_valid["prediction"].values
            print("   ✅ Added predictions")

        else:
            raise ValueError("No geometry source available")

        # Verify data before saving
        print(f"\n{'='*60}")
        print(f"🔍 VERIFICATION:")
        print(f"   Rows in valid_gdf: {len(valid_gdf)}")
        print(f"   '{target}' exists: {target in valid_gdf.columns}")
        print(f"   Sample actual values: {valid_gdf[target].head().tolist()}")
        print(f"   'prediction' exists: {'prediction' in valid_gdf.columns}")
        print(f"   Sample predictions: {valid_gdf['prediction'].head().tolist()}")
        print(f"{'='*60}\n")

        shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
        os.makedirs(shp_pred_dir, exist_ok=True)
        shp_pred_path = os.path.join(shp_pred_dir, "RandomForest_Predicted.shp")
        valid_gdf = valid_gdf.drop(columns=['__original_index__'], errors='ignore')
        valid_gdf.to_file(shp_pred_path)
        print(f"   ✅ Shapefile saved: {shp_pred_path}")

        # Create ZIP
        zip_out = os.path.join(export_path, "RandomForest_Predicted.zip")
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in os.listdir(shp_pred_dir):
                z.write(os.path.join(shp_pred_dir, f), f)
        print(f"   ✅ ZIP created: {zip_out}")

        # Optional: save to database
        if is_db_mode and schema:
            try:
                predicted_table = f"{table_name}_RF_Predicted"
                provincial_code = get_provincial_code_from_schema(schema)
                db_session_save = get_user_database_session(provincial_code)
                engine_save = db_session_save.get_bind()
                
                valid_gdf.to_postgis(
                    name=predicted_table,
                    con=engine_save,
                    schema=schema,
                    if_exists="replace",
                    index=False,
                )
                
                engine_save.dispose()
                db_session_save.close()
                print(f"   ✅ Saved to {schema}.{predicted_table}")
            except Exception as e:
                print(f"   ⚠️ Could not save to DB: {e}")

    except Exception as e:
        print(f"⚠️ Shapefile export error: {e}")
        import traceback
        traceback.print_exc()
        zip_out = None

    # Collect paths for frontend
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
async def train_random_forest_model(
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

        # 1️⃣ Detect input mode and load data
        if schema and schema.strip() and table_name and table_name.strip():
            is_db_mode = True
            print(f"✅ RF DB mode: schema={schema}, table={table_name}")
            df_full = df_from_db(schema.strip(), table_name.strip())
        else:
            print("✅ RF File mode detected")
            gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
            file_gdf = gdf.copy()
            df_full = gdf.drop(
                columns=[c for c in gdf.columns if c.lower() in GEOM_NAMES],
                errors="ignore",
            )

        if df_full.empty:
            return JSONResponse(
                status_code=400, content={"error": "No data loaded."}
            )

        # 🔑 CRITICAL: Store original indices BEFORE any filtering
        df_full['__original_index__'] = df_full.index
        print(f"📍 Stored original indices for {len(df_full)} rows")

        # Extract PIN early
        pin_series, pin_colname = extract_pin_column(df_full)
        
        # Remove PIN from training features if present
        if pin_colname and pin_colname in df_full.columns:
            df_full = df_full.drop(columns=[pin_colname])
            print(f"   🔧 Removed PIN column '{pin_colname}' from training features")

        # 2️⃣ Parse fields
        indep = (
            json.loads(independent_vars)
            if isinstance(independent_vars, str)
            else independent_vars
        )
        target = dependent_var

        if target not in df_full.columns:
            return JSONResponse(
                status_code=400,
                content={"error": f"Dependent variable '{target}' not found in data."},
            )

        for col in indep:
            if col not in df_full.columns:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Independent variable '{col}' not found in data."},
                )

        # 3️⃣ Convert numeric columns using safe_to_float
        for col in indep + [target]:
            df_full[col] = df_full[col].map(safe_to_float)

        # 4️⃣ Create working dataset with selected columns + original index tracker
        df_model = df_full[indep + [target] + ['__original_index__']].copy()
        
        # Drop rows with NaNs in training columns
        before = len(df_model)
        df_model = df_model.dropna(subset=indep + [target])
        after = len(df_model)
        print(f"🔢 RF dropped {before - after} rows with NaNs in selected columns.")

        if df_model.empty:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid rows after cleaning."},
            )

        # 5️⃣ Handle excluded indices
        try:
            excluded_list = (
                json.loads(excluded_indices) if excluded_indices else []
            )
            if not isinstance(excluded_list, list):
                excluded_list = []
        except Exception:
            excluded_list = []

        # Ensure excluded_list contains integers
        excluded_list = [
            int(i)
            for i in excluded_list
            if isinstance(i, int) or (isinstance(i, str) and i.isdigit())
        ]

        if excluded_list:
            # Filter out rows whose __original_index__ is in excluded_list
            excluded_mask = df_model['__original_index__'].isin(excluded_list)
            excluded_count = excluded_mask.sum()
            df_model = df_model[~excluded_mask].copy()
            print(f"🧹 RF excluding {excluded_count} rows before training...")
        else:
            print("✅ RF: no excluded rows received.")

        if df_model.empty:
            return JSONResponse(
                status_code=400,
                content={"error": "All rows were excluded. Nothing to train."},
            )

        print(f"📊 Final training dataset: {len(df_model)} rows")
        print(f"📊 Original indices range: {df_model['__original_index__'].min()} to {df_model['__original_index__'].max()}")

        # 6️⃣ Prepare X, y
        X = df_model[indep].values
        y = df_model[target].values

        # 7️⃣ Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # 8️⃣ Train Random Forest model
        print("🚀 Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # 9️⃣ Predictions on test set
        y_pred = model.predict(X_test)

        # 🔟 Prepare df_valid (cleaned dataset used for training)
        # This preserves __original_index__ for geometry mapping
        df_valid = df_model.copy()

        # 1️⃣1️⃣ Export artifacts
        model_version = get_next_model_version("rf")
        export_id = f"rf_{model_version}"
        export_path = os.path.join(EXPORT_DIR, export_id)
        os.makedirs(export_path, exist_ok=True)

        print(f"📦 Creating export folder: {export_id} (Version {model_version})")
        artifacts = export_rf_report_and_artifacts(
            export_path=export_path,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            feature_names=indep,
            df_full=df_full,
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

        plots_paths = artifacts["plots"]
        downloads_paths = artifacts["downloads"]
        metrics = artifacts["metrics"]

        # 1️⃣2️⃣ Interactive data for frontend
        residuals = y_test - y_pred
        counts, bin_edges = np.histogram(residuals, bins=20)
        residual_bins = bin_edges.tolist()
        residual_counts = counts.tolist()

        y_test_array = y_test.tolist()
        preds_array = y_pred.tolist()

        # 🆕 Compute variable distributions
        print("📊 Computing variable distributions for RF...")
        variable_distributions = compute_variable_distributions(
            pd.DataFrame(X_train, columns=indep),
            indep
        )
        print(f"✅ Computed distributions for {len(variable_distributions)} variables")

        # 🆕 CREATE TRAINING RESULT PREVIEW (with PIN)
        print("📋 Creating training result preview...")
        preview_df = df_valid.copy()
        
        # Add predictions
        preds_valid = model.predict(df_valid[indep].values)
        preview_df["prediction"] = preds_valid
        
        # Build preview columns
        preview_cols = []
        if pin_series is not None:
            try:
                preview_df["PIN"] = pin_series.iloc[preview_df.index].values
                preview_cols.append("PIN")
                print("   ✅ Added PIN to preview")
            except Exception as e:
                print(f"   ⚠️ Could not add PIN to preview: {e}")
        
        preview_cols.extend(indep)
        preview_cols.append(target)
        preview_cols.append("prediction")
        
        # Take first 100 rows for preview
        cama_preview = preview_df[preview_cols].head(100).to_dict('records')
        print(f"   ✅ Created preview with {len(cama_preview)} rows")

        # 1️⃣3️⃣ Wrap paths into API URLs
        base_url = "/api/ai-tools/download"
        geojson_url = "/api/ai-tools/preview-geojson"

        downloads = {}
        for key, path in downloads_paths.items():
            if not path:
                continue
            downloads[key] = f"{base_url}?file={path}"

        # Add GeoJSON preview if shapefile exists
        shp = downloads_paths.get("shapefile")
        if shp:
            downloads["geojson"] = f"{geojson_url}?file_path={shp}"

        # Wrap plot paths
        plots = {
            name: (f"{base_url}?file={p}" if p else None)
            for name, p in plots_paths.items()
        }

        return {
            "model_version": model_version,
            "model_id": export_id,     
            "message": "RANDOM FOREST COMPLETE!!",
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
                "y_test": y_test_array,
                "preds": preds_array,
            },
            "variable_distributions": variable_distributions,
            "cama_preview": cama_preview,
            "plots": plots,
            "downloads": downloads,
            "is_db_mode": is_db_mode,
            "isRunMode": False,
            "record_count": int(len(df_model)),
        }

    except Exception as e:
        import traceback
        print(f"❌ RF TRAIN ERROR: {e}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})