from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Tuple, Dict, Any
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile, os, joblib, json, zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from scipy import stats
from sqlalchemy import text
from AITools.pdf_summary_generator import generate_model_summary_page
from db import get_user_database_session
from AITools.ai_utils import (
    extract_pin_column,
    compute_variable_distributions,
    get_next_model_version,
    upsert_pin_field,
    drop_duplicate_pin_fields,
)

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)

def wrap_plot_urls(plots: Dict[str, Optional[str]], prefix: str) -> Dict[str, Optional[str]]:
    return {
        key: (f"{prefix}?file={path}" if path else None)
        for key, path in plots.items()
    }

def get_provincial_code_from_schema(schema: str) -> str:
    """PH0403406_Calauan -> PH04034 ; PH0402118_Silang -> PH04021"""
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
    """Load table from PostGIS, excluding geometry-like columns."""
    provincial_code = get_provincial_code_from_schema(schema)
    db_session = get_user_database_session(provincial_code)
    try:
        # collect columns first
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
            # fallback * (let SQL raise meaningful error if table is empty)
            keep_sql = "*"
        else:
            keep_sql = ", ".join(f'"{c}"' for c in keep)

        rows = db_session.execute(
            text(f'SELECT {keep_sql} FROM "{schema}"."{table}"')
        ).fetchall()

        # if we used * then fields order must be recomputed:
        if keep_sql == "*":
            df = pd.DataFrame(rows, columns=colnames)
            # still drop geometry if present
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
        # 1) Check if the requested table has geometry
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
            # Table has geometry - load directly
            print(f"✅ {table} has geometry column: {geom_check[0]}")
            sql = f'SELECT * FROM "{schema}"."{table}"'
            gdf = gpd.read_postgis(sql, engine, geom_col=geom_check[0])
            return gdf
        
        # 2) No geometry - find spatial tables in schema
        print(f"ℹ️ {table} has no geometry. Searching for spatial tables in {schema}...")
        
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
        print(f"📍 Found spatial tables: {list(spatial_tables.keys())}")
        
        # 3) Get column names from target table for join key detection
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
        
        # Common join key patterns (case-insensitive)
        join_key_candidates = [
            "PIN", "ARPN", "ARP_PIN", "TD_NO", "PROPERTY_ID", 
            "PARCEL_ID", "TAX_DEC_NO", "OID", "OBJECTID", "FID"
        ]
        
        # 4) Try to find matching spatial table and join key
        for spatial_table, geom_col in spatial_tables.items():
            # Get columns from spatial table
            spatial_cols_rows = db_session.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :table
                """),
                {"schema": schema, "table": spatial_table}
            ).fetchall()
            spatial_cols = {row[0].upper() for row in spatial_cols_rows}
            
            # Find common join keys
            common_keys = []
            for key in join_key_candidates:
                if key in target_cols and key in spatial_cols:
                    common_keys.append(key)
            
            if common_keys:
                join_key = common_keys[0]  # Use first matching key
                print(f"✅ Found join key '{join_key}' between {table} and {spatial_table}")
                
                # Perform JOIN
                join_sql = f'''
                    SELECT 
                        t.*, 
                        s."{geom_col}" as geometry
                    FROM "{schema}"."{table}" t
                    INNER JOIN "{schema}"."{spatial_table}" s
                      ON t."{join_key}" = s."{join_key}"
                '''
                
                print(f"🔗 Joining {table} with {spatial_table} on {join_key}")
                gdf = gpd.read_postgis(join_sql, engine, geom_col="geometry")
                print(f"✅ Successfully joined! Result has {len(gdf)} rows")
                return gdf
        
        # 5) No suitable join found
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


def export_full_report_and_artifacts(
    export_path: str,
    model,
    scaler,
    independent_vars: List[str],
    target: str,
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
    preds: np.ndarray,
    residuals: pd.Series,
    X_train_unscaled: pd.DataFrame = None,
    model_version: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:

    mse = float(np.mean((y_test - preds) ** 2))
    mae = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(mse))
    # R2 from sklearn r2_score for consistency
    from sklearn.metrics import r2_score
    r2 = float(r2_score(y_test, preds))

    # standardized importance
    std_X = np.std(X_train_scaled, axis=0)
    std_y = np.std(y_train)
    with np.errstate(divide="ignore", invalid="ignore"):
        importance = np.where(std_y == 0, 0, model.coef_ * std_X / std_y)

    # residual t-test
    t_stat, p_val = stats.ttest_1samp(residuals, 0)
    residual_ttest = {"t_stat": float(t_stat), "p_value": float(p_val)}

    n = len(y_train)
    k = len(independent_vars)
    dof = n - k - 1
    
    residual_std_error = np.sqrt(np.sum(residuals ** 2) / dof)
    
    try:
        XtX_inv = np.linalg.inv(X_train_scaled.T @ X_train_scaled)
        var_coef = residual_std_error ** 2 * np.diag(XtX_inv)
        std_errors = np.sqrt(var_coef)
        
        t_stats = model.coef_ / std_errors
        
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
        
        intercept_var = residual_std_error ** 2 * (1/n + np.mean(X_train_scaled, axis=0) @ XtX_inv @ np.mean(X_train_scaled, axis=0).T)
        intercept_std_err = np.sqrt(intercept_var)
        intercept_t = model.intercept_ / intercept_std_err
        intercept_p = 2 * (1 - stats.t.cdf(np.abs(intercept_t), dof))
        coef_ttests = []
        coef_ttests.append({
            "variable": "Intercept",
            "coef": float(model.intercept_),
            "std_err": float(intercept_std_err),
            "t": float(intercept_t),
            "p": float(intercept_p),
            "significant": bool(intercept_p < 0.05) 
        })
        for i, var in enumerate(independent_vars):
            coef_ttests.append({
                "variable": var,
                "coef": float(model.coef_[i]),
                "std_err": float(std_errors[i]),
                "t": float(t_stats[i]),
                "p": float(p_values[i]),
                "significant": bool(p_values[i] < 0.05)
            })
    except Exception as e:
        print(f"⚠️ Could not calculate coefficient t-tests: {e}")
        coef_ttests = None

    metrics = {
        "r2": float(r2),
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
    }
    
    accent = "#1e88e5"
    png_paths: Dict[str, str] = {}

    pdf_path = os.path.join(export_path, f"LR_Report_v{model_version}.pdf")
    with PdfPages(pdf_path) as pp:
        # Metrics table
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis("off")
        table = ax.table(
            cellText=[["Model", "MSE", "MAE", "RMSE", "R²"],
                      ["Linear Regression", f"{mse:.2f}", f"{mae:.2f}", f"{rmse:.2f}", f"{r2:.2f}"]],
            loc="center", cellLoc="center",
        )
        table.scale(1, 2)
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor(accent)
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0')
        pp.savefig(fig, facecolor="white")
        metrics_png = os.path.join(export_path, "metrics_table.png")
        fig.savefig(metrics_png, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        png_paths["metrics"] = metrics_png

        # Feature importance
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(independent_vars, importance, color=accent)
        ax.set_ylabel("Standardized Coefficient")
        ax.set_title("Feature Importance", color=accent, fontsize=13, weight='bold', pad=10)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        pp.savefig(fig, facecolor="white")
        fi_png = os.path.join(export_path, "feature_importance.png")
        fig.savefig(fi_png, bbox_inches="tight", facecolor="white"); plt.close(fig)
        png_paths["feature_importance"] = fi_png

        # Residual distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, ax=ax, color=accent, edgecolor="black")
        ax.set_title("Residual Distribution (Normal Curve)", color=accent, fontsize=13, weight='bold', pad=10)
        ax.set_xlabel("Residual"); ax.set_ylabel("Frequency")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); pp.savefig(fig, facecolor="white")
        resid_png = os.path.join(export_path, "residual_distribution.png")
        fig.savefig(resid_png, bbox_inches="tight", facecolor="white"); plt.close(fig)
        png_paths["residual_distribution"] = resid_png

        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, preds, alpha=0.6, color=accent, edgecolor="black", linewidth=0.5)
        minv = min(min(y_test), min(preds))
        maxv = max(max(y_test), max(preds))
        ax.plot([minv, maxv], [minv, maxv], "k--", lw=1.5, label="Perfect Prediction")
        ax.set_xlabel("Actual Values"); ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Scatter Plot", color=accent, fontsize=13, weight='bold', pad=10)
        ax.legend(); plt.tight_layout(); pp.savefig(fig, facecolor="white")
        scatter_png = os.path.join(export_path, "actual_vs_predicted.png")
        fig.savefig(scatter_png, bbox_inches="tight", facecolor="white"); plt.close(fig)
        png_paths["actual_vs_predicted"] = scatter_png

        # Residuals vs Predicted
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(preds, residuals, alpha=0.6, color="#e53935", edgecolor="black", linewidth=0.5)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, label="Zero Line")
        ax.set_xlabel("Predicted Values"); ax.set_ylabel("Residuals (Actual - Predicted)")
        ax.set_title("Residuals vs Predicted Values", color="#e53935", fontsize=13, weight='bold', pad=10)
        ax.legend(); plt.tight_layout(); pp.savefig(fig, facecolor="white")
        resid_pred_png = os.path.join(export_path, "residuals_vs_predicted.png")
        fig.savefig(resid_pred_png, bbox_inches="tight", facecolor="white"); plt.close(fig)
        png_paths["residuals_vs_predicted"] = resid_pred_png

        # Residual t-test page
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5,
                f"T-test on Residuals:\nT-statistic = {t_stat:.4f}\nP-value = {p_val:.4f}",
                fontsize=12, ha="center", va="center", color="black",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=accent, edgecolor="black", alpha=0.2))
        pp.savefig(fig, facecolor="white")
        plt.close(fig)
        
        if coef_ttests:
            fig, ax = plt.subplots(figsize=(10, max(3, len(coef_ttests) * 0.4)))
            ax.axis("off")
            
            # Prepare table data
            table_data = [["Variable", "Coefficient", "Std Error", "t-statistic", "p-value", "Significant"]]
            for test in coef_ttests:
                sig_marker = "✓" if test["significant"] else "✗"
                table_data.append([
                    test["variable"],
                    f"{test['coef']:.6f}",
                    f"{test['std_err']:.6f}",
                    f"{test['t']:.4f}",
                    f"{test['p']:.4f}",
                    sig_marker
                ])
            
            # Create table
            table = ax.table(
                cellText=table_data,
                loc="center",
                cellLoc="center",
            )
            table.scale(1, 2)
            
            # Style header row
            for j in range(6):
                cell = table[(0, j)]
                cell.set_facecolor(accent)
                cell.set_text_props(weight='bold', color='white', size=10)
            
            # Style data rows
            for i in range(1, len(table_data)):
                for j in range(6):
                    cell = table[(i, j)]
                    if j == 5:  # Significant column
                        if table_data[i][5] == "✓":
                            cell.set_facecolor('#c8e6c9')  # light green
                        else:
                            cell.set_facecolor('#ffccbc')  # light red
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else '#ffffff')
                    cell.set_text_props(size=9)
            
            ax.set_title("Coefficient T-Tests (α = 0.05)", 
                        fontsize=14, weight='bold', color=accent, pad=20)
            
            pp.savefig(fig, facecolor="white")
            plt.close(fig)

        # Per independent var distributions
        if X_train_unscaled is not None:
            for col in independent_vars:
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    col_data = X_train_unscaled[col].dropna()
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
                    
                except Exception as e:
                    print(f"⚠️ Could not create distribution for {col}: {e}")
                    # Fallback: create a simple text placeholder
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.axis('off')
                    ax.text(0.5, 0.5, f"Distribution of {col}\n(Unable to render)",
                        ha="center", va="center", fontsize=12)
                    pp.savefig(fig, facecolor="white")
                    plt.close(fig)
                
            pdf_metrics = {
                "R²": metrics["r2"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "MSE": metrics["mse"],
            }
                    
            generate_model_summary_page(
                pp=pp,
                model_type="Linear Regression",
                metrics=pdf_metrics,
                features=independent_vars,
                importance_values=importance,
                target_variable=target,
                n_samples=len(y_train),
                accent_color=accent
            )

    t_tests = {"residuals": residual_ttest, "coefficients": coef_ttests}
    return metrics, png_paths, t_tests, pdf_path

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
    try:
        file_gdf = None
        is_db_mode = False 
        if schema and schema.strip() and table_name and table_name.strip():
            is_db_mode = True
            print(f"✅ Database mode detected: schema={schema}, table={table_name}")
            df_full = df_from_db(schema.strip(), table_name.strip())
        else:
            print(f"✅ File mode detected")
            gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
            file_gdf = gdf.copy()
            df_full = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

        total_rows_before = len(df_full)

        # 2️⃣ Apply exclusions FIRST (before storing original index)
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

        # 🔑 CRITICAL: Store original indices AFTER exclusions but BEFORE any other filtering
        df_full['__original_index__'] = df_full.index
        print(f"📍 Stored original indices for {len(df_full)} rows after exclusions")

        # Parse variables
        if independent_vars.startswith("["):
            indep = json.loads(independent_vars)
        else:
            indep = [v.strip() for v in independent_vars.split(",")]
        indep = [v for v in indep if v]
        target = dependent_var.strip()

        lower_map = {c.lower(): c for c in df_full.columns}
        missing = [v for v in indep + [target] if v.lower() not in lower_map]
        if missing:
            return JSONResponse(status_code=400, content={"error": f"Missing variables: {missing}"})
        
        df_full.columns = [c.lower() for c in df_full.columns]
        pin_series, pin_colname = extract_pin_column(df_full)
        indep = [v.lower() for v in indep]
        target = target.lower()

        for col in indep + [target]:
            df_full[col] = df_full[col].map(safe_to_float)

        df_valid = df_full.dropna(subset=indep + [target])
        if df_valid.empty:
            return JSONResponse(status_code=400, content={"error": "No valid numeric data found."})

        # Train/test split
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        X = df_valid[indep]
        y = df_valid[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        residuals = y_test - preds

        model_version = get_next_model_version("linear")
        export_id = f"linear_{model_version}"
        export_path = os.path.join(EXPORT_DIR, export_id)
        os.makedirs(export_path, exist_ok=True)

        print(f"📦 Creating export folder: {export_id} (Version {model_version})")

        model_path = os.path.join(export_path, f"LR_model_{model_version}.pkl")
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "features": [v.lower() for v in indep],
                "dependent_var": target.lower(),
                "version": model_version,  # ✅ Store version
                "model_type": "lr",
                "trained_at": datetime.now().isoformat(),
            },
            model_path,
        )
        print(f"💾 Saved model: LR_model_{model_version}.pkl")
        # Generate report
        metrics, png_paths, t_tests, pdf_path = export_full_report_and_artifacts(
            export_path, model, scaler, indep, target,
            X_train_scaled, y_train, X_test_scaled, y_test, preds, residuals,
            X_train_unscaled=X_train,
            model_version=model_version,
        )

        # Export CSV with predictions
        preds_valid = model.predict(scaler.transform(df_valid[indep]))
        df_valid = df_valid.copy()
        df_valid["prediction"] = preds_valid

        # Needed for shapefile export + response payload
        safe_target_name = "actual_val" if len(target) > 10 else target

        # Define output CSV path
        csv_path = os.path.join(export_path, f"LR_Training_Result_v{model_version}.csv")

        csv_df = df_valid[indep + [target, "prediction"]].copy()

        if pin_series is not None:
            csv_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)

        csv_df.to_csv(csv_path, index=False)
        print(f"✅ Exported cleaned CSV (excluded rows removed): {csv_path}")

        # 9️⃣ Export shapefile/ZIP (only trained rows)
        zip_out = None
        try:
            # 🔑 Get ORIGINAL indices
            original_indices = df_valid['__original_index__'].tolist()
            print(f"📍 Original indices to map: {original_indices[:10]}... (showing first 10)")

            if is_db_mode:
                # ============ DATABASE MODE ============
                print("✅ Database mode: fetching geometry for export")
                gdf_db = gdf_from_db_with_geometry(schema, table_name)
                
                # Use ORIGINAL indices (iloc is position-based)
                valid_gdf = gdf_db.iloc[original_indices].copy()
                
                print(f"   📊 GeoDataFrame shape: {valid_gdf.shape}")
                print(f"   📊 Valid rows count: {len(df_valid)}")
                
                # 🔥 DROP ORIGINAL UNIT_VALUE/MARKET_VAL if exists
                columns_to_drop = []
                for col in valid_gdf.columns:
                    if col.upper() == 'UNIT_VALUE':
                        columns_to_drop.append(col)
                        print(f"   🗑️ Dropping original column '{col}' to avoid collision")
                    elif col.upper() == 'MARKET_VAL':
                        columns_to_drop.append(col)
                        print(f"   🗑️ Dropping original column '{col}'")
                
                if columns_to_drop:
                    valid_gdf = valid_gdf.drop(columns=columns_to_drop, errors='ignore')
                
                if pin_series is not None:
                    try:
                        upsert_pin_field(valid_gdf, pin_series.iloc[original_indices].values, preferred_name="PIN")
                        drop_duplicate_pin_fields(valid_gdf, keep_name="PIN")
                        print("   ✅ PIN field updated (no duplicates)")
                    except Exception as e:
                        print(f"   ⚠️ Could not update PIN field: {e}")
                
                # ✅ ADD ACTUAL VALUES using safe field name
                valid_gdf[safe_target_name] = df_valid[target].values
                print(f"   ✅ Added actual values as '{safe_target_name}'")
                
                # Add predictions
                valid_gdf["prediction"] = df_valid["prediction"].values
                print(f"   ✅ Added prediction column")
                
                # Verify data
                print(f"\n{'='*60}")
                print(f"🔍 VERIFICATION:")
                print(f"   Rows in valid_gdf: {len(valid_gdf)}")
                print(f"   '{safe_target_name}' exists: {safe_target_name in valid_gdf.columns}")
                print(f"   Sample actual values: {valid_gdf[safe_target_name].head().tolist()}")
                print(f"   'prediction' exists: {'prediction' in valid_gdf.columns}")
                print(f"   Sample predictions: {valid_gdf['prediction'].head().tolist()}")
                print(f"{'='*60}\n")
                
                # Export shapefile
                shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
                os.makedirs(shp_pred_dir, exist_ok=True)
                shp_pred_path = os.path.join(shp_pred_dir, "predicted_output.shp")
                
                # Remove helper column before saving
                valid_gdf = valid_gdf.drop(columns=['__original_index__'], errors='ignore')
                valid_gdf.to_file(shp_pred_path)
                print(f"   ✅ Shapefile saved: {shp_pred_path}")
                
                # Create ZIP
                zip_out = os.path.join(export_path, "predicted_output.zip")
                with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in os.listdir(shp_pred_dir):
                        z.write(os.path.join(shp_pred_dir, f), f)
                print(f"   ✅ ZIP created: {zip_out}")
            
            elif file_gdf is not None:
                # ============ FILE MODE ============
                print("✅ File mode: using uploaded geometry for export")
                
                # Use ORIGINAL indices
                valid_gdf = file_gdf.iloc[original_indices].copy()
                
                print(f"   📊 GeoDataFrame shape: {valid_gdf.shape}")
                print(f"   📊 Columns: {valid_gdf.columns.tolist()}")
                
                # 🔥 DROP ORIGINAL UNIT_VALUE/MARKET_VAL if exists (to avoid collision)
                columns_to_drop = []
                for col in valid_gdf.columns:
                    if col.upper() == 'UNIT_VALUE':
                        columns_to_drop.append(col)
                        print(f"   🗑️ Dropping original column '{col}' to avoid collision")
                    elif col.upper() == 'MARKET_VAL':
                        columns_to_drop.append(col)
                        print(f"   🗑️ Dropping original column '{col}' (too large values)")
                
                if columns_to_drop:
                    valid_gdf = valid_gdf.drop(columns=columns_to_drop, errors='ignore')
                
                if pin_series is not None:
                    try:
                        upsert_pin_field(valid_gdf, pin_series.iloc[original_indices].values, preferred_name="PIN")
                        drop_duplicate_pin_fields(valid_gdf, keep_name="PIN")
                        print("   ✅ PIN field updated (no duplicates)")
                    except Exception as e:
                        print(f"   ⚠️ Could not update PIN field: {e}")
                
                # ✅ ADD ACTUAL VALUES using safe field name
                # Now safe_target_name will be used without collision
                valid_gdf[safe_target_name] = df_valid[target].values
                print(f"   ✅ Added actual values as '{safe_target_name}'")
                
                # Verify the column exists
                print(f"   🔍 Checking column '{safe_target_name}': {safe_target_name in valid_gdf.columns}")
                print(f"   🔍 Sample values: {valid_gdf[safe_target_name].head(3).tolist()}")
                
                # Add predictions
                valid_gdf["prediction"] = df_valid["prediction"].values
                print(f"   ✅ Added prediction column")
                
                print(f"\n   📋 Final columns in shapefile ({len(valid_gdf.columns)} total):")
                for col in valid_gdf.columns:
                    print(f"      - {col} (length: {len(col)})")
                
                # Export shapefile
                shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
                os.makedirs(shp_pred_dir, exist_ok=True)
                shp_pred_path = os.path.join(shp_pred_dir, "predicted_output.shp")
                
                # Remove helper column before saving
                valid_gdf = valid_gdf.drop(columns=['__original_index__'], errors='ignore')
                
                print(f"\n   💾 Saving shapefile with these key columns:")
                print(f"      - {safe_target_name}: {safe_target_name in valid_gdf.columns}")
                print(f"      - prediction: {'prediction' in valid_gdf.columns}")
                
                valid_gdf.to_file(shp_pred_path)
                print(f"   ✅ Shapefile created with {len(valid_gdf)} features")
                
                # Create ZIP
                zip_out = os.path.join(export_path, "predicted_output.zip")
                with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in os.listdir(shp_pred_dir):
                        z.write(os.path.join(shp_pred_dir, f), f)
                
                print(f"   ✅ Created ZIP: {zip_out}")
            
            else:
                print("ℹ️ No geometry data available (no shapefile output)")

        except Exception as e:
            print(f"⚠️ Error creating shapefile output: {e}")
            import traceback
            traceback.print_exc()
            zip_out = None

        # 🔟 Build response payload
        counts, bins = np.histogram(residuals, bins=20)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # 🆕 Compute variable distributions
        print("📊 Computing variable distributions...")
        variable_distributions = compute_variable_distributions(
            df_valid, 
            indep
        )
        print(f"✅ Computed distributions for {len(variable_distributions)} variables")

        base_url = "/api/ai-tools/download"

        # plots (PNG) -> download URLs
        plots = {key: f"{base_url}?file={path}" for key, path in png_paths.items()}

        # base downloads
        downloads = {
            "model": f"{base_url}?file={model_path}",
            "report": f"{base_url}?file={pdf_path}",
            "cama_csv": f"{base_url}?file={csv_path}",
        }

        # optional shapefile + geojson preview
        if zip_out:
            downloads["shapefile"] = f"{base_url}?file={zip_out}"
            downloads["geojson"] = f"/api/ai-tools/preview-geojson?file_path={zip_out}"
            
        print("📋 Creating training result preview...")
        preview_df = df_valid.copy()

        if pin_series is not None:
            preview_df = preview_df.copy()
            preview_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)

        preview_cols = []
        if pin_series is not None:
            preview_cols.append("PIN")

        preview_cols.extend(indep)
        preview_cols.append(target)
        preview_cols.append("prediction")

        cama_preview = preview_df[preview_cols].head(100).to_dict("records")
        print(f"   ✅ Created preview with {len(cama_preview)} rows")
        
        metrics = {k: float(v) for k, v in metrics.items()}


        return {
            "model_version": model_version,
            "model_id": export_id,
            "dependent_var": safe_target_name,
            "original_dependent_var": target,
            "metrics": metrics,
            "features": indep,
            "importance": [  # ✅ Array format (matches RF/XGB)
                {"feature": feat, "value": float(val)}
                for feat, val in zip(indep, model.coef_)
            ],
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
            "message": "Model trained successfully (excluded rows removed from exports).",
        }

    except Exception as e:
        import traceback
        print(f"❌ TRAIN ERROR: {e}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})