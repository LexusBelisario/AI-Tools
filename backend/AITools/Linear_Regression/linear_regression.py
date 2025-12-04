# backend/Predictive_Model_Tools/linear_regression.py

from fastapi import APIRouter, UploadFile, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Tuple, Dict, Any
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile, os, joblib, json, zipfile, shutil
import matplotlib
matplotlib.use("Agg")  # for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sqlalchemy import text

from db import get_user_database_session

router = APIRouter(prefix="/linear-regressions", tags=["AI Model Tools"])

# persistent export dir (never auto-deleted)
EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)


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
    """
    Load table from PostGIS WITH geometry.
    If the specified table has no geometry, attempts to find a related spatial table
    and join them on common key fields (PIN, ARPN, ARP_PIN, etc.)
    """
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

    metrics = {"R²": r2, "MSE": mse, "MAE": mae, "RMSE": rmse}
    
    accent = "#1e88e5"
    png_paths: Dict[str, str] = {}

    pdf_path = os.path.join(export_path, "regression_report.pdf")
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

    t_tests = {"residuals": residual_ttest, "coefficients": coef_ttests}
    return metrics, png_paths, t_tests, pdf_path


@router.post("/fields")
async def extract_fields(shapefiles: List[UploadFile]):
    try:
        gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=None)
        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        fields = df.columns.tolist()
        return {"fields": fields}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.post("/fields-zip")
async def extract_fields_zip(zip_file: UploadFile):
    try:
        gdf = gdf_from_zip_or_parts(shapefiles=None, zip_file=zip_file)
        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        fields = df.columns.tolist()
        return {"fields": fields}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.post("/preview")
async def file_preview(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    limit: int = Form(100),
    offset: int = Form(0),
):
    try:
        gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
        if gdf.geometry is not None:
            gdf = gdf.drop(columns=[gdf.geometry.name], errors="ignore")
        total = len(gdf)
        page_df = gdf.iloc[int(offset): int(offset) + int(limit)].copy()
        fields = list(page_df.columns)
        def _py(v):
            if pd.isna(v):
                return None
            try:
                return v.item()
            except Exception:
                return v
        rows = [{k: _py(v) for k, v in rec.items()} for rec in page_df.to_dict(orient="records")]
        return {"rows": rows, "total": int(total), "fields": fields}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/train")
async def train_linear_regression(
    # any of the three input modes can be used
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),

    # required modeling params
    independent_vars: str = Form(...),
    dependent_var: str = Form(...),
):
    """
    Unified trainer:
      - shapefiles[] OR
      - zip_file OR
      - schema + table_name (DB)
    """
    try:
        # 1) Determine input mode and build df_full
        file_gdf = None
        is_db_mode = False
        
        # Check if database mode (both schema and table_name must be non-empty)
        if schema and schema.strip() and table_name and table_name.strip():
            is_db_mode = True
            print(f"✅ Database mode detected: schema={schema}, table={table_name}")
            df_full = df_from_db(schema.strip(), table_name.strip())
        else:
            # file-based mode
            print(f"✅ File mode detected")
            gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
            file_gdf = gdf.copy()
            df_full = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

        # 2) Parse variables
        if independent_vars.startswith("["):
            indep = json.loads(independent_vars)
        else:
            indep = [v.strip() for v in independent_vars.split(",")]
        indep = [v for v in indep if v]
        target = dependent_var.strip()

        # 3) Validate fields (case-insensitive)
        lower_map = {c.lower(): c for c in df_full.columns}
        missing = [v for v in indep + [target] if v.lower() not in lower_map]
        if missing:
            return JSONResponse(status_code=400, content={"error": f"Missing variables: {missing}"})

        # normalize dataframe columns to lower for modeling
        df_full.columns = [c.lower() for c in df_full.columns]
        indep = [v.lower() for v in indep]
        target = target.lower()

        # 4) Convert to numeric
        for col in indep + [target]:
            df_full[col] = df_full[col].map(safe_to_float)

        df_valid = df_full.dropna(subset=indep + [target])
        if df_valid.empty:
            return JSONResponse(status_code=400, content={"error": "No valid numeric data found."})

        # 5) Train/test split & scale
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

        # 6) Export bundle dir
        export_id = f"linear_{np.random.randint(100000, 999999)}"
        export_path = os.path.join(EXPORT_DIR, export_id)
        os.makedirs(export_path, exist_ok=True)

        # 7) Save model bundle (lowercase feature names for robust matching)
        model_path = os.path.join(export_path, "trained_model.pkl")
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "features": [v.lower() for v in indep],
            "dependent_var": target.lower(),
        }, model_path)

        # 8) Build report & PNGs (blue accent), residual t-test
        metrics, png_paths, t_tests, pdf_path = export_full_report_and_artifacts(
            export_path, model, scaler, indep, target,
            X_train_scaled, y_train, X_test_scaled, y_test, preds, residuals,
            X_train_unscaled=X_train
        )

        # 9) Predict on full dataset and export CSV
        df_full["prediction"] = np.nan
        preds_valid = model.predict(scaler.transform(df_valid[indep]))
        df_full.loc[df_valid.index, "prediction"] = preds_valid
        csv_path = os.path.join(export_path, "LinearRegression_CAMA.csv")
        df_full[indep + [target, "prediction"]].to_csv(csv_path, index=False)

        # 10) If file-based, export shapefile + zip (skip for database mode)
        zip_out = None
        try:
            if is_db_mode:
    # Database mode: fetch geometry from DB and create shapefile
                print("✅ Database mode: fetching geometry and creating shapefile")
                
                provincial_code = get_provincial_code_from_schema(schema)
                db_session = get_user_database_session(provincial_code)
                engine = db_session.get_bind()
                
                try:
                    # Read original table WITH geometry
                    sql = f'SELECT * FROM "{schema}"."{table_name}"'
                    try:
    # Read original table WITH geometry (with auto-join if needed)
                        print(f"📍 Loading spatial data for {schema}.{table_name}")
                        gdf_db = gdf_from_db_with_geometry(schema, table_name)
                        
                        print(f"📊 GeoDataFrame shape: {gdf_db.shape}")
                        print(f"📊 Predictions shape: {df_full['prediction'].shape}")
                        print(f"📊 GeoDataFrame columns: {list(gdf_db.columns)}")
                        
                        # Normalize column names to lowercase for alignment
                        gdf_db.columns = [c.lower() for c in gdf_db.columns]
                        
                        # Reset index to ensure proper alignment
                        gdf_db = gdf_db.reset_index(drop=True)
                        df_full_reset = df_full.reset_index(drop=True)
                        
                        # Add predictions - properly aligned by index
                        if len(gdf_db) == len(df_full_reset):
                            gdf_db["prediction"] = df_full_reset["prediction"]
                            print(f"✅ Added {gdf_db['prediction'].notna().sum()} predictions")
                        else:
                            print(f"⚠️ Length mismatch: gdf={len(gdf_db)}, df={len(df_full_reset)}")
                            # Fallback: fill with NaN
                            gdf_db["prediction"] = np.nan
                        
                        # Create shapefile output (same as file mode)
                        shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
                        os.makedirs(shp_pred_dir, exist_ok=True)
                        shp_pred_path = os.path.join(shp_pred_dir, "predicted_output.shp")
                        
                        print(f"📁 Writing shapefile to: {shp_pred_path}")
                        gdf_db.to_file(shp_pred_path)
                        print(f"✅ Shapefile created successfully")
                        
                        # Create ZIP
                        zip_out = os.path.join(export_path, "predicted_output.zip")
                        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                            for f in os.listdir(shp_pred_dir):
                                full_path = os.path.join(shp_pred_dir, f)
                                z.write(full_path, f)
                        
                        print(f"✅ Created ZIP file: {zip_out}")
                        print(f"✅ Shapefile output complete for database mode")
                        
                        # Optionally: also save to a new table in DB
                        try:
                            predicted_table = f"{table_name}_Predicted"
                            
                            # Get a fresh engine for saving (avoid connection issues)
                            provincial_code = get_provincial_code_from_schema(schema)
                            db_session_save = get_user_database_session(provincial_code)
                            engine_save = db_session_save.get_bind()
                            
                            gdf_db.to_postgis(
                                name=predicted_table,
                                con=engine_save,
                                schema=schema,
                                if_exists="replace",
                                index=False
                            )
                            print(f"✅ Also saved predictions to {schema}.{predicted_table}")
                            
                            engine_save.dispose()
                            db_session_save.close()
                        except Exception as save_err:
                            print(f"⚠️ Could not save to DB table (non-critical): {save_err}")
                            import traceback
                            traceback.print_exc()
                        
                    except Exception as db_err:
                        print(f"❌ Database mode shapefile creation failed: {db_err}")
                        import traceback
                        traceback.print_exc()
                        raise  # Re-raise to see the full error
                    
                    print(f"📊 GeoDataFrame shape: {gdf_db.shape}")
                    print(f"📊 Predictions shape: {df_full['prediction'].shape}")
                    
                    # Normalize column names to lowercase for alignment
                    gdf_db.columns = [c.lower() for c in gdf_db.columns]
                    
                    # Reset index to ensure proper alignment
                    gdf_db = gdf_db.reset_index(drop=True)
                    df_full_reset = df_full.reset_index(drop=True)
                    
                    # Add predictions - properly aligned by index
                    if len(gdf_db) == len(df_full_reset):
                        gdf_db["prediction"] = df_full_reset["prediction"]
                        print(f"✅ Added {gdf_db['prediction'].notna().sum()} predictions")
                    else:
                        print(f"⚠️ Length mismatch: gdf={len(gdf_db)}, df={len(df_full_reset)}")
                        # Fallback: fill with NaN
                        gdf_db["prediction"] = np.nan
                    
                    # Create shapefile output (same as file mode)
                    shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
                    os.makedirs(shp_pred_dir, exist_ok=True)
                    shp_pred_path = os.path.join(shp_pred_dir, "predicted_output.shp")
                    
                    print(f"📁 Writing shapefile to: {shp_pred_path}")
                    gdf_db.to_file(shp_pred_path)
                    print(f"✅ Shapefile created successfully")
                    
                    # Create ZIP
                    zip_out = os.path.join(export_path, "predicted_output.zip")
                    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                        for f in os.listdir(shp_pred_dir):
                            full_path = os.path.join(shp_pred_dir, f)
                            z.write(full_path, f)
                    
                    print(f"✅ Created ZIP file: {zip_out}")
                    print(f"✅ Shapefile output complete for database mode")
                    
                    # Optionally: also save to a new table in DB
                    try:
                        predicted_table = f"{table_name}_Predicted"
                        gdf_db.to_postgis(
                            name=predicted_table,
                            con=engine,
                            schema=schema,
                            if_exists="replace",
                            index=False
                        )
                        print(f"✅ Also saved predictions to {schema}.{predicted_table}")
                    except Exception as save_err:
                        print(f"⚠️ Could not save to DB table (non-critical): {save_err}")
                        import traceback
                        traceback.print_exc()
                    
                except Exception as db_err:
                    print(f"❌ Database mode shapefile creation failed: {db_err}")
                    import traceback
                    traceback.print_exc()
                    raise  # Re-raise to see the full error
                finally:
                    engine.dispose()
                    db_session.close()
            
            elif file_gdf is not None:
                # File mode: create shapefile (existing logic)
                print("✅ File mode: creating shapefile output")
                file_gdf = file_gdf.copy()
                file_gdf["prediction"] = df_full["prediction"].values
                shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
                os.makedirs(shp_pred_dir, exist_ok=True)
                shp_pred_path = os.path.join(shp_pred_dir, "predicted_output.shp")
                file_gdf.to_file(shp_pred_path)

                zip_out = os.path.join(export_path, "predicted_output.zip")
                with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in os.listdir(shp_pred_dir):
                        z.write(os.path.join(shp_pred_dir, f), f)
                
                print(f"✅ Created shapefile output for file mode: {zip_out}")
            
            else:
                print("ℹ️ No spatial output (no geometry data)")

        except Exception as e:
            print(f"⚠️ Error creating shapefile output: {e}")
            import traceback
            traceback.print_exc()
            zip_out = None

        # 11) Basic interactive payload bits (histogram bins)
        counts, bins = np.histogram(residuals, bins=20)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        base_url = "/api/linear-regression/download"
        plots = {key: f"{base_url}?file={path}" for key, path in png_paths.items()}
        downloads = {
            "model": f"{base_url}?file={model_path}",
            "report": f"{base_url}?file={pdf_path}",
            "cama_csv": f"{base_url}?file={csv_path}",
        }

        # Add shapefile if created (works for BOTH file and database mode)
        if zip_out:
            downloads["shapefile"] = f"{base_url}?file={zip_out}"
            # Also add GeoJSON preview URL for map visualization
            downloads["geojson"] = f"{base_url.replace('/download', '/preview-geojson')}?file_path={zip_out}"

        return {
            "dependent_var": target,
            "metrics": metrics,
            "coefficients": {k: float(v) for k, v in zip(indep, model.coef_)},
            "intercept": float(model.intercept_),
            "t_test": t_tests,
            "interactive_data": {
                "residuals": residuals.tolist(),
                "residual_bins": bin_centers.tolist(),
                "residual_counts": counts.tolist(),
                "y_test": y_test.tolist(),
                "preds": preds.tolist(),
                "importance": {k: float(v) for k, v in zip(indep, getattr(model, "coef_", np.zeros(len(indep))))},
            },
            "plots": plots,
            "downloads": downloads,
            "is_db_mode": is_db_mode,  # ✅ Let frontend know the mode
            "message": "Model trained successfully and files ready for download.",
        }

    except Exception as e:
        import traceback
        print(f"❌ TRAIN ERROR: {e}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/download")
async def download_file(file: str = Query(...)):
    try:
        if not os.path.exists(file):
            return JSONResponse(status_code=404, content={"error": "File not found."})
        filename = os.path.basename(file)
        return FileResponse(path=file, filename=filename, media_type="application/octet-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/db-tables")
async def get_db_tables_for_schema(schema: str):
    """
    Returns ONLY CAMA_Table if it exists in the given schema.
    """
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)
        try:
            q = text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema
                  AND table_type = 'BASE TABLE'
                  AND LOWER(table_name) = 'cama_table'
                ORDER BY table_name
            """)
            rows = db_session.execute(q, {"schema": schema}).fetchall()
            tables = [r[0] for r in rows]
            return {"tables": tables, "schema": schema}
        finally:
            db_session.close()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/db-fields")
async def get_db_fields_for_table(table: str, schema: str):
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)
        try:
            fields_query = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ordinal_position
            """)
            rows = db_session.execute(fields_query, {"schema": schema, "table": table}).fetchall()
            fields = [r[0] for r in rows]
            if not fields:
                return JSONResponse(status_code=404, content={"error": f"Table '{table}' not found in schema '{schema}'"})
            return {"fields": fields, "table": table, "schema": schema}
        finally:
            db_session.close()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/db-preview")
async def preview_db_table(
    table: str,
    schema: str,
    limit: int = 100,
    offset: int = 0
):
    """Single, non-duplicated /db-preview."""
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)
        try:
            # fields
            fields_rows = db_session.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :table
                    ORDER BY ordinal_position
                """),
                {"schema": schema, "table": table},
            ).fetchall()
            fields_all = [r[0] for r in fields_rows]
            fields = [c for c in fields_all if c.lower() not in GEOM_NAMES]
            if not fields:
                return JSONResponse(status_code=404, content={"error": "No non-geometry columns to preview."})

            # total
            total = db_session.execute(
                text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
            ).scalar()

            # rows
            col_sql = ", ".join([f'"{c}"' for c in fields])  # ✅ build safely outside the f-string
            data_query = text(
                f'SELECT {col_sql} FROM "{schema}"."{table}" LIMIT :limit OFFSET :offset'
            )
            res = db_session.execute(data_query, {"limit": limit, "offset": offset})
            rows = [dict(zip(fields, r)) for r in res]
            return {"fields": fields, "rows": rows, "total": total, "schema": schema, "table": table}
        finally:
            db_session.close()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/run-saved-model")
async def run_saved_model_unified(
    model_file: UploadFile,
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    schema: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
):
    """
    Unified run-saved-model endpoint
    Supports:
      - Local shapefile or ZIP
      - Database table (schema + table_name)
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # --- Load model bundle ---
            model_path = os.path.join(tmpdir, model_file.filename)
            with open(model_path, "wb") as f:
                f.write(await model_file.read())

            bundle = joblib.load(model_path)
            model = bundle["model"]
            scaler = bundle.get("scaler", None)
            features = [f.lower() for f in bundle.get("features", [])]
            print(f"✅ Loaded model with features: {features}")

            # --- Detect input source ---
            if schema and table_name:
                # ========== DATABASE MODE ==========
                print(f"🗄️ Running model on database: {schema}.{table_name}")
                gdf = gdf_from_db_with_geometry(schema, table_name)
            else:
                # ========== FILE MODE ==========
                print(f"📂 Running model on shapefile/zip input")
                gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)

            # --- Prepare dataframe ---
            df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
            df.columns = [c.lower() for c in df.columns]

            missing = [f for f in features if f not in df.columns]
            if missing:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Missing features in input: {missing}"},
                )

            X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
            preds = (
                model.predict(scaler.transform(X))
                if scaler is not None
                else model.predict(X)
            )

            gdf["prediction"] = preds

            # --- Export results ---
            export_path = os.path.join(EXPORT_DIR, f"run_{np.random.randint(100000, 999999)}")
            os.makedirs(export_path, exist_ok=True)

            shp_dir = os.path.join(export_path, "predicted_shapefile")
            os.makedirs(shp_dir, exist_ok=True)
            shp_path = os.path.join(shp_dir, "predicted_output.shp")
            gdf.to_file(shp_path)

            zip_out = os.path.join(export_path, "predicted_output.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(shp_dir):
                    z.write(os.path.join(shp_dir, f), f)

            pdf_path = os.path.join(export_path, "run_report.pdf")
            with PdfPages(pdf_path) as pp:
                fig, ax = plt.subplots(figsize=(7, 2))
                ax.axis("off")
                ax.text(
                    0.5, 0.5,
                    f"Predictions completed\n{len(preds)} records processed.",
                    ha="center", va="center", fontsize=13, weight="bold",
                )
                pp.savefig(fig); plt.close(fig)

            # --- Optional: save to DB table ---
            if schema and table_name:
                try:
                    provincial_code = get_provincial_code_from_schema(schema)
                    db_session = get_user_database_session(provincial_code)
                    engine = db_session.get_bind()
                    out_table = f"{table_name}_PredictedRun"
                    gdf.to_postgis(
                        name=out_table,
                        con=engine,
                        schema=schema,
                        if_exists="replace",
                        index=False
                    )
                    print(f"✅ Saved to {schema}.{out_table}")
                    engine.dispose()
                    db_session.close()
                except Exception as e:
                    print(f"⚠️ Could not save to DB: {e}")

            base_url = "/api/linear-regression/download"
            return {
                "message": "✅ Model run completed successfully.",
                "downloads": {
                    "report": f"{base_url}?file={pdf_path}",
                    "shapefile": f"{base_url}?file={zip_out}",
                },
                "record_count": len(gdf),
            }

    except Exception as e:
        import traceback
        print("❌ RUN ERROR:", e)
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})



@router.post("/run-saved-model-db")
async def run_saved_model_db(
    model_file: UploadFile,
    table_name: str = Form(...),
):
    try:
        schema_part, table_part = ("public", table_name)
        if "." in table_name:
            schema_part, table_part = table_name.split(".", 1)

        provincial_code = get_provincial_code_from_schema(schema_part)
        db_session = get_user_database_session(provincial_code)
        engine = db_session.get_bind()
        try:
            sql = f'SELECT * FROM "{schema_part}"."{table_part}"'
            gdf = gpd.read_postgis(sql, engine, geom_col="geometry")

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, model_file.filename)
                with open(model_path, "wb") as f:
                    f.write(await model_file.read())
                bundle = joblib.load(model_path)
                model, scaler = bundle["model"], bundle.get("scaler")
                features = [f.lower() for f in bundle.get("features", [])]

            gdf.columns = [c.lower() for c in gdf.columns]
            X = gdf[features].apply(pd.to_numeric, errors="coerce").fillna(0)
            preds = model.predict(scaler.transform(X)) if scaler else model.predict(X)
            gdf["prediction"] = preds

            export_path = os.path.join(EXPORT_DIR, f"run_{np.random.randint(100000,999999)}")
            os.makedirs(export_path, exist_ok=True)
            shp_dir = os.path.join(export_path, "predicted_shapefile")
            os.makedirs(shp_dir, exist_ok=True)
            shp_path = os.path.join(shp_dir, "predicted_output.shp")
            gdf.to_file(shp_path)

            zip_out = os.path.join(export_path, "predicted_output.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(shp_dir):
                    z.write(os.path.join(shp_dir, f), f)

            pdf_path = os.path.join(export_path, "predicted_report.pdf")
            with PdfPages(pdf_path) as pdf:
                plt.hist(gdf["prediction"], bins=20, color="skyblue", edgecolor="black")
                plt.title("Distribution of Predicted Values")
                pdf.savefig(); plt.close()

            base_url = "/api/linear-regression/download"
            return {
                "message": "Predictions completed successfully.",
                "downloads": {
                    "report": f"{base_url}?file={pdf_path}",
                    "shapefile": f"{base_url}?file={zip_out}"
                },
                "record_count": len(gdf)
            }
        finally:
            engine.dispose()
            db_session.close()

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/save-to-db")
async def save_to_db(payload: dict):
    shapefile_url = payload.get("shapefile_url")
    table_name = payload.get("table_name", "Predicted_Output")
    if not shapefile_url or not shapefile_url.endswith(".zip"):
        return JSONResponse(status_code=400, content={"error": "Invalid shapefile URL."})
    try:
        file_path = shapefile_url.split("?file=")[-1]
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "Shapefile not found."})

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(tmpdir)
            shp_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_files:
                return JSONResponse(status_code=400, content={"error": "No .shp file found in zip."})
            shp_path = shp_files[0]
            gdf = gpd.read_file(shp_path)

            schema = payload.get("schema", "public")
            provincial_code = get_provincial_code_from_schema(schema)
            db_session = get_user_database_session(provincial_code)
            engine = db_session.get_bind()

            gdf.to_postgis(
                name=table_name,
                con=engine,
                schema=schema,
                if_exists="replace",
                index=False
            )
            engine.dispose()
        return {"message": f"Saved successfully to {schema}.{table_name}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/predicted-geojson")
def get_predicted_geojson(table: str = "Predicted_Output", schema: str = "public"):
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)
        engine = db_session.get_bind()
        sql = f'SELECT * FROM "{schema}"."{table}"'
        gdf = gpd.read_postgis(sql, engine, geom_col="geometry")
        engine.dispose()
        return json.loads(gdf.to_json())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/preview-geojson")
def preview_geojson(file_path: str):
    """
    Convert predicted shapefile ZIP output to GeoJSON for preview map.
    Works even before saving to DB.
    """
    import urllib.parse
    try:
        if file_path.startswith("/api/linear-regression/download"):
            parsed = urllib.parse.urlparse(file_path)
            query_params = urllib.parse.parse_qs(parsed.query)
            file_path = query_params.get("file", [None])[0]
            if not file_path:
                return JSONResponse({"error": "Invalid file parameter in URL."}, status_code=400)
            file_path = urllib.parse.unquote(file_path)

        file_path = file_path.strip('"').strip("'")
        if not os.path.exists(file_path):
            return JSONResponse({"error": f"File not found: {file_path}"}, status_code=404)

        tmpdir = tempfile.mkdtemp()
        extract_dir = os.path.join(tmpdir, "unzipped")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        shp_file = next(
            (os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith(".shp")),
            None,
        )
        if not shp_file:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return JSONResponse({"error": "No .shp found inside ZIP."}, status_code=400)

        gdf = gpd.read_file(shp_file)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        geojson_data = json.loads(gdf.to_json())
        shutil.rmtree(tmpdir, ignore_errors=True)
        return geojson_data
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
