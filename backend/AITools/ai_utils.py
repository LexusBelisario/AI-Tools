from typing import List, Optional, Dict, Tuple
import os
import tempfile
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import text
from fastapi import UploadFile
from db import get_user_database_session
GEOM_NAMES = {"geom", "geometry", "wkb_geometry", "the_geom"}

def get_provincial_code_from_schema(schema: str) -> str:
    if not schema:
        return ""
    return schema[:7] if len(schema) >= 7 else schema


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


def extract_pin_column(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    for col in df.columns:
        if col.lower() == "pin":
            return df[col], col
    return None, None

def compute_variable_distributions(
    df: pd.DataFrame, 
    columns: List[str], 
    bins: int = 30
) -> Dict[str, Dict]:
    distributions = {}
    
    for col in columns:
        try:
            # Get column data and drop NaNs
            data = df[col].dropna()
            
            if len(data) == 0:
                print(f"   ⚠️ No valid data for column '{col}'")
                continue
            
            # Compute histogram
            counts, bin_edges = np.histogram(data, bins=bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            # Compute statistics
            distributions[col] = {
                "values": data.tolist()[:1000],  # Limit to 1000 points for performance
                "bins": bin_centers.tolist(),
                "counts": counts.tolist(),
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "count": int(len(data)),
            }
            
            print(f"   ✅ {col}: mean={data.mean():.2f}, median={data.median():.2f}, std={data.std():.2f}")
            
        except Exception as e:
            print(f"   ⚠️ Could not compute distribution for '{col}': {e}")
            continue
    
    return distributions

def df_from_db(schema: str, table: str) -> pd.DataFrame:
    provincial_code = get_provincial_code_from_schema(schema)
    db_session = get_user_database_session(provincial_code)

    try:
        # Get columns
        cols = db_session.execute(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :s AND table_name = :t
                ORDER BY ordinal_position
            """),
            {"s": schema, "t": table},
        ).fetchall()

        colnames = [r[0] for r in cols]
        keep = [c for c in colnames if c.lower() not in GEOM_NAMES]
        keep_sql = "*" if not keep else ", ".join(f'"{c}"' for c in keep)

        rows = db_session.execute(
            text(f'SELECT {keep_sql} FROM "{schema}"."{table}"')
        ).fetchall()

        df = pd.DataFrame(rows, columns=colnames if keep_sql == "*" else keep)
        df = df[[c for c in df.columns if c.lower() not in GEOM_NAMES]]
        return df

    finally:
        db_session.close()


def gdf_from_db_with_geometry(schema: str, table: str) -> gpd.GeoDataFrame:
    provincial_code = get_provincial_code_from_schema(schema)
    db_session = get_user_database_session(provincial_code)
    engine = db_session.get_bind()

    try:
        geom_check = db_session.execute(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema 
                  AND table_name = :table
                  AND udt_name = 'geometry'
            """),
            {"schema": schema, "table": table},
        ).fetchone()

        # Direct load if geometry exists
        if geom_check:
            geom_col = geom_check[0]
            return gpd.read_postgis(
                f'SELECT * FROM "{schema}"."{table}"',
                engine,
                geom_col=geom_col
            )

        # Else find spatial table in schema
        spatial_tables = db_session.execute(
            text("""
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND udt_name = 'geometry'
            """),
            {"schema": schema},
        ).fetchall()

        if not spatial_tables:
            raise ValueError(f"No spatial tables in schema '{schema}'.")

        # Candidate join keys
        join_keys = [
            "PIN","PIN_NEW","ARP","ARP_NO","PARCEL_ID","LOT_ID",
            "TAXDECL_NO","TD_NO","MAP_NO","MAP_ID","ID"
        ]

        # Try find a join key match
        target_cols = {
            r[0].upper()
            for r in db_session.execute(
                text("""SELECT column_name FROM information_schema.columns
                        WHERE table_schema=:s AND table_name=:t"""),
                {"s": schema, "t": table},
            )
        }

        for sp_table, geom_col in spatial_tables:
            sp_cols = {
                r[0].upper()
                for r in db_session.execute(
                    text("""SELECT column_name FROM information_schema.columns
                            WHERE table_schema=:s AND table_name=:t"""),
                    {"s": schema, "t": sp_table},
                )
            }

            for key in join_keys:
                if key in target_cols and key in sp_cols:
                    join_sql = f"""
                        SELECT t.*, s."{geom_col}" AS geometry
                        FROM "{schema}"."{table}" t
                        JOIN "{schema}"."{sp_table}" s
                        ON t."{key}" = s."{key}"
                    """
                    return gpd.read_postgis(join_sql, engine, geom_col="geometry")

        raise ValueError(f"Could not auto-join {table} with a spatial table.")

    finally:
        engine.dispose()
        db_session.close()

def gdf_from_zip_or_parts(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame from either ZIP file or individual shapefile components.
    """
    print("=" * 60)
    print("🔍 gdf_from_zip_or_parts called")
    print(f"   zip_file: {zip_file.filename if zip_file else None}")
    print(f"   shapefiles: {len(shapefiles) if shapefiles else 0} files")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📁 Temp directory: {tmpdir}")
        shp_path = None

        # ========================================
        # HANDLE ZIP FILE
        # ========================================
        if zip_file:
            print(f"\n📦 Processing ZIP file: {zip_file.filename}")
            zip_path = os.path.join(tmpdir, zip_file.filename)
            
            try:
                # Read and save ZIP
                content = zip_file.file.read()
                print(f"   ✅ Read {len(content):,} bytes from upload")
                
                with open(zip_path, "wb") as f:
                    f.write(content)
                print(f"   ✅ Saved ZIP to: {zip_path}")
                
                # Extract ZIP
                with zipfile.ZipFile(zip_path, "r") as z:
                    file_list = z.namelist()
                    print(f"   📋 ZIP contains {len(file_list)} files:")
                    for fname in file_list[:10]:  # Show first 10
                        print(f"      - {fname}")
                    if len(file_list) > 10:
                        print(f"      ... and {len(file_list) - 10} more")
                    
                    z.extractall(tmpdir)
                    print(f"   ✅ Extracted to: {tmpdir}")
                
            except zipfile.BadZipFile as e:
                print(f"   ❌ BAD ZIP FILE: {e}")
                raise ValueError(f"Invalid ZIP file: {e}")
            except Exception as e:
                print(f"   ❌ ERROR processing ZIP: {e}")
                import traceback
                traceback.print_exc()
                raise

        # ========================================
        # HANDLE INDIVIDUAL SHAPEFILES
        # ========================================
        elif shapefiles:
            print(f"\n📁 Processing {len(shapefiles)} individual files")
            
            for idx, uf in enumerate(shapefiles):
                try:
                    file_path = os.path.join(tmpdir, uf.filename)
                    content = uf.file.read()
                    
                    with open(file_path, "wb") as out:
                        out.write(content)
                    
                    print(f"   [{idx+1}/{len(shapefiles)}] ✅ {uf.filename} ({len(content):,} bytes)")
                
                except Exception as e:
                    print(f"   [{idx+1}/{len(shapefiles)}] ❌ Error saving {uf.filename}: {e}")
                    raise

        else:
            print("❌ ERROR: No files provided!")
            raise ValueError("Either zip_file or shapefiles must be provided")

        # ========================================
        # FIND .SHP FILE
        # ========================================
        print(f"\n🔍 Searching for .shp file in {tmpdir}")
        
        all_files = []
        for root, dirs, files in os.walk(tmpdir):
            for fn in files:
                full_path = os.path.join(root, fn)
                all_files.append(full_path)
                print(f"   📄 Found: {fn}")
                
                if fn.lower().endswith(".shp"):
                    shp_path = full_path
                    print(f"   ✅ SHAPEFILE FOUND: {shp_path}")
                    break
            
            if shp_path:
                break

        if not shp_path:
            print(f"\n❌ NO .SHP FILE FOUND!")
            print(f"   Total files in temp dir: {len(all_files)}")
            print(f"   Files list:")
            for f in all_files:
                print(f"      - {f}")
            raise ValueError("No .shp file found in uploaded files")

        # ========================================
        # READ SHAPEFILE
        # ========================================
        print(f"\n📖 Reading shapefile: {shp_path}")
        
        try:
            gdf = gpd.read_file(shp_path)
            print(f"   ✅ SUCCESS!")
            print(f"   📊 Rows: {len(gdf)}")
            print(f"   📊 Columns: {len(gdf.columns)}")
            print(f"   📋 Column names: {gdf.columns.tolist()}")
            print(f"   🗺️  CRS: {gdf.crs}")
            print("=" * 60)
            return gdf
            
        except Exception as e:
            print(f"   ❌ ERROR reading shapefile: {e}")
            import traceback
            traceback.print_exc()
            
            # Check if required files exist
            base_name = shp_path[:-4]  # Remove .shp
            required_files = [".shp", ".shx", ".dbf"]
            print(f"\n   🔍 Checking for required files:")
            for ext in required_files:
                req_file = base_name + ext
                exists = os.path.exists(req_file)
                print(f"      {ext}: {'✅' if exists else '❌'} {req_file}")
            
            raise ValueError(f"Failed to read shapefile: {e}")


# -------------------------------------------------
# 🔧 PLOT HELPERS FOR RANDOM FOREST
# -------------------------------------------------
def plot_rf_feature_importance(importance, feature_names, ax=None):
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4))) if ax is None else (ax.figure, ax)
    idx = np.argsort(importance)
    ax.barh(range(len(idx)), importance[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    return fig


def plot_rf_residual_distribution(residuals, ax=None):
    fig, ax = plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax)
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Residual Distribution")
    return fig


def plot_rf_actual_vs_predicted(y_test, y_pred, ax=None):
    fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (ax.figure, ax)
    ax.scatter(y_test, y_pred, alpha=0.6)
    minv, maxv = min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))
    ax.plot([minv, maxv], [minv, maxv], "r--")
    ax.set_title("Actual vs Predicted")
    return fig


def plot_rf_residuals_vs_predicted(y_pred, residuals, ax=None):
    fig, ax = plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax)
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Residuals vs Predicted")
    return fig

def export_rf_artifacts(export_path, model, X_train, X_test, y_train, y_test,
                        y_pred, feature_names, df_full, df_valid, indep, target,
                        excluded_indices, is_db_mode, schema, table_name, file_gdf):

    os.makedirs(export_path, exist_ok=True)
    plots_dir = os.path.join(export_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # RF metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

    # Feature importance
    importance = model.feature_importances_
    fi_fig = plot_rf_feature_importance(importance, feature_names)
    fi_path = os.path.join(plots_dir, "fi.png")
    fi_fig.savefig(fi_path, dpi=200)
    plt.close(fi_fig)

    # Residual distribution
    residuals = y_test - y_pred
    rd_fig = plot_rf_residual_distribution(residuals)
    rd_path = os.path.join(plots_dir, "residual_dist.png")
    rd_fig.savefig(rd_path, dpi=200)
    plt.close(rd_fig)

    # A vs P
    avp_fig = plot_rf_actual_vs_predicted(y_test, y_pred)
    avp_path = os.path.join(plots_dir, "avp.png")
    avp_fig.savefig(
        avp_path, dpi=200
    )
    plt.close(avp_fig)

    # Residuals vs Predicted
    rvp_fig = plot_rf_residuals_vs_predicted(y_pred, residuals)
    rvp_path = os.path.join(plots_dir, "rvp.png")
    rvp_fig.savefig(
        rvp_path, dpi=200
    )
    plt.close(rvp_fig)

    # Save model
    import pickle
    model_path = os.path.join(export_path, "RandomForest_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": indep, "target": target}, f)

    # Export CSV
    preds_valid = model.predict(df_valid[indep].values)
    df_valid = df_valid.copy()
    df_valid["prediction"] = preds_valid

    csv_path = os.path.join(export_path, "RF_cleaned.csv")
    df_valid.to_csv(csv_path, index=False)

    # Export shapefile
    zip_out = None
    try:
        if is_db_mode:
            gdf_db = gdf_from_db_with_geometry(schema, table_name)
            gdf_db = gdf_db.loc[df_valid.index].copy()
            gdf_db["prediction"] = df_valid["prediction"]
            shp_dir = os.path.join(export_path, "shp")
            os.makedirs(shp_dir, exist_ok=True)
            shp_path = os.path.join(shp_dir, "RF_pred.shp")
            gdf_db.to_file(shp_path)
            zip_out = os.path.join(export_path, "RF_pred.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(shp_dir):
                    z.write(os.path.join(shp_dir, f), f)
        else:
            file_gdf = file_gdf.loc[df_valid.index].copy()
            file_gdf["prediction"] = df_valid["prediction"]
            shp_dir = os.path.join(export_path, "shp")
            os.makedirs(shp_dir, exist_ok=True)
            shp_path = os.path.join(shp_dir, "RF_pred.shp")
            file_gdf.to_file(shp_path)
            zip_out = os.path.join(export_path, "RF_pred.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(shp_dir):
                    z.write(os.path.join(shp_dir, f), f)
    except:
        pass

    downloads = {
        "model": model_path,
        "fi": fi_path,
        "residual_dist": rd_path,
        "avp": avp_path,
        "rvp": rvp_path,
        "csv": csv_path,
        "shapefile": zip_out,
    }

    return {
        "metrics": metrics,
        "plots": {
            "fi": fi_path,
            "residual_dist": rd_path,
            "avp": avp_path,
            "rvp": rvp_path,
        },
        "downloads": downloads,
    }

def export_lr_artifacts(export_path, model, scaler, indep, target,
                        X_train_scaled, y_train, X_test_scaled, y_test,
                        preds, residuals, df_valid):

    os.makedirs(export_path, exist_ok=True)
    png_paths = {}
    accent = "#1e88e5"

    from sklearn.metrics import r2_score
    mse = float(np.mean((y_test - preds) ** 2))
    mae = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, preds))

    metrics = {"R²": r2, "MSE": mse, "MAE": mae, "RMSE": rmse}

    pdf_path = os.path.join(export_path, "lr_report.pdf")
    with PdfPages(pdf_path) as pp:

        # Metrics table
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis("off")
        table = ax.table(
            cellText=[["Model", "MSE", "MAE", "RMSE", "R²"],
                      ["Linear Regression",
                       f"{mse:.2f}", f"{mae:.2f}",
                       f"{rmse:.2f}", f"{r2:.2f}"]],
            loc="center", cellLoc="center",
        )
        table.scale(1, 2)
        pp.savefig(fig)
        plt.close(fig)

        # Feature importance (coefficients)
        importance = model.coef_
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(indep, importance, color=accent)
        ax.set_title("Feature Importance (LR)")
        plt.xticks(rotation=45)
        fi_png = os.path.join(export_path, "lr_fi.png")
        fig.savefig(fi_png, bbox_inches="tight")
        plt.close(fig)
        png_paths["fi"] = fi_png

        # Residual distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, ax=ax, color=accent)
        ax.set_title("Residual Distribution (LR)")
        rd_png = os.path.join(export_path, "lr_residual_dist.png")
        fig.savefig(rd_png, bbox_inches="tight")
        plt.close(fig)
        png_paths["residual_dist"] = rd_png

        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, preds, alpha=0.6, edgecolor="black")
        minv, maxv = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
        ax.plot([minv, maxv], [minv, maxv], "k--")
        ax.set_title("Actual vs Predicted (LR)")
        avp_png = os.path.join(export_path, "lr_avp.png")
        fig.savefig(avp_png, bbox_inches="tight")
        plt.close(fig)
        png_paths["avp"] = avp_png

        # Residuals vs Predicted
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(preds, residuals, alpha=0.6)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title("Residuals vs Predicted (LR)")
        rvp_png = os.path.join(export_path, "lr_rvp.png")
        fig.savefig(rvp_png, bbox_inches="tight")
        plt.close(fig)
        png_paths["rvp"] = rvp_png

    # CSV export
    preds_valid = model.predict(scaler.transform(df_valid[indep]))
    df_valid = df_valid.copy()
    df_valid["prediction"] = preds_valid
    csv_path = os.path.join(export_path, "lr_cleaned.csv")
    df_valid.to_csv(csv_path, index=False)

    downloads = {
        "report": pdf_path,
        "csv": csv_path,
        "fi": png_paths["fi"],
        "residual_dist": png_paths["residual_dist"],
        "avp": png_paths["avp"],
        "rvp": png_paths["rvp"],
    }

    return {"metrics": metrics, "plots": png_paths, "downloads": downloads}

def export_xgb_artifacts(export_path, model, scaler, X_train, X_test, y_train, y_test,
                         y_pred, feature_names, df_valid, indep, target):

    os.makedirs(export_path, exist_ok=True)
    png_paths = {}

    from sklearn.metrics import r2_score
    mse = float(np.mean((y_test - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    metrics = {"MSE": mse, "MAE": mae, "RMSE": rmse, "R²": r2}

    # Feature importance
    importance = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(feature_names, importance, color="#1e88e5")
    ax.set_title("Feature Importance (XGB)")
    plt.xticks(rotation=45)
    fi_png = os.path.join(export_path, "xgb_fi.png")
    fig.savefig(fi_png, bbox_inches="tight")
    plt.close(fig)
    png_paths["fi"] = fi_png

    # Residual dist
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals, kde=True, ax=ax, color="#1e88e5")
    rd_png = os.path.join(export_path, "xgb_residual_dist.png")
    fig.savefig(rd_png, bbox_inches="tight")
    plt.close(fig)
    png_paths["residual_dist"] = rd_png

    # Actual vs Pred
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolor="black")
    minv, maxv = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([minv, maxv], [minv, maxv], "k--")
    avp_png = os.path.join(export_path, "xgb_avp.png")
    fig.savefig(avp_png, bbox_inches="tight")
    plt.close(fig)
    png_paths["avp"] = avp_png

    # Residuals vs Pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="black", linestyle="--")
    rvp_png = os.path.join(export_path, "xgb_rvp.png")
    fig.savefig(rvp_png, bbox_inches="tight")
    plt.close(fig)
    png_paths["rvp"] = rvp_png

    # CSV
    if scaler:
        preds_valid = model.predict(scaler.transform(df_valid[indep]))
    else:
        preds_valid = model.predict(df_valid[indep])

    df_valid = df_valid.copy()
    df_valid["prediction"] = preds_valid
    csv_path = os.path.join(export_path, "xgb_cleaned.csv")
    df_valid.to_csv(csv_path, index=False)

    downloads = {
        "fi": fi_png,
        "residual_dist": rd_png,
        "avp": avp_png,
        "rvp": rvp_png,
        "csv": csv_path,
    }

    return {"metrics": metrics, "plots": png_paths, "downloads": downloads}


def get_next_model_version(model_type: str) -> int:
    import os
    import re
    
    EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
    
    # Pattern to match: linear_1, rf_2, xgb_3, etc.
    pattern = re.compile(rf"^{model_type}_(\d+)$")
    
    versions = []
    if os.path.exists(EXPORT_DIR):
        for folder in os.listdir(EXPORT_DIR):
            match = pattern.match(folder)
            if match:
                versions.append(int(match.group(1)))
    
    return max(versions) + 1 if versions else 1


def find_pin_field(columns) -> Optional[str]:
    for c in columns:
        if str(c).lower() == "pin":
            return c
    return None


def upsert_pin_field(gdf: "gpd.GeoDataFrame", pin_values, preferred_name: str = "PIN") -> str:
    existing = find_pin_field(gdf.columns)

    if existing:
        gdf[existing] = pin_values
        return str(existing)

    gdf[preferred_name] = pin_values
    return preferred_name


def drop_duplicate_pin_fields(gdf: "gpd.GeoDataFrame", keep_name: str = "PIN"):
    pin_cols = [c for c in gdf.columns if str(c).lower() == "pin"]
    if len(pin_cols) <= 1:
        return

    keep = None
    for c in pin_cols:
        if str(c) == keep_name:
            keep = c
            break
    if keep is None:
        keep = pin_cols[0]

    for c in pin_cols:
        if c != keep:
            gdf.drop(columns=[c], inplace=True, errors="ignore")


def find_pin_field(columns):
    for c in columns:
        if str(c).lower() == "pin":
            return c
    return None


def upsert_pin_field(gdf, pin_values, preferred_name: str = "PIN") -> str:
    existing = find_pin_field(gdf.columns)
    if existing:
        gdf[existing] = pin_values
        return str(existing)

    gdf[preferred_name] = pin_values
    return preferred_name


def drop_duplicate_pin_fields(gdf, keep_name: str = "PIN"):
    pin_cols = [c for c in gdf.columns if str(c).lower() == "pin"]
    if len(pin_cols) <= 1:
        return

    keep = None
    for c in pin_cols:
        if str(c) == keep_name:
            keep = c
            break
    if keep is None:
        keep = pin_cols[0]

    for c in pin_cols:
        if c != keep:
            gdf.drop(columns=[c], inplace=True, errors="ignore")
