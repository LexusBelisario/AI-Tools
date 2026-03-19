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
from db import get_user_database_session
from AITools.ai_utils import (
    extract_pin_column,
    compute_variable_distributions,
    upsert_pin_field,
    drop_duplicate_pin_fields,
)

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")
os.makedirs(EXPORT_DIR, exist_ok=True)

REPORT_ACCENT = "#1e88e5"
REPORT_DARK = "#1f2937"
REPORT_LIGHT = "#f8fafc"
REPORT_BORDER = "#d0d7de"


def build_artifact_base_name(model_used: str) -> str:
    now = datetime.now()
    return f"{model_used}_{now.strftime('%Y-%b-%d_%I-%M-%S%p')}"


def wrap_plot_urls(plots: Dict[str, Optional[str]], prefix: str) -> Dict[str, Optional[str]]:
    return {
        key: (f"{prefix}?file={path}" if path else None)
        for key, path in plots.items()
    }


def get_provincial_code_from_schema(schema: str) -> str:
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
    provincial_code = get_provincial_code_from_schema(schema)
    db_session = get_user_database_session(provincial_code)
    try:
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
            keep_sql = "*"
        else:
            keep_sql = ", ".join(f'"{c}"' for c in keep)

        rows = db_session.execute(
            text(f'SELECT {keep_sql} FROM "{schema}"."{table}"')
        ).fetchall()

        if keep_sql == "*":
            df = pd.DataFrame(rows, columns=colnames)
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
            print(f"{table} has geometry column: {geom_check[0]}")
            sql = f'SELECT * FROM "{schema}"."{table}"'
            gdf = gpd.read_postgis(sql, engine, geom_col=geom_check[0])
            return gdf

        print(f"{table} has no geometry. Searching for spatial tables in {schema}...")

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
        print(f"Found spatial tables: {list(spatial_tables.keys())}")

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

        join_key_candidates = [
            "PIN", "ARPN", "ARP_PIN", "TD_NO", "PROPERTY_ID",
            "PARCEL_ID", "TAX_DEC_NO", "OID", "OBJECTID", "FID"
        ]

        for spatial_table, geom_col in spatial_tables.items():
            spatial_cols_rows = db_session.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :table
                """),
                {"schema": schema, "table": spatial_table}
            ).fetchall()
            spatial_cols = {row[0].upper() for row in spatial_cols_rows}

            common_keys = []
            for key in join_key_candidates:
                if key in target_cols and key in spatial_cols:
                    common_keys.append(key)

            if common_keys:
                join_key = common_keys[0]
                print(f"Found join key '{join_key}' between {table} and {spatial_table}")

                join_sql = f'''
                    SELECT
                        t.*,
                        s."{geom_col}" as geometry
                    FROM "{schema}"."{table}" t
                    INNER JOIN "{schema}"."{spatial_table}" s
                      ON t."{join_key}" = s."{join_key}"
                '''

                print(f"Joining {table} with {spatial_table} on {join_key}")
                gdf = gpd.read_postgis(join_sql, engine, geom_col="geometry")
                print(f"Successfully joined. Result has {len(gdf)} rows")
                return gdf

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


def _new_page(figsize=(8.27, 11.69)):
    fig = plt.figure(figsize=figsize, facecolor="white")
    return fig


def _add_page_header(fig, title: str, subtitle: Optional[str] = None):
    fig.text(0.07, 0.965, title, fontsize=20, fontweight="bold", color=REPORT_ACCENT, va="top")
    if subtitle:
        fig.text(0.07, 0.938, subtitle, fontsize=10.5, color="#5f6b7a", va="top")
    fig.lines.append(
        plt.Line2D(
            [0.07, 0.93],
            [0.922, 0.922],
            transform=fig.transFigure,
            color=REPORT_BORDER,
            linewidth=1.2
        )
    )


def _add_footer(fig, artifact_base: str, page_label: str):
    fig.lines.append(
        plt.Line2D(
            [0.07, 0.93],
            [0.05, 0.05],
            transform=fig.transFigure,
            color=REPORT_BORDER,
            linewidth=0.8
        )
    )
    fig.text(0.07, 0.028, f"Model Report | {artifact_base}", fontsize=8.5, color="#6b7280")
    fig.text(0.93, 0.028, page_label, fontsize=8.5, color="#6b7280", ha="right")


def _save_png(fig, export_path: str, filename: str) -> str:
    out = os.path.join(export_path, filename)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=200)
    return out


def _wrap_text(text: str, chars_per_line: int) -> List[str]:
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > chars_per_line:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_wrapped_text(
    ax,
    x: float,
    y_start: float,
    text: str,
    chars_per_line: int,
    fontsize: float,
    color: str,
    line_gap: float = 0.12,
    fontweight: Optional[str] = None,
):
    lines = _wrap_text(text, chars_per_line)
    y = y_start
    for line in lines:
        ax.text(
            x,
            y,
            line,
            fontsize=fontsize,
            color=color,
            va="top",
            fontweight=fontweight,
            clip_on=True,
        )
        y -= line_gap
    return y


def _draw_paragraph_in_box(
    ax,
    text: str,
    x: float = 0.03,
    y_top: float = 0.78,
    width_chars: int = 82,
    fontsize: float = 10.5,
    color: str = REPORT_DARK,
    line_gap: float = 0.16,
):
    lines = _wrap_text(text, width_chars)
    y = y_top
    for line in lines:
        ax.text(x, y, line, fontsize=fontsize, color=color, va="top", clip_on=True)
        y -= line_gap
    return y


def _draw_feature_tags(ax, features: List[str], x_start=0.03, y_start=0.66):
    x = x_start
    y = y_start
    row_height = 0.16
    pad_x = 0.015
    max_x = 0.95

    for feat in features:
        feat = str(feat)
        est_w = min(0.012 * len(feat) + 0.06, 0.22)

        if x + est_w > max_x:
            x = x_start
            y -= row_height

        ax.text(
            x,
            y,
            feat,
            fontsize=9.2,
            color=REPORT_DARK,
            va="center",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.24,rounding_size=0.06",
                facecolor="#eef6ff",
                edgecolor="#b6d4fe",
                linewidth=0.9,
            ),
            clip_on=True,
        )
        x += est_w + pad_x

    return y

def _estimate_feature_box_height(features: List[str]) -> float:
    if not features:
        return 0.16

    rows = 1
    current_width = 0.03
    max_x = 0.95

    for feat in features:
        est_w = min(0.012 * len(str(feat)) + 0.06, 0.22)
        if current_width + est_w > max_x:
            rows += 1
            current_width = 0.03 + est_w + 0.015
        else:
            current_width += est_w + 0.015

    return max(0.22, min(0.42, 0.12 + rows * 0.075))

def _metrics_interpretation_text(
    r2: float,
    rmse: float,
    mae: float,
    top_feature: Optional[str],
    top_value: Optional[float],
) -> List[str]:
    if r2 >= 0.75:
        perf_text = f"R² = {r2:.3f} indicates strong explanatory power."
    elif r2 >= 0.50:
        perf_text = f"R² = {r2:.3f} indicates moderate explanatory power."
    elif r2 >= 0.25:
        perf_text = f"R² = {r2:.3f} indicates limited explanatory power."
    else:
        perf_text = f"R² = {r2:.3f} indicates weak explanatory power."

    error_text = (
        f"Average prediction error is around MAE = {mae:.2f}, "
        f"while RMSE = {rmse:.2f} suggests the effect of larger errors."
    )

    if top_feature is not None and top_value is not None:
        feature_text = (
            f"Most influential predictor: {top_feature} "
            f"(standardized effect = {top_value:.4f})."
        )
    else:
        feature_text = "Feature importance could not be ranked."

    return [perf_text, error_text, feature_text]


def _chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]


def _style_table(table, header_fontsize=9, body_fontsize=8.8, significant_col: Optional[int] = None):
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#222222")
        cell.set_linewidth(0.8)
        if i == 0:
            cell.set_facecolor(REPORT_ACCENT)
            cell.set_text_props(weight="bold", color="white", fontsize=header_fontsize)
        else:
            if significant_col is not None and j == significant_col:
                txt = cell.get_text().get_text()
                cell.set_facecolor("#d1fae5" if txt == "Yes" else "#fee2e2")
            else:
                cell.set_facecolor("#f5f7fa" if i % 2 == 1 else "white")
            cell.set_text_props(color=REPORT_DARK, fontsize=body_fontsize)


def _build_cover_page(pp: PdfPages, artifact_base: str, target: str, features: List[str], n_samples: int, page_num: int) -> int:
    fig = _new_page()
    fig.text(0.07, 0.88, "Linear Regression Model Report", fontsize=24, fontweight="bold", color=REPORT_ACCENT)
    fig.text(0.07, 0.84, "Structured training documentation", fontsize=13, color="#5f6b7a")

    meta_ax = fig.add_axes([0.07, 0.54, 0.86, 0.23])
    meta_ax.axis("off")
    meta_ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2)
    )
    meta_ax.add_patch(
        plt.Rectangle((0, 0.83), 1, 0.17, facecolor="#f4f9ff", edgecolor=REPORT_BORDER, linewidth=1.2)
    )
    meta_ax.text(0.03, 0.915, "Model Information", fontsize=12, fontweight="bold", color=REPORT_ACCENT, va="center")

    meta_lines = [
        ("Model Type", "Linear Regression"),
        ("Model Name", artifact_base),
        ("Target Variable", target),
        ("Feature Count", str(len(features))),
        ("Training Samples", f"{n_samples:,}"),
        ("Generated At", datetime.now().strftime("%Y-%b-%d %I:%M:%S %p")),
    ]

    y = 0.72
    for label, value in meta_lines:
        meta_ax.text(0.03, y, label, fontsize=11, fontweight="bold", color=REPORT_DARK, va="center")
        meta_ax.text(0.30, y, value, fontsize=11, color=REPORT_DARK, va="center")
        y -= 0.11

    def_ax = fig.add_axes([0.07, 0.30, 0.86, 0.22])
    def_ax.axis("off")
    def_ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, fill=True, facecolor="#f0f7ff", edgecolor=REPORT_BORDER, linewidth=1.2)
    )
    def_ax.text(0.03, 0.88, "What is Linear Regression?", fontsize=11, fontweight="bold", color=REPORT_ACCENT, va="top")

    definition = (
        "Linear Regression is a method that estimates how the target value changes based on the "
        "input variables. In simple terms, it finds the best-fitting straight-line relationship "
        "between the features and the value you want to predict. Each feature is given a weight "
        "called a coefficient, and those coefficients are used to compute predictions."
    )

    _draw_paragraph_in_box(
        def_ax,
        definition,
        x=0.03,
        y_top=0.68,
        width_chars=95,
        fontsize=9.5,
        color=REPORT_DARK,
        line_gap=0.14,
    )

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


def _build_executive_summary_page(
    pp: PdfPages,
    artifact_base: str,
    metrics: Dict[str, float],
    independent_vars: List[str],
    importance: np.ndarray,
    residual_ttest: Dict[str, float],
    page_num: int,
) -> int:
    fig = _new_page()
    _add_page_header(fig, "Executive Summary", "Key results and top-level interpretation")

    sorted_pairs = sorted(
        zip(independent_vars, importance),
        key=lambda x: abs(float(x[1])),
        reverse=True
    )
    top_feature = sorted_pairs[0][0] if sorted_pairs else None
    top_value = float(sorted_pairs[0][1]) if sorted_pairs else None

    summary_lines = _metrics_interpretation_text(
        r2=metrics["r2"],
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        top_feature=top_feature,
        top_value=top_value,
    )

    left_ax = fig.add_axes([0.07, 0.52, 0.40, 0.32])
    left_ax.axis("off")
    left_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    left_ax.text(0.04, 0.93, "Performance Summary", fontsize=12, fontweight="bold", color=REPORT_ACCENT)

    y = 0.80
    for line in summary_lines:
        wrapped = _wrap_text(line, chars_per_line=42)
        left_ax.text(0.05, y, f"• {wrapped[0]}", fontsize=9.5, color=REPORT_DARK, va="top")
        sub_y = y - 0.10
        for extra in wrapped[1:]:
            left_ax.text(0.08, sub_y, extra, fontsize=9.5, color=REPORT_DARK, va="top")
            sub_y -= 0.10
        y = sub_y - 0.04

    right_ax = fig.add_axes([0.53, 0.52, 0.40, 0.32])
    right_ax.axis("off")
    right_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    right_ax.text(0.04, 0.93, "Residual Check", fontsize=12, fontweight="bold", color=REPORT_ACCENT)

    p_val = residual_ttest["p_value"]
    if p_val < 0.05:
        residual_msg = "Residual mean is statistically different from zero at alpha = 0.05."
    else:
        residual_msg = "Residual mean is not statistically different from zero at alpha = 0.05."

    residual_lines = [
        f"T-statistic: {residual_ttest['t_stat']:.4f}",
        f"P-value: {residual_ttest['p_value']:.4f}",
        residual_msg,
    ]

    y = 0.80
    for line in residual_lines:
        wrapped = _wrap_text(line, chars_per_line=42)
        right_ax.text(0.05, y, f"• {wrapped[0]}", fontsize=9.5, color=REPORT_DARK, va="top")
        sub_y = y - 0.10
        for extra in wrapped[1:]:
            right_ax.text(0.08, sub_y, extra, fontsize=9.5, color=REPORT_DARK, va="top")
            sub_y -= 0.10
        y = sub_y - 0.04

    rec_ax = fig.add_axes([0.07, 0.19, 0.86, 0.23])
    rec_ax.axis("off")
    rec_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    rec_ax.text(0.02, 0.90, "Recommended Reading of this Report", fontsize=12, fontweight="bold", color=REPORT_ACCENT)

    rec_text = (
        "Use the metrics page to evaluate overall fit, the feature pages to review standardized "
        "effects and coefficient significance, and the diagnostics page to inspect bias and "
        "error behavior. Variable distribution pages provide context for predictor spread."
    )
    _draw_wrapped_text(rec_ax, 0.02, 0.68, rec_text, chars_per_line=100, fontsize=10.5, color=REPORT_DARK, line_gap=0.14)

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


def _build_metrics_table_page(
    pp: PdfPages,
    artifact_base: str,
    metrics: Dict[str, float],
    export_path: str,
    page_num: int,
) -> Tuple[str, int]:
    fig = _new_page()
    _add_page_header(fig, "Model Performance Metrics", "Core evaluation results")

    ax = fig.add_axes([0.10, 0.63, 0.80, 0.16])
    ax.axis("off")

    table = ax.table(
        cellText=[
            ["Metric", "Value", "Interpretation"],
            ["R²", f"{metrics['r2']:.4f}", "Explained variance of the model"],
            ["RMSE", f"{metrics['rmse']:.2f}", "Penalizes larger prediction errors"],
            ["MAE", f"{metrics['mae']:.2f}", "Average absolute error"],
            ["MSE", f"{metrics['mse']:.2f}", "Mean squared error"],
        ],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.6)
    _style_table(table, header_fontsize=10, body_fontsize=9)

    notes_ax = fig.add_axes([0.10, 0.29, 0.80, 0.20])
    notes_ax.axis("off")
    notes_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.0))
    notes_ax.text(0.03, 0.86, "Interpretation Notes", fontsize=12, fontweight="bold", color=REPORT_ACCENT)

    notes_text = (
        f"This model achieved R² = {metrics['r2']:.4f}. RMSE and MAE should be interpreted "
        f"relative to the scale of the target variable. Lower values generally indicate better fit, "
        f"but diagnostic plots are still needed to assess whether the model behaves well across the data range."
    )
    _draw_paragraph_in_box(
        notes_ax,
        notes_text,
        x=0.03,
        y_top=0.62,
        width_chars=88,
        fontsize=10.3,
        color=REPORT_DARK,
        line_gap=0.15,
    )

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    out = _save_png(fig, export_path, "metrics_table.png")
    plt.close(fig)
    return out, page_num + 1


def _build_feature_bar_chart(ax, feat_names: List[str], feat_vals: List[float]):
    ax.barh(feat_names, feat_vals, color=REPORT_ACCENT, edgecolor="#1f1f1f", linewidth=0.5)
    ax.set_title("Feature Importance", fontsize=12, fontweight="bold", color=REPORT_ACCENT, pad=10)
    ax.set_xlabel("Coefficient", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()


def _build_feature_importance_pages(
    pp: PdfPages,
    artifact_base: str,
    independent_vars: List[str],
    importance: np.ndarray,
    coef_ttests: Optional[List[Dict[str, Any]]],
    export_path: str,
    page_num: int,
) -> Tuple[str, int]:
    sorted_pairs = sorted(
        zip(independent_vars, importance),
        key=lambda x: abs(float(x[1])),
        reverse=True
    )
    feat_names = [x[0] for x in sorted_pairs]
    feat_vals = [float(x[1]) for x in sorted_pairs]

    preview_fig = _new_page()
    _add_page_header(preview_fig, "Feature Analysis", "Coefficient importance and significance tests")
    preview_ax = preview_fig.add_axes([0.18, 0.56, 0.72, 0.28])
    _build_feature_bar_chart(preview_ax, feat_names, feat_vals)
    preview_png_path = _save_png(preview_fig, export_path, "feature_importance.png")
    plt.close(preview_fig)

    if coef_ttests:
        row_chunks = _chunk_list(coef_ttests, 12)
    else:
        row_chunks = [[]]

    for idx, chunk in enumerate(row_chunks):
        fig = _new_page()

        if idx == 0:
            _add_page_header(fig, "Feature Analysis", "Coefficient importance and significance tests")
            chart_ax = fig.add_axes([0.18, 0.56, 0.72, 0.28])
            _build_feature_bar_chart(chart_ax, feat_names, feat_vals)
            table_ax = fig.add_axes([0.08, 0.12, 0.84, 0.36])
            table_ax.axis("off")
        else:
            _add_page_header(fig, "Feature Analysis (Continued)", "Additional coefficient significance rows")
            table_ax = fig.add_axes([0.08, 0.12, 0.84, 0.74])
            table_ax.axis("off")

        if coef_ttests:
            table_data = [["Variable", "Coefficient", "Std Error", "t-stat", "p-value", "Significant"]]
            for row in chunk:
                table_data.append([
                    row["variable"],
                    f"{row['coef']:.6f}",
                    f"{row['std_err']:.6f}",
                    f"{row['t']:.4f}",
                    f"{row['p']:.4f}",
                    "Yes" if row["significant"] else "No"
                ])

            col_widths = [0.20, 0.16, 0.16, 0.14, 0.14, 0.12]
            table = table_ax.table(
                cellText=table_data,
                loc="center",
                cellLoc="center",
                colWidths=col_widths,
            )

            table.scale(1, 1.4 if idx == 0 else 1.7)
            _style_table(table, header_fontsize=8.8, body_fontsize=8.1, significant_col=5)
        else:
            table_ax.text(
                0.5,
                0.5,
                "Coefficient t-test details are not available.",
                ha="center",
                va="center",
                fontsize=11,
                color=REPORT_DARK,
            )

        _add_footer(fig, artifact_base, f"Page {page_num}")
        pp.savefig(fig, facecolor="white")
        plt.close(fig)
        page_num += 1

    return preview_png_path, page_num


def _build_diagnostics_page(
    pp: PdfPages,
    artifact_base: str,
    y_test: pd.Series,
    preds: np.ndarray,
    residuals: pd.Series,
    export_path: str,
    page_num: int,
) -> Tuple[str, str, int]:
    fig = _new_page()
    _add_page_header(fig, "Prediction Diagnostics", "Observed fit and residual behavior")

    ax1 = fig.add_axes([0.14, 0.54, 0.76, 0.32])
    ax1.scatter(y_test, preds, alpha=0.65, color=REPORT_ACCENT, edgecolor="black", linewidth=0.4)
    minv = min(float(np.min(y_test)), float(np.min(preds)))
    maxv = max(float(np.max(y_test)), float(np.max(preds)))
    ax1.plot([minv, maxv], [minv, maxv], "k--", lw=1.3, label="Perfect Prediction")
    ax1.set_title("Actual vs Predicted", fontsize=12, fontweight="bold", color=REPORT_ACCENT, pad=8)
    ax1.set_xlabel("Actual Values", fontsize=9)
    ax1.set_ylabel("Predicted Values", fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.20)

    ax2 = fig.add_axes([0.14, 0.13, 0.76, 0.28])
    ax2.scatter(preds, residuals, alpha=0.65, color="#ef4444", edgecolor="black", linewidth=0.4)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1.3, label="Zero Line")
    ax2.set_title("Residuals vs Predicted", fontsize=12, fontweight="bold", color="#dc2626", pad=8)
    ax2.set_xlabel("Predicted Values", fontsize=9)
    ax2.set_ylabel("Residuals", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.20)

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")

    diag_scatter_path = _save_png(fig, export_path, "actual_vs_predicted.png")
    plt.close(fig)

    resid_fig, resid_ax = plt.subplots(figsize=(6, 5), facecolor="white")
    resid_fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.12)
    resid_ax.scatter(preds, residuals, alpha=0.65, color="#ef4444", edgecolor="black", linewidth=0.4)
    resid_ax.axhline(y=0, color="black", linestyle="--", linewidth=1.3, label="Zero Line")
    resid_ax.set_title("Residuals vs Predicted", fontsize=12, fontweight="bold", color="#dc2626", pad=8)
    resid_ax.set_xlabel("Predicted Values", fontsize=9)
    resid_ax.set_ylabel("Residuals", fontsize=9)
    resid_ax.tick_params(labelsize=8)
    resid_ax.legend(fontsize=8)
    resid_ax.grid(alpha=0.20)
    resid_pred_path = os.path.join(export_path, "residuals_vs_predicted.png")
    resid_fig.savefig(resid_pred_path, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(resid_fig)

    return diag_scatter_path, resid_pred_path, page_num + 1


def _build_residual_distribution_page(
    pp: PdfPages,
    artifact_base: str,
    residuals: pd.Series,
    residual_ttest: Dict[str, float],
    export_path: str,
    page_num: int,
) -> Tuple[str, int]:
    fig = _new_page()
    _add_page_header(fig, "Residual Analysis", "Residual distribution and one-sample t-test")

    ax = fig.add_axes([0.10, 0.52, 0.80, 0.32])
    sns.histplot(residuals, kde=True, ax=ax, color=REPORT_ACCENT, edgecolor="black", bins=20)
    ax.set_title("Residual Distribution", fontsize=13, fontweight="bold", color=REPORT_ACCENT, pad=12)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.20)

    if residual_ttest["p_value"] < 0.05:
        conclusion = "Conclusion: residual mean differs significantly from zero."
    else:
        conclusion = "Conclusion: residual mean is not significantly different from zero."

    info_ax = fig.add_axes([0.10, 0.16, 0.80, 0.22])
    info_ax.axis("off")
    info_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.0))
    info_ax.text(0.03, 0.88, "Residual t-test", fontsize=12, fontweight="bold", color=REPORT_ACCENT)
    info_ax.text(0.03, 0.68, f"T-statistic: {residual_ttest['t_stat']:.4f}", fontsize=10.5, color=REPORT_DARK)
    info_ax.text(0.03, 0.50, f"P-value: {residual_ttest['p_value']:.4f}", fontsize=10.5, color=REPORT_DARK)

    _draw_paragraph_in_box(
        info_ax,
        conclusion,
        x=0.03,
        y_top=0.30,
        width_chars=86,
        fontsize=10.5,
        color=REPORT_DARK,
        line_gap=0.14,
    )

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    out = _save_png(fig, export_path, "residual_distribution.png")
    plt.close(fig)
    return out, page_num + 1


def _build_variable_distribution_pages(
    pp: PdfPages,
    artifact_base: str,
    X_train_unscaled: pd.DataFrame,
    independent_vars: List[str],
    page_num: int,
) -> int:
    plots_per_page = 2

    for start_idx in range(0, len(independent_vars), plots_per_page):
        cols = independent_vars[start_idx:start_idx + plots_per_page]

        fig = _new_page()
        _add_page_header(fig, "Variable Distributions", "Predictor spread and basic descriptive statistics")

        axes_positions = [
            [0.10, 0.54, 0.80, 0.28],
            [0.10, 0.14, 0.80, 0.28],
        ]

        for pos, col in zip(axes_positions, cols):
            ax = fig.add_axes(pos)
            try:
                col_data = X_train_unscaled[col].dropna()
                sns.histplot(col_data, kde=True, ax=ax, color=REPORT_ACCENT, edgecolor="black", bins=25)
                ax.set_title(f"Distribution of {col}", fontsize=12.5, fontweight="bold", color=REPORT_ACCENT, pad=10)
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="y", alpha=0.18)

                mean_val = float(col_data.mean())
                median_val = float(col_data.median())
                std_val = float(col_data.std())

                stats_text = (
                    f"Mean: {mean_val:.2f}\n"
                    f"Median: {median_val:.2f}\n"
                    f"Std: {std_val:.2f}"
                )
                ax.text(
                    0.98,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor=REPORT_ACCENT, alpha=0.92)
                )
            except Exception as e:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"Unable to render distribution for {col}\n{str(e)}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=REPORT_DARK
                )

        _add_footer(fig, artifact_base, f"Page {page_num}")
        pp.savefig(fig, facecolor="white")
        plt.close(fig)
        page_num += 1

    return page_num


def _build_final_summary_page(
    pp: PdfPages,
    artifact_base: str,
    metrics: Dict[str, float],
    independent_vars: List[str],
    importance: np.ndarray,
    target: str,
    n_samples: int,
    residual_ttest: Dict[str, float],
    page_num: int,
) -> int:
    fig = _new_page()
    _add_page_header(fig, "Final Interpretation", "Concise model documentation summary")

    sorted_pairs = sorted(
        zip(independent_vars, importance),
        key=lambda x: abs(float(x[1])),
        reverse=True
    )

    top_items = sorted_pairs[:3]

    box1 = fig.add_axes([0.07, 0.58, 0.40, 0.26])
    box1.axis("off")
    box1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_ACCENT, linewidth=1.3))
    box1.text(0.04, 0.88, "Model Summary", fontsize=12, fontweight="bold", color=REPORT_ACCENT)

    summary_text = (
        f"Target variable: {target}\n"
        f"Training samples: {n_samples:,}\n"
        f"R²: {metrics['r2']:.4f}\n"
        f"RMSE: {metrics['rmse']:.2f}\n"
        f"MAE: {metrics['mae']:.2f}"
    )
    box1.text(0.04, 0.70, summary_text, fontsize=10.5, color=REPORT_DARK, va="top")

    box2 = fig.add_axes([0.53, 0.58, 0.40, 0.26])
    box2.axis("off")
    box2.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    box2.text(0.04, 0.88, "Top Predictors", fontsize=12, fontweight="bold", color=REPORT_ACCENT)

    y = 0.72
    if top_items:
        for idx, (feat, val) in enumerate(top_items, start=1):
            box2.text(0.05, y, f"{idx}. {feat} ({float(val):.4f})", fontsize=10.5, color=REPORT_DARK)
            y -= 0.18
    else:
        box2.text(0.05, 0.72, "No ranked predictors available.", fontsize=10.5, color=REPORT_DARK)

    box3 = fig.add_axes([0.07, 0.20, 0.86, 0.24])
    box3.axis("off")
    box3.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    box3.text(0.02, 0.82, "Documentation Notes", fontsize=12, fontweight="bold", color=REPORT_ACCENT)

    residual_note = (
        "Residual mean is significantly different from zero."
        if residual_ttest["p_value"] < 0.05
        else "Residual mean is not significantly different from zero."
    )

    notes = [
        f"This linear regression model was trained using {len(independent_vars)} predictor(s).",
        f"Overall fit should be judged mainly through R² and diagnostic plots. Current R² is {metrics['r2']:.4f}.",
        residual_note,
        "Use this report together with business or domain validation before deployment.",
    ]

    y = 0.62
    for line in notes:
        wrapped = _wrap_text(line, 105)
        box3.text(0.03, y, f"• {wrapped[0]}", fontsize=10.3, color=REPORT_DARK, va="top")
        y -= 0.11
        for extra in wrapped[1:]:
            box3.text(0.06, y, extra, fontsize=10.3, color=REPORT_DARK, va="top")
            y -= 0.11
        y -= 0.02

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


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
    artifact_base: str = "LR",
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str], str]:

    mse = float(np.mean((y_test - preds) ** 2))
    mae = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(mse))

    from sklearn.metrics import r2_score
    r2 = float(r2_score(y_test, preds))

    std_X = np.std(X_train_scaled, axis=0)
    std_y = np.std(y_train)
    with np.errstate(divide="ignore", invalid="ignore"):
        importance = np.where(std_y == 0, 0, model.coef_ * std_X / std_y)

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

        intercept_var = residual_std_error ** 2 * (
            1 / n + np.mean(X_train_scaled, axis=0) @ XtX_inv @ np.mean(X_train_scaled, axis=0).T
        )
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
        print(f"Could not calculate coefficient t-tests: {e}")
        coef_ttests = None

    metrics = {
        "r2": float(r2),
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
    }

    png_paths: Dict[str, str] = {}

    pdf_path = os.path.join(export_path, f"{artifact_base}.pdf")
    with PdfPages(pdf_path) as pp:
        page_num = 1

        page_num = _build_cover_page(
            pp=pp,
            artifact_base=artifact_base,
            target=target,
            features=independent_vars,
            n_samples=len(y_train),
            page_num=page_num,
        )

        page_num = _build_executive_summary_page(
            pp=pp,
            artifact_base=artifact_base,
            metrics=metrics,
            independent_vars=independent_vars,
            importance=importance,
            residual_ttest=residual_ttest,
            page_num=page_num,
        )

        png_paths["metrics"], page_num = _build_metrics_table_page(
            pp=pp,
            artifact_base=artifact_base,
            metrics=metrics,
            export_path=export_path,
            page_num=page_num,
        )

        png_paths["feature_importance"], page_num = _build_feature_importance_pages(
            pp=pp,
            artifact_base=artifact_base,
            independent_vars=independent_vars,
            importance=importance,
            coef_ttests=coef_ttests,
            export_path=export_path,
            page_num=page_num,
        )

        actual_vs_pred_path, residuals_vs_pred_path, page_num = _build_diagnostics_page(
            pp=pp,
            artifact_base=artifact_base,
            y_test=y_test,
            preds=preds,
            residuals=residuals,
            export_path=export_path,
            page_num=page_num,
        )
        png_paths["actual_vs_predicted"] = actual_vs_pred_path
        png_paths["residuals_vs_predicted"] = residuals_vs_pred_path

        png_paths["residual_distribution"], page_num = _build_residual_distribution_page(
            pp=pp,
            artifact_base=artifact_base,
            residuals=residuals,
            residual_ttest=residual_ttest,
            export_path=export_path,
            page_num=page_num,
        )

        if X_train_unscaled is not None and len(independent_vars) > 0:
            page_num = _build_variable_distribution_pages(
                pp=pp,
                artifact_base=artifact_base,
                X_train_unscaled=X_train_unscaled,
                independent_vars=independent_vars,
                page_num=page_num,
            )

        page_num = _build_final_summary_page(
            pp=pp,
            artifact_base=artifact_base,
            metrics=metrics,
            independent_vars=independent_vars,
            importance=importance,
            target=target,
            n_samples=len(y_train),
            residual_ttest=residual_ttest,
            page_num=page_num,
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
            print(f"Database mode detected: schema={schema}, table={table_name}")
            df_full = df_from_db(schema.strip(), table_name.strip())
        else:
            print("File mode detected")
            gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
            file_gdf = gdf.copy()
            df_full = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

        total_rows_before = len(df_full)

        try:
            excluded = json.loads(excluded_indices or "[]")
            excluded_count = len(excluded)
            if excluded_count:
                print(f"Excluding {excluded_count} rows before training...")
                df_full = df_full.drop(df_full.index[excluded]).reset_index(drop=True)
            else:
                print("No excluded rows received.")
        except Exception as e:
            print(f"Could not parse excluded_indices: {e}")
            excluded_count = 0

        df_full["__original_index__"] = df_full.index
        print(f"Stored original indices for {len(df_full)} rows after exclusions")

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

        artifact_base = build_artifact_base_name("LR")
        export_path = os.path.join(EXPORT_DIR, artifact_base)
        os.makedirs(export_path, exist_ok=True)

        model_path = os.path.join(export_path, f"{artifact_base}.pkl")
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "features": [v.lower() for v in indep],
                "dependent_var": target.lower(),
                "model_type": "lr",
                "trained_at": datetime.now().isoformat(),
            },
            model_path,
        )
        print(f"Saved model: {os.path.basename(model_path)}")

        metrics, png_paths, t_tests, pdf_path = export_full_report_and_artifacts(
            export_path,
            model,
            scaler,
            indep,
            target,
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            preds,
            residuals,
            X_train_unscaled=X_train,
            artifact_base=artifact_base,
        )

        preds_valid = model.predict(scaler.transform(df_valid[indep]))
        df_valid = df_valid.copy()
        df_valid["prediction"] = preds_valid

        safe_target_name = "actual_val" if len(target) > 10 else target

        csv_path = os.path.join(export_path, f"{artifact_base}.csv")
        csv_df = df_valid[indep + [target, "prediction"]].copy()

        if pin_series is not None:
            csv_df.insert(0, "PIN", pin_series.iloc[df_valid.index].values)

        csv_df.to_csv(csv_path, index=False)
        print(f"Exported cleaned CSV (excluded rows removed): {csv_path}")

        zip_out = None
        try:
            original_indices = df_valid["__original_index__"].tolist()
            print(f"Original indices to map: {original_indices[:10]}... (showing first 10)")

            if is_db_mode:
                print("Database mode: fetching geometry for export")
                gdf_db = gdf_from_db_with_geometry(schema, table_name)
                valid_gdf = gdf_db.iloc[original_indices].copy()

                print(f"GeoDataFrame shape: {valid_gdf.shape}")
                print(f"Valid rows count: {len(df_valid)}")

                columns_to_drop = []
                for col in valid_gdf.columns:
                    if col.upper() == "UNIT_VALUE":
                        columns_to_drop.append(col)
                        print(f"Dropping original column '{col}' to avoid collision")
                    elif col.upper() == "MARKET_VAL":
                        columns_to_drop.append(col)
                        print(f"Dropping original column '{col}'")

                if columns_to_drop:
                    valid_gdf = valid_gdf.drop(columns=columns_to_drop, errors="ignore")

                if pin_series is not None:
                    try:
                        upsert_pin_field(valid_gdf, pin_series.iloc[original_indices].values, preferred_name="PIN")
                        drop_duplicate_pin_fields(valid_gdf, keep_name="PIN")
                        print("PIN field updated (no duplicates)")
                    except Exception as e:
                        print(f"Could not update PIN field: {e}")

                valid_gdf[safe_target_name] = df_valid[target].values
                print(f"Added actual values as '{safe_target_name}'")

                valid_gdf["prediction"] = df_valid["prediction"].values
                print("Added prediction column")

                print(f"\n{'=' * 60}")
                print("VERIFICATION:")
                print(f"Rows in valid_gdf: {len(valid_gdf)}")
                print(f"'{safe_target_name}' exists: {safe_target_name in valid_gdf.columns}")
                print(f"Sample actual values: {valid_gdf[safe_target_name].head().tolist()}")
                print(f"'prediction' exists: {'prediction' in valid_gdf.columns}")
                print(f"Sample predictions: {valid_gdf['prediction'].head().tolist()}")
                print(f"{'=' * 60}\n")

                shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
                os.makedirs(shp_pred_dir, exist_ok=True)
                shp_pred_path = os.path.join(shp_pred_dir, "predicted_output.shp")

                valid_gdf = valid_gdf.drop(columns=["__original_index__"], errors="ignore")
                valid_gdf.to_file(shp_pred_path)
                print(f"Shapefile saved: {shp_pred_path}")

                zip_out = os.path.join(export_path, "predicted_output.zip")
                with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in os.listdir(shp_pred_dir):
                        z.write(os.path.join(shp_pred_dir, f), f)
                print(f"ZIP created: {zip_out}")

            elif file_gdf is not None:
                print("File mode: using uploaded geometry for export")
                valid_gdf = file_gdf.iloc[original_indices].copy()

                print(f"GeoDataFrame shape: {valid_gdf.shape}")
                print(f"Columns: {valid_gdf.columns.tolist()}")

                columns_to_drop = []
                for col in valid_gdf.columns:
                    if col.upper() == "UNIT_VALUE":
                        columns_to_drop.append(col)
                        print(f"Dropping original column '{col}' to avoid collision")
                    elif col.upper() == "MARKET_VAL":
                        columns_to_drop.append(col)
                        print(f"Dropping original column '{col}' (too large values)")

                if columns_to_drop:
                    valid_gdf = valid_gdf.drop(columns=columns_to_drop, errors="ignore")

                if pin_series is not None:
                    try:
                        upsert_pin_field(valid_gdf, pin_series.iloc[original_indices].values, preferred_name="PIN")
                        drop_duplicate_pin_fields(valid_gdf, keep_name="PIN")
                        print("PIN field updated (no duplicates)")
                    except Exception as e:
                        print(f"Could not update PIN field: {e}")

                valid_gdf[safe_target_name] = df_valid[target].values
                print(f"Added actual values as '{safe_target_name}'")

                print(f"Checking column '{safe_target_name}': {safe_target_name in valid_gdf.columns}")
                print(f"Sample values: {valid_gdf[safe_target_name].head(3).tolist()}")

                valid_gdf["prediction"] = df_valid["prediction"].values
                print("Added prediction column")

                print(f"\nFinal columns in shapefile ({len(valid_gdf.columns)} total):")
                for col in valid_gdf.columns:
                    print(f" - {col} (length: {len(col)})")

                shp_pred_dir = os.path.join(export_path, "predicted_shapefile")
                os.makedirs(shp_pred_dir, exist_ok=True)
                shp_pred_path = os.path.join(shp_pred_dir, "predicted_output.shp")

                valid_gdf = valid_gdf.drop(columns=["__original_index__"], errors="ignore")

                print("\nSaving shapefile with these key columns:")
                print(f" - {safe_target_name}: {safe_target_name in valid_gdf.columns}")
                print(f" - prediction: {'prediction' in valid_gdf.columns}")

                valid_gdf.to_file(shp_pred_path)
                print(f"Shapefile created with {len(valid_gdf)} features")

                zip_out = os.path.join(export_path, "predicted_output.zip")
                with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in os.listdir(shp_pred_dir):
                        z.write(os.path.join(shp_pred_dir, f), f)

                print(f"Created ZIP: {zip_out}")

            else:
                print("No geometry data available (no shapefile output)")

        except Exception as e:
            print(f"Error creating shapefile output: {e}")
            import traceback
            traceback.print_exc()
            zip_out = None

        counts, bins = np.histogram(residuals, bins=20)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        print("Computing variable distributions...")
        variable_distributions = compute_variable_distributions(df_valid, indep)
        print(f"Computed distributions for {len(variable_distributions)} variables")

        base_url = "/api/ai-tools/download"
        plots = {key: f"{base_url}?file={path}" for key, path in png_paths.items()}

        downloads = {
            "model": f"{base_url}?file={model_path}",
            "report": f"{base_url}?file={pdf_path}",
            "cama_csv": f"{base_url}?file={csv_path}",
        }

        if zip_out:
            downloads["shapefile"] = f"{base_url}?file={zip_out}"
            downloads["geojson"] = f"/api/ai-tools/preview-geojson?file_path={zip_out}"

        print("Creating training result preview...")
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
        print(f"Created preview with {len(cama_preview)} rows")

        metrics = {k: float(v) for k, v in metrics.items()}

        return {
            "model_name": artifact_base,
            "model_id": artifact_base,
            "dependent_var": safe_target_name,
            "original_dependent_var": target,
            "metrics": metrics,
            "features": indep,
            "importance": [
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
        print(f"TRAIN ERROR: {e}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})