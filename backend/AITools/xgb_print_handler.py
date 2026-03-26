"""
xgb_print_handler.py
--------------------
PDF report generation handler for the XGBoost training module.
Same design system as lr_print_handler.py — shared constants, layout,
typography, and helper functions — but content is XGBoost-specific:
model hyperparameters on the cover, feature importances (not coefficients),
no t-test pages, and an XGBoost-specific final summary.
"""

from typing import List, Optional, Tuple, Dict, Any
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from scipy import stats


# ---------------------------------------------------------------------------
# Report color constants  (identical to lr_print_handler)
# ---------------------------------------------------------------------------

REPORT_ACCENT = "#1e88e5"
REPORT_DARK   = "#1f2937"
REPORT_LIGHT  = "#f8fafc"
REPORT_BORDER = "#d0d7de"


# ---------------------------------------------------------------------------
# Low-level drawing helpers  (identical to lr_print_handler)
# ---------------------------------------------------------------------------

def _new_page(figsize=(8.27, 11.69)):
    fig = plt.figure(figsize=figsize, facecolor="white")
    return fig


def _add_page_header(fig, title: str, subtitle: Optional[str] = None):
    fig.text(0.07, 0.972, title, fontsize=20, fontweight="bold", color=REPORT_ACCENT, va="top")
    if subtitle:
        fig.text(0.07, 0.945, subtitle, fontsize=10.5, color="#5f6b7a", va="top")
    fig.lines.append(
        plt.Line2D(
            [0.07, 0.93],
            [0.928, 0.928],
            transform=fig.transFigure,
            color=REPORT_BORDER,
            linewidth=1.2,
        )
    )


def _add_footer(fig, artifact_base: str, page_label: str):
    fig.lines.append(
        plt.Line2D(
            [0.07, 0.93],
            [0.06, 0.06],
            transform=fig.transFigure,
            color=REPORT_BORDER,
            linewidth=0.8,
        )
    )
    fig.text(0.07, 0.035, f"Model Report | {artifact_base}", fontsize=8.5, color="#6b7280")
    fig.text(0.93, 0.035, page_label, fontsize=8.5, color="#6b7280", ha="right")


def _save_png(fig, export_path: str, filename: str) -> str:
    out = os.path.join(export_path, filename)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=200)
    return out



# ---------------------------------------------------------------------------
# Physical layout constants
# ---------------------------------------------------------------------------
_FIG_W_IN   = 8.27
_FIG_H_IN   = 11.69
_CHAR_W_10  = 0.069
_LINE_H_10  = 0.158


def _wrap_text(text: str, chars_per_line: int) -> list:
    words = str(text).split()
    lines, current = [], ""
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


def _cpl(axes_width: float, x_pad: float, fontsize: float) -> int:
    usable_in = _FIG_W_IN * axes_width * (1.0 - x_pad * 2)
    return max(20, int(usable_in / (_CHAR_W_10 * fontsize / 10.0)))


def _text_box(
    fig,
    left: float,
    bottom: float,
    width: float,
    text: str,
    title: Optional[str] = None,
    fontsize: float = 10.3,
    title_fontsize: float = 11.0,
    x_pad: float = 0.04,
    pad_top_in: float = 0.16,
    pad_bot_in: float = 0.16,
    title_gap_in: float = 0.26,
    facecolor: str = "white",
    edgecolor: str = REPORT_BORDER,
    title_color: str = REPORT_ACCENT,
    text_color: str = REPORT_DARK,
) -> float:
    """Auto-sizing text box. Returns total height in figure-fraction units."""
    chars       = _cpl(width, x_pad, fontsize)
    lines       = _wrap_text(text, chars)
    line_h_in   = _LINE_H_10 * fontsize / 10.0
    title_h_in  = title_gap_in if title else 0.0
    total_h_in  = pad_top_in + title_h_in + len(lines) * line_h_in + pad_bot_in
    total_h_frac = total_h_in / _FIG_H_IN

    ax = fig.add_axes([left, bottom, width, total_h_frac])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True,
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2))

    pad_top_ax = pad_top_in  / total_h_in
    title_h_ax = title_h_in  / total_h_in
    line_h_ax  = line_h_in   / total_h_in

    y = 1.0 - pad_top_ax
    if title:
        ax.text(x_pad, y, title, fontsize=title_fontsize, fontweight="bold",
                color=title_color, va="top", clip_on=True)
        y -= title_h_ax
    for line in lines:
        ax.text(x_pad, y, line, fontsize=fontsize, color=text_color,
                va="top", clip_on=True)
        y -= line_h_ax

    return total_h_frac


def _chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]


def _style_table(table, header_fontsize=9, body_fontsize=8.8, highlight_col: Optional[int] = None):
    """Style a matplotlib table. highlight_col marks a column with green/red shading."""
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#222222")
        cell.set_linewidth(0.8)
        if i == 0:
            cell.set_facecolor(REPORT_ACCENT)
            cell.set_text_props(weight="bold", color="white", fontsize=header_fontsize)
        else:
            if highlight_col is not None and j == highlight_col:
                txt = cell.get_text().get_text()
                cell.set_facecolor("#d1fae5" if txt == "Yes" else "#fee2e2")
            else:
                cell.set_facecolor("#f5f7fa" if i % 2 == 1 else "white")
            cell.set_text_props(color=REPORT_DARK, fontsize=body_fontsize)


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
        f"while RMSE = {rmse:.2f} captures the effect of larger errors."
    )

    if top_feature is not None and top_value is not None:
        feature_text = (
            f"Most influential predictor: {top_feature} "
            f"(importance score = {top_value:.4f})."
        )
    else:
        feature_text = "Feature importance could not be ranked."

    return [perf_text, error_text, feature_text]


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _build_cover_page(
    pp: PdfPages,
    artifact_base: str,
    target: str,
    features: List[str],
    n_samples: int,
    hyperparams: Dict[str, Any],
    scaler_choice: str,
    page_num: int,
) -> int:
    fig = _new_page()
    fig.text(0.07, 0.88, "XGBoost Model Report", fontsize=24, fontweight="bold", color=REPORT_ACCENT)

    # --- Definition box (first) ---
    definition = (
        "XGBoost (Extreme Gradient Boosting) is an ensemble machine learning algorithm that builds "
        "a series of decision trees sequentially, where each tree corrects the errors of the previous "
        "one. It uses gradient boosting to minimize prediction error, and is known for its speed, "
        "accuracy, and ability to capture complex non-linear relationships between features and the "
        "target variable."
    )
    _text_box(fig, 0.07, 0.64, 0.86, definition,
              title="What is XGBoost?",
              fontsize=9.5, title_fontsize=11,
              facecolor="#f0f7ff", pad_top_in=0.12, pad_bot_in=0.12, title_gap_in=0.22)

    # --- Hyperparameters box ---
    hyp_ax = fig.add_axes([0.07, 0.40, 0.86, 0.20])
    hyp_ax.axis("off")
    hyp_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, facecolor="#f4f9ff", edgecolor=REPORT_BORDER, linewidth=1.2))
    hyp_ax.text(0.03, 0.88, "Model Hyperparameters", fontsize=11, fontweight="bold", color=REPORT_ACCENT, va="top")

    hyp_items = [
        ("n_estimators",     str(hyperparams.get("n_estimators", 300))),
        ("learning_rate",    str(hyperparams.get("learning_rate", 0.05))),
        ("max_depth",        str(hyperparams.get("max_depth", 6))),
        ("subsample",        str(hyperparams.get("subsample", 0.8))),
        ("colsample_bytree", str(hyperparams.get("colsample_bytree", 0.8))),
        ("objective",        str(hyperparams.get("objective", "reg:squarederror"))),
    ]

    col_break = len(hyp_items) // 2 + len(hyp_items) % 2
    y = 0.66
    for idx, (label, value) in enumerate(hyp_items):
        col_x = 0.03 if idx < col_break else 0.52
        row_y = y - (idx % col_break) * 0.13
        hyp_ax.text(col_x,        row_y, label, fontsize=9.5, fontweight="bold", color=REPORT_DARK, va="top")
        hyp_ax.text(col_x + 0.22, row_y, value, fontsize=9.5, color=REPORT_DARK, va="top")

    # --- Model info box (last) ---
    meta_ax = fig.add_axes([0.07, 0.13, 0.86, 0.23])
    meta_ax.axis("off")
    meta_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    meta_ax.add_patch(plt.Rectangle((0, 0.84), 1, 0.16, facecolor="#f4f9ff", edgecolor=REPORT_BORDER, linewidth=1.2))
    meta_ax.text(0.03, 0.92, "Model Information", fontsize=12, fontweight="bold", color=REPORT_ACCENT, va="center")

    meta_lines = [
        ("Model Type",       "XGBoost Regressor"),
        ("Model Name",       artifact_base),
        ("Target Variable",  target),
        ("Feature Count",    str(len(features))),
        ("Training Samples", f"{n_samples:,}"),
        ("Scaler Used",      scaler_choice),
        ("Generated At",     datetime.now().strftime("%Y-%b-%d %I:%M:%S %p")),
    ]

    y = 0.74
    for label, value in meta_lines:
        meta_ax.text(0.03, y, label, fontsize=10.5, fontweight="bold", color=REPORT_DARK, va="center")
        meta_ax.text(0.30, y, value, fontsize=10.5, color=REPORT_DARK, va="center")
        y -= 0.097

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


def _build_executive_summary_page(
    pp: PdfPages,
    artifact_base: str,
    metrics: Dict[str, float],
    feature_names: List[str],
    importance: np.ndarray,
    page_num: int,
) -> int:
    fig = _new_page()
    _add_page_header(fig, "Executive Summary", "Key results and top-level interpretation")

    sorted_pairs = sorted(
        zip(feature_names, importance),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    top_feature = sorted_pairs[0][0] if sorted_pairs else None
    top_value   = float(sorted_pairs[0][1]) if sorted_pairs else None

    summary_lines = _metrics_interpretation_text(
        r2=metrics["R²"],
        rmse=metrics["RMSE"],
        mae=metrics["MAE"],
        top_feature=top_feature,
        top_value=top_value,
    )

    # Left box — performance
    BOX_LEFT_X   = 0.07
    BOX_RIGHT_X  = 0.52
    BOX_WIDTH    = 0.41
    BOX_BOTTOM   = 0.68
    BOX_HEIGHT   = 0.20
    TITLE_Y      = 0.91
    CONTENT_Y    = 0.80
    CONT_STEP    = 0.095

    left_ax = fig.add_axes([BOX_LEFT_X, BOX_BOTTOM, BOX_WIDTH, BOX_HEIGHT])
    left_ax.axis("off")
    left_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    left_ax.text(0.04, TITLE_Y, "Performance Summary", fontsize=12, fontweight="bold", color=REPORT_ACCENT)

    y = CONTENT_Y
    for line in summary_lines:
        wrapped = _wrap_text(line, chars_per_line=42)
        left_ax.text(0.05, y, f"• {wrapped[0]}", fontsize=9.5, color=REPORT_DARK, va="top")
        sub_y = y - CONT_STEP
        for extra in wrapped[1:]:
            left_ax.text(0.08, sub_y, extra, fontsize=9.5, color=REPORT_DARK, va="top")
            sub_y -= CONT_STEP
        y = sub_y - 0.04

    # Right box — top features
    right_ax = fig.add_axes([BOX_RIGHT_X, BOX_BOTTOM, BOX_WIDTH, BOX_HEIGHT])
    right_ax.axis("off")
    right_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    right_ax.text(0.04, TITLE_Y, "Top Feature Importances", fontsize=12, fontweight="bold", color=REPORT_ACCENT)

    y = CONTENT_Y
    for rank, (feat, val) in enumerate(sorted_pairs[:5], start=1):
        line = f"{rank}. {feat}  ({float(val):.4f})"
        right_ax.text(0.05, y, line, fontsize=9.5, color=REPORT_DARK, va="top")
        y -= 0.13

    # Bottom box — reading guide
    rec_text = (
        "Use the metrics page to evaluate overall fit, the feature importance page to review "
        "which predictors contributed most, and the diagnostics page to inspect prediction "
        "bias and residual behavior. Variable distribution pages provide context for predictor spread."
    )
    _text_box(fig, 0.07, 0.48, 0.86, rec_text,
              title="Recommended Reading of this Report",
              fontsize=10.5, title_fontsize=12,
              x_pad=0.02, facecolor="white", pad_top_in=0.12, pad_bot_in=0.12, title_gap_in=0.24)

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

    ax = fig.add_axes([0.10, 0.74, 0.80, 0.14])
    ax.axis("off")

    table = ax.table(
        cellText=[
            ["Metric", "Value", "Interpretation"],
            ["R²",   f"{metrics['R²']:.4f}",   "Explained variance of the model"],
            ["RMSE", f"{metrics['RMSE']:.2f}",  "Penalizes larger prediction errors"],
            ["MAE",  f"{metrics['MAE']:.2f}",   "Average absolute error"],
            ["MSE",  f"{metrics['MSE']:.2f}",   "Mean squared error"],
        ],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.6)
    _style_table(table, header_fontsize=10, body_fontsize=9)

    notes_text = (
        f"This XGBoost model achieved R² = {metrics['R²']:.4f}. RMSE and MAE should be interpreted "
        f"relative to the scale of the target variable. Unlike linear models, XGBoost captures "
        f"non-linear interactions between features, so residual patterns may differ from classical regression."
    )
    _text_box(fig, 0.07, 0.60, 0.86, notes_text,
              title="Interpretation Notes",
              fontsize=10.3, title_fontsize=12,
              x_pad=0.02, facecolor="white", pad_top_in=0.12, pad_bot_in=0.12, title_gap_in=0.24)

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    out = _save_png(fig, export_path, "metrics_table.png")
    plt.close(fig)
    return out, page_num + 1


def _build_feature_importance_page(
    pp: PdfPages,
    artifact_base: str,
    feature_names: List[str],
    importance: np.ndarray,
    export_path: str,
    page_num: int,
) -> Tuple[str, int]:
    sorted_pairs = sorted(
        zip(feature_names, importance),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    feat_names = [x[0] for x in sorted_pairs]
    feat_vals  = [float(x[1]) for x in sorted_pairs]

    fig = _new_page()
    _add_page_header(fig, "Feature Importance", "XGBoost gain-based feature contribution scores")

    # Bar chart
    chart_ax = fig.add_axes([0.15, 0.50, 0.76, 0.36])
    chart_ax.barh(feat_names, feat_vals, color=REPORT_ACCENT, edgecolor="#1f1f1f", linewidth=0.5)
    chart_ax.set_title("Feature Importance (Gain)", fontsize=12, fontweight="bold", color=REPORT_ACCENT, pad=10)
    chart_ax.set_xlabel("Importance Score", fontsize=9)
    chart_ax.tick_params(axis="y", labelsize=8)
    chart_ax.tick_params(axis="x", labelsize=8.5)
    chart_ax.spines["top"].set_visible(False)
    chart_ax.spines["right"].set_visible(False)
    chart_ax.grid(axis="x", alpha=0.25)
    chart_ax.invert_yaxis()

    # Importance table
    table_ax = fig.add_axes([0.10, 0.16, 0.80, 0.28])
    table_ax.axis("off")

    row_chunks = _chunk_list(
        [[f, f"{v:.6f}", f"{v / sum(feat_vals) * 100:.1f}%" if sum(feat_vals) > 0 else "N/A"]
         for f, v in zip(feat_names, feat_vals)],
        12,
    )
    chunk = row_chunks[0]  # first page shows top 12

    table_data = [["Feature", "Importance Score", "% of Total"]] + chunk
    table = table_ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.40, 0.30, 0.20],
    )
    table.scale(1, 1.4)
    _style_table(table, header_fontsize=9, body_fontsize=8.3)

    preview_png_path = _save_png(fig, export_path, "feature_importance.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    page_num += 1

    # Overflow pages for >12 features
    for extra_chunk in row_chunks[1:]:
        fig = _new_page()
        _add_page_header(fig, "Feature Importance (Continued)", "Additional feature contribution scores")

        table_ax = fig.add_axes([0.10, 0.12, 0.80, 0.74])
        table_ax.axis("off")
        table_data = [["Feature", "Importance Score", "% of Total"]] + extra_chunk
        table = table_ax.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
            colWidths=[0.40, 0.30, 0.20],
        )
        table.scale(1, 1.7)
        _style_table(table, header_fontsize=9, body_fontsize=8.3)

        _add_footer(fig, artifact_base, f"Page {page_num}")
        pp.savefig(fig, facecolor="white")
        plt.close(fig)
        page_num += 1

    return preview_png_path, page_num


def _build_diagnostics_page(
    pp: PdfPages,
    artifact_base: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    export_path: str,
    page_num: int,
) -> Tuple[str, str, int]:
    fig = _new_page()
    _add_page_header(fig, "Prediction Diagnostics", "Observed fit and residual behavior")

    ax1 = fig.add_axes([0.14, 0.56, 0.76, 0.29])
    ax1.scatter(y_test, y_pred, alpha=0.65, color=REPORT_ACCENT, edgecolor="black", linewidth=0.4)
    minv = min(float(np.min(y_test)), float(np.min(y_pred)))
    maxv = max(float(np.max(y_test)), float(np.max(y_pred)))
    ax1.plot([minv, maxv], [minv, maxv], "k--", lw=1.3, label="Perfect Prediction")
    ax1.set_title("Actual vs Predicted", fontsize=12, fontweight="bold", color=REPORT_ACCENT, pad=8)
    ax1.set_xlabel("Actual Values", fontsize=9)
    ax1.set_ylabel("Predicted Values", fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.20)

    ax2 = fig.add_axes([0.14, 0.15, 0.76, 0.24])
    ax2.scatter(y_pred, residuals, alpha=0.65, color="#ef4444", edgecolor="black", linewidth=0.4)
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

    # Standalone residuals-vs-predicted PNG
    resid_fig, resid_ax = plt.subplots(figsize=(6, 5), facecolor="white")
    resid_fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.12)
    resid_ax.scatter(y_pred, residuals, alpha=0.65, color="#ef4444", edgecolor="black", linewidth=0.4)
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
    residuals: np.ndarray,
    export_path: str,
    page_num: int,
) -> Tuple[str, int]:
    t_stat, p_val = stats.ttest_1samp(residuals, 0)

    fig = _new_page()
    _add_page_header(fig, "Residual Analysis", "Residual distribution and one-sample t-test")

    ax = fig.add_axes([0.12, 0.32, 0.76, 0.52])
    sns.histplot(residuals, kde=True, ax=ax, color=REPORT_ACCENT, edgecolor="black", bins=20)

    # Linear y-axis ticks: 0, 250, 500, 750, 1k, 2k, 4k, 6k, 8k, 10k
    custom_ticks = [0, 250, 500, 750, 1000, 2000, 4000, 6000, 8000, 10000]
    data_max = ax.get_ylim()[1]
    visible_ticks = [t for t in custom_ticks if t <= data_max * 1.10]
    if len(visible_ticks) < 2:
        visible_ticks = custom_ticks[:4]
    ax.set_yticks(visible_ticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: '0' if x == 0 else (f'{int(x)}' if x < 1000 else f'{int(x/1000):.0f}k')
    ))
    ax.set_ylim(0, max(visible_ticks) * 1.10)

    ax.set_title("Residual Distribution", fontsize=13, fontweight="bold", color=REPORT_ACCENT, pad=12)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.20)

    conclusion = (
        "Conclusion: residual mean differs significantly from zero."
        if p_val < 0.05
        else "Conclusion: residual mean is not significantly different from zero."
    )

    ttest_text = (
        f"T-statistic: {t_stat:.4f}     "
        f"P-value: {p_val:.4f}     "
        f"{conclusion}"
    )
    _text_box(fig, 0.10, 0.17, 0.80, ttest_text,
              title="Residual t-test",
              fontsize=10.5, title_fontsize=12,
              facecolor="white", pad_top_in=0.12, pad_bot_in=0.12, title_gap_in=0.24)

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    out = _save_png(fig, export_path, "residual_distribution.png")
    plt.close(fig)
    return out, page_num + 1


def _build_variable_distribution_pages(
    pp: PdfPages,
    artifact_base: str,
    df_valid: pd.DataFrame,
    independent_vars: List[str],
    page_num: int,
) -> int:
    plots_per_page = 2
    axes_positions = [
        [0.10, 0.57, 0.80, 0.24],
        [0.10, 0.16, 0.80, 0.24],
    ]

    for start_idx in range(0, len(independent_vars), plots_per_page):
        cols = independent_vars[start_idx:start_idx + plots_per_page]
        fig = _new_page()
        _add_page_header(fig, "Variable Distributions", "Predictor spread and basic descriptive statistics")

        for pos, col in zip(axes_positions, cols):
            ax = fig.add_axes(pos)
            try:
                col_data = df_valid[col].dropna()
                sns.histplot(col_data, kde=True, ax=ax, color=REPORT_ACCENT, edgecolor="black", bins=25)
                ax.set_title(f"Distribution of {col}", fontsize=12.5, fontweight="bold", color=REPORT_ACCENT, pad=10)
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="y", alpha=0.18)

                stats_text = (
                    f"Mean: {float(col_data.mean()):.2f}\n"
                    f"Median: {float(col_data.median()):.2f}\n"
                    f"Std: {float(col_data.std()):.2f}"
                )
                ax.text(
                    0.98, 0.95, stats_text,
                    transform=ax.transAxes,
                    va="top", ha="right", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor=REPORT_ACCENT, alpha=0.92),
                )
            except Exception as e:
                ax.axis("off")
                ax.text(
                    0.5, 0.5,
                    f"Unable to render distribution for {col}\n{str(e)}",
                    ha="center", va="center",
                    fontsize=11, color=REPORT_DARK,
                )

        _add_footer(fig, artifact_base, f"Page {page_num}")
        pp.savefig(fig, facecolor="white")
        plt.close(fig)
        page_num += 1

    return page_num



def _draw_doc_notes_box(fig, notes: List[str], bottom: float, chars_per_line: int = 72) -> None:
    """
    Draw the Documentation Notes box with a height that auto-expands
    to fit all bullet lines — never clips, never overflows right edge.
    """
    line_gap  = 0.024
    title_h   = 0.042
    pad_top   = 0.022
    pad_bot   = 0.022
    indent    = "    "  # continuation-line indent

    # Pre-compute all display lines
    display_lines = []
    for note in notes:
        wrapped = _wrap_text(note, chars_per_line)
        display_lines.append(f"• {wrapped[0]}")
        for extra in wrapped[1:]:
            display_lines.append(f"{indent}{extra}")

    total_h = pad_top + title_h + len(display_lines) * line_gap + pad_bot
    total_h = max(total_h, 0.12)   # minimum sensible height

    ax = fig.add_axes([0.07, bottom, 0.86, total_h])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1,
        fill=False, edgecolor=REPORT_BORDER, linewidth=1.2
    ))

    # Title
    title_y = 1.0 - pad_top / total_h
    ax.text(0.03, title_y, "Documentation Notes",
            fontsize=12, fontweight="bold", color=REPORT_ACCENT, va="top", clip_on=True)

    # Bullets
    line_gap_ax = line_gap / total_h      # convert to axes coords
    y = title_y - title_h / total_h
    for line in display_lines:
        ax.text(0.03, y, line, fontsize=10.3, color=REPORT_DARK, va="top", clip_on=True)
        y -= line_gap_ax


def _build_final_summary_page(
    pp: PdfPages,
    artifact_base: str,
    metrics: Dict[str, float],
    feature_names: List[str],
    importance: np.ndarray,
    target: str,
    n_samples: int,
    hyperparams: Dict[str, Any],
    page_num: int,
) -> int:
    fig = _new_page()
    _add_page_header(fig, "Final Interpretation", "Concise model documentation summary")

    sorted_pairs = sorted(
        zip(feature_names, importance),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    top_items = sorted_pairs[:3]

    # Left — model summary
    BOX_LEFT_X  = 0.07
    BOX_RIGHT_X = 0.52
    BOX_WIDTH   = 0.41
    BOX_BOTTOM  = 0.70
    BOX_HEIGHT  = 0.20

    box1 = fig.add_axes([BOX_LEFT_X, BOX_BOTTOM, BOX_WIDTH, BOX_HEIGHT])
    box1.axis("off")
    box1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    box1.text(0.04, 0.91, "Model Summary", fontsize=12, fontweight="bold", color=REPORT_ACCENT, va="top")

    summary_lines_list = [
        f"Target variable: {target}",
        f"Training samples: {n_samples:,}",
        f"R²: {metrics['R²']:.4f}",
        f"RMSE: {metrics['RMSE']:.2f}",
        f"MAE: {metrics['MAE']:.2f}",
    ]
    sy = 0.76
    for sline in summary_lines_list:
        box1.text(0.04, sy, sline, fontsize=10.5, color=REPORT_DARK, va="top")
        sy -= 0.13

    # Right — top predictors
    box2 = fig.add_axes([BOX_RIGHT_X, BOX_BOTTOM, BOX_WIDTH, BOX_HEIGHT])
    box2.axis("off")
    box2.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    box2.text(0.04, 0.91, "Top Predictors", fontsize=12, fontweight="bold", color=REPORT_ACCENT, va="top")

    y = 0.76
    if top_items:
        for idx, (feat, val) in enumerate(top_items, start=1):
            box2.text(0.05, y, f"{idx}. {feat}  ({float(val):.4f})", fontsize=10.5, color=REPORT_DARK, va="top")
            y -= 0.18
    else:
        box2.text(0.05, y, "No ranked predictors available.", fontsize=10.5, color=REPORT_DARK, va="top")

    # Bottom — documentation notes

    notes = [
        f"This XGBoost model was trained using {len(feature_names)} predictor(s) with "
        f"{hyperparams.get('n_estimators', 300)} estimators at learning rate {hyperparams.get('learning_rate', 0.05)}.",
        f"Overall fit should be judged through R² and diagnostic plots. Current R² is {metrics['R²']:.4f}.",
        "Feature importance scores reflect gain contribution and are not directly comparable to regression coefficients.",
        "Use this report together with domain validation before deploying predictions.",
    ]

    _draw_doc_notes_box(fig, notes, bottom=0.33)

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def export_xgb_report_and_artifacts(
    export_path: str,
    model,
    scaler,
    feature_names: List[str],
    target: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    df_valid: pd.DataFrame,
    scaler_choice: str = "None",
    artifact_base: str = "XGB",
) -> Tuple[Dict[str, float], Dict[str, str], str]:
    """
    Build the full PDF report and return:
      metrics   — dict with R², RMSE, MAE, MSE
      png_paths — dict of standalone PNG paths for the frontend
      pdf_path  — path to the generated PDF
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    os.makedirs(export_path, exist_ok=True)

    mse  = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    metrics: Dict[str, float] = {
        "R²":   r2,
        "RMSE": rmse,
        "MAE":  mae,
        "MSE":  mse,
    }

    residuals = y_test - y_pred

    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        importance = np.zeros(len(feature_names))

    hyperparams = {
        "n_estimators":    getattr(model, "n_estimators",    300),
        "learning_rate":   getattr(model, "learning_rate",   0.05),
        "max_depth":       getattr(model, "max_depth",       6),
        "subsample":       getattr(model, "subsample",       0.8),
        "colsample_bytree":getattr(model, "colsample_bytree",0.8),
        "objective":       getattr(model, "objective",       "reg:squarederror"),
    }

    png_paths: Dict[str, str] = {}

    pdf_path = os.path.join(export_path, f"{artifact_base}.pdf")
    with PdfPages(pdf_path) as pp:
        page_num = 1

        page_num = _build_cover_page(
            pp=pp, artifact_base=artifact_base,
            target=target, features=feature_names,
            n_samples=len(y_train), hyperparams=hyperparams,
            scaler_choice=scaler_choice, page_num=page_num,
        )

        page_num = _build_executive_summary_page(
            pp=pp, artifact_base=artifact_base,
            metrics=metrics, feature_names=feature_names,
            importance=importance, page_num=page_num,
        )

        png_paths["metrics"], page_num = _build_metrics_table_page(
            pp=pp, artifact_base=artifact_base,
            metrics=metrics, export_path=export_path,
            page_num=page_num,
        )

        png_paths["feature_importance"], page_num = _build_feature_importance_page(
            pp=pp, artifact_base=artifact_base,
            feature_names=feature_names, importance=importance,
            export_path=export_path, page_num=page_num,
        )

        actual_vs_pred_path, residuals_vs_pred_path, page_num = _build_diagnostics_page(
            pp=pp, artifact_base=artifact_base,
            y_test=y_test, y_pred=y_pred,
            residuals=residuals, export_path=export_path,
            page_num=page_num,
        )
        png_paths["actual_vs_predicted"]    = actual_vs_pred_path
        png_paths["residuals_vs_predicted"] = residuals_vs_pred_path

        png_paths["residual_distribution"], page_num = _build_residual_distribution_page(
            pp=pp, artifact_base=artifact_base,
            residuals=residuals, export_path=export_path,
            page_num=page_num,
        )

        if len(feature_names) > 0:
            page_num = _build_variable_distribution_pages(
                pp=pp, artifact_base=artifact_base,
                df_valid=df_valid, independent_vars=feature_names,
                page_num=page_num,
            )

        page_num = _build_final_summary_page(
            pp=pp, artifact_base=artifact_base,
            metrics=metrics, feature_names=feature_names,
            importance=importance, target=target,
            n_samples=len(y_train), hyperparams=hyperparams,
            page_num=page_num,
        )

    return metrics, png_paths, pdf_path