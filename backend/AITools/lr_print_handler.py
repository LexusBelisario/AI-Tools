"""
lr_print_handler.py
-------------------
PDF report generation handler for the Linear Regression training module.
Contains all page-builder functions, drawing helpers, and the main
export_full_report_and_artifacts() orchestrator.
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
# Report color constants
# ---------------------------------------------------------------------------

REPORT_ACCENT = "#1e88e5"
REPORT_DARK   = "#1f2937"
REPORT_LIGHT  = "#f8fafc"
REPORT_BORDER = "#d0d7de"


# ---------------------------------------------------------------------------
# Low-level drawing helpers
# ---------------------------------------------------------------------------

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
            linewidth=1.2,
        )
    )


def _add_footer(fig, artifact_base: str, page_label: str):
    fig.lines.append(
        plt.Line2D(
            [0.07, 0.93],
            [0.05, 0.05],
            transform=fig.transFigure,
            color=REPORT_BORDER,
            linewidth=0.8,
        )
    )
    fig.text(0.07, 0.028, f"Model Report | {artifact_base}", fontsize=8.5, color="#6b7280")
    fig.text(0.93, 0.028, page_label, fontsize=8.5, color="#6b7280", ha="right")


def _save_png(fig, export_path: str, filename: str) -> str:
    out = os.path.join(export_path, filename)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=200)
    return out



# ---------------------------------------------------------------------------
# Physical layout constants
# ---------------------------------------------------------------------------
_FIG_W_IN   = 8.27    # A4 width in inches
_CHAR_W_10  = 0.069   # avg char width in inches at fontsize 10pt (DejaVu Sans)
_LINE_H_10  = 0.158   # line height in inches at fontsize 10pt (≈ 1.2 × cap height)


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
    """Characters per line for a given axes width fraction and fontsize."""
    usable_in = _FIG_W_IN * axes_width * (1.0 - x_pad * 2)
    return max(20, int(usable_in / (_CHAR_W_10 * fontsize / 10.0)))


def _line_h_frac(axes_height_in: float, fontsize: float) -> float:
    """Line height as a fraction of axes height."""
    return (_LINE_H_10 * fontsize / 10.0) / axes_height_in


def _text_box(
    fig,
    left: float,
    bottom: float,
    width: float,
    text: str,
    title: Optional[str] = None,
    fontsize: float = 10.5,
    title_fontsize: float = 11.0,
    x_pad: float = 0.03,
    pad_top_in: float = 0.12,
    pad_bot_in: float = 0.12,
    title_gap_in: float = 0.22,
    facecolor: str = "white",
    edgecolor: str = REPORT_BORDER,
    title_color: str = REPORT_ACCENT,
    text_color: str = REPORT_DARK,
) -> float:
    """
    Draw a text box that auto-sizes its height to fit content.
    Returns the bottom y of the box (same as `bottom` — useful for stacking).
    """
    chars = _cpl(width, x_pad, fontsize)
    lines = _wrap_text(text, chars)

    line_h_in   = _LINE_H_10 * fontsize / 10.0
    content_h   = len(lines) * line_h_in
    title_h     = title_gap_in if title else 0.0
    total_h_in  = pad_top_in + title_h + content_h + pad_bot_in
    total_h_frac = total_h_in / (_FIG_W_IN * (11.69 / 8.27))   # convert to figure height fraction

    ax = fig.add_axes([left, bottom, width, total_h_frac])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1,
        fill=True, facecolor=facecolor,
        edgecolor=edgecolor, linewidth=1.2
    ))

    # y positions in axes coords (0=bottom, 1=top)
    pad_top_ax  = pad_top_in  / total_h_in
    pad_bot_ax  = pad_bot_in  / total_h_in   # noqa: F841
    title_h_ax  = title_h     / total_h_in
    line_h_ax   = line_h_in   / total_h_in

    y = 1.0 - pad_top_ax
    if title:
        ax.text(x_pad, y, title,
                fontsize=title_fontsize, fontweight="bold",
                color=title_color, va="top", clip_on=True)
        y -= title_h_ax

    for line in lines:
        ax.text(x_pad, y, line,
                fontsize=fontsize, color=text_color,
                va="top", clip_on=True)
        y -= line_h_ax

    return bottom   # caller can use bottom + total_h_frac to stack


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
            x, y, feat,
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


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _build_cover_page(
    pp: PdfPages,
    artifact_base: str,
    target: str,
    features: List[str],
    n_samples: int,
    page_num: int,
) -> int:
    fig = _new_page()
    fig.text(0.07, 0.88, "Linear Regression Model Report", fontsize=24, fontweight="bold", color=REPORT_ACCENT)

    definition = (
        "Linear Regression is a method that estimates how the target value changes based on the "
        "input variables. In simple terms, it finds the best-fitting straight-line relationship "
        "between the features and the value you want to predict. Each feature is given a weight "
        "called a coefficient, and those coefficients are used to compute predictions."
    )
    _text_box(fig, 0.07, 0.63, 0.86, definition,
              title="What is Linear Regression?",
              fontsize=9.5, title_fontsize=11,
              facecolor="#f0f7ff", pad_top_in=0.12, pad_bot_in=0.12, title_gap_in=0.22)

    meta_ax = fig.add_axes([0.07, 0.33, 0.86, 0.26])
    meta_ax.axis("off")
    meta_ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2)
    )
    meta_ax.add_patch(
        plt.Rectangle((0, 0.85), 1, 0.15, facecolor="#f4f9ff", edgecolor=REPORT_BORDER, linewidth=1.2)
    )
    meta_ax.text(0.03, 0.925, "Model Information", fontsize=12, fontweight="bold", color=REPORT_ACCENT, va="center")

    meta_lines = [
        ("Model Type",       "Linear Regression"),
        ("Model Name",       artifact_base),
        ("Target Variable",  target),
        ("Feature Count",    str(len(features))),
        ("Training Samples", f"{n_samples:,}"),
        ("Generated At",     datetime.now().strftime("%Y-%b-%d %I:%M:%S %p")),
    ]

    y = 0.74
    for label, value in meta_lines:
        meta_ax.text(0.03, y, label, fontsize=11, fontweight="bold", color=REPORT_DARK, va="center")
        meta_ax.text(0.30, y, value, fontsize=11, color=REPORT_DARK, va="center")
        y -= 0.11

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
        reverse=True,
    )
    top_feature = sorted_pairs[0][0] if sorted_pairs else None
    top_value   = float(sorted_pairs[0][1]) if sorted_pairs else None

    summary_lines = _metrics_interpretation_text(
        r2=metrics["r2"],
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        top_feature=top_feature,
        top_value=top_value,
    )

    left_ax = fig.add_axes([0.07, 0.64, 0.40, 0.24])
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

    right_ax = fig.add_axes([0.53, 0.64, 0.40, 0.24])
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

    rec_text = (
        "Use the metrics page to evaluate overall fit, the feature pages to review standardized "
        "effects and coefficient significance, and the diagnostics page to inspect bias and "
        "error behavior. Variable distribution pages provide context for predictor spread."
    )
    _text_box(fig, 0.07, 0.38, 0.86, rec_text,
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

    ax = fig.add_axes([0.10, 0.63, 0.80, 0.16])
    ax.axis("off")

    table = ax.table(
        cellText=[
            ["Metric", "Value", "Interpretation"],
            ["R²",   f"{metrics['r2']:.4f}",   "Explained variance of the model"],
            ["RMSE", f"{metrics['rmse']:.2f}",  "Penalizes larger prediction errors"],
            ["MAE",  f"{metrics['mae']:.2f}",   "Average absolute error"],
            ["MSE",  f"{metrics['mse']:.2f}",   "Mean squared error"],
        ],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.6)
    _style_table(table, header_fontsize=10, body_fontsize=9)

    notes_text = (
        f"This model achieved R² = {metrics['r2']:.4f}. RMSE and MAE should be interpreted "
        f"relative to the scale of the target variable. Lower values generally indicate better fit, "
        f"but diagnostic plots are still needed to assess whether the model behaves well across the data range."
    )
    _text_box(fig, 0.10, 0.07, 0.80, notes_text,
              title="Interpretation Notes",
              fontsize=10.3, title_fontsize=12,
              facecolor="white", pad_top_in=0.12, pad_bot_in=0.12, title_gap_in=0.24)

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    out = _save_png(fig, export_path, "metrics_table.png")
    plt.close(fig)
    return out, page_num + 1


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
        reverse=True,
    )
    feat_names = [x[0] for x in sorted_pairs]
    feat_vals  = [float(x[1]) for x in sorted_pairs]

    preview_fig = _new_page()
    _add_page_header(preview_fig, "Feature Analysis", "Coefficient importance and significance tests")
    preview_ax = preview_fig.add_axes([0.18, 0.56, 0.72, 0.28])
    _build_feature_bar_chart(preview_ax, feat_names, feat_vals)
    preview_png_path = _save_png(preview_fig, export_path, "feature_importance.png")
    plt.close(preview_fig)

    row_chunks = _chunk_list(coef_ttests, 12) if coef_ttests else [[]]

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
                    "Yes" if row["significant"] else "No",
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
                0.5, 0.5,
                "Coefficient t-test details are not available.",
                ha="center", va="center",
                fontsize=11, color=REPORT_DARK,
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

    ttest_text = (
        f"T-statistic: {residual_ttest['t_stat']:.4f}     "
        f"P-value: {residual_ttest['p_value']:.4f}     "
        f"{conclusion}"
    )
    _text_box(fig, 0.10, 0.07, 0.80, ttest_text,
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
    X_train_unscaled: pd.DataFrame,
    independent_vars: List[str],
    page_num: int,
) -> int:
    plots_per_page = 2
    axes_positions = [
        [0.10, 0.54, 0.80, 0.28],
        [0.10, 0.14, 0.80, 0.28],
    ]

    for start_idx in range(0, len(independent_vars), plots_per_page):
        cols = independent_vars[start_idx:start_idx + plots_per_page]
        fig = _new_page()
        _add_page_header(fig, "Variable Distributions", "Predictor spread and basic descriptive statistics")

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

                mean_val   = float(col_data.mean())
                median_val = float(col_data.median())
                std_val    = float(col_data.std())

                stats_text = (
                    f"Mean: {mean_val:.2f}\n"
                    f"Median: {median_val:.2f}\n"
                    f"Std: {std_val:.2f}"
                )
                ax.text(
                    0.98, 0.95, stats_text,
                    transform=ax.transAxes,
                    va="top", ha="right",
                    fontsize=9,
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
    line_gap  = 0.022   # height per line in figure-fraction units
    title_h   = 0.038   # space for the "Documentation Notes" title
    pad_top   = 0.018
    pad_bot   = 0.018
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
        reverse=True,
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

    _draw_doc_notes_box(fig, notes, bottom=0.07)

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

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

    from sklearn.metrics import r2_score

    mse  = float(np.mean((y_test - preds) ** 2))
    mae  = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_test, preds))

    std_X = np.std(X_train_scaled, axis=0)
    std_y = np.std(y_train)
    with np.errstate(divide="ignore", invalid="ignore"):
        importance = np.where(std_y == 0, 0, model.coef_ * std_X / std_y)

    t_stat, p_val = stats.ttest_1samp(residuals, 0)
    residual_ttest = {"t_stat": float(t_stat), "p_value": float(p_val)}

    n   = len(y_train)
    k   = len(independent_vars)
    dof = n - k - 1

    if dof <= 0:
        print(f"⚠️ Warning: dof={dof} (n={n}, k={k}). Dataset too small for reliable t-statistics.")
    residual_std_error = np.sqrt(np.sum(residuals ** 2) / dof) if dof > 0 else float("nan")

    try:
        XtX_inv   = np.linalg.inv(X_train_scaled.T @ X_train_scaled)
        var_coef  = residual_std_error ** 2 * np.diag(XtX_inv)
        std_errors = np.sqrt(var_coef)

        t_stats  = model.coef_ / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

        intercept_var     = residual_std_error ** 2 * (
            1 / n + np.mean(X_train_scaled, axis=0) @ XtX_inv @ np.mean(X_train_scaled, axis=0).T
        )
        intercept_std_err = np.sqrt(intercept_var)
        intercept_t       = model.intercept_ / intercept_std_err
        intercept_p       = 2 * (1 - stats.t.cdf(np.abs(intercept_t), dof))

        coef_ttests = [{
            "variable":    "Intercept",
            "coef":        float(model.intercept_),
            "std_err":     float(intercept_std_err),
            "t":           float(intercept_t),
            "p":           float(intercept_p),
            "significant": bool(intercept_p < 0.05),
        }]
        for i, var in enumerate(independent_vars):
            coef_ttests.append({
                "variable":    var,
                "coef":        float(model.coef_[i]),
                "std_err":     float(std_errors[i]),
                "t":           float(t_stats[i]),
                "p":           float(p_values[i]),
                "significant": bool(p_values[i] < 0.05),
            })
    except Exception as e:
        print(f"Could not calculate coefficient t-tests: {e}")
        coef_ttests = None

    metrics: Dict[str, float] = {
        "r2":   float(r2),
        "mse":  float(mse),
        "mae":  float(mae),
        "rmse": float(rmse),
    }

    png_paths: Dict[str, str] = {}

    pdf_path = os.path.join(export_path, f"{artifact_base}.pdf")
    with PdfPages(pdf_path) as pp:
        page_num = 1

        page_num = _build_cover_page(
            pp=pp, artifact_base=artifact_base,
            target=target, features=independent_vars,
            n_samples=len(y_train), page_num=page_num,
        )

        page_num = _build_executive_summary_page(
            pp=pp, artifact_base=artifact_base,
            metrics=metrics, independent_vars=independent_vars,
            importance=importance, residual_ttest=residual_ttest,
            page_num=page_num,
        )

        png_paths["metrics"], page_num = _build_metrics_table_page(
            pp=pp, artifact_base=artifact_base,
            metrics=metrics, export_path=export_path,
            page_num=page_num,
        )

        png_paths["feature_importance"], page_num = _build_feature_importance_pages(
            pp=pp, artifact_base=artifact_base,
            independent_vars=independent_vars, importance=importance,
            coef_ttests=coef_ttests, export_path=export_path,
            page_num=page_num,
        )

        actual_vs_pred_path, residuals_vs_pred_path, page_num = _build_diagnostics_page(
            pp=pp, artifact_base=artifact_base,
            y_test=y_test, preds=preds,
            residuals=residuals, export_path=export_path,
            page_num=page_num,
        )
        png_paths["actual_vs_predicted"]    = actual_vs_pred_path
        png_paths["residuals_vs_predicted"] = residuals_vs_pred_path

        png_paths["residual_distribution"], page_num = _build_residual_distribution_page(
            pp=pp, artifact_base=artifact_base,
            residuals=residuals, residual_ttest=residual_ttest,
            export_path=export_path, page_num=page_num,
        )

        if X_train_unscaled is not None and len(independent_vars) > 0:
            page_num = _build_variable_distribution_pages(
                pp=pp, artifact_base=artifact_base,
                X_train_unscaled=X_train_unscaled,
                independent_vars=independent_vars,
                page_num=page_num,
            )

        page_num = _build_final_summary_page(
            pp=pp, artifact_base=artifact_base,
            metrics=metrics, independent_vars=independent_vars,
            importance=importance, target=target,
            n_samples=len(y_train), residual_ttest=residual_ttest,
            page_num=page_num,
        )

    t_tests = {"residuals": residual_ttest, "coefficients": coef_ttests}
    return metrics, png_paths, t_tests, pdf_path