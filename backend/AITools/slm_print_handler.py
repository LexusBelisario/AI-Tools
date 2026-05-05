"""
slm_print_handler.py
--------------------
PDF report generation for the Spatial Lag Model training module.
Mirrors the structure of lr_print_handler.py.
"""

from typing import List, Optional, Tuple, Dict, Any
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPORT_ACCENT = "#2563eb"
REPORT_DARK   = "#1f2937"
REPORT_LIGHT  = "#f8fafc"
REPORT_BORDER = "#d0d7de"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_page(figsize=(8.27, 11.69)):
    return plt.figure(figsize=figsize, facecolor="white")


def _add_header(fig, title: str, subtitle: Optional[str] = None):
    fig.text(0.07, 0.965, title, fontsize=20, fontweight="bold", color=REPORT_ACCENT, va="top")
    if subtitle:
        fig.text(0.07, 0.938, subtitle, fontsize=10.5, color="#5f6b7a", va="top")
    fig.lines.append(plt.Line2D([0.07, 0.93], [0.922, 0.922],
                                transform=fig.transFigure, color=REPORT_BORDER, linewidth=1.2))


def _add_footer(fig, artifact_base: str, page_label: str):
    fig.lines.append(plt.Line2D([0.07, 0.93], [0.05, 0.05],
                                transform=fig.transFigure, color=REPORT_BORDER, linewidth=0.8))
    fig.text(0.07, 0.028, f"SLM Report | {artifact_base}", fontsize=8.5, color="#6b7280")
    fig.text(0.93, 0.028, page_label, fontsize=8.5, color="#6b7280", ha="right")


def _save_png(fig, export_path: str, filename: str) -> str:
    out = os.path.join(export_path, filename)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=180)
    return out


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _build_cover_page(pp, artifact_base, target, features, n_samples, rho, page_num):
    fig = _new_page()
    # Title block
    fig.text(0.5, 0.72, "Spatial Lag Model", fontsize=32, fontweight="bold",
             color=REPORT_ACCENT, ha="center", va="center")
    fig.text(0.5, 0.655, "Training Report", fontsize=18, color=REPORT_DARK, ha="center")
    fig.lines.append(plt.Line2D([0.2, 0.8], [0.635, 0.635],
                                transform=fig.transFigure, color=REPORT_BORDER, linewidth=1))

    info_lines = [
        f"Model ID  : {artifact_base}",
        f"Target    : {target}",
        f"Predictors: {len(features)}",
        f"Samples   : {n_samples:,}",
        f"Spatial ρ : {rho:.4f}",
        f"Weights   : Queen Contiguity (row-standardized)",
    ]
    y = 0.59
    for line in info_lines:
        fig.text(0.5, y, line, fontsize=11, color=REPORT_DARK, ha="center")
        y -= 0.04

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


def _build_metrics_page(pp, export_path, artifact_base, metrics, rho, moran_i, moran_p, page_num):
    fig = _new_page()
    _add_header(fig, "Model Metrics", "Performance and spatial diagnostics")

    ax = fig.add_axes([0.07, 0.55, 0.86, 0.35])
    ax.axis("off")

    rows = [
        ("R² (holdout)",      f"{metrics.get('r2', 0):.4f}",    "Variance explained on test set"),
        ("Pseudo R²",         f"{metrics.get('pseudo_r2', 0):.4f}", "Spreg in-sample pseudo R²"),
        ("RMSE",              f"{metrics.get('rmse', 0):.4f}",   "Root mean squared error"),
        ("MAE",               f"{metrics.get('mae', 0):.4f}",    "Mean absolute error"),
        ("MSE",               f"{metrics.get('mse', 0):.4f}",    "Mean squared error"),
        ("ρ (Spatial Lag)",   f"{rho:.4f}",                       "Strength of spatial autocorrelation"),
        ("Moran's I (resid)", f"{moran_i:.4f}" if moran_i is not None else "N/A",
                                                                  "Residual spatial autocorrelation"),
        ("Moran's I p-value", f"{moran_p:.4f}" if moran_p is not None else "N/A",
                                                                  "Significance of Moran's I"),
    ]

    col_labels = ["Metric", "Value", "Description"]
    col_widths = [0.25, 0.15, 0.60]
    header_y   = 0.92

    for j, (label, width) in enumerate(zip(col_labels, col_widths)):
        ax.text(sum(col_widths[:j]) + 0.01, header_y, label,
                fontsize=10, fontweight="bold", color=REPORT_ACCENT, va="top",
                transform=ax.transAxes)

    row_h = 0.09
    for i, (metric, value, desc) in enumerate(rows):
        y = header_y - (i + 1) * row_h
        bg = "#f0f6ff" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - 0.01), 1, row_h,
                                   transform=ax.transAxes, facecolor=bg, edgecolor="none"))
        ax.text(0.01, y + 0.03, metric, fontsize=10, color=REPORT_DARK, va="top", transform=ax.transAxes)
        ax.text(0.26, y + 0.03, value,  fontsize=10, color="#2563eb",   va="top", transform=ax.transAxes)
        ax.text(0.41, y + 0.03, desc,   fontsize=9,  color="#6b7280",   va="top", transform=ax.transAxes)

    png_path = _save_png(fig, export_path, f"{artifact_base}_metrics.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


def _build_coefficients_page(pp, export_path, artifact_base, coeff_table, rho, page_num):
    fig = _new_page()
    _add_header(fig, "Coefficients", "Spatial lag model coefficient estimates (GM_Lag)")

    ax = fig.add_axes([0.07, 0.12, 0.86, 0.78])
    ax.axis("off")

    col_labels  = ["Variable", "Coef", "Std Err", "z-stat", "p-value", "Sig."]
    col_x       = [0.0, 0.30, 0.46, 0.58, 0.72, 0.87]
    header_y    = 0.97

    for j, label in enumerate(col_labels):
        ax.text(col_x[j], header_y, label, fontsize=9.5, fontweight="bold",
                color=REPORT_ACCENT, va="top", transform=ax.transAxes)

    row_h = min(0.065, 0.80 / max(len(coeff_table) + 1, 1))
    for i, row in enumerate(coeff_table):
        y   = header_y - (i + 1) * row_h
        bg  = "#f0f6ff" if i % 2 == 0 else "white"
        sig = "★" if row.get("significant") else ""
        ax.add_patch(plt.Rectangle((0, y - 0.005), 1, row_h,
                                   transform=ax.transAxes, facecolor=bg, edgecolor="none"))
        vals = [row["variable"], f"{row['coef']:.4f}", f"{row['std_err']:.4f}",
                f"{row['z']:.4f}", f"{row['p']:.4f}", sig]
        for j, val in enumerate(vals):
            color = "#16a34a" if j == 5 and sig else REPORT_DARK
            ax.text(col_x[j], y + row_h * 0.6, val, fontsize=9, color=color,
                    va="top", transform=ax.transAxes)

    # Rho row
    rho_y = header_y - (len(coeff_table) + 1) * row_h
    ax.add_patch(plt.Rectangle((0, rho_y - 0.005), 1, row_h,
                               transform=ax.transAxes, facecolor="#fff7e6", edgecolor="none"))
    ax.text(col_x[0], rho_y + row_h * 0.6, "W*y  (ρ)", fontsize=9, fontweight="bold",
            color="#b45309", va="top", transform=ax.transAxes)
    ax.text(col_x[1], rho_y + row_h * 0.6, f"{rho:.4f}", fontsize=9,
            color="#b45309", va="top", transform=ax.transAxes)

    ax.text(0.0, -0.04, "★ = significant at p < 0.05   |   W*y (ρ) = spatial lag coefficient",
            fontsize=8.5, color="#6b7280", va="top", transform=ax.transAxes)

    png_path = _save_png(fig, export_path, f"{artifact_base}_coefficients.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


def _build_diagnostics_page(pp, export_path, artifact_base, y_full, preds_full, residuals, page_num):
    fig = _new_page()
    _add_header(fig, "Diagnostics", "Actual vs Predicted · Residuals")

    # Actual vs Predicted
    ax1 = fig.add_axes([0.07, 0.54, 0.40, 0.33])
    ax1.scatter(y_full, preds_full, alpha=0.45, s=18, color="#2563eb", edgecolors="none")
    mn, mx = min(y_full.min(), preds_full.min()), max(y_full.max(), preds_full.max())
    ax1.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="Perfect fit")
    ax1.set_xlabel("Actual", fontsize=9)
    ax1.set_ylabel("Predicted", fontsize=9)
    ax1.set_title("Actual vs Predicted", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    avp_path = _save_png(fig, export_path, f"{artifact_base}_actual_vs_predicted.png")

    # Residuals vs Predicted
    ax2 = fig.add_axes([0.55, 0.54, 0.40, 0.33])
    ax2.scatter(preds_full, residuals, alpha=0.45, s=18, color="#7c3aed", edgecolors="none")
    ax2.axhline(0, color="red", linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Predicted", fontsize=9)
    ax2.set_ylabel("Residual", fontsize=9)
    ax2.set_title("Residuals vs Predicted", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    rvp_path = _save_png(fig, export_path, f"{artifact_base}_residuals_vs_predicted.png")

    # Residual histogram
    ax3 = fig.add_axes([0.07, 0.14, 0.86, 0.30])
    ax3.hist(residuals, bins=30, color="#2563eb", alpha=0.75, edgecolor="white")
    ax3.axvline(0, color="red", linewidth=1.2, linestyle="--")
    ax3.set_xlabel("Residual", fontsize=9)
    ax3.set_ylabel("Frequency", fontsize=9)
    ax3.set_title("Residual Distribution", fontsize=10, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    rd_path = _save_png(fig, export_path, f"{artifact_base}_residual_distribution.png")

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return avp_path, rvp_path, rd_path, page_num + 1


def _build_feature_importance_page(pp, export_path, artifact_base, coeff_table, page_num):
    fig = _new_page()
    _add_header(fig, "Feature Importance", "Absolute coefficient magnitudes")

    sorted_ct = sorted(coeff_table, key=lambda x: abs(x["coef"]), reverse=True)
    feats  = [r["variable"] for r in sorted_ct]
    values = [abs(r["coef"]) for r in sorted_ct]
    colors = ["#16a34a" if r["significant"] else "#94a3b8" for r in sorted_ct]

    ax = fig.add_axes([0.07, 0.12, 0.86, 0.72])
    bars = ax.barh(feats[::-1], values[::-1], color=colors[::-1], edgecolor="white", height=0.6)
    ax.set_xlabel("|Coefficient|", fontsize=10)
    ax.set_title("Feature Importance (|β|)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5, color=REPORT_DARK)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#16a34a", label="p < 0.05"),
                        Patch(color="#94a3b8", label="p ≥ 0.05")],
              fontsize=8.5, loc="lower right")

    png_path = _save_png(fig, export_path, f"{artifact_base}_feature_importance.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


def _build_summary_page(pp, artifact_base, metrics, indep, rho, moran_i, moran_p, n_samples, page_num):
    fig = _new_page()
    _add_header(fig, "Final Interpretation", "Spatial Lag Model summary")

    box1 = fig.add_axes([0.07, 0.58, 0.40, 0.28])
    box1.axis("off")
    box1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_ACCENT, linewidth=1.3))
    box1.text(0.04, 0.90, "Model Summary", fontsize=12, fontweight="bold", color=REPORT_ACCENT)
    summary = (
        f"Samples   : {n_samples:,}\n"
        f"Predictors: {len(indep)}\n"
        f"R² (test) : {metrics.get('r2', 0):.4f}\n"
        f"Pseudo R² : {metrics.get('pseudo_r2', 0):.4f}\n"
        f"RMSE      : {metrics.get('rmse', 0):.2f}\n"
        f"MAE       : {metrics.get('mae', 0):.2f}"
    )
    box1.text(0.04, 0.72, summary, fontsize=10, color=REPORT_DARK, va="top")

    box2 = fig.add_axes([0.53, 0.58, 0.40, 0.28])
    box2.axis("off")
    box2.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    box2.text(0.04, 0.90, "Spatial Diagnostics", fontsize=12, fontweight="bold", color=REPORT_ACCENT)
    rho_interp  = "Strong" if abs(rho) > 0.5 else ("Moderate" if abs(rho) > 0.2 else "Weak")
    mi_sig      = "significant" if (moran_p is not None and moran_p < 0.05) else "not significant"
    spatial_txt = (
        f"ρ (rho)   : {rho:.4f}  [{rho_interp} spatial lag]\n"
        f"Moran's I : {moran_i:.4f}" if moran_i is not None else f"ρ (rho)   : {rho:.4f}"
    )
    if moran_i is not None:
        spatial_txt += f"\n  → Residuals are {mi_sig} (p={moran_p:.4f})"
    box2.text(0.04, 0.72, spatial_txt, fontsize=10, color=REPORT_DARK, va="top")

    notes = [
        "The Spatial Lag Model accounts for spatial dependence by including a spatially lagged",
        "dependent variable (W*y) as a predictor. ρ close to 1 indicates strong spatial clustering.",
        f"Moran's I on residuals being {mi_sig} suggests the model {'adequately' if mi_sig == 'not significant' else 'may not fully'} captures spatial structure.",
        "Queen contiguity weights were used (row-standardized). Validate results against domain knowledge.",
    ]
    y = 0.46
    for note in notes:
        fig.text(0.07, y, note, fontsize=9.5, color="#374151")
        y -= 0.038

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def export_slm_report_and_artifacts(
    export_path: str,
    artifact_base: str,
    model,
    indep: List[str],
    target: str,
    y_full: np.ndarray,
    preds_full: np.ndarray,
    residuals: np.ndarray,
    metrics: Dict[str, Any],
    coeff_table: List[Dict],
    rho: float,
    moran_i: Optional[float],
    moran_p: Optional[float],
    df_valid: pd.DataFrame,
) -> Tuple[Dict[str, str], str]:

    png_paths: Dict[str, str] = {}
    pdf_path = os.path.join(export_path, f"{artifact_base}.pdf")

    with PdfPages(pdf_path) as pp:
        page_num = 1

        page_num = _build_cover_page(
            pp, artifact_base, target, indep, len(df_valid), rho, page_num
        )

        png_paths["metrics"], page_num = _build_metrics_page(
            pp, export_path, artifact_base, metrics, rho, moran_i, moran_p, page_num
        )

        png_paths["coefficients"], page_num = _build_coefficients_page(
            pp, export_path, artifact_base, coeff_table, rho, page_num
        )

        avp, rvp, rd, page_num = _build_diagnostics_page(
            pp, export_path, artifact_base, y_full, preds_full, residuals, page_num
        )
        png_paths["actual_vs_predicted"]    = avp
        png_paths["residuals_vs_predicted"] = rvp
        png_paths["residual_distribution"]  = rd

        png_paths["feature_importance"], page_num = _build_feature_importance_page(
            pp, export_path, artifact_base, coeff_table, page_num
        )

        _build_summary_page(
            pp, artifact_base, metrics, indep, rho, moran_i, moran_p, len(df_valid), page_num
        )

    print(f"✅ SLM PDF report: {pdf_path}")
    return png_paths, pdf_path