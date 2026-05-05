"""
hybrid_slm_rf_print_handler.py
-------------------------------
PDF report generation for the Hybrid SLM + RF model.
"""
from typing import List, Optional, Tuple, Dict, Any
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ACCENT  = "#7c3aed"
ACCENT2 = "#2563eb"
ACCENT3 = "#16a34a"
DARK    = "#1f2937"
BORDER  = "#d0d7de"


def _new_page(figsize=(8.27, 11.69)):
    return plt.figure(figsize=figsize, facecolor="white")


def _add_header(fig, title, subtitle=None):
    fig.text(0.07, 0.965, title, fontsize=20, fontweight="bold", color=ACCENT, va="top")
    if subtitle:
        fig.text(0.07, 0.938, subtitle, fontsize=10.5, color="#5f6b7a", va="top")
    fig.lines.append(plt.Line2D([0.07, 0.93], [0.922, 0.922],
                                transform=fig.transFigure, color=BORDER, linewidth=1.2))


def _add_footer(fig, artifact_base, page_label):
    fig.lines.append(plt.Line2D([0.07, 0.93], [0.05, 0.05],
                                transform=fig.transFigure, color=BORDER, linewidth=0.8))
    fig.text(0.07, 0.028, f"Hybrid SLM+RF Report | {artifact_base}", fontsize=8.5, color="#6b7280")
    fig.text(0.93, 0.028, page_label, fontsize=8.5, color="#6b7280", ha="right")


def _save_png(fig, export_path, filename):
    out = os.path.join(export_path, filename)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=180)
    return out


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _build_cover(pp, artifact_base, target, features, n_samples, rho, page_num):
    fig = _new_page()
    fig.text(0.5, 0.74, "Hybrid Spatial Lag Model", fontsize=30, fontweight="bold",
             color=ACCENT, ha="center")
    fig.text(0.5, 0.675, "+ Random Forest", fontsize=22, color=ACCENT2, ha="center")
    fig.text(0.5, 0.625, "Training Report", fontsize=16, color=DARK, ha="center")
    fig.lines.append(plt.Line2D([0.2, 0.8], [0.605, 0.605],
                                transform=fig.transFigure, color=BORDER, linewidth=1))
    info_lines = [
        f"Model ID  : {artifact_base}",
        f"Target    : {target}",
        f"Predictors: {len(features)}",
        f"Samples   : {n_samples:,}",
        f"Spatial ρ : {rho:.4f}",
        f"Stage 1   : Spatial Lag Model (GM_Lag, Queen contiguity)",
        f"Stage 2   : Random Forest on SLM residuals (300 trees)",
    ]
    y = 0.59
    for line in info_lines:
        fig.text(0.5, y, line, fontsize=10.5, color=DARK, ha="center")
        y -= 0.038
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


def _build_metrics_page(pp, export_path, artifact_base, metrics, rho,
                         moran_i_slm, moran_p_slm, moran_i_hybrid, moran_p_hybrid, page_num):
    fig = _new_page()
    _add_header(fig, "Model Metrics", "Stage 1 (SLM) · Stage 2 (RF) · Combined Hybrid")

    ax = fig.add_axes([0.07, 0.12, 0.86, 0.78])
    ax.axis("off")

    sections = [
        ("Hybrid (Final)", [
            ("R² (holdout)",     f"{metrics.get('r2', 0):.4f}",    "Hybrid variance explained on test set"),
            ("RMSE",             f"{metrics.get('rmse', 0):.4f}",   "Hybrid root mean squared error"),
            ("MAE",              f"{metrics.get('mae', 0):.4f}",    "Hybrid mean absolute error"),
            ("Moran's I (hyb.)", f"{moran_i_hybrid:.4f}" if moran_i_hybrid is not None else "N/A",
                                                                    "Residual spatial autocorrelation after hybrid"),
        ]),
        ("Stage 1 — SLM", [
            ("R² SLM (holdout)", f"{metrics.get('r2_slm', 0):.4f}", "SLM-only variance explained"),
            ("RMSE SLM",         f"{metrics.get('rmse_slm', 0):.4f}", "SLM-only RMSE"),
            ("Pseudo R²",        f"{metrics.get('pseudo_r2', 0):.4f}", "Spreg in-sample pseudo R²"),
            ("ρ (Spatial Lag)",  f"{rho:.4f}",                       "Strength of spatial autocorrelation"),
            ("Moran's I (SLM)",  f"{moran_i_slm:.4f}" if moran_i_slm is not None else "N/A",
                                                                      "SLM residual spatial autocorrelation"),
        ]),
    ]

    row_h = 0.065
    y     = 0.97
    col_x = [0.0, 0.35, 0.52, 0.92]

    for section_title, rows in sections:
        ax.text(0.0, y, section_title, fontsize=10, fontweight="bold",
                color=ACCENT, va="top", transform=ax.transAxes)
        y -= 0.045
        for i, (metric, value, desc) in enumerate(rows):
            bg = "#f5f3ff" if i % 2 == 0 else "white"
            ax.add_patch(plt.Rectangle((0, y - 0.008), 1, row_h,
                                       transform=ax.transAxes, facecolor=bg, edgecolor="none"))
            ax.text(col_x[0] + 0.01, y + 0.03, metric, fontsize=9.5, color=DARK, va="top", transform=ax.transAxes)
            ax.text(col_x[1],        y + 0.03, value,  fontsize=9.5, color=ACCENT2, va="top", transform=ax.transAxes)
            ax.text(col_x[2],        y + 0.03, desc,   fontsize=8.5, color="#6b7280", va="top", transform=ax.transAxes)
            y -= row_h
        y -= 0.03

    png_path = _save_png(fig, export_path, f"{artifact_base}_metrics.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


def _build_diagnostics_page(pp, export_path, artifact_base, y_full, pred_slm, pred_hybrid, residuals_hybrid, page_num):
    fig = _new_page()
    _add_header(fig, "Diagnostics", "Actual vs Predicted (SLM vs Hybrid) · Residual Distribution")

    # Actual vs Predicted comparison
    ax1 = fig.add_axes([0.07, 0.54, 0.86, 0.33])
    mn = min(y_full.min(), pred_hybrid.min())
    mx = max(y_full.max(), pred_hybrid.max())
    ax1.scatter(y_full, pred_slm,     alpha=0.35, s=12, color=ACCENT2,  label="SLM", edgecolors="none")
    ax1.scatter(y_full, pred_hybrid,  alpha=0.45, s=12, color=ACCENT,   label="Hybrid", edgecolors="none")
    ax1.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="Perfect fit")
    ax1.set_xlabel("Actual", fontsize=9)
    ax1.set_ylabel("Predicted", fontsize=9)
    ax1.set_title("Actual vs Predicted — SLM vs Hybrid", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    avp_path = _save_png(fig, export_path, f"{artifact_base}_actual_vs_predicted.png")

    # Residual histogram (hybrid)
    ax2 = fig.add_axes([0.07, 0.14, 0.86, 0.30])
    ax2.hist(residuals_hybrid, bins=30, color=ACCENT, alpha=0.75, edgecolor="white")
    ax2.axvline(0, color="red", linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Residual (Hybrid)", fontsize=9)
    ax2.set_ylabel("Frequency", fontsize=9)
    ax2.set_title("Hybrid Residual Distribution", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    rd_path = _save_png(fig, export_path, f"{artifact_base}_residual_distribution.png")

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return avp_path, rd_path, page_num + 1


def _build_stage_comparison_page(pp, export_path, artifact_base, y_full, pred_slm, pred_rf_correction, pred_hybrid, page_num):
    fig = _new_page()
    _add_header(fig, "Stage Decomposition", "SLM prediction · RF correction · Final hybrid")

    # SLM predictions vs actual
    ax1 = fig.add_axes([0.07, 0.55, 0.40, 0.30])
    ax1.scatter(y_full, pred_slm, alpha=0.4, s=12, color=ACCENT2, edgecolors="none")
    mn, mx = y_full.min(), y_full.max()
    ax1.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    ax1.set_title("Stage 1: SLM", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Actual", fontsize=8)
    ax1.set_ylabel("ŷ_SLM", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # RF correction distribution
    ax2 = fig.add_axes([0.55, 0.55, 0.40, 0.30])
    ax2.hist(pred_rf_correction, bins=30, color=ACCENT3, alpha=0.75, edgecolor="white")
    ax2.axvline(0, color="red", linewidth=1, linestyle="--")
    ax2.set_title("Stage 2: RF Correction (ε̂_RF)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Correction value", fontsize=8)
    ax2.set_ylabel("Frequency", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Hybrid vs actual
    ax3 = fig.add_axes([0.07, 0.14, 0.86, 0.30])
    ax3.scatter(y_full, pred_hybrid, alpha=0.45, s=12, color=ACCENT, edgecolors="none")
    mn2, mx2 = min(y_full.min(), pred_hybrid.min()), max(y_full.max(), pred_hybrid.max())
    ax3.plot([mn2, mx2], [mn2, mx2], "r--", linewidth=1.2, label="Perfect fit")
    ax3.set_title("Stage 3: Hybrid (ŷ_SLM + ε̂_RF)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Actual", fontsize=8)
    ax3.set_ylabel("ŷ_hybrid", fontsize=8)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    stage_path = _save_png(fig, export_path, f"{artifact_base}_stage_decomposition.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return stage_path, page_num + 1


def _build_importance_page(pp, export_path, artifact_base, importance, page_num):
    fig = _new_page()
    _add_header(fig, "Feature Importance", "RF Stage 2 — importance on SLM residuals")

    sorted_imp = sorted(importance, key=lambda x: x["value"], reverse=True)
    feats  = [r["feature"] for r in sorted_imp]
    values = [r["value"]   for r in sorted_imp]
    colors = [ACCENT if v >= np.median(values) else "#94a3b8" for v in values]

    ax = fig.add_axes([0.07, 0.12, 0.86, 0.72])
    bars = ax.barh(feats[::-1], values[::-1], color=colors[::-1], edgecolor="white", height=0.6)
    ax.set_xlabel("|Feature Importance| (RF on residuals)", fontsize=10)
    ax.set_title("RF Stage 2 — Feature Importance", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5, color=DARK)

    png_path = _save_png(fig, export_path, f"{artifact_base}_feature_importance.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


def _build_summary_page(pp, artifact_base, metrics, indep, rho,
                         moran_i_slm, moran_p_slm, moran_i_hybrid, moran_p_hybrid, n_samples, page_num):
    fig = _new_page()
    _add_header(fig, "Final Interpretation", "Hybrid SLM + RF summary")

    # Hybrid box
    box1 = fig.add_axes([0.07, 0.60, 0.40, 0.26])
    box1.axis("off")
    box1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=ACCENT, linewidth=1.3))
    box1.text(0.04, 0.92, "Hybrid (Final)", fontsize=11, fontweight="bold", color=ACCENT)
    box1.text(0.04, 0.74,
              f"R²    : {metrics.get('r2', 0):.4f}\n"
              f"RMSE  : {metrics.get('rmse', 0):.2f}\n"
              f"MAE   : {metrics.get('mae', 0):.2f}",
              fontsize=10, color=DARK, va="top")

    # SLM box
    box2 = fig.add_axes([0.53, 0.60, 0.40, 0.26])
    box2.axis("off")
    box2.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=ACCENT2, linewidth=1.2))
    box2.text(0.04, 0.92, "Stage 1 — SLM", fontsize=11, fontweight="bold", color=ACCENT2)
    mi_sig = "significant" if (moran_p_slm is not None and moran_p_slm < 0.05) else "not significant"
    box2.text(0.04, 0.74,
              f"R²    : {metrics.get('r2_slm', 0):.4f}\n"
              f"ρ     : {rho:.4f}\n"
              f"Moran : {moran_i_slm:.4f} ({mi_sig})" if moran_i_slm is not None else
              f"R²    : {metrics.get('r2_slm', 0):.4f}\nρ: {rho:.4f}",
              fontsize=10, color=DARK, va="top")

    notes = [
        f"Samples: {n_samples:,}  |  Predictors: {len(indep)}  |  Weights: Queen contiguity",
        "",
        "The Hybrid model combines SLM's spatial structure with RF's nonlinear correction.",
        f"ρ = {rho:.4f} indicates {'strong' if abs(rho) > 0.5 else 'moderate' if abs(rho) > 0.2 else 'weak'} spatial dependence in the target variable.",
        f"Hybrid Moran's I = {moran_i_hybrid:.4f}" if moran_i_hybrid is not None else "",
        "RF feature importances reflect contribution to the nonlinear residual, not direct effects on y.",
        "For causal interpretation, use SLM coefficients (ρ, β) only.",
    ]
    y = 0.54
    for note in [n for n in notes if n]:
        fig.text(0.07, y, note, fontsize=9.5, color="#374151")
        y -= 0.038

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def export_hybrid_report_and_artifacts(
    export_path, artifact_base, indep, target,
    y_full, pred_slm, pred_rf_correction, pred_hybrid, residuals_hybrid,
    metrics, slm_coefficients, importance,
    rho, moran_i_slm, moran_p_slm, moran_i_hybrid, moran_p_hybrid,
    df_valid,
):
    png_paths = {}
    pdf_path  = os.path.join(export_path, f"{artifact_base}.pdf")

    with PdfPages(pdf_path) as pp:
        page_num = 1

        page_num = _build_cover(
            pp, artifact_base, target, indep, len(df_valid), rho, page_num
        )

        png_paths["metrics"], page_num = _build_metrics_page(
            pp, export_path, artifact_base, metrics, rho,
            moran_i_slm, moran_p_slm, moran_i_hybrid, moran_p_hybrid, page_num
        )

        avp, rd, page_num = _build_diagnostics_page(
            pp, export_path, artifact_base, y_full, pred_slm, pred_hybrid, residuals_hybrid, page_num
        )
        png_paths["actual_vs_predicted"]   = avp
        png_paths["residual_distribution"] = rd

        png_paths["stage_decomposition"], page_num = _build_stage_comparison_page(
            pp, export_path, artifact_base, y_full, pred_slm, pred_rf_correction, pred_hybrid, page_num
        )

        png_paths["feature_importance"], page_num = _build_importance_page(
            pp, export_path, artifact_base, importance, page_num
        )

        _build_summary_page(
            pp, artifact_base, metrics, indep, rho,
            moran_i_slm, moran_p_slm, moran_i_hybrid, moran_p_hybrid, len(df_valid), page_num
        )

    print(f"✅ Hybrid PDF report: {pdf_path}")
    return png_paths, pdf_path