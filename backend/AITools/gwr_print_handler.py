"""
gwr_print_handler.py
--------------------
PDF report generation for the Geographically Weighted Regression (GWR) training module.

Page structure:
  1. Cover
  2. Metrics (global R², RMSE, AICc, bandwidth, Moran's I)
  3. Local β Surface Summary (min/Q1/mean/Q3/max per predictor)
  4. Local β IQR Box Plot (spatial variation per predictor)
  5. Feature Importance (mean |β| ranked)
  6. Diagnostics (Actual vs Predicted · Residual Distribution)
  7. Local R² Distribution
  8. Summary / Interpretation
"""

from typing import List, Optional, Tuple, Dict, Any
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

ACCENT  = "#7c3aed"   # purple — GWR identity
ACCENT2 = "#0f766e"   # teal secondary
DARK    = "#1f2937"
LIGHT   = "#f5f3ff"
BORDER  = "#ddd6fe"
BLUE    = "#2563eb"
ORANGE  = "#ea580c"
GREEN   = "#16a34a"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _new_page(figsize=(8.27, 11.69)):
    return plt.figure(figsize=figsize, facecolor="white")


def _add_header(fig, title: str, subtitle: Optional[str] = None):
    fig.text(0.07, 0.965, title, fontsize=20, fontweight="bold", color=ACCENT, va="top")
    if subtitle:
        fig.text(0.07, 0.938, subtitle, fontsize=10.5, color="#5f6b7a", va="top")
    fig.lines.append(plt.Line2D([0.07, 0.93], [0.922, 0.922],
                                transform=fig.transFigure, color=BORDER, linewidth=1.2))


def _add_footer(fig, artifact_base: str, page_label: str):
    fig.lines.append(plt.Line2D([0.07, 0.93], [0.05, 0.05],
                                transform=fig.transFigure, color=BORDER, linewidth=0.8))
    fig.text(0.07, 0.028, f"GWR Report | {artifact_base}", fontsize=8.5, color="#6b7280")
    fig.text(0.93, 0.028, page_label, fontsize=8.5, color="#6b7280", ha="right")


def _save_png(fig, export_path: str, filename: str) -> str:
    out = os.path.join(export_path, filename)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=180)
    return out


# ---------------------------------------------------------------------------
# Page 1 — Cover
# ---------------------------------------------------------------------------

def _build_cover(pp, artifact_base, target, features, n_samples, bw, page_num):
    fig = _new_page()

    fig.text(0.5, 0.74, "Geographically Weighted", fontsize=26, fontweight="bold",
             color=ACCENT, ha="center")
    fig.text(0.5, 0.695, "Regression (GWR)", fontsize=26, fontweight="bold",
             color=ACCENT, ha="center")
    fig.text(0.5, 0.645, "Training Report", fontsize=16, color=DARK, ha="center")
    fig.lines.append(plt.Line2D([0.2, 0.8], [0.625, 0.625],
                                transform=fig.transFigure, color=BORDER, linewidth=1))

    # Formula box
    ax_f = fig.add_axes([0.10, 0.535, 0.80, 0.075])
    ax_f.axis("off")
    ax_f.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=LIGHT, edgecolor=BORDER,
                                  linewidth=1.2, transform=ax_f.transAxes))
    ax_f.text(0.5, 0.55,
              "y_i = β_0(u_i,v_i) + Σ β_k(u_i,v_i) · x_ik + ε_i",
              fontsize=11, fontfamily="monospace", color=DARK,
              ha="center", va="center", transform=ax_f.transAxes)

    info_lines = [
        f"Model ID   : {artifact_base}",
        f"Target     : {target}",
        f"Predictors : {len(features)}",
        f"Samples    : {n_samples:,}",
        f"Bandwidth  : {int(bw)} nearest neighbors (adaptive bi-square)",
        f"Selection  : AICc minimization",
        f"Output     : {n_samples:,} local coefficient sets (one per spatial unit)",
    ]
    y = 0.505
    for line in info_lines:
        fig.text(0.5, y, line, fontsize=10.5, color=DARK, ha="center")
        y -= 0.038

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


# ---------------------------------------------------------------------------
# Page 2 — Metrics
# ---------------------------------------------------------------------------

def _build_metrics_page(pp, export_path, artifact_base, metrics, moran_i, moran_p, page_num):
    fig = _new_page()
    _add_header(fig, "Model Metrics", "Global performance · GWR diagnostics · Spatial autocorrelation")

    ax = fig.add_axes([0.07, 0.45, 0.86, 0.44])
    ax.axis("off")

    rows = [
        ("R² (holdout)",         f"{metrics.get('r2', 0):.4f}",       "Variance explained on 20% test set"),
        ("RMSE",                 f"{metrics.get('rmse', 0):.4f}",      "Root mean squared error"),
        ("MAE",                  f"{metrics.get('mae', 0):.4f}",       "Mean absolute error"),
        ("Mean local R²",        f"{metrics.get('mean_r2', 0):.4f}",   "Average local goodness-of-fit across all units"),
        ("AICc",                 f"{metrics.get('aicc', 0):.2f}",      "Corrected AIC — bandwidth selection criterion"),
        ("Effective df (tr S)",  f"{metrics.get('eff_df', 0):.2f}",    "Trace of hat matrix — effective degrees of freedom"),
        ("Bandwidth (k-NN)",     f"{int(metrics.get('bandwidth', 0))}", "Adaptive nearest-neighbor bandwidth selected by AICc"),
        ("Moran's I (residuals)",f"{moran_i:.4f}" if moran_i is not None else "N/A",
                                                                        "Spatial autocorrelation in GWR residuals"),
        ("Moran's I p-value",    f"{moran_p:.4f}" if moran_p is not None else "N/A",
                                                                        "Significance — non-significant = good fit"),
    ]

    col_x  = [0.0, 0.35, 0.52]
    header_y = 0.97
    ax.add_patch(plt.Rectangle((0, header_y - 0.02), 1, 0.055,
                               transform=ax.transAxes, facecolor="#ede9fe", edgecolor="none"))
    for j, label in enumerate(["Metric", "Value", "Description"]):
        ax.text(col_x[j] + 0.01, header_y, label, fontsize=10, fontweight="bold",
                color=ACCENT, va="top", transform=ax.transAxes)

    row_h = 0.088
    for i, (metric, value, desc) in enumerate(rows):
        y  = header_y - (i + 1.2) * row_h
        bg = LIGHT if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - 0.005), 1, row_h,
                                   transform=ax.transAxes, facecolor=bg, edgecolor="none"))
        ax.text(col_x[0] + 0.01, y + 0.04, metric, fontsize=9.5, color=DARK,   va="top", transform=ax.transAxes)
        ax.text(col_x[1],        y + 0.04, value,  fontsize=9.5, color=ACCENT, va="top", transform=ax.transAxes)
        ax.text(col_x[2],        y + 0.04, desc,   fontsize=8.5, color="#6b7280", va="top", transform=ax.transAxes)

    png_path = _save_png(fig, export_path, f"{artifact_base}_metrics.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 3 — Local β Surface Summary Table
# ---------------------------------------------------------------------------

def _build_local_beta_summary_page(pp, export_path, artifact_base, coeff_summary, page_num):
    fig = _new_page()
    _add_header(fig, "Local β Surface Summary",
                "Per-predictor statistics across all spatial units · % significant = |t| ≥ 1.96")

    ax = fig.add_axes([0.04, 0.10, 0.92, 0.79])
    ax.axis("off")

    col_labels = ["Variable", "Min", "Q1", "Mean", "Median", "Q3", "Max", "IQR", "% Sig."]
    col_x      = [0.00, 0.18, 0.28, 0.38, 0.49, 0.59, 0.69, 0.79, 0.89]
    header_y   = 0.97

    ax.add_patch(plt.Rectangle((0, header_y - 0.022), 1, 0.052,
                               transform=ax.transAxes, facecolor="#ede9fe", edgecolor="none"))
    for j, label in enumerate(col_labels):
        ax.text(col_x[j] + 0.005, header_y, label, fontsize=9, fontweight="bold",
                color=ACCENT, va="top", transform=ax.transAxes)

    row_h = min(0.075, 0.84 / max(len(coeff_summary) + 1, 1))

    for i, row in enumerate(coeff_summary):
        y  = header_y - (i + 1.5) * row_h
        bg = LIGHT if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - 0.005), 1, row_h,
                                   transform=ax.transAxes, facecolor=bg, edgecolor="none"))

        pct_sig = row.get("pct_sig", 0)
        sig_color = GREEN if pct_sig >= 50 else (ORANGE if pct_sig >= 25 else "#94a3b8")

        vals   = [
            row["variable"],
            f"{row['beta_min']:.4f}",
            f"{row['beta_q1']:.4f}",
            f"{row['beta_mean']:.4f}",
            f"{row['beta_median']:.4f}",
            f"{row['beta_q3']:.4f}",
            f"{row['beta_max']:.4f}",
            f"{row['beta_iqr']:.4f}",
            f"{pct_sig:.1f}%",
        ]
        colors = [DARK, BLUE, BLUE, ACCENT, ACCENT, ORANGE, ORANGE, "#374151", sig_color]

        for j, (val, color) in enumerate(zip(vals, colors)):
            ax.text(col_x[j] + 0.005, y + row_h * 0.55, val,
                    fontsize=8.5, color=color, va="top", transform=ax.transAxes)

    ax.text(0.0, -0.04,
            "IQR = Q3 − Q1 (spread of local effects)  ·  % Sig. = share of units where |t| ≥ 1.96  ·  "
            "High IQR = strong spatial non-stationarity",
            fontsize=7.5, color="#6b7280", va="top", transform=ax.transAxes)

    png_path = _save_png(fig, export_path, f"{artifact_base}_local_beta_summary.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 4 — Local β IQR Box Plot (spatial variation)
# ---------------------------------------------------------------------------

def _build_beta_boxplot_page(pp, export_path, artifact_base, coeff_summary, page_num):
    fig = _new_page()
    _add_header(fig, "Local β Spatial Variation",
                "Box plots of local coefficient distributions — wide IQR = spatial non-stationarity")

    ax = fig.add_axes([0.12, 0.12, 0.82, 0.72])

    feats      = [r["variable"] for r in coeff_summary]
    box_data   = [r["local_betas"] for r in coeff_summary]
    n_feats    = len(feats)

    bp = ax.boxplot(
        box_data,
        vert=True,
        patch_artist=True,
        labels=feats,
        medianprops=dict(color="#dc2626", linewidth=2),
        whiskerprops=dict(color="#374151", linewidth=1.2),
        capprops=dict(color="#374151", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="#94a3b8", markersize=3, alpha=0.4),
        widths=0.5,
    )

    colors = [ACCENT, ACCENT2, BLUE, ORANGE, GREEN, "#0891b2", "#be185d", "#d97706"]
    for patch, color in zip(bp["boxes"], colors * ((n_feats // len(colors)) + 1)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color="#dc2626", linewidth=1, linestyle="--", alpha=0.7, label="β = 0")
    ax.set_xlabel("Predictor", fontsize=10)
    ax.set_ylabel("Local β value", fontsize=10)
    ax.set_title("Distribution of Local Coefficients Across Spatial Units", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    ax.text(0.01, -0.14,
            "Each box = distribution of β_k across all n spatial units.  "
            "Median line = central tendency.  Box width = IQR.  "
            "Dots beyond whiskers = outlier locations.",
            fontsize=8, color="#6b7280", transform=ax.transAxes)

    png_path = _save_png(fig, export_path, f"{artifact_base}_beta_boxplot.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 5 — Feature Importance (mean |β|)
# ---------------------------------------------------------------------------

def _build_feature_importance_page(pp, export_path, artifact_base, coeff_summary, page_num):
    fig = _new_page()
    _add_header(fig, "Feature Importance", "Ranked by mean |β| across all spatial units")

    sorted_cs = sorted(coeff_summary, key=lambda x: abs(x["beta_mean"]), reverse=True)
    feats     = [r["variable"]              for r in sorted_cs]
    means     = [abs(r["beta_mean"])        for r in sorted_cs]
    pct_sigs  = [r.get("pct_sig", 0)       for r in sorted_cs]
    colors    = [GREEN if p >= 50 else (ORANGE if p >= 25 else "#94a3b8") for p in pct_sigs]

    ax = fig.add_axes([0.07, 0.12, 0.86, 0.72])
    bars = ax.barh(feats[::-1], means[::-1], color=colors[::-1], edgecolor="white", height=0.6)
    ax.set_xlabel("Mean |β| across spatial units", fontsize=10)
    ax.set_title("Feature Importance — Mean Absolute Local Coefficient", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    max_val = max(means) if means else 1
    for bar, val, pct in zip(bars, means[::-1], pct_sigs[::-1]):
        ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}  ({pct:.0f}% sig.)", va="center", fontsize=8.5, color=DARK)

    ax.legend(handles=[
        Patch(color=GREEN,      label="≥ 50% of units significant"),
        Patch(color=ORANGE,     label="25–49% significant"),
        Patch(color="#94a3b8",  label="< 25% significant"),
    ], fontsize=8.5, loc="lower right")

    png_path = _save_png(fig, export_path, f"{artifact_base}_feature_importance.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 6 — Diagnostics (Actual vs Predicted + Residuals)
# ---------------------------------------------------------------------------

def _build_diagnostics_page(pp, export_path, artifact_base,
                              y_full, preds_full, residuals, page_num):
    fig = _new_page()
    _add_header(fig, "Diagnostics", "Actual vs Predicted · Residual Distribution")

    # Actual vs Predicted
    ax1 = fig.add_axes([0.07, 0.54, 0.86, 0.32])
    mn, mx = min(y_full.min(), preds_full.min()), max(y_full.max(), preds_full.max())
    ax1.scatter(y_full, preds_full, alpha=0.4, s=12, color=ACCENT, edgecolors="none")
    ax1.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="Perfect fit")
    ax1.set_xlabel("Actual", fontsize=9)
    ax1.set_ylabel("Predicted (GWR)", fontsize=9)
    ax1.set_title("Actual vs Predicted", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    avp_path = _save_png(fig, export_path, f"{artifact_base}_actual_vs_predicted.png")

    # Residual Distribution
    ax2 = fig.add_axes([0.07, 0.14, 0.86, 0.30])
    ax2.hist(residuals, bins=30, color=ACCENT, alpha=0.72, edgecolor="white")
    ax2.axvline(0, color="red", linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Residual", fontsize=9)
    ax2.set_ylabel("Frequency", fontsize=9)
    ax2.set_title("Residual Distribution", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    rd_path = _save_png(fig, export_path, f"{artifact_base}_residual_distribution.png")

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return avp_path, rd_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 7 — Local R² Distribution
# ---------------------------------------------------------------------------

def _build_local_r2_page(pp, export_path, artifact_base, local_r2, page_num):
    fig = _new_page()
    _add_header(fig, "Local R² Distribution",
                "Goodness-of-fit per spatial unit — low R² areas signal missing local predictors")

    ax1 = fig.add_axes([0.07, 0.54, 0.86, 0.32])
    ax1.hist(local_r2, bins=30, color=ACCENT2, alpha=0.75, edgecolor="white")
    ax1.axvline(float(np.mean(local_r2)), color="#dc2626", linewidth=1.5,
                linestyle="--", label=f"Mean = {np.mean(local_r2):.4f}")
    ax1.axvline(float(np.median(local_r2)), color=ORANGE, linewidth=1.5,
                linestyle=":", label=f"Median = {np.median(local_r2):.4f}")
    ax1.set_xlabel("Local R²", fontsize=9)
    ax1.set_ylabel("Frequency", fontsize=9)
    ax1.set_title("Distribution of Local R² Across Spatial Units", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    lr2_path = _save_png(fig, export_path, f"{artifact_base}_local_r2.png")

    # Summary stats table
    ax2 = fig.add_axes([0.07, 0.14, 0.50, 0.30])
    ax2.axis("off")
    stats_rows = [
        ("Min local R²",    f"{local_r2.min():.4f}"),
        ("Q1 local R²",     f"{np.percentile(local_r2, 25):.4f}"),
        ("Mean local R²",   f"{local_r2.mean():.4f}"),
        ("Median local R²", f"{np.median(local_r2):.4f}"),
        ("Q3 local R²",     f"{np.percentile(local_r2, 75):.4f}"),
        ("Max local R²",    f"{local_r2.max():.4f}"),
        ("% units R²>0.5",  f"{np.mean(local_r2 > 0.5)*100:.1f}%"),
        ("% units R²>0.7",  f"{np.mean(local_r2 > 0.7)*100:.1f}%"),
    ]
    row_h = 0.11
    y     = 0.96
    for i, (label, val) in enumerate(stats_rows):
        bg = LIGHT if i % 2 == 0 else "white"
        ax2.add_patch(plt.Rectangle((0, y - 0.02), 1, row_h,
                                    transform=ax2.transAxes, facecolor=bg, edgecolor="none"))
        ax2.text(0.02, y + 0.03, label, fontsize=9.5, color=DARK,   va="top", transform=ax2.transAxes)
        ax2.text(0.72, y + 0.03, val,   fontsize=9.5, color=ACCENT, va="top", transform=ax2.transAxes)
        y -= row_h

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return lr2_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 8 — Summary / Interpretation
# ---------------------------------------------------------------------------

def _build_summary_page(pp, artifact_base, metrics, indep, moran_i, moran_p, n_samples, bw, page_num):
    fig = _new_page()
    _add_header(fig, "Final Interpretation", "Geographically Weighted Regression summary")

    # Box 1 — Global metrics
    box1 = fig.add_axes([0.07, 0.60, 0.40, 0.27])
    box1.axis("off")
    box1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=ACCENT, linewidth=1.3))
    box1.text(0.04, 0.92, "Global Performance", fontsize=11, fontweight="bold", color=ACCENT)
    box1.text(0.04, 0.74,
              f"R² (test)   : {metrics.get('r2', 0):.4f}\n"
              f"RMSE        : {metrics.get('rmse', 0):.2f}\n"
              f"Mean local R²: {metrics.get('mean_r2', 0):.4f}\n"
              f"AICc        : {metrics.get('aicc', 0):.2f}\n"
              f"Samples     : {n_samples:,}  |  Predictors: {len(indep)}",
              fontsize=9.5, color=DARK, va="top")

    # Box 2 — GWR diagnostics
    box2 = fig.add_axes([0.53, 0.60, 0.40, 0.27])
    box2.axis("off")
    box2.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=ACCENT2, linewidth=1.2))
    box2.text(0.04, 0.92, "GWR Diagnostics", fontsize=11, fontweight="bold", color=ACCENT2)
    mi_sig = "not significant ✓" if (moran_p is not None and moran_p >= 0.05) else "significant ★"
    box2.text(0.04, 0.74,
              f"Kernel     : Adaptive bi-square\n"
              f"Bandwidth  : {int(bw)} nearest neighbors\n"
              f"Eff. df    : {metrics.get('eff_df', 0):.2f}\n"
              f"Moran's I  : {moran_i:.4f} ({mi_sig})" if moran_i is not None else
              f"Kernel     : Adaptive bi-square\n"
              f"Bandwidth  : {int(bw)} nearest neighbors\n"
              f"Eff. df    : {metrics.get('eff_df', 0):.2f}",
              fontsize=9.5, color=DARK, va="top")

    notes = [
        "GWR fits a separate weighted regression at every spatial unit using kernel-weighted",
        "observations — producing n sets of local coefficients β_k(u_i, v_i).",
        "Coefficients vary continuously across space; map them to identify spatial heterogeneity.",
        "Always map local β_k alongside local t-values — grey out cells where |t| < 1.96.",
        "High IQR in the β surface = strong spatial non-stationarity for that predictor.",
        "Low local R² areas signal missing predictors or model mis-specification at those locations.",
        "Bandwidth controls locality: smaller k-NN = more local (high variance); larger = smoother.",
        "Run Monte Carlo non-stationarity test to confirm GWR is justified over global OLS.",
        "Compare GWR AICc vs OLS AICc — prefer GWR only if ΔAICc > 3.",
        f"Moran's I on residuals is {mi_sig} — "
        f"{'spatial structure is well captured.' if 'not' in mi_sig else 'residual autocorrelation remains.'}",
    ]
    y = 0.54
    for note in notes:
        fig.text(0.07, y, note, fontsize=9, color="#374151")
        y -= 0.034

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def export_gwr_report_and_artifacts(
    export_path:   str,
    artifact_base: str,
    indep:         List[str],
    target:        str,
    y_full:        np.ndarray,
    preds_full:    np.ndarray,
    residuals:     np.ndarray,
    local_r2:      np.ndarray,
    coeff_summary: List[Dict],
    metrics:       Dict[str, Any],
    moran_i:       Optional[float],
    moran_p:       Optional[float],
    df_valid:      pd.DataFrame,
    coords:        np.ndarray,
) -> Tuple[Dict[str, str], str]:

    png_paths: Dict[str, str] = {}
    pdf_path = os.path.join(export_path, f"{artifact_base}.pdf")
    bw       = metrics.get("bandwidth", 0)

    with PdfPages(pdf_path) as pp:
        page_num = 1

        page_num = _build_cover(
            pp, artifact_base, target, indep, len(df_valid), bw, page_num
        )

        png_paths["metrics"], page_num = _build_metrics_page(
            pp, export_path, artifact_base, metrics, moran_i, moran_p, page_num
        )

        png_paths["local_beta_summary"], page_num = _build_local_beta_summary_page(
            pp, export_path, artifact_base, coeff_summary, page_num
        )

        png_paths["beta_boxplot"], page_num = _build_beta_boxplot_page(
            pp, export_path, artifact_base, coeff_summary, page_num
        )

        png_paths["feature_importance"], page_num = _build_feature_importance_page(
            pp, export_path, artifact_base, coeff_summary, page_num
        )

        avp, rd, page_num = _build_diagnostics_page(
            pp, export_path, artifact_base, y_full, preds_full, residuals, page_num
        )
        png_paths["actual_vs_predicted"]   = avp
        png_paths["residual_distribution"] = rd

        png_paths["local_r2"], page_num = _build_local_r2_page(
            pp, export_path, artifact_base, local_r2, page_num
        )

        _build_summary_page(
            pp, artifact_base, metrics, indep, moran_i, moran_p, len(df_valid), bw, page_num
        )

    print(f"✅ GWR PDF report: {pdf_path}")
    return png_paths, pdf_path