"""
sdm_xgb_print_handler.py
------------------------
PDF report generation for the Hybrid SDM + XGBoost model.

Page structure:
  1. Cover
  2. Metrics (SDM stage + Hybrid final)
  3. Coefficients (β vs θ table — SDM-specific)
  4. Impacts Decomposition (LeSage-Pace)
  5. Diagnostics (Actual vs Predicted · Residual distribution)
  6. Stage Decomposition (SDM prediction · XGB correction · Hybrid final)
  7. XGBoost Feature Importance (gain-based on SDM residuals)
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

# Colour palette — teal/orange hybrid identity
ACCENT   = "#0f766e"   # teal  — SDM stage
ACCENT2  = "#ea580c"   # orange — XGBoost stage
ACCENT3  = "#7c3aed"   # purple — hybrid final
BLUE     = "#2563eb"   # β own effect
ORANGE   = "#ea580c"   # θ spillover
DARK     = "#1f2937"
LIGHT    = "#f0fdfa"
BORDER   = "#99f6e4"
PURPLE   = "#7c3aed"


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
    fig.text(0.07, 0.028, f"Hybrid SDM+XGBoost Report | {artifact_base}", fontsize=8.5, color="#6b7280")
    fig.text(0.93, 0.028, page_label, fontsize=8.5, color="#6b7280", ha="right")


def _save_png(fig, export_path: str, filename: str) -> str:
    out = os.path.join(export_path, filename)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=180)
    return out


# ---------------------------------------------------------------------------
# Page 1 — Cover
# ---------------------------------------------------------------------------

def _build_cover(pp, artifact_base, target, features, n_samples, rho, page_num):
    fig = _new_page()
    fig.text(0.5, 0.74, "Hybrid Spatial Durbin Model", fontsize=28, fontweight="bold",
             color=ACCENT, ha="center")
    fig.text(0.5, 0.685, "+ XGBoost", fontsize=22, color=ACCENT2, ha="center")
    fig.text(0.5, 0.635, "Training Report", fontsize=16, color=DARK, ha="center")
    fig.lines.append(plt.Line2D([0.2, 0.8], [0.615, 0.615],
                                transform=fig.transFigure, color=BORDER, linewidth=1))

    # Formula box
    ax_f = fig.add_axes([0.12, 0.525, 0.76, 0.075])
    ax_f.axis("off")
    ax_f.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=LIGHT, edgecolor=BORDER,
                                  linewidth=1.2, transform=ax_f.transAxes))
    ax_f.text(0.5, 0.55,
              "y = ρWy + Xβ + WXθ + ε   →   ŷ_hybrid = ŷ_SDM + ε̂_XGB",
              fontsize=11, fontfamily="monospace", color=DARK,
              ha="center", va="center", transform=ax_f.transAxes)

    info_lines = [
        f"Model ID   : {artifact_base}",
        f"Target     : {target}",
        f"Predictors : {len(features)}  (+ {len(features)} WX spillover terms)",
        f"Samples    : {n_samples:,}",
        f"Spatial ρ  : {rho:.4f}",
        f"Stage 1    : Spatial Durbin Model (GM_Lag on [X|WX], Queen contiguity)",
        f"Stage 2    : XGBoost on SDM residuals (300 estimators)",
    ]
    y = 0.50
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

def _build_metrics_page(pp, export_path, artifact_base, metrics, rho,
                         moran_i_sdm, moran_p_sdm, moran_i_hybrid, moran_p_hybrid, page_num):
    fig = _new_page()
    _add_header(fig, "Model Metrics", "Stage 1 (SDM) · Stage 2 (XGBoost) · Combined Hybrid")

    ax = fig.add_axes([0.07, 0.12, 0.86, 0.78])
    ax.axis("off")

    sections = [
        ("Hybrid (Final)", [
            ("R² (holdout)",      f"{metrics.get('r2', 0):.4f}",     "Hybrid variance explained on test set"),
            ("RMSE",              f"{metrics.get('rmse', 0):.4f}",    "Hybrid root mean squared error"),
            ("MAE",               f"{metrics.get('mae', 0):.4f}",     "Hybrid mean absolute error"),
            ("Moran's I (hyb.)",  f"{moran_i_hybrid:.4f}" if moran_i_hybrid is not None else "N/A",
                                                                       "Residual spatial autocorrelation after hybrid"),
        ]),
        ("Stage 1 — SDM", [
            ("R² SDM (holdout)",  f"{metrics.get('r2_sdm', 0):.4f}", "SDM-only variance explained"),
            ("RMSE SDM",          f"{metrics.get('rmse_sdm', 0):.4f}", "SDM-only RMSE"),
            ("Pseudo R²",         f"{metrics.get('pseudo_r2', 0):.4f}", "Spreg in-sample pseudo R²"),
            ("ρ (Spatial Lag)",   f"{rho:.4f}",                        "Strength of spatial autocorrelation"),
            ("Moran's I (SDM)",   f"{moran_i_sdm:.4f}" if moran_i_sdm is not None else "N/A",
                                                                        "SDM residual spatial autocorrelation"),
            ("Moran's I p-value", f"{moran_p_sdm:.4f}" if moran_p_sdm is not None else "N/A",
                                                                        "Significance of SDM Moran's I"),
        ]),
    ]

    row_h = 0.062
    y     = 0.97
    col_x = [0.0, 0.35, 0.52, 0.92]

    for section_title, rows in sections:
        ax.text(0.0, y, section_title, fontsize=10, fontweight="bold",
                color=ACCENT, va="top", transform=ax.transAxes)
        y -= 0.045
        for i, (metric, value, desc) in enumerate(rows):
            bg = LIGHT if i % 2 == 0 else "white"
            ax.add_patch(plt.Rectangle((0, y - 0.008), 1, row_h,
                                       transform=ax.transAxes, facecolor=bg, edgecolor="none"))
            ax.text(col_x[0] + 0.01, y + 0.03, metric, fontsize=9.5, color=DARK,   va="top", transform=ax.transAxes)
            ax.text(col_x[1],        y + 0.03, value,  fontsize=9.5, color=BLUE,   va="top", transform=ax.transAxes)
            ax.text(col_x[2],        y + 0.03, desc,   fontsize=8.5, color="#6b7280", va="top", transform=ax.transAxes)
            y -= row_h
        y -= 0.03

    png_path = _save_png(fig, export_path, f"{artifact_base}_metrics.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 3 — Coefficients (β vs θ)
# ---------------------------------------------------------------------------

def _build_coefficients_page(pp, export_path, artifact_base, coeff_table, rho, page_num):
    fig = _new_page()
    _add_header(fig, "Coefficients",
                "β = own effect (X)  |  θ = spillover effect (WX)  |  ★ p < 0.05")

    ax = fig.add_axes([0.04, 0.10, 0.92, 0.79])
    ax.axis("off")

    col_labels = ["Variable", "β", "SE(β)", "p(β)", "θ", "SE(θ)", "p(θ)", "Spillover"]
    col_x      = [0.00, 0.22, 0.33, 0.43, 0.54, 0.65, 0.75, 0.86]
    header_y   = 0.97

    ax.add_patch(plt.Rectangle((0, header_y - 0.025), 1, 0.055,
                               transform=ax.transAxes, facecolor="#ccfbf1", edgecolor="none"))
    ax.text(0.225, header_y - 0.005, "── own ──",       fontsize=7, color=BLUE,   va="top", transform=ax.transAxes)
    ax.text(0.545, header_y - 0.005, "── spillover ──", fontsize=7, color=ORANGE, va="top", transform=ax.transAxes)

    for j, label in enumerate(col_labels):
        color = BLUE   if label in ("β", "SE(β)", "p(β)") else \
                ORANGE if label in ("θ", "SE(θ)", "p(θ)") else ACCENT
        ax.text(col_x[j] + 0.005, header_y, label, fontsize=9, fontweight="bold",
                color=color, va="top", transform=ax.transAxes)

    row_h = min(0.072, 0.82 / max(len(coeff_table) + 2, 1))
    spillover_colors = {"positive": "#16a34a", "negative": "#dc2626", "none": "#94a3b8"}

    for i, row in enumerate(coeff_table):
        y  = header_y - (i + 1.6) * row_h
        bg = LIGHT if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - 0.005), 1, row_h,
                                   transform=ax.transAxes, facecolor=bg, edgecolor="none"))

        b_sig  = "★" if row.get("beta_sig")  else ""
        th_sig = "★" if row.get("theta_sig") else ""
        sp     = row.get("spillover_type", "none")

        vals   = [
            row["variable"],
            f"{row['beta']:.4f}{b_sig}",
            f"{row['beta_se']:.4f}",
            f"{row['beta_p']:.4f}",
            f"{row['theta']:.4f}{th_sig}",
            f"{row['theta_se']:.4f}",
            f"{row['theta_p']:.4f}",
            sp.upper(),
        ]
        colors = [
            DARK,
            BLUE   if row.get("beta_sig")  else "#374151",
            "#6b7280", "#6b7280",
            ORANGE if row.get("theta_sig") else "#374151",
            "#6b7280", "#6b7280",
            spillover_colors.get(sp, "#94a3b8"),
        ]
        for j, (val, color) in enumerate(zip(vals, colors)):
            ax.text(col_x[j] + 0.005, y + row_h * 0.55, val,
                    fontsize=8.5, color=color, va="top", transform=ax.transAxes)

    # Rho row
    rho_y = header_y - (len(coeff_table) + 2.2) * row_h
    ax.add_patch(plt.Rectangle((0, rho_y - 0.005), 1, row_h,
                               transform=ax.transAxes, facecolor="#fff7e6", edgecolor="none"))
    ax.text(col_x[0] + 0.005, rho_y + row_h * 0.55, "W*y  (ρ)",
            fontsize=9, fontweight="bold", color="#b45309", va="top", transform=ax.transAxes)
    ax.text(col_x[1] + 0.005, rho_y + row_h * 0.55, f"{rho:.4f}",
            fontsize=9, color="#b45309", va="top", transform=ax.transAxes)
    ax.text(col_x[4] + 0.005, rho_y + row_h * 0.55, "spatial lag of y",
            fontsize=8, color="#b45309", va="top", transform=ax.transAxes)

    ax.text(0.0, -0.04,
            "★ = significant at p < 0.05   |   β = own predictor effect   |   "
            "θ = neighbor spillover effect   |   W*y (ρ) = spatial lag coefficient",
            fontsize=7.5, color="#6b7280", va="top", transform=ax.transAxes)

    png_path = _save_png(fig, export_path, f"{artifact_base}_coefficients.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 4 — Impacts Decomposition (LeSage-Pace)
# ---------------------------------------------------------------------------

def _build_impacts_page(pp, export_path, artifact_base, coeff_table, page_num):
    fig = _new_page()
    _add_header(fig, "Impacts Decomposition",
                "LeSage-Pace average direct, indirect (spillover), and total effects · S_k(W) = (I-ρW)⁻¹(β_k·I + θ_k·W)")

    ax = fig.add_axes([0.04, 0.52, 0.92, 0.36])
    ax.axis("off")

    col_labels = ["Variable", "Avg Direct", "Avg Indirect", "Avg Total", "Spillover Type"]
    col_x      = [0.00, 0.28, 0.46, 0.64, 0.82]
    header_y   = 0.97

    ax.add_patch(plt.Rectangle((0, header_y - 0.025), 1, 0.07,
                               transform=ax.transAxes, facecolor="#ccfbf1", edgecolor="none"))
    for j, label in enumerate(col_labels):
        ax.text(col_x[j] + 0.005, header_y, label, fontsize=9.5, fontweight="bold",
                color=ACCENT, va="top", transform=ax.transAxes)

    row_h      = min(0.10, 0.85 / max(len(coeff_table) + 1, 1))
    sp_colors  = {"positive": "#16a34a", "negative": "#dc2626", "none": "#94a3b8"}

    for i, row in enumerate(coeff_table):
        y  = header_y - (i + 1.5) * row_h
        bg = LIGHT if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - 0.01), 1, row_h,
                                   transform=ax.transAxes, facecolor=bg, edgecolor="none"))
        d   = row.get("avg_direct")
        ind = row.get("avg_indirect")
        t   = row.get("avg_total")
        sp  = row.get("spillover_type", "none")

        vals   = [
            row["variable"],
            f"{d:.4f}"   if d   is not None else "N/A",
            f"{ind:.4f}" if ind is not None else "N/A",
            f"{t:.4f}"   if t   is not None else "N/A",
            sp.upper(),
        ]
        colors = [DARK, BLUE, ORANGE, PURPLE, sp_colors.get(sp, "#94a3b8")]
        for j, (val, color) in enumerate(zip(vals, colors)):
            ax.text(col_x[j] + 0.005, y + row_h * 0.55, val,
                    fontsize=9, color=color, va="top", transform=ax.transAxes)

    # Bar chart — avg direct vs indirect vs total
    ax2 = fig.add_axes([0.07, 0.10, 0.86, 0.36])
    feats     = [r["variable"]              for r in coeff_table]
    directs   = [r.get("avg_direct")   or 0 for r in coeff_table]
    indirects = [r.get("avg_indirect") or 0 for r in coeff_table]
    totals    = [r.get("avg_total")    or 0 for r in coeff_table]

    x = np.arange(len(feats))
    w = 0.25
    ax2.bar(x - w, directs,   width=w, label="Avg Direct",   color=BLUE,   alpha=0.85)
    ax2.bar(x,     indirects, width=w, label="Avg Indirect", color=ORANGE, alpha=0.85)
    ax2.bar(x + w, totals,    width=w, label="Avg Total",    color=PURPLE, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(feats, rotation=30, ha="right", fontsize=9)
    ax2.axhline(0, color="#374151", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Effect magnitude", fontsize=9)
    ax2.set_title("Average Direct · Indirect · Total Impacts per Predictor", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8.5)
    ax2.grid(axis="y", alpha=0.3)

    png_path = _save_png(fig, export_path, f"{artifact_base}_impacts.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 5 — Diagnostics
# ---------------------------------------------------------------------------

def _build_diagnostics_page(pp, export_path, artifact_base,
                              y_full, pred_sdm, pred_hybrid, residuals_hybrid, page_num):
    fig = _new_page()
    _add_header(fig, "Diagnostics", "Actual vs Predicted (SDM vs Hybrid) · Residual Distribution")

    ax1 = fig.add_axes([0.07, 0.54, 0.86, 0.33])
    mn  = min(y_full.min(), pred_hybrid.min())
    mx  = max(y_full.max(), pred_hybrid.max())
    ax1.scatter(y_full, pred_sdm,    alpha=0.35, s=12, color=ACCENT,  label="SDM",    edgecolors="none")
    ax1.scatter(y_full, pred_hybrid, alpha=0.45, s=12, color=PURPLE,  label="Hybrid", edgecolors="none")
    ax1.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="Perfect fit")
    ax1.set_xlabel("Actual", fontsize=9)
    ax1.set_ylabel("Predicted", fontsize=9)
    ax1.set_title("Actual vs Predicted — SDM vs Hybrid", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    avp_path = _save_png(fig, export_path, f"{artifact_base}_actual_vs_predicted.png")

    ax2 = fig.add_axes([0.07, 0.14, 0.86, 0.30])
    ax2.hist(residuals_hybrid, bins=30, color=PURPLE, alpha=0.75, edgecolor="white")
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


# ---------------------------------------------------------------------------
# Page 6 — Stage Decomposition
# ---------------------------------------------------------------------------

def _build_stage_decomposition_page(pp, export_path, artifact_base,
                                     y_full, pred_sdm, pred_xgb_correction, pred_hybrid, page_num):
    fig = _new_page()
    _add_header(fig, "Stage Decomposition", "SDM prediction · XGBoost correction · Final hybrid")

    # Stage 1: SDM vs actual
    ax1 = fig.add_axes([0.07, 0.55, 0.40, 0.30])
    mn, mx = y_full.min(), y_full.max()
    ax1.scatter(y_full, pred_sdm, alpha=0.4, s=12, color=ACCENT, edgecolors="none")
    ax1.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    ax1.set_title("Stage 1: SDM", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Actual", fontsize=8)
    ax1.set_ylabel("ŷ_SDM", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Stage 2: XGBoost correction distribution
    ax2 = fig.add_axes([0.55, 0.55, 0.40, 0.30])
    ax2.hist(pred_xgb_correction, bins=30, color=ACCENT2, alpha=0.75, edgecolor="white")
    ax2.axvline(0, color="red", linewidth=1, linestyle="--")
    ax2.set_title("Stage 2: XGBoost Correction (ε̂_XGB)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Correction value", fontsize=8)
    ax2.set_ylabel("Frequency", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Final hybrid
    ax3 = fig.add_axes([0.07, 0.14, 0.86, 0.30])
    mn2, mx2 = min(y_full.min(), pred_hybrid.min()), max(y_full.max(), pred_hybrid.max())
    ax3.scatter(y_full, pred_hybrid, alpha=0.45, s=12, color=PURPLE, edgecolors="none")
    ax3.plot([mn2, mx2], [mn2, mx2], "r--", linewidth=1.2, label="Perfect fit")
    ax3.set_title("Stage 3: Hybrid (ŷ_SDM + ε̂_XGB)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Actual", fontsize=8)
    ax3.set_ylabel("ŷ_hybrid", fontsize=8)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    stage_path = _save_png(fig, export_path, f"{artifact_base}_stage_decomposition.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return stage_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 7 — XGBoost Feature Importance
# ---------------------------------------------------------------------------

def _build_xgb_importance_page(pp, export_path, artifact_base, xgb_importance, page_num):
    fig = _new_page()
    _add_header(fig, "XGBoost Feature Importance", "Stage 2 — gain-based importance on SDM residuals")

    sorted_imp = sorted(xgb_importance, key=lambda x: x["value"], reverse=True)
    feats  = [r["feature"] for r in sorted_imp]
    values = [r["value"]   for r in sorted_imp]
    colors = [ACCENT2 if v >= np.median(values) else "#fdba74" for v in values]

    ax = fig.add_axes([0.07, 0.12, 0.86, 0.72])
    bars = ax.barh(feats[::-1], values[::-1], color=colors[::-1], edgecolor="white", height=0.6)
    ax.set_xlabel("Feature Importance (XGBoost gain on SDM residuals)", fontsize=10)
    ax.set_title("XGBoost Stage 2 — Feature Importance", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    max_val = max(values) if values else 1
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5, color=DARK)

    ax.text(0.0, -0.06,
            "Note: XGBoost importances reflect contribution to ε_SDM (nonlinear residual), not to y directly.",
            fontsize=8, color="#6b7280", transform=ax.transAxes)

    png_path = _save_png(fig, export_path, f"{artifact_base}_xgb_importance.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 8 — Summary / Interpretation
# ---------------------------------------------------------------------------

def _build_summary_page(pp, artifact_base, metrics, indep, rho,
                         moran_i_sdm, moran_p_sdm, moran_i_hybrid, moran_p_hybrid,
                         n_samples, page_num):
    fig = _new_page()
    _add_header(fig, "Final Interpretation", "Hybrid SDM + XGBoost summary")

    # Hybrid box
    box1 = fig.add_axes([0.07, 0.60, 0.40, 0.27])
    box1.axis("off")
    box1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=PURPLE, linewidth=1.3))
    box1.text(0.04, 0.92, "Hybrid (Final)", fontsize=11, fontweight="bold", color=PURPLE)
    box1.text(0.04, 0.74,
              f"R²    : {metrics.get('r2', 0):.4f}\n"
              f"RMSE  : {metrics.get('rmse', 0):.2f}\n"
              f"MAE   : {metrics.get('mae', 0):.2f}",
              fontsize=10, color=DARK, va="top")

    # SDM box
    box2 = fig.add_axes([0.53, 0.60, 0.40, 0.27])
    box2.axis("off")
    box2.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=ACCENT, linewidth=1.2))
    box2.text(0.04, 0.92, "Stage 1 — SDM", fontsize=11, fontweight="bold", color=ACCENT)
    mi_sig = "significant" if (moran_p_sdm is not None and moran_p_sdm < 0.05) else "not significant"
    sdm_txt = (
        f"R²    : {metrics.get('r2_sdm', 0):.4f}\n"
        f"ρ     : {rho:.4f}\n"
    )
    if moran_i_sdm is not None:
        sdm_txt += f"Moran : {moran_i_sdm:.4f} ({mi_sig})"
    box2.text(0.04, 0.74, sdm_txt, fontsize=10, color=DARK, va="top")

    rho_str   = f"Strong" if abs(rho) > 0.5 else ("Moderate" if abs(rho) > 0.2 else "Weak")
    hyb_mi    = f"Hybrid Moran's I = {moran_i_hybrid:.4f}" if moran_i_hybrid is not None else ""

    notes = [
        f"Samples: {n_samples:,}  |  Predictors: {len(indep)} (+ {len(indep)} WX terms)  |  Weights: Queen contiguity",
        "",
        "The Hybrid SDM+XGBoost combines SDM's parametric spatial structure (ρWy + WXθ)",
        "with XGBoost's nonlinear correction on the SDM residuals.",
        f"ρ = {rho:.4f} indicates {rho_str.lower()} spatial dependence in the target variable.",
        hyb_mi,
        "SDM stage provides interpretable direct/indirect/total impacts via S_k(W).",
        "XGBoost importances reflect contribution to ε_SDM only — not direct effects on y.",
        "For policy/causal interpretation, always lead with SDM impacts decomposition.",
        "SHAP values (Stage 2) sum to ε̂_XGB per observation — the nonlinear correction only.",
    ]
    y = 0.54
    for note in [n for n in notes if n]:
        fig.text(0.07, y, note, fontsize=9.5, color="#374151")
        y -= 0.036

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def export_sdm_xgb_report_and_artifacts(
    export_path:         str,
    artifact_base:       str,
    indep:               List[str],
    target:              str,
    y_full:              np.ndarray,
    pred_sdm:            np.ndarray,
    pred_xgb_correction: np.ndarray,
    pred_hybrid:         np.ndarray,
    residuals_hybrid:    np.ndarray,
    metrics:             Dict[str, Any],
    coeff_table:         List[Dict],
    xgb_importance:      List[Dict],
    rho:                 float,
    moran_i_sdm:         Optional[float],
    moran_p_sdm:         Optional[float],
    moran_i_hybrid:      Optional[float],
    moran_p_hybrid:      Optional[float],
    df_valid:            pd.DataFrame,
) -> Tuple[Dict[str, str], str]:

    png_paths: Dict[str, str] = {}
    pdf_path = os.path.join(export_path, f"{artifact_base}.pdf")

    with PdfPages(pdf_path) as pp:
        page_num = 1

        page_num = _build_cover(
            pp, artifact_base, target, indep, len(df_valid), rho, page_num
        )

        png_paths["metrics"], page_num = _build_metrics_page(
            pp, export_path, artifact_base, metrics, rho,
            moran_i_sdm, moran_p_sdm, moran_i_hybrid, moran_p_hybrid, page_num
        )

        png_paths["coefficients"], page_num = _build_coefficients_page(
            pp, export_path, artifact_base, coeff_table, rho, page_num
        )

        png_paths["impacts"], page_num = _build_impacts_page(
            pp, export_path, artifact_base, coeff_table, page_num
        )

        avp, rd, page_num = _build_diagnostics_page(
            pp, export_path, artifact_base,
            y_full, pred_sdm, pred_hybrid, residuals_hybrid, page_num
        )
        png_paths["actual_vs_predicted"]   = avp
        png_paths["residual_distribution"] = rd

        png_paths["stage_decomposition"], page_num = _build_stage_decomposition_page(
            pp, export_path, artifact_base,
            y_full, pred_sdm, pred_xgb_correction, pred_hybrid, page_num
        )

        png_paths["xgb_importance"], page_num = _build_xgb_importance_page(
            pp, export_path, artifact_base, xgb_importance, page_num
        )

        _build_summary_page(
            pp, artifact_base, metrics, indep, rho,
            moran_i_sdm, moran_p_sdm, moran_i_hybrid, moran_p_hybrid,
            len(df_valid), page_num
        )

    print(f"✅ Hybrid SDM+XGB PDF report: {pdf_path}")
    return png_paths, pdf_path