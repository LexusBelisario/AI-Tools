"""
sdm_print_handler.py
--------------------
PDF report generation for the Spatial Durbin Model training module.
Extends slm_print_handler with:
  - β vs θ side-by-side coefficient table
  - LeSage-Pace impacts decomposition table (avg direct, indirect, total)
  - Spillover type visualization per predictor
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

REPORT_ACCENT  = "#0f766e"   # teal — distinct from SLM blue
REPORT_DARK    = "#1f2937"
REPORT_LIGHT   = "#f0fdfa"
REPORT_BORDER  = "#99f6e4"
REPORT_ORANGE  = "#ea580c"   # WX / theta highlight (matches reference doc)
REPORT_BLUE    = "#2563eb"   # beta highlight
REPORT_PURPLE  = "#7c3aed"


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
    fig.text(0.07, 0.028, f"SDM Report | {artifact_base}", fontsize=8.5, color="#6b7280")
    fig.text(0.93, 0.028, page_label, fontsize=8.5, color="#6b7280", ha="right")


def _save_png(fig, export_path: str, filename: str) -> str:
    out = os.path.join(export_path, filename)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=180)
    return out


# ---------------------------------------------------------------------------
# Page 1 — Cover
# ---------------------------------------------------------------------------

def _build_cover_page(pp, artifact_base, target, features, n_samples, rho, page_num):
    fig = _new_page()

    fig.text(0.5, 0.74, "Spatial Durbin Model", fontsize=30, fontweight="bold",
             color=REPORT_ACCENT, ha="center", va="center")
    fig.text(0.5, 0.675, "Training Report", fontsize=18, color=REPORT_DARK, ha="center")
    fig.lines.append(plt.Line2D([0.2, 0.8], [0.655, 0.655],
                                transform=fig.transFigure, color=REPORT_BORDER, linewidth=1))

    # Formula box
    ax_f = fig.add_axes([0.18, 0.56, 0.64, 0.07])
    ax_f.axis("off")
    ax_f.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#f0fdfa", edgecolor=REPORT_BORDER,
                                  linewidth=1.2, transform=ax_f.transAxes))
    ax_f.text(0.5, 0.55, "y = ρWy + Xβ + WXθ + ε",
              fontsize=13, fontfamily="monospace", color=REPORT_DARK,
              ha="center", va="center", transform=ax_f.transAxes)

    info_lines = [
        f"Model ID   : {artifact_base}",
        f"Target     : {target}",
        f"Predictors : {len(features)}  (+ {len(features)} WX spillover terms)",
        f"Samples    : {n_samples:,}",
        f"Spatial ρ  : {rho:.4f}",
        f"Weights    : Queen Contiguity (row-standardized)",
        f"Estimator  : GM_Lag on augmented [X | WX]",
    ]
    y = 0.52
    for line in info_lines:
        fig.text(0.5, y, line, fontsize=10.5, color=REPORT_DARK, ha="center")
        y -= 0.038

    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return page_num + 1


# ---------------------------------------------------------------------------
# Page 2 — Metrics
# ---------------------------------------------------------------------------

def _build_metrics_page(pp, export_path, artifact_base, metrics, rho, moran_i, moran_p, page_num):
    fig = _new_page()
    _add_header(fig, "Model Metrics", "Performance and spatial diagnostics")

    ax = fig.add_axes([0.07, 0.50, 0.86, 0.40])
    ax.axis("off")

    rows = [
        ("R² (holdout)",       f"{metrics.get('r2', 0):.4f}",       "Variance explained on test set"),
        ("Pseudo R²",          f"{metrics.get('pseudo_r2', 0):.4f}", "Spreg in-sample pseudo R²"),
        ("RMSE",               f"{metrics.get('rmse', 0):.4f}",      "Root mean squared error"),
        ("MAE",                f"{metrics.get('mae', 0):.4f}",       "Mean absolute error"),
        ("MSE",                f"{metrics.get('mse', 0):.4f}",       "Mean squared error"),
        ("ρ (Spatial Lag)",    f"{rho:.4f}",                          "Strength of spatial dependence in y"),
        ("Moran's I (resid)",  f"{moran_i:.4f}" if moran_i is not None else "N/A",
                                                                      "Residual spatial autocorrelation"),
        ("Moran's I p-value",  f"{moran_p:.4f}" if moran_p is not None else "N/A",
                                                                      "Significance of Moran's I"),
    ]

    col_labels = ["Metric", "Value", "Description"]
    col_widths = [0.28, 0.14, 0.58]
    header_y   = 0.94

    for j, label in enumerate(col_labels):
        ax.text(sum(col_widths[:j]) + 0.01, header_y, label,
                fontsize=10, fontweight="bold", color=REPORT_ACCENT, va="top",
                transform=ax.transAxes)

    row_h = 0.09
    for i, (metric, value, desc) in enumerate(rows):
        y  = header_y - (i + 1) * row_h
        bg = REPORT_LIGHT if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - 0.01), 1, row_h,
                                   transform=ax.transAxes, facecolor=bg, edgecolor="none"))
        ax.text(0.01, y + 0.03, metric, fontsize=10, color=REPORT_DARK,  va="top", transform=ax.transAxes)
        ax.text(0.29, y + 0.03, value,  fontsize=10, color=REPORT_ACCENT, va="top", transform=ax.transAxes)
        ax.text(0.43, y + 0.03, desc,   fontsize=9,  color="#6b7280",    va="top", transform=ax.transAxes)

    png_path = _save_png(fig, export_path, f"{artifact_base}_metrics.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 3 — β vs θ Coefficient Table (SDM-specific)
# ---------------------------------------------------------------------------

def _build_coefficients_page(pp, export_path, artifact_base, coeff_table, rho, page_num):
    fig = _new_page()
    _add_header(fig, "Coefficients",
                "β = own effect (X)  |  θ = spillover effect (WX)  |  ★ p < 0.05")

    ax = fig.add_axes([0.04, 0.10, 0.92, 0.79])
    ax.axis("off")

    # Column layout: Variable | β | SE(β) | p(β) | θ | SE(θ) | p(θ) | Spillover
    col_labels = ["Variable", "β", "SE(β)", "p(β)", "θ", "SE(θ)", "p(θ)", "Spillover"]
    col_x      = [0.00, 0.22, 0.33, 0.43, 0.54, 0.65, 0.75, 0.86]
    header_y   = 0.97

    # Header background
    ax.add_patch(plt.Rectangle((0, header_y - 0.025), 1, 0.055,
                               transform=ax.transAxes, facecolor="#ccfbf1", edgecolor="none"))

    for j, label in enumerate(col_labels):
        color = REPORT_BLUE if label in ("β", "SE(β)", "p(β)") else \
                REPORT_ORANGE if label in ("θ", "SE(θ)", "p(θ)") else REPORT_ACCENT
        ax.text(col_x[j] + 0.005, header_y, label, fontsize=9, fontweight="bold",
                color=color, va="top", transform=ax.transAxes)

    # Divider under β group and θ group
    ax.text(0.225, header_y - 0.005, "── own ──", fontsize=7, color=REPORT_BLUE,
            va="top", transform=ax.transAxes)
    ax.text(0.545, header_y - 0.005, "── spillover ──", fontsize=7, color=REPORT_ORANGE,
            va="top", transform=ax.transAxes)

    row_h = min(0.072, 0.82 / max(len(coeff_table) + 2, 1))

    spillover_colors = {"positive": "#16a34a", "negative": "#dc2626", "none": "#94a3b8"}

    for i, row in enumerate(coeff_table):
        y   = header_y - (i + 1.6) * row_h
        bg  = REPORT_LIGHT if i % 2 == 0 else "white"
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
            REPORT_DARK,
            REPORT_BLUE if row.get("beta_sig")  else "#374151",
            "#6b7280", "#6b7280",
            REPORT_ORANGE if row.get("theta_sig") else "#374151",
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
                color=REPORT_ACCENT, va="top", transform=ax.transAxes)

    row_h = min(0.10, 0.85 / max(len(coeff_table) + 1, 1))
    sp_colors = {"positive": "#16a34a", "negative": "#dc2626", "none": "#94a3b8"}

    for i, row in enumerate(coeff_table):
        y   = header_y - (i + 1.5) * row_h
        bg  = REPORT_LIGHT if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - 0.01), 1, row_h,
                                   transform=ax.transAxes, facecolor=bg, edgecolor="none"))

        d  = row.get("avg_direct")
        ind = row.get("avg_indirect")
        t  = row.get("avg_total")
        sp = row.get("spillover_type", "none")

        vals   = [
            row["variable"],
            f"{d:.4f}"   if d   is not None else "N/A",
            f"{ind:.4f}" if ind is not None else "N/A",
            f"{t:.4f}"   if t   is not None else "N/A",
            sp.upper(),
        ]
        colors = [
            REPORT_DARK,
            REPORT_BLUE,
            REPORT_ORANGE,
            REPORT_PURPLE,
            sp_colors.get(sp, "#94a3b8"),
        ]
        for j, (val, color) in enumerate(zip(vals, colors)):
            ax.text(col_x[j] + 0.005, y + row_h * 0.55, val,
                    fontsize=9, color=color, va="top", transform=ax.transAxes)

    # Bar chart — avg direct vs indirect vs total
    ax2 = fig.add_axes([0.07, 0.10, 0.86, 0.36])
    feats    = [r["variable"] for r in coeff_table]
    directs  = [r.get("avg_direct")   or 0 for r in coeff_table]
    indirects= [r.get("avg_indirect") or 0 for r in coeff_table]
    totals   = [r.get("avg_total")    or 0 for r in coeff_table]

    x    = np.arange(len(feats))
    w    = 0.25
    ax2.bar(x - w, directs,   width=w, label="Avg Direct",   color=REPORT_BLUE,   alpha=0.85)
    ax2.bar(x,     indirects, width=w, label="Avg Indirect", color=REPORT_ORANGE, alpha=0.85)
    ax2.bar(x + w, totals,    width=w, label="Avg Total",    color=REPORT_PURPLE, alpha=0.85)
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
# Page 5 — Diagnostics (Actual vs Predicted · Residuals)
# ---------------------------------------------------------------------------

def _build_diagnostics_page(pp, export_path, artifact_base, y_full, preds_full, residuals, page_num):
    fig = _new_page()
    _add_header(fig, "Diagnostics", "Actual vs Predicted · Residual analysis")

    ax1 = fig.add_axes([0.07, 0.54, 0.40, 0.33])
    ax1.scatter(y_full, preds_full, alpha=0.45, s=18, color=REPORT_ACCENT, edgecolors="none")
    mn, mx = min(y_full.min(), preds_full.min()), max(y_full.max(), preds_full.max())
    ax1.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="Perfect fit")
    ax1.set_xlabel("Actual", fontsize=9)
    ax1.set_ylabel("Predicted", fontsize=9)
    ax1.set_title("Actual vs Predicted", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    avp_path = _save_png(fig, export_path, f"{artifact_base}_actual_vs_predicted.png")

    ax2 = fig.add_axes([0.55, 0.54, 0.40, 0.33])
    ax2.scatter(preds_full, residuals, alpha=0.45, s=18, color=REPORT_PURPLE, edgecolors="none")
    ax2.axhline(0, color="red", linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Predicted", fontsize=9)
    ax2.set_ylabel("Residual", fontsize=9)
    ax2.set_title("Residuals vs Predicted", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    rvp_path = _save_png(fig, export_path, f"{artifact_base}_residuals_vs_predicted.png")

    ax3 = fig.add_axes([0.07, 0.14, 0.86, 0.30])
    ax3.hist(residuals, bins=30, color=REPORT_ACCENT, alpha=0.75, edgecolor="white")
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


# ---------------------------------------------------------------------------
# Page 6 — Feature Importance (|β| ranked)
# ---------------------------------------------------------------------------

def _build_feature_importance_page(pp, export_path, artifact_base, coeff_table, page_num):
    fig = _new_page()
    _add_header(fig, "Feature Importance", "Ranked by |β| (own effect magnitude)")

    sorted_ct = sorted(coeff_table, key=lambda x: abs(x["beta"]), reverse=True)
    feats     = [r["variable"] for r in sorted_ct]
    b_vals    = [abs(r["beta"])  for r in sorted_ct]
    th_vals   = [abs(r["theta"]) for r in sorted_ct]
    b_colors  = ["#16a34a" if r["beta_sig"]  else "#93c5fd" for r in sorted_ct]
    th_colors = ["#ea580c" if r["theta_sig"] else "#fdba74" for r in sorted_ct]

    ax = fig.add_axes([0.07, 0.12, 0.86, 0.72])
    x  = np.arange(len(feats))
    w  = 0.35
    bars_b  = ax.barh(x + w / 2, b_vals[::-1],  height=w, color=b_colors[::-1],  label="|β| own effect")
    bars_th = ax.barh(x - w / 2, th_vals[::-1], height=w, color=th_colors[::-1], label="|θ| spillover")

    ax.set_yticks(x)
    ax.set_yticklabels(feats[::-1], fontsize=9)
    ax.set_xlabel("|Coefficient|", fontsize=10)
    ax.set_title("Feature Importance — Own (β) vs Spillover (θ)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    max_val = max(b_vals + th_vals) if b_vals or th_vals else 1
    for bar, val in zip(bars_b, b_vals[::-1]):
        ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color=REPORT_DARK)
    for bar, val in zip(bars_th, th_vals[::-1]):
        ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color=REPORT_DARK)

    ax.legend(handles=[
        Patch(color="#16a34a", label="|β| sig (p<0.05)"),
        Patch(color="#93c5fd", label="|β| not sig"),
        Patch(color="#ea580c", label="|θ| sig (p<0.05)"),
        Patch(color="#fdba74", label="|θ| not sig"),
    ], fontsize=8.5, loc="lower right")

    png_path = _save_png(fig, export_path, f"{artifact_base}_feature_importance.png")
    _add_footer(fig, artifact_base, f"Page {page_num}")
    pp.savefig(fig, facecolor="white")
    plt.close(fig)
    return png_path, page_num + 1


# ---------------------------------------------------------------------------
# Page 7 — Summary / Interpretation
# ---------------------------------------------------------------------------

def _build_summary_page(pp, artifact_base, metrics, indep, rho, moran_i, moran_p, n_samples, page_num):
    fig = _new_page()
    _add_header(fig, "Final Interpretation", "Spatial Durbin Model summary")

    box1 = fig.add_axes([0.07, 0.60, 0.40, 0.27])
    box1.axis("off")
    box1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_ACCENT, linewidth=1.3))
    box1.text(0.04, 0.90, "Model Summary", fontsize=12, fontweight="bold", color=REPORT_ACCENT)
    summary = (
        f"Samples    : {n_samples:,}\n"
        f"Predictors : {len(indep)}  (+{len(indep)} WX terms)\n"
        f"R² (test)  : {metrics.get('r2', 0):.4f}\n"
        f"Pseudo R²  : {metrics.get('pseudo_r2', 0):.4f}\n"
        f"RMSE       : {metrics.get('rmse', 0):.2f}\n"
        f"MAE        : {metrics.get('mae', 0):.2f}"
    )
    box1.text(0.04, 0.72, summary, fontsize=10, color=REPORT_DARK, va="top")

    box2 = fig.add_axes([0.53, 0.60, 0.40, 0.27])
    box2.axis("off")
    box2.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=REPORT_BORDER, linewidth=1.2))
    box2.text(0.04, 0.90, "Spatial Diagnostics", fontsize=12, fontweight="bold", color=REPORT_ACCENT)
    rho_interp = "Strong" if abs(rho) > 0.5 else ("Moderate" if abs(rho) > 0.2 else "Weak")
    mi_sig     = "significant" if (moran_p is not None and moran_p < 0.05) else "not significant"
    spatial_txt = f"ρ (rho)    : {rho:.4f}  [{rho_interp} spatial lag]\n"
    if moran_i is not None:
        spatial_txt += f"Moran's I  : {moran_i:.4f} (p={moran_p:.4f})\n"
        spatial_txt += f"  → Residuals are {mi_sig}"
    box2.text(0.04, 0.72, spatial_txt, fontsize=10, color=REPORT_DARK, va="top")

    notes = [
        "The Spatial Durbin Model extends the Spatial Lag Model by adding spatially lagged",
        "predictor variables (WX) alongside the lagged dependent variable (Wy).",
        "β captures the own effect of each predictor; θ captures the neighbor spillover effect.",
        "Positive θ = complementarity (neighbor's X raises local y).",
        "Negative θ = competition/substitution (neighbor's X depresses local y).",
        "Always interpret impacts from S_k(W), not raw β and θ — the spatial multiplier",
        "(I-ρW)⁻¹ amplifies effects through feedback across the network.",
        f"Moran's I on residuals is {mi_sig} — the model {'adequately' if mi_sig == 'not significant' else 'may not fully'} captures spatial structure.",
        "Queen contiguity weights used (row-standardized). Validate against domain knowledge.",
    ]
    y = 0.52
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

def export_sdm_report_and_artifacts(
    export_path:   str,
    artifact_base: str,
    model,
    indep:         List[str],
    target:        str,
    y_full:        np.ndarray,
    preds_full:    np.ndarray,
    residuals:     np.ndarray,
    metrics:       Dict[str, Any],
    coeff_table:   List[Dict],
    rho:           float,
    moran_i:       Optional[float],
    moran_p:       Optional[float],
    df_valid:      pd.DataFrame,
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

        png_paths["impacts"], page_num = _build_impacts_page(
            pp, export_path, artifact_base, coeff_table, page_num
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

    print(f"✅ SDM PDF report: {pdf_path}")
    return png_paths, pdf_path