"""
lr_print_handler.py
-------------------
PDF report generation for Linear Regression.
Layout engine: ReportLab (text, boxes, tables)
Charts:        Matplotlib / Seaborn (saved as PNG, embedded by ReportLab)
"""

from typing import List, Optional, Tuple, Dict, Any
import os, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

# ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Table, TableStyle, Image, KeepTogether, HRFlowable,
    NextPageTemplate, PageBreak,
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas as rl_canvas


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_ACCENT = colors.HexColor("#1e88e5")
C_DARK   = colors.HexColor("#1f2937")
C_LIGHT  = colors.HexColor("#f8fafc")
C_BORDER = colors.HexColor("#d0d7de")
C_SUBTTL = colors.HexColor("#5f6b7a")
C_HDRFIL = colors.HexColor("#f4f9ff")
C_DEFBOX = colors.HexColor("#f0f7ff")
C_TBLODD = colors.HexColor("#f5f7fa")
C_GREEN  = colors.HexColor("#d1fae5")
C_RED    = colors.HexColor("#fee2e2")

# ---------------------------------------------------------------------------
# Page geometry
# ---------------------------------------------------------------------------
PW, PH   = A4                        # 595.27 x 841.89 pts
ML = MR  = 0.75 * inch               # left / right margin
MT       = 0.65 * inch               # top margin  (below header)
MB       = 0.70 * inch               # bottom margin (above footer)
TW       = PW - ML - MR              # text (content) width
HEADER_H = 0.55 * inch
FOOTER_H = 0.35 * inch


# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------
def _sty(name, **kw):
    base = dict(fontName="Helvetica", fontSize=10, leading=14,
                textColor=C_DARK, spaceAfter=0, spaceBefore=0)
    base.update(kw)
    return ParagraphStyle(name, **base)

S_TITLE    = _sty("title",   fontName="Helvetica-Bold", fontSize=22, textColor=C_ACCENT, leading=26)
S_H1       = _sty("h1",      fontName="Helvetica-Bold", fontSize=20, textColor=C_ACCENT, leading=24)
S_H2       = _sty("h2",      fontName="Helvetica-Bold", fontSize=12, textColor=C_ACCENT, leading=16)
S_BODY     = _sty("body",    fontSize=10, leading=14, alignment=TA_JUSTIFY)
S_BODY_L   = _sty("bodyl",   fontSize=10, leading=14, alignment=TA_LEFT)
S_SMALL    = _sty("small",   fontSize=9,  leading=13, alignment=TA_JUSTIFY)
S_SMALL_L  = _sty("smalll",  fontSize=9,  leading=13, alignment=TA_LEFT)
S_BULLET   = _sty("bullet",  fontSize=9.5, leading=14, leftIndent=10,
                  firstLineIndent=-10, alignment=TA_LEFT)
S_META_KEY = _sty("mkey",    fontName="Helvetica-Bold", fontSize=10.5, leading=15)
S_META_VAL = _sty("mval",    fontSize=10.5, leading=15)
S_FOOTER   = _sty("footer",  fontSize=8.5, textColor=colors.HexColor("#6b7280"))
S_SUBTITLE = _sty("subtitle",fontSize=10.5, textColor=C_SUBTTL, leading=14)
S_CAPTION  = _sty("caption", fontSize=9, textColor=C_SUBTTL, leading=12, alignment=TA_CENTER)
S_TBL_HDR  = _sty("tblhdr",  fontName="Helvetica-Bold", fontSize=9.5,
                  textColor=colors.white, alignment=TA_CENTER)
S_TBL_BODY = _sty("tblbody", fontSize=9, alignment=TA_CENTER)


# ---------------------------------------------------------------------------
# Header / Footer canvas callback
# ---------------------------------------------------------------------------
def _draw_header_footer(c: rl_canvas.Canvas, doc, artifact_base: str, page_num: int):
    c.saveState()
    # Header title + subtitle stored in doc._lr_page_meta
    meta = getattr(doc, "_lr_page_meta", {}).get(page_num, {})
    title    = meta.get("title", "")
    subtitle = meta.get("subtitle", "")

    if title:
        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(C_ACCENT)
        c.drawString(ML, PH - 0.42 * inch, title)
    if subtitle:
        c.setFont("Helvetica", 10.5)
        c.setFillColor(C_SUBTTL)
        c.drawString(ML, PH - 0.58 * inch, subtitle)

    # Header rule
    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.8)
    c.line(ML, PH - 0.65 * inch, PW - MR, PH - 0.65 * inch)

    # Footer rule
    c.line(ML, MB - 0.12 * inch, PW - MR, MB - 0.12 * inch)
    c.setFont("Helvetica", 8.5)
    c.setFillColor(colors.HexColor("#6b7280"))
    c.drawString(ML, MB - 0.30 * inch, f"Model Report | {artifact_base}")
    c.drawRightString(PW - MR, MB - 0.30 * inch, f"Page {page_num}")
    c.restoreState()


# ---------------------------------------------------------------------------
# Reusable flowable: InfoBox  (titled bordered box, body = list of Paragraphs)
# ---------------------------------------------------------------------------
class InfoBox(Flowable):
    """A bordered box with optional tinted title bar and body content."""

    def __init__(self, title: str, body_paras: list,
                 bg=colors.white, title_bg=C_HDRFIL,
                 border=C_BORDER, width=None,
                 pad_h=8, pad_v=7, title_pad=6):
        super().__init__()
        self._title      = title
        self._paras      = body_paras
        self._bg         = bg
        self._title_bg   = title_bg
        self._border     = border
        self._width      = width or TW
        self._pad_h      = pad_h
        self._pad_v      = pad_v
        self._title_pad  = title_pad
        self.width       = self._width
        self._title_h    = 22
        self._body_h     = 0

    def _inner_w(self):
        return self._width - 2 * self._pad_h

    def wrap(self, aW, aH):
        iw = self._inner_w()
        body_h = 0
        for p in self._paras:
            w, h = p.wrap(iw, 9999)
            body_h += h + 3
        self._body_h = body_h
        total = self._title_h + self._pad_v + body_h + self._pad_v
        self.height = total
        return self._width, total

    def draw(self):
        w, h = self._width, self.height
        c = self.canv
        c.saveState()

        # Outer box
        c.setFillColor(self._bg)
        c.setStrokeColor(self._border)
        c.setLineWidth(0.8)
        c.roundRect(0, 0, w, h, 4, fill=1, stroke=1)

        # Title bar
        c.setFillColor(self._title_bg)
        c.roundRect(0, h - self._title_h, w, self._title_h, 4, fill=1, stroke=0)
        c.rect(0, h - self._title_h, w, self._title_h / 2, fill=1, stroke=0)

        # Title text
        c.setFont("Helvetica-Bold", 11.5)
        c.setFillColor(C_ACCENT)
        c.drawString(self._pad_h, h - self._title_h + self._title_pad, self._title)

        # Body paragraphs
        y = h - self._title_h - self._pad_v
        iw = self._inner_w()
        for p in self._paras:
            pw, ph = p.wrap(iw, 9999)
            y -= ph
            p.drawOn(c, self._pad_h, y)
            y -= 3

        c.restoreState()


# ---------------------------------------------------------------------------
# Helpers: build InfoBox quickly
# ---------------------------------------------------------------------------
def _info_box(title: str, rows: List[Tuple[str, str]],
              bg=colors.white, title_bg=C_HDRFIL) -> InfoBox:
    """Two-column key/value rows."""
    paras = []
    for k, v in rows:
        paras.append(Paragraph(
            f'<font name="Helvetica-Bold">{k}</font>&nbsp;&nbsp;&nbsp;{v}',
            S_BODY_L))
    return InfoBox(title, paras, bg=bg, title_bg=title_bg)


def _bullet_box(title: str, bullets: List[str],
                fontsize=9.5, bg=colors.white) -> InfoBox:
    """Bulleted list inside an InfoBox."""
    style = ParagraphStyle("bul2", parent=S_BULLET, fontSize=fontsize,
                           leading=int(fontsize * 1.45))
    paras = [Paragraph(f"• {b}", style) for b in bullets]
    return InfoBox(title, paras, bg=bg)


def _para_box(title: str, text: str, bg=colors.white) -> InfoBox:
    """Single justified paragraph inside an InfoBox."""
    return InfoBox(title, [Paragraph(text, S_BODY)], bg=bg)


# ---------------------------------------------------------------------------
# ReportLab table helper
# ---------------------------------------------------------------------------
def _rl_table(headers: List[str], rows: List[List[str]],
              col_widths: List[float] = None,
              sig_col: int = None) -> Table:
    data = [[Paragraph(h, S_TBL_HDR) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), S_TBL_BODY) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)

    style = [
        ("BACKGROUND",  (0, 0), (-1, 0),  C_ACCENT),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_TBLODD, colors.white]),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
    ]
    if sig_col is not None:
        for i, row in enumerate(rows, start=1):
            val = str(row[sig_col])
            bg  = C_GREEN if val == "Yes" else C_RED
            style.append(("BACKGROUND", (sig_col, i), (sig_col, i), bg))
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------------
# Matplotlib chart helpers  (return PNG path)
# ---------------------------------------------------------------------------
def _tmp_png(prefix="chart"):
    fd, path = tempfile.mkstemp(suffix=".png", prefix=prefix)
    os.close(fd)
    return path


def _chart_scatter(y_test, preds, title, xlabel, ylabel, color, w=6, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    ax.scatter(y_test, preds, alpha=0.65, color=color, edgecolor="black", linewidth=0.4)
    mn = min(float(np.min(y_test)), float(np.min(preds)))
    mx = max(float(np.max(y_test)), float(np.max(preds)))
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.3, label="Perfect Prediction")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1e88e5", pad=8)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8); ax.legend(fontsize=8); ax.grid(alpha=0.20)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = _tmp_png("scatter"); fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    return p


def _chart_residuals(preds, residuals, w=6, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    ax.scatter(preds, residuals, alpha=0.65, color="#ef4444", edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linestyle="--", lw=1.3, label="Zero Line")
    ax.set_title("Residuals vs Predicted", fontsize=12, fontweight="bold", color="#dc2626", pad=8)
    ax.set_xlabel("Predicted Values", fontsize=9); ax.set_ylabel("Residuals", fontsize=9)
    ax.tick_params(labelsize=8); ax.legend(fontsize=8); ax.grid(alpha=0.20)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = _tmp_png("resid"); fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    return p


def _chart_hist_residuals(residuals, w=6, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    sns.histplot(residuals, kde=True, ax=ax, color="#1e88e5", edgecolor="black", bins=20)
    custom_ticks = [0, 2000, 4000, 6000, 8000, 10000]
    data_max = ax.get_ylim()[1]
    visible = [t for t in custom_ticks if t <= data_max * 1.10] or custom_ticks[:3]
    ax.set_yticks(visible)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: "0" if x == 0 else f"{int(x/1000):.0f}k"))
    ax.set_ylim(0, max(visible) * 1.10)
    ax.set_title("Residual Distribution", fontsize=13, fontweight="bold", color="#1e88e5", pad=12)
    ax.set_xlabel("Residual"); ax.set_ylabel("Frequency")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    p = _tmp_png("reshist"); fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    return p


def _chart_feature_bar(feat_names, feat_vals, xlabel="Coefficient", title="Feature Importance", w=6, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    ax.barh(feat_names, feat_vals, color="#1e88e5", edgecolor="#1f1f1f", linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1e88e5", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis="y", labelsize=8); ax.tick_params(axis="x", labelsize=8.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25); ax.invert_yaxis()
    fig.tight_layout()
    p = _tmp_png("featbar"); fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    return p


def _chart_histogram(col_data, col_name, mean_val, median_val, std_val, w=5.5, h=3.0):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    sns.histplot(col_data, kde=True, ax=ax, color="#1e88e5", edgecolor="black", bins=25)
    ax.set_title(f"Distribution of {col_name}", fontsize=12, fontweight="bold",
                 color="#1e88e5", pad=10)
    ax.set_xlabel(col_name); ax.set_ylabel("Frequency")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.text(0.98, 0.95,
            f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#1e88e5", alpha=0.92))
    fig.tight_layout()
    p = _tmp_png("hist"); fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Simple doc builder  (one-pass, page-by-page)
# ---------------------------------------------------------------------------
class _LRDoc:
    """Builds pages as a list of (meta, story) tuples, then writes PDF."""

    def __init__(self, pdf_path: str, artifact_base: str):
        self.pdf_path      = pdf_path
        self.artifact_base = artifact_base
        self._pages: List[Tuple[dict, list]] = []   # [(meta, flowables), …]

    def add_page(self, title: str, subtitle: str, story: list):
        self._pages.append({"title": title, "subtitle": subtitle}, story)

    def _add(self, meta: dict, story: list):
        self._pages.append((meta, story))

    def save(self):
        from reportlab.pdfgen import canvas as C
        from reportlab.platypus import SimpleDocTemplate

        ab = self.artifact_base

        # Build one big story with PageBreaks between sections
        full_story = []
        page_metas = {}   # page_number → meta  (filled during build)

        class _PageCounter:
            def __init__(self): self.n = 0
            def __call__(self, canvas, doc):
                self.n += 1
                meta = self._metas.get(self.n, {})
                _draw_header_footer(canvas, doc, ab, self.n)

        # Use a SimpleDocTemplate with custom onPage
        doc = SimpleDocTemplate(
            self.pdf_path,
            pagesize=A4,
            leftMargin=ML, rightMargin=MR,
            topMargin=MT + HEADER_H, bottomMargin=MB + FOOTER_H,
        )
        doc._lr_page_meta = {}
        doc._lr_ab        = ab

        # Flatten pages → story, inserting PageBreaks and meta markers
        story = []
        for i, (meta, flowables) in enumerate(self._pages):
            if i > 0:
                story.append(PageBreak())
            story.append(_MetaMarker(meta, i + 1))
            story.extend(flowables)

        page_counter = [0]
        current_meta = [{}]

        def on_page(canvas, doc):
            page_counter[0] += 1
            _draw_header_footer(canvas, doc, ab, page_counter[0])

        doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


class _MetaMarker(Flowable):
    """Zero-height flowable that injects page meta into the doc."""
    def __init__(self, meta, page_idx):
        super().__init__()
        self.meta      = meta
        self.page_idx  = page_idx
        self.width = self.height = 0

    def draw(self):
        # Store meta keyed by actual page number at render time
        pass

    def wrap(self, *_): return 0, 0


# ---------------------------------------------------------------------------
# Story builders  (return list of Flowables)
# ---------------------------------------------------------------------------
GAP  = Spacer(1, 10)
GAP2 = Spacer(1, 18)


def _story_cover(artifact_base, target, features, n_samples, model=None):
    fit_intercept = getattr(model, "fit_intercept", True)  if model else True
    normalize     = getattr(model, "normalize",     False) if model else False
    positive      = getattr(model, "positive",      False) if model else False

    return [
        Paragraph("Linear Regression Model Report", S_TITLE),
        Spacer(1, 14),
        HRFlowable(width=TW, thickness=0.8, color=C_BORDER),
        Spacer(1, 14),
        InfoBox("What is Linear Regression?", [Paragraph(
            "Linear Regression is a method that estimates how the target value changes "
            "based on the input variables. In simple terms, it finds the best-fitting "
            "straight-line relationship between the features and the value you want to "
            "predict. Each feature is given a weight called a coefficient, and those "
            "coefficients are used to compute predictions.",
            S_BODY)], bg=C_DEFBOX, title_bg=C_DEFBOX),
        GAP2,
        _info_box("Model Hyperparameters", [
            ("fit_intercept", str(fit_intercept)),
            ("normalize",     str(normalize)),
            ("positive",      str(positive)),
        ]),
        GAP2,
        _info_box("Model Information", [
            ("Model Type",       "Linear Regression"),
            ("Model Name",       artifact_base),
            ("Target Variable",  target),
            ("Feature Count",    str(len(features))),
            ("Training Samples", f"{n_samples:,}"),
            ("Generated At",     datetime.now().strftime("%Y-%b-%d %I:%M:%S %p")),
        ]),
    ]


def _story_exec_summary(metrics, independent_vars, importance, residual_ttest):
    sorted_pairs = sorted(zip(independent_vars, importance),
                          key=lambda x: abs(float(x[1])), reverse=True)
    top_feature = sorted_pairs[0][0] if sorted_pairs else "N/A"
    top_value   = float(sorted_pairs[0][1]) if sorted_pairs else 0.0

    r2, rmse, mae = metrics["r2"], metrics["rmse"], metrics["mae"]
    if r2 >= 0.75:   perf = f"R² = {r2:.3f} indicates strong explanatory power."
    elif r2 >= 0.50: perf = f"R² = {r2:.3f} indicates moderate explanatory power."
    else:            perf = f"R² = {r2:.3f} indicates limited explanatory power."

    perf_bullets = [
        perf,
        f"Average prediction error is around MAE = {mae:.2f}, while RMSE = {rmse:.2f} "
        f"suggests the effect of larger errors.",
        f"Most influential predictor: {top_feature} (standardized effect = {top_value:.4f}).",
    ]

    p_val = residual_ttest["p_value"]
    res_msg = ("Residual mean is statistically different from zero at alpha = 0.05."
               if p_val < 0.05
               else "Residual mean is not statistically different from zero at alpha = 0.05.")
    res_bullets = [
        f"T-statistic: {residual_ttest['t_stat']:.4f}",
        f"P-value: {p_val:.4f}",
        res_msg,
    ]

    rec_text = (
        "Use the metrics page to evaluate overall fit, the feature pages to review "
        "standardized effects and coefficient significance, and the diagnostics page "
        "to inspect bias and error behavior. Variable distribution pages provide "
        "context for predictor spread."
    )

    return [
        _bullet_box("Performance Summary", perf_bullets),
        GAP,
        _bullet_box("Residual Check", res_bullets),
        GAP,
        _para_box("Recommended Reading of this Report", rec_text),
    ]


def _story_metrics(metrics, model_type="Linear Regression"):
    r2, rmse, mae, mse = metrics["r2"], metrics["rmse"], metrics["mae"], metrics["mse"]
    rows = [
        ["R²",   f"{r2:.4f}",   "Explained variance of the model"],
        ["RMSE", f"{rmse:.2f}", "Penalizes larger prediction errors"],
        ["MAE",  f"{mae:.2f}",  "Average absolute error"],
        ["MSE",  f"{mse:.2f}",  "Mean squared error"],
    ]
    col_w = [TW * 0.22, TW * 0.22, TW * 0.56]
    notes = (
        f"This model achieved R² = {r2:.4f}. RMSE and MAE should be interpreted relative "
        f"to the scale of the target variable. Lower values generally indicate better fit, "
        f"but diagnostic plots are still needed to assess whether the model behaves well "
        f"across the data range."
    )
    return [
        _rl_table(["Metric", "Value", "Interpretation"], rows, col_widths=col_w),
        GAP2,
        _para_box("Interpretation Notes", notes),
    ]


def _story_feature_importance(independent_vars, importance, coef_ttests, export_path):
    sorted_pairs = sorted(zip(independent_vars, importance),
                          key=lambda x: abs(float(x[1])), reverse=True)
    feat_names = [x[0] for x in sorted_pairs]
    feat_vals  = [float(x[1]) for x in sorted_pairs]

    chart_png = _chart_feature_bar(feat_names, feat_vals,
                                   xlabel="Coefficient",
                                   title="Feature Importance (Standardized Coefficients)")

    story = [Image(chart_png, width=TW, height=TW * 0.52)]
    story.append(GAP)

    if coef_ttests:
        rows = []
        for r in coef_ttests:
            rows.append([r["variable"], f"{r['coef']:.6f}", f"{r['std_err']:.6f}",
                         f"{r['t']:.4f}", f"{r['p']:.4f}",
                         "Yes" if r["significant"] else "No"])
        cw = [TW*0.20, TW*0.16, TW*0.16, TW*0.14, TW*0.14, TW*0.12]
        story.append(_rl_table(
            ["Variable","Coefficient","Std Error","t-stat","p-value","Significant"],
            rows, col_widths=cw, sig_col=5))
    return story


def _story_diagnostics(y_test, preds, residuals, export_path):
    scatter_png = _chart_scatter(y_test, preds,
                                 "Actual vs Predicted",
                                 "Actual Values", "Predicted Values", "#1e88e5")
    resid_png   = _chart_residuals(preds, residuals)
    return [
        Image(scatter_png, width=TW, height=TW * 0.55),
        GAP,
        Image(resid_png,   width=TW, height=TW * 0.46),
    ]


def _story_residual_dist(residuals, residual_ttest):
    hist_png = _chart_hist_residuals(residuals)
    t, p = residual_ttest["t_stat"], residual_ttest["p_value"]
    conclusion = ("Conclusion: residual mean differs significantly from zero."
                  if p < 0.05
                  else "Conclusion: residual mean is not significantly different from zero.")
    ttest_text = f"T-statistic: {t:.4f}     P-value: {p:.4f}     {conclusion}"
    return [
        Image(hist_png, width=TW, height=TW * 0.56),
        GAP,
        _para_box("Residual t-test", ttest_text),
    ]


def _story_var_distributions(X_train_unscaled, independent_vars):
    story = []
    pairs = [independent_vars[i:i+2] for i in range(0, len(independent_vars), 2)]
    for pair in pairs:
        row_imgs = []
        for col in pair:
            try:
                d = X_train_unscaled[col].dropna()
                p = _chart_histogram(d, col, float(d.mean()), float(d.median()), float(d.std()),
                                     w=5.2, h=3.0)
                row_imgs.append(p)
            except Exception:
                row_imgs.append(None)
        # side-by-side if 2 charts, else full width
        if len(row_imgs) == 2 and all(row_imgs):
            w2 = TW / 2 - 4
            t = Table([[Image(row_imgs[0], width=w2, height=w2*0.58),
                        Image(row_imgs[1], width=w2, height=w2*0.58)]],
                      colWidths=[w2, w2])
            t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                                   ("LEFTPADDING",(0,0),(-1,-1),2),
                                   ("RIGHTPADDING",(0,0),(-1,-1),2)]))
            story.append(t)
        elif row_imgs[0]:
            story.append(Image(row_imgs[0], width=TW, height=TW*0.42))
        story.append(GAP)
    return story


def _story_final_summary(metrics, independent_vars, importance, target,
                          n_samples, residual_ttest):
    sorted_pairs = sorted(zip(independent_vars, importance),
                          key=lambda x: abs(float(x[1])), reverse=True)
    top3 = sorted_pairs[:3]

    residual_note = (
        "Residual mean is significantly different from zero."
        if residual_ttest["p_value"] < 0.05
        else "Residual mean is not significantly different from zero."
    )

    # Two side-by-side boxes
    box1 = _info_box("Model Summary", [
        ("Target variable:",   target),
        ("Training samples:",  f"{n_samples:,}"),
        ("R²:",                f"{metrics['r2']:.4f}"),
        ("RMSE:",              f"{metrics['rmse']:.2f}"),
        ("MAE:",               f"{metrics['mae']:.2f}"),
    ], title_bg=C_HDRFIL)

    box2 = _bullet_box("Top Predictors",
                       [f"{i+1}. {f}  ({float(v):.4f})" for i, (f, v) in enumerate(top3)])

    half = TW / 2 - 6
    box1.width = half; box1._width = half
    box2.width = half; box2._width = half

    tbl = Table([[box1, box2]], colWidths=[half, half])
    tbl.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
        ("ALIGN",        (0,0),(-1,-1), "LEFT"),
    ]))

    notes = [
        f"This linear regression model was trained using {len(independent_vars)} predictor(s).",
        f"Overall fit should be judged mainly through R² and diagnostic plots. Current R² is {metrics['r2']:.4f}.",
        residual_note,
        "Use this report together with business or domain validation before deployment.",
    ]
    notes_box = _bullet_box("Documentation Notes", notes)

    return [tbl, GAP2, notes_box]


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
    from reportlab.platypus import SimpleDocTemplate, PageBreak

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

    n, k = len(y_train), len(independent_vars)
    dof  = n - k - 1
    residual_std_error = np.sqrt(np.sum(residuals ** 2) / dof)

    try:
        XtX_inv    = np.linalg.inv(X_train_scaled.T @ X_train_scaled)
        var_coef   = residual_std_error ** 2 * np.diag(XtX_inv)
        std_errors = np.sqrt(var_coef)
        t_stats    = model.coef_ / std_errors
        p_values   = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
        iv         = residual_std_error ** 2 * (
            1/n + np.mean(X_train_scaled,axis=0) @ XtX_inv @ np.mean(X_train_scaled,axis=0).T)
        i_se       = np.sqrt(iv)
        i_t        = model.intercept_ / i_se
        i_p        = 2 * (1 - stats.t.cdf(abs(i_t), dof))
        coef_ttests = [{"variable":"Intercept","coef":float(model.intercept_),
                        "std_err":float(i_se),"t":float(i_t),"p":float(i_p),
                        "significant":bool(i_p<0.05)}]
        for i, var in enumerate(independent_vars):
            coef_ttests.append({"variable":var,"coef":float(model.coef_[i]),
                                 "std_err":float(std_errors[i]),"t":float(t_stats[i]),
                                 "p":float(p_values[i]),"significant":bool(p_values[i]<0.05)})
    except Exception as e:
        print(f"Could not calculate coefficient t-tests: {e}")
        coef_ttests = None

    metrics = {"r2": float(r2), "mse": float(mse), "mae": float(mae), "rmse": float(rmse)}

    # ── Page definitions: (title, subtitle, story) ──────────────────────────
    pages = [
        ("",                    "",                                          _story_cover(artifact_base, target, independent_vars, len(y_train), model)),
        ("Executive Summary",   "Key results and top-level interpretation",  _story_exec_summary(metrics, independent_vars, importance, residual_ttest)),
        ("Model Performance Metrics", "Core evaluation results",             _story_metrics(metrics)),
        ("Feature Analysis",    "Coefficient importance and significance",   _story_feature_importance(independent_vars, importance, coef_ttests, export_path)),
        ("Prediction Diagnostics","Observed fit and residual behavior",      _story_diagnostics(y_test, preds, residuals, export_path)),
        ("Residual Analysis",   "Residual distribution and one-sample t-test", _story_residual_dist(residuals, residual_ttest)),
    ]
    if X_train_unscaled is not None and len(independent_vars) > 0:
        pages.append(("Variable Distributions", "Predictor spread and basic descriptive statistics",
                       _story_var_distributions(X_train_unscaled, independent_vars)))
    pages.append(("Final Interpretation", "Concise model documentation summary",
                  _story_final_summary(metrics, independent_vars, importance,
                                       target, len(y_train), residual_ttest)))

    # ── Build PDF ────────────────────────────────────────────────────────────
    pdf_path = os.path.join(export_path, f"{artifact_base}.pdf")
    ab       = artifact_base

    page_metas = {i+1: {"title": t, "subtitle": s} for i, (t, s, _) in enumerate(pages)}
    page_counter = [0]

    def on_page(canvas, doc):
        page_counter[0] += 1
        meta = page_metas.get(page_counter[0], {})
        # Patch doc meta for header
        doc._lr_page_meta = meta
        _draw_header_footer(canvas, doc, ab, page_counter[0])

    full_story = []
    for i, (title, subtitle, story) in enumerate(pages):
        if i > 0:
            full_story.append(PageBreak())
        full_story.extend(story)

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=MT + HEADER_H,
        bottomMargin=MB + FOOTER_H,
    )
    doc.build(full_story, onFirstPage=on_page, onLaterPages=on_page)

    png_paths: Dict[str, str] = {}
    t_tests = {"residuals": residual_ttest, "coefficients": coef_ttests}
    return metrics, png_paths, t_tests, pdf_path