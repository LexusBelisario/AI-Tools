import React, { useState } from "react";
import Plot from "react-plotly.js";
import {
  SCATTER_LAYOUT,
  BAR_LAYOUT,
  DISTRIBUTION_LAYOUT,
  SCATTER_MARKER,
  BAR_MARKER,
  PLOT_CONFIG,
  getDashedLine,
  getAxisTitle,
  getFeatureImportanceMarker,
  FONT_FAMILY,
} from "./components/plotStyles.js";

// --- Results Tab (top-level) ---

export default function ResultsTabUI({
  results,
  trainErrors = {},
  activeModelTab,
  setActiveModelTab,
  onShowMap,
  setLoadingMap,
  setLoadingFieldName,
  setSaveModalOpen,
  setSaveConfig,
  userSchema,
}) {
  const hasLR = !!results.lr;
  const hasRF = !!results.rf;
  const hasXGB = !!results.xgb;
  const hasSLM = !!results.slm;
  const hasLRError = !hasLR && !!trainErrors.lr;
  const hasRFError = !hasRF && !!trainErrors.rf;
  const hasXGBError = !hasXGB && !!trainErrors.xgb;
  const hasSLMError = !hasSLM && !!trainErrors.slm;
  const hasHybridSLM = !!results.hybrid_slm;
  const hasHybridSLMError = !hasHybridSLM && !!trainErrors.hybrid;
  const hasHybrid = !!results.hybrid;
  const hasHybridError = !hasHybrid && !!trainErrors.hybrid;

  return (
    <div className="blgf-ai-content">
      <div className="blgf-ai-modeltabs">
        {hasLR && (
          <div
            className={`blgf-ai-modeltab ${
              activeModelTab === "lr" ? "active" : ""
            }`}
            onClick={() => setActiveModelTab("lr")}
          >
            Linear Regression
          </div>
        )}
        {hasLRError && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "lr_err" ? "active" : ""}`}
            onClick={() => setActiveModelTab("lr_err")}
            style={{ color: "#f87171" }}
          >
            Linear Regression ✕
          </div>
        )}

        {hasRF && (
          <div
            className={`blgf-ai-modeltab ${
              activeModelTab === "rf" ? "active" : ""
            }`}
            onClick={() => setActiveModelTab("rf")}
          >
            Random Forest
          </div>
        )}
        {hasRFError && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "rf_err" ? "active" : ""}`}
            onClick={() => setActiveModelTab("rf_err")}
            style={{ color: "#f87171" }}
          >
            Random Forest ✕
          </div>
        )}

        {hasXGB && (
          <div
            className={`blgf-ai-modeltab ${
              activeModelTab === "xgb" ? "active" : ""
            }`}
            onClick={() => setActiveModelTab("xgb")}
          >
            XGBoost
          </div>
        )}
        {hasXGBError && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "xgb_err" ? "active" : ""}`}
            onClick={() => setActiveModelTab("xgb_err")}
            style={{ color: "#f87171" }}
          >
            XGBoost ✕
          </div>
        )}

        {hasSLM && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "slm" ? "active" : ""}`}
            onClick={() => setActiveModelTab("slm")}
          >
            Spatial Lag Model
          </div>
        )}
        {hasSLMError && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "slm_err" ? "active" : ""}`}
            onClick={() => setActiveModelTab("slm_err")}
            style={{ color: "#f87171" }}
          >
            Spatial Lag Model ✕
          </div>
        )}

        {hasHybridSLM && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "hybrid_slm" ? "active" : ""}`}
            onClick={() => setActiveModelTab("hybrid_slm")}
            style={activeModelTab === "hybrid_slm" ? { borderColor: "#2563eb" } : {}}
          >
            Spatial Lag Model
          </div>
        )}
        {hasHybridSLMError && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "hybrid_slm_err" ? "active" : ""}`}
            onClick={() => setActiveModelTab("hybrid_slm_err")}
            style={{ color: "#f87171" }}
          >
            Spatial Lag Model ✕
          </div>
        )}

        {hasHybrid && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "hybrid" ? "active" : ""}`}
            onClick={() => setActiveModelTab("hybrid")}
            style={activeModelTab === "hybrid" ? { borderColor: "#7c3aed" } : {}}
          >
            Hybrid Spatial Lag Model + Random Forest
          </div>
        )}
        {hasHybridError && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "hybrid_err" ? "active" : ""}`}
            onClick={() => setActiveModelTab("hybrid_err")}
            style={{ color: "#f87171" }}
          >
            Hybrid Spatial Lag Model + Random Forest ✕
          </div>
        )}
      </div>

      {!activeModelTab && (
        <div className="blgf-ai-empty-placeholder">
          Train a model first to display results.
        </div>
      )}

      {/* Error panel for failed models */}
      {activeModelTab && activeModelTab.endsWith("_err") && (
        <div className="blgf-ai-block" style={{ marginTop: 16 }}>
          <div style={{
            background: "#1e1010",
            border: "1px solid #f87171",
            borderRadius: 8,
            padding: "16px",
          }}>
            <div style={{ color: "#f87171", fontWeight: 600, marginBottom: 8 }}>
              ⚠️ Training Failed
            </div>
            <div style={{ color: "#fca5a5", fontSize: 13, fontFamily: "monospace", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
              {trainErrors[activeModelTab.replace("_err", "")]}
            </div>
          </div>
        </div>
      )}

      {activeModelTab && !activeModelTab.endsWith("_err") && (
        <ModelSection
          modelType={activeModelTab}
          modelResult={results[activeModelTab]}
          onShowMap={onShowMap}
          setLoadingMap={setLoadingMap}
          setLoadingFieldName={setLoadingFieldName}
          setSaveModalOpen={setSaveModalOpen}
          setSaveConfig={setSaveConfig}
          userSchema={userSchema}
        />
      )}
    </div>
  );
}

// --- Model Section (per-model view with sub-tabs) ---

function ModelSection({
  modelType,
  modelResult,
  onShowMap,
  setLoadingMap,
  setLoadingFieldName,
  setSaveModalOpen,
  setSaveConfig,
  userSchema,
}) {
  const [subTab, setSubTab] = useState("metrics");

  if (!modelResult) return null;

  const niceName =
    modelType === "lr"         ? "Linear Regression"
    : modelType === "rf"       ? "Random Forest"
    : modelType === "slm"      ? "Spatial Lag Model"
    : modelType === "hybrid_slm" ? "Spatial Lag Model"
    : modelType === "hybrid"   ? "Hybrid Spatial Lag Model + Random Forest"
    : "XGBoost";

  return (
    <div className="blgf-ai-result">
      <div className="blgf-ai-result-header">
        <div className="blgf-ai-modeltitle">{niceName}</div>
        <div className="blgf-ai-subtabs">
          <div
            className={`blgf-ai-subtab ${subTab === "metrics" ? "active" : ""}`}
            onClick={() => setSubTab("metrics")}
          >
            Metrics
          </div>

          <div
            className={`blgf-ai-subtab ${subTab === "plots" ? "active" : ""}`}
            onClick={() => setSubTab("plots")}
          >
            Plots
          </div>

          <div
            className={`blgf-ai-subtab ${subTab === "dist" ? "active" : ""}`}
            onClick={() => setSubTab("dist")}
          >
            Distributions
          </div>
        </div>
      </div>

      {subTab === "metrics" && (
        <>
          <MetricsSection
            modelType={modelType}
            result={modelResult}
            onShowMap={onShowMap}
            setLoadingMap={setLoadingMap}
            setLoadingFieldName={setLoadingFieldName}
            setSaveModalOpen={setSaveModalOpen}
            setSaveConfig={setSaveConfig}
            userSchema={userSchema}
          />
          <ImportanceSection modelType={modelType} result={modelResult} />
          {modelType === "lr" && <LRCoefficientsSection result={modelResult} />}
          {(modelType === "slm" || modelType === "hybrid_slm") && <SLMCoefficientsSection result={modelResult} />}
          {(modelType === "hybrid" || modelType === "hybrid_slm") && <HybridDiagnosticsSection result={modelResult} />}
          <ModelCAMA result={modelResult} />
        </>
      )}

      {subTab === "plots" && (
        <PlotsSection modelType={modelType} result={modelResult} />
      )}

      {subTab === "dist" && (
        <VariableDistributions modelType={modelType} result={modelResult} />
      )}
    </div>
  );
}

// --- Metrics ---

function MetricsSection({
  modelType,
  result,
  onShowMap,
  setLoadingMap,
  setLoadingFieldName,
  setSaveModalOpen,
  setSaveConfig,
  userSchema,
}) {
  const metrics = result?.metrics || {};
  return (
    <>
      <ModelDownloads
        modelType={modelType}
        result={result}
        onShowMap={onShowMap}
        setLoadingMap={setLoadingMap}
        setLoadingFieldName={setLoadingFieldName}
        setSaveModalOpen={setSaveModalOpen}
        setSaveConfig={setSaveConfig}
        userSchema={userSchema}
      />

      <div className="blgf-ai-card">
        <div className="blgf-ai-subtitle2">Performance Metrics</div>

        <table className="blgf-ai-table narrow">
          <tbody>
            {[
              ["MSE", metrics.MSE ?? metrics.mse],
              ["RMSE", metrics.RMSE ?? metrics.rmse],
              ["MAE", metrics.MAE ?? metrics.mae],
              ["R²", metrics["R²"] ?? metrics.r2],
              ...(modelType === "slm" ? [
                ["Pseudo R²", metrics.pseudo_r2],
                ["ρ (Spatial Lag)", metrics.rho],
                ["Moran's I (residuals)", metrics.moran_i],
                ["Moran's I p-value", metrics.moran_p],
              ] : []),
              ...(modelType === "hybrid" ? [
                ["R² — SLM Stage", metrics.r2_slm],
                ["RMSE — SLM Stage", metrics.rmse_slm],
                ["Pseudo R² (SLM)", metrics.pseudo_r2],
                ["ρ (Spatial Lag)", metrics.rho],
                ["Moran's I — SLM residuals", metrics.moran_i_slm],
                ["Moran's I — Hybrid residuals", metrics.moran_i_hybrid],
              ] : []),
            ]
              .filter(([, v]) => v !== undefined && v !== null)
              .map(([label, value]) => (
                <tr key={label}>
                  <td>{label}</td>
                  <td className="align-right">
                    {typeof value === "number" ? value.toFixed(6) : value}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

// --- Downloads / Export ---

function ModelDownloads({
  modelType,
  result,
  onShowMap,
  setLoadingMap,
  setLoadingFieldName,
  setSaveModalOpen,
  setSaveConfig,
  userSchema,
}) {
  const dl = result?.downloads || {};

  const normalize = (p) => {
    if (!p) return null;
    if (p.startsWith("/api")) return p;
    return `/api/ai-tools/download?file=${encodeURIComponent(p)}`;
  };

  const extractFilePath = (url) => {
    if (!url) return null;
    if (url.includes("/api/ai-tools/download?file=")) {
      const match = url.match(/file=([^&]+)/);
      return match ? decodeURIComponent(match[1]) : url;
    }
    return url;
  };

  const modelRawPath = dl.model_raw || extractFilePath(dl.model);
  const shapefileRawPath = dl.shapefile_raw || extractFilePath(dl.shapefile);

  const calculatePredictionRange = () => {
    const interactiveData = result?.interactive_data || {};
    const preds = interactiveData.preds || [];
    if (preds.length === 0) return null;
    return { min: Math.min(...preds), max: Math.max(...preds) };
  };

  const calculateActualRange = () => {
    const interactiveData = result?.interactive_data || {};
    const actuals = interactiveData.y_test || [];
    if (actuals.length === 0) return null;
    return { min: Math.min(...actuals), max: Math.max(...actuals) };
  };

  return (
    <div className="blgf-ai-card">
      <div className="blgf-ai-subtitle2">Export & Actions</div>

      <div className="blgf-ai-actions-grid">
        <ul className="blgf-ai-downloads">
          {dl.model && (
            <li>
              <a href={normalize(dl.model)} target="_blank">
                Model File (.pkl)
              </a>
            </li>
          )}
          {dl.report && (
            <li>
              <a href={normalize(dl.report)} target="_blank">
                PDF Report
              </a>
            </li>
          )}
          {dl.shapefile && (
            <li>
              <a href={normalize(dl.shapefile)} target="_blank">
                Shapefile (.zip)
              </a>
            </li>
          )}
          {dl.cama_csv && (
            <li>
              <a href={normalize(dl.cama_csv)} target="_blank">
                CAMA CSV
              </a>
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}

// --- LR Coefficients (Linear Regression only) ---

function LRCoefficientsSection({ result }) {
  const coeffs = result?.coefficients || [];
  const tTests = result?.t_test?.coefficients || [];
  const residualTest = result?.t_test?.residuals || {};

  return (
    <div className="blgf-ai-card">
      <div className="blgf-ai-subtitle2">Coefficients Analysis</div>

      {coeffs.length > 0 && (
        <>
          <div className="blgf-ai-subtitle3">Raw Coefficients</div>
          <table className="blgf-ai-table narrow">
            <thead>
              <tr>
                <th>Variable</th>
                <th className="align-right">Coefficient</th>
              </tr>
            </thead>
            <tbody>
              {coeffs.map((c, i) => (
                <tr key={i}>
                  <td>{c.variable}</td>
                  <td className="align-right">{c.value.toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {tTests.length > 0 && (
        <>
          <div className="blgf-ai-subtitle3">T-Tests</div>
          <table className="blgf-ai-table narrow">
            <thead>
              <tr>
                <th>Variable</th>
                <th className="align-right">Std Error</th>
                <th className="align-right">t-Value</th>
                <th className="align-right">p-Value</th>
              </tr>
            </thead>
            <tbody>
              {tTests.map((row, i) => (
                <tr key={i}>
                  <td>{row.variable}</td>
                  <td className="align-right">{row.std_err?.toFixed(6)}</td>
                  <td className="align-right">{row.t?.toFixed(6)}</td>
                  <td className="align-right">{row.p?.toExponential(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {residualTest?.t_stat !== undefined && (
        <>
          <div className="blgf-ai-subtitle3">Residual T-Test</div>
          <table className="blgf-ai-table narrow">
            <tbody>
              <tr>
                <td>t-Statistic</td>
                <td className="align-right">
                  {residualTest.t_stat.toFixed(6)}
                </td>
              </tr>
              <tr>
                <td>p-Value</td>
                <td className="align-right">
                  {residualTest.p_value.toExponential(3)}
                </td>
              </tr>
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

// --- SLM Coefficients & Spatial Diagnostics ---

function SLMCoefficientsSection({ result }) {
  const coeffs  = result?.coefficients || [];
  const rho     = result?.rho;
  const moranI  = result?.moran_i;
  const moranP  = result?.moran_p;
  const metrics = result?.metrics || {};

  return (
    <div className="blgf-ai-card">
      <div className="blgf-ai-subtitle2">Spatial Diagnostics</div>

      {/* Spatial summary badges */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
        {rho !== undefined && rho !== null && (
          <div style={{ background: "#fff7e6", border: "1px solid #f59e0b", borderRadius: 8, padding: "8px 14px" }}>
            <div style={{ fontSize: 11, color: "#92400e", fontWeight: 700, marginBottom: 2 }}>ρ (Spatial Lag)</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: "#b45309" }}>{rho.toFixed(4)}</div>
          </div>
        )}
        {moranI !== undefined && moranI !== null && (
          <div style={{ background: "#eff6ff", border: "1px solid #3b82f6", borderRadius: 8, padding: "8px 14px" }}>
            <div style={{ fontSize: 11, color: "#1d4ed8", fontWeight: 700, marginBottom: 2 }}>Moran's I (residuals)</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: "#1d4ed8" }}>{moranI.toFixed(4)}</div>
            {moranP !== null && (
              <div style={{ fontSize: 10, color: moranP < 0.05 ? "#dc2626" : "#6b7280" }}>
                p = {moranP.toFixed(4)} {moranP < 0.05 ? "★ significant" : ""}
              </div>
            )}
          </div>
        )}
        {metrics.pseudo_r2 !== undefined && (
          <div style={{ background: "#f0fdf4", border: "1px solid #22c55e", borderRadius: 8, padding: "8px 14px" }}>
            <div style={{ fontSize: 11, color: "#166534", fontWeight: 700, marginBottom: 2 }}>Pseudo R²</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: "#15803d" }}>{metrics.pseudo_r2.toFixed(4)}</div>
          </div>
        )}
      </div>

      {/* Coefficients table */}
      {coeffs.length > 0 && (
        <>
          <div className="blgf-ai-subtitle3">Coefficient Estimates (z-test)</div>
          <table className="blgf-ai-table narrow">
            <thead>
              <tr>
                <th>Variable</th>
                <th className="align-right">Coef</th>
                <th className="align-right">Std Err</th>
                <th className="align-right">z-stat</th>
                <th className="align-right">p-value</th>
                <th className="align-right">Sig.</th>
              </tr>
            </thead>
            <tbody>
              {coeffs.map((row, i) => (
                <tr key={i} style={row.significant ? { background: "rgba(34,197,94,0.06)" } : {}}>
                  <td>{row.variable}</td>
                  <td className="align-right">{row.coef?.toFixed(4)}</td>
                  <td className="align-right">{row.std_err?.toFixed(4)}</td>
                  <td className="align-right">{row.z?.toFixed(4)}</td>
                  <td className="align-right">{row.p?.toExponential(3)}</td>
                  <td className="align-right" style={{ color: row.significant ? "#16a34a" : "#94a3b8" }}>
                    {row.significant ? "★" : "—"}
                  </td>
                </tr>
              ))}
              {/* Rho row */}
              {rho !== undefined && (
                <tr style={{ background: "rgba(245,158,11,0.08)", fontWeight: 600 }}>
                  <td>W*y (ρ)</td>
                  <td className="align-right">{rho.toFixed(4)}</td>
                  <td className="align-right" colSpan={4} style={{ color: "#92400e", fontSize: 12 }}>
                    Spatial lag coefficient
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          <div style={{ fontSize: 11, color: "#6b7280", marginTop: 8 }}>
            ★ = significant at p &lt; 0.05 · Weights: Queen contiguity (row-standardized)
          </div>
        </>
      )}
    </div>
  );
}

// --- Hybrid SLM+RF Diagnostics Section ---

function HybridDiagnosticsSection({ result }) {
  const rho        = result?.rho;
  const moranISLM  = result?.moran_i_slm;
  const moranPSLM  = result?.moran_p_slm;
  const moranIHyb  = result?.moran_i_hybrid;
  const moranPHyb  = result?.moran_p_hybrid;
  const metrics    = result?.metrics || {};
  const coeffs     = result?.slm_coefficients || [];

  return (
    <div className="blgf-ai-card">
      <div className="blgf-ai-subtitle2">Spatial Diagnostics</div>

      {/* Badges */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
        {rho !== undefined && rho !== null && (
          <div style={{ background: "#fff7e6", border: "1px solid #f59e0b", borderRadius: 8, padding: "8px 14px" }}>
            <div style={{ fontSize: 11, color: "#92400e", fontWeight: 700, marginBottom: 2 }}>ρ — Stage 1 SLM</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: "#b45309" }}>{rho.toFixed(4)}</div>
          </div>
        )}
        {moranISLM !== undefined && moranISLM !== null && (
          <div style={{ background: "#eff6ff", border: "1px solid #3b82f6", borderRadius: 8, padding: "8px 14px" }}>
            <div style={{ fontSize: 11, color: "#1d4ed8", fontWeight: 700, marginBottom: 2 }}>Moran's I — SLM residuals</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: "#1d4ed8" }}>{moranISLM.toFixed(4)}</div>
            {moranPSLM !== null && (
              <div style={{ fontSize: 10, color: moranPSLM < 0.05 ? "#dc2626" : "#6b7280" }}>
                p = {moranPSLM.toFixed(4)} {moranPSLM < 0.05 ? "★ significant" : ""}
              </div>
            )}
          </div>
        )}
        {moranIHyb !== undefined && moranIHyb !== null && (
          <div style={{ background: "#f5f3ff", border: "1px solid #7c3aed", borderRadius: 8, padding: "8px 14px" }}>
            <div style={{ fontSize: 11, color: "#5b21b6", fontWeight: 700, marginBottom: 2 }}>Moran's I — Hybrid residuals</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: "#7c3aed" }}>{moranIHyb.toFixed(4)}</div>
            {moranPHyb !== null && (
              <div style={{ fontSize: 10, color: moranPHyb < 0.05 ? "#dc2626" : "#16a34a" }}>
                p = {moranPHyb.toFixed(4)} {moranPHyb < 0.05 ? "★ still significant" : "✓ not significant"}
              </div>
            )}
          </div>
        )}
        {metrics.r2_slm !== undefined && (
          <div style={{ background: "#f0fdf4", border: "1px solid #22c55e", borderRadius: 8, padding: "8px 14px" }}>
            <div style={{ fontSize: 11, color: "#166534", fontWeight: 700, marginBottom: 2 }}>R² improvement</div>
            <div style={{ fontSize: 18, fontWeight: 800, color: "#15803d" }}>
              +{((metrics.r2 - metrics.r2_slm) * 100).toFixed(2)}%
            </div>
            <div style={{ fontSize: 10, color: "#6b7280" }}>Hybrid vs SLM alone</div>
          </div>
        )}
      </div>

      {/* SLM Stage 1 coefficients */}
      {coeffs.length > 0 && (
        <>
          <div className="blgf-ai-subtitle3">Stage 1 — SLM Coefficients (z-test)</div>
          <table className="blgf-ai-table narrow">
            <thead>
              <tr>
                <th>Variable</th>
                <th className="align-right">Coef</th>
                <th className="align-right">Std Err</th>
                <th className="align-right">z-stat</th>
                <th className="align-right">p-value</th>
                <th className="align-right">Sig.</th>
              </tr>
            </thead>
            <tbody>
              {coeffs.map((row, i) => (
                <tr key={i} style={row.significant ? { background: "rgba(34,197,94,0.06)" } : {}}>
                  <td>{row.variable}</td>
                  <td className="align-right">{row.coef?.toFixed(4)}</td>
                  <td className="align-right">{row.std_err?.toFixed(4)}</td>
                  <td className="align-right">{row.z?.toFixed(4)}</td>
                  <td className="align-right">{row.p?.toExponential(3)}</td>
                  <td className="align-right" style={{ color: row.significant ? "#16a34a" : "#94a3b8" }}>
                    {row.significant ? "★" : "—"}
                  </td>
                </tr>
              ))}
              {rho !== undefined && (
                <tr style={{ background: "rgba(245,158,11,0.08)", fontWeight: 600 }}>
                  <td>W*y (ρ)</td>
                  <td className="align-right">{rho.toFixed(4)}</td>
                  <td className="align-right" colSpan={4} style={{ color: "#92400e", fontSize: 12 }}>
                    Spatial lag coefficient
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          <div style={{ fontSize: 11, color: "#6b7280", marginTop: 8 }}>
            ★ = significant at p &lt; 0.05 · Stage 2 RF feature importances shown above
          </div>
        </>
      )}
    </div>
  );
}

// --- Feature Importance ---

function ImportanceSection({ modelType, result }) {
  const imp = result?.importance || [];

  if (!imp.length)
    return (
      <div className="blgf-ai-card">
        <div className="blgf-ai-subtitle2">Feature Importance</div>
        <div className="blgf-ai-empty-text">
          No feature importance available.
        </div>
      </div>
    );

  const features = imp.map((i) => i.feature);
  const values = imp.map((i) => i.value);

  return (
    <div className="blgf-ai-card">
      <div className="blgf-ai-chart-container">
        <div className="blgf-ai-chart-title">
          Feature Importance ({modelType.toUpperCase()})
        </div>
        <table className="blgf-ai-table narrow mb-4">
          <thead>
            <tr>
              <th>Feature</th>
              <th className="align-right">Importance</th>
            </tr>
          </thead>
          <tbody>
            {imp.map((row, i) => (
              <tr key={i}>
                <td>{row.feature}</td>
                <td className="align-right">{row.value.toFixed(6)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Plot
          data={[
            {
              type: "bar",
              orientation: "h",
              x: values,
              y: features,
              marker: getFeatureImportanceMarker(values),
            },
          ]}
          layout={{
            height: features.length * 35 + 80,
            margin: { l: 150, r: 20, t: 30, b: 40 },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            font: { color: "#e2e8f0", family: FONT_FAMILY },
            xaxis: {
              gridcolor: "#334155",
              title: getAxisTitle("Importance", "#f7c800"),
              tickfont: { family: FONT_FAMILY },
            },
            yaxis: {
              gridcolor: "rgba(0,0,0,0)",
              tickfont: { color: "#cbd5e1", family: FONT_FAMILY },
            },
          }}
          config={PLOT_CONFIG}
          style={{ width: "100%" }}
        />
      </div>
    </div>
  );
}

// --- Plots (Actual vs Predicted + Residual Distribution) ---

function PlotsSection({ modelType, result }) {
  const d = result?.interactive_data || {};
  const y = d.y_test || [];
  const preds = d.preds || [];
  const bins = d.residual_bins || [];
  const binCounts = d.residual_counts || [];

  return (
    <>
      <div className="blgf-ai-card">
        <div className="blgf-ai-chart-container">
          <div className="blgf-ai-chart-title">Actual vs Predicted</div>
          <Plot
            data={[
              {
                x: y,
                y: preds,
                mode: "markers",
                type: "scatter",
                name: "Predictions",
                marker: SCATTER_MARKER.predictions,
              },
              getDashedLine(
                Math.min(...y),
                Math.max(...y),
                Math.min(...y),
                Math.max(...y),
                "Actual Value",
              ),
            ]}
            layout={{
              ...SCATTER_LAYOUT,
              xaxis: {
                ...SCATTER_LAYOUT.xaxis,
                title: getAxisTitle("Actual Values"),
              },
              yaxis: {
                ...SCATTER_LAYOUT.yaxis,
                title: getAxisTitle("Predicted Values"),
              },
            }}
            config={PLOT_CONFIG}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      <div className="blgf-ai-card">
        <div className="blgf-ai-chart-container">
          <div className="blgf-ai-chart-title">Residual Distribution</div>
          <Plot
            data={[
              {
                x: bins,
                y: binCounts,
                type: "bar",
                name: "Residual Frequency",
                marker: BAR_MARKER.residual,
              },
            ]}
            layout={{
              ...BAR_LAYOUT,
              xaxis: {
                ...BAR_LAYOUT.xaxis,
                title: getAxisTitle("Residual"),
              },
              yaxis: {
                ...BAR_LAYOUT.yaxis,
                title: getAxisTitle("Frequency"),
              },
            }}
            config={PLOT_CONFIG}
            style={{ width: "100%" }}
          />
        </div>
      </div>
    </>
  );
}

// --- Variable Distributions ---

function VariableDistributions({ modelType, result }) {
  const dist = result?.variable_distributions || {};
  const vars = Object.keys(dist);

  if (!vars.length)
    return (
      <div className="blgf-ai-card">
        <div className="blgf-ai-subtitle2">Variable Distributions</div>
        <div className="blgf-ai-empty-text">
          No distribution data available.
        </div>
      </div>
    );

  return (
    <>
      {vars.map((v) => {
        const varData = dist[v];

        if (!varData || !varData.bins || !varData.counts) {
          return (
            <div className="blgf-ai-card" key={v}>
              <div className="blgf-ai-subtitle2">{v}</div>
              <div className="blgf-ai-empty-text">No data available</div>
            </div>
          );
        }

        return (
          <div className="blgf-ai-card" key={v}>
            <div className="blgf-ai-chart-container">
              <div className="blgf-ai-chart-title">Distribution of {v}</div>

              <div className="blgf-ai-stats-row">
                <div className="blgf-ai-stat-badge">
                  <strong>Mean:</strong> {varData.mean?.toFixed(2) || "N/A"}
                </div>
                <div className="blgf-ai-stat-badge">
                  <strong>Median:</strong> {varData.median?.toFixed(2) || "N/A"}
                </div>
                <div className="blgf-ai-stat-badge">
                  <strong>Std Dev:</strong> {varData.std?.toFixed(2) || "N/A"}
                </div>
                {varData.count && (
                  <div className="blgf-ai-stat-badge">
                    <strong>Samples:</strong> {varData.count}
                  </div>
                )}
              </div>

              <Plot
                data={[
                  {
                    x: varData.bins,
                    y: varData.counts,
                    type: "bar",
                    name: v,
                    marker: BAR_MARKER.distribution,
                  },
                ]}
                layout={{
                  ...DISTRIBUTION_LAYOUT,
                  xaxis: {
                    ...DISTRIBUTION_LAYOUT.xaxis,
                    title: getAxisTitle(v, "#f7c800"),
                  },
                  yaxis: {
                    ...DISTRIBUTION_LAYOUT.yaxis,
                    title: getAxisTitle("Frequency", "#f7c800"),
                  },
                }}
                config={PLOT_CONFIG}
                style={{ width: "100%" }}
              />
            </div>
          </div>
        );
      })}
    </>
  );
}

// --- CAMA Preview Table ---

function ModelCAMA({ result }) {
  const rows = result?.cama_preview || [];
  if (!rows.length)
    return (
      <div className="blgf-ai-card">
        <div className="blgf-ai-subtitle2">Training Result Preview</div>
        <div className="blgf-ai-empty-text">
          No training data preview available.
        </div>
      </div>
    );
  const columns = Object.keys(rows[0]);
  return (
    <div className="blgf-ai-card">
      <div className="blgf-ai-subtitle2">Training Result Preview</div>
      <div className="blgf-ai-table-wrap" style={{ maxHeight: "260px" }}>
        <table className="blgf-ai-table narrow">
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c}>{String(r[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}