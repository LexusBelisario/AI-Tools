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
  const hasLRError = !hasLR && !!trainErrors.lr;
  const hasRFError = !hasRF && !!trainErrors.rf;
  const hasXGBError = !hasXGB && !!trainErrors.xgb;

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
    modelType === "lr"
      ? "Linear Regression"
      : modelType === "rf"
        ? "Random Forest"
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