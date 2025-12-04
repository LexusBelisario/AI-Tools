import React, { useState } from "react";
import "./LinearRegression.css";
import Plot from "react-plotly.js";
import API from "../../../api.js";

const LinearRegression = ({ onClose }) => {
  const [files, setFiles] = useState([]);
  const [fields, setFields] = useState([]);
  const [independentVars, setIndependentVars] = useState([]);
  const [dependentVar, setDependentVar] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showRunModal, setShowRunModal] = useState(false);
  const [modelFile, setModelFile] = useState(null);
  const [runFiles, setRunFiles] = useState([]);
  const [showResultsPanel, setShowResultsPanel] = useState(false);
  const [selectedGraph, setSelectedGraph] = useState(null);
  const [fullscreenGraph, setFullscreenGraph] = useState(null);

  // === Handle file upload ===
  const handleFileChange = async (e) => {
    const selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length === 0) return;

    setFiles(selectedFiles);
    setFields([]);
    setIndependentVars([]);
    setDependentVar("");
    setResult(null);

    try {
      const hasZip = selectedFiles.some((f) =>
        f.name.toLowerCase().endsWith(".zip")
      );
      const hasParts = selectedFiles.some((f) =>
        [".shp", ".dbf", ".shx", ".prj"].some((ext) =>
          f.name.toLowerCase().endsWith(ext)
        )
      );

      if (hasZip && hasParts) {
        alert(
          "Please select only a single ZIP file or a complete shapefile set."
        );
        return;
      }

      const formData = new FormData();
      let endpoint;
      if (hasZip) {
        if (selectedFiles.length > 1) {
          alert(
            "Multiple ZIP files detected. Please upload only one ZIP file."
          );
          return;
        }
        formData.append("zip_file", selectedFiles[0]);
        endpoint = `${API}/linear-regression/fields-zip`;
      } else {
        selectedFiles.forEach((f) => formData.append("shapefiles", f));
        endpoint = `${API}/linear-regression/fields`;
      }

      const res = await fetch(endpoint, { method: "POST", body: formData });
      const data = await res.json();
      if (res.ok && data.fields) setFields(data.fields);
      else alert(data.error || "Unable to extract fields.");
    } catch (err) {
      console.error("Error reading shapefile fields:", err);
      alert("Error reading shapefile fields. See console for details.");
    }
  };

  const toggleIndependentVar = (field) => {
    setIndependentVars((prev) =>
      prev.includes(field) ? prev.filter((f) => f !== field) : [...prev, field]
    );
  };

  // === Train model ===
  const handleTrainModel = async () => {
    if (files.length === 0)
      return alert("Please upload your shapefile or ZIP.");
    if (independentVars.length === 0)
      return alert("Select independent variables.");
    if (!dependentVar) return alert("Select dependent variable.");

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    let endpoint;

    if (files.length === 1 && files[0].name.toLowerCase().endsWith(".zip")) {
      formData.append("zip_file", files[0]);
      endpoint = `${API}/linear-regression/train-zip`;
    } else {
      files.forEach((f) => formData.append("shapefiles", f));
      endpoint = `${API}/linear-regression/train`;
    }

    formData.append("independent_vars", JSON.stringify(independentVars));
    formData.append("dependent_var", dependentVar);

    try {
      const res = await fetch(endpoint, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) {
        console.error("Training error:", data);
        alert(`Error: ${data.error || res.statusText}`);
      } else {
        setResult(data);
      }
    } catch (err) {
      console.error("Training fetch error:", err);
      alert("Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  // === Run saved model ===
  const handleRunModel = async () => {
    if (!modelFile) return alert("Please select a .pkl model file.");
    if (runFiles.length === 0)
      return alert("Please upload a shapefile or ZIP.");

    const hasZip = runFiles.some((f) => f.name.toLowerCase().endsWith(".zip"));
    const hasParts = runFiles.some((f) =>
      [".shp", ".dbf", ".shx", ".prj"].some((ext) =>
        f.name.toLowerCase().endsWith(ext)
      )
    );
    if (hasZip && hasParts) {
      alert(
        "Multiple file types detected. Please select only a ZIP or shapefile set."
      );
      return;
    }

    const formData = new FormData();
    formData.append("model_file", modelFile);
    if (hasZip) formData.append("zip_file", runFiles[0]);
    else runFiles.forEach((f) => formData.append("shapefiles", f));

    setLoading(true);
    setResult(null);

    try {
      const res = await fetch(`${API}/linear-regression/run-saved-model`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        console.error("Run Model Error:", data);
        alert(`Error: ${data.error || res.statusText}`);
      } else setResult(data);
    } catch (err) {
      console.error("Run Model Fetch Error:", err);
      alert("Failed to connect to backend.");
    } finally {
      setLoading(false);
      setShowRunModal(false);
      setModelFile(null);
      setRunFiles([]);
    }
  };

  // === Plotly defaults ===
  const plotConfig = (filename) => ({
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    scrollZoom: true,
    toImageButtonOptions: { format: "png", filename },
    modeBarButtonsToRemove: ["select2d", "lasso2d"],
  });

  const plotLayoutBase = {
    paper_bgcolor: "#000",
    plot_bgcolor: "#000",
    font: { color: "white" },
    hoverlabel: {
      bgcolor: "#111",
      bordercolor: "#00ff9d",
      font: { color: "white" },
    },
    margin: { l: 60, r: 30, t: 60, b: 60 },
  };

  return (
    <div className="lr-overlay">
      <div className="lr-panel">
        <div className="lr-header">
          <h3>Linear Regression Tool</h3>
          <button className="lr-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="lr-content">
          {/* === Upload + Variables === */}
          <label>Upload Shapefile (.shp, .dbf, .shx, .prj) or ZIP</label>
          <div className="file-upload">
            <input
              type="file"
              id="shpInput"
              accept=".shp,.dbf,.shx,.prj,.zip"
              multiple
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            <button
              className="choose-file-btn"
              onClick={() => document.getElementById("shpInput").click()}
            >
              Choose Files
            </button>
            <span className="file-name">
              {files.length > 0
                ? files.map((f) => f.name).join(", ")
                : "No files chosen"}
            </span>
          </div>

          <hr className="divider" />

          <label>Independent Variables</label>
          <div className="checkbox-list">
            {fields.length > 0 ? (
              fields.map((f) => (
                <label key={f} className="checkbox-item">
                  <input
                    type="checkbox"
                    checked={independentVars.includes(f)}
                    onChange={() => toggleIndependentVar(f)}
                  />
                  {f}
                </label>
              ))
            ) : (
              <p className="placeholder-text">No fields loaded yet.</p>
            )}
          </div>

          <label>Dependent Variable</label>
          <select
            value={dependentVar}
            onChange={(e) => setDependentVar(e.target.value)}
          >
            <option value="">-- Select --</option>
            {fields.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>

          <div className="lr-buttons">
            <button
              className="run-btn"
              onClick={handleTrainModel}
              disabled={loading}
            >
              {loading ? "Training..." : "Train Model"}
            </button>
            <button
              className="run-btn secondary"
              onClick={() => setShowRunModal(true)}
              disabled={loading}
            >
              Run Saved Model
            </button>
          </div>

          {/* === Model Summary === */}
          {result && (
            <div className="model-summary-box">
              <h3 className="summary-title">🧠 Model Summary</h3>
              <p className="summary-sub">
                Dependent Variable: <span>{result.dependent_var}</span>
              </p>
              <table className="summary-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.metrics || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{v.toFixed(6)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <h4 className="coef-header">Regression Coefficients</h4>
              <table className="coef-table">
                <thead>
                  <tr>
                    <th>Variable</th>
                    <th>Coefficient</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.coefficients || {}).map(([v, c]) => (
                    <tr key={v}>
                      <td>{v}</td>
                      <td>{c.toFixed(6)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <p className="intercept-line">
                Intercept: {result.intercept.toFixed(6)}
              </p>
              {result?.t_test && (
                <div className="t-test-section mt-4">
                  <h3 className="text-green-400 text-lg font-semibold mb-1">
                    🧮 T-test on Residuals
                  </h3>
                  <table className="w-full border-collapse text-sm bg-black/30 text-white">
                    <thead>
                      <tr className="bg-black/50 text-green-300">
                        <th className="border border-green-700 p-1">Metric</th>
                        <th className="border border-green-700 p-1">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border border-green-700 p-1">
                          T-statistic
                        </td>
                        <td className="border border-green-700 p-1">
                          {result.t_test.t_stat?.toFixed(4)}
                        </td>
                      </tr>
                      <tr>
                        <td className="border border-green-700 p-1">P-value</td>
                        <td className="border border-green-700 p-1">
                          {result.t_test.p_value?.toFixed(4)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}

              {/* Downloads */}
              <div className="download-links">
                <h4>Downloads</h4>
                <ul>
                  <li>
                    <a
                      href={result.downloads.model}
                      target="_blank"
                      rel="noreferrer"
                    >
                      📦 Model (.pkl)
                    </a>
                  </li>
                  <li>
                    <a
                      href={result.downloads.report}
                      target="_blank"
                      rel="noreferrer"
                    >
                      📄 PDF Report
                    </a>
                  </li>
                  <li>
                    <a
                      href={result.downloads.shapefile}
                      target="_blank"
                      rel="noreferrer"
                    >
                      🗺️ Predicted Shapefile (.zip)
                    </a>
                  </li>
                </ul>
              </div>

              <div className="graphs-button-container">
                <button
                  className="show-graphs-btn"
                  onClick={() => setShowResultsPanel(!showResultsPanel)}
                >
                  {showResultsPanel
                    ? "Hide Graphs & Tables"
                    : "Show Graphs & Tables"}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* === Graphs Modal === */}
      {showResultsPanel && result && (
        <div className="graphs-modal">
          <div className="graphs-modal-content">
            <button
              className="graphs-close"
              onClick={() => setShowResultsPanel(false)}
            >
              ✕
            </button>
            <h2 className="graphs-title">📊 Linear Regression Results</h2>
            <p className="graphs-subtitle">
              Interactive model performance & diagnostics dashboard
            </p>

            {/* === Graph Grid === */}
            <div className="graphs-grid">
              {[
                { key: "importance", title: "Feature Importance" },
                { key: "residuals", title: "Residual Distribution" },
                { key: "actual_pred", title: "Actual vs Predicted" },
                { key: "resid_pred", title: "Residuals vs Predicted" },
              ].map((g) => (
                <div
                  key={g.key}
                  className="graph-card"
                  onClick={() => setFullscreenGraph(g.key)}
                >
                  <h4>{g.title}</h4>
                  {g.key === "importance" && (
                    <Plot
                      data={[
                        {
                          x: Object.keys(result.interactive_data.importance),
                          y: Object.values(result.interactive_data.importance),
                          type: "bar",
                          marker: { color: "#00ff9d" },
                        },
                      ]}
                      layout={{ ...plotLayoutBase, title: g.title }}
                      config={plotConfig(g.key)}
                      useResizeHandler
                      style={{ width: "100%", height: "100%" }}
                    />
                  )}
                  {g.key === "residuals" && (
                    <Plot
                      data={[
                        {
                          type: "bar",
                          x: result.interactive_data.residual_bins,
                          y: result.interactive_data.residual_counts,
                          marker: {
                            color: "#00ff9d",
                            opacity: 0.85,
                            line: { color: "#0f0f0f", width: 1.2 },
                          },
                          // 🟩 Slightly thinner bars to give visible spacing even on zoom
                          width:
                            0.6 *
                            ((Math.max(
                              ...result.interactive_data.residual_bins
                            ) -
                              Math.min(
                                ...result.interactive_data.residual_bins
                              )) /
                              result.interactive_data.residual_bins.length),
                        },
                      ]}
                      layout={{
                        ...plotLayoutBase,
                        title: {
                          text: "Residual Distribution (with spacing)",
                          font: { color: "#00ff9d" },
                        },
                        xaxis: {
                          title: "Residual",
                          showgrid: true,
                          gridcolor: "#222",
                        },
                        yaxis: { title: "Frequency" },
                        bargap: 0.35, // 💡 slightly wider gaps between bars
                        bargroupgap: 0.25,
                        autosize: true,
                        responsive: true,
                      }}
                      config={{
                        ...plotConfig("residual_distribution"),
                        scrollZoom: true,
                      }}
                      useResizeHandler
                      style={{ width: "100%", height: "100%" }}
                    />
                  )}

                  {g.key === "actual_pred" && (
                    <Plot
                      data={[
                        {
                          x: result.interactive_data.y_test,
                          y: result.interactive_data.preds,
                          mode: "markers",
                          type: "scatter",
                          marker: { color: "#00ff9d", size: 8, opacity: 0.8 },
                        },
                        {
                          x: result.interactive_data.y_test,
                          y: result.interactive_data.y_test,
                          mode: "lines",
                          line: { color: "white", dash: "dash" },
                        },
                      ]}
                      layout={{ ...plotLayoutBase, title: g.title }}
                      config={plotConfig(g.key)}
                      useResizeHandler
                      style={{ width: "100%", height: "100%" }}
                    />
                  )}
                  {g.key === "resid_pred" && (
                    <Plot
                      data={[
                        {
                          x: result.interactive_data.preds,
                          y: result.interactive_data.residuals,
                          mode: "markers",
                          type: "scatter",
                          marker: { color: "#ff6363", size: 8, opacity: 0.8 },
                        },
                        {
                          x: result.interactive_data.preds,
                          y: Array(result.interactive_data.preds?.length).fill(
                            0
                          ),
                          mode: "lines",
                          line: { color: "white", dash: "dash" },
                        },
                      ]}
                      layout={{ ...plotLayoutBase, title: g.title }}
                      config={plotConfig(g.key)}
                      useResizeHandler
                      style={{ width: "100%", height: "100%" }}
                    />
                  )}
                </div>
              ))}
            </div>

            {/* === CAMA PREVIEW === */}
            {result.cama_preview && (
              <div className="cama-preview-modal">
                <h3 className="cama-header-modal">
                  🏠 CAMA Attribute Table (Preview)
                </h3>
                <p className="graphs-subtitle">
                  Click a column name to see its data distribution.
                </p>
                <div className="cama-table-wrapper-modal">
                  <table className="cama-table-modal">
                    <thead>
                      <tr>
                        {Object.keys(result.cama_preview[0] || {}).map(
                          (col) => (
                            <th
                              key={col}
                              onClick={() => {
                                const columnValues = result.cama_preview
                                  .map((row) => parseFloat(row[col]))
                                  .filter((v) => !isNaN(v)); // only numeric
                                if (columnValues.length > 0) {
                                  setSelectedGraph({
                                    title: `Distribution of ${col}`,
                                    column: col,
                                    values: columnValues,
                                  });
                                } else {
                                  alert(
                                    `No numeric data found for column: ${col}`
                                  );
                                }
                              }}
                              style={{ cursor: "pointer" }}
                            >
                              {col}
                              {result.distributions?.[col] && (
                                <span
                                  style={{ fontSize: "12px", color: "#00ff9d" }}
                                >
                                  {" "}
                                  📊
                                </span>
                              )}
                            </th>
                          )
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {result.cama_preview.map((row, i) => (
                        <tr key={i}>
                          {Object.entries(row).map(([col, val]) => (
                            <td
                              key={col}
                              className={
                                col === "prediction" ? "prediction-col" : ""
                              }
                            >
                              {val !== "" ? val : "—"}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="cama-download-container">
                  <a
                    className="download-csv-btn"
                    href={result.downloads.cama_csv}
                    target="_blank"
                    rel="noreferrer"
                  >
                    ⬇️ Download Full CAMA Table (CSV)
                  </a>
                </div>
              </div>
            )}

            {/* === Popup distribution viewer === */}
            {selectedGraph && (
              <div
                className="graph-viewer-overlay"
                onClick={() => setSelectedGraph(null)}
              >
                <div
                  className="graph-viewer-box"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    className="graph-viewer-close"
                    onClick={() => setSelectedGraph(null)}
                  >
                    ✕
                  </button>
                  <h3 style={{ color: "#00ff9d" }}>{selectedGraph.title}</h3>
                  {/* 🧠 Render Plotly histogram dynamically */}
                  <Plot
                    data={[
                      {
                        x: selectedGraph.values,
                        type: "histogram",
                        histnorm: "probability density",
                        marker: { color: "#00ff9d", opacity: 0.75 },
                      },
                    ]}
                    layout={{
                      paper_bgcolor: "#000",
                      plot_bgcolor: "#000",
                      font: { color: "white" },
                      hoverlabel: {
                        bgcolor: "#111",
                        bordercolor: "#00ff9d",
                        font: { color: "white" },
                      },
                      margin: { l: 50, r: 30, t: 60, b: 60 },
                      xaxis: { title: selectedGraph.column },
                      yaxis: { title: "Density" },
                      bargap: 0.3,
                    }}
                    config={{
                      responsive: true,
                      displaylogo: false,
                      scrollZoom: true,
                      toImageButtonOptions: {
                        format: "png",
                        filename: selectedGraph.column,
                      },
                    }}
                    useResizeHandler
                    style={{ width: "100%", height: "70vh" }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* === Fullscreen Chart === */}
      {fullscreenGraph && (
        <div
          className="graph-viewer-overlay"
          onClick={() => setFullscreenGraph(null)}
        >
          <div
            className="graph-viewer-box"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="graph-viewer-close"
              onClick={() => setFullscreenGraph(null)}
            >
              ✕
            </button>
            <h3 style={{ color: "#00ff9d" }}>
              {fullscreenGraph.replace("_", " ").toUpperCase()}
            </h3>
            <Plot
              data={
                fullscreenGraph === "importance"
                  ? [
                      {
                        x: Object.keys(result.interactive_data.importance),
                        y: Object.values(result.interactive_data.importance),
                        type: "bar",
                        marker: { color: "#00ff9d" },
                      },
                    ]
                  : fullscreenGraph === "residuals"
                  ? [
                      {
                        type: "bar",
                        x: result.interactive_data.residual_bins,
                        y: result.interactive_data.residual_counts,
                        marker: {
                          color: "#00ff9d",
                          opacity: 0.85,
                          line: { color: "#0f0f0f", width: 1.2 },
                        },
                        // ✅ keep the same width ratio and spacing
                        width:
                          0.6 *
                          ((Math.max(...result.interactive_data.residual_bins) -
                            Math.min(
                              ...result.interactive_data.residual_bins
                            )) /
                            result.interactive_data.residual_bins.length),
                      },
                    ]
                  : fullscreenGraph === "actual_pred"
                  ? [
                      {
                        x: result.interactive_data.y_test,
                        y: result.interactive_data.preds,
                        mode: "markers",
                        type: "scatter",
                        marker: { color: "#00ff9d", size: 10 },
                      },
                      {
                        x: result.interactive_data.y_test,
                        y: result.interactive_data.y_test,
                        mode: "lines",
                        line: { color: "white", dash: "dash" },
                      },
                    ]
                  : [
                      {
                        x: result.interactive_data.preds,
                        y: result.interactive_data.residuals,
                        mode: "markers",
                        type: "scatter",
                        marker: { color: "#ff6363", size: 10 },
                      },
                      {
                        x: result.interactive_data.preds,
                        y: Array(result.interactive_data.preds?.length).fill(0),
                        mode: "lines",
                        line: { color: "white", dash: "dash" },
                      },
                    ]
              }
              layout={{
                ...plotLayoutBase,
                title: `${fullscreenGraph
                  .replace("_", " ")
                  .toUpperCase()} (Fullscreen)`,
              }}
              config={plotConfig(`${fullscreenGraph}_full`)}
              useResizeHandler
              style={{ width: "100%", height: "80vh" }}
            />
          </div>
        </div>
      )}

      {/* === Run Model Modal === */}
      {showRunModal && (
        <div className="lr-modal">
          <div className="lr-modal-content">
            <h4>Run Saved Model</h4>
            <label>Upload Model (.pkl)</label>
            <input
              type="file"
              accept=".pkl"
              onChange={(e) => setModelFile(e.target.files[0])}
            />
            <label>Upload Shapefile (.zip or .shp/.dbf/.shx/.prj)</label>
            <input
              type="file"
              accept=".zip,.shp,.dbf,.shx,.prj"
              multiple
              onChange={(e) => setRunFiles(Array.from(e.target.files))}
            />
            <div className="lr-modal-buttons">
              <button onClick={handleRunModel} disabled={loading}>
                {loading ? "Running..." : "Run"}
              </button>
              <button
                className="cancel-btn"
                onClick={() => setShowRunModal(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LinearRegression;
