import React, { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import API from "../../api.js";
import TrainingLoader from "./components/trainingLoader.jsx";
import MapLoader from "./components/MapLoader.jsx";
import SaveToDBModal from "./SaveToDBModal.jsx";
import "./components/aitoolsmodal.css";
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
import RunSavedTabUI from "./RunSavedTabUI.jsx";

export default function AIToolsModal({
  isOpen,
  onClose,
  onShowMap,
  schema: externalSchema = null,
  token = "",
  shouldDisconnect = false,
}) {
  const [availableTables, setAvailableTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState("");
  const [fields, setFields] = useState([]);
  const [dependentVar, setDependentVar] = useState("");
  const [independentVars, setIndependentVars] = useState([]);
  const [excludedIndices, setExcludedIndices] = useState([]);
  const [previewRows, setPreviewRows] = useState([]);
  const [previewTotal, setPreviewTotal] = useState(0);
  const [previewPage, setPreviewPage] = useState(1);
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [saveConfig, setSaveConfig] = useState(null);

  const PAGE_SIZE = 100;

  const [activeTab, setActiveTab] = useState("inputs");

  const [modelChecks, setModelChecks] = useState({
    lr: false,
    rf: false,
    xgb: false,
  });

  const [results, setResults] = useState({
    lr: null,
    rf: null,
    xgb: null,
  });

  const [activeModelTab, setActiveModelTab] = useState(null);
  const [training, setTraining] = useState(false);
  const [loadingMap, setLoadingMap] = useState(false);
  const [loadingFieldName, setLoadingFieldName] = useState("");

  const decodeJwtPayload = (tok) => {
    try {
      const part = tok.split(".")[1];
      const b64 = part.replace(/-/g, "+").replace(/_/g, "/");
      const json = decodeURIComponent(
        atob(b64)
          .split("")
          .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
          .join("")
      );
      return JSON.parse(json);
    } catch {
      return {};
    }
  };

  const tokenPayload = decodeJwtPayload(token);

  // try to find db in token payload (adjust keys depending on your GIS token)
  const userDb =
    tokenPayload.db ||
    tokenPayload.db_name ||
    tokenPayload.dbname ||
    tokenPayload.prov_dbname ||
    null;

  const authFetch = (url, options = {}) => {
    const headers = { ...(options.headers || {}) };
    if (token) headers.Authorization = `Bearer ${token}`;

    if (userSchema) headers["X-Target-Schema"] = userSchema;
    if (userDb) headers["X-Target-DB"] = userDb;

    return fetch(url, { ...options, headers });
  };

  const [commonStatus, setCommonStatus] = useState({
    connected: false,
    context: null,
  });

  const [commonBusy, setCommonBusy] = useState(false);
  const [commonError, setCommonError] = useState("");

  const resolvedSchema = commonStatus?.context?.schema || null;
  const userSchema = externalSchema || resolvedSchema;

  const loadCommonStatus = async () => {
    if (!token) {
      setCommonStatus({ connected: false, context: null });
      return;
    }
    try {
      setCommonError("");
      const res = await authFetch(`${API}/common/status`);
      const data = await res.json();
      setCommonStatus(data);
    } catch (e) {
      setCommonStatus({ connected: false, context: null });
      setCommonError("Unable to check common connection status.");
    }
  };

  const connectCommon = async () => {
    if (!token) {
      setCommonError("No token received.");
      return;
    }

    setCommonBusy(true);
    setCommonError("");

    console.log("🔄 Connecting with schema:", externalSchema); // 👈 ADD THIS

    try {
      const res = await authFetch(`${API}/common/connect`, { method: "POST" });

      let data = null;
      try {
        data = await res.json();
      } catch {
        data = { detail: await res.text() };
      }

      if (!res.ok) {
        console.log("❌ CONNECT ERROR:", data);
        throw new Error(data?.detail || "Connect failed");
      }

      console.log("✅ Connected to:", data.context); // 👈 ADD THIS
      setCommonStatus(data);
    } catch (e) {
      setCommonStatus({ connected: false, context: null });
      setCommonError(e.message || "Connect failed");
    } finally {
      setCommonBusy(false);
    }
  };

  const disconnectCommon = async () => {
    setCommonStatus({ connected: false, context: null });
    setCommonError("");
  };

  const loadDatabasePreview = async (page) => {
    if (!selectedTable || !userSchema) return;

    try {
      setPreviewPage(page);

      const fd = new FormData();
      fd.append("schema", userSchema);
      fd.append("table_name", selectedTable);
      fd.append("limit", PAGE_SIZE);
      fd.append("offset", (page - 1) * PAGE_SIZE);

      const res = await authFetch(`${API}/ai-tools/preview-db`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const errorText = await res.text();
        alert(`Server error: ${res.status} - ${errorText}`);
        return;
      }

      const data = await res.json();
      setPreviewRows(data.rows || []);
      setPreviewTotal(data.total || 0);
    } catch (err) {
      alert("Preview failed: " + err.message);
    }
  };

  const loadAvailableTables = async () => {
    if (!userSchema) return;

    try {
      const fd = new FormData();
      fd.append("schema", userSchema);

      const res = await authFetch(`${API}/ai-tools/list-tables`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        await res.text();
        alert(`Failed to load tables: ${res.status}`);
        return;
      }

      const data = await res.json();

      if (data.tables && data.tables.length > 0) {
        setAvailableTables(data.tables);
      } else {
        setAvailableTables([]);
        alert("No Training_Table found in this schema.");
      }
    } catch (err) {
      alert(`Failed to load tables: ${err.message}`);
    }
  };

  const loadTableFields = async () => {
    if (!selectedTable || !userSchema) return;

    try {
      const fd = new FormData();
      fd.append("schema", userSchema);
      fd.append("table_name", selectedTable);

      const res = await authFetch(`${API}/ai-tools/fields-db`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const errorText = await res.text();
        alert(`Server error: ${res.status} - ${errorText}`);
        return;
      }

      const data = await res.json();

      if (data.fields) {
        setFields(data.fields);
      } else {
        alert("No fields found in the table.");
      }
    } catch (err) {
      alert(`Failed to load fields: ${err.message}`);
    }
  };

  const toggleExcludedRow = (index) => {
    setExcludedIndices((prev) =>
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index]
    );
  };

  const handleTrain = async () => {
    const selected = Object.keys(modelChecks).filter((m) => modelChecks[m]);
    if (!selected.length) return alert("Select at least one model.");

    if (!dependentVar || !independentVars.length)
      return alert("Select dependent and independent variables.");

    if (!userSchema || !selectedTable) {
      alert("Please select a training table first.");
      return;
    }

    // 1. Start Loading
    setTraining(true);
    setActiveTab("results");

    try {
      const fdBase = new FormData();
      fdBase.append("schema", userSchema);
      fdBase.append("table_name", selectedTable);
      fdBase.append("dependent_var", dependentVar);
      fdBase.append("independent_vars", JSON.stringify(independentVars));
      fdBase.append("excluded_indices", JSON.stringify(excludedIndices));

      const newResults = { lr: null, rf: null, xgb: null };

      const calls = selected.map(async (m) => {
        const fd = new FormData();
        for (const [key, val] of fdBase.entries()) fd.append(key, val);

        const endpoint =
          m === "lr"
            ? "/ai-tools/train-lr/train"
            : m === "rf"
              ? "/ai-tools/train-rf/train"
              : "/ai-tools/train-xgb/train";

        try {
          const res = await authFetch(`${API}${endpoint}`, {
            method: "POST",
            body: fd,
          });

          if (!res.ok) throw new Error(`Model ${m} failed`);

          newResults[m] = await res.json();
        } catch (err) {
          console.error(`Error training ${m}:`, err);
          // Optional: alert user specific model failed
        }
      });

      // Wait for all API calls
      await Promise.all(calls);

      setResults(newResults);

      const first = selected.find((m) => newResults[m]);
      if (first) {
        setActiveModelTab(first);
      }

      console.log(
        "🔄 Auto-saving training results to Common Table Database..."
      );
      await autoSaveToCommonDB(newResults, selected);
    } catch (error) {
      console.error("Critical error during training sequence:", error);
      alert("An error occurred during the training process.");
    } finally {
      // 2. STOP LOADING (This guarantees the loader disappears)
      setTraining(false);
    }
  };
  const autoSaveToCommonDB = async (results, trainedModels) => {
    for (const modelType of trainedModels) {
      const result = results[modelType];
      if (!result) continue;

      try {
        console.log(
          `📤 Auto-saving ${modelType.toUpperCase()} to Common DB...`
        );

        const formData = new FormData();

        // Model file path
        if (result.downloads?.model) {
          const modelPath = result.downloads.model.includes("?file=")
            ? decodeURIComponent(result.downloads.model.split("?file=")[1])
            : result.downloads.model;
          formData.append("model_path", modelPath);
        }

        // Shapefile path for predictions
        if (result.downloads?.shapefile_raw || result.downloads?.shapefile) {
          const shpPath =
            result.downloads.shapefile_raw ||
            (result.downloads.shapefile.includes("?file=")
              ? decodeURIComponent(
                  result.downloads.shapefile.split("?file=")[1]
                )
              : result.downloads.shapefile);
          formData.append("shapefile_path", shpPath);
        }

        formData.append("model_type", modelType);
        formData.append("model_version", result.model_version || 1);
        formData.append(
          "dependent_var",
          result.dependent_var || result.original_dependent_var || ""
        );
        formData.append("features_json", JSON.stringify(result.features || []));
        formData.append("metrics_json", JSON.stringify(result.metrics || {}));

        const response = await authFetch(
          `${API}/common/auto-save-training-results`,
          {
            method: "POST",
            headers: {
              "X-Target-Schema": userSchema,
            },
            body: formData,
          }
        );

        if (response.ok) {
          const data = await response.json();
          console.log(
            `✅ ${modelType.toUpperCase()} auto-saved to Common DB:`,
            data
          );
        } else {
          const error = await response.text();
          console.warn(`⚠️ Failed to auto-save ${modelType}:`, error);
        }
      } catch (err) {
        console.error(`❌ Auto-save error for ${modelType}:`, err);
      }
    }
  };

  const hasResults = !!(results.lr || results.rf || results.xgb);

  useEffect(() => {
    if (isOpen) loadCommonStatus();
  }, [isOpen]);

  useEffect(() => {
    if (userSchema) {
      loadAvailableTables();
    }
  }, [userSchema]);

  useEffect(() => {
    if (selectedTable) {
      loadTableFields();
    }
  }, [selectedTable]);

  useEffect(() => {
    if (!isOpen) {
      // Clear everything when modal closes
      setCommonStatus({ connected: false, context: null });
      setSelectedTable("");
      setFields([]);
      setPreviewRows([]);
      setPreviewTotal(0);
      setDependentVar("");
      setIndependentVars([]);
      setExcludedIndices([]);
      setResults({ lr: null, rf: null, xgb: null });
      setActiveModelTab(null);
      setAvailableTables([]);
      return;
    }

    // When modal opens, force fresh connection
    if (token) {
      connectCommon();
    }
  }, [isOpen, token]);

  useEffect(() => {
    setSelectedTable("");
    setFields([]);
    setPreviewRows([]);
    setPreviewTotal(0);
    setDependentVar("");
    setIndependentVars([]);
    setExcludedIndices([]);
    setResults({ lr: null, rf: null, xgb: null });
    setActiveModelTab(null);
  }, [userSchema]);

  if (!isOpen) return null;

  useEffect(() => {
    if (shouldDisconnect) {
      console.log("🔌 Handling disconnect signal from parent");

      // Clear all state
      setCommonStatus({ connected: false, context: null });
      setSelectedTable("");
      setFields([]);
      setPreviewRows([]);
      setPreviewTotal(0);
      setDependentVar("");
      setIndependentVars([]);
      setExcludedIndices([]);
      setResults({ lr: null, rf: null, xgb: null });
      setActiveModelTab(null);
      setAvailableTables([]);
      setCommonError("");

      console.log("✅ AI Tools state cleared");
    }
  }, [shouldDisconnect]);

  return (
    <div className="blgf-ai-root">
      <MapLoader isLoading={loadingMap} fieldName={loadingFieldName} />
      <div className="blgf-ai-panel">
        <TrainingLoader isTraining={training} />

        <div className="blgf-ai-header">
          <div>
            <div className="blgf-ai-title">AI Tools</div>
            <div className="blgf-ai-subtitle">
              Train models and explore outputs
              {userSchema && (
                <span className="blgf-ai-schema-tag">{userSchema}</span>
              )}
            </div>
          </div>

          <button className="blgf-ai-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="blgf-ai-block" style={{ marginTop: 12 }}>
          <div className="blgf-ai-label">Common Table Connection</div>

          <div
            style={{
              display: "flex",
              gap: 10,
              alignItems: "center",
              flexWrap: "wrap",
            }}
          >
            <div style={{ fontSize: 12, opacity: 0.8 }}>
              {commonStatus?.connected
                ? `Connected: ${commonStatus?.context?.db}.${commonStatus?.context?.schema}`
                : "Not connected"}
            </div>

            <div
              style={{
                marginLeft: "auto",
                display: "flex",
                gap: 10,
                alignItems: "center",
              }}
            >
              {commonStatus?.connected ? (
                <button
                  className="blgf-ai-btn-secondary"
                  disabled={commonBusy}
                  onClick={disconnectCommon}
                >
                  Disconnect
                </button>
              ) : (
                <button
                  className="blgf-ai-btn-primary"
                  disabled={commonBusy || !token}
                  onClick={connectCommon}
                >
                  {commonBusy ? "Connecting..." : "Connect"}
                </button>
              )}
            </div>
          </div>

          {commonError && (
            <div className="blgf-ai-helper-text error" style={{ marginTop: 8 }}>
              {commonError}
            </div>
          )}
        </div>

        <div className="blgf-ai-tabs">
          <div
            className={`blgf-ai-tab ${activeTab === "inputs" ? "active" : ""}`}
            onClick={() => setActiveTab("inputs")}
          >
            Train
          </div>

          <div
            className={`blgf-ai-tab ${activeTab === "results" ? "active" : ""} ${
              !hasResults ? "disabled" : ""
            }`}
            onClick={() => {
              if (hasResults) {
                setActiveTab("results");
              }
            }}
          >
            Results
          </div>

          <div
            className={`blgf-ai-tab ${
              activeTab === "run-saved" ? "active" : ""
            }`}
            onClick={() => setActiveTab("run-saved")}
          >
            Run Saved
          </div>
        </div>

        {activeTab === "inputs" && (
          <InputsTabUI
            fields={fields}
            dependentVar={dependentVar}
            independentVars={independentVars}
            previewRows={previewRows}
            previewTotal={previewTotal}
            previewPage={previewPage}
            PAGE_SIZE={PAGE_SIZE}
            setDependentVar={setDependentVar}
            setIndependentVars={setIndependentVars}
            modelChecks={modelChecks}
            setModelChecks={setModelChecks}
            toggleExcludedRow={toggleExcludedRow}
            excludedIndices={excludedIndices}
            setExcludedIndices={setExcludedIndices}
            handleTrain={handleTrain}
            training={training}
            userSchema={userSchema}
            availableTables={availableTables}
            selectedTable={selectedTable}
            setSelectedTable={setSelectedTable}
            loadDatabasePreview={loadDatabasePreview}
          />
        )}

        {activeTab === "results" && (
          <ResultsTabUI
            results={results}
            activeModelTab={activeModelTab}
            setActiveModelTab={setActiveModelTab}
            onShowMap={onShowMap}
            setLoadingMap={setLoadingMap}
            setLoadingFieldName={setLoadingFieldName}
            setSaveModalOpen={setSaveModalOpen}
            setSaveConfig={setSaveConfig}
            userSchema={userSchema}
          />
        )}

        {activeTab === "run-saved" && (
          <RunSavedTabUI
            onShowMap={onShowMap}
            userSchema={userSchema}
            setLoadingMap={setLoadingMap}
            setLoadingFieldName={setLoadingFieldName}
          />
        )}
      </div>

      <SaveToDBModal
        isOpen={saveModalOpen}
        onClose={() => {
          setSaveModalOpen(false);
          setSaveConfig(null);
        }}
        userSchema={userSchema}
        token={token}
        saveType={saveConfig?.saveType}
        modelType={saveConfig?.modelType}
        modelPath={saveConfig?.modelPath}
        dependentVar={saveConfig?.dependentVar}
        independentVars={saveConfig?.independentVars}
        shapefilePath={saveConfig?.shapefilePath}
      />
    </div>
  );
}

function AttributePreviewTable({
  previewRows,
  previewPage,
  PAGE_SIZE,
  dependentVar,
  independentVars,
  excludedIndices,
  toggleExcludedRow,
}) {
  const [sortConfig, setSortConfig] = useState({ key: null, direction: null });

  const displayColumns = React.useMemo(() => {
    return [...(dependentVar ? [dependentVar] : []), ...independentVars];
  }, [dependentVar, independentVars]);

  const sortedRows = React.useMemo(() => {
    if (!sortConfig.key || displayColumns.length === 0) return previewRows;

    return [...previewRows].sort((a, b) => {
      const aVal = parseFloat(a[sortConfig.key]) || 0;
      const bVal = parseFloat(b[sortConfig.key]) || 0;

      if (sortConfig.direction === "asc") {
        return aVal - bVal;
      } else {
        return bVal - aVal;
      }
    });
  }, [previewRows, sortConfig, displayColumns]);

  const handleSort = (column) => {
    setSortConfig((prev) => ({
      key: column,
      direction:
        prev.key === column && prev.direction === "desc" ? "asc" : "desc",
    }));
  };

  if (displayColumns.length === 0) {
    return (
      <div className="blgf-ai-table-wrap empty">
        <div>Select dependent and independent variables to preview data</div>
      </div>
    );
  }

  if (previewRows.length === 0 || sortedRows.length === 0) {
    return (
      <div className="blgf-ai-table-wrap empty">
        <div>No preview data available</div>
      </div>
    );
  }

  return (
    <div className="blgf-ai-table-wrap">
      <table className="blgf-ai-table">
        <thead>
          <tr>
            <th style={{ width: "50px" }}>Use</th>
            {displayColumns.map((col) => (
              <th
                key={col}
                className={`sortable ${
                  sortConfig.key === col
                    ? sortConfig.direction === "asc"
                      ? "sorted-asc"
                      : "sorted-desc"
                    : ""
                }`}
                onClick={() => handleSort(col)}
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {sortedRows.map((row, idx) => {
            const globalIdx = (previewPage - 1) * PAGE_SIZE + idx;
            const isExcluded = excludedIndices.includes(globalIdx);

            return (
              <tr key={idx} className={isExcluded ? "excluded" : ""}>
                <td>
                  <input
                    type="checkbox"
                    checked={!isExcluded}
                    onChange={() => toggleExcludedRow(globalIdx)}
                  />
                </td>

                {displayColumns.map((col) => (
                  <td key={col}>
                    {row[col] !== undefined && row[col] !== null
                      ? String(row[col])
                      : "-"}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function InputsTabUI({
  fields,
  dependentVar,
  independentVars,
  previewRows,
  previewTotal,
  previewPage,
  PAGE_SIZE,
  setDependentVar,
  setIndependentVars,
  modelChecks,
  setModelChecks,
  excludedIndices,
  setExcludedIndices,
  toggleExcludedRow,
  handleTrain,
  training,
  userSchema,
  availableTables,
  selectedTable,
  setSelectedTable,
  loadDatabasePreview,
}) {
  return (
    <div className="blgf-ai-content">
      <div className="blgf-ai-block">
        <div className="blgf-ai-label">Select Models</div>
        <div className="blgf-ai-models-grid">
          <div
            className={`blgf-ai-model-card ${modelChecks.lr ? "active" : ""}`}
            onClick={() => setModelChecks((p) => ({ ...p, lr: !p.lr }))}
          >
            <div className="blgf-ai-model-card-header">
              <span className="blgf-ai-model-name">Linear Regression</span>
              <div className="blgf-ai-checkbox-indicator">
                {modelChecks.lr && "✓"}
              </div>
            </div>
            <div className="blgf-ai-model-desc">
              Base statistical model for continuous target prediction.
            </div>
          </div>

          <div
            className={`blgf-ai-model-card ${modelChecks.rf ? "active" : ""}`}
            onClick={() => setModelChecks((p) => ({ ...p, rf: !p.rf }))}
          >
            <div className="blgf-ai-model-card-header">
              <span className="blgf-ai-model-name">Random Forest</span>
              <div className="blgf-ai-checkbox-indicator">
                {modelChecks.rf && "✓"}
              </div>
            </div>
            <div className="blgf-ai-model-desc">
              Ensemble learning method using multiple decision trees.
            </div>
          </div>

          <div
            className={`blgf-ai-model-card ${modelChecks.xgb ? "active" : ""}`}
            onClick={() => setModelChecks((p) => ({ ...p, xgb: !p.xgb }))}
          >
            <div className="blgf-ai-model-card-header">
              <span className="blgf-ai-model-name">XGBoost</span>
              <div className="blgf-ai-checkbox-indicator">
                {modelChecks.xgb && "✓"}
              </div>
            </div>
            <div className="blgf-ai-model-desc">
              Gradient boosting framework for high performance.
            </div>
          </div>
        </div>
      </div>

      <div className="blgf-ai-data-grid">
        <div className="blgf-ai-col-left">
          <div className="blgf-ai-block">
            <div className="blgf-ai-label">Training Table</div>
            <select
              value={selectedTable}
              onChange={(e) => setSelectedTable(e.target.value)}
              className="blgf-ai-select"
              disabled={availableTables.length === 0}
            >
              <option value="">Select a table</option>
              {availableTables.map((table) => (
                <option key={table} value={table}>
                  {table}
                </option>
              ))}
            </select>
            {availableTables.length === 0 && (
              <div className="blgf-ai-helper-text error">
                No tables found in schema {userSchema}
              </div>
            )}
          </div>

          <div className="blgf-ai-block">
            <div className="blgf-ai-label">Dependent Variable (Target)</div>
            <select
              value={dependentVar}
              onChange={(e) => setDependentVar(e.target.value)}
              className="blgf-ai-select"
              disabled={!fields.length}
            >
              <option value="">Select target</option>
              {fields.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="blgf-ai-col-right">
          <div className="blgf-ai-block full-height">
            <div className="blgf-ai-label">
              Independent Variables (Features)
            </div>
            <div className="blgf-ai-list">
              {fields.length === 0 && (
                <div className="blgf-ai-empty-list">
                  Select a table to load fields
                </div>
              )}
              {fields.map((f) => (
                <label key={f} className="blgf-ai-checkbox">
                  <input
                    type="checkbox"
                    checked={independentVars.includes(f)}
                    onChange={() =>
                      setIndependentVars((p) =>
                        p.includes(f) ? p.filter((x) => x !== f) : [...p, f]
                      )
                    }
                  />
                  <span>{f}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="blgf-ai-block">
        <div className="blgf-ai-preview-header">
          <div className="blgf-ai-label">Data Preview</div>
          <div className="blgf-ai-preview-actions">
            <button
              className="blgf-ai-btn-text"
              onClick={() => setExcludedIndices([])}
            >
              Select All
            </button>
            <button
              className="blgf-ai-btn-text"
              onClick={() =>
                setExcludedIndices(
                  Array.from({ length: previewTotal }, (_, i) => i)
                )
              }
            >
              Deselect All
            </button>
          </div>
        </div>

        <AttributePreviewTable
          previewRows={previewRows}
          previewPage={previewPage}
          PAGE_SIZE={PAGE_SIZE}
          dependentVar={dependentVar}
          independentVars={independentVars}
          excludedIndices={excludedIndices}
          toggleExcludedRow={toggleExcludedRow}
        />

        <div className="blgf-ai-pagination">
          <button
            onClick={() => {
              if (previewPage > 1) {
                loadDatabasePreview(previewPage - 1);
              }
            }}
            disabled={previewPage <= 1}
          >
            Previous
          </button>

          <span>
            Page {previewPage} / {Math.ceil(previewTotal / PAGE_SIZE) || 1}
          </span>

          <button
            onClick={() => {
              if (previewPage * PAGE_SIZE < previewTotal) {
                loadDatabasePreview(previewPage + 1);
              }
            }}
            disabled={previewPage * PAGE_SIZE >= previewTotal}
          >
            Next
          </button>
        </div>
      </div>

      <div className="blgf-ai-footer">
        <button
          className="blgf-ai-btn-primary"
          disabled={training}
          onClick={handleTrain}
        >
          {training ? "Training in Progress..." : "Train Selected Models"}
        </button>
      </div>
    </div>
  );
}

function ResultsTabUI({
  results,
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
      </div>

      {!activeModelTab && (
        <div className="blgf-ai-empty-placeholder">
          Train a model first to display results.
        </div>
      )}

      {activeModelTab && (
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
            {Object.entries(metrics).map(([k, v]) => (
              <tr key={k}>
                <td>{k}</td>
                <td className="align-right">
                  {typeof v === "number" ? v.toFixed(6) : v}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

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
                "Actual Value"
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

        <div className="blgf-ai-action-buttons">
          {dl.shapefile && (
            <button
              className="blgf-ai-btn-primary wide"
              onClick={async () => {
                const actualField = result?.dependent_var || "unit_value";
                setLoadingFieldName(actualField.toUpperCase());
                setLoadingMap(true);

                const rawPath =
                  dl.shapefile_raw || extractFilePath(dl.shapefile);
                const enc = encodeURIComponent(rawPath);
                const url = `/api/ai-tools/preview-geojson?file_path=${enc}`;

                const predictionRange = calculatePredictionRange();
                const actualRange = calculateActualRange();

                onShowMap({
                  url,
                  label:
                    modelType === "lr"
                      ? "Linear Regression"
                      : modelType === "rf"
                        ? "Random Forest"
                        : "XGBoost",
                  predictionField: "prediction",
                  actualField: result?.dependent_var || "actual_val",
                  predictionRange: predictionRange,
                  actualRange: actualRange,
                });

                setTimeout(() => {
                  setLoadingMap(false);
                }, 1000);
              }}
            >
              Visualize on Map
            </button>
          )}

          {dl.shapefile && (
            <button
              className="blgf-ai-btn-secondary wide"
              onClick={() => {
                setSaveConfig({
                  saveType: "model",
                  modelType: modelType,
                  modelPath: modelRawPath,
                  dependentVar: result?.dependent_var || "",
                  independentVars: result?.independent_vars || [],
                  shapefilePath: shapefileRawPath,
                });
                setSaveModalOpen(true);
              }}
            >
              Save to Database
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

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
