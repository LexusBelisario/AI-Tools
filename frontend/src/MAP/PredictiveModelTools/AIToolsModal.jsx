import React, { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import API from "../../api.js";
import { useSchema } from "../SchemaContext.jsx";
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
export default function AIToolsModal({ isOpen, onClose, onShowMap }) {
  if (!isOpen) return null;

  const [inputMode, setInputMode] = useState("file");
  const [availableTables, setAvailableTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState("");
  const { schema: userSchema } = useSchema();
  const [files, setFiles] = useState([]);
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

  // 🆕 Load preview from database
  const loadDatabasePreview = async (page) => {
    if (!selectedTable || !userSchema) return;

    try {
      setPreviewPage(page);

      const fd = new FormData();
      fd.append("schema", userSchema);
      fd.append("table_name", selectedTable);
      fd.append("limit", PAGE_SIZE);
      fd.append("offset", (page - 1) * PAGE_SIZE);

      const res = await fetch(`${API}/ai-tools/preview-db`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const errorText = await res.text();
        console.error("❌ Response error:", errorText);
        alert(`Server error: ${res.status} - ${errorText}`);
        return;
      }

      const data = await res.json();

      setPreviewRows(data.rows || []);
      setPreviewTotal(data.total || 0);
    } catch (err) {
      console.error("❌ Preview error:", err);
      alert("Preview failed: " + err.message);
    }
  };

  // Tabs: inputs | results
  const [activeTab, setActiveTab] = useState("Train");

  // Selected models
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
  const loadAvailableTables = async () => {
    if (!userSchema) {
      console.warn("⚠️ No schema selected");
      return;
    }

    try {
      console.log(`📊 Loading training tables from schema: ${userSchema}`);

      const fd = new FormData();
      fd.append("schema", userSchema);

      const res = await fetch(`${API}/ai-tools/list-tables`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const errorText = await res.text();
        console.error("❌ Failed to load tables:", errorText);
        alert(`Failed to load tables: ${res.status}`);
        return;
      }

      const data = await res.json();
      console.log("✅ Available tables:", data);

      if (data.tables && data.tables.length > 0) {
        setAvailableTables(data.tables);
      } else {
        setAvailableTables([]);
        alert("No Training_Table found in this schema.");
      }
    } catch (err) {
      console.error("❌ Error loading tables:", err);
      alert(`Failed to load tables: ${err.message}`);
    }
  };

  const hasResults = !!(results.lr || results.rf || results.xgb);

  const loadTableFields = async () => {
    if (!selectedTable || !userSchema) return;

    try {
      console.log(`📊 Loading fields from ${userSchema}.${selectedTable}`);

      const fd = new FormData();
      fd.append("schema", userSchema);
      fd.append("table_name", selectedTable);

      const res = await fetch(`${API}/ai-tools/fields-db`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const errorText = await res.text();
        console.error("❌ Response error:", errorText);
        alert(`Server error: ${res.status} - ${errorText}`);
        return;
      }

      const data = await res.json();
      console.log("✅ Fields loaded:", data);

      if (data.fields) {
        setFields(data.fields);
      } else {
        alert("No fields found in the table.");
      }
    } catch (err) {
      console.error("❌ Error loading fields:", err);
      alert(`Failed to load fields: ${err.message}`);
    }
  };

  const loadFilePreview = async (page) => {
    try {
      setPreviewPage(page);
      if (!files.length) return;

      const fd = new FormData();
      const hasZip = files.some((f) => f.name.toLowerCase().endsWith(".zip"));

      if (hasZip) {
        fd.append(
          "zip_file",
          files.find((f) => f.name.toLowerCase().endsWith(".zip"))
        );
      } else {
        files.forEach((f) => fd.append("shapefiles", f));
      }

      fd.append("limit", PAGE_SIZE);
      fd.append("offset", (page - 1) * PAGE_SIZE);

      const res = await fetch(`${API}/ai-tools/preview`, {
        method: "POST",
        body: fd,
      });

      const data = await res.json();
      setPreviewRows(data.rows || []);
      setPreviewTotal(data.total || 0);
    } catch (err) {
      console.error(err);
      alert("Preview failed.");
    }
  };

  const handleFileChange = async (e) => {
    const selected = Array.from(e.target.files || []);
    if (!selected.length) return;

    console.log(
      "📁 Files selected:",
      selected.map((f) => f.name)
    );

    setFiles(selected);
    setFields([]);
    setDependentVar("");
    setIndependentVars([]);
    setPreviewRows([]);
    setPreviewTotal(0);

    try {
      const fd = new FormData();
      const hasZip = selected.some((f) =>
        f.name.toLowerCase().endsWith(".zip")
      );

      console.log("📦 Has ZIP:", hasZip);

      let url = `${API}/ai-tools/fields`;

      if (hasZip) {
        const zipFile = selected.find((f) =>
          f.name.toLowerCase().endsWith(".zip")
        );
        fd.append("zip_file", zipFile);
        url = `${API}/ai-tools/fields-zip`;
        console.log("🎯 Using ZIP endpoint:", url);
      } else {
        selected.forEach((f) => fd.append("shapefiles", f));
        console.log("🎯 Using shapefiles endpoint:", url);
      }

      console.log("🌐 Making request to:", url);

      const res = await fetch(url, { method: "POST", body: fd });

      console.log("📡 Response status:", res.status);
      console.log("📡 Response ok:", res.ok);

      if (!res.ok) {
        const errorText = await res.text();
        console.error("❌ Response error:", errorText);
        alert(`Server error: ${res.status} - ${errorText}`);
        return;
      }

      const data = await res.json();
      console.log("✅ Response data:", data);

      if (data.fields) {
        console.log("📋 Fields found:", data.fields);
        setFields(data.fields);
      } else {
        console.warn("⚠️ No fields in response");
        alert("No fields found in the uploaded file.");
      }
    } catch (err) {
      console.error("❌ Error in handleFileChange:", err);
      alert(`Failed to read shapefile: ${err.message}`);
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

    setTraining(true);
    setActiveTab("results");

    // Prepare base FormData
    const fdBase = new FormData();

    // 🆕 ADD MODE-SPECIFIC DATA
    if (inputMode === "database") {
      if (!userSchema || !selectedTable) {
        alert("Please select a training table first.");
        setTraining(false);
        return;
      }
      fdBase.append("schema", userSchema);
      fdBase.append("table_name", selectedTable);
    } else {
      // File mode
      const hasZip = files.some((f) => f.name.toLowerCase().endsWith(".zip"));
      if (hasZip) {
        fdBase.append(
          "zip_file",
          files.find((f) => f.name.toLowerCase().endsWith(".zip"))
        );
      } else {
        files.forEach((f) => fdBase.append("shapefiles", f));
      }
    }

    fdBase.append("dependent_var", dependentVar);
    fdBase.append("independent_vars", JSON.stringify(independentVars));
    fdBase.append("excluded_indices", JSON.stringify(excludedIndices));

    const newResults = { lr: null, rf: null, xgb: null };

    // Train each model in parallel
    const calls = selected.map(async (m) => {
      const fd = new FormData();
      for (const [key, val] of fdBase.entries()) fd.append(key, val);

      const endpoint =
        m === "lr"
          ? "/ai-tools/train-lr/train"
          : m === "rf"
            ? "/ai-tools/train-rf/train"
            : "/ai-tools/train-xgb/train";

      const res = await fetch(`${API}${endpoint}`, {
        method: "POST",
        body: fd,
      });

      newResults[m] = await res.json();
    });

    await Promise.all(calls);
    setResults(newResults);

    const first = selected.find((m) => newResults[m]);
    setActiveModelTab(first);

    setTraining(false);
  };
  useEffect(() => {
    if (inputMode === "database" && userSchema) {
      loadAvailableTables();
    }
  }, [inputMode, userSchema]);

  // Load fields when table is selected
  useEffect(() => {
    if (selectedTable && inputMode === "database") {
      loadTableFields();
    }
  }, [selectedTable, inputMode]);

  useEffect(() => {
    if (inputMode === "database") {
      setSelectedTable("");
      setFields([]);
      setPreviewRows([]);
      setPreviewTotal(0);
      setDependentVar("");
      setIndependentVars([]);
    }
  }, [userSchema]);

  useEffect(() => {
    // Only load if we have variables selected
    if (!dependentVar && independentVars.length === 0) {
      // Clear preview if no variables selected
      setPreviewRows([]);
      setPreviewTotal(0);
      return;
    }

    // Load preview based on mode
    if (inputMode === "database" && selectedTable) {
      loadDatabasePreview(1);
    } else if (inputMode === "file" && files.length > 0) {
      loadFilePreview(1);
    }
  }, [dependentVar, JSON.stringify(independentVars)]);

  return (
    <div className="blgf-ai-root">
      <MapLoader isLoading={loadingMap} fieldName={loadingFieldName} />{" "}
      <div className="blgf-ai-panel" style={{ position: "relative" }}>
        {/* 🆕 ILAGAY DITO - Inside the panel */}
        <TrainingLoader isTraining={training} />

        {/* HEADER */}
        <div className="blgf-ai-header">
          <div>
            <div className="blgf-ai-title">AI Tools</div>
            <div className="blgf-ai-subtitle">
              Train models and explore outputs
              {userSchema && (
                <span
                  style={{
                    marginLeft: "10px",
                    color: "#3b82f6",
                    fontWeight: "600",
                  }}
                >
                  📍 {userSchema}
                </span>
              )}
            </div>
          </div>

          <button className="blgf-ai-close" onClick={onClose}>
            ✕
          </button>
        </div>

        {/* MAIN TABS */}
        <div className="blgf-ai-tabs">
          <div
            className={`blgf-ai-tab ${activeTab === "inputs" ? "active" : ""}`}
            onClick={() => setActiveTab("inputs")}
          >
            Train
          </div>

          <div
            className={`blgf-ai-tab ${activeTab === "results" ? "active" : ""} ${!hasResults ? "disabled" : ""}`}
            onClick={() => {
              if (hasResults) {
                setActiveTab("results");
              }
            }}
            title={!hasResults ? "Train a model first to view results" : ""}
            style={{
              cursor: !hasResults ? "not-allowed" : "pointer",
              opacity: !hasResults ? 0.5 : 1,
            }}
          >
            Results
          </div>

          {/* ✅ ADD THIS */}
          <div
            className={`blgf-ai-tab ${activeTab === "run-saved" ? "active" : ""}`}
            onClick={() => setActiveTab("run-saved")}
          >
            Run Saved
          </div>
        </div>

        {activeTab === "inputs" && (
          <InputsTabUI
            files={files}
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
            handleFileChange={handleFileChange}
            handleTrain={handleTrain}
            loadFilePreview={loadFilePreview}
            training={training}
            inputMode={inputMode}
            setInputMode={setInputMode}
            userSchema={userSchema} // Pass from context
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
            setSaveModalOpen={setSaveModalOpen} // ✅ Add
            setSaveConfig={setSaveConfig} // ✅ Add
            userSchema={userSchema} // ✅ Add
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
        shapefilePath={saveConfig?.shapefilePath}
        userSchema={userSchema}
        saveType={saveConfig?.saveType}
        modelType={saveConfig?.modelType}
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
      <div className="blgf-ai-table-wrap">
        <div style={{ padding: "20px", textAlign: "center", color: "#64748b" }}>
          Select dependent and independent variables to preview data
        </div>
      </div>
    );
  }

  if (previewRows.length === 0 || sortedRows.length === 0) {
    return (
      <div className="blgf-ai-table-wrap">
        <div style={{ padding: "20px", textAlign: "center", color: "#64748b" }}>
          No preview data available
        </div>
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
  files,
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
  toggleExcludedRow,
  handleFileChange,
  handleTrain,
  loadFilePreview,
  training,
  inputMode,
  setInputMode,
  userSchema, // 🆕 From context
  availableTables, // 🆕 List of tables
  selectedTable, // 🆕 Selected table
  setSelectedTable, // 🆕 Setter
  loadDatabasePreview,
}) {
  return (
    <div className="blgf-ai-content">
      {/* ==================== MODE SELECTOR ==================== */}
      <div className="blgf-ai-block">
        <div className="blgf-ai-label">📂 Data Input Mode</div>

        <div className="blgf-ai-mode-radio-group">
          <label
            className={`blgf-ai-mode-radio-label ${inputMode === "file" ? "active" : ""}`}
          >
            <input
              type="radio"
              value="file"
              checked={inputMode === "file"}
              onChange={(e) => setInputMode(e.target.value)}
            />
            <span className="blgf-ai-mode-radio-text">
              📄 Upload File (Shapefile/ZIP)
            </span>
          </label>

          <label
            className={`blgf-ai-mode-radio-label ${inputMode === "database" ? "active" : ""}`}
          >
            <input
              type="radio"
              value="database"
              checked={inputMode === "database"}
              onChange={(e) => setInputMode(e.target.value)}
            />
            <span className="blgf-ai-mode-radio-text">
              🗄️ Database (Training Table)
            </span>
          </label>
        </div>
      </div>

      {/* ==================== 🆕 DATABASE MODE - TABLE DROPDOWN ==================== */}
      {inputMode === "database" && (
        <div className="blgf-ai-block">
          {/* Show current schema */}
          <div className="blgf-ai-info-box schema">
            📍 <strong>Current Schema:</strong> {userSchema || "Not selected"}
          </div>

          {/* Table selector dropdown */}
          <div style={{ marginBottom: "15px" }}>
            <div className="blgf-ai-label">Select Training Table</div>
            <select
              value={selectedTable}
              onChange={(e) => setSelectedTable(e.target.value)}
              className="blgf-ai-select"
              disabled={availableTables.length === 0}
            >
              <option value="">-- Select a table --</option>
              {availableTables.map((table) => (
                <option key={table} value={table}>
                  {table}
                </option>
              ))}
            </select>
          </div>

          {/* No tables warning */}
          {availableTables.length === 0 && (
            <div className="blgf-ai-info-box warning">
              ⚠️ No Training_Table found in schema <strong>{userSchema}</strong>
            </div>
          )}

          {/* Selected table confirmation */}
          {selectedTable && (
            <div className="blgf-ai-info-box success">
              ✅ Using table: <strong>{selectedTable}</strong>
            </div>
          )}
        </div>
      )}
      {/* ==================== FILE UPLOAD MODE ==================== */}
      {inputMode === "file" && (
        <div className="blgf-ai-block">
          <div className="blgf-ai-label">Upload Shapefile or ZIP</div>
          <input
            type="file"
            multiple
            accept=".zip,.shp,.dbf,.shx,.prj"
            onChange={handleFileChange}
            className="blgf-ai-input-file"
          />

          <div className="blgf-ai-filelist">
            {files.map((f, i) => (
              <span key={i}>{f.name}</span>
            ))}
          </div>
        </div>
      )}

      {/* ----------------------- DEPENDENT VARIABLE ----------------------- */}
      <div className="blgf-ai-block">
        <div className="blgf-ai-label">Dependent Variable</div>
        <select
          value={dependentVar}
          onChange={(e) => setDependentVar(e.target.value)}
          className="blgf-ai-select"
        >
          <option value="">-- Select --</option>
          {fields.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
      </div>

      {/* -------------------- INDEPENDENT VARIABLES -------------------- */}
      <div className="blgf-ai-block">
        <div className="blgf-ai-label">Independent Variables</div>

        <div className="blgf-ai-list">
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
              {f}
            </label>
          ))}
        </div>
      </div>

      {/* ------------------------------ MODEL CHECKBOXES ------------------------------ */}
      <div className="blgf-ai-block">
        <div className="blgf-ai-label">Models to Train</div>

        <div className="blgf-ai-models">
          <label
            className={`blgf-ai-model-checkbox-label ${modelChecks.lr ? "checked" : ""}`}
          >
            <input
              type="checkbox"
              checked={modelChecks.lr}
              onChange={() => setModelChecks((p) => ({ ...p, lr: !p.lr }))}
            />
            <span className="blgf-ai-model-checkbox-text">
              Linear Regression
            </span>
          </label>

          <label
            className={`blgf-ai-model-checkbox-label ${modelChecks.rf ? "checked" : ""}`}
          >
            <input
              type="checkbox"
              checked={modelChecks.rf}
              onChange={() => setModelChecks((p) => ({ ...p, rf: !p.rf }))}
            />
            <span className="blgf-ai-model-checkbox-text">Random Forest</span>
          </label>

          <label
            className={`blgf-ai-model-checkbox-label ${modelChecks.xgb ? "checked" : ""}`}
          >
            <input
              type="checkbox"
              checked={modelChecks.xgb}
              onChange={() => setModelChecks((p) => ({ ...p, xgb: !p.xgb }))}
            />
            <span className="blgf-ai-model-checkbox-text">XGBoost</span>
          </label>
        </div>
      </div>

      {/* ---------------------------- ATTRIBUTE TABLE ---------------------------- */}
      <div className="blgf-ai-block">
        <div className="blgf-ai-preview-header">
          <div className="blgf-ai-subtitle2">Attribute Preview</div>
          <div className="blgf-ai-preview-actions">
            <button
              className="blgf-ai-btn-secondary"
              onClick={() => setExcludedIndices([])}
            >
              ✓ Select All
            </button>
            <button
              className="blgf-ai-btn-secondary"
              onClick={() =>
                setExcludedIndices(
                  Array.from({ length: previewTotal }, (_, i) => i)
                )
              }
            >
              ✗ Deselect All
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

        {/* Pagination */}
        <div className="blgf-ai-pagination">
          <button
            onClick={() => {
              if (previewPage > 1) {
                if (inputMode === "database") {
                  loadDatabasePreview(previewPage - 1);
                } else {
                  loadFilePreview(previewPage - 1);
                }
              }
            }}
          >
            Prev
          </button>

          <span>
            Page {previewPage} / {Math.ceil(previewTotal / PAGE_SIZE) || 1}
          </span>

          <button
            onClick={() => {
              if (previewPage * PAGE_SIZE < previewTotal) {
                if (inputMode === "database") {
                  loadDatabasePreview(previewPage + 1);
                } else {
                  loadFilePreview(previewPage + 1);
                }
              }
            }}
          >
            Next
          </button>
        </div>
      </div>

      {/* ------------------------------ TRAIN BUTTON ------------------------------ */}
      <div className="blgf-ai-btnrow">
        <button
          className="blgf-ai-btn-primary"
          disabled={training}
          onClick={handleTrain}
        >
          {training ? "Training…" : "Train Models"}
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
  setSaveModalOpen, // ✅ Add
  setSaveConfig, // ✅ Add
  userSchema,
}) {
  const hasLR = !!results.lr;
  const hasRF = !!results.rf;
  const hasXGB = !!results.xgb;

  return (
    <div className="blgf-ai-content">
      {/* ---------------- MODEL SELECTION TABS ---------------- */}
      <div className="blgf-ai-modeltabs">
        {hasLR && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "lr" ? "active" : ""}`}
            onClick={() => setActiveModelTab("lr")}
          >
            Linear Regression
          </div>
        )}

        {hasRF && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "rf" ? "active" : ""}`}
            onClick={() => setActiveModelTab("rf")}
          >
            Random Forest
          </div>
        )}

        {hasXGB && (
          <div
            className={`blgf-ai-modeltab ${activeModelTab === "xgb" ? "active" : ""}`}
            onClick={() => setActiveModelTab("xgb")}
          >
            XGBoost
          </div>
        )}
      </div>

      {/* ---------------- EMPTY STATE ---------------- */}
      {!activeModelTab && (
        <div style={{ color: "#94a3b8", padding: "20px" }}>
          Train a model first to display results.
        </div>
      )}

      {/* ---------------- MODEL SECTION ---------------- */}
      {activeModelTab && (
        <ModelSection
          modelType={activeModelTab}
          modelResult={results[activeModelTab]}
          onShowMap={onShowMap}
          setLoadingMap={setLoadingMap}
          setLoadingFieldName={setLoadingFieldName}
          setSaveModalOpen={setSaveModalOpen} // ✅ Add
          setSaveConfig={setSaveConfig} // ✅ Add
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
  setSaveModalOpen, // ✅ Add
  setSaveConfig, // ✅ Add
  userSchema, // ✅ Add
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
      <div className="blgf-ai-modeltitle">{niceName}</div>

      {/* SUB-TABS BAR */}
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

      {/* ================= CONTENT ================= */}

      {subTab === "metrics" && (
        <>
          <MetricsSection
            modelType={modelType}
            result={modelResult}
            onShowMap={onShowMap}
            setLoadingMap={setLoadingMap}
            setLoadingFieldName={setLoadingFieldName}
            setSaveModalOpen={setSaveModalOpen} // ✅ Already here
            setSaveConfig={setSaveConfig} // ✅ Already here
            userSchema={userSchema} // ✅ Already here
          />

          {/* Show Feature Importance for ALL models */}
          <ImportanceSection modelType={modelType} result={modelResult} />

          {/* LR-specific coefficient details */}
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
  setSaveModalOpen, // ✅ Add
  setSaveConfig, // ✅ Add
  userSchema, // ✅ Add
}) {
  const metrics = result?.metrics || {};
  return (
    <>
      {/* DOWNLOADS CARD */}
      <ModelDownloads
        modelType={modelType}
        result={result}
        onShowMap={onShowMap}
        setLoadingMap={setLoadingMap}
        setLoadingFieldName={setLoadingFieldName}
        setSaveModalOpen={setSaveModalOpen} // ✅ Add
        setSaveConfig={setSaveConfig} // ✅ Add
        userSchema={userSchema} // ✅ Add
      />

      {/* METRICS CARD */}
      <div className="blgf-ai-card">
        <div className="blgf-ai-subtitle2">Metrics</div>

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
      <div className="blgf-ai-subtitle2">Coefficients</div>

      {/* RAW COEFFICIENTS */}
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

      {/* COEFFICIENT T-TESTS */}
      {tTests.length > 0 && (
        <>
          <div className="blgf-ai-subtitle3">Coefficient T-Tests</div>

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

      {/* RESIDUALS T-TEST */}
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
        <div style={{ color: "#94a3b8" }}>No feature importance available.</div>
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
        {/* TABLE */}
        <table
          className="blgf-ai-table narrow"
          style={{
            marginBottom: "20px",
            fontFamily: "Plus Jakarta Sans, system-ui, sans-serif",
          }}
        >
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
        {/* PLOTLY BAR CHART */}
        jsx
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
  const residuals = d.residuals || [];
  const bins = d.residual_bins || [];
  const binCounts = d.residual_counts || [];

  return (
    <>
      {/* Actual vs Predicted */}
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

      {/* Residual Distribution */}
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
        <div style={{ color: "#94a3b8" }}>No distribution data available.</div>
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
              <div
                style={{
                  color: "#94a3b8",
                  padding: "20px",
                  textAlign: "center",
                }}
              >
                No data available for this variable
              </div>
            </div>
          );
        }

        return (
          <div className="blgf-ai-card" key={v}>
            <div className="blgf-ai-chart-container">
              <div className="blgf-ai-chart-title">Distribution of {v}</div>

              {/* Stats Badges */}
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

              {/* Histogram */}
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

  // ✅ Extract raw path from shapefile URL
  const shapefileRawPath = dl.shapefile_raw || extractFilePath(dl.shapefile);

  const calculatePredictionRange = () => {
    console.log("🔍 Calculating prediction range...");
    const interactiveData = result?.interactive_data || {};
    const preds = interactiveData.preds || [];

    if (preds.length === 0) {
      console.warn("⚠️ No predictions found");
      return null;
    }

    const min = Math.min(...preds);
    const max = Math.max(...preds);
    console.log("✅ Prediction range:", { min, max });
    return { min, max };
  };

  const calculateActualRange = () => {
    console.log("🔍 Calculating actual values range...");
    console.log("   Result object:", result);
    const interactiveData = result?.interactive_data || {};
    const actuals = interactiveData.y_test || [];

    if (actuals.length === 0) {
      console.warn("⚠️ No actual values found");
      return null;
    }

    const min = Math.min(...actuals);
    const max = Math.max(...actuals);
    console.log("✅ Actual range:", { min, max });
    return { min, max };
  };

  return (
    <div className="blgf-ai-card">
      <div className="blgf-ai-subtitle2">Downloads</div>

      <ul className="blgf-ai-downloads">
        {dl.model && (
          <li>
            <a href={normalize(dl.model)} target="_blank">
              📦 {modelType.toUpperCase()} Model (.pkl)
            </a>
          </li>
        )}

        {dl.report && (
          <li>
            <a href={normalize(dl.report)} target="_blank">
              📄 PDF Report
            </a>
          </li>
        )}

        {dl.shapefile && (
          <li>
            <a href={normalize(dl.shapefile)} target="_blank">
              🗺️ Predicted Shapefile (.zip)
            </a>
          </li>
        )}

        {dl.cama_csv && (
          <li>
            <a href={normalize(dl.cama_csv)} target="_blank">
              📊 CAMA CSV
            </a>
          </li>
        )}
      </ul>

      {dl.shapefile && (
        <button
          className="blgf-ai-btn-primary wide"
          onClick={async () => {
            const actualField = result?.dependent_var || "unit_value";
            setLoadingFieldName(actualField.toUpperCase());
            setLoadingMap(true);

            const rawPath = dl.shapefile_raw || extractFilePath(dl.shapefile);
            const enc = encodeURIComponent(rawPath);
            const url = `/api/ai-tools/preview-geojson?file_path=${enc}`;

            const predictionRange = calculatePredictionRange();
            const actualRange = calculateActualRange();

            console.log("📦 Sending to map:", {
              url,
              predictionRange,
              actualRange,
              dependentVar: result?.dependent_var,
            });

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
          🗺️ Show On Map
        </button>
      )}
      {dl.shapefile && (
        <button
          className="blgf-ai-btn-secondary wide"
          style={{ marginTop: "10px" }}
          onClick={() => {
            console.log("💾 Save button clicked:", {
              shapefileRawPath,
              modelType,
            });
            setSaveConfig({
              shapefilePath: shapefileRawPath,
              saveType: "training",
              modelType: modelType,
            });
            setSaveModalOpen(true);
          }}
        >
          💾 Save to Database
        </button>
      )}
    </div>
  );
}
function ModelCAMA({ result }) {
  const rows = result?.cama_preview || [];
  if (!rows.length)
    return (
      <div className="blgf-ai-card">
        <div className="blgf-ai-subtitle2">Training Result Preview</div>{" "}
        <div style={{ color: "#94a3b8" }}>
          No training data preview available.
        </div>{" "}
      </div>
    );
  const columns = Object.keys(rows[0]);
  return (
    <div className="blgf-ai-card">
      <div className="blgf-ai-subtitle2">Training Result Preview</div>{" "}
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
