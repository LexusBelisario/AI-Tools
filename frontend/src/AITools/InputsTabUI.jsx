import React, { useState, useMemo } from "react";

// --- Attribute Preview Table ---

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

  const displayColumns = useMemo(() => {
    return [...(dependentVar ? [dependentVar] : []), ...independentVars];
  }, [dependentVar, independentVars]);

  const sortedRows = useMemo(() => {
    if (!sortConfig.key || displayColumns.length === 0) return previewRows;
    return [...previewRows].sort((a, b) => {
      const aVal = parseFloat(a[sortConfig.key]) || 0;
      const bVal = parseFloat(b[sortConfig.key]) || 0;
      return sortConfig.direction === "asc" ? aVal - bVal : bVal - aVal;
    });
  }, [previewRows, sortConfig, displayColumns]);

  const handleSort = (column) => {
    setSortConfig((prev) => ({
      key: column,
      direction: prev.key === column && prev.direction === "desc" ? "asc" : "desc",
    }));
  };

  if (displayColumns.length === 0) {
    return (
      <div className="blgf-ai-table-wrap empty">
        <div style={{ padding: "40px", textAlign: "center", color: "#94a3b8" }}>
          Select dependent and independent variables above to preview data
        </div>
      </div>
    );
  }

  if (previewRows.length === 0 || sortedRows.length === 0) {
    return (
      <div className="blgf-ai-table-wrap empty">
        <div style={{ padding: "40px", textAlign: "center", color: "#94a3b8" }}>
          No data available in selected table
        </div>
      </div>
    );
  }

  const availableColumns = displayColumns.filter((col) => sortedRows[0].hasOwnProperty(col));

  if (availableColumns.length === 0) {
    return (
      <div className="blgf-ai-table-wrap empty">
        <div style={{ padding: "40px", textAlign: "center", color: "#f59e0b" }}>
          ⚠️ Selected variables not found in table data
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
            {availableColumns.map((col) => (
              <th
                key={col}
                className={`sortable ${sortConfig.key === col ? (sortConfig.direction === "asc" ? "sorted-asc" : "sorted-desc") : ""}`}
                onClick={() => handleSort(col)}
                style={{ cursor: "pointer", userSelect: "none" }}
              >
                {col}
                {sortConfig.key === col && (
                  <span style={{ marginLeft: "8px", fontSize: "12px" }}>
                    {sortConfig.direction === "asc" ? "▲" : "▼"}
                  </span>
                )}
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
                  <input type="checkbox" checked={!isExcluded} onChange={() => toggleExcludedRow(globalIdx)} />
                </td>
                {availableColumns.map((col) => (
                  <td key={col}>{row[col] !== undefined && row[col] !== null ? String(row[col]) : "-"}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// --- Model Card ---

function ModelCard({ label, description, checked, onChange, accent, disabled }) {
  return (
    <div
      className={`blgf-ai-model-card ${checked ? "active" : ""} ${disabled ? "disabled" : ""}`}
      onClick={disabled ? undefined : onChange}
      style={{
        ...(checked && accent ? { borderColor: accent, boxShadow: `0 0 0 2px ${accent}22` } : {}),
        ...(disabled ? { opacity: 0.4, cursor: "not-allowed", pointerEvents: "none" } : {}),
      }}
    >
      <div className="blgf-ai-model-card-header">
        <span className="blgf-ai-model-name">{label}</span>
        <div className="blgf-ai-checkbox-indicator" style={checked && accent ? { background: accent, color: "#fff" } : {}}>
          {checked && "✓"}
        </div>
      </div>
      <div className="blgf-ai-model-desc">{description}</div>
    </div>
  );
}

// --- Inputs Tab ---

export default function InputsTabUI({
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

  const toggleModel = (key) => {
    if (key === "hybrid") {
      const next = !modelChecks.hybrid;
      // When enabling hybrid: uncheck all other models; when disabling: just uncheck hybrid
      setModelChecks((p) => ({
        lr:     next ? false : p.lr,
        rf:     next ? false : p.rf,
        xgb:    next ? false : p.xgb,
        slm:    next ? false : p.slm,
        hybrid: next,
      }));
    } else {
      // Cannot toggle other models while hybrid is selected
      if (modelChecks.hybrid) return;
      setModelChecks((p) => ({ ...p, [key]: !p[key] }));
    }
  };

  const hybridActive = modelChecks.hybrid;

  return (
    <div className="blgf-ai-content">
      <div className="blgf-ai-block">
        <div className="blgf-ai-label">Select Models</div>

        {/* --- AI Tools --- */}
        <div className="blgf-ai-model-group-label">AI Models</div>
        <div className="blgf-ai-models-grid">
          <ModelCard
            label="Linear Regression"
            description="Base statistical model for continuous target prediction."
            checked={modelChecks.lr}
            onChange={() => toggleModel("lr")}
            disabled={hybridActive}
          />
          <ModelCard
            label="Random Forest"
            description="Ensemble learning method using multiple decision trees."
            checked={modelChecks.rf}
            onChange={() => toggleModel("rf")}
            disabled={hybridActive}
          />
          <ModelCard
            label="XGBoost"
            description="Gradient boosting framework for high performance."
            checked={modelChecks.xgb}
            onChange={() => toggleModel("xgb")}
            disabled={hybridActive}
          />
        </div>

        {/* --- Geospatial Models --- */}
        <div className="blgf-ai-model-group-label" style={{ marginTop: 16 }}>Spatial Models</div>
        <div className="blgf-ai-models-grid">
          <ModelCard
            label="Spatial Lag Model"
            description="Accounts for spatial autocorrelation using a spatially lagged dependent variable."
            checked={modelChecks.slm}
            onChange={() => toggleModel("slm")}
            accent="#2563eb"
            disabled={hybridActive}
          />
        </div>

        {/* --- Hybrid Models --- */}
        <div className="blgf-ai-model-group-label" style={{ marginTop: 16 }}>Hybrid Models</div>
        <div className="blgf-ai-models-grid">
          <ModelCard
            label="Spatial Lag Model + Random Forest"
            description="Two-stage model: SLM captures spatial structure, Random Forest corrects nonlinear residuals."
            checked={modelChecks.hybrid}
            onChange={() => toggleModel("hybrid")}
            accent="#7c3aed"
          />
        </div>
      </div>

      <div className="blgf-ai-data-grid">
        <div className="blgf-ai-col-left">
          <div className="blgf-ai-block">
            <div className="blgf-ai-label">Select Training Table</div>
            <select
              value={selectedTable}
              onChange={(e) => setSelectedTable(e.target.value)}
              className="blgf-ai-select"
              disabled={availableTables.length === 0}
            >
              <option value="">Select a table</option>
              {availableTables.map((table) => (
                <option key={table} value={table}>{table}</option>
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
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="blgf-ai-col-right">
          <div className="blgf-ai-block full-height">
            <div className="blgf-ai-label">Independent Variables (Features)</div>
            <div className="blgf-ai-list">
              {fields.length === 0 && (
                <div className="blgf-ai-empty-list">Select a table to load fields</div>
              )}
              {fields.map((f) => (
                <label key={f} className="blgf-ai-checkbox">
                  <input
                    type="checkbox"
                    checked={independentVars.includes(f)}
                    onChange={() =>
                      setIndependentVars((p) =>
                        p.includes(f) ? p.filter((x) => x !== f) : [...p, f],
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
            <button className="blgf-ai-btn-text" onClick={() => setExcludedIndices([])}>
              Select All
            </button>
            <button
              className="blgf-ai-btn-text"
              onClick={() => setExcludedIndices(Array.from({ length: previewTotal }, (_, i) => i))}
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
          <button onClick={() => { if (previewPage > 1) loadDatabasePreview(previewPage - 1); }} disabled={previewPage <= 1}>
            Previous
          </button>
          <span>Page {previewPage} / {Math.ceil(previewTotal / PAGE_SIZE) || 1}</span>
          <button onClick={() => { if (previewPage * PAGE_SIZE < previewTotal) loadDatabasePreview(previewPage + 1); }} disabled={previewPage * PAGE_SIZE >= previewTotal}>
            Next
          </button>
        </div>
      </div>

      <div className="blgf-ai-footer">
        <button className="blgf-ai-btn-primary" disabled={training} onClick={handleTrain}>
          {training ? "Training in Progress..." : "Train Selected Models"}
        </button>
      </div>
    </div>
  );
}