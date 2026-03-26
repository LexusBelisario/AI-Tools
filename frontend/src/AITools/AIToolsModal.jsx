import React, { useState, useEffect } from "react";
import API from "../api.js";
import TrainingLoader from "./components/trainingLoader.jsx";
import "./components/aitoolsmodal.css";
import InputsTabUI from "./InputsTabUI.jsx";
import ResultsTabUI from "./ResultsTabUI.jsx";
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
          .join(""),
      );
      return JSON.parse(json);
    } catch {
      return {};
    }
  };

  const tokenPayload = decodeJwtPayload(token);

  const userDb =
    tokenPayload.province_access ||
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

  const connectCommon = async ({ tokenOnly = false } = {}) => {
    if (!token) {
      setCommonError("No token received.");
      return;
    }

    setCommonBusy(true);
    setCommonError("");

    console.log("🔄 Connecting with schema:", externalSchema);

    try {
      // When tokenOnly is true, skip the stale X-Target-Schema / X-Target-DB
      // overrides and let the backend resolve context purely from the token.
      const headers = { Authorization: `Bearer ${token}` };
      if (!tokenOnly) {
        if (userSchema) headers["X-Target-Schema"] = userSchema;
        if (userDb) headers["X-Target-DB"] = userDb;
      }

      const res = await fetch(`${API}/common/connect`, {
        method: "POST",
        headers,
      });

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

      console.log("✅ Connected to:", data.context);
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

      console.log("📊 Preview data received:", {
        rowCount: data.rows?.length,
        totalRows: data.total,
        sampleRow: data.rows?.[0],
      });

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
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index],
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
        }
      });

      await Promise.all(calls);

      setResults(newResults);

      const first = selected.find((m) => newResults[m]);
      if (first) {
        setActiveModelTab(first);
      }

      console.log(
        "🔄 Auto-saving training results to Common Table Database...",
      );
      await autoSaveToCommonDB(newResults, selected);

      // Notify parent window that training is complete
      if (window.parent !== window) {
        const trainedModels = selected.filter((m) => newResults[m]);
        window.parent.postMessage(
          {
            type: "AI_TOOLS_TRAINING_COMPLETE",
            status: trainedModels.length > 0 ? "success" : "failed",
            models_trained: trainedModels,
            metrics: Object.fromEntries(
              trainedModels.map((m) => [m, newResults[m]?.metrics || null]),
            ),
            schema: userSchema,
            table: selectedTable,
            dependent_var: dependentVar,
            timestamp: new Date().toISOString(),
          },
          "*",
        );
        console.log("📨 Sent AI_TOOLS_TRAINING_COMPLETE to parent");
      }
    } catch (error) {
      console.error("Critical error during training sequence:", error);
      alert("An error occurred during the training process.");
    } finally {
      setTraining(false);
    }
  };

  const autoSaveToCommonDB = async (results, trainedModels) => {
    for (const modelType of trainedModels) {
      const result = results[modelType];
      if (!result) continue;

      try {
        console.log(
          `📤 Auto-saving ${modelType.toUpperCase()} to Common DB...`,
        );

        const formData = new FormData();

        if (result.downloads?.model) {
          const modelPath = result.downloads.model.includes("?file=")
            ? decodeURIComponent(result.downloads.model.split("?file=")[1])
            : result.downloads.model;
          formData.append("model_path", modelPath);
        }

        if (result.downloads?.shapefile_raw || result.downloads?.shapefile) {
          const shpPath =
            result.downloads.shapefile_raw ||
            (result.downloads.shapefile.includes("?file=")
              ? decodeURIComponent(
                  result.downloads.shapefile.split("?file=")[1],
                )
              : result.downloads.shapefile);
          formData.append("shapefile_path", shpPath);
        }

        formData.append("model_type", modelType);
        formData.append("model_version", result.model_version || 1);
        formData.append(
          "dependent_var",
          result.dependent_var || result.original_dependent_var || "",
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
          },
        );

        if (response.ok) {
          const data = await response.json();
          console.log(
            `✅ ${modelType.toUpperCase()} auto-saved to Common DB:`,
            data,
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

  // --- Helper to reset all form/connection state ---
  const resetAllState = () => {
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
  };

  // --- Effects ---

  useEffect(() => {
    if (userSchema) {
      loadAvailableTables();
    }
  }, [userSchema]);

  useEffect(() => {
    if (selectedTable && userSchema) {
      loadTableFields();
    }
  }, [selectedTable]);

  useEffect(() => {
    if (selectedTable && (dependentVar || independentVars.length > 0)) {
      loadDatabasePreview(1);
    }
  }, [dependentVar, independentVars, selectedTable]);

  // FIX: When token changes while modal is already open, reset stale
  // commonStatus BEFORE reconnecting. Without this, authFetch sends the
  // old userSchema in X-Target-Schema which overrides the new token's
  // claims on the backend, causing the "stuck on old LGU" bug.
  useEffect(() => {
    if (!isOpen) {
      resetAllState();
      return;
    }

    if (token) {
      // Clear stale connection context so the UI resets for the new LGU.
      resetAllState();
      // Use tokenOnly so the connect call does NOT send stale
      // X-Target-Schema / X-Target-DB from the previous render.
      connectCommon({ tokenOnly: true });
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

  // NOTE: Moved above the early return so React hooks are always called
  // in the same order (hooks must not be conditional / after early returns).
  useEffect(() => {
    if (shouldDisconnect) {
      console.log("🔌 Handling disconnect signal from parent");

      resetAllState();
      setCommonError("");

      console.log("✅ AI Tools state cleared");
    }
  }, [shouldDisconnect]);

  if (!isOpen) return null;

  // --- Render ---

  return (
    <div className="blgf-ai-root">
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
            token={token}
            userDb={userDb}
          />
        )}
      </div>
    </div>
  );
}