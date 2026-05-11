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
  preSelectedRunModel = null,
  openRunSaved = false,
  onPreSelectedRunModelConsumed = null,
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
    slm: false,
    sdm: false,
    gwr: false,
    hybrid: false,
    hybrid_sdm_xgb: false,
  });

  const [results, setResults] = useState({
    lr: null,
    rf: null,
    xgb: null,
    slm: null,
    sdm: null,
    gwr: null,
    hybrid_slm: null,
    hybrid: null,
    hybrid_sdm: null,
    hybrid_sdm_xgb: null,
  });

  const [trainErrors, setTrainErrors] = useState({});
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

      const newResults = {
        lr: null, rf: null, xgb: null, slm: null, sdm: null, gwr: null,
        hybrid_slm: null, hybrid: null,
        hybrid_sdm: null, hybrid_sdm_xgb: null,
      };
      const trainErrors = {};

      const calls = selected.map(async (m) => {
        const fd = new FormData();
        for (const [key, val] of fdBase.entries()) fd.append(key, val);

        const endpoint =
          m === "lr"              ? "/ai-tools/train-lr/train"
          : m === "rf"            ? "/ai-tools/train-rf/train"
          : m === "slm"           ? "/ai-tools/train-slm/train"
          : m === "sdm"           ? "/ai-tools/train-sdm/train"
          : m === "gwr"           ? "/ai-tools/train-gwr/train"
          : m === "hybrid"        ? "/ai-tools/train-hybrid-slm-rf/train"
          : m === "hybrid_sdm_xgb" ? "/ai-tools/train-hybrid-sdm-xgb/train"
          : "/ai-tools/train-xgb/train";

        try {
          const res = await authFetch(`${API}${endpoint}`, {
            method: "POST",
            body: fd,
          });

          if (!res.ok) {
            const errBody = await res.text().catch(() => "");
            let detail = `HTTP ${res.status}`;
            try {
              const parsed = JSON.parse(errBody);
              detail = parsed.error || parsed.detail || detail;
            } catch {
              if (errBody) detail = errBody;
            }
            trainErrors[m] = detail;
            throw new Error(detail);
          }

          const data = await res.json();
          if (m === "hybrid") {
            newResults["hybrid_slm"] = data.slm_stage;
            newResults["hybrid"]     = data.hybrid_stage;
          } else if (m === "hybrid_sdm_xgb") {
            newResults["hybrid_sdm"]     = data.sdm_stage;
            newResults["hybrid_sdm_xgb"] = data.hybrid_stage;
          } else {
            newResults[m] = data;
          }
        } catch (err) {
          if (!trainErrors[m]) trainErrors[m] = err.message;
          console.error(`Error training ${m}:`, err);
        }
      });

      await Promise.all(calls);

      setResults(newResults);
      setTrainErrors(trainErrors);

      const expandedSelected = [];
      for (const m of selected) {
        if (m === "hybrid")              { expandedSelected.push("hybrid_slm", "hybrid"); }
        else if (m === "hybrid_sdm_xgb") { expandedSelected.push("hybrid_sdm", "hybrid_sdm_xgb"); }
        else                             { expandedSelected.push(m); }
      }
      const first = expandedSelected.find((m) => newResults[m]);
      if (first) {
        setActiveModelTab(first);
      } else {
        const firstFailed = selected.find((m) => trainErrors[m]);
        if (firstFailed) setActiveModelTab(`${firstFailed}_err`);
      }

      console.log("🔄 Auto-saving training results to Common Table Database...");
      await autoSaveToCommonDB(newResults, selected);

      if (window.parent !== window) {
        const trainedModels = selected.filter((m) => newResults[m]);
        const failedModels = selected.filter((m) => !newResults[m]);
        window.parent.postMessage(
          {
            type: "AI_TOOLS_TRAINING_COMPLETE",
            status: trainedModels.length > 0 ? "success" : "failed",
            models_trained: trainedModels,
            failed_models: failedModels,
            errors: Object.keys(trainErrors).length > 0 ? trainErrors : null,
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

      if (window.parent !== window) {
        window.parent.postMessage(
          {
            type: "AI_TOOLS_TRAINING_COMPLETE",
            status: "error",
            models_trained: [],
            failed_models: Object.keys(modelChecks).filter((m) => modelChecks[m]),
            errors: { _critical: error.message },
            metrics: null,
            schema: userSchema,
            table: selectedTable,
            dependent_var: dependentVar,
            timestamp: new Date().toISOString(),
          },
          "*",
        );
      }
    } finally {
      setTraining(false);
    }
  };

  const autoSaveToCommonDB = async (results, trainedModels) => {
    const expandedModels = [];
    for (const m of trainedModels) {
      if (m === "hybrid") {
        if (results["hybrid_slm"]) expandedModels.push("hybrid_slm");
        if (results["hybrid"])     expandedModels.push("hybrid");
      } else if (m === "hybrid_sdm_xgb") {
        if (results["hybrid_sdm"])     expandedModels.push("hybrid_sdm");
        if (results["hybrid_sdm_xgb"]) expandedModels.push("hybrid_sdm_xgb");
      } else {
        expandedModels.push(m);
      }
    }

    for (const modelType of expandedModels) {
      const result = results[modelType];
      if (!result) continue;

      const backendModelType =
        modelType === "hybrid_slm"       ? "slm"
        : modelType === "hybrid"         ? "hybrid_slm_rf"
        : modelType === "hybrid_sdm"     ? "sdm"
        : modelType === "hybrid_sdm_xgb" ? "hybrid_sdm_xgb"
        : modelType;

      try {
        console.log(`📤 Auto-saving ${modelType.toUpperCase()} to Common DB...`);

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
              ? decodeURIComponent(result.downloads.shapefile.split("?file=")[1])
              : result.downloads.shapefile);
          formData.append("shapefile_path", shpPath);
        }

        formData.append("model_type", backendModelType);
        formData.append("model_version", result.model_version || 1);
        formData.append("dependent_var", result.dependent_var || result.original_dependent_var || "");
        formData.append("features_json", JSON.stringify(result.features || []));
        formData.append("metrics_json", JSON.stringify(result.metrics || {}));
        formData.append("importance_json", JSON.stringify(result.importance || []));
        formData.append("t_tests_json", JSON.stringify(result.t_test || null));

        const response = await authFetch(`${API}/common/auto-save-training-results`, {
          method: "POST",
          headers: { "X-Target-Schema": userSchema },
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          console.log(`✅ ${modelType.toUpperCase()} auto-saved to Common DB:`, data);
        } else {
          const error = await response.text();
          console.warn(`⚠️ Failed to auto-save ${modelType}:`, error);
        }
      } catch (err) {
        console.error(`❌ Auto-save error for ${modelType}:`, err);
      }
    }
  };

  const hasResults = !!(
    results.lr || results.rf || results.xgb ||
    results.slm || results.sdm || results.gwr ||
    results.hybrid_slm || results.hybrid ||
    results.hybrid_sdm || results.hybrid_sdm_xgb
  );

  const resetAllState = () => {
    setActiveTab("inputs");
    setCommonStatus({ connected: false, context: null });
    setSelectedTable("");
    setFields([]);
    setPreviewRows([]);
    setPreviewTotal(0);
    setDependentVar("");
    setIndependentVars([]);
    setExcludedIndices([]);
    setResults({
      lr: null, rf: null, xgb: null, slm: null, sdm: null, gwr: null,
      hybrid_slm: null, hybrid: null,
      hybrid_sdm: null, hybrid_sdm_xgb: null,
    });
    setTrainErrors({});
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

  useEffect(() => {
    if (!isOpen) {
      resetAllState();
      return;
    }

    if (token) {
      resetAllState();
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
    setResults({
      lr: null, rf: null, xgb: null, slm: null, sdm: null, gwr: null,
      hybrid_slm: null, hybrid: null,
      hybrid_sdm: null, hybrid_sdm_xgb: null,
    });
    setActiveModelTab(null);
  }, [userSchema]);

  useEffect(() => {
    if (shouldDisconnect) {
      console.log("🔌 Handling disconnect signal from parent");
      resetAllState();
      setCommonError("");
      console.log("✅ AI Tools state cleared");
    }
  }, [shouldDisconnect]);

  useEffect(() => {
    if (openRunSaved) {
      setActiveTab("run-saved");
    } else {
      setActiveTab("inputs");
    }
  }, [openRunSaved]);

  if (!isOpen) return null;

  return (
    <div className="blgf-ai-root">
      <div className="blgf-ai-panel">
        <TrainingLoader isTraining={training} selectedModels={Object.keys(modelChecks).filter(m => modelChecks[m])} />

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

          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <div style={{ fontSize: 12, opacity: 0.8 }}>
              {commonStatus?.connected
                ? `Connected: ${commonStatus?.context?.db}.${commonStatus?.context?.schema}`
                : "Not connected"}
            </div>

            <div style={{ marginLeft: "auto", display: "flex", gap: 10, alignItems: "center" }}>
              {commonStatus?.connected ? (
                <button className="blgf-ai-btn-secondary" disabled={commonBusy} onClick={disconnectCommon}>
                  Disconnect
                </button>
              ) : (
                <button className="blgf-ai-btn-primary" disabled={commonBusy || !token} onClick={connectCommon}>
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
            className={`blgf-ai-tab ${activeTab === "results" ? "active" : ""} ${!hasResults ? "disabled" : ""}`}
            onClick={() => { if (hasResults) setActiveTab("results"); }}
          >
            Results
          </div>

          <div
            className={`blgf-ai-tab ${activeTab === "run-saved" ? "active" : ""}`}
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
            trainErrors={trainErrors}
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
            preSelectedModel={preSelectedRunModel}
            openRunSaved={openRunSaved}
            onPreSelectedConsumed={onPreSelectedRunModelConsumed}
          />
        )}
      </div>
    </div>
  );
}