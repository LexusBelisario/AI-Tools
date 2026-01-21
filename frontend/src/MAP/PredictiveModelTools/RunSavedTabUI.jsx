import React, { useState, useEffect } from "react";
import API from "../../api.js";

export default function RunSavedTabUI({
  onShowMap,
  userSchema,
  setLoadingMap,
  setLoadingFieldName,
  token = "",
  userDb = null,
}) {
  // === MODEL INPUT STATE ===
  const [modelSource, setModelSource] = useState("db"); // "upload" or "db"
  const [modelFile, setModelFile] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedDbModel, setSelectedDbModel] = useState("");

  // === DATA INPUT STATE ===
  const [inputSource, setInputSource] = useState("db"); // "db" or "file"
  const [tableName, setTableName] = useState("LandParcel");
  const [shapefiles, setShapefiles] = useState([]);
  const [zipFile, setZipFile] = useState(null);

  // === UI STATE ===
  const [loading, setLoading] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

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

  // fallback kung hindi pinasa ang userDb
  const tokenPayload = decodeJwtPayload(token);
  const resolvedUserDb =
    userDb ||
    tokenPayload.db ||
    tokenPayload.db_name ||
    tokenPayload.dbname ||
    tokenPayload.prov_dbname ||
    null;

  // Auth helper
  const authFetch = (url, options = {}) => {
    const headers = { ...(options.headers || {}) };

    if (token) headers.Authorization = `Bearer ${token}`;
    if (userSchema) headers["X-Target-Schema"] = userSchema;
    if (resolvedUserDb) headers["X-Target-DB"] = resolvedUserDb;

    return fetch(url, { ...options, headers });
  };

  const loadAvailableModels = async () => {
    if (!token) {
      setError("Authentication token is missing. Please reconnect.");
      setAvailableModels([]);
      return;
    }
    if (!userSchema) {
      setError("No schema selected. Please connect to Common Database first.");
      setAvailableModels([]);
      return;
    }

    setLoadingModels(true);
    setError(null);

    try {
      const res = await authFetch(`${API}/ai-tools/list-models`, {
        method: "GET",
      });

      if (!res.ok) {
        const errorText = await res.text();
        setAvailableModels([]);
        setError(`Failed to load models: ${errorText}`);
        return;
      }

      const data = await res.json();
      setAvailableModels(data.models || []);
    } catch (err) {
      setAvailableModels([]);
      setError(`Error loading models: ${err.message}`);
    } finally {
      setLoadingModels(false);
    }
  };

  // Load models when DB source is selected
  useEffect(() => {
    if (modelSource === "db") {
      loadAvailableModels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelSource, userSchema, token, resolvedUserDb]);

  const handleModelFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith(".pkl")) {
      setModelFile(file);
      setError(null);
    } else {
      setError("Please upload a valid .pkl model file");
    }
  };

  const handleShapefilesChange = (e) => {
    setShapefiles(Array.from(e.target.files));
    setZipFile(null);
  };

  const handleZipFileChange = (e) => {
    setZipFile(e.target.files[0]);
    setShapefiles([]);
  };

  const handleRunModel = async () => {
    setError(null);
    setResult(null);

    if (!token) {
      setError("Authentication token is missing. Please reconnect.");
      return;
    }

    if (modelSource === "upload" && !modelFile) {
      setError("Please upload a model file (.pkl)");
      return;
    }
    if (modelSource === "db" && !selectedDbModel) {
      setError("Please select a model from the database");
      return;
    }

    if (inputSource === "db" && !userSchema) {
      setError("No schema selected. Please connect to Common Database first.");
      return;
    }
    if (inputSource === "file" && shapefiles.length === 0 && !zipFile) {
      setError("Please upload shapefile(s) or a ZIP file");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();

      // model info
      formData.append("model_source", modelSource);
      if (modelSource === "upload") {
        formData.append("model_file", modelFile);
      } else {
        formData.append("model_id", selectedDbModel);
      }

      // data info
      if (inputSource === "db") {
        formData.append("schema", userSchema);
        formData.append("table_name", tableName);
      } else {
        if (zipFile) {
          formData.append("zip_file", zipFile);
        } else {
          shapefiles.forEach((file) => {
            formData.append("shapefiles", file);
          });
        }
      }

      const response = await authFetch(`${API}/ai-tools/run-saved-model`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || data.detail || "Failed to run model");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setModelFile(null);
    setSelectedDbModel("");
    setShapefiles([]);
    setZipFile(null);
    setResult(null);
    setError(null);
    setTableName("LandParcel");
  };

  return (
    <div className="blgf-ai-content">
      <div className="blgf-ai-data-grid" style={{ gridTemplateColumns: "1fr" }}>
        {/* STEP 1 */}
        <div className="blgf-ai-block">
          <div className="blgf-ai-label">1. Select Model Source</div>

          <div className="blgf-ai-models-grid" style={{ marginBottom: "20px" }}>
            <div
              className={`blgf-ai-model-card ${modelSource === "db" ? "active" : ""}`}
              onClick={() => setModelSource("db")}
            >
              <div className="blgf-ai-model-card-header">
                <span className="blgf-ai-model-name">From Common Database</span>
                <div className="blgf-ai-checkbox-indicator">
                  {modelSource === "db" && "✓"}
                </div>
              </div>
              <div className="blgf-ai-model-desc">
                Select a saved model from the Common Table Database.
              </div>
            </div>

            <div
              className={`blgf-ai-model-card ${modelSource === "upload" ? "active" : ""}`}
              onClick={() => setModelSource("upload")}
            >
              <div className="blgf-ai-model-card-header">
                <span className="blgf-ai-model-name">Upload File</span>
                <div className="blgf-ai-checkbox-indicator">
                  {modelSource === "upload" && "✓"}
                </div>
              </div>
              <div className="blgf-ai-model-desc">
                Upload a local .pkl model file.
              </div>
            </div>
          </div>

          <div className="blgf-ai-card">
            {modelSource === "db" ? (
              <>
                <div className="blgf-ai-label">Select Saved Model</div>

                {loadingModels ? (
                  <div className="blgf-ai-helper-text">Loading models...</div>
                ) : (
                  <>
                    <select
                      value={selectedDbModel}
                      onChange={(e) => setSelectedDbModel(e.target.value)}
                      className="blgf-ai-select"
                      disabled={availableModels.length === 0}
                    >
                      <option value="">-- Select a model --</option>
                      {availableModels.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.display_name}
                        </option>
                      ))}
                    </select>

                    {availableModels.length === 0 && !loadingModels && (
                      <div
                        className="blgf-ai-helper-text error"
                        style={{ marginTop: "10px" }}
                      >
                        No saved models found in Common Database. Train a model
                        first.
                      </div>
                    )}
                  </>
                )}
              </>
            ) : (
              <>
                <div className="blgf-ai-label">Upload Model (.pkl)</div>
                <input
                  type="file"
                  accept=".pkl"
                  onChange={handleModelFileChange}
                  className="blgf-ai-select"
                  style={{ padding: "10px" }}
                />
                {modelFile && (
                  <div
                    className="blgf-ai-filelist"
                    style={{ marginTop: "10px" }}
                  >
                    <span>{modelFile.name}</span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* STEP 2 */}
        <div className="blgf-ai-block">
          <div className="blgf-ai-label">2. Select Data Source</div>
          <div className="blgf-ai-models-grid">
            <div
              className={`blgf-ai-model-card ${inputSource === "db" ? "active" : ""}`}
              onClick={() => setInputSource("db")}
            >
              <div className="blgf-ai-model-card-header">
                <span className="blgf-ai-model-name">
                  Common Database (LandParcel)
                </span>
                <div className="blgf-ai-checkbox-indicator">
                  {inputSource === "db" && "✓"}
                </div>
              </div>
              <div className="blgf-ai-model-desc">
                Use the existing 'LandParcel' table in the Common Database
                schema.
              </div>
            </div>

            <div
              className={`blgf-ai-model-card ${inputSource === "file" ? "active" : ""}`}
              onClick={() => setInputSource("file")}
            >
              <div className="blgf-ai-model-card-header">
                <span className="blgf-ai-model-name">Upload Shapefile/ZIP</span>
                <div className="blgf-ai-checkbox-indicator">
                  {inputSource === "file" && "✓"}
                </div>
              </div>
              <div className="blgf-ai-model-desc">
                Upload new spatial data files to process.
              </div>
            </div>
          </div>
        </div>

        {/* STEP 3 */}
        {inputSource === "db" && (
          <div className="blgf-ai-block">
            <div
              className="blgf-ai-schema-tag"
              style={{ marginBottom: "10px", display: "inline-block" }}
            >
              Schema: {userSchema || "None"}
            </div>

            {!userSchema ? (
              <div className="blgf-ai-helper-text error">
                Please connect to Common Database first.
              </div>
            ) : (
              <>
                <div
                  className="blgf-ai-helper-text"
                  style={{ marginBottom: "10px" }}
                >
                  Ready to process from Common Database
                </div>

                <div className="blgf-ai-label">Select Table</div>
                <select
                  value={tableName}
                  onChange={(e) => setTableName(e.target.value)}
                  className="blgf-ai-select"
                >
                  <option value="LandParcel">LandParcel</option>
                  <option value="Training_Table">Training_Table</option>
                </select>
              </>
            )}
          </div>
        )}

        {inputSource === "file" && (
          <div className="blgf-ai-card">
            <div className="blgf-ai-label">Upload Files</div>
            <div
              style={{
                display: "grid",
                gap: "20px",
                gridTemplateColumns: "1fr 1fr",
              }}
            >
              <div>
                <div className="blgf-ai-helper-text">Option A: Shapefiles</div>
                <input
                  type="file"
                  multiple
                  accept=".shp,.dbf,.shx,.prj"
                  onChange={handleShapefilesChange}
                  className="blgf-ai-select"
                  style={{ marginTop: "5px" }}
                />
              </div>

              <div>
                <div className="blgf-ai-helper-text">Option B: ZIP Archive</div>
                <input
                  type="file"
                  accept=".zip"
                  onChange={handleZipFileChange}
                  className="blgf-ai-select"
                  style={{ marginTop: "5px" }}
                />
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="blgf-ai-block">
            <div
              className="blgf-ai-helper-text error"
              style={{ fontSize: "14px" }}
            >
              {error}
            </div>
          </div>
        )}

        {result && (
          <div className="blgf-ai-result" style={{ marginTop: "20px" }}>
            <div className="blgf-ai-result-header">
              <div className="blgf-ai-modeltitle">Processing Complete</div>
            </div>

            <div className="blgf-ai-stats-row">
              <div className="blgf-ai-stat-badge">
                <strong>Records Processed:</strong> {result.record_count}
              </div>
              <div className="blgf-ai-stat-badge">
                <strong>Model Type:</strong> {result.model_type?.toUpperCase()}
              </div>
            </div>

            <div className="blgf-ai-card">
              <div className="blgf-ai-subtitle2">Downloads & Actions</div>

              <div className="blgf-ai-actions-grid">
                <ul className="blgf-ai-downloads">
                  {result.downloads?.report && (
                    <li>
                      <a
                        href={result.downloads.report}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Download PDF Report
                      </a>
                    </li>
                  )}
                  {result.downloads?.shapefile && (
                    <li>
                      <a
                        href={result.downloads.shapefile}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Download Predicted Shapefile (.zip)
                      </a>
                    </li>
                  )}
                </ul>

                <div className="blgf-ai-action-buttons">
                  {result.downloads?.shapefile_raw && (
                    <button
                      className="blgf-ai-btn-primary wide"
                      onClick={() => {
                        setLoadingFieldName(
                          result.actual_field
                            ? result.actual_field.toUpperCase()
                            : "PREDICTION",
                        );
                        setLoadingMap(true);

                        const rawPath = result.downloads.shapefile_raw;
                        const enc = encodeURIComponent(rawPath);
                        const url = `/api/ai-tools/preview-geojson?file_path=${enc}`;

                        onShowMap({
                          url,
                          label: "Run Saved Model Result",
                          predictionField:
                            result.prediction_field || "prediction",
                          actualField: result.actual_field || null,
                          predictionRange: result.prediction_range || null,
                          actualRange: result.actual_range || null,
                        });

                        setTimeout(() => {
                          setLoadingMap(false);
                        }, 1000);
                      }}
                    >
                      Visualize on Map
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="blgf-ai-footer">
        <div style={{ display: "flex", gap: "10px" }}>
          <button
            onClick={handleReset}
            className="blgf-ai-btn-secondary"
            style={{ width: "30%" }}
          >
            Reset
          </button>
          <button
            onClick={handleRunModel}
            disabled={loading}
            className="blgf-ai-btn-primary"
            style={{ width: "70%" }}
          >
            {loading ? "Processing..." : "Run Model"}
          </button>
        </div>
      </div>
    </div>
  );
}
