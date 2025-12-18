import React, { useState, useEffect } from "react";
import API from "../../api.js";

export default function RunSavedTabUI({
  onShowMap,
  userSchema,
  setLoadingMap,
  setLoadingFieldName,
}) {
  // === MODEL INPUT STATE ===
  const [modelSource, setModelSource] = useState("upload"); // "upload" or "db"
  const [modelFile, setModelFile] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedDbModel, setSelectedDbModel] = useState("");

  // === DATA INPUT STATE ===
  const [inputSource, setInputSource] = useState("db"); // "db" or "file"
  const [shapefiles, setShapefiles] = useState([]);
  const [zipFile, setZipFile] = useState(null);

  // === UI STATE ===
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Load models from DB when "Database" source is selected
  useEffect(() => {
    if (modelSource === "db") {
      loadAvailableModels();
    }
  }, [modelSource]);

  const loadAvailableModels = async () => {
    try {
      // NOTE: Ensure this endpoint exists in your backend
      const res = await fetch(`${API}/ai-tools/list-models`, {
        method: "GET", // or POST depending on your backend
      });
      if (res.ok) {
        const data = await res.json();
        setAvailableModels(data.models || []);
      } else {
        console.error("Failed to load models");
      }
    } catch (err) {
      console.error("Error loading models:", err);
    }
  };

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

    // Validation
    if (modelSource === "upload" && !modelFile) {
      setError("Please upload a model file (.pkl)");
      return;
    }
    if (modelSource === "db" && !selectedDbModel) {
      setError("Please select a model from the database");
      return;
    }

    if (inputSource === "db" && !userSchema) {
      setError("No schema selected. Please select a region first.");
      return;
    }
    if (inputSource === "file" && shapefiles.length === 0 && !zipFile) {
      setError("Please upload shapefile(s) or a ZIP file");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();

      // Append Model Info
      formData.append("model_source", modelSource);
      if (modelSource === "upload") {
        formData.append("model_file", modelFile);
      } else {
        formData.append("model_name", selectedDbModel);
      }

      // Append Data Info
      if (inputSource === "db") {
        formData.append("schema", userSchema);
        formData.append("table_name", "JoinedTable");
      } else {
        if (zipFile) {
          formData.append("zip_file", zipFile);
        } else {
          shapefiles.forEach((file) => {
            formData.append("shapefiles", file);
          });
        }
      }

      const response = await fetch(`${API}/ai-tools/run-saved-model`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to run model");
      }

      console.log("✅ Run saved model response:", data);
      setResult(data);
    } catch (err) {
      console.error("Run model error:", err);
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
  };

  return (
    <div className="blgf-ai-content">
      <div className="blgf-ai-data-grid" style={{ gridTemplateColumns: "1fr" }}>
        {/* === STEP 1: SELECT MODEL === */}
        <div className="blgf-ai-block">
          <div className="blgf-ai-label">1. Select Model Source</div>

          <div className="blgf-ai-models-grid" style={{ marginBottom: "20px" }}>
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

            <div
              className={`blgf-ai-model-card ${modelSource === "db" ? "active" : ""}`}
              onClick={() => setModelSource("db")}
            >
              <div className="blgf-ai-model-card-header">
                <span className="blgf-ai-model-name">From Database</span>
                <div className="blgf-ai-checkbox-indicator">
                  {modelSource === "db" && "✓"}
                </div>
              </div>
              <div className="blgf-ai-model-desc">
                Select a saved model from the server.
              </div>
            </div>
          </div>

          {/* Model Selection UI based on Source */}
          <div className="blgf-ai-card">
            {modelSource === "upload" ? (
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
                    <span>📦 {modelFile.name}</span>
                  </div>
                )}
              </>
            ) : (
              <>
                <div className="blgf-ai-label">Select Saved Model</div>
                <select
                  value={selectedDbModel}
                  onChange={(e) => setSelectedDbModel(e.target.value)}
                  className="blgf-ai-select"
                  disabled={availableModels.length === 0}
                >
                  <option value="">-- Select a model --</option>
                  {availableModels.map((m, i) => (
                    <option key={i} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
                {availableModels.length === 0 && (
                  <div className="blgf-ai-helper-text">
                    {loading ? "Loading models..." : "No saved models found."}
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* === STEP 2: SELECT DATA SOURCE === */}
        <div className="blgf-ai-block">
          <div className="blgf-ai-label">2. Select Data Source</div>
          <div className="blgf-ai-models-grid">
            <div
              className={`blgf-ai-model-card ${inputSource === "db" ? "active" : ""}`}
              onClick={() => setInputSource("db")}
            >
              <div className="blgf-ai-model-card-header">
                <span className="blgf-ai-model-name">
                  Database (JoinedTable)
                </span>
                <div className="blgf-ai-checkbox-indicator">
                  {inputSource === "db" && "✓"}
                </div>
              </div>
              <div className="blgf-ai-model-desc">
                Use the existing 'JoinedTable' in the selected schema.
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

        {/* === STEP 3: CONTEXT SPECIFIC INPUTS === */}
        {inputSource === "db" && (
          <div className="blgf-ai-block">
            <div
              className="blgf-ai-schema-tag"
              style={{ marginBottom: "10px", display: "inline-block" }}
            >
              📍 Schema: {userSchema || "None"}
            </div>

            {!userSchema ? (
              <div className="blgf-ai-helper-text error">
                ⚠️ Please select a region/schema from the main map first.
              </div>
            ) : (
              <div className="blgf-ai-helper-text" style={{ color: "#4ade80" }}>
                ✅ Ready to process JoinedTable
              </div>
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
                {shapefiles.length > 0 && (
                  <div
                    className="blgf-ai-filelist"
                    style={{ marginTop: "10px" }}
                  >
                    {shapefiles.map((f, i) => (
                      <div key={i}>📄 {f.name}</div>
                    ))}
                  </div>
                )}
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
                {zipFile && (
                  <div
                    className="blgf-ai-filelist"
                    style={{ marginTop: "10px" }}
                  >
                    <div>🤐 {zipFile.name}</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* === ERROR MESSAGE === */}
        {error && (
          <div className="blgf-ai-block">
            <div
              className="blgf-ai-helper-text error"
              style={{ fontSize: "14px" }}
            >
              ⚠️ {error}
            </div>
          </div>
        )}

        {/* === RESULTS SECTION === */}
        {result && (
          <div className="blgf-ai-result" style={{ marginTop: "20px" }}>
            <div className="blgf-ai-result-header">
              <div className="blgf-ai-modeltitle">Processing Complete</div>
            </div>

            <div className="blgf-ai-stats-row">
              <div className="blgf-ai-stat-badge">
                <strong>Records Processed:</strong> {result.record_count}
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
                        📄 Download PDF Report
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
                        🗺️ Download Predicted Shapefile (.zip)
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
                            : "PREDICTION"
                        );
                        setLoadingMap(true);

                        const rawPath = result.downloads.shapefile_raw;
                        const enc = encodeURIComponent(rawPath);
                        const url = `/api/ai-tools/preview-geojson?file_path=${enc}`;

                        onShowMap({
                          url,
                          label: `Run Saved Model Result`,
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

                  {result.downloads?.shapefile_raw && (
                    <button
                      className="blgf-ai-btn-secondary wide"
                      onClick={() => {
                        console.log("Saving to DB logic here...");
                        // Trigger save modal logic if props are passed
                      }}
                    >
                      Save to Database
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* === FOOTER ACTION BUTTONS === */}
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
