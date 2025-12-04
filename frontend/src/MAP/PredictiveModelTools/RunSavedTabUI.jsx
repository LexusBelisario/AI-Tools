import React, { useState } from "react";
import API from "../../api.js";

export default function RunSavedTabUI({
  onShowMap,
  userSchema,
  setLoadingMap,
  setLoadingFieldName,
}) {
  const [modelType, setModelType] = useState("lr");
  const [modelFile, setModelFile] = useState(null);
  const [inputSource, setInputSource] = useState("db"); // "db" or "file"

  // File input (for shapefile/zip)
  const [shapefiles, setShapefiles] = useState([]);
  const [zipFile, setZipFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleModelFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith(".pkl")) {
      setModelFile(file);
      setError(null);
    } else {
      setError("Please upload a .pkl model file");
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

    if (!modelFile) {
      setError("Please upload a model file (.pkl)");
      return;
    }

    if (inputSource === "db") {
      if (!userSchema) {
        setError("No schema selected. Please select a region first.");
        return;
      }
    } else {
      if (shapefiles.length === 0 && !zipFile) {
        setError("Please upload shapefile(s) or a ZIP file");
        return;
      }
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("model_type", modelType);
      formData.append("model_file", modelFile);

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

      // ✅ Show map if we have shapefile (DON'T auto-show, let user click button)
      // Just set the result, user can click "Show on Map" button later
    } catch (err) {
      console.error("Run model error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setModelFile(null);
    setShapefiles([]);
    setZipFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="blgf-ai-content">
      {/* Model Type Selection */}
      <div className="blgf-ai-block">
        <div className="blgf-ai-label">Model Type</div>
        <select
          value={modelType}
          onChange={(e) => setModelType(e.target.value)}
          className="blgf-ai-select"
        >
          <option value="lr">Linear Regression</option>
          <option value="rf">Random Forest</option>
          <option value="xgb">XGBoost</option>
        </select>
      </div>

      {/* Model File Upload */}
      <div className="blgf-ai-block">
        <div className="blgf-ai-label">Upload Model File (.pkl)</div>
        <input
          type="file"
          accept=".pkl"
          onChange={handleModelFileChange}
          className="blgf-ai-input-file"
        />
        {modelFile && (
          <div className="blgf-ai-filelist">
            <span>{modelFile.name}</span>
          </div>
        )}
      </div>

      {/* Input Source Selection */}
      <div className="blgf-ai-block">
        <div className="blgf-ai-label">Data Input Mode</div>
        <div className="blgf-ai-mode-radio-group">
          <label
            className={`blgf-ai-mode-radio-label ${inputSource === "db" ? "active" : ""}`}
          >
            <input
              type="radio"
              value="db"
              checked={inputSource === "db"}
              onChange={(e) => setInputSource(e.target.value)}
            />
            <span className="blgf-ai-mode-radio-text">
              Database (JoinedTable)
            </span>
          </label>

          <label
            className={`blgf-ai-mode-radio-label ${inputSource === "file" ? "active" : ""}`}
          >
            <input
              type="radio"
              value="file"
              checked={inputSource === "file"}
              onChange={(e) => setInputSource(e.target.value)}
            />
            <span className="blgf-ai-mode-radio-text">
              Upload File (Shapefile/ZIP)
            </span>
          </label>
        </div>
      </div>

      {/* ✅ Database Mode - Show current schema and JoinedTable */}
      {inputSource === "db" && (
        <div className="blgf-ai-block">
          <div className="blgf-ai-info-box schema">
            📍 <strong>Current Schema:</strong> {userSchema || "Not selected"}
          </div>

          <div className="blgf-ai-info-box success">
            ✅ Using table: <strong>JoinedTable</strong>
          </div>

          {!userSchema && (
            <div className="blgf-ai-info-box warning">
              ⚠️ Please select a region/schema first
            </div>
          )}
        </div>
      )}

      {/* File Input */}
      {inputSource === "file" && (
        <div className="blgf-ai-block">
          <div style={{ marginBottom: "15px" }}>
            <div className="blgf-ai-label">
              Upload Shapefiles (multiple files)
            </div>
            <input
              type="file"
              multiple
              accept=".shp,.dbf,.shx,.prj"
              onChange={handleShapefilesChange}
              className="blgf-ai-input-file"
            />
            {shapefiles.length > 0 && (
              <div className="blgf-ai-filelist">
                {shapefiles.map((f, i) => (
                  <span key={i}>{f.name}</span>
                ))}
              </div>
            )}
          </div>

          <div
            style={{ textAlign: "center", color: "#64748b", margin: "10px 0" }}
          >
            OR
          </div>

          <div>
            <div className="blgf-ai-label">Upload ZIP File</div>
            <input
              type="file"
              accept=".zip"
              onChange={handleZipFileChange}
              className="blgf-ai-input-file"
            />
            {zipFile && (
              <div className="blgf-ai-filelist">
                <span>{zipFile.name}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="blgf-ai-block">
          <div className="blgf-ai-info-box warning">⚠️ {error}</div>
        </div>
      )}

      {result && (
        <div className="blgf-ai-block">
          <div className="blgf-ai-info-box success">
            ✅ Model run successful! Processed {result.record_count} records
          </div>

          <div className="blgf-ai-card" style={{ marginTop: "15px" }}>
            <div className="blgf-ai-subtitle2">Downloads</div>
            <ul className="blgf-ai-downloads">
              {result && (
                <div className="blgf-ai-block">
                  <div className="blgf-ai-info-box success">
                    ✅ Model run successful! Processed {result.record_count}{" "}
                    records
                  </div>

                  <div className="blgf-ai-card" style={{ marginTop: "15px" }}>
                    <div className="blgf-ai-subtitle2">Downloads</div>
                    <ul className="blgf-ai-downloads">
                      {result.downloads?.report && (
                        <li>
                          <a
                            href={result.downloads.report}
                            target="_blank"
                            rel="noreferrer"
                          >
                            📄 PDF Report
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
                            🗺️ Predicted Shapefile (.zip)
                          </a>
                        </li>
                      )}
                    </ul>

                    {/* ✅ ADD Save to Database Button for Run Saved Model */}
                    {result.downloads?.shapefile_raw && (
                      <button
                        className="blgf-ai-btn-secondary wide"
                        style={{ marginTop: "10px" }}
                        onClick={() => {
                          // TODO: Implement save for run saved model
                          // Will need to pass setSaveModalOpen and setSaveConfig as props
                          console.log("Save run results to DB");
                        }}
                      >
                        💾 Save to Database
                      </button>
                    )}
                  </div>
                </div>
              )}
            </ul>

            {/* ✅ ADD SHOW ON MAP BUTTON */}
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

                  console.log("🗺️ Loading map with URL:", url);
                  console.log("📊 Prediction range:", result.prediction_range);
                  console.log("📊 Actual range:", result.actual_range);
                  console.log("📊 Actual field:", result.actual_field);

                  onShowMap({
                    url,
                    label: `Run ${modelType.toUpperCase()} Model`,
                    predictionField: result.prediction_field || "prediction",
                    actualField: result.actual_field || null,
                    predictionRange: result.prediction_range || null,
                    actualRange: result.actual_range || null,
                  });

                  setTimeout(() => {
                    setLoadingMap(false);
                  }, 1000);
                }}
              >
                🗺️ Show On Map
              </button>
            )}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="blgf-ai-btnrow">
        <button
          onClick={handleReset}
          className="blgf-ai-btn-secondary"
          style={{ marginRight: "10px" }}
        >
          Reset
        </button>
        <button
          onClick={handleRunModel}
          disabled={loading}
          className="blgf-ai-btn-primary"
        >
          {loading ? "Running..." : "Run Model"}
        </button>
      </div>
    </div>
  );
}
