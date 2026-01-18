import React, { useState } from "react";
import API from "../../api.js";

export default function SaveToDBModal({
  isOpen,
  onClose,
  userSchema,
  token,
  saveType,
  modelType,
  modelPath,
  dependentVar,
  independentVars,
  shapefilePath,
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  if (!isOpen) return null;

  const handleSave = async () => {
    setError(null);
    setSuccess(null);
    setLoading(true);

    try {
      if (!token) {
        throw new Error(
          "No authentication token. Please refresh your session."
        );
      }
      if (!userSchema) {
        throw new Error("No schema selected.");
      }

      const fd = new FormData();

      // IMPORTANT: This now saves to GIS Database (manual/optional)
      let url = "";
      if (saveType === "model") {
        if (!modelPath) throw new Error("Missing model path.");

        fd.append("model_path", modelPath);
        fd.append("model_type", modelType || "unknown");
        fd.append("dependent_var", dependentVar || "");

        const featuresPayload = Array.isArray(independentVars)
          ? independentVars
          : [];
        fd.append("features_json", JSON.stringify(featuresPayload));

        // 👇 NEW: Save to GIS Database (not Common DB)
        url = `${API}/ai-tools/save-model-to-gis-db`;
      } else {
        if (!shapefilePath) throw new Error("Missing shapefile path.");

        fd.append("shapefile_path", shapefilePath);
        fd.append("model_type", modelType || "unknown");
        fd.append("save_type", saveType || "training");

        // 👇 NEW: Save to GIS Database (not Common DB)
        url = `${API}/ai-tools/save-predictions-to-gis-db`;
      }

      const res = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "X-Target-Schema": userSchema,
        },
        body: fd,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || data.detail || "Save failed");
      }

      setSuccess({
        message: "Successfully saved to GIS Database (local).",
        table: data.table_name,
        schema: data.schema,
      });

      setTimeout(() => {
        onClose();
      }, 1200);
    } catch (e) {
      setError(e.message || "Save failed");
    } finally {
      setLoading(false);
    }
  };

  const title =
    saveType === "model"
      ? "Save Trained Model to Common Table Database"
      : "Save Prediction Results to Common Table Database";

  const destLabel =
    saveType === "model" ? "ai_trained_models" : "PredictionResults";

  return (
    <>
      <div
        className="fixed inset-0 bg-black/60 z-[1400] transition-opacity"
        onClick={onClose}
      />

      <div className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[520px] bg-[#0f172a] border border-gray-700 rounded-xl shadow-2xl z-[1500] p-6">
        <h3 className="text-xl font-bold text-[#F7C800] mb-4">{title}</h3>

        <div className="mb-6 p-3 bg-[#1e293b] border border-gray-600 rounded-lg">
          <div className="text-xs text-gray-400 mb-1">Destination:</div>
          <div className="text-sm font-semibold text-[#F7C800]">
            {userSchema} → {destLabel}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Model: {(modelType || "unknown").toUpperCase()}
          </div>
        </div>

        {saveType === "model" && modelPath && (
          <div className="mb-6 p-3 bg-blue-900/20 border border-blue-500/30 rounded-lg">
            <div className="text-xs text-gray-300">
              Source: {String(modelPath)}
            </div>
          </div>
        )}

        {error && (
          <div className="mb-4 p-3 bg-red-900/20 border border-red-500 rounded text-sm text-red-400">
            {error}
          </div>
        )}

        {success && (
          <div className="mb-4 p-3 bg-green-900/20 border border-green-500 rounded text-sm text-green-400">
            {success.message}
            {success.id != null && (
              <div className="text-xs mt-1 text-green-300">
                ID: {success.id} | Schema: {success.schema}
              </div>
            )}
          </div>
        )}

        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 bg-gray-700 hover:bg-gray-600 text-white font-semibold py-2.5 px-4 rounded-lg transition"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={loading || !userSchema || !token}
            className="flex-1 bg-[#F7C800] hover:bg-[#d4ad00] text-black font-semibold py-2.5 px-4 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </>
  );
}
