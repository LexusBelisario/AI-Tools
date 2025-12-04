import React, { useState } from "react";
import API from "../../api.js";

export default function SaveToDBModal({ 
  isOpen, 
  onClose, 
  shapefilePath,
  userSchema,
  saveType, // "training" or "run"
  modelType, // for training: "lr" | "rf" | "xgb"
}) {
  const [destination, setDestination] = useState("local");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  if (!isOpen) return null;

  const handleSave = async () => {
    setError(null);
    setSuccess(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("shapefile_path", shapefilePath);
      formData.append("destination", destination);
      
      if (destination === "local") {
        if (!userSchema) {
          setError("No schema selected. Please select a region first.");
          setLoading(false);
          return;
        }
        formData.append("schema", userSchema);
      }

      let endpoint = "";
      if (saveType === "training") {
        formData.append("model_type", modelType);
        endpoint = `${API}/ai-tools/save-training-result`;
      } else {
        endpoint = `${API}/ai-tools/save-run-result`;
      }

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to save to database");
      }

      setSuccess(data);
      
      setTimeout(() => {
        onClose();
      }, 2000);

    } catch (err) {
      console.error("Save to DB error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 z-[1400] transition-opacity"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[450px] bg-[#0f172a] border border-gray-700 rounded-xl shadow-2xl z-[1500] p-6">
        <h3 className="text-xl font-bold text-[#F7C800] mb-4">
          Save to Database
        </h3>

        {/* Destination Selection */}
        <div className="mb-6">
          <div className="text-sm font-semibold text-gray-300 mb-3">
            Select Destination
          </div>
          
          <div className="space-y-3">
            <label
              className={`block p-4 border-2 rounded-lg cursor-pointer transition ${
                destination === "local"
                  ? "border-[#F7C800] bg-[#F7C800]/10"
                  : "border-gray-600 hover:border-gray-500"
              }`}
            >
              <input
                type="radio"
                value="local"
                checked={destination === "local"}
                onChange={(e) => setDestination(e.target.value)}
                className="mr-3"
              />
              <span className="text-white font-medium">Local Database</span>
              <div className="text-xs text-gray-400 ml-6 mt-1">
                Save to: <span className="text-[#F7C800]">{userSchema || "No schema selected"}"JoinedTable"</span>
                {saveType === "run" && " → JoinedTable"}
              </div>
            </label>

            <label
              className={`block p-4 border-2 rounded-lg cursor-pointer transition ${
                destination === "shared"
                  ? "border-[#F7C800] bg-[#F7C800]/10"
                  : "border-gray-600 hover:border-gray-500"
              }`}
            >
              <input
                type="radio"
                value="shared"
                checked={destination === "shared"}
                onChange={(e) => setDestination(e.target.value)}
                className="mr-3"
              />
              <span className="text-white font-medium">SharedDatabase</span>
              <div className="text-xs text-gray-400 ml-6 mt-1">
                Save to: <span className="text-blue-400">JoinedTable</span>
                {saveType === "run" && " → JoinedTable"}
              </div>
            </label>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-4 p-3 bg-red-900/20 border border-red-500 rounded text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Success */}
        {success && (
          <div className="mb-4 p-3 bg-green-900/20 border border-green-500 rounded text-sm text-green-400">
            ✓ Saved successfully! ({success.record_count} records)
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 bg-gray-700 hover:bg-gray-600 text-white font-semibold py-2.5 px-4 rounded-lg transition"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={loading || (destination === "local" && !userSchema)}
            className="flex-1 bg-[#F7C800] hover:bg-[#d4ad00] text-black font-semibold py-2.5 px-4 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Saving..." : "Save to Database"}
          </button>
        </div>
      </div>
    </>
  );
}