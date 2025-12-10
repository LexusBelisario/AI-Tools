import { Routes, Route, Navigate } from "react-router-dom";
import { useState } from "react";
import AIToolsModal from "./MAP/PredictiveModelTools/AIToolsModal";

export const API_URL =
  process.env.NODE_ENV === "production"
    ? "https://your-ai-tools-backend-url"
    : "http://localhost:8000";

export default function App() {
  const [panelOpen, setPanelOpen] = useState(true); 
  const handleShowMap = (payload) => {
    console.log("🗺️ Show on map:", payload);
  };

  return (
    <Routes>
      {/* Default redirect */}
      <Route path="/" element={<Navigate to="/ai/viewer" replace />} />

      {/* AI Tools Viewer */}
      <Route
        path="/ai/viewer"
        element={
          <AIToolsModal
            isOpen={panelOpen}
            onClose={() => setPanelOpen(false)}
            onShowMap={handleShowMap}
          />
        }
      />
    </Routes>
  );
}
