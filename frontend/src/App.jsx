import { Routes, Route, Navigate } from "react-router-dom";
import AIToolsModal from "./MAP/PredictiveModelTools/AIToolsModal";
import { useEffect, useState } from "react";

export const API_URL =
  process.env.NODE_ENV === "production"
    ? "https://your-ai-tools-backend-url"
    : "http://localhost:8001";

export default function App() {
  const [panelOpen, setPanelOpen] = useState(true);
  const handleShowMap = (payload) => {
    console.log("🗺️ Show on map:", payload);
  };
  const GIS_ORIGIN = import.meta.env.VITE_GIS_ORIGIN || "http://localhost:5173";

  useEffect(() => {
    const handler = (event) => {
      if (event.origin !== GIS_ORIGIN) return;
      if (event.data?.type !== "AI_TOOLS_AUTH") return;

      const token = event.data?.token;
      if (!token) return;

      localStorage.setItem("access_token", token);
    };

    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, []);

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
