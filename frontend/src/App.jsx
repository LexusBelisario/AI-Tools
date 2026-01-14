import { Routes, Route, Navigate } from "react-router-dom";
import AIToolsModal from "./MAP/PredictiveModelTools/AIToolsModal";
import { useEffect, useState } from "react";

export default function App() {
  const [panelOpen, setPanelOpen] = useState(true);
  const [token, setToken] = useState(
    () => localStorage.getItem("access_token") || ""
  );

  const handleShowMap = (payload) => {
    console.log("🗺️ Show on map:", payload);
  };

  // Change this if GIS runs on a different origin
  const GIS_ORIGIN = import.meta.env.VITE_GIS_ORIGIN || "http://localhost:5173";

  useEffect(() => {
    const handler = (event) => {
      // Security: only accept messages from GIS
      if (event.origin !== GIS_ORIGIN) return;

      if (event.data?.type !== "AI_TOOLS_AUTH") return;

      const t = event.data?.token;
      if (!t) return;

      localStorage.setItem("access_token", t);
      setToken(t); // ✅ crucial: re-render immediately
    };

    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, [GIS_ORIGIN]);

  return (
    <Routes>
      <Route path="/" element={<Navigate to="/ai/viewer" replace />} />
      <Route
        path="/ai/viewer"
        element={
          <AIToolsModal
            isOpen={panelOpen}
            onClose={() => setPanelOpen(false)}
            onShowMap={handleShowMap}
            token={token} // ✅ pass token
          />
        }
      />
    </Routes>
  );
}
