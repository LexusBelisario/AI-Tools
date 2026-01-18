import { Routes, Route, Navigate } from "react-router-dom";
import AIToolsModal from "./MAP/PredictiveModelTools/AIToolsModal";
import { useEffect, useState } from "react";

export default function App() {
  const [panelOpen, setPanelOpen] = useState(true);
  const [token, setToken] = useState(
    () => localStorage.getItem("access_token") || ""
  );
  const [shouldDisconnect, setShouldDisconnect] = useState(false);

  const handleShowMap = (payload) => {
    console.log("🗺️ Show on map:", payload);
  };

  // Change this if GIS runs on a different origin
  const GIS_ORIGIN = import.meta.env.VITE_GIS_ORIGIN || "http://localhost:5173";
  const API = import.meta.env.VITE_API_URL || "http://localhost:8001";

  useEffect(() => {
    const handler = async (event) => {
      // Security: only accept messages from GIS
      if (event.origin !== GIS_ORIGIN) {
        console.warn("⚠️ Message from untrusted origin:", event.origin);
        return;
      }

      const { type, token: receivedToken } = event.data;

      // Handle token authentication
      if (type === "AI_TOOLS_AUTH") {
        if (!receivedToken) return;

        console.log("✅ Received token from GIS");
        localStorage.setItem("access_token", receivedToken);
        setToken(receivedToken);
        setShouldDisconnect(false); // Reset disconnect flag
      }

      // Handle disconnect request
      if (type === "AI_TOOLS_DISCONNECT") {
        console.log("🔌 Disconnect requested by GIS");

        // Call disconnect API
        try {
          const currentToken = localStorage.getItem("access_token");
          if (currentToken) {
            const response = await fetch(`${API}/api/common/disconnect`, {
              method: "POST",
              headers: {
                Authorization: `Bearer ${currentToken}`,
              },
            });

            if (response.ok) {
              console.log("✅ Disconnected from Common DB");
            } else {
              console.warn("⚠️ Disconnect API returned:", response.status);
            }
          }
        } catch (err) {
          console.error("❌ Disconnect error:", err);
        }

        // Clear local state
        localStorage.removeItem("access_token");
        setToken("");
        setShouldDisconnect(true); // Signal to child components
      }
    };

    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, [GIS_ORIGIN, API]);

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
            token={token}
            shouldDisconnect={shouldDisconnect} // Pass disconnect signal
          />
        }
      />
    </Routes>
  );
}
