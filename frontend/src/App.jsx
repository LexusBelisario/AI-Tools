import { Routes, Route, Navigate } from "react-router-dom";
import AIToolsModal from "./AITools/AIToolsModal.jsx";
import { useEffect, useState } from "react";

export default function App() {
  const [panelOpen, setPanelOpen] = useState(true);
  const [token, setToken] = useState(
    () => localStorage.getItem("access_token") || "",
  );
  const [shouldDisconnect, setShouldDisconnect] = useState(false);
  const [preSelectedRunModel, setPreSelectedRunModel] = useState(null);
  const [openRunSaved, setOpenRunSaved] = useState(false);

  const handleShowMap = (payload) => {
    console.log("🗺️ Show on map:", payload);
  };

  // Comma-separated list of trusted origins that can send tokens.
  // Set VITE_TRUSTED_ORIGINS in .env to add more, e.g.:
  //   VITE_TRUSTED_ORIGINS=http://localhost:5173,https://partner-app.example.com
  // "null" is included to allow local HTML file testing (file:// origin)
  const TRUSTED_ORIGINS = (
    import.meta.env.VITE_TRUSTED_ORIGINS ||
    "http://localhost:5173,http://localhost:5174,http://localhost:9000,https://cama-core-14282293226.asia-southeast1.run.app,http://35.194.255.28:8000,http://localhost:8000,null"
  )
    .split(",")
    .map((o) => o.trim());

  // When VITE_API_URL is empty (Docker/production), API calls use relative
  // paths which resolve to the same origin the frontend is served from.
  const API = import.meta.env.VITE_API_URL || 
  (window.location.port === "8003" ? "" : "http://localhost:8001");

  // Notify parent iframe that AI Tools is closing
  const handleClose = () => {
    if (window.parent !== window) {
      window.parent.postMessage({ type: "AI_TOOLS_CLOSE" }, "*");
    }
    setPanelOpen(false);
  };

  useEffect(() => {
    const handler = async (event) => {
      // Security: only accept messages from trusted origins
      if (!TRUSTED_ORIGINS.includes(event.origin)) {
        console.warn("⚠️ Message from untrusted origin:", event.origin);
        return;
      }

      const { type, token: receivedToken } = event.data;

      // Handle token authentication
      if (type === "AI_TOOLS_AUTH") {
        if (!receivedToken) return;
        console.log("✅ Received token from:", event.origin);
        localStorage.setItem("access_token", receivedToken);
        setToken(receivedToken);
        setShouldDisconnect(false);
        setPreSelectedRunModel(null);
        setOpenRunSaved(false);
      }

      // Handle open run saved tab with pre-selected model
      if (type === "AI_TOOLS_OPEN_RUN_SAVED") {
        const { model_name } = event.data;
        console.log("📨 Received AI_TOOLS_OPEN_RUN_SAVED:", model_name);
        if (model_name) setPreSelectedRunModel(model_name);
        setOpenRunSaved(true);
        setPanelOpen(true);
      }

      // Handle disconnect request
      if (type === "AI_TOOLS_DISCONNECT") {
        console.log("🔌 Disconnect requested by:", event.origin);

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

        localStorage.removeItem("access_token");
        setToken("");
        setShouldDisconnect(true);
      }
    };

    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, [TRUSTED_ORIGINS, API]);

  return (
    <Routes>
      <Route path="/" element={<Navigate to="/ai/viewer" replace />} />
      <Route
        path="/ai/viewer"
        element={
          <AIToolsModal
            isOpen={panelOpen}
            onClose={handleClose}
            onShowMap={handleShowMap}
            token={token}
            shouldDisconnect={shouldDisconnect}
            preSelectedRunModel={preSelectedRunModel}
            openRunSaved={openRunSaved}
            onPreSelectedRunModelConsumed={() => setPreSelectedRunModel(null)}
          />
        }
      />
    </Routes>
  );
}