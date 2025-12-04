import { useState } from "react";
import { MapContainer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import BaseMapSelector from "./BasemapSelector/BaseMapSelector.jsx";
import SchemaSelector from "./SchemaSelector/SchemaSelector";
import AdminBoundaries from "./AdminBoundaries/AdminBoundaries.jsx";
import Orthophoto from "./Orthophoto/Orthophoto.jsx";
import ParcelLoader from "./ParcelLoader";
import LoadingHandler from "./LoadingHandler";
import Toolbar from "./Toolbar/toolbar.jsx";

import CoordinatesDisplay from "./CoordinatesDisplay/CoordinatesDisplay.jsx";
import MapRefRegisterer from "./MapRefRegister.jsx";
import RightControls from "./RightSideControls.jsx";
import { useNavigate } from "react-router-dom";

function MapView() {
  const [activeTool, setActiveTool] = useState(null);
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const navigate = useNavigate();

  // 🚪 Logout function
  const handleLogout = async () => {
    setIsLoggingOut(true);

    try {
      const token = localStorage.getItem("access_token");

      if (!token) {
        // If no token, just clear and redirect
        localStorage.clear();
        navigate("/login");
        return;
      }

      // Call logout endpoint
      const response = await fetch("http://localhost:8000/auth/logout", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        console.log(data.message); // Log success message

        // Clear all stored data
        localStorage.removeItem("access_token");
        localStorage.removeItem("user_type");
        localStorage.removeItem("username");
        // Add any other keys you're storing

        // Redirect to login
        navigate("/login");
      } else {
        // Even if logout fails on backend, clear local storage
        console.error("Logout failed, but clearing local session");
        localStorage.clear();
        navigate("/login");
      }
    } catch (error) {
      console.error("Logout error:", error);
      // Force logout on client side even if backend fails
      localStorage.clear();
      navigate("/login");
    } finally {
      setIsLoggingOut(false);
    }
  };

  return (
    <MapContainer
      center={[12.8797, 121.774]}
      zoom={6}
      zoomControl={false}
      attributionControl={false}
      style={{
        height: "100vh",
        width: "100%",
        position: "relative",
        zIndex: 0,
      }}
      className="leaflet-map-container"
    >
      {/* 🌍 Base Layers + Map registration */}
      <BaseMapSelector />
      <MapRefRegisterer />
      <ParcelLoader />
      <LoadingHandler />
      <CoordinatesDisplay />

      {/* 🧭 LEFT TOOLBAR (GIS Tools) */}
      <Toolbar />

      {/* 🧩 RIGHT PANEL (Zoom + Tools) */}
      <RightControls activeTool={activeTool} setActiveTool={setActiveTool} />

      {/* 🔘 Tool Panels — only show when active */}
      <SchemaSelector
        isVisible={activeTool === "schema"}
        onClose={() => setActiveTool(null)}
      />
      <Orthophoto
        isVisible={activeTool === "ortho"}
        onClose={() => setActiveTool(null)}
      />
      <AdminBoundaries
        isVisible={activeTool === "admin"}
        onClose={() => setActiveTool(null)}
      />

      {/* 🔵 BOTTOM LEFT BUTTONS */}
      <div className="absolute bottom-4 left-4 z-[1100] flex gap-2">
        <button
          onClick={() => navigate("/ai/viewer")}
          className="bg-[#111] text-white px-3 py-2 rounded-lg border border-gray-600 hover:bg-[#333] transition-colors"
        >
          AI Viewer
        </button>

        {/* 🚪 LOGOUT BUTTON */}
        <button
          onClick={handleLogout}
          disabled={isLoggingOut}
          className="bg-red-600 text-white px-3 py-2 rounded-lg border border-red-700 hover:bg-red-700 disabled:bg-red-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
        >
          {isLoggingOut ? (
            <>
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Logging out...
            </>
          ) : (
            <>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
                />
              </svg>
              Logout
            </>
          )}
        </button>
      </div>
    </MapContainer>
  );
}

export default MapView;
