import { Routes, Route, Navigate } from "react-router-dom";
import { useState, useEffect } from "react";
import Map from "./MAP/Map.jsx";
import LoginPage from "./user_login/login_page.jsx";
import RegisterPage from "./user_login/register_page.jsx";
import { SchemaProvider } from "./MAP/SchemaContext.jsx";
import Admin from "./ADMIN/admin.jsx";
import AllAIMapView from "./MAP/PredictiveModelTools/AllMapView.jsx";
import LinearRegressionView from "./MAP/PredictiveModelTools/LinearRegression/LinearRegression.jsx";

export const API_URL =
  process.env.NODE_ENV === "production"
    ? "https://blgf-gis-webapp-nq0r.onrender.com"
    : "http://localhost:8000";

// ✅ Override fetch globally
const originalFetch = window.fetch;
window.fetch = function (...args) {
  const [url, options = {}] = args;

  if (url.includes("/api/")) {
    const token = localStorage.getItem("accessToken");
    if (token) {
      options.headers = {
        ...options.headers,
        Authorization: `Bearer ${token}`,
      };
    }
  }

  return originalFetch(url, options).then((response) => {
    if (response.status === 401 && url.includes("/api/")) {
      localStorage.removeItem("accessToken");
      window.location.href = "/login";
    }
    return response;
  });
};

// ✅ Protected routes
const ProtectedRoute = ({ children, adminOnly = false }) => {
  const token = localStorage.getItem("accessToken");

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  if (adminOnly) {
    try {
      const payload = JSON.parse(atob(token.split(".")[1]));
      if (payload.user_type !== "admin") {
        return <Navigate to="/map" replace />;
      }
    } catch (e) {
      console.error("Error decoding token:", e);
      return <Navigate to="/login" replace />;
    }
  }

  return children;
};

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem("accessToken");
    setIsAuthenticated(!!token);
  }, []);

  return (
    <SchemaProvider>
      <Routes>
        {/* Root route */}
        <Route path="/" element={<Navigate to="/login" replace />} />
        {/* Public */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} /> {/* ✅ added */}
        {/* User Map */}
        <Route
          path="/map"
          element={
            <ProtectedRoute>
              <Map />
            </ProtectedRoute>
          }
        />
        <Route
          path="/ai/viewer"
          element={
            <ProtectedRoute>
              <AllAIMapView />
            </ProtectedRoute>
          }
        />
        <Route
          path="/ai/linear"
          element={
            <ProtectedRoute>
              <LinearRegressionView />
            </ProtectedRoute>
          }
        />
        {/* Admin */}
        <Route
          path="/admin/*"
          element={
            <ProtectedRoute adminOnly={true}>
              <Admin />
            </ProtectedRoute>
          }
        />
      </Routes>
    </SchemaProvider>
  );
};

export default App;
