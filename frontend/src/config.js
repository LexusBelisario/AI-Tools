// config.js - Fully dynamic, no hardcoding!
export const API_URL = (() => {
  const { protocol, hostname, port } = window.location;

  // If localhost, use localhost backend
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return "http://localhost:8000";
  }

  // For production, use same hostname but port 8000
  const baseUrl = `${protocol}//${hostname}`;
  return `${baseUrl}:8000`;
})();

export const API_BASE = `${API_URL}/api`;
