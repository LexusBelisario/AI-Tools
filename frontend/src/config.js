const isLocalhost = (hostname) =>
  hostname === "localhost" || hostname === "127.0.0.1";

export const AI_API_URL = (() => {
  // ✅ Highest priority: explicit env override
  const fromEnv = import.meta.env.VITE_AI_API_URL;
  if (fromEnv && String(fromEnv).trim()) return String(fromEnv).trim();

  const { protocol, hostname } = window.location;

  // ✅ Local dev: AI Tools backend runs on 8001
  if (isLocalhost(hostname)) {
    return "http://localhost:8001";
  }

  // ✅ Production default:
  // If you deploy behind a reverse proxy, you can set VITE_AI_API_URL="/"
  // or point to your actual backend domain.
  return `${protocol}//${hostname}`;
})();

export const API_BASE = `${AI_API_URL}/api`;
