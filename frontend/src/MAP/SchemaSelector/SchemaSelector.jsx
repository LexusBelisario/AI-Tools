// SchemaSelector.jsx
import { useState, useEffect, useRef } from "react";
import { ApiService } from "../../api_service.js";
import { useSchema } from "../SchemaContext";
import { useMap } from "react-leaflet";

const SchemaSelector = () => {
  const [schemas, setSchemas] = useState([]);
  const [selectedSchema, setSelectedSchema] = useState("");
  const [loading, setLoading] = useState(true);
  const [userAccess, setUserAccess] = useState(null);
  const [error, setError] = useState(null);
  const { setSchema } = useSchema();
  const map = useMap();
  const hasZoomedToProvince = useRef(false);
  const hasAutoSelected = useRef(false);

  const normalize = (s) =>
    (s ?? "")
      .toString()
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");

  // ======================================================
  // 🔹 Helper: Browser Warning (license / invalid location)
  // ======================================================
  const showLicenseWarning = (message) => {
    const existing = document.getElementById("license-warning");
    if (existing) existing.remove();

    const div = document.createElement("div");
    div.id = "license-warning";
    div.style.cssText = `
      position: fixed;
      top: 25px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(220,53,69,0.95);
      color: #fff;
      padding: 10px 20px;
      border-radius: 6px;
      font-weight: 600;
      z-index: 9999;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
      font-family: sans-serif;
    `;
    div.textContent = message || "Invalid location — outside licensed LGU area.";
    document.body.appendChild(div);

    setTimeout(() => div.remove(), 6000);
  };

  // ======================================================
  // 🔹 Helper: Fetch and zoom to provincial bounds
  // ======================================================
  const fetchProvincialBoundsAndZoom = async () => {
    try {
      const res = await ApiService.get("/province/provincial-bounds");
      if (res?.status === "success" && Array.isArray(res.bounds) && res.bounds.length === 4) {
        const [xmin, ymin, xmax, ymax] = res.bounds;
        const bounds = [
          [ymin, xmin],
          [ymax, xmax],
        ];
        console.log(`🗺️ Smooth zoom to provincial bounds (${res.prov_code}):`, bounds);

        const center = [(ymin + ymax) / 2, (xmin + xmax) / 2];
        const zoom = map.getBoundsZoom(bounds, false);
        map.flyTo(center, zoom, { animate: true, duration: 1.5 });
      } else {
        console.warn("⚠️ Invalid or missing provincial bounds:", res);
      }
    } catch (err) {
      console.error("❌ Failed to fetch provincial bounds:", err);
    }
  };

  // ======================================================
  // 🔹 Load schemas (and always zoom to provincial bounds)
  // ======================================================
  useEffect(() => {
    const loadSchemas = async () => {
      try {
        setLoading(true);
        setError(null);

        const data = await ApiService.get("/list-schemas");
        console.log("🔍 Schemas response:", data);

        if (data?.schemas) {
          setSchemas(data.schemas);
          setUserAccess(data.user_access);

          // ✅ Always zoom to province once
          if (!hasZoomedToProvince.current) {
            await fetchProvincialBoundsAndZoom();
            hasZoomedToProvince.current = true;
          }

          // --- Determine municipal access
          const municipalAccessRaw =
            data.user_access?.municipal_access ?? data.user_access?.municipal ?? "";
          const municipalAccess = municipalAccessRaw.toString().trim();
          const isAllAccess = /^all$/i.test(municipalAccess);

          if (!hasAutoSelected.current && !isAllAccess && data.schemas.length > 0) {
            const accessKey = normalize(municipalAccess);
            const targetSchema =
              data.schemas.find((s) => normalize(s).startsWith(accessKey)) ||
              data.schemas.find((s) => normalize(s).startsWith(accessKey.slice(0, 9))) ||
              null;

            if (targetSchema) {
              console.log(`✅ Auto-selecting schema: ${targetSchema}`);
              setSelectedSchema(targetSchema);
              setSchema(targetSchema);
              hasAutoSelected.current = true;
            }
          }
        }
      } catch (err) {
        console.error("❌ Failed to load schemas:", err);
        setError("Failed to load schemas. Please try again.");
        setSchemas([]);
      } finally {
        setLoading(false);
      }
    };

    loadSchemas();
  }, [setSchema, map]);

  // ======================================================
  // 🔹 Handle manual schema change
  // ======================================================
  const handleSchemaChange = (schema) => {
    console.log("🔄 Manual schema change:", schema);
    setSelectedSchema(schema);
    setSchema(schema);
  };

  // ======================================================
  // 🔹 Fetch municipal bounds & enforce license
  // ======================================================
  useEffect(() => {
    if (!selectedSchema) return;

    const fetchBoundsAndZoom = async () => {
      try {
        console.log("📍 Fetching bounds for schema:", selectedSchema);
        const res = await ApiService.get(`/municipal-bounds?schema=${selectedSchema}`);

        // 🚫 License check failed
        if (res?.status === "invalid_bounds") {
          console.warn("🚫 License violation detected:", res.message);
          showLicenseWarning(res.message);

          // Reset map and stop everything else
          map.setView([12.8797, 121.7740], 7); // Philippines view
          return;
        }

        // ✅ Valid schema bounds
        if (res?.status === "success" && Array.isArray(res.bounds) && res.bounds.length === 4) {
          const [xmin, ymin, xmax, ymax] = res.bounds;
          const bounds = [
            [ymin, xmin],
            [ymax, xmax],
          ];
          console.log(`📦 Zoom to bounds of ${selectedSchema}:`, bounds);

          const center = [(ymin + ymax) / 2, (xmin + xmax) / 2];
          const zoom = map.getBoundsZoom(bounds, false);
          map.flyTo(center, zoom, { animate: true, duration: 1.5 });
        } else {
          console.warn("⚠️ Bounds not found or invalid:", res);
        }
      } catch (err) {
        console.error("❌ Failed to fetch bounds:", err);
      }
    };

    fetchBoundsAndZoom();
  }, [selectedSchema, map]);

  // ======================================================
  // 🔹 Expose state globally for UI panel
  // ======================================================
  useEffect(() => {
    window._schemaSelectorData = {
      schemas,
      selectedSchema,
      loading,
      error,
      userAccess,
    };
  }, [schemas, selectedSchema, loading, error, userAccess]);

  useEffect(() => {
    window._handleSchemaChange = handleSchemaChange;
    return () => {
      delete window._handleSchemaChange;
      delete window._schemaSelectorData;
    };
  }, [selectedSchema]);

  return null;
};

export default SchemaSelector;
