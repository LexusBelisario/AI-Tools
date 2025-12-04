// src/components/Labels/LandmarkLabel.jsx
import { useEffect, useRef } from "react";
import L from "leaflet";
import { API_BASE } from "../../config";
import { useSchema } from "../SchemaContext.jsx";

const LandmarkLabel = ({ map }) => {
  const { schema } = useSchema();
  const layerGroupRef = useRef(L.layerGroup()); // ✅ Create immediately
  const zoomLimits = [16, 25]; // Same as parcel visibility

  // === Apply zoom-based visibility ===
  const updateVisibility = () => {
    if (!map || !layerGroupRef.current) return;
    const zoom = map.getZoom();
    const [min, max] = zoomLimits;
    const visible = zoom >= min && zoom <= max;

    if (visible) {
      if (!map.hasLayer(layerGroupRef.current)) map.addLayer(layerGroupRef.current);
    } else {
      if (map.hasLayer(layerGroupRef.current)) map.removeLayer(layerGroupRef.current);
    }
  };

  // === Fetch and render landmark labels ===
  const loadLandmarks = async () => {
    if (!map || !schema) return;

    const token =
      localStorage.getItem("access_token") ||
      localStorage.getItem("accessToken");

    try {
      const res = await fetch(`${API_BASE}/landmarks/${schema}`, {
        headers: {
          "Content-Type": "application/json",
          ...(token && { Authorization: `Bearer ${token}` }),
        },
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const features = data?.features || [];

      if (!features.length) {
        console.warn(`ℹ️ No landmarks found in schema: ${schema}`);
        return;
      }

      const group = layerGroupRef.current;
      group.clearLayers(); // ✅ Clear before adding new

      features.forEach((f) => {
        const name = f.properties?.name;
        if (!name || !f.geometry || f.geometry.type !== "Point") return;
        const [lng, lat] = f.geometry.coordinates;

        const label = L.divIcon({
          html: `<div style="
            color: white;
            font-weight: bold;
            text-shadow: 0 0 3px black, 1px 1px 2px black;
            font-size: 11px;
            white-space: nowrap;
            pointer-events: none;
          ">${name}</div>`,
          className: "",
        });

        L.marker([lat, lng], { icon: label, interactive: false }).addTo(group);
      });

      updateVisibility();
      console.log(`✅ ${features.length} landmark labels loaded for ${schema}`);
    } catch (err) {
      console.error("❌ Failed to load landmark labels:", err);
    }
  };

  // === Lifecycle ===
  useEffect(() => {
    if (!map || !schema) return;

    const group = layerGroupRef.current;
    group.addTo(map); // ✅ Ensure the group exists immediately

    loadLandmarks();
    updateVisibility();

    const handleZoom = () => updateVisibility();
    map.on("zoomend", handleZoom);

    return () => {
      console.log("🧹 Removing landmark labels");
      map.off("zoomend", handleZoom);

      // ✅ Ensure layer removal even if async fetch didn’t finish
      if (map.hasLayer(group)) map.removeLayer(group);
      group.clearLayers();
    };
  }, [map, schema]);

  return null;
};

export default LandmarkLabel;
