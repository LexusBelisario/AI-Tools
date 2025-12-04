// src/components/Labels/StreetLabel.jsx
import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet-textpath";
import { API_BASE } from "../../config";
import { useSchema } from "../SchemaContext.jsx";

const StreetLabel = ({ map }) => {
  const { schema } = useSchema();
  const layerGroupRef = useRef(null);
  const zoomLimits = [16, 25]; // same as parcel visibility

  // === Visibility handler ===
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

  useEffect(() => {
    if (!map || !schema) return;

    const group = L.layerGroup().addTo(map);
    layerGroupRef.current = group;
    console.log(`📍 Loading street labels for schema=${schema}`);

    const fetchStreets = async () => {
      try {
        const token =
          localStorage.getItem("access_token") ||
          localStorage.getItem("accessToken");

        const url = `${API_BASE}/single-table?schema=${encodeURIComponent(schema)}&table=RoadNetwork`;
        const res = await fetch(url, {
          headers: {
            "Content-Type": "application/json",
            ...(token && { Authorization: `Bearer ${token}` }),
          },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const features = data?.features || [];

        if (!features.length) {
          console.warn(`ℹ️ No street data found for ${schema}`);
          return;
        }

        features.forEach((f) => {
          const name =
            f.properties?.road_name ||
            f.properties?.name ||
            f.properties?.label ||
            f.properties?.roadlabel;
          if (!name || !f.geometry) return;

          let coords = f.geometry.coordinates;
          if (f.geometry.type === "MultiLineString") coords = coords[0];
          if (!coords?.length) return;

          // ensure text direction is readable
          const dx = coords[coords.length - 1][0] - coords[0][0];
          const dy = coords[coords.length - 1][1] - coords[0][1];
          const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
          if (angle > 90 || angle < -90) coords.reverse();

          const latlngs = coords.map(([lng, lat]) => [lat, lng]);

          // invisible polyline for text path
          const line = L.polyline(latlngs, { opacity: 0 }).addTo(group);

          // add road name text with strong black outline
          line.setText(name, {
            repeat: false,
            center: true,
            offset: 0,
            attributes: {
              fill: "white",
              "font-size": "12px",
              "font-weight": "bold",
              stroke: "black",            // black outline
              "stroke-width": "3px",      // outline thickness
              "paint-order": "stroke",    // ensures stroke is rendered behind text fill
              "stroke-linejoin": "round", // smoother joins for curved roads
              "text-shadow": "none",      // remove blur shadows for crisp edges
            },
          });
        });

        updateVisibility();
        map.on("zoomend", updateVisibility);
        console.log(`✅ ${features.length} street labels rendered for ${schema}`);
      } catch (err) {
        console.error("❌ Failed to load street labels:", err);
      }
    };

    fetchStreets();

    return () => {
      map.off("zoomend", updateVisibility);
      if (layerGroupRef.current && map.hasLayer(layerGroupRef.current)) {
        map.removeLayer(layerGroupRef.current);
      }
      layerGroupRef.current = null;
      console.log("🧹 Street labels removed");
    };
  }, [map, schema]);

  return null;
};

export default StreetLabel;
