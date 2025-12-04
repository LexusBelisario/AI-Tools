import { useEffect, useState, useRef } from "react";
import L from "leaflet";
import "leaflet-textpath";
import * as turf from "@turf/turf";


const LABEL_FIELDS = [
  { key: "barangay", label: "Barangay", color: "#1e90ff" },
  { key: "section", label: "Section", color: "#28a745" },
  { key: "pin", label: "PIN", color: "#ff5733" },
];

const Multilabel = ({ map, activeSchemas }) => {
  const [features, setFeatures] = useState([]);
  const [visibleFields, setVisibleFields] = useState([]);
  const labelGroupRef = useRef(L.layerGroup([], { pane: "markerPane" }));
  const MAX_LABELS = 200;

  useEffect(() => {
    if (!map || !activeSchemas.length) return;

    const fetchData = async () => {
      const query = activeSchemas.map(s => `schemas=${encodeURIComponent(s)}`).join("&");
      const url = `${API}/all-barangays?${query}`;
      try {
        const res = await fetch(url);
        const geojson = await res.json();
        setFeatures(geojson.features || []);
      } catch (err) {
        console.error("❌ Multilabel fetch error:", err);
      }
    };

    fetchData();
  }, [map, activeSchemas]);

  useEffect(() => {
    if (!map || !features.length) return;

    const labelGroup = labelGroupRef.current;

    const getPolygonCenter = (geometry) => {
      try {
        const centroid = turf.centroid({ type: "Feature", geometry });
        const [lng, lat] = centroid.geometry.coordinates;
        return L.latLng(lat, lng);
      } catch {
        return null;
      }
    };

    const renderLabels = () => {
      labelGroup.clearLayers();
      if (!visibleFields.length) return;

      const bounds = map.getBounds();
      const visibleFeatures = [];

      features.forEach((feature) => {
        const { barangay } = feature.properties || {};
        const geom = feature.geometry;
        const center = getPolygonCenter(geom);
        if (!geom || !barangay || !center) return;
        if (!bounds.contains(center)) return;
        visibleFeatures.push({ feature, center });
      });

      if (visibleFeatures.length > MAX_LABELS) {
        console.warn(`⚠️ Skipping label render — too many features in view (${visibleFeatures.length})`);
        return;
      }

      visibleFeatures.forEach(({ feature, center }) => {
        let htmlLines = [];

        visibleFields.forEach((fieldKey) => {
          const fieldConfig = LABEL_FIELDS.find(f => f.key === fieldKey);
          if (!fieldConfig) return;

          const value = feature.properties?.[fieldKey];
          if (value) {
            const labelText = fieldConfig.key === "pin" ? `PIN: ${value}` : value;
            htmlLines.push(`<div style="color:${fieldConfig.color};">${labelText}</div>`);
          }
        });

        if (!htmlLines.length) return;

        const html = `
          <div style="text-align:center; font-weight:bold; font-size:12px; line-height:1.2;">
            ${htmlLines.join("")}
          </div>
        `;

        const marker = L.marker(center, {
          icon: L.divIcon({
            className: "multi-label",
            html,
            iconSize: null
          }),
          interactive: false,
          zIndexOffset: 1000
        });

        labelGroup.addLayer(marker);
      });

      map.addLayer(labelGroup);
    };

    let timeout;
    const handleMoveEnd = () => {
      clearTimeout(timeout);
      timeout = setTimeout(renderLabels, 150);
    };

    renderLabels();
    map.on("moveend", handleMoveEnd);
    map.on("zoomend", handleMoveEnd);

    return () => {
      map.off("moveend", handleMoveEnd);
      map.off("zoomend", handleMoveEnd);
      map.removeLayer(labelGroup);
      labelGroup.clearLayers();
    };
  }, [map, features, visibleFields]);

  const toggleField = (fieldKey) => {
    setVisibleFields((prev) => {
      const isActive = prev.includes(fieldKey);
      return isActive ? prev.filter((k) => k !== fieldKey) : [...prev.filter(k => k !== fieldKey), fieldKey];
    });
  };

  return (
    <div className="label-subfilters" style={{
      position: "absolute",
      bottom: 180,
      left: 0,
      zIndex: 9999,
      background: "#fff",
      padding: "10px 14px",
      borderRadius: "6px",
      boxShadow: "0 0 6px rgba(0,0,0,0.2)",
      fontSize: "14px",
      maxHeight: "50vh",
      overflowY: "auto"
    }}>
      {LABEL_FIELDS.map(field => (
        <div key={field.key}>
          <label style={{ color: field.color, fontWeight: "bold" }}>
            <input
              type="checkbox"
              checked={visibleFields.includes(field.key)}
              onChange={() => toggleField(field.key)}
            /> {field.label}
          </label>
        </div>
      ))}
    </div>
  );
};

export default Multilabel;
