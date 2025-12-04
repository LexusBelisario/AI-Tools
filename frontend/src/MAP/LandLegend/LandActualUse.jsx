// LandActualUse.jsx
import React, { useEffect, useState } from "react";
import Table_Column from "./Table_Column.jsx";
import L from "leaflet";
import API from "../../api.js";
import { useSchema } from "../SchemaContext";
import { useMap } from "react-leaflet";

let actualUseLayer = null;

const LandActualUse = () => {
  const map = useMap();
  const { schema } = useSchema();
  const [column, setColumn] = useState("land_ause");
  const [showSelector, setShowSelector] = useState(false);

  const generateColorMap = (features, key) => {
    const categories = [...new Set(features.map((f) => f.properties[key]))];
    const total = categories.length || 1;
    const colorMap = {};
    categories.forEach((cat, i) => {
      const hue = Math.floor((360 / total) * i);
      colorMap[cat] = `hsl(${hue}, 65%, 55%)`;
    });
    return colorMap;
  };

  const renderLayer = (features, col) => {
    if (actualUseLayer) {
      map.removeLayer(actualUseLayer);
      window.removeLandInfoLegend?.("actualuse");
      actualUseLayer = null;
    }

    const colorMap = generateColorMap(features, col);

    actualUseLayer = L.geoJSON(features, {
      style: (feature) => ({
        color: "#444",
        weight: 1,
        fillOpacity: 0.6,
        fillColor: colorMap[feature.properties[col]] || "#cccccc",
      }),
      onEachFeature: (feature, layer) => {
        layer.bindPopup(`<strong>${col}:</strong> ${feature.properties[col]}`);
      },
    }).addTo(map);

    window.addLandInfoLegend?.(
      "actualuse",
      <>
        <strong>Land Actual Use</strong>
        <div className="legend-items">
          {Object.entries(colorMap).map(([value, color]) => (
            <div key={value}>
              <span
                className="legend-swatch"
                style={{ backgroundColor: color }}
              ></span>
              {value}
            </div>
          ))}
        </div>
      </>
    );
  };

  const fetchAndRender = (col) => {
    if (!schema) {
      console.warn("⛔ fetchAndRender: No schema selected");
      return;
    }

    const geoUrl = `${API}/all-barangays?schemas=${schema}`;
    const attrUrl = `${API}/search/attribute-table?schema=${schema}`; // ✅ updated path

    Promise.all([fetch(geoUrl), fetch(attrUrl)])
      .then(([geoRes, attrRes]) => Promise.all([geoRes.json(), attrRes.json()]))
      .then(([geoData, attrData]) => {
        if (!geoData.features?.length) {
          console.warn("⚠️ No geometry features returned");
          return;
        }
        if (attrData.status !== "success" || !attrData.data) {
          console.warn("⚠️ No attribute data returned");
          return;
        }

        const attrMap = {};
        attrData.data.forEach((attr) => {
          attrMap[attr.pin] = attr;
        });

        const mergedFeatures = geoData.features.map((f) => ({
          ...f,
          properties: { ...f.properties, ...(attrMap[f.properties.pin] || {}) },
        }));

        renderLayer(mergedFeatures, col);
      })
      .catch((err) => console.error("❌ Fetch error:", err));
  };

  useEffect(() => {
    if (!map || !schema) {
      console.warn(
        "⛔ LandActualUse not ready. map:",
        !!map,
        "schema:",
        schema
      );
      return;
    }

    window.toggleActualUseLayer = () => {
      if (actualUseLayer) {
        map.removeLayer(actualUseLayer);
        window.removeLandInfoLegend?.("actualuse");
        actualUseLayer = null;
        window.setActiveTool?.(null);
        return;
      }

      fetchAndRender(column);
    };

    return () => {
      window.toggleActualUseLayer = null;
      if (actualUseLayer) {
        map.removeLayer(actualUseLayer);
        window.removeLandInfoLegend?.("actualuse");
        actualUseLayer = null;
      }
    };
  }, [map, schema]);

  const handleColumnApply = (col) => {
    setShowSelector(false);
    setColumn(col);
    fetchAndRender(col);
    window.setActiveTool?.("actualuse");
  };

  return (
    <>
      {showSelector && (
        <Table_Column
          schema={schema}
          table="JoinedTable"
          onApply={handleColumnApply}
          onClose={() => setShowSelector(false)}
        />
      )}
    </>
  );
};

export default LandActualUse;
