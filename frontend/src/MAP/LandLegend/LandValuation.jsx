// LandValuation.jsx
import React, { useEffect, useState } from "react";
import L from "leaflet";
import Table_Column from "./Table_Column";
import API from "../../api.js";
import { useSchema } from "../SchemaContext";
import { useMap } from "react-leaflet";

let landValuationLayer = null;

const LandValuation = () => {
  const map = useMap();
  const { schema } = useSchema();
  const [showColumnSelector, setShowColumnSelector] = useState(false);
  const [selectedColumn, setSelectedColumn] = useState("value");

  const generateColorMap = (features, key) => {
    const categories = [
      ...new Set(
        features
          .map((f) => f.properties[key])
          .filter((v) => v !== null && v !== undefined && v !== "")
      ),
    ];
    const total = categories.length || 1;
    const colorMap = {};
    categories.forEach((cat, i) => {
      const hue = Math.floor((360 / total) * i);
      colorMap[cat] = `hsl(${hue}, 65%, 55%)`;
    });
    return colorMap;
  };

  const renderLayer = (features, col) => {
    if (landValuationLayer) {
      map.removeLayer(landValuationLayer);
      window.removeLandInfoLegend?.("landvaluation");
      landValuationLayer = null;
    }

    const colorMap = generateColorMap(features, col);

    landValuationLayer = L.geoJSON(features, {
      style: (feature) => {
        const val = feature.properties[col];
        return {
          color: "#444",
          weight: 1,
          fillOpacity: 0.6,
          fillColor: val ? colorMap[val] : "#cccccc",
        };
      },
      onEachFeature: (feature, layer) => {
        const val = feature.properties[col];
        layer.bindPopup(`<strong>${col}:</strong> ${val ?? "N/A"}`);
      },
    }).addTo(map);

    window.addLandInfoLegend?.(
      "landvaluation",
      <>
        <strong>Land Valuation ({col})</strong>
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
    if (!schema) return;

    const geoUrl = `${API}/all-barangays?schemas=${schema}`;
    const attrUrl = `${API}/search/attribute-table?schema=${schema}`; // ✅ updated path

    Promise.all([fetch(geoUrl), fetch(attrUrl)])
      .then(([geoRes, attrRes]) => Promise.all([geoRes.json(), attrRes.json()]))
      .then(([geoData, attrData]) => {
        if (!geoData.features?.length) return;
        if (attrData.status !== "success" || !attrData.data) return;

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
      .catch((err) => console.error("❌ Failed to reload:", err));
  };

  useEffect(() => {
    if (!map || !schema) return;

    window.toggleLandValuationLayer = () => {
      if (landValuationLayer) {
        map.removeLayer(landValuationLayer);
        window.removeLandInfoLegend?.("landvaluation");
        landValuationLayer = null;
        window.setActiveTool?.(null);
        return;
      }

      const geoUrl = `${API}/all-barangays?schemas=${schema}`;
      const attrUrl = `${API}/search/attribute-table?schema=${schema}`; // ✅ updated path

      Promise.all([fetch(geoUrl), fetch(attrUrl)])
        .then(([geoRes, attrRes]) =>
          Promise.all([geoRes.json(), attrRes.json()])
        )
        .then(([geoData, attrData]) => {
          if (!geoData.features?.length) return;
          if (attrData.status !== "success" || !attrData.data) return;

          const attrMap = {};
          attrData.data.forEach((attr) => {
            attrMap[attr.pin] = attr;
          });

          const mergedFeatures = geoData.features.map((f) => ({
            ...f,
            properties: {
              ...f.properties,
              ...(attrMap[f.properties.pin] || {}),
            },
          }));

          const props = mergedFeatures[0].properties;
          const match = Object.keys(props).find(
            (k) => k.toLowerCase() === "value"
          );

          if (match) {
            setSelectedColumn(match);
            renderLayer(mergedFeatures, match);
            window.setActiveTool?.("landvaluation");
          } else {
            setShowColumnSelector(true);
          }
        })
        .catch((err) => console.error("❌ Fetch error:", err));
    };

    return () => {
      window.toggleLandValuationLayer = null;
      if (landValuationLayer) {
        map.removeLayer(landValuationLayer);
        window.removeLandInfoLegend?.("landvaluation");
        landValuationLayer = null;
      }
    };
  }, [map, schema]);

  const handleColumnApply = (col) => {
    setShowColumnSelector(false);
    setSelectedColumn(col);
    fetchAndRender(col);
    window.setActiveTool?.("landvaluation");
  };

  return (
    <>
      {showColumnSelector && (
        <Table_Column
          schema={schema}
          table="JoinedTable"
          onApply={handleColumnApply}
          onClose={() => setShowColumnSelector(false)}
        />
      )}
    </>
  );
};

export default LandValuation;
