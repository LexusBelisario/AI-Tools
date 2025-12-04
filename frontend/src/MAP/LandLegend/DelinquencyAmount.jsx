// DelinquencyAmount.jsx
import React, { useEffect, useState } from "react";
import L from "leaflet";
import Table_Column from "./Table_Column.jsx";
import API from "../../api.js";
import { useSchema } from "../SchemaContext";
import { useMap } from "react-leaflet";

let delinqAmountLayer = null;

const DelinquencyAmount = () => {
  const map = useMap();
  const { schema } = useSchema();
  const [showColumnSelector, setShowColumnSelector] = useState(false);
  const [selectedColumn, setSelectedColumn] = useState("del_amount");

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
    if (delinqAmountLayer) {
      map.removeLayer(delinqAmountLayer);
      window.removeLandInfoLegend?.("delinqamount");
      delinqAmountLayer = null;
    }

    const colorMap = generateColorMap(features, col);

    delinqAmountLayer = L.geoJSON(features, {
      style: (feature) => {
        const val = feature.properties[col];
        return {
          color: "#444",
          weight: 1,
          fillOpacity: 0.6,
          fillColor:
            val !== null && val !== undefined && val !== ""
              ? colorMap[val]
              : "#cccccc",
        };
      },
      onEachFeature: (feature, layer) => {
        const val = feature.properties[col];
        layer.bindPopup(`<strong>${col}:</strong> ${val ?? "N/A"}`);
      },
    }).addTo(map);

    // Create legend using CSS custom properties for reliable color display
    const legendContent = (
      <>
        <strong>Delinquency Amount</strong>
        <div className="legend-items">
          {Object.entries(colorMap).map(([value, color]) => (
            <div key={value}>
              <span
                className="legend-swatch"
                style={{
                  "--swatch-color": color,
                  backgroundColor: color, // fallback
                }}
              ></span>
              <span>{value}</span>
            </div>
          ))}
        </div>
      </>
    );

    window.addLandInfoLegend?.("delinqamount", legendContent);
  };

  const fetchAndRender = (col) => {
    if (!schema) return;

    const geoUrl = `${API}/all-barangays?schemas=${schema}`;
    const attrUrl = `${API}/attribute-table?schema=${schema}`;

    Promise.all([fetch(geoUrl), fetch(attrUrl)])
      .then(([geoRes, attrRes]) => Promise.all([geoRes.json(), attrRes.json()]))
      .then(([geoData, attrData]) => {
        if (!geoData.features || geoData.features.length === 0) return;
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
      .catch((err) => console.error("⚠ Failed to reload:", err));
  };

  useEffect(() => {
    if (!map || !schema) return;

    window.toggleDelinqAmountLayer = () => {
      if (delinqAmountLayer) {
        map.removeLayer(delinqAmountLayer);
        window.removeLandInfoLegend?.("delinqamount");
        delinqAmountLayer = null;
        window.setActiveTool?.(null);
        return;
      }

      const geoUrl = `${API}/all-barangays?schemas=${schema}`;
      const attrUrl = `${API}/attribute-table?schema=${schema}`;

      Promise.all([fetch(geoUrl), fetch(attrUrl)])
        .then(([geoRes, attrRes]) =>
          Promise.all([geoRes.json(), attrRes.json()])
        )
        .then(([geoData, attrData]) => {
          if (!geoData.features || geoData.features.length === 0) return;
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
            (k) => k.toLowerCase() === "del_amount"
          );

          if (match) {
            setSelectedColumn(match);
            renderLayer(mergedFeatures, match);
            window.setActiveTool?.("delinqamount");
          } else {
            setShowColumnSelector(true);
          }
        })
        .catch((err) => console.error("⚠ Fetch error:", err));
    };

    return () => {
      window.toggleDelinqAmountLayer = null;
      if (delinqAmountLayer) {
        map.removeLayer(delinqAmountLayer);
        window.removeLandInfoLegend?.("delinqamount");
        delinqAmountLayer = null;
      }
    };
  }, [map, schema]);

  const handleColumnApply = (col) => {
    setShowColumnSelector(false);
    setSelectedColumn(col);
    fetchAndRender(col);
    window.setActiveTool?.("delinqamount");
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

export default DelinquencyAmount;
