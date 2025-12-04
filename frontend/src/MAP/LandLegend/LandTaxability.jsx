// LandTaxability.jsx
import React, { useEffect, useState } from "react";
import L from "leaflet";
import Table_Column from "./Table_Column.jsx";
import API from "../../api.js";
import { useSchema } from "../SchemaContext";
import { useMap } from "react-leaflet";

let taxabilityLayer = null;

const LandTaxability = () => {
  const map = useMap();
  const { schema } = useSchema();
  const [column, setColumn] = useState("taxability");
  const [showSelector, setShowSelector] = useState(false);

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
    if (taxabilityLayer) {
      map.removeLayer(taxabilityLayer);
      window.removeLandInfoLegend?.("taxability");
      taxabilityLayer = null;
    }

    const colorMap = generateColorMap(features, col);

    taxabilityLayer = L.geoJSON(features, {
      style: (feature) => ({
        color: "#444",
        weight: 1,
        fillOpacity: 0.6,
        fillColor: colorMap[feature.properties[col]] || "#cccccc",
      }),
      onEachFeature: (feature, layer) => {
        layer.bindPopup(
          `<strong>${col}:</strong> ${feature.properties[col] ?? "N/A"}`
        );
      },
    }).addTo(map);

    window.addLandInfoLegend?.(
      "taxability",
      <>
        <strong>Land Taxability</strong>
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
      .catch((err) => console.error("❌ Fetch error:", err));
  };

  useEffect(() => {
    if (!map || !schema) return;

    window.toggleTaxabilityLayer = () => {
      if (taxabilityLayer) {
        map.removeLayer(taxabilityLayer);
        window.removeLandInfoLegend?.("taxability");
        taxabilityLayer = null;
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
            (k) => k.toLowerCase() === "taxability"
          );

          if (match) {
            setColumn(match);
            renderLayer(mergedFeatures, match);
            window.setActiveTool?.("taxability");
          } else {
            setShowSelector(true);
          }
        })
        .catch((err) => console.error("❌ Fetch error (toggle):", err));
    };

    return () => {
      window.toggleTaxabilityLayer = null;
      if (taxabilityLayer) {
        map.removeLayer(taxabilityLayer);
        window.removeLandInfoLegend?.("taxability");
        taxabilityLayer = null;
      }
    };
  }, [map, schema]);

  const handleColumnApply = (col) => {
    setShowSelector(false);
    setColumn(col);
    fetchAndRender(col);
    window.setActiveTool?.("taxability");
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

export default LandTaxability;
