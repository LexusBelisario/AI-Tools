// src/components/Thematic/CLUP.jsx
import React, { useEffect, useState, useRef } from "react";
import { useMap } from "react-leaflet";
import { useSchema } from "../SchemaContext.jsx";
import Table_Column from "../LandLegend/Table_Column.jsx";
import API from "../../api.js";

const CLUP = () => {
  const map = useMap();
  const { schema } = useSchema();
  const [showColumnSelector, setShowColumnSelector] = useState(false);
  const [selectedColumn, setSelectedColumn] = useState(null);
  const [geojsonData, setGeojsonData] = useState(null);
  const isBusy = useRef(false);

  const generateColor = () => {
    const r = Math.floor(100 + Math.random() * 155);
    const g = Math.floor(100 + Math.random() * 155);
    const b = Math.floor(100 + Math.random() * 155);
    return `rgb(${r},${g},${b})`;
  };

  const hasZoneColumn = (features) => {
    if (!features || features.length === 0) return false;
    const props = features[0].properties || {};
    return Object.keys(props).some((k) => k.toLowerCase() === "zone");
  };

  useEffect(() => {
    if (!schema) {
      console.log("⚠️ LandUse: no schema, toggle not registered");
      return;
    }

    console.log("✅ LandUse: registering toggle for schema =", schema);

    const updateButton = (active) => {
      const btn = document.getElementById("btnLandUse");
      if (btn) btn.classList.toggle("active-tool", active);
    };

    window.toggleLandUseLayer = async () => {
      console.log("🖱️ toggleLandUseLayer called");

      if (!map || !schema || isBusy.current) {
        console.log("⏸️ LandUse: map/schema missing or busy");
        return;
      }
      isBusy.current = true;

      try {
        // === Turn OFF
        if (window.landUseLayerGroup) {
          console.log("🔄 Removing LandUse layer");
          if (map.hasLayer(window.landUseLayerGroup)) {
            map.removeLayer(window.landUseLayerGroup);
          }
          window.landUseLayerGroup = null;
          updateButton(false);
          window.removeThematicLegend?.("landuse");
          isBusy.current = false;
          return;
        }

        // === Turn ON
        const url = `${API}/single-table?schema=${schema}&table=LandUse`;
        console.log("🌐 Fetching:", url);

        // ✅ ADD AUTHENTICATION HEADERS
        const token =
          localStorage.getItem("access_token") ||
          localStorage.getItem("accessToken");
        const res = await fetch(url, {
          headers: {
            "Content-Type": "application/json",
            ...(token && { Authorization: `Bearer ${token}` }),
          },
        });

        // ✅ CHECK RESPONSE STATUS
        if (!res.ok) {
          if (res.status === 401 || res.status === 403) {
            console.error("❌ Authentication error");
            localStorage.removeItem("access_token");
            localStorage.removeItem("accessToken");
            window.location.href = "/login";
            return;
          }
          throw new Error(`HTTP error! status: ${res.status}`);
        }

        const geojson = await res.json();
        setGeojsonData(geojson);

        let columnToStyle = selectedColumn;
        if (!columnToStyle) {
          if (hasZoneColumn(geojson.features)) {
            const props = geojson.features[0].properties;
            columnToStyle = Object.keys(props).find(
              (k) => k.toLowerCase() === "zone"
            );
            console.log(`✅ Using default Zone column: ${columnToStyle}`);
          } else {
            console.log("⚠️ No 'Zone' column found, showing selector");
            setShowColumnSelector(true);
            isBusy.current = false;
            return;
          }
        }

        renderLayer(geojson, columnToStyle);
        updateButton(true);
      } catch (err) {
        console.error("❌ LandUse fetch error:", err);
      } finally {
        isBusy.current = false;
      }
    };
  }, [map, schema, selectedColumn]);

  const renderLayer = (geojson, columnName) => {
    console.log("🎨 Rendering LandUse by column:", columnName);

    const values = geojson.features.map((f) => f.properties?.[columnName]);
    const unique = Array.from(
      new Set(values.map((v) => v?.toString().toUpperCase()).filter(Boolean))
    );

    const colorMap = {};
    unique.forEach((v) => {
      colorMap[v] = generateColor();
    });

    const group = L.geoJSON(geojson, {
      style: (feature) => {
        const val = feature.properties?.[columnName]?.toString().toUpperCase();
        return {
          color: "#444",
          weight: 1,
          fillColor: colorMap[val] || "#ccc",
          fillOpacity: 0.6,
        };
      },
      onEachFeature: (feature, layer) => {
        const val = feature.properties?.[columnName];
        layer.bindPopup(
          `<strong>Land Use ${columnName}:</strong> ${val || "N/A"}`
        );
      },
    });

    group.addTo(map);
    window.landUseLayerGroup = group;

    if (window.addThematicLegend) {
      window.addThematicLegend(
        "landuse",
        <>
          <strong>Land Use ({columnName}):</strong>
          <div className="legend-items">
            {unique.map((val, i) => (
              <div key={i}>
                <span
                  className="legend-swatch"
                  style={{ backgroundColor: colorMap[val] }}
                ></span>
                {val}
              </div>
            ))}
          </div>
        </>
      );
    }
  };

  const handleColumnSelection = (columnName) => {
    console.log("📊 User selected LandUse column:", columnName);
    setShowColumnSelector(false);
    setSelectedColumn(columnName);
    if (geojsonData) {
      renderLayer(geojsonData, columnName);
      document.getElementById("btnLandUse")?.classList.add("active-tool");
    }
  };

  return (
    <>
      {showColumnSelector && (
        <Table_Column
          schema={schema}
          table="LandUse"
          onApply={handleColumnSelection}
          onClose={() => setShowColumnSelector(false)}
        />
      )}
    </>
  );
};

export default CLUP;
