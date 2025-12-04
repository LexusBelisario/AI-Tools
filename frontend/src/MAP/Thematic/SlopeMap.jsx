// src/components/Thematic/SlopeMap.jsx
import React, { useEffect, useState, useRef } from "react";
import { useMap } from "react-leaflet";
import { useSchema } from "../SchemaContext";
import Table_Column from "../LandLegend/Table_Column.jsx";
import API from "../../api.js"; // ✅ CHANGED from api_service.js to api.js

const SlopeMap = () => {
  const map = useMap();
  const { schema } = useSchema();
  const [selectedColumn, setSelectedColumn] = useState("slope");
  const [showColumnPopup, setShowColumnPopup] = useState(false);
  const [geojsonData, setGeojsonData] = useState(null);
  const isBusy = useRef(false);

  useEffect(() => {
    if (!schema) {
      console.log("⚠️ SlopeMap: no schema, toggle not registered");
      return;
    }

    console.log("✅ SlopeMap: registering toggle for schema =", schema);

    const updateButton = (active) => {
      const btn = document.getElementById("btnSlopeMap");
      if (btn) btn.classList.toggle("active-tool", active);
    };

    window.toggleSlopeLayer = async () => {
      console.log("🖱️ toggleSlopeLayer called");

      if (!map || !schema || isBusy.current) {
        console.log("⏸️ SlopeMap: map/schema missing or busy");
        return;
      }
      isBusy.current = true;

      try {
        // === Turn OFF
        if (window.slopeLayerGroup) {
          console.log("🔄 Removing Slope layer");
          if (map.hasLayer(window.slopeLayerGroup)) {
            map.removeLayer(window.slopeLayerGroup);
          }
          window.slopeLayerGroup = null;
          updateButton(false);
          window.removeThematicLegend?.("slope");
          isBusy.current = false;
          return;
        }

        // === Turn ON
        const url = `${API}/single-table?schema=${schema}&table=SlopeMap`;
        console.log("🌐 Fetching:", url);

        // ✅ ADD AUTH HEADERS
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

        const props = geojson.features?.[0]?.properties || {};
        if (!props.hasOwnProperty("slope")) {
          console.warn("⚠️ SlopeMap: no 'slope' column, showing picker");
          setShowColumnPopup(true);
          return;
        }

        renderLayer(geojson, "slope");
        updateButton(true);
      } catch (err) {
        console.error("❌ SlopeMap fetch error:", err);
      } finally {
        isBusy.current = false;
      }
    };
  }, [map, schema]);

  const renderLayer = (geojson, column) => {
    console.log("🎨 Rendering SlopeMap by column:", column);

    const slopes = geojson.features
      .map((f) => Number((f.properties?.[column] || "").toString().trim()))
      .filter((v) => !isNaN(v));

    if (slopes.length === 0) {
      console.warn("⚠️ No numeric slope values found");
      return;
    }

    const min = Math.min(...slopes);
    const max = Math.max(...slopes);
    const step = (max - min) / 7;

    const getColor = (value) => {
      const r = Math.max(0, Math.min(1, (value - min) / (max - min)));
      if (r >= 0.833) return "#ff69b4"; // Very steep
      if (r >= 0.666) return "#800080"; // Extremely steep
      if (r >= 0.5) return "#ff0000"; // Steep
      if (r >= 0.333) return "#ffa500"; // Moderately steep
      if (r >= 0.166) return "#ffff00"; // Gentle
      if (r >= 0.083) return "#3cb44b"; // Nearly level
      return "#8b4513"; // Flat
    };

    const breaks = [
      { color: "#8b4513", label: `≤ ${(min + step).toFixed(1)}°` },
      {
        color: "#3cb44b",
        label: `${(min + step).toFixed(1)}–${(min + 2 * step).toFixed(1)}°`,
      },
      {
        color: "#ffff00",
        label: `${(min + 2 * step).toFixed(1)}–${(min + 3 * step).toFixed(1)}°`,
      },
      {
        color: "#ffa500",
        label: `${(min + 3 * step).toFixed(1)}–${(min + 4 * step).toFixed(1)}°`,
      },
      {
        color: "#ff0000",
        label: `${(min + 4 * step).toFixed(1)}–${(min + 5 * step).toFixed(1)}°`,
      },
      {
        color: "#800080",
        label: `${(min + 5 * step).toFixed(1)}–${(min + 6 * step).toFixed(1)}°`,
      },
      { color: "#ff69b4", label: `≥ ${(min + 6 * step).toFixed(1)}°` },
    ];

    const group = L.geoJSON(geojson, {
      style: (feature) => {
        const val = Number(
          (feature.properties?.[column] || "").toString().trim()
        );
        const color = !isNaN(val) ? getColor(val) : "#ccc";
        return {
          color,
          fillColor: color,
          fillOpacity: 0.6,
          weight: 0.4,
        };
      },
      onEachFeature: (feature, layer) => {
        layer.on("click", () => {
          if (window.openRoadInfoOnly) {
            window.openRoadInfoOnly(feature.properties);
          }
        });
      },
    });

    group.addTo(map);
    window.slopeLayerGroup = group;

    if (window.addThematicLegend) {
      window.addThematicLegend(
        "slope",
        <>
          <strong>Slope Legend (°):</strong>
          <div className="legend-items">
            {breaks.map((b, i) => (
              <div key={i}>
                <span
                  className="legend-swatch"
                  style={{ backgroundColor: b.color }}
                ></span>
                {b.label}
              </div>
            ))}
          </div>
        </>
      );
    }
  };

  const handleApplyColumn = (col) => {
    console.log("📊 User applied SlopeMap column:", col);
    setSelectedColumn(col);
    setShowColumnPopup(false);
    if (geojsonData) {
      renderLayer(geojsonData, col);
      document.getElementById("btnSlopeMap")?.classList.add("active-tool");
    }
  };

  return (
    <>
      {showColumnPopup && (
        <Table_Column
          schema={schema}
          table="SlopeMap"
          onApply={handleApplyColumn}
          onClose={() => setShowColumnPopup(false)}
        />
      )}
    </>
  );
};

export default SlopeMap;
