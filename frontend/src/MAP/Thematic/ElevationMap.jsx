// src/components/Thematic/Elevation.jsx
import React, { useEffect, useState, useRef } from "react";
import { useMap } from "react-leaflet";
import { useSchema } from "../SchemaContext.jsx";
import Table_Column from "../LandLegend/Table_Column.jsx";
import API from "../../api.js";

const ElevationMap = ({ setRange }) => {
  const map = useMap();
  const { schema } = useSchema();
  const [selectedColumn, setSelectedColumn] = useState("elevation");
  const [showColumnPopup, setShowColumnPopup] = useState(false);
  const [geojsonData, setGeojsonData] = useState(null);
  const isBusy = useRef(false);

  useEffect(() => {
    if (!schema) {
      console.log("⚠️ Elevation: no schema, toggle not registered");
      return;
    }

    console.log("✅ Elevation: registering toggle for schema =", schema);

    const updateButton = (active) => {
      const btn = document.getElementById("btnElevationMap");
      if (btn) btn.classList.toggle("active-tool", active);
    };

    window.toggleElevationLayer = async () => {
      console.log("🖱️ toggleElevationLayer called");

      if (!map || !schema || isBusy.current) {
        console.log("⏸️ Elevation: map/schema missing or busy");
        return;
      }
      isBusy.current = true;

      try {
        // === Turn OFF
        if (window.elevationLayerGroup) {
          console.log("🔄 Removing Elevation layer");
          if (map.hasLayer(window.elevationLayerGroup)) {
            map.removeLayer(window.elevationLayerGroup);
          }
          window.elevationLayerGroup = null;
          updateButton(false);
          window.removeThematicLegend?.("elevation");
          isBusy.current = false;
          return;
        }

        // === Turn ON
        const url = `${API}/single-table?schema=${schema}&table=ElevationMap`;
        console.log("🌐 Fetching:", url);

        // ✅ ADD AUTH HEADERS HERE!
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
        if (!props.hasOwnProperty("elevation")) {
          console.warn("⚠️ Elevation: no 'elevation' column, showing picker");
          setShowColumnPopup(true);
          return;
        }

        renderLayer(geojson, "elevation");
        updateButton(true);
      } catch (err) {
        console.error("❌ Elevation fetch error:", err);
      } finally {
        isBusy.current = false;
      }
    };
  }, [map, schema]);

  const renderLayer = (geojson, column) => {
    console.log("🎨 Rendering Elevation by column:", column);

    const elevations = geojson.features
      .map((f) => Number((f.properties?.[column] || "").toString().trim()))
      .filter((v) => !isNaN(v));

    if (elevations.length === 0) {
      console.warn("⚠️ No numeric elevation values found");
      return;
    }

    const min = Math.min(...elevations);
    const max = Math.max(...elevations);
    const step = (max - min) / 7;
    if (setRange) setRange({ min, max });

    const getColor = (value) => {
      if (value <= min + step) return "#8b4513";
      if (value <= min + 2 * step) return "#3cb44b";
      if (value <= min + 3 * step) return "#ffff00";
      if (value <= min + 4 * step) return "#ffa500";
      if (value <= min + 5 * step) return "#ff0000";
      if (value <= min + 6 * step) return "#800080";
      return "#ff69b4";
    };

    const breaks = [
      { color: "#8b4513", label: `≤ ${(min + step).toFixed(1)} m` },
      {
        color: "#3cb44b",
        label: `${(min + step).toFixed(1)}–${(min + 2 * step).toFixed(1)} m`,
      },
      {
        color: "#ffff00",
        label: `${(min + 2 * step).toFixed(1)}–${(min + 3 * step).toFixed(
          1
        )} m`,
      },
      {
        color: "#ffa500",
        label: `${(min + 3 * step).toFixed(1)}–${(min + 4 * step).toFixed(
          1
        )} m`,
      },
      {
        color: "#ff0000",
        label: `${(min + 4 * step).toFixed(1)}–${(min + 5 * step).toFixed(
          1
        )} m`,
      },
      {
        color: "#800080",
        label: `${(min + 5 * step).toFixed(1)}–${(min + 6 * step).toFixed(
          1
        )} m`,
      },
      { color: "#ff69b4", label: `≥ ${(min + 6 * step).toFixed(1)} m` },
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
    window.elevationLayerGroup = group;

    if (window.addThematicLegend) {
      window.addThematicLegend(
        "elevation",
        <>
          <strong>Elevation Legend (m):</strong>
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
    console.log("📊 User applied Elevation column:", col);
    setSelectedColumn(col);
    setShowColumnPopup(false);
    if (geojsonData) {
      renderLayer(geojsonData, col);
      document.getElementById("btnElevationMap")?.classList.add("active-tool");
    }
  };

  return (
    <>
      {showColumnPopup && (
        <Table_Column
          schema={schema}
          table="ElevationMap"
          onApply={handleApplyColumn}
          onClose={() => setShowColumnPopup(false)}
        />
      )}
    </>
  );
};

export default ElevationMap;
