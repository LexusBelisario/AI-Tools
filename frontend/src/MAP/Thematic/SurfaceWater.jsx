// src/components/Thematic/SurfaceWater.jsx
import React, { useEffect, useState, useRef } from "react";
import { useMap } from "react-leaflet";
import { useSchema } from "../SchemaContext";
import Table_Column from "../LandLegend/Table_Column.jsx";
import API from "../../api.js"; // ✅ CHANGED from api_service.js to api.js

const SurfaceWater = () => {
  const map = useMap();
  const { schema } = useSchema();
  const [selectedColumn, setSelectedColumn] = useState("type");
  const [showColumnPopup, setShowColumnPopup] = useState(false);
  const [geojsonData, setGeojsonData] = useState(null);
  const isBusy = useRef(false);

  // === Color generator
  const generateColor = (index, total) => {
    const hue = 60 + (index / Math.max(total - 1, 1)) * 180; // yellow to blue
    return `hsl(${hue}, 70%, 50%)`;
  };

  useEffect(() => {
    if (!schema) {
      console.log("⚠️ SurfaceWater: no schema, toggle not registered");
      return;
    }

    console.log("✅ SurfaceWater: registering toggle for schema =", schema);

    const updateButton = (active) => {
      const btn = document.getElementById("btnSurfaceWater");
      if (btn) btn.classList.toggle("active-tool", active);
    };

    window.toggleSurfaceWater = async () => {
      console.log("🖱️ toggleSurfaceWater called");

      if (!map || !schema || isBusy.current) {
        console.log("⏸️ SurfaceWater: map/schema missing or busy");
        return;
      }
      isBusy.current = true;

      try {
        // === Turn OFF
        if (window.surfaceWaterLayerGroup) {
          console.log("🔄 Removing SurfaceWater layer");
          if (map.hasLayer(window.surfaceWaterLayerGroup)) {
            map.removeLayer(window.surfaceWaterLayerGroup);
          }
          window.surfaceWaterLayerGroup = null;
          updateButton(false);
          window.removeThematicLegend?.("surfacewater");
          isBusy.current = false;
          return;
        }

        // === Turn ON
        const url = `${API}/single-table?schema=${schema}&table=SurfaceWater`;
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
        if (!props.hasOwnProperty("type")) {
          console.warn(
            "⚠️ SurfaceWater: no 'type' column, showing column picker"
          );
          setShowColumnPopup(true);
          return;
        }

        renderLayer(geojson, "type");
        updateButton(true);
      } catch (err) {
        console.error("❌ SurfaceWater fetch error:", err);
      } finally {
        isBusy.current = false;
      }
    };
  }, [map, schema]);

  // === Draw layer and legend
  const renderLayer = (geojson, column) => {
    console.log("🎨 Rendering SurfaceWater layer by column:", column);

    const values = Array.from(
      new Set(
        geojson.features.map((f) =>
          (f.properties?.[column] || "Unknown").toString().trim().toLowerCase()
        )
      )
    ).sort();

    const colorMap = {};
    values.forEach((val, i) => {
      colorMap[val] = generateColor(i, values.length);
    });

    const group = L.geoJSON(geojson, {
      style: (feature) => {
        const val = (feature.properties?.[column] || "unknown")
          .toString()
          .trim()
          .toLowerCase();
        return {
          color: colorMap[val],
          fillColor: colorMap[val],
          fillOpacity: 0.4,
          weight: 1,
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
    window.surfaceWaterLayerGroup = group;

    if (window.addThematicLegend) {
      window.addThematicLegend(
        "surfacewater",
        <>
          <strong>Surface Water Legend:</strong>
          <div className="legend-items">
            {values.map((val, idx) => (
              <div key={idx}>
                <span
                  className="legend-swatch"
                  style={{ backgroundColor: colorMap[val] }}
                ></span>
                {val.charAt(0).toUpperCase() + val.slice(1)}
              </div>
            ))}
          </div>
        </>
      );
    }
  };

  // === Handle column picker
  const handleApplyColumn = (col) => {
    console.log("📊 User applied column:", col);
    setSelectedColumn(col);
    setShowColumnPopup(false);
    if (geojsonData) {
      renderLayer(geojsonData, col);
      document.getElementById("btnSurfaceWater")?.classList.add("active-tool");
    }
  };

  return (
    <>
      {showColumnPopup && (
        <Table_Column
          schema={schema}
          table="SurfaceWater"
          onApply={handleApplyColumn}
          onClose={() => setShowColumnPopup(false)}
        />
      )}
    </>
  );
};

export default SurfaceWater;
