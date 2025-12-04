// src/components/Thematic/SoilType.jsx
import React, { useEffect, useState, useRef } from "react";
import { useMap } from "react-leaflet";
import Table_Column from "../LandLegend/Table_Column.jsx";
import API from "../../api.js"; // ✅ CHANGED from api_service.js to api.js
import { useSchema } from "../SchemaContext";

const SoilType = () => {
  const map = useMap();
  const { schema } = useSchema();
  const [selectedColumn, setSelectedColumn] = useState("type");
  const [showColumnPopup, setShowColumnPopup] = useState(false);
  const [geojsonData, setGeojsonData] = useState(null);
  const isBusy = useRef(false);

  const predefinedColors = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
  ];
  const generateColor = (index) =>
    predefinedColors[index % predefinedColors.length];

  useEffect(() => {
    if (!schema) {
      console.log("⚠️ SoilType: no schema, toggle not registered");
      return;
    }

    console.log("✅ SoilType: registering toggle for schema =", schema);

    const updateButton = (active) => {
      const btn = document.getElementById("btnSoilType");
      if (btn) btn.classList.toggle("active-tool", active);
    };

    window.toggleSoilType = async () => {
      console.log("🖱️ toggleSoilType called");

      if (!map || !schema || isBusy.current) {
        console.log("⏸️ SoilType: map/schema missing or busy");
        return;
      }
      isBusy.current = true;

      // === Turn off layer ===
      if (window.soilTypeLayerGroup) {
        console.log("🔄 Removing SoilType layer");
        if (map.hasLayer(window.soilTypeLayerGroup)) {
          map.removeLayer(window.soilTypeLayerGroup);
        }
        window.soilTypeLayerGroup = null;
        updateButton(false);
        if (window.removeThematicLegend)
          window.removeThematicLegend("soiltype");
        isBusy.current = false;
        return;
      }

      // === Turn on: Fetch ===
      const url = `${API}/single-table?schema=${schema}&table=SoilType`;
      try {
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
          console.warn("⚠️ SoilType: no 'type' column, showing column picker");
          setShowColumnPopup(true);
          return;
        }

        renderLayer(geojson, "type");
        updateButton(true);
      } catch (err) {
        console.error("❌ SoilType fetch error:", err);
      } finally {
        isBusy.current = false;
      }
    };
  }, [map, schema]);

  const renderLayer = (geojson, column) => {
    console.log("🎨 Rendering SoilType layer by column:", column);

    const types = Array.from(
      new Set(
        geojson.features.map((f) =>
          (f.properties?.[column] || "Unknown").toString().trim().toLowerCase()
        )
      )
    ).sort();

    const colorMap = {};
    types.forEach((type, i) => {
      colorMap[type] = generateColor(i);
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
          fillOpacity: 0.5,
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
    window.soilTypeLayerGroup = group;

    if (window.addThematicLegend) {
      window.addThematicLegend(
        "soiltype",
        <>
          <strong>Soil Type Legend:</strong>
          <div className="legend-items">
            {types.map((val, idx) => (
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

  const handleApplyColumn = (col) => {
    console.log("📊 User applied column:", col);
    setSelectedColumn(col);
    setShowColumnPopup(false);
    if (geojsonData) {
      renderLayer(geojsonData, col);
      document.getElementById("btnSoilType")?.classList.add("active-tool");
    }
  };

  return (
    <>
      {showColumnPopup && (
        <Table_Column
          schema={schema}
          table="SoilType"
          onApply={handleApplyColumn}
          onClose={() => setShowColumnPopup(false)}
        />
      )}
    </>
  );
};

export default SoilType;
