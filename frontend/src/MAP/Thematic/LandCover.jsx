// src/components/Thematic/LandCover.jsx
import React, { useEffect, useState, useRef } from "react";
import { useMap } from "react-leaflet";
import { useSchema } from "../SchemaContext";
import Table_Column from "../LandLegend/Table_Column.jsx";
import API from "../../api.js";

const LandCover = () => {
  const map = useMap();
  const { schema } = useSchema();
  const [selectedColumn, setSelectedColumn] = useState("class_name");
  const [showColumnPopup, setShowColumnPopup] = useState(false);
  const [geojsonData, setGeojsonData] = useState(null);
  const isBusy = useRef(false);

  // ✅ Reuse same palette as SoilType
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
      console.log("⚠️ LandCover: no schema, toggle not registered");
      return;
    }

    console.log("✅ LandCover: registering toggle for schema =", schema);

    const updateButton = (active) => {
      const btn = document.getElementById("btnLandCover");
      if (btn) btn.classList.toggle("active-tool", active);
    };

    window.toggleLandCoverLayer = async () => {
      console.log("🖱️ toggleLandCoverLayer called");

      if (!map || !schema || isBusy.current) {
        console.log("⏸️ LandCover: map/schema missing or busy");
        return;
      }
      isBusy.current = true;

      try {
        // === Turn OFF
        if (window.landCoverLayerGroup) {
          console.log("🔄 Removing LandCover layer");
          if (map.hasLayer(window.landCoverLayerGroup)) {
            map.removeLayer(window.landCoverLayerGroup);
          }
          window.landCoverLayerGroup = null;
          updateButton(false);
          window.removeThematicLegend?.("landcover");
          isBusy.current = false;
          return;
        }

        // === Turn ON
        const url = `${API}/single-table?schema=${schema}&table=LandCover`;
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
        if (!props.hasOwnProperty("class_name")) {
          console.warn("⚠️ LandCover: no 'class_name' column, showing picker");
          setShowColumnPopup(true);
          return;
        }

        renderLayer(geojson, "class_name");
        updateButton(true);
      } catch (err) {
        console.error("❌ LandCover fetch error:", err);
      } finally {
        isBusy.current = false;
      }
    };
  }, [map, schema]);

  const renderLayer = (geojson, column) => {
    console.log("🎨 Rendering LandCover by column:", column);

    const rawValues = geojson.features.map((f) =>
      (f.properties?.[column] || "unknown").toString().trim().toLowerCase()
    );
    const unique = Array.from(new Set(rawValues)).sort();

    const colorMap = {};
    unique.forEach((val, i) => {
      colorMap[val] = generateColor(i);
    });

    const group = L.geoJSON(geojson, {
      style: (feature) => {
        const val = (feature.properties?.[column] || "unknown")
          .toString()
          .trim()
          .toLowerCase();
        return {
          color: colorMap[val] || "#ccc",
          fillColor: colorMap[val] || "#ccc",
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
    window.landCoverLayerGroup = group;

    if (window.addThematicLegend) {
      window.addThematicLegend(
        "landcover",
        <>
          <strong>Land Cover Legend:</strong>
          <div className="legend-items">
            {unique.map((val, i) => (
              <div key={i}>
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
    console.log("📊 User applied LandCover column:", col);
    setSelectedColumn(col);
    setShowColumnPopup(false);
    if (geojsonData) {
      renderLayer(geojsonData, col);
      document.getElementById("btnLandCover")?.classList.add("active-tool");
    }
  };

  return (
    <>
      {showColumnPopup && (
        <Table_Column
          schema={schema}
          table="LandCover"
          onApply={handleApplyColumn}
          onClose={() => setShowColumnPopup(false)}
        />
      )}
    </>
  );
};

export default LandCover;
