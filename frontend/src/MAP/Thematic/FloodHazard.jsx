// src/components/Thematic/FloodHazard.jsx
import React, { useEffect, useState, useRef } from "react";
import { useMap } from "react-leaflet";
import { useSchema } from "../SchemaContext";
import Table_Column from "../LandLegend/Table_Column.jsx";
import API from "../../api.js";

const FloodHazard = () => {
  const map = useMap();
  const { schema } = useSchema();
  const [selectedColumn, setSelectedColumn] = useState("rating");
  const [showColumnPopup, setShowColumnPopup] = useState(false);
  const [geojsonData, setGeojsonData] = useState(null);
  const isBusy = useRef(false);

  // Color + labels
  const smartColorMap = {
    high: "#ff0000",
    moderate: "#ffa500",
    low: "#ffff00",
    none: "#008000",
    floodway: "#800080",
    unknown: "#ccc",
  };

  const categoryLabels = {
    high: "High",
    moderate: "Moderate",
    low: "Low",
    none: "Not Susceptible",
    floodway: "Floodway",
    unknown: "Unknown",
  };

  const classifyCategory = (rawValue) => {
    const v = (rawValue || "").toString().toLowerCase().trim();
    if (!v || v === "null" || v === "undefined") return "none";
    if (["0", "none", "not susceptible", "n/a"].includes(v)) return "none";
    if (["3", "level 3", "lvl 3"].includes(v)) return "high";
    if (["2", "level 2", "lvl 2"].includes(v)) return "moderate";
    if (["1", "level 1", "lvl 1"].includes(v)) return "low";
    if (v.includes("floodway")) return "floodway";
    if (v.includes("high")) return "high";
    if (v.includes("moderate") || v.includes("medium")) return "moderate";
    if (v.includes("low")) return "low";
    if (v.includes("not") || v.includes("none")) return "none";
    return "unknown";
  };

  useEffect(() => {
    if (!schema) {
      console.log("⚠️ FloodHazard: no schema, toggle not registered");
      return;
    }

    console.log("✅ FloodHazard: registering toggle for schema =", schema);

    const updateButton = (active) => {
      const btn = document.getElementById("btnFloodHazard");
      if (btn) btn.classList.toggle("active-tool", active);
    };

    window.toggleFloodHazard = async () => {
      console.log("🖱️ toggleFloodHazard called");

      if (!map || !schema || isBusy.current) {
        console.log("⏸️ FloodHazard: map/schema missing or busy");
        return;
      }
      isBusy.current = true;

      try {
        // === Turn OFF
        if (window.floodHazardLayerGroup) {
          console.log("🔄 Removing FloodHazard layer");
          if (map.hasLayer(window.floodHazardLayerGroup)) {
            map.removeLayer(window.floodHazardLayerGroup);
          }
          window.floodHazardLayerGroup = null;
          updateButton(false);
          window.removeThematicLegend?.("floodhazard");
          isBusy.current = false;
          return;
        }

        // === Turn ON
        const url = `${API}/single-table?schema=${schema}&table=FloodHazard`;
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

        const props = geojson.features?.[0]?.properties || {};
        if (!props.hasOwnProperty("rating")) {
          console.warn("⚠️ FloodHazard: no 'rating' column, showing picker");
          setShowColumnPopup(true);
          return;
        }

        renderLayer(geojson, "rating");
        updateButton(true);
      } catch (err) {
        console.error("❌ FloodHazard fetch error:", err);
      } finally {
        isBusy.current = false;
      }
    };
  }, [map, schema]);

  const renderLayer = (geojson, column) => {
    console.log("🎨 Rendering FloodHazard by column:", column);

    const valueToCategory = {};
    const valueToColor = {};

    geojson.features.forEach((f) => {
      const raw = f.properties?.[column];
      const val = (raw || "").toString().trim();
      const cat = classifyCategory(val);
      valueToCategory[val] = cat;
      valueToColor[val] = smartColorMap[cat] || smartColorMap.unknown;
    });

    const priority = { high: 1, moderate: 2, low: 3 };
    const uniqueValues = Object.keys(valueToColor).sort((a, b) => {
      const catA = valueToCategory[a];
      const catB = valueToCategory[b];
      return (priority[catA] || 999) - (priority[catB] || 999);
    });

    const group = L.geoJSON(geojson, {
      style: (feature) => {
        const val = (feature.properties?.[column] || "").toString().trim();
        return {
          color: valueToColor[val] || "#ccc",
          fillColor: valueToColor[val] || "#ccc",
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
    window.floodHazardLayerGroup = group;

    if (window.addThematicLegend) {
      window.addThematicLegend(
        "floodhazard",
        <>
          <strong>Flood Hazard Legend:</strong>
          <div className="legend-items">
            {uniqueValues.map((val, i) => {
              const cat = valueToCategory[val];
              return (
                <div key={i}>
                  <span
                    className="legend-swatch"
                    style={{ backgroundColor: valueToColor[val] || "#ccc" }}
                  ></span>
                  {categoryLabels[cat] || val}
                </div>
              );
            })}
          </div>
        </>
      );
    }
  };

  const handleApplyColumn = (col) => {
    console.log("📊 User applied FloodHazard column:", col);
    setSelectedColumn(col);
    setShowColumnPopup(false);
    if (geojsonData) {
      renderLayer(geojsonData, col);
      document.getElementById("btnFloodHazard")?.classList.add("active-tool");
    }
  };

  return (
    <>
      {showColumnPopup && (
        <Table_Column
          schema={schema}
          table="FloodHazard"
          onApply={handleApplyColumn}
          onClose={() => setShowColumnPopup(false)}
        />
      )}
    </>
  );
};

export default FloodHazard;
