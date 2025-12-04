// src/components/Thematic/RoadNetwork.jsx
import React, { useEffect, useState, useRef } from "react";
import { useMap } from "react-leaflet";
import { useSchema } from "../SchemaContext";
import Table_Column from "../LandLegend/Table_Column.jsx";
import API from "../../api.js"; // ✅ CHANGED from api_service.js to api.js

const RoadNetwork = () => {
  const map = useMap();
  const { schema } = useSchema();
  const [selectedColumn, setSelectedColumn] = useState("road_type");
  const [showColumnPopup, setShowColumnPopup] = useState(false);
  const [geojsonData, setGeojsonData] = useState(null);
  const isBusy = useRef(false);

  const predefinedColors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
  ];
  const generateColor = (index) =>
    predefinedColors[index % predefinedColors.length];

  useEffect(() => {
    if (!schema) {
      console.log("⚠️ RoadNetwork: no schema, toggle not registered");
      return;
    }

    console.log("✅ RoadNetwork: registering toggle for schema =", schema);

    const updateButton = (active) => {
      const btn = document.getElementById("btnRoadNetwork");
      if (btn) btn.classList.toggle("active-tool", active);
    };

    window.toggleRoadNetwork = async () => {
      console.log("🖱️ toggleRoadNetwork called");

      if (!map || !schema || isBusy.current) {
        console.log("⏸️ RoadNetwork: map/schema missing or busy");
        return;
      }
      isBusy.current = true;

      try {
        // === Turn OFF
        if (window.roadLayerGroup) {
          console.log("🔄 Removing RoadNetwork layer");
          if (map.hasLayer(window.roadLayerGroup)) {
            map.removeLayer(window.roadLayerGroup);
          }
          window.roadLayerGroup = null;
          updateButton(false);
          window.removeThematicLegend?.("roadnetwork");
          isBusy.current = false;
          return;
        }

        // === Turn ON
        const url = `${API}/single-table?schema=${schema}&table=RoadNetwork`;
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
        if (!props.hasOwnProperty("road_type")) {
          console.warn("⚠️ RoadNetwork: no 'road_type' column, showing picker");
          setShowColumnPopup(true);
          return;
        }

        renderLayer(geojson, "road_type");
        updateButton(true);
      } catch (err) {
        console.error("❌ RoadNetwork fetch error:", err);
      } finally {
        isBusy.current = false;
      }
    };
  }, [map, schema]);

  const renderLayer = (geojson, column) => {
    console.log("🎨 Rendering RoadNetwork by column:", column);

    const types = Array.from(
      new Set(
        geojson.features.map((f) =>
          (f.properties?.[column] || "unknown").toString().trim().toLowerCase()
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
          weight: 2,
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
    window.roadLayerGroup = group;

    if (window.addThematicLegend) {
      window.addThematicLegend(
        "roadnetwork",
        <>
          <strong>Road Network Legend:</strong>
          <div className="legend-items">
            {types.map((type, idx) => (
              <div key={idx}>
                <span
                  className="legend-swatch"
                  style={{ backgroundColor: colorMap[type] }}
                ></span>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </div>
            ))}
          </div>
        </>
      );
    }
  };

  const handleApplyColumn = (col) => {
    console.log("📊 User applied RoadNetwork column:", col);
    setSelectedColumn(col);
    setShowColumnPopup(false);
    if (geojsonData) {
      renderLayer(geojsonData, col);
      document.getElementById("btnRoadNetwork")?.classList.add("active-tool");
    }
  };

  return (
    <>
      {showColumnPopup && (
        <Table_Column
          schema={schema}
          table="RoadNetwork"
          onApply={handleApplyColumn}
          onClose={() => setShowColumnPopup(false)}
        />
      )}
    </>
  );
};

export default RoadNetwork;
