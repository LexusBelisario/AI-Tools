// src/components/Thematic/LandClass.jsx
import React, { useEffect, useState, useRef } from "react";
import { useMap } from "react-leaflet";
import { useSchema } from "../SchemaContext";
import Table_Column from "../LandLegend/Table_Column.jsx";
import API from "../../api.js";

const LandClass = () => {
  const map = useMap(); // ✅ get map from react-leaflet
  const { schema } = useSchema(); // ✅ schema from context
  const [showColumnSelector, setShowColumnSelector] = useState(false);
  const [selectedColumn, setSelectedColumn] = useState(null);
  const [geojsonData, setGeojsonData] = useState(null);
  const isBusy = useRef(false); // ✅ persistent flag
  const [layer, setLayer] = useState(null);

  const generateColor = () => {
    const r = Math.floor(100 + Math.random() * 155);
    const g = Math.floor(100 + Math.random() * 155);
    const b = Math.floor(100 + Math.random() * 155);
    return `rgb(${r},${g},${b})`;
  };

  const hasZoneColumn = (features) => {
    if (!features || features.length === 0) return false;
    const props = features[0].properties || {};
    return "Zone" in props || "zone" in props || "ZONE" in props;
  };

  useEffect(() => {
    if (!schema) {
      console.log("⚠️ LandClass: no schema, toggle not registered");
      return;
    }

    console.log("✅ LandClass: registering toggle for schema =", schema);

    const updateButton = (active) => {
      const btn = document.getElementById("btnLandClass");
      if (btn) btn.classList.toggle("active-tool", active);
    };

    window.toggleLandClassLayer = async () => {
      console.log("🖱️ toggleLandClassLayer called");

      if (!map || !schema || isBusy.current) {
        console.log("⏸️ LandClass: map/schema missing or busy");
        return;
      }
      isBusy.current = true;

      try {
        // === Turn OFF
        if (layer) {
          console.log("🔄 Removing LandClass layer");
          if (map.hasLayer(layer)) {
            map.removeLayer(layer);
          }
          setLayer(null);
          updateButton(false);
          window.removeThematicLegend?.("landclass");
          isBusy.current = false;
          return;
        }

        // === Turn ON
        const url = `${API}/single-table?schema=${schema}&table=LandClass`;
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

        let columnToStyle = selectedColumn;

        if (!columnToStyle) {
          if (hasZoneColumn(geojson.features)) {
            const props = geojson.features[0].properties;
            if ("Zone" in props) columnToStyle = "Zone";
            else if ("zone" in props) columnToStyle = "zone";
            else if ("ZONE" in props) columnToStyle = "ZONE";
            console.log(`✅ Using default Zone column: ${columnToStyle}`);
          } else {
            console.log("⚠️ No 'Zone' column found, showing column selector");
            setShowColumnSelector(true);
            isBusy.current = false;
            return;
          }
        }

        renderLayer(geojson, columnToStyle);
        updateButton(true);
      } catch (err) {
        console.error("❌ LandClass fetch error:", err);
      } finally {
        isBusy.current = false;
      }
    };
  }, [map, schema, layer, selectedColumn]);

  const renderLayer = (geojson, columnName) => {
    console.log("🎨 Rendering LandClass by column:", columnName);

    const columnValues = geojson.features.map(
      (f) => f.properties?.[columnName]
    );
    const uniqueValues = Array.from(
      new Set(
        columnValues.map((v) => v?.toString().toUpperCase()).filter((v) => v)
      )
    );

    const colorMap = {};
    uniqueValues.forEach((v) => {
      colorMap[v] = generateColor();
    });

    const group = L.geoJSON(geojson, {
      style: (feature) => {
        const v = feature.properties?.[columnName]?.toString().toUpperCase();
        return {
          color: "#444",
          weight: 1,
          fillColor: colorMap[v] || "#ccc",
          fillOpacity: 0.6,
        };
      },
      onEachFeature: (feature, layer) => {
        const val = feature.properties?.[columnName];
        layer.bindPopup(`<strong>${columnName}:</strong> ${val || "N/A"}`);
      },
    });

    group.addTo(map);
    setLayer(group);

    if (window.addThematicLegend) {
      window.addThematicLegend(
        "landclass",
        <>
          <strong>Land Classification ({columnName}):</strong>
          <div className="legend-items">
            {uniqueValues.map((val, idx) => (
              <div key={idx}>
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
    setShowColumnSelector(false);
    setSelectedColumn(columnName);
    if (geojsonData) {
      renderLayer(geojsonData, columnName);
      document.getElementById("btnLandClass")?.classList.add("active-tool");
    }
  };

  const handleCloseColumnSelector = () => {
    setShowColumnSelector(false);
  };

  return (
    <>
      {showColumnSelector && (
        <Table_Column
          schema={schema}
          table="LandClass"
          onApply={handleColumnSelection}
          onClose={handleCloseColumnSelector}
        />
      )}
    </>
  );
};

export default LandClass;
