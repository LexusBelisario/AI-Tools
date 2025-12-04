import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import * as L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./PredictedMapModal.css";
import API from "../../api.js";

// 🌈 Enhanced color scale
const getColor = (val, min, max) => {
  if (val == null || isNaN(val)) return "#ccc";

  const range = max - min;
  if (range === 0) return "#3388ff";

  const normalized = (val - min) / range;

  if (normalized < 0.2) return "#d73027";
  if (normalized < 0.4) return "#fc8d59";
  if (normalized < 0.6) return "#fee08b";
  if (normalized < 0.8) return "#d9ef8b";
  return "#1a9850";
};

const FitBoundsOnce = ({ data }) => {
  const map = useMap();
  useEffect(() => {
    if (data?.features?.length) {
      const geojson = L.geoJSON(data);
      const bounds = geojson.getBounds();
      if (bounds.isValid()) map.fitBounds(bounds);
    }
  }, [data, map]);
  return null;
};

const PredictedMapModal = ({ onClose, geojsonUrl = null }) => {
  const [geojson, setGeojson] = useState(null);
  const [loading, setLoading] = useState(false);
  const [minMax, setMinMax] = useState([0, 0]);
  const [ranges, setRanges] = useState([]);

  useEffect(() => {
    const fetchGeo = async () => {
      setLoading(true);
      try {
        const url =
          geojsonUrl && geojsonUrl.includes("/preview-geojson")
            ? geojsonUrl
            : geojsonUrl ||
              `${API}/linear-regression/predicted-geojson?table=Predicted_Output`;

        console.log("=".repeat(60));
        console.log("🗺️ PREDICTED MAP MODAL DEBUG");
        console.log("=".repeat(60));
        console.log("🌐 URL:", url);

        const res = await fetch(url);
        console.log("📡 Response status:", res.status);

        const data = await res.json();
        console.log("📦 Data received:", data);
        console.log("📦 Features count:", data?.features?.length);

        if (!data?.features) {
          throw new Error("Invalid GeoJSON data.");
        }

        // Sample first feature
        console.log("📋 First feature:", data.features[0]);
        console.log(
          "📋 First feature properties:",
          data.features[0]?.properties
        );

        // Extract predictions
        const preds = data.features
          .map((f, i) => {
            const val = parseFloat(f.properties.prediction);
            if (i < 3)
              console.log(
                `   Feature ${i} prediction:`,
                f.properties.prediction,
                "→",
                val
              );
            return val;
          })
          .filter((v) => !isNaN(v));

        console.log("📊 Total valid predictions:", preds.length);
        console.log("📊 Sample predictions:", preds.slice(0, 10));

        if (preds.length === 0) {
          throw new Error("No valid prediction values found");
        }

        const min = Math.min(...preds);
        const max = Math.max(...preds);

        console.log("📊 MIN:", min);
        console.log("📊 MAX:", max);
        console.log("📊 RANGE:", max - min);

        setMinMax([min, max]);

        // Create ranges
        const range = max - min;
        let bucketSize;

        if (range < 50) bucketSize = 5;
        else if (range < 100) bucketSize = 10;
        else if (range < 500) bucketSize = 50;
        else if (range < 1000) bucketSize = 100;
        else if (range < 5000) bucketSize = 500;
        else if (range < 10000) bucketSize = 1000;
        else bucketSize = Math.ceil(range / 10 / 1000) * 1000;

        console.log("📊 Bucket size:", bucketSize);

        const startValue = Math.floor(min / bucketSize) * bucketSize;
        console.log("📊 Start value:", startValue);

        const newRanges = [];
        let current = startValue;

        while (current < max && newRanges.length < 20) {
          const rangeEnd = current + bucketSize;
          newRanges.push([current, rangeEnd]);
          console.log(`   Range ${newRanges.length}: ${current} - ${rangeEnd}`);
          current = rangeEnd;
        }

        console.log("✅ Final ranges array:", newRanges);
        console.log("✅ Ranges count:", newRanges.length);

        setRanges(newRanges);
        setGeojson(data);

        console.log("=".repeat(60));
      } catch (err) {
        console.error("❌ ERROR:", err);
        alert(`Failed to load map: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    fetchGeo();
  }, [geojsonUrl]);

  console.log("🎨 RENDER - ranges state:", ranges);
  console.log("🎨 RENDER - minMax state:", minMax);
  console.log("🎨 RENDER - geojson:", geojson ? "loaded" : "null");

  const styleFeature = (feature) => {
    const predValue = feature.properties.prediction;

    return {
      color: "#000",
      weight: 0.5,
      fillColor: getColor(predValue, minMax[0], minMax[1]),
      fillOpacity: 0.75,
    };
  };

  return (
    <div className="predictmap-overlay" onClick={onClose}>
      <div className="predictmap-box" onClick={(e) => e.stopPropagation()}>
        <button className="predictmap-close" onClick={onClose}>
          ✕
        </button>
        <h3 className="predictmap-title">🗺️ Predicted Values Thematic Map</h3>

        {loading ? (
          <div
            style={{ textAlign: "center", marginTop: "2rem", color: "#00ff9d" }}
          >
            Loading predicted map...
          </div>
        ) : geojson ? (
          <>
            <div className="predictmap-map">
              <MapContainer
                zoom={14}
                minZoom={5}
                maxZoom={20}
                center={[12.8797, 121.774]}
              >
                <TileLayer
                  attribution='&copy; <a href="https://osm.org">OpenStreetMap</a>'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <GeoJSON
                  data={geojson}
                  style={styleFeature}
                  onEachFeature={(feature, layer) => {
                    const val = feature.properties.prediction;
                    const formatted =
                      typeof val === "number"
                        ? val.toLocaleString(undefined, {
                            maximumFractionDigits: 2,
                          })
                        : "N/A";

                    layer.bindTooltip(
                      `<div style="font-size:13px;">
                        <div style="color: #00ff9d; margin-bottom: 3px;">Prediction</div>
                        <div style="font-size: 14px; font-weight: 600;">${formatted}</div>
                      </div>`,
                      { sticky: true, direction: "top", opacity: 0.95 }
                    );
                  }}
                />
                <FitBoundsOnce data={geojson} />
              </MapContainer>
            </div>

            <div className="predictmap-legend">
              <div className="legend-header">
                <h4>Legend (Predicted Values)</h4>
                <div className="legend-stats">
                  <span>Min: {minMax[0].toFixed(0)}</span>
                  <span>Max: {minMax[1].toFixed(0)}</span>
                </div>
              </div>

              {/* 🔍 DEBUG INFO */}
              <div
                style={{
                  fontSize: "10px",
                  color: "#ff6",
                  marginBottom: "8px",
                  fontFamily: "monospace",
                }}
              >
                DEBUG: ranges.length = {ranges.length} | minMax = [{minMax[0]},{" "}
                {minMax[1]}]
              </div>

              <div className="legend-items">
                {ranges.length === 0 ? (
                  <div style={{ color: "#ff4d4d", fontSize: "12px" }}>
                    ⚠️ No ranges generated! Check console.
                  </div>
                ) : (
                  ranges.map(([rangeMin, rangeMax], i) => {
                    const midpoint = (rangeMin + rangeMax) / 2;

                    return (
                      <div key={i} className="legend-item">
                        <span
                          className="legend-color"
                          style={{
                            background: getColor(
                              midpoint,
                              minMax[0],
                              minMax[1]
                            ),
                          }}
                        ></span>
                        <span className="legend-label">
                          {rangeMin.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}{" "}
                          –{" "}
                          {rangeMax.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}
                        </span>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </>
        ) : (
          <p
            style={{ textAlign: "center", marginTop: "2rem", color: "#ff4d4d" }}
          >
            No map data available.
          </p>
        )}
      </div>
    </div>
  );
};

export default PredictedMapModal;
