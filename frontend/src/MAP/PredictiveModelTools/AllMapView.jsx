import React, { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { useNavigate } from "react-router-dom";

import BaseMapSelector from "../BasemapSelector/BaseMapSelector.jsx";
import AIToolsModal from "./AIToolsModal.jsx";

const DEFAULT_CENTER = [12.8797, 121.774];
const DEFAULT_ZOOM = 6;

const colorFor = (val, min, max) => {
  if (val == null || isNaN(val)) return "#ccc";

  const range = Math.max(1e-9, max - min);
  const normalized = (val - min) / range; // 0 to 1

  if (normalized < 0.2) return "#1a9850"; // Green
  if (normalized < 0.4) return "#d9ef8b"; // Light green
  if (normalized < 0.6) return "#fee08b"; // Yellow
  if (normalized < 0.8) return "#fc8d59"; // Orange
  return "#d73027"; // Red
};
function FitOnce({ data }) {
  const map = useMap();
  useEffect(() => {
    if (!data) return;
    try {
      const gj = new window.L.geoJSON(data);
      const b = gj.getBounds();
      if (b.isValid()) map.fitBounds(b);
    } catch {}
  }, [data, map]);
  return null;
}

export default function AllAIMapView() {
  const navigate = useNavigate();
  const [panelOpen, setPanelOpen] = useState(false);
  const [overlay, setOverlay] = useState(null);
  const [viewMode, setViewMode] = useState("predicted");

  const handleShowMap = async ({
    url,
    label = "AI Result",
    predictionField = "prediction",
    actualField = null,
    predictionRange = null,
    actualRange = null,
  }) => {
    try {
      console.log("🗺️ Loading map overlay:", {
        url,
        label,
        predictionField,
        actualField,
        predictionRange,
        actualRange,
      });

      const res = await fetch(url);
      const gj = await res.json();

      if (!gj?.features?.length) {
        alert("No GeoJSON features returned.");
        return;
      }

      console.log(`✅ Loaded ${gj.features.length} features`);

      // ✅ Calculate ranges and residuals from GeoJSON
      let finalPredictionRange = predictionRange;
      let finalActualRange = actualRange;
      const residuals = [];

      if (!finalPredictionRange && predictionField) {
        const vals = gj.features
          .map((f) => parseFloat(f.properties?.[predictionField]))
          .filter((v) => !isNaN(v));
        if (vals.length > 0) {
          finalPredictionRange = {
            min: Math.min(...vals),
            max: Math.max(...vals),
          };
        }
      }

      if (!finalActualRange && actualField) {
        const vals = gj.features
          .map((f) => parseFloat(f.properties?.[actualField]))
          .filter((v) => !isNaN(v));
        if (vals.length > 0) {
          finalActualRange = {
            min: Math.min(...vals),
            max: Math.max(...vals),
          };
        }
      }

      // ✅ Calculate residuals for each feature
      if (actualField && predictionField) {
        gj.features.forEach((f) => {
          const actual = parseFloat(f.properties?.[actualField]);
          const pred = parseFloat(f.properties?.[predictionField]);

          if (!isNaN(actual) && !isNaN(pred)) {
            const residual = actual - pred;
            residuals.push(residual);
          }
        });
      }

      // ✅ Calculate residual statistics
      let residualStats = null;
      if (residuals.length > 0) {
        const mean = residuals.reduce((a, b) => a + b, 0) / residuals.length;
        const mae =
          residuals.reduce((a, b) => a + Math.abs(b), 0) / residuals.length;
        const rmse = Math.sqrt(
          residuals.reduce((a, b) => a + b * b, 0) / residuals.length
        );

        residualStats = {
          min: Math.min(...residuals),
          max: Math.max(...residuals),
          mean: mean,
          mae: mae,
          rmse: rmse,
        };

        console.log("📊 Residual stats:", residualStats);
      }

      setOverlay({
        label,
        predictionField,
        actualField,
        data: gj,
        predictionRange: finalPredictionRange,
        actualRange: finalActualRange,
        residualStats: residualStats,
      });

      setViewMode("predicted");
    } catch (e) {
      console.error("❌ Map load error:", e);
      alert("Failed to load map preview.");
    }
  };

  const clearOverlay = () => {
    setOverlay(null);
    setViewMode("predicted");
  };

  const currentField =
    viewMode === "predicted" ? overlay?.predictionField : overlay?.actualField;

  const currentRange =
    viewMode === "predicted" ? overlay?.predictionRange : overlay?.actualRange;

  const minMax = useMemo(() => {
    if (!currentRange) return [0, 0];
    return [currentRange.min, currentRange.max];
  }, [currentRange]);

  return (
    <div className="relative w-full h-screen bg-[#0a0a0a] text-white">
      {/* MAP */}
      <MapContainer
        center={DEFAULT_CENTER}
        zoom={DEFAULT_ZOOM}
        minZoom={5}
        style={{ width: "100%", height: "100%" }}
        className="z-0"
      >
        <BaseMapSelector />

        <TileLayer
          attribution='&copy; <a href="https://osm.org">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {overlay?.data && currentField && (
          <>
            <GeoJSON
              key={`${viewMode}-${currentField}`}
              data={overlay.data}
              style={(feature) => ({
                color: "#000",
                weight: 0.5,
                fillColor: colorFor(
                  feature.properties?.[currentField],
                  minMax[0],
                  minMax[1]
                ),
                fillOpacity: 0.75,
              })}
              onEachFeature={(feature, layer) => {
                if (Math.random() < 0.01) {
                  // Log ~1% of features
                  console.log("🔍 Feature properties:", feature.properties);
                  console.log(
                    "   Available keys:",
                    Object.keys(feature.properties)
                  );
                  console.log("   predictionField:", overlay.predictionField);
                  console.log("   actualField:", overlay.actualField);
                }

                const predVal = feature.properties?.[overlay.predictionField];

                // ✅ Case-insensitive field lookup for actual field
                let actualVal = null;
                if (overlay.actualField) {
                  // Try exact match first
                  actualVal = feature.properties?.[overlay.actualField];

                  // If not found, try case-insensitive search
                  if (actualVal === undefined || actualVal === null) {
                    const actualFieldLower = overlay.actualField.toLowerCase();
                    const matchingKey = Object.keys(feature.properties).find(
                      (key) => key.toLowerCase() === actualFieldLower
                    );

                    if (matchingKey) {
                      actualVal = feature.properties[matchingKey];
                      if (Math.random() < 0.01) {
                        console.log(
                          `✅ Matched '${overlay.actualField}' to '${matchingKey}' (case-insensitive)`
                        );
                      }
                    }
                  }
                }

                // ✅ Calculate residual
                const residual =
                  actualVal != null && predVal != null
                    ? actualVal - predVal
                    : null;

                // ✅ EXACT values - no rounding unless necessary
                const predText =
                  typeof predVal === "number"
                    ? predVal.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })
                    : "N/A";

                const actualText =
                  typeof actualVal === "number"
                    ? actualVal.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })
                    : "N/A";

                const residualText =
                  typeof residual === "number"
                    ? residual.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })
                    : "N/A";

                const tooltipContent = `
    <div style="font-size:13px; font-family: monospace;">
      <div style="font-weight: bold; margin-bottom: 8px; color: #F7C800; border-bottom: 1px solid #F7C800; padding-bottom: 4px;">
        ${overlay.label}
      </div>
      ${
        overlay.actualField
          ? `
        <div style="margin-bottom: 6px; padding: 4px; background: rgba(59, 130, 246, 0.1); border-radius: 3px;">
          <div style="color: #94a3b8; font-size: 10px; margin-bottom: 2px;">Actual (${overlay.actualField})</div>
          <div style="font-weight: 700; font-size: 15px; color: #3b82f6;">${actualText}</div>
        </div>
      `
          : ""
      }
      <div style="margin-bottom: 6px; padding: 4px; background: rgba(16, 185, 129, 0.1); border-radius: 3px;">
        <div style="color: #94a3b8; font-size: 10px; margin-bottom: 2px;">Predicted</div>
        <div style="font-weight: 700; font-size: 15px; color: #10b981;">${predText}</div>
      </div>
      ${
        residual !== null
          ? `
        <div style="margin-top: 8px; padding: 8px; background: rgba(30, 41, 59, 0.9); border-radius: 4px; border-left: 3px solid ${residual < 0 ? "#ef4444" : "#10b981"};">
          <div style="font-size: 10px; color: #94a3b8; margin-bottom: 3px;">RESIDUAL (Actual - Predicted)</div>
          <div style="font-weight: 700; font-size: 15px; color: ${residual < 0 ? "#ef4444" : "#10b981"};">
            ${residual >= 0 ? "+" : ""}${residualText}
          </div>
          <div style="font-size: 9px; color: #64748b; margin-top: 3px;">
            ${Math.abs(residual) < 100 ? "✓ Good prediction" : residual < 0 ? "⚠ Under-predicted" : "⚠ Over-predicted"}
          </div>
        </div>
      `
          : ""
      }
    </div>
  `;

                layer.bindTooltip(tooltipContent, {
                  sticky: true,
                  direction: "top",
                  opacity: 0.95,
                  className: "custom-tooltip",
                });
              }}
            />
            <FitOnce data={overlay.data} />
          </>
        )}
      </MapContainer>

      {/* BACK BUTTON */}
      <button
        onClick={() => navigate("/map")}
        className="absolute top-4 left-4 bg-[#111827] text-white px-4 py-2 rounded-lg shadow hover:bg-[#1f2937] border border-gray-700 z-[1100]"
      >
        ← Back
      </button>

      {/* AI TOOLS PANEL */}
      <div className="absolute top-20 left-4 z-[1100] w-[320px]">
        <button
          onClick={() => setPanelOpen(true)}
          className="w-full bg-[#0f172a]/95 border border-gray-700 rounded-2xl shadow-xl px-5 py-4 text-left hover:bg-[#1e293b] transition"
        >
          <h3 className="text-[#F7C800] text-lg font-bold tracking-wide">
            AI Tools
          </h3>
          <p className="text-xs text-gray-300 mt-1">
            Train, evaluate, and view model predictions.
          </p>
        </button>

        {overlay?.data && (
          <div className="mt-4 rounded-xl border border-gray-700 p-4 bg-[#0f172a]/95 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="text-sm font-semibold text-[#F7C800]">
                  {overlay.label}
                </div>
                <div className="text-xs text-gray-400 mt-0.5">
                  Viewing: {viewMode === "predicted" ? "Predicted" : "Actual"}{" "}
                  Values
                </div>
              </div>

              <button
                onClick={clearOverlay}
                className="text-xs bg-[#7f1d1d] hover:bg-[#b91c1c] text-white px-3 py-1.5 rounded transition"
              >
                Clear
              </button>
            </div>

            {/* Toggle Between Actual and Predicted */}
            {overlay.actualField && (
              <div className="mb-3 p-2 bg-[#1e293b] rounded-lg">
                <div className="flex gap-2">
                  <button
                    onClick={() => setViewMode("actual")}
                    className={`flex-1 px-3 py-2 rounded text-xs font-medium transition ${
                      viewMode === "actual"
                        ? "bg-[#0038A8] text-white"
                        : "bg-[#374151] text-gray-300 hover:bg-[#4b5563]"
                    }`}
                  >
                    📊 Actual
                  </button>
                  <button
                    onClick={() => setViewMode("predicted")}
                    className={`flex-1 px-3 py-2 rounded text-xs font-medium transition ${
                      viewMode === "predicted"
                        ? "bg-[#10b981] text-white"
                        : "bg-[#374151] text-gray-300 hover:bg-[#4b5563]"
                    }`}
                  >
                    🎯 Predicted
                  </button>
                </div>
              </div>
            )}

            {/* Legend */}
            <div className="space-y-3">
              <Legend
                min={minMax[0]}
                max={minMax[1]}
                title={
                  viewMode === "predicted"
                    ? "Predicted Values"
                    : "Actual Values"
                }
              />

              {/* ✅ Comparison Stats with Residuals */}
              {overlay.actualRange && overlay.predictionRange && (
                <div className="mt-3 pt-3 border-t border-gray-700">
                  <div className="text-xs font-semibold text-[#F7C800] mb-2">
                    📊 Comparison Stats
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Actual Range:</span>
                      <span className="font-mono text-white">
                        {overlay.actualRange.min.toFixed(0)} -{" "}
                        {overlay.actualRange.max.toFixed(0)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Predicted Range:</span>
                      <span className="font-mono text-white">
                        {overlay.predictionRange.min.toFixed(0)} -{" "}
                        {overlay.predictionRange.max.toFixed(0)}
                      </span>
                    </div>

                    {/* ✅ Residual Statistics */}
                    {overlay.residualStats && (
                      <>
                        <div className="pt-2 mt-2 border-t border-gray-800">
                          <div className="text-xs font-semibold text-[#a78bfa] mb-2">
                            🎯 Residual Statistics
                          </div>
                          <div className="space-y-1.5">
                            <div className="flex justify-between">
                              <span className="text-gray-400">
                                Mean Residual:
                              </span>
                              <span
                                className={`font-mono ${
                                  Math.abs(overlay.residualStats.mean) < 100
                                    ? "text-green-400"
                                    : "text-yellow-400"
                                }`}
                              >
                                {overlay.residualStats.mean >= 0 ? "+" : ""}
                                {overlay.residualStats.mean.toFixed(2)}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">MAE:</span>
                              <span className="font-mono text-white">
                                {overlay.residualStats.mae.toFixed(2)}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">RMSE:</span>
                              <span className="font-mono text-white">
                                {overlay.residualStats.rmse.toFixed(2)}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">
                                Min Residual:
                              </span>
                              <span className="font-mono text-red-400">
                                {overlay.residualStats.min.toFixed(0)}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">
                                Max Residual:
                              </span>
                              <span className="font-mono text-green-400">
                                {overlay.residualStats.max.toFixed(0)}
                              </span>
                            </div>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* AI TOOLS SLIDE-IN PANEL */}
      <AIToolsModal
        isOpen={panelOpen}
        onClose={() => setPanelOpen(false)}
        onShowMap={handleShowMap}
      />
    </div>
  );
}

// ✅ Updated Legend Component with title prop
function Legend({ min, max, title = "Prediction Range" }) {
  console.log("🎨 LEGEND RENDER -", title, "- min:", min, "max:", max);

  if (!min && !max) {
    return (
      <div className="text-xs text-gray-400 text-center py-2">
        No range available.
      </div>
    );
  }

  if (!isFinite(min) || !isFinite(max)) {
    return (
      <div className="text-xs text-gray-400 text-center py-2">
        Invalid range values.
      </div>
    );
  }

  if (min === max) {
    return (
      <div className="space-y-1.5">
        <div className="text-xs font-semibold text-gray-300 mb-2 pb-1 border-b border-gray-700">
          {title}
        </div>
        <div className="text-xs text-gray-400">
          All values are{" "}
          {min.toLocaleString(undefined, { maximumFractionDigits: 2 })}
        </div>
      </div>
    );
  }

  const range = max - min;
  console.log("📊 Calculating ranges for:", { min, max, range });

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
  const ranges = [];
  let current = startValue;

  while (current < max && ranges.length < 15) {
    const rangeEnd = current + bucketSize;
    ranges.push({
      min: current,
      max: rangeEnd,
      midpoint: (current + rangeEnd) / 2,
    });
    current = rangeEnd;
  }

  console.log("✅ Generated ranges:", ranges);

  return (
    <div className="space-y-1.5">
      <div className="text-xs font-semibold text-gray-300 mb-2 pb-1 border-b border-gray-700">
        {title}
      </div>

      <div className="max-h-52 overflow-y-auto pr-1 space-y-1">
        {ranges.map((r, i) => (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span
              className="inline-block w-8 h-4 rounded border border-gray-600 shadow-sm flex-shrink-0"
              style={{ background: colorFor(r.midpoint, min, max) }}
            />
            <span className="text-gray-300 font-mono text-[10px]">
              {r.min.toLocaleString(undefined, { maximumFractionDigits: 0 })} –{" "}
              {r.max.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span>
          </div>
        ))}
      </div>

      {/* Summary stats */}
      <div className="mt-2 pt-2 border-t border-gray-700 text-xs text-gray-400 space-y-1">
        <div className="flex justify-between">
          <span>Min:</span>
          <span className="font-mono text-white">
            {Math.floor(min).toLocaleString()}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Max:</span>
          <span className="font-mono text-white">
            {Math.ceil(max).toLocaleString()}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Range:</span>
          <span className="font-mono text-white">
            {Math.ceil(range).toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
}
