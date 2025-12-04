// // === Layer Arrays ===
// export const parcelLayers = [];
// import API from "../api";
// window.parcelLayers = parcelLayers;

// // ============================================================
// // 🟡 Persistent Highlight Layer (Info/Edit Mode)
// // ============================================================
// window.highlightLayer = L.geoJSON(null, {
//   style: {
//     color: "yellow",
//     weight: 2,
//     fillColor: "yellow",
//     fillOpacity: 0.4,
//   },
// });

// window.highlightFeature = function (feature) {
//   if (!feature || !feature.geometry) return;
//   window.highlightLayer.clearLayers();
//   window.highlightLayer.addData(feature);
//   if (window.map && !window.map.hasLayer(window.highlightLayer)) {
//     window.highlightLayer.addTo(window.map);
//   }
// };

// window.clearHighlight = function () {
//   window.highlightLayer.clearLayers();
// };

// // ============================================================
// // 🔵 Persistent Consolidate Layer (Multi-Select Mode)
// // ============================================================
// window.consolidateLayer = L.geoJSON(null, {
//   style: {
//     color: "blue",
//     weight: 2,
//     fillColor: "blue",
//     fillOpacity: 0.4,
//   },
// });

// window.addConsolidateFeature = function (feature) {
//   if (!feature || !feature.geometry) return;
//   if (window.map && !window.map.hasLayer(window.consolidateLayer)) {
//     window.consolidateLayer.addTo(window.map);
//   }
//   window.consolidateLayer.addData(feature);
// };

// window.removeConsolidateFeature = function (pin) {
//   if (!window.consolidateLayer) return;
//   const layersToRemove = [];
//   window.consolidateLayer.eachLayer((layer) => {
//     const props = layer.feature?.properties;
//     if (props?.pin === pin) layersToRemove.push(layer);
//   });
//   layersToRemove.forEach((layer) => window.consolidateLayer.removeLayer(layer));
// };

// window.clearConsolidateHighlights = function () {
//   if (window.consolidateLayer) window.consolidateLayer.clearLayers();
// };

// // ============================================================
// // 🟠 Persistent Subdivide Result Layer (Post-Save Highlights)
// // ============================================================
// window.subdivideLayer = L.geoJSON(null, {
//   style: {
//     color: "orange",
//     weight: 2,
//     fillColor: "orange",
//     fillOpacity: 0.5,
//   },
// });

// window.addSubdivideFeatures = function (features) {
//   if (!features || !features.length) return;
//   if (window.map && !window.map.hasLayer(window.subdivideLayer)) {
//     window.subdivideLayer.addTo(window.map);
//   }
//   window.subdivideLayer.clearLayers();
//   features.forEach((f) => {
//     if (f?.geometry) window.subdivideLayer.addData(f);
//   });
// };

// window.clearSubdivideHighlights = function () {
//   if (window.subdivideLayer) window.subdivideLayer.clearLayers();
// };

// // ============================================================
// // 🟣 Persistent Base Parcel Layer (Chosen Parcel Before Split)
// // ============================================================
// window.subdivideBaseLayer = L.geoJSON(null, {
//   style: {
//     color: "#FF00FF", // vivid magenta outline
//     weight: 3,
//     fillColor: "none",
//     fillOpacity: 0,
//   },
// });

// window.setSubdivideBaseParcel = function (feature) {
//   if (!feature || !feature.geometry) return;
//   if (window.map && !window.map.hasLayer(window.subdivideBaseLayer)) {
//     window.subdivideBaseLayer.addTo(window.map);
//   }
//   window.subdivideBaseLayer.clearLayers();
//   window.subdivideBaseLayer.addData(feature);
// };

// window.clearSubdivideBaseParcel = function () {
//   if (window.subdivideBaseLayer) window.subdivideBaseLayer.clearLayers();
// };

// // ============================================================
// // 🗺️ Load All Parcel Features
// // ============================================================
// export async function loadAllGeoTables(map, selectedSchemas = []) {
//   if (!selectedSchemas.length) {
//     if (window.setLoadingProgress) window.setLoadingProgress(false);
//     return;
//   }

//   if (window.setLoadingProgress) window.setLoadingProgress(true);

//   const query = selectedSchemas
//     .map((s) => `schemas=${encodeURIComponent(s)}`)
//     .join("&");
//   const url = `${API}/all-barangays?${query}`;

//   try {
//     const token =
//       localStorage.getItem("access_token") ||
//       localStorage.getItem("accessToken");

//     const response = await fetch(url, {
//       headers: {
//         "Content-Type": "application/json",
//         ...(token && { Authorization: `Bearer ${token}` }),
//       },
//     });

//     if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
//     const data = await response.json();

//     // ✅ Show toolbar when data is loaded
//     const toggleBtn = document.getElementById("toggleToolbarBtn");
//     if (toggleBtn) toggleBtn.style.display = "block";

//     // 🧹 Remove existing parcel layers
//     parcelLayers.forEach(({ layer }) => map.removeLayer(layer));
//     parcelLayers.length = 0;

//     // 🆕 Create and style new parcel layer
//     const newLayer = L.geoJSON(data, {
//       style: {
//         color: "black",
//         weight: 0.5,
//         fillColor: "white",
//         fillOpacity: 0,
//       },
//       onEachFeature: (feature, layer) => {
//         parcelLayers.push({ feature, layer });
//       },
//     });

//     newLayer.addTo(map);

//     // ✅ Keep all persistent overlay layers always on top
//     if (!map.hasLayer(window.highlightLayer))
//       window.highlightLayer.addTo(map);
//     if (!map.hasLayer(window.consolidateLayer))
//       window.consolidateLayer.addTo(map);
//     if (!map.hasLayer(window.subdivideLayer))
//       window.subdivideLayer.addTo(map);
//     if (!map.hasLayer(window.subdivideBaseLayer))
//       window.subdivideBaseLayer.addTo(map);

//     // 🔔 Notify AdminBoundaries that parcels are ready
//     if (window.onParcelsLoaded) window.onParcelsLoaded();

//     // ✅ Reapply visibility logic so parcels/sections show correctly
//     if (window._updateBoundaryVisibility) window._updateBoundaryVisibility();

//     if (window.setLoadingProgress) window.setLoadingProgress(false);
//   } catch (err) {
//     console.error("❌ Failed to load geographic data:", err);
//     if (window.setLoadingProgress) window.setLoadingProgress(false);
//   }
// }

// // ============================================================
// // 🔁 Reload a Single Parcel Table
// // ============================================================
// export async function loadGeoTable(map, schema, table) {
//   const url = `${API}/single-table?schema=${encodeURIComponent(
//     schema
//   )}&table=${encodeURIComponent(table)}`;

//   if (window.setLoadingProgress) window.setLoadingProgress(true);

//   try {
//     const token =
//       localStorage.getItem("access_token") ||
//       localStorage.getItem("accessToken");

//     const response = await fetch(url, {
//       headers: {
//         "Content-Type": "application/json",
//         ...(token && { Authorization: `Bearer ${token}` }),
//       },
//     });

//     if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
//     const data = await response.json();

//     // 🧹 Remove existing layers for this schema/table
//     const toRemove = parcelLayers.filter(
//       (p) =>
//         p.feature.properties.source_table === table &&
//         p.feature.properties.source_schema === schema
//     );

//     toRemove.forEach((p) => map.removeLayer(p.layer));
//     for (let i = parcelLayers.length - 1; i >= 0; i--) {
//       const f = parcelLayers[i].feature.properties;
//       if (f.source_table === table && f.source_schema === schema) {
//         parcelLayers.splice(i, 1);
//       }
//     }

//     // 🆕 Create new layer
//     const newLayer = L.geoJSON(data, {
//       style: {
//         color: "black",
//         weight: 1,
//         fillColor: "white",
//         fillOpacity: 0.1,
//       },
//       onEachFeature: (feature, layer) => {
//         parcelLayers.push({ feature, layer });
//       },
//     });

//     newLayer.addTo(map);

//     // ✅ Keep all persistent layers always on top
//     if (!map.hasLayer(window.highlightLayer))
//       window.highlightLayer.addTo(map);
//     if (!map.hasLayer(window.consolidateLayer))
//       window.consolidateLayer.addTo(map);
//     if (!map.hasLayer(window.subdivideLayer))
//       window.subdivideLayer.addTo(map);
//     if (!map.hasLayer(window.subdivideBaseLayer))
//       window.subdivideBaseLayer.addTo(map);

//     // 🔔 Notify other modules
//     if (window.onParcelsLoaded) window.onParcelsLoaded();

//     // ✅ Reapply visibility logic
//     if (window._updateBoundaryVisibility) window._updateBoundaryVisibility();

//     if (window.setLoadingProgress) window.setLoadingProgress(false);
//   } catch (err) {
//     console.error("❌ Failed to reload parcel table:", err);
//     if (window.setLoadingProgress) window.setLoadingProgress(false);
//   }
// }

// // ============================================================
// // 🌐 Export to Global Scope
// // ============================================================
// window.loadAllGeoTables = loadAllGeoTables;
// window.loadGeoTable = loadGeoTable;
