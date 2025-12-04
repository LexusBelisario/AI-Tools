// ParcelLoader.jsx — FINAL REMAKE (safe, clean, with loading handler restored)

import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";
import { useSchema } from "./SchemaContext.jsx";
import API from "../api";
import L from "leaflet";

// ============================================================
// 🔹 STRICT VISIBILITY
// ============================================================
function shouldShowParcels() {
  return window._parcelVisibilityState === true;
}

// ============================================================
// 🔹 Load ALL parcel tables for selected schemas
// ============================================================
async function loadAllGeoTables(map, selectedSchemas = []) {
  if (!selectedSchemas.length) return;

  const query = selectedSchemas
    .map((s) => `schemas=${encodeURIComponent(s)}`)
    .join("&");

  const url = `${API}/all-barangays?${query}`;

  if (window.setLoadingProgress) window.setLoadingProgress(true);

  try {
    const token =
      localStorage.getItem("access_token") ||
      localStorage.getItem("accessToken");

    const response = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        ...(token && { Authorization: `Bearer ${token}` }),
      },
    });

    if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

    const data = await response.json();

    // ========================================================
    // 🧹 Remove existing parcel layers (keep overlays)
    // ========================================================
    if (!window.parcelLayers) window.parcelLayers = [];

    window.parcelLayers.forEach(({ layer }) => map.removeLayer(layer));
    window.parcelLayers.length = 0;

    // ========================================================
    // 🆕 Draw new parcels, DO NOT auto-add
    // ========================================================
    const newLayer = L.geoJSON(data, {
      style: {
        color: "black",
        weight: 0.5,
        fillColor: "white",
        fillOpacity: 0,
      },
      onEachFeature: (feature, layer) => {
        window.parcelLayers.push({ feature, layer });
      },
    });

    // ========================================================
    // 🔹 Only add if AdminBoundaries says so
    // ========================================================
    if (shouldShowParcels()) {
      newLayer.addTo(map);
    }

    // ========================================================
    // 🔹 Bring overlays above parcels
    // ========================================================
    [
      window.highlightLayer,
      window.consolidateLayer,
      window.subdivideLayer,
      window.subdivideBaseLayer,
    ].forEach((l) => {
      if (l) {
        if (!map.hasLayer(l)) l.addTo(map);
        l.bringToFront();
      }
    });

    // 🔧 Ask AdminBoundaries to re-evaluate visibility rules
    if (window._updateBoundaryVisibility)
      window._updateBoundaryVisibility(true);
  } catch (err) {
    console.error("❌ Failed loading parcels:", err);
  } finally {
    if (window.setLoadingProgress) window.setLoadingProgress(false);
  }
}

// ============================================================
// 🔹 Load a SINGLE parcel/road/landmark table
// ============================================================
async function loadGeoTable(map, schema, table) {
  const url = `${API}/single-table?schema=${encodeURIComponent(
    schema
  )}&table=${encodeURIComponent(table)}`;

  if (window.setLoadingProgress) window.setLoadingProgress(true);

  try {
    const token =
      localStorage.getItem("access_token") ||
      localStorage.getItem("accessToken");

    const response = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        ...(token && { Authorization: `Bearer ${token}` }),
      },
    });

    if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

    const data = await response.json();

    if (!window.parcelLayers) window.parcelLayers = [];

    // Remove only previous layers of same schema/table
    const toRemove = window.parcelLayers.filter(
      (p) =>
        p.feature.properties.source_schema === schema &&
        p.feature.properties.source_table === table
    );

    toRemove.forEach((p) => map.removeLayer(p.layer));

    window.parcelLayers = window.parcelLayers.filter(
      (p) =>
        !(
          p.feature.properties.source_schema === schema &&
          p.feature.properties.source_table === table
        )
    );

    // Draw new (not auto-added)
    const newLayer = L.geoJSON(data, {
      style: {
        color: "black",
        weight: 1,
        fillColor: "white",
        fillOpacity: 0.1,
      },
      onEachFeature: (feature, layer) => {
        window.parcelLayers.push({ feature, layer });
      },
    });

    if (shouldShowParcels()) {
      newLayer.addTo(map);
    }

    // Keep overlays above
    [
      window.highlightLayer,
      window.consolidateLayer,
      window.subdivideLayer,
      window.subdivideBaseLayer,
    ].forEach((l) => {
      if (l) {
        if (!map.hasLayer(l)) l.addTo(map);
        l.bringToFront();
      }
    });

    if (window._updateBoundaryVisibility)
      window._updateBoundaryVisibility(true);
  } catch (err) {
    console.error("❌ Failed loading table:", err);
  } finally {
    if (window.setLoadingProgress) window.setLoadingProgress(false);
  }
}

// ============================================================
// 🔹 ParcelLoader Component
// ============================================================
function ParcelLoader() {
  const { schema } = useSchema();
  const map = useMap();

  const highlightRef = useRef(null);
  const consolidateRef = useRef(null);
  const subdivideRef = useRef(null);
  const subdivideBaseRef = useRef(null);

  // ============================================================
  // 🔹 Initialize overlays once
  // ============================================================
  useEffect(() => {
    if (!map) return;

    window.parcelLayers = [];

    highlightRef.current = L.geoJSON(null, {
      style: { color: "yellow", weight: 2, fillColor: "yellow", fillOpacity: 0.4 },
    });

    consolidateRef.current = L.geoJSON(null, {
      style: { color: "blue", weight: 2, fillColor: "blue", fillOpacity: 0.4 },
    });

    subdivideRef.current = L.geoJSON(null, {
      style: { color: "orange", weight: 2, fillColor: "orange", fillOpacity: 0.5 },
    });

    subdivideBaseRef.current = L.geoJSON(null, {
      style: { color: "#FF00FF", weight: 3, fillColor: "none", fillOpacity: 0 },
    });

    [
      highlightRef.current,
      consolidateRef.current,
      subdivideRef.current,
      subdivideBaseRef.current,
    ].forEach((l) => {
      if (!map.hasLayer(l)) l.addTo(map);
      l.bringToFront();
    });

    // Global overlay references
    window.highlightLayer = highlightRef.current;
    window.consolidateLayer = consolidateRef.current;
    window.subdivideLayer = subdivideRef.current;
    window.subdivideBaseLayer = subdivideBaseRef.current;

    // ============================================================
    // 🔹 Highlighter Utilities
    // ============================================================
    window.highlightFeature = (f) => {
      if (!f?.geometry) return;
      window.highlightLayer.clearLayers();
      window.highlightLayer.addData(f);
      window.highlightLayer.bringToFront();
    };

    window.clearHighlight = () =>
      window.highlightLayer && window.highlightLayer.clearLayers();

    window.addConsolidateFeature = (f) => {
      if (!f?.geometry) return;
      window.consolidateLayer.addData(f);
      window.consolidateLayer.bringToFront();
    };

    window.clearConsolidateHighlights = () =>
      window.consolidateLayer && window.consolidateLayer.clearLayers();

    window.addSubdivideFeatures = (features) => {
      window.subdivideLayer.clearLayers();
      features.forEach((f) => f?.geometry && window.subdivideLayer.addData(f));
      window.subdivideLayer.bringToFront();
    };

    window.clearSubdivideHighlights = () =>
      window.subdivideLayer && window.subdivideLayer.clearLayers();

    window.setSubdivideBaseParcel = (f) => {
      window.subdivideBaseLayer.clearLayers();
      window.subdivideBaseLayer.addData(f);
      window.subdivideBaseLayer.bringToFront();
    };

    window.clearSubdivideBaseParcel = () =>
      window.subdivideBaseLayer && window.subdivideBaseLayer.clearLayers();

    // ============================================================
    // 🔹 Expose global loading + data loaders
    // ============================================================
    window.loadAllGeoTables = loadAllGeoTables;
    window.loadGeoTable = loadGeoTable;
  }, [map]);

  // ============================================================
  // 🔹 Auto-load parcels when schema changes
  // ============================================================
  useEffect(() => {
    if (schema && map) loadAllGeoTables(map, [schema]);
  }, [schema, map]);

  return null;
}

export default ParcelLoader;
export { loadAllGeoTables, loadGeoTable };
