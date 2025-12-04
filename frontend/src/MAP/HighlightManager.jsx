import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";

/**
 * HighlightManager.jsx
 * Handles highlight, consolidate, and subdivide overlay layers globally.
 */
export default function HighlightManager() {
  const map = useMap();
  const highlightLayerRef = useRef(null);
  const consolidateLayerRef = useRef(null);
  const subdivideLayerRef = useRef(null);
  const subdivideBaseLayerRef = useRef(null);

  useEffect(() => {
    if (!map) return;

    // Create overlay layers once
    highlightLayerRef.current = L.geoJSON(null, {
      style: { color: "yellow", weight: 2, fillColor: "yellow", fillOpacity: 0.4 },
    });
    consolidateLayerRef.current = L.geoJSON(null, {
      style: { color: "blue", weight: 2, fillColor: "blue", fillOpacity: 0.4 },
    });
    subdivideLayerRef.current = L.geoJSON(null, {
      style: { color: "orange", weight: 2, fillColor: "orange", fillOpacity: 0.5 },
    });
    subdivideBaseLayerRef.current = L.geoJSON(null, {
      style: { color: "#FF00FF", weight: 3, fillColor: "none", fillOpacity: 0 },
    });

    // Add to map
    const layers = [
      highlightLayerRef.current,
      consolidateLayerRef.current,
      subdivideLayerRef.current,
      subdivideBaseLayerRef.current,
    ];
    layers.forEach(layer => {
      if (!map.hasLayer(layer)) layer.addTo(map);
      layer.bringToFront();
    });

    // Expose globals
    window.highlightLayer = highlightLayerRef.current;
    window.consolidateLayer = consolidateLayerRef.current;
    window.subdivideLayer = subdivideLayerRef.current;
    window.subdivideBaseLayer = subdivideBaseLayerRef.current;

    // --- Highlight handlers
    window.highlightFeature = feature => {
      if (!feature?.geometry || !window.highlightLayer) return;
      window.highlightLayer.clearLayers();
      window.highlightLayer.addData(feature);
      if (!map.hasLayer(window.highlightLayer)) map.addLayer(window.highlightLayer);
      window.highlightLayer.bringToFront();
    };

    window.clearHighlight = () => {
      window.highlightLayer?.clearLayers();
    };

    // --- Consolidate handlers
    window.addConsolidateFeature = feature => {
      if (!feature?.geometry || !window.consolidateLayer) return;
      if (!map.hasLayer(window.consolidateLayer)) map.addLayer(window.consolidateLayer);
      window.consolidateLayer.addData(feature);
      window.consolidateLayer.bringToFront();
    };

    window.clearConsolidateHighlights = () => {
      window.consolidateLayer?.clearLayers();
    };

    // --- Subdivide handlers
    window.addSubdivideFeatures = features => {
      if (!Array.isArray(features) || !window.subdivideLayer) return;
      if (!map.hasLayer(window.subdivideLayer)) map.addLayer(window.subdivideLayer);
      window.subdivideLayer.clearLayers();
      features.forEach(f => f?.geometry && window.subdivideLayer.addData(f));
      window.subdivideLayer.bringToFront();
    };

    window.clearSubdivideHighlights = () => {
      window.subdivideLayer?.clearLayers();
    };

    window.setSubdivideBaseParcel = feature => {
      if (!feature?.geometry || !window.subdivideBaseLayer) return;
      if (!map.hasLayer(window.subdivideBaseLayer)) map.addLayer(window.subdivideBaseLayer);
      window.subdivideBaseLayer.clearLayers();
      window.subdivideBaseLayer.addData(feature);
      window.subdivideBaseLayer.bringToFront();
    };

    window.clearSubdivideBaseParcel = () => {
      window.subdivideBaseLayer?.clearLayers();
    };

    return () => {
      layers.forEach(layer => map.removeLayer(layer));
    };
  }, [map]);

  return null;
}
