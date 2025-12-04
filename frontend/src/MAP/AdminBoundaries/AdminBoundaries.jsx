import { useEffect, useState, useRef, useCallback } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./AdminBoundaries.css";
import { useSchema } from "../SchemaContext";
import API from "../../api";

function AdminBoundaries() {
  const map = useMap();
  const { schema } = useSchema();

  const [showMunicipal, setShowMunicipal] = useState(true);
  const [showBarangay, setShowBarangay] = useState(true);
  const [showSection, setShowSection] = useState(true);
  const [showParcels, setShowParcels] = useState(true);

  const municipalRef = useRef(null);
  const barangayLayerRef = useRef(null);
  const sectionLayerRef = useRef(null);
  const barangayLabelsRef = useRef([]);

  const limits = {
    municipal: [4, 13],
    barangay: [12, 15],
    section: [14, 19],
    parcel: [16, 25],
  };

  const getParcelStyle = () => ({
    color: window.parcelOutlineColor || "black",
    weight: 1.2,
    opacity: 1,
    fillColor: "black",
    fillOpacity: 0.1,
  });

  // 🏙️ Initialize Municipal Boundary (WMS)
  useEffect(() => {
    if (!map) return;
    municipalRef.current = L.tileLayer.wms(
      "http://104.199.142.35:8080/geoserver/MapBoundaries/wms",
      {
        layers: "MapBoundaries:PH_MunicipalMap",
        format: "image/png",
        transparent: true,
        version: "1.1.1",
        crs: L.CRS.EPSG4326,
      }
    );
  }, [map]);

  // 🗺️ Load Barangay and Section Boundaries
  useEffect(() => {
    if (!map || !schema) return;

    [barangayLayerRef, sectionLayerRef].forEach((ref) => {
      if (ref.current && map.hasLayer(ref.current)) map.removeLayer(ref.current);
      ref.current = null;
    });
    barangayLabelsRef.current.forEach((l) => map.removeLayer(l));
    barangayLabelsRef.current = [];

    const loadData = async () => {
      try {
        const res = await fetch(`${API}/municipal-boundaries?schema=${schema}`);
        const data = await res.json();
        if (data.status !== "success") return;

        const palette = [
          "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
          "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
          "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
          "#aaffc3", "#808000", "#b8926dff", "#000075", "#808080"
        ];

        const barangayColors = {};
        let brgyIndex = 0;
        data.barangay?.features?.forEach((f) => {
          const n = f.properties.barangay;
          if (n && !barangayColors[n])
            barangayColors[n] = palette[brgyIndex++ % palette.length];
        });

        const sectionColors = {};
        let secIndex = 0;
        data.section?.features?.forEach((f) => {
          const n = f.properties.barangay;
          if (n && !sectionColors[n])
            sectionColors[n] = palette[secIndex++ % palette.length];
        });

        barangayLayerRef.current = L.geoJSON(data.barangay || null, {
          style: (f) => ({
            color: "#000",
            weight: 1.2,
            fillColor: barangayColors[f.properties.barangay] || "#999",
            fillOpacity: 0.5,
          }),
          onEachFeature: (feature, layer) => {
            const name = feature.properties?.barangay;
            if (name) {
              const center = layer.getBounds().getCenter();
              const label = L.tooltip({
                permanent: true,
                direction: "center",
                className: "barangay-label",
              }).setContent(
                `<div style="
                  color:white;
                  font-size:12px;
                  font-weight:700;
                  text-shadow:
                    -1px -1px 0 #000,
                     1px -1px 0 #000,
                    -1px  1px 0 #000,
                     1px  1px 0 #000;">${name}</div>`
              ).setLatLng(center);
              barangayLabelsRef.current.push(label);
            }
          },
          interactive: false,
        });

        sectionLayerRef.current = L.geoJSON(data.section || null, {
          style: (f) => ({
            color: "#202020",
            weight: 2.0,
            fillColor: sectionColors[f.properties.barangay] || "#999",
            fillOpacity: 0.15,
          }),
          interactive: false,
        });

        updateVisibility(true);
      } catch (err) {
        console.error("❌ Boundary fetch error:", err);
      }
    };

    loadData();
  }, [map, schema]);

  // 🔍 Visibility Controller
  const updateVisibility = useCallback(
    (force = false, overrides = {}) => {
      if (!map) return;
      const zoom = map.getZoom();
      const inZoom = (min, max) => zoom >= min && zoom <= max;

      // --- Municipal
      if (municipalRef.current) {
        const visible = showMunicipal && inZoom(...limits.municipal);
        if (visible && !map.hasLayer(municipalRef.current))
          map.addLayer(municipalRef.current);
        else if (!visible && map.hasLayer(municipalRef.current))
          map.removeLayer(municipalRef.current);
      }

      // --- Barangay
      if (barangayLayerRef.current) {
        const visible = showBarangay && inZoom(...limits.barangay);
        if (visible && !map.hasLayer(barangayLayerRef.current)) {
          map.addLayer(barangayLayerRef.current);
          barangayLabelsRef.current.forEach((l) => map.addLayer(l));
        } else if (!visible && map.hasLayer(barangayLayerRef.current)) {
          map.removeLayer(barangayLayerRef.current);
          barangayLabelsRef.current.forEach((l) => map.removeLayer(l));
        }
      }

      // --- Section
      if (sectionLayerRef.current) {
        const visible = showSection && inZoom(...limits.section);
        if (visible && !map.hasLayer(sectionLayerRef.current))
          map.addLayer(sectionLayerRef.current);
        else if (!visible && map.hasLayer(sectionLayerRef.current))
          map.removeLayer(sectionLayerRef.current);
      }

      // --- Parcels (fully managed)
      if (window.parcelLayers && Array.isArray(window.parcelLayers)) {
        const zoomOK = inZoom(...limits.parcel);
        const shouldShow = showParcels && zoomOK;

        // ✅ Share visibility with ParcelLoader
        window._parcelVisibilityState = shouldShow;

        window.parcelLayers.forEach(({ layer }) => {
          if (!layer) return;
          if (shouldShow) {
            if (!map.hasLayer(layer)) map.addLayer(layer);
            layer.setStyle(getParcelStyle());
          } else if (map.hasLayer(layer)) {
            map.removeLayer(layer);
          }
        });
      }
    },
    [map, showMunicipal, showBarangay, showSection, showParcels]
  );

  // 🎯 React to zoom + checkbox changes
  useEffect(() => {
    if (!map) return;
    const handler = () => updateVisibility();
    map.on("zoomend", handler);
    return () => map.off("zoomend", handler);
  }, [map, updateVisibility]);

  useEffect(() => {
    updateVisibility();
  }, [showMunicipal, showBarangay, showSection, showParcels]);

  // 🌐 Expose Global Controls
  useEffect(() => {
    window._updateBoundaryVisibility = (force, overrides) =>
      updateVisibility(force, overrides);
    window._setShowMunicipal = (v) => setShowMunicipal(v);
    window._setShowBarangay = (v) => setShowBarangay(v);
    window._setShowSection = (v) => setShowSection(v);
    window._setShowParcels = (v) => setShowParcels(v);
  }, [updateVisibility]);

  // 🧹 Cleanup
  useEffect(() => {
    return () => {
      [barangayLayerRef, sectionLayerRef].forEach((ref) => {
        if (ref.current && map.hasLayer(ref.current)) map.removeLayer(ref.current);
      });
      barangayLabelsRef.current.forEach((l) => map.removeLayer(l));
      barangayLabelsRef.current = [];
      if (municipalRef.current && map.hasLayer(municipalRef.current))
        map.removeLayer(municipalRef.current);
    };
  }, [map]);

  return null;
}

export default AdminBoundaries;
