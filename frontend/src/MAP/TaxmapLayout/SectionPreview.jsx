import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./SectionPreview.css";

const SectionPreview = ({ payload, onClose }) => {
  const mapRef = useRef(null);
  const mapContainerRef = useRef(null);

  const [sectionShapes, setSectionShapes] = useState([]);
  const [otherBarangays, setOtherBarangays] = useState([]);
  const [bounds, setBounds] = useState(null);

  // ============================================================
  // 1) FETCH SECTION SHAPES
  // ============================================================
  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(
          `/api/taxmaplayout/section-boundary?schema=${payload.schema}&barangay=${payload.barangay}`
        );
        const data = await res.json();

        if (!data || !data.features || data.features.length === 0) {
          console.warn("No section geometry returned.");
          return;
        }

        setSectionShapes(data.features);
        setBounds(data.bounds);
      } catch (err) {
        console.error("Error loading section boundary:", err);
      }
    };

    load();
  }, [payload]);

  // ============================================================
  // 2) FETCH OTHER BARANGAYS
  // ============================================================
  useEffect(() => {
    const loadOthers = async () => {
      try {
        const res = await fetch(
          `/api/taxmaplayout/other-barangays?schema=${payload.schema}&exclude=${payload.barangay}`
        );
        const data = await res.json();

        if (!data || !data.features) return;

        setOtherBarangays(data.features);
      } catch (err) {
        console.error("Error loading other barangays:", err);
      }
    };

    loadOthers();
  }, [payload]);

  // ============================================================
  // 3) INITIALIZE MAP ONCE
  // ============================================================
  useEffect(() => {
    if (!mapRef.current && mapContainerRef.current) {
      mapRef.current = L.map(mapContainerRef.current, {
        zoomControl: false,
        attributionControl: false
      }).setView([14.2, 121.3], 13);
    }
  }, []);

  // ============================================================
  // 4) DRAW LAYERS (OTHER BARANGAYS → THEIR LABELS → SECTIONS → SECTION LABELS)
  // ============================================================
  useEffect(() => {
    if (!mapRef.current) return;
    if (!sectionShapes || sectionShapes.length === 0) return;

    const map = mapRef.current;

    // --- Clear old layers
    map.eachLayer((layer) => map.removeLayer(layer));

    // =====================================================
    // A) OTHER BARANGAYS (broken lines)
    // =====================================================
    let otherLayer = null;
    if (otherBarangays.length > 0) {
      otherLayer = L.geoJSON(otherBarangays, {
        style: {
          color: "#333",
          weight: 1,
          fillOpacity: 0,
          dashArray: "6 6"
        }
      }).addTo(map);
    }

    // =====================================================
    // B) OTHER BARANGAY LABELS
    // =====================================================
    if (otherBarangays.length > 0) {
      otherBarangays.forEach((feature) => {
        const name = feature?.properties?.barangay;
        const geom = feature?.geometry;
        if (!geom) return;

        const latlng = L.geoJSON(feature).getBounds().getCenter();

        const labelIcon = L.divIcon({
          className: "barangay-label",
          html: name || "",
        });

        L.marker(latlng, { icon: labelIcon }).addTo(map);
      });
    }

    // =====================================================
    // C) SECTION POLYGONS (black)
    // =====================================================
    const sectionLayer = L.geoJSON(sectionShapes, {
      style: {
        color: "#000000",
        weight: 2,
        fillOpacity: 0.1
      }
    }).addTo(map);

    // =====================================================
    // D) SECTION LABELS
    // =====================================================
    sectionShapes.forEach((feature) => {
      const sectionName = feature?.properties?.section;
      const geom = feature?.geometry;

      if (!geom) return;
      const latlng = L.geoJSON(feature).getBounds().getCenter();

      const labelIcon = L.divIcon({
        className: "section-label",
        html: sectionName || "",
      });

      L.marker(latlng, { icon: labelIcon }).addTo(map);
    });

    // =====================================================
    // E) FIT TO SECTION BOUNDS
    // =====================================================
    const b = sectionLayer.getBounds();
    if (b && b.isValid()) map.fitBounds(b);

  }, [sectionShapes, otherBarangays]);

  // ============================================================
  // RENDER
  // ============================================================
  return (
    <div className="sectionpreview-overlay">
      <div className="sectionpreview-container">
        <button className="sectionpreview-close" onClick={onClose}>×</button>

        <div
          ref={mapContainerRef}
          style={{
            width: "100%",
            height: "100%",
            background: "#eee",
          }}
        />
      </div>
    </div>
  );
};

export default SectionPreview;
