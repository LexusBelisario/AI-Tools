import React, { useState, useEffect } from "react";
import { createPortal } from "react-dom";
import L from "leaflet";
import API from "../../api.js";
import { useSchema } from "../SchemaContext";
import { clearParcelHighlights } from "./SearchHelpers";
import LandmarkInfotool from "../LandmarkTools/LandmarkInfotool.jsx";
import "./Search.css";

function getIconPath(type) {
  const map = {
    "commercial entities": "Commercial Entities.svg",
    "educational entities": "Educational Entities.svg",
    "financial entities": "Financial Entities.svg",
    "fire station": "Fire Station.svg",
    "gas station": "Gas Station.svg",
    "government entities": "Government Entities.svg",
    "industrial entities": "Industrial Entities.svg",
    "medical entities": "Medical Entities.svg",
    "police station": "Police Station.svg",
    "recreational entities": "Recreational Entities.svg",
    "religious entities": "Religious Entities.svg",
    "subdivision": "Subdivision.svg",
    "telecommunication entities": "Telecommunication Entities.svg",
    "transportation entities": "Transportation Entities.svg",
  };
  const file = map[type?.toLowerCase()] || "default.svg";
  return `/icons/LandmarkIcons/${file}`;
}

const LandmarkSearch = () => {
  const { schema } = useSchema();

  const [query, setQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState("");
  const [barangayFilter, setBarangayFilter] = useState("");

  const [results, setResults] = useState([]);
  const [features, setFeatures] = useState([]);
  const [selectedFeature, setSelectedFeature] = useState(null);

  const [typeOptions, setTypeOptions] = useState([]);
  const [barangayOptions, setBarangayOptions] = useState([]);

  const [showInfo, setShowInfo] = useState(false);

  const map = window.map;
  if (!window.landmarkLayerGroup) {
    window.landmarkLayerGroup = L.layerGroup().addTo(map);
  }
  const landmarkLayerGroup = window.landmarkLayerGroup;

  // === Helper to render features on map ===
  const renderLandmarks = (landmarks) => {
    landmarkLayerGroup.clearLayers();
    landmarks.forEach((f) => {
      const props = f.properties;
      const icon = L.icon({
        iconUrl: getIconPath(props.type),
        iconSize: [32, 32],
        iconAnchor: [16, 32],
        popupAnchor: [0, -32],
      });
      const marker = L.marker(
        [f.geometry.coordinates[1], f.geometry.coordinates[0]],
        { icon }
      );
      marker.feature = f;
      marker.addTo(landmarkLayerGroup);
    });
  };

  // === Load all landmarks ===
  useEffect(() => {
    if (!schema) return;

    const load = async () => {
      try {
        const token =
          localStorage.getItem("access_token") ||
          localStorage.getItem("accessToken");

        const res = await fetch(
          `${API}/single-table?schema=${schema}&table=Landmarks`,
          {
            headers: {
              "Content-Type": "application/json",
              ...(token && { Authorization: `Bearer ${token}` }),
            },
          }
        );

        const json = await res.json();
        const feats = json.features || [];
        setFeatures(feats);

        // Populate dropdowns
        const uniqueTypes = [
          ...new Set(feats.map((f) => f.properties?.type).filter(Boolean)),
        ].sort();
        const uniqueBarangays = [
          ...new Set(feats.map((f) => f.properties?.barangay).filter(Boolean)),
        ].sort();

        setTypeOptions(uniqueTypes);
        setBarangayOptions(uniqueBarangays);

        // ✅ Automatically show all landmarks when tool is opened
        renderLandmarks(feats);
      } catch (err) {
        console.error("Failed to load landmarks:", err);
        setFeatures([]);
        setTypeOptions([]);
        setBarangayOptions([]);
      }
    };

    load();
  }, [schema]);

  // === Filter landmarks based on inputs ===
  const handleSearch = () => {
    clearParcelHighlights();
    setShowInfo(false);

    const nameTerm = query.trim().toLowerCase();
    const typeTerm = typeFilter.trim().toLowerCase();
    const barangayTerm = barangayFilter.trim().toLowerCase();

    const filtered = features.filter((f) => {
      const p = f.properties || {};
      const nameMatch =
        !nameTerm || p.name?.toLowerCase().includes(nameTerm);
      const typeMatch = !typeTerm || p.type?.toLowerCase() === typeTerm;
      const brgyMatch =
        !barangayTerm || p.barangay?.toLowerCase() === barangayTerm;
      return nameMatch && typeMatch && brgyMatch;
    });

    renderLandmarks(filtered);
    setResults(filtered);
  };

  // === Handle click on search result ===
  const handleClickResult = (f) => {
    const bounds = L.geoJSON(f.geometry).getBounds();
    map.fitBounds(bounds, { maxZoom: 18 });

    // ✅ Fix: find the actual marker object and change its icon safely
    landmarkLayerGroup.eachLayer((layer) => {
      if (layer instanceof L.Marker && layer.feature) {
        const isSelected = layer.feature === f;
        const icon = L.icon({
          iconUrl: getIconPath(layer.feature.properties.type),
          iconSize: [isSelected ? 48 : 32, isSelected ? 48 : 32],
          iconAnchor: [24, 48],
        });
        layer.setIcon(icon);
      }
    });

    setSelectedFeature(f);
    setShowInfo(true);
  };

  // === Clear form and reset map ===
  const handleClear = () => {
    setQuery("");
    setTypeFilter("");
    setBarangayFilter("");
    setResults([]);
    setSelectedFeature(null);
    clearParcelHighlights();
    landmarkLayerGroup.clearLayers();
    renderLandmarks(features);
    setShowInfo(false);
  };

  return (
    <>
      <div className="tab-content">
        {/* === Filters === */}
        <div className="field-grid">
          <div className="field-cell" style={{ gridColumn: "span 3" }}>
            <label>Landmark Name</label>
            <input
              type="text"
              value={query}
              placeholder="Search by name..."
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            />
          </div>

          <div className="field-cell" style={{ gridColumn: "span 2" }}>
            <label>Type</label>
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
            >
              <option value="">-- Select Type --</option>
              {typeOptions.map((t, i) => (
                <option key={i} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          <div className="field-cell" style={{ gridColumn: "span 2" }}>
            <label>Barangay</label>
            <select
              value={barangayFilter}
              onChange={(e) => setBarangayFilter(e.target.value)}
            >
              <option value="">-- Select Barangay --</option>
              {barangayOptions.map((b, i) => (
                <option key={i} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* === Buttons === */}
        <div className="button-row">
          <button className="search-btn" onClick={handleSearch}>
            Search
          </button>
          <button className="clear-btn" onClick={handleClear}>
            Clear
          </button>
        </div>

        {/* === Results === */}
        {results.length > 0 && (
          <div className="search-results">
            <p>
              <b>Results:</b> {results.length}
            </p>
            <ul>
              {results.map((f, idx) => (
                <li
                  key={idx}
                  className={
                    selectedFeature?.properties?.id === f.properties?.id
                      ? "selected"
                      : ""
                  }
                  onClick={() => handleClickResult(f)}
                >
                  {f.properties?.name || "Unnamed Landmark"}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* ✅ Info Panel Portal */}
      {showInfo &&
        createPortal(
          <LandmarkInfotool
            visible={showInfo}
            data={selectedFeature}
            onClose={() => setShowInfo(false)}
          />,
          document.body
        )}
    </>
  );
};

export default LandmarkSearch;
