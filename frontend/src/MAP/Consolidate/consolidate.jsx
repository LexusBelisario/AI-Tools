import React, { useEffect, useState } from "react";
import { useMap } from "react-leaflet";
import { loadGeoTable } from "../ParcelLoader.jsx"; // ✅ CHANGED
import "./consolidate.css";
import API from "../../api.js";
import { useSchema } from "../SchemaContext";
import ParcelClickHandler from "../ParcelClickHandler";

const Consolidate = ({ onClose }) => {
  const { schema } = useSchema();
  const map = useMap();
  const [selected, setSelected] = useState([]);
  const [newPin, setNewPin] = useState("");
  const [showInstructions, setShowInstructions] = useState(true);
  const [lastCreatedPin, setLastCreatedPin] = useState(null);

  // === Parcel selection callback
  const handleParcelSelect = (pin, feature, isSelected) => {
    if (isSelected) setSelected((prev) => [...prev, feature]);
    else setSelected((prev) => prev.filter((f) => f.properties.pin !== pin));
  };

  // === Suggest next PIN based on prefix
  useEffect(() => {
    if (selected.length < 2) {
      setNewPin("");
      return;
    }

    const sample = selected[0].properties;
    const fullParts = sample.pin.split("-");
    if (fullParts.length !== 5) {
      setNewPin("");
      return;
    }

    const prefix = fullParts.slice(0, 4).join("-");
    const matches = (window.parcelLayers || [])
      .map((p) => p.feature?.properties?.pin)
      .filter((p) => p?.startsWith(prefix));

    const suffixes = matches
      .map((p) => {
        const parts = p.split("-");
        return parseInt(parts[4]);
      })
      .filter((n) => !isNaN(n));

    const nextSuffix = Math.max(...suffixes, 0) + 1;
    setNewPin(`${prefix}-${String(nextSuffix).padStart(3, "0")}`);
  }, [selected]);

  const removeFromList = (pin) => {
    setSelected((prev) => prev.filter((f) => f.properties.pin !== pin));
  };

  // === Highlight the newly created parcel
  const highlightNewParcel = (pin, schema, table) => {
    if (!window.parcelLayers?.length) {
      setTimeout(() => highlightNewParcel(pin, schema, table), 500);
      return;
    }

    const target = window.parcelLayers.find(
      (p) =>
        p.feature?.properties?.pin === pin &&
        p.feature?.properties?.source_schema === schema &&
        p.feature?.properties?.source_table === table
    );

    if (target?.layer) {
      window.parcelLayers.forEach(({ layer }) => {
        layer.setStyle?.({
          color: "black",
          weight: 1,
          fillColor: "white",
          fillOpacity: 0.1,
        });
      });

      target.layer.setStyle?.({
        color: "black",
        weight: 2,
        fillColor: "yellow",
        fillOpacity: 0.4,
      });

      map.fitBounds(target.layer.getBounds(), { maxZoom: 18 });
    } else {
      setTimeout(() => highlightNewParcel(pin, schema, table), 500);
    }
  };

  // === Perform merge
  const mergeAndSend = async () => {
    if (selected.length < 2 || !schema) return;

    try {
      const validGeometries = selected
        .map((f) => f.geometry)
        .filter((g) => g?.type && g?.coordinates);
      if (validGeometries.length < 2) return;

      const table = selected[0].properties.source_table;
      const firstPin = selected[0].properties.pin;

      const infoRes = await fetch(
        `${API}/parcel-info?pin=${encodeURIComponent(
          firstPin
        )}&schema=${encodeURIComponent(schema)}`
      );
      const infoJson = await infoRes.json();
      if (infoJson.status !== "success") return;

      const baseProps = {
        ...infoJson.data,
        pin: newPin,
        parcel: "",
        section: "",
      };

      Object.keys(baseProps).forEach((k) => {
        if (baseProps[k] === null || baseProps[k] === undefined) baseProps[k] = "";
      });

      const res = await fetch(`${API}/merge-parcels-postgis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          schema,
          table,
          base_props: baseProps,
          original_pins: selected.map((f) => f.properties.pin),
          geometries: validGeometries,
        }),
      });

      const json = await res.json();
      if (json.status !== "success") return;

      const createdPin = json.new_pin || newPin;
      setLastCreatedPin(createdPin);

      // Remove old layers
      window.parcelLayers
        .filter(
          (p) =>
            p.feature.properties.source_table === table &&
            p.feature.properties.source_schema === schema
        )
        .forEach((p) => map.removeLayer(p.layer));

      for (let i = window.parcelLayers.length - 1; i >= 0; i--) {
        const f = window.parcelLayers[i].feature.properties;
        if (f.source_table === table && f.source_schema === schema) {
          window.parcelLayers.splice(i, 1);
        }
      }

      // Reload updated parcels and highlight new one
      await loadGeoTable(map, schema, table);
      highlightNewParcel(createdPin, schema, table);

      // Keep the tool open and ready for next consolidation
      setSelected([]);
      setNewPin("");
    } catch (err) {
      console.error("Consolidate error:", err);
    }
  };

  const clearAndClose = () => {
    setSelected([]);
    onClose();
  };

  return (
    <>
      <ParcelClickHandler
        activeTool="consolidate"
        onConsolidateSelect={handleParcelSelect}
      />

      <div id="consolidatePopup" className="visible">
        <div className="consolidate-header">
          <h3>Consolidate Tool</h3>
          <button className="close-btn" onClick={clearAndClose}>
            Close
          </button>
        </div>

        <div className="collapsible-header">
          <button
            className="toggle-instructions-btn"
            onClick={() => setShowInstructions(!showInstructions)}
          >
            {showInstructions ? "Hide Instructions ▲" : "Show Instructions ▼"}
          </button>
        </div>

        {showInstructions && (
          <div className="consolidate-instructions">
            <p>1. Click parcels on the map to select them.</p>
            <p>
              2. Selected parcels will be highlighted in <b>blue</b> and listed
              below.
            </p>
            <p>3. A new PIN is suggested below but you can edit it.</p>
            <p>
              4. Click <strong>Consolidate and Save</strong> to merge. The new
              parcel will be highlighted automatically.
            </p>
          </div>
        )}

        {selected.length > 0 && (
          <div className="consolidate-selected-label">
            Parcels selected for consolidation (PIN):
          </div>
        )}
        {selected.length === 0 && !lastCreatedPin && (
          <p style={{ color: "gray", fontSize: "12px" }}>
            No parcels selected yet.
          </p>
        )}

        <ul className="consolidate-pin-list">
          {selected.map((f) => (
            <li key={f.properties.pin}>
              {f.properties.pin}
              <button
                className="remove-btn"
                onClick={() => removeFromList(f.properties.pin)}
                title="Remove from list"
              >
                ×
              </button>
            </li>
          ))}
        </ul>

        {selected.length >= 2 && (
          <div style={{ marginTop: "12px", paddingLeft: "12px" }}>
            <label style={{ fontSize: "13px", fontWeight: "bold" }}>
              New PIN to assign:
            </label>
            <input
              type="text"
              value={newPin}
              onChange={(e) => setNewPin(e.target.value)}
              style={{
                width: "90%",
                marginTop: "4px",
                padding: "6px",
                fontSize: "13px",
                border: "1px solid #ccc",
                borderRadius: "4px",
              }}
            />
          </div>
        )}

        {lastCreatedPin && (
          <p style={{ fontSize: "13px", marginTop: "10px", color: "#006400" }}>
            Last created parcel: <b>{lastCreatedPin}</b>
          </p>
        )}

        <div className="consolidate-popup-buttons">
          <button
            className="consolidate-btn confirm"
            onClick={mergeAndSend}
            disabled={selected.length < 2}
            style={{
              opacity: selected.length < 2 ? 0.5 : 1,
              cursor: selected.length < 2 ? "not-allowed" : "pointer",
            }}
          >
            Consolidate and Save
          </button>

          <button className="consolidate-btn cancel" onClick={clearAndClose}>
            Cancel and Discard
          </button>
        </div>
      </div>
    </>
  );
};

export default Consolidate;
