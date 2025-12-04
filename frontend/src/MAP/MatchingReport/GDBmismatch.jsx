// frontend/src/MAP/MatchingReport/GDBMismatch.jsx
import React, { useEffect, useState, useMemo } from "react";
import "./MatchingReport.css";

const GDBMismatch = ({ rows = [], loading = false, error = null }) => {
  const [selectedBrgy, setSelectedBrgy] = useState("All");

  // ======================================================
  // 🧹 Cleanup highlight when tool closes
  // ======================================================
  useEffect(() => {
    return () => {
      if (window.clearHighlight) {
        console.log("🧹 Clearing parcel highlights (GDB Mismatch closed)");
        window.clearHighlight();
      }
    };
  }, []);

  // ======================================================
  // 📋 Collect unique barangay names
  // ======================================================
  const barangays = useMemo(() => {
    const names = new Set();
    rows.forEach((r) => {
      if (r.brgy_nm) names.add(r.brgy_nm.trim());
      else if (r.brgy_code) names.add(r.brgy_code.trim());
    });
    return ["All", ...Array.from(names).sort()];
  }, [rows]);

  // ======================================================
  // 🔎 Filter rows by selected barangay
  // ======================================================
  const filteredRows = useMemo(() => {
    if (selectedBrgy === "All") return rows;
    return rows.filter(
      (r) => r.brgy_nm === selectedBrgy || r.brgy_code === selectedBrgy
    );
  }, [rows, selectedBrgy]);

  // ======================================================
  // 📍 Zoom to parcel by PIN
  // ======================================================
  const zoomToParcel = (pin) => {
    try {
      const map = window.map;
      const parcelLayers = window.parcelLayers;
      if (!map || !parcelLayers?.length) {
        alert("Map or parcel layers not found.");
        return;
      }

      const parcelEntry = parcelLayers.find(
        (p) => p.feature?.properties?.pin === pin
      );
      if (!parcelEntry) {
        alert(`Parcel not found for PIN: ${pin}`);
        return;
      }

      if (window.clearHighlight) window.clearHighlight();
      if (window.highlightFeature) window.highlightFeature(parcelEntry.feature);
      const bounds = L.geoJSON(parcelEntry.feature).getBounds();
      map.fitBounds(bounds, { maxZoom: 18 });

      console.log(`✅ Zoomed to parcel PIN: ${pin}`);
    } catch (err) {
      console.error("❌ Error zooming to parcel:", err);
    }
  };

  // ======================================================
  // 🧱 Render
  // ======================================================
  return (
    <div className="matching-body">
      <p className="matching-description">
        Displays all <strong>PINs</strong> that are present in the{" "}
        <strong>GIS database (GDB)</strong> but missing from the{" "}
        <strong>RPIS database (RDB)</strong>.
      </p>

      <div className="matching-table-container scrollable">
        <table className="matching-table left-align">
          <thead>
            <tr>
              <th>PIN</th>
              <th>
                <select
                  className="matching-filter-dropdown"
                  value={selectedBrgy}
                  onChange={(e) => setSelectedBrgy(e.target.value)}
                >
                  {barangays.map((b, i) => (
                    <option key={i} value={b} title={b}>
                      {b === "All" ? "Barangay: All" : `Barangay: ${b}`}
                    </option>
                  ))}
                </select>
              </th>
            </tr>
          </thead>

          <tbody>
            {loading ? (
              <tr>
                <td colSpan={2} className="no-data-cell">
                  Checking records...
                </td>
              </tr>
            ) : filteredRows.length === 0 ? (
              <tr>
                <td colSpan={2} className="no-data-cell">
                  No missing PINs found in GIS.
                </td>
              </tr>
            ) : (
              filteredRows.map((row, i) => (
                <tr
                  key={i}
                  className="clickable-row"
                  onClick={() => zoomToParcel(row.pin)}
                >
                  <td>{row.pin}</td>
                  <td>{row.brgy_nm || row.brgy_code || "-"}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {error && <div className="matching-error">⚠️ {error}</div>}
    </div>
  );
};

export default GDBMismatch;
