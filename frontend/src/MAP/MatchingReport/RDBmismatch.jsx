// frontend/src/MAP/MatchingReport/RDBMismatch.jsx
import React, { useMemo, useState } from "react";
import "./MatchingReport.css";

const RDBMismatch = ({ rows = [], columns = [], loading = false, error = null }) => {
  const [selectedBrgy, setSelectedBrgy] = useState("All");

  // Determine which column to use as barangay reference
  const brgyField = useMemo(() => {
    if (!columns || columns.length === 0) return null;

    if (columns.includes("brgy_nm")) return "brgy_nm";
    if (columns.includes("brgy_code")) return "brgy_code";

    // Fallback: first column whose name contains "brgy"
    const fallback = columns.find((c) => c.toLowerCase().includes("brgy"));
    return fallback || null;
  }, [columns]);

  // Collect distinct barangay values from the chosen field
  const barangays = useMemo(() => {
    if (!brgyField) return ["All"];

    const set = new Set();
    rows.forEach((row) => {
      const val = row[brgyField];
      if (val && String(val).trim() !== "") {
        set.add(String(val).trim());
      }
    });

    return ["All", ...Array.from(set).sort()];
  }, [rows, brgyField]);

  // Filter rows based on selected barangay
  const filteredRows = useMemo(() => {
    if (!brgyField || selectedBrgy === "All") return rows;
    return rows.filter((row) => String(row[brgyField]).trim() === selectedBrgy);
  }, [rows, selectedBrgy, brgyField]);

  return (
    <div className="matching-body">
      <p className="matching-description">
        Displays all <strong>PINs</strong> and corresponding records that exist in the{" "}
        <strong>RPIS database (RDB)</strong> but are missing from the{" "}
        <strong>GIS database (GDB)</strong>.
      </p>

      {/* Barangay filter toolbar (top-right, compact) */}
      {brgyField && (
        <div className="matching-toolbar">
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
        </div>
      )}

      <div className="matching-table-container scrollable">
        <table className="matching-table left-align">
          <thead>
            <tr>
              {columns && columns.length > 0 ? (
                columns.map((col, index) => (
                  <th key={index}>{col.toUpperCase()}</th>
                ))
              ) : (
                <th>No Columns Found</th>
              )}
            </tr>
          </thead>

          <tbody>
            {loading ? (
              <tr>
                <td colSpan={columns.length || 1} className="no-data-cell">
                  Checking records...
                </td>
              </tr>
            ) : filteredRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length || 1} className="no-data-cell">
                  No RPIS-only PINs found.
                </td>
              </tr>
            ) : (
              filteredRows.map((row, i) => (
                <tr key={i}>
                  {columns.map((col) => (
                    <td key={col}>{row[col] ?? "-"}</td>
                  ))}
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

export default RDBMismatch;
