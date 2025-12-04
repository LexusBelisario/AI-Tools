// frontend/src/MAP/MatchingReport/MatchSummary.jsx
import React from "react";
import "./MatchingReport.css";

const MatchSummary = ({ rows = [], loading = false, error = null }) => {
  // ======================================================
  // 🔹 Render UI
  // ======================================================
  return (
    <div className="matching-body">
      <p className="matching-description">
        Compare PINs between the <strong>GIS database (GDB)</strong> and the{" "}
        <strong>RPIS database (RDB)</strong> grouped by barangay.
      </p>

      <div className="matching-table-container scrollable">
        <table className="matching-table">
          <thead>
            <tr>
              <th>Code</th>
              <th>Barangay</th>
              <th>Geographic Database (GDB)</th>
              <th>RPTA Database (RDB)</th>
              <th>RDB - GDB</th>
              <th>Total Match</th>
              <th>RDB / GDB</th>
              <th>Total Match / GDB</th>
              <th>Total Match / RDB</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan="9" className="no-data-cell">
                  Checking records...
                </td>
              </tr>
            ) : rows.length === 0 ? (
              <tr>
                <td colSpan="9" className="no-data-cell">
                  No data available yet. Click{" "}
                  <strong>"Run Matching Report"</strong> to generate results.
                </td>
              </tr>
            ) : (
              rows.map((row, i) => (
                <tr key={i}>
                  <td>{row.code}</td>
                  <td>{row.barangay}</td>
                  <td>{row.gdb}</td>
                  <td>{row.rdb}</td>
                  <td
                    style={{
                      color:
                        row.rdb_minus_gdb > 0
                          ? "green"
                          : row.rdb_minus_gdb < 0
                          ? "red"
                          : "black",
                    }}
                  >
                    {row.rdb_minus_gdb}
                  </td>
                  <td>{row.total_match}</td>
                  <td>{row.rdb_over_gdb?.toFixed(2)}%</td>
                  <td>{row.match_over_gdb?.toFixed(2)}%</td>
                  <td>{row.match_over_rdb?.toFixed(2)}%</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {error && (
        <div className="matching-error">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
};

export default MatchSummary;
