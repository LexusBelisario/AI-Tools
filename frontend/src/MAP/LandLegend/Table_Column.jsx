import React, { useEffect, useState } from "react";
import "./Table_Column.css";
import API from "../../api.js";

const Table_Column = ({ schema, table, onApply, onClose }) => {
  const [columns, setColumns] = useState([]);
  const [selected, setSelected] = useState("");

  useEffect(() => {
    const loadColumns = async () => {
      try {
        let url;
        if (table) {
          // For specific table requests (like LandClass)
          url = `${API}/single-table?schema=${schema}&table=${table}`;
        } else {
          // For all-barangays requests (like LandValuation)
          url = `${API}/all-barangays?schemas=${schema}`;
        }

        const res = await fetch(url);
        if (!res.ok) {
          console.error("❌ Request failed:", res.status, res.statusText);
          return;
        }

        const data = await res.json();
        if (data.features && data.features.length > 0) {
          const keys = Object.keys(data.features[0].properties);
          setColumns(keys);
        }
      } catch (err) {
        console.error("❌ Failed to load columns:", err);
      }
    };

    if (schema) loadColumns();
  }, [schema, table]);

  const handleSubmit = () => {
    if (selected) onApply(selected);
  };

  return (
    <div id="tableColumnPopup" className="visible">
      <div className="table-column-header">
        <h3>Select Column to Style</h3>
        <button className="close-btn" onClick={onClose}>
          Close
        </button>
      </div>

      <div style={{ padding: "12px 16px" }}>
        <label>Available Columns:</label>
        <select
          style={{
            marginTop: "8px",
            padding: "6px",
            fontSize: "14px",
            width: "100%",
          }}
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          <option value="">-- Select Column --</option>
          {columns.map((col) => (
            <option key={col} value={col}>
              {col}
            </option>
          ))}
        </select>

        <button
          className="table-column-btn"
          onClick={handleSubmit}
          disabled={!selected}
          style={{
            opacity: selected ? 1 : 0.5,
            cursor: selected ? "pointer" : "not-allowed",
          }}
        >
          Apply
        </button>
      </div>
    </div>
  );
};

export default Table_Column;
