// RoadInfo.jsx

import React from "react";

// === Embedded CSS Styles ===
// These styles define a fixed-position panel on the right side of the map
const styles = `
  #roadInfoPanel {
    position: fixed;                /* Keeps the panel always visible */
    top: 80px;                      /* Offset from top to avoid toolbar */
    right: 0;                       /* Align to right edge of viewport */
    width: 300px;
    height: auto;
    max-height: 60vh;              /* Limit vertical height to 60% of screen */
    background-color: #f2f2f2;
    box-shadow: -2px 0 12px rgba(0, 0, 0, 0.3); /* Subtle shadow on the left edge */
    z-index: 2100;                  /* Stacks above map layers */
    display: flex;
    flex-direction: column;
  }

  .info-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 16px 4px;
    font-weight: bold;
  }

  .info-content {
    padding: 16px;
  }

  .info-grid {
    display: grid;
    grid-template-columns: 1fr;    /* Single column layout */
    gap: 12px;
  }

  .info-grid label {
    display: flex;
    flex-direction: column;
    font-size: 13px;
    font-weight: bold;
  }

  .info-grid input {
    font-size: 13px;
    padding: 4px 6px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background-color: #f9f9f9;
  }

  .close-btn {
    font-size: 13px;
    font-weight: bold;
    padding: 6px 10px;
    border-radius: 4px;
    border: 1px solid #000;
    background-color: #fff;
    cursor: pointer;
  }

  .close-btn:hover {
    background-color: #ad1414;
    color: white;
  }
`;

// === Component Definition ===
// This panel shows road information (road name and type) when visible is true
const RoadInfo = ({ visible, onClose, data }) => {
  if (!visible || !data) return null; // Do not render if not visible or no data

  return (
    <>
      <style>{styles}</style> {/* Inline style injection into the component */}
      <div id="roadInfoPanel">
        <div className="info-header">
          <h3>Road Information</h3>
          <button className="close-btn" onClick={onClose}>Close</button>
        </div>
        <div className="info-content">
          <div className="info-grid">
            <label>
              ROAD NAME
              <input value={data.road_name || ""} readOnly />
            </label>
            <label>
              ROAD TYPE
              <input value={data.road_type || ""} readOnly />
            </label>
          </div>
        </div>
      </div>
    </>
  );
};

export default RoadInfo;
