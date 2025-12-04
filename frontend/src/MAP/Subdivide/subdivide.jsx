import React, { useState } from "react";
import { useMap } from "react-leaflet";
import AddLine from "./AddLine.jsx";
import AddPoints from "./AddPoints.jsx";
import BearingDistance from "./BearingDistance.jsx";
import "./subdivide.css";

const Subdivide = ({ visible, onClose }) => {
  const map = useMap();
  const [activeTab, setActiveTab] = useState("line");

  const tabs = [
    { id: "line", label: "Add Line" },
    { id: "points", label: "Add Points" },
    { id: "bearing", label: "Bearing Distance" },
  ];

  return (
    <div id="subdividePopup" className={visible ? "visible" : ""}>
      {/* === Header === */}
      <div className="subdivide-header">
        <h3>Subdivide Tool</h3>
        <button
          className="subdivide-close-btn"
          onClick={() => {
            if (window.clearSubdivideHighlights) {
              window.clearSubdivideHighlights();
            }
            onClose();
          }}
        >
          ✕
        </button>
      </div>

      {/* === Tabs === */}
      <div className="tab-buttons">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={activeTab === tab.id ? "active" : ""}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* === Mode Components === */}
      <div style={{ padding: "8px" }}>
        {activeTab === "line" && <AddLine map={map} />}
        {activeTab === "points" && <AddPoints map={map} />}
        {activeTab === "bearing" && <BearingDistance map={map} />}
      </div>
    </div>
  );
};

export default Subdivide;
