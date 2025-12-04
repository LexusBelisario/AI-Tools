import React, { useState } from "react";
import "./Search.css";
import { useSchema } from "../SchemaContext";
import PropertySearch from "./PropertySearch";
import RoadSearch from "./RoadSearch";
import LandmarkSearch from "./LandmarkSearch";

const Search = ({ visible, onClose }) => {
  const { schema } = useSchema();
  const [activeTab, setActiveTab] = useState("property");

  // 🔹 Handle switching between Property / Road / Landmark tabs
  const handleTabSwitch = (tab) => {
    setActiveTab(tab);
    if (window.closeReactInfoTool) window.closeReactInfoTool();
  };

  // 🔹 Handle close button
  const handleClose = () => {
    onClose?.();
    if (window.closeReactInfoTool) window.closeReactInfoTool();
  };

  // 🔹 If popup is hidden, render nothing
  if (!visible) return null;

  return (
    <div className={`search-popup ${visible ? "visible" : ""}`} id="searchPopup">
      <div id="searchPopupInner">
        {/* === Header === */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ margin: 0 }}>Search</h2>
          <button className="search-close-btn" onClick={handleClose}>
            <strong>CLOSE</strong>
          </button>
        </div>

        {/* === Tabs === */}
        <div className="button-row" style={{ marginTop: "10px" }}>
          {[
            { key: "property", label: "Property" },
            { key: "road", label: "Road" },
            { key: "landmark", label: "Landmark" }
          ].map(({ key, label }) => (
            <button
              key={key}
              className={`search-btn ${activeTab === key ? "active" : ""}`}
              onClick={() => handleTabSwitch(key)}
            >
              {label}
            </button>
          ))}
        </div>

        {/* === Schema Reminder === */}
        {!schema && (
          <p style={{ color: "#a00", fontStyle: "italic", marginTop: "12px" }}>
            Please select a municipality/city schema first.
          </p>
        )}

        {/* === Tab Content === */}
        {schema && (
          <>
            {activeTab === "property" && <PropertySearch />}
            {activeTab === "road" && <RoadSearch />}
            {activeTab === "landmark" && <LandmarkSearch />}
          </>
        )}
      </div>
    </div>
  );
};

export default Search;
