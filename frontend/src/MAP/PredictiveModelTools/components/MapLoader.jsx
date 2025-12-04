import React from "react";
import "./MapLoader.css";

export default function MapLoader({ isLoading, fieldName = "map data" }) {
  if (!isLoading) return null;

  return (
    <div className="map-loader-overlay">
      <div className="map-loader-container">
        <div className="earth-icon">🌍</div>
        <div className="map-loader-message">
          Loading {fieldName} visualization
        </div>
        <div className="map-loader-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>
  );
}