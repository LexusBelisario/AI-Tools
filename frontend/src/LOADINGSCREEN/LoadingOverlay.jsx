import React from "react";
import "./Loading.css";

const LoadingOverlay = ({ visible }) => {
  if (!visible) return null;

  return (
    <div className="loading-overlay">
      <div className="loading-content">
        <div className="spinner"></div>
        <p>Loading Parcels. Please Wait...</p>
      </div>
    </div>
  );
};

export default LoadingOverlay;
