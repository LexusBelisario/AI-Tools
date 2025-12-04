import React, { useState } from "react";
import SectionMap from "./SectionMap.jsx";
import PropertyIDMap from "./PropertyIDMap.jsx";
import SectionPreview from "./SectionPreview.jsx";
import "./TaxmapLayout.css";

const TaxmapLayout = ({ isVisible, onClose }) => {
  if (!isVisible) return null;

  const [activeTab, setActiveTab] = useState("section");
  const [previewData, setPreviewData] = useState(null);

  const handleApply = (payload) => {
    setPreviewData(payload);
  };

  const closePreview = () => setPreviewData(null);

  return (
    <>
      <div className="tml-overlay">
        <div className="tml-container">

          {/* Header */}
          <div className="tml-header">
            <h2 className="tml-title">Taxmap Layout</h2>
            <button className="tml-close-btn" onClick={onClose}>×</button>
          </div>

          {/* Tabs */}
          <div className="tml-tabs">
            <div
              className={`tml-tab ${activeTab === "section" ? "active" : ""}`}
              onClick={() => setActiveTab("section")}
            >
              Section
            </div>

            <div
              className={`tml-tab ${activeTab === "pid" ? "active" : ""}`}
              onClick={() => setActiveTab("pid")}
            >
              Property Identification
            </div>

            <div
              className={`tml-tab ${activeTab === "vicinity" ? "active" : ""}`}
              onClick={() => setActiveTab("vicinity")}
            >
              Vicinity
            </div>
          </div>

          {/* Content */}
          <div className="tml-content">
            {activeTab === "section" && (
              <div className="tml-section-wrapper">
                <SectionMap onApply={handleApply} onClose={onClose} />
              </div>
            )}

            {activeTab === "pid" && (
              <div className="tml-section-wrapper">
                <PropertyIDMap onApply={handleApply} onClose={onClose} />
              </div>
            )}

            {activeTab === "vicinity" && (
              <div className="tml-placeholder">
                Vicinity map layout coming soon.
              </div>
            )}
          </div>

        </div>
      </div>

      {/* FULLSCREEN PREVIEW (separate, highest layer) */}
      {previewData && (
        <SectionPreview
          payload={previewData}
          onClose={closePreview}
        />
      )}
    </>
  );
};

export default TaxmapLayout;
