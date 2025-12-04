import React, { useState } from "react";
import "./toolbar.css";
import TBMainToolbar from "./TBMainToolbar.jsx";
import TBLandLegendTools from "./TBLandLegendTools.jsx";
import TBThematicMaps from "./TBThematicMaps.jsx";
import TBLabelTools from "./TBLabelTools.jsx";
import TBLandmarkTools from "./TBLandmarkTools.jsx";
import { useSchema } from "../SchemaContext.jsx"; // ✅ added

const tabs = [
  { id: "parcel", label: "Parcel Main Toolbar" },
  { id: "land", label: "Land Legend Tools" },
  { id: "thematic", label: "Thematic Maps" },
  { id: "label", label: "Label Tools" },
  { id: "landmark", label: "Landmark Tools" },
];

const Toolbar = ({
  onSearchClick,
  onInfoClick,
  onAttributeEditClick,
  onConsolidateClick,
  onSubdivideClick,
  onLandmarkInfoClick,
  onShowLandmarksClick,
}) => {
  const [visible, setVisible] = useState(false);
  const [activeTool, setActiveTool] = useState(null);
  const [activeTab, setActiveTab] = useState("parcel");
  const { schema } = useSchema(); // ✅ get current schema

  const handleToggle = () => {
    setVisible((prev) => !prev);
  };

  // === Renders the buttons depending on selected tab ===
  const renderButtons = () => {
    switch (activeTab) {
      case "parcel":
        return (
          <TBMainToolbar
            activeTool={activeTool}
            setActiveTool={setActiveTool}
            onSearchClick={onSearchClick}
            onInfoClick={onInfoClick}
            onAttributeEditClick={onAttributeEditClick}
            onConsolidateClick={onConsolidateClick}
            onSubdivideClick={onSubdivideClick}
          />
        );

      case "land":
        return (
          <TBLandLegendTools
            activeTool={activeTool}
            setActiveTool={setActiveTool}
          />
        );

      case "thematic":
        return <TBThematicMaps />;

      case "label":
        return <TBLabelTools />;

      case "landmark":
        return (
          <TBLandmarkTools
            activeTool={activeTool}
            setActiveTool={setActiveTool}
            onLandmarkInfoClick={onLandmarkInfoClick}
            onShowLandmarksClick={onShowLandmarksClick}
          />
        );


      default:
        return null;
    }
  };

  return (
    <>
      {/* === Toggle Button === */}
      <button id="toggleToolbarBtn" onClick={handleToggle}>
        ☰ GISTools
      </button>

      <div
        className={`toolbar-panel ${visible ? "visible" : ""}`}
        id="toolbarPanel"
      >
        {/* === Tab Selector === */}
        <select
          className="toolbar-dropdown"
          value={activeTab}
          onChange={(e) => setActiveTab(e.target.value)}
        >
          {tabs.map((tab) => (
            <option key={tab.id} value={tab.id}>
              {tab.label}
            </option>
          ))}
        </select>

        {/* === Tools Grid === */}
        <div className="toolbar-grid" style={{ position: "relative" }}>
          {renderButtons()}

          {/* === Blur overlay if no schema selected === */}
          {!schema && (
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                backdropFilter: "blur(5px)",
                backgroundColor: "rgba(0, 0, 0, 0.35)",
                zIndex: 5,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                textAlign: "center",
                color: "white",
                fontSize: "14px",
                fontWeight: "500",
                borderRadius: "6px",
                pointerEvents: "auto",
              }}
            >
              <div>
                <p>No Mun/City selected yet.</p>
                <p>Please select a Mun/City first.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default Toolbar;
