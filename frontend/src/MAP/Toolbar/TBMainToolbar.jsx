import React, { useState, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";

import ParcelClickHandler from "../ParcelClickHandler.jsx";
import Search from "../Search/Search.jsx";
import InfoTool from "../InfoTool/InfoTool.jsx";
import Consolidate from "../Consolidate/consolidate.jsx";
import Subdivide from "../Subdivide/subdivide.jsx";
import MatchingReport from "../MatchingReport/MatchingReport.jsx";
import TMCR from "../TMCR/TMCR.jsx";
import TaxmapLayout from "../TaxmapLayout/TaxmapLayout.jsx";

import "./toolbar.css";

const TBMainToolbar = ({ activeTool, setActiveTool }) => {
  const [infoProps, setInfoProps] = useState({});
  const [editHeader, setEditHeader] = useState("Land Parcel Information");

  // central unified state for mutually exclusive tools
  const [openTool, setOpenTool] = useState(null);

  // Unified tool closer
  const closeAll = useCallback(() => {
    setOpenTool(null);
    setActiveTool(null);
  }, [setActiveTool]);

  // Sync external tool state
  useEffect(() => {
    window.currentActiveTool = activeTool;
    return () => {
      window.currentActiveTool = null;
    };
  }, [activeTool]);

  // Expose global helper functions
  useEffect(() => {
    window.switchToEditMode = () => {
      setOpenTool("info");
      setActiveTool("edit");
    };
    window.switchToInfoMode = () => {
      setOpenTool("info");
      setActiveTool("info");
    };
    window.setReactInfoToolData = (parcelData) => {
      setInfoProps(parcelData);
      setOpenTool("info");
      setActiveTool("info");
    };

    return () => {
      delete window.switchToEditMode;
      delete window.switchToInfoMode;
      delete window.setReactInfoToolData;
    };
  }, [setActiveTool]);

  // Mutually exclusive openers FIXED
  const openExclusive = useCallback(
    (toolName) => {
      // 🔥 Prevent reopening same tool (fixes the typing bug)
      if (openTool === toolName) return;

      setOpenTool(toolName);
      setActiveTool(toolName);
    },
    [openTool, setActiveTool]
  );

  const infoOrEditActive = openTool === "info";

  return (
    <>
      {/* === TOOLBAR BUTTONS === */}
      <>
        <button
          className={`tool-button ${openTool === "search" ? "active" : ""}`}
          onClick={() => openExclusive("search")}
          title="Search for parcels or attributes"
        >
          <img src="/icons/property_search_tool_icon.png" alt="Search" />
          <span>Search</span>
        </button>

        <button
          className={`tool-button ${openTool === "info" ? "active" : ""}`}
          onClick={() => openExclusive("info")}
          title="Parcel info"
        >
          <img src="/icons/land_parcel_info_icon.png" alt="Info" />
          <span>Land Parcel Info Tool</span>
        </button>

        <button
          className={`tool-button ${openTool === "taxmap" ? "active" : ""}`}
          onClick={() => openExclusive("taxmap")}
          title="Create map layout (Section, PID, Vicinity)"
        >
          <img src="/icons/taxmap_layout_icon.png" alt="Layout" />
          <span>Taxmap Layout</span>
        </button>

        <button
          className={`tool-button ${openTool === "tmcr" ? "active" : ""}`}
          onClick={() => openExclusive("tmcr")}
          title="Tax Map Control Roll"
        >
          <img src="/icons/tmcr_icon.png" alt="TMCR" />
          <span>TMCR Report</span>
        </button>

        <button
          className={`tool-button ${openTool === "match" ? "active" : ""}`}
          onClick={() => openExclusive("match")}
          title="Matching Report (GIS vs RPT)"
        >
          <img src="/icons/matching_report_icon.png" alt="Match" />
          <span>Matching Report</span>
        </button>

        <button
          className={`tool-button ${openTool === "consolidate" ? "active" : ""}`}
          onClick={() => openExclusive("consolidate")}
          title="Consolidate parcel geometries"
        >
          <img src="/icons/consolidate_icon.png" alt="Consolidate" />
          <span>Consolidate</span>
        </button>

        <button
          className={`tool-button ${openTool === "subdivide" ? "active" : ""}`}
          onClick={() => openExclusive("subdivide")}
          title="Subdivide parcels"
        >
          <img src="/icons/subdivide_icon.png" alt="Subdivide" />
          <span>Subdivide</span>
        </button>
      </>

      {/* === PANELS VIA PORTALS === */}

      {openTool === "search" &&
        createPortal(
          <Search visible={true} onClose={closeAll} />,
          document.body
        )}

      {openTool === "taxmap" &&
        createPortal(
          <TaxmapLayout isVisible={true} onClose={closeAll} />,
          document.body
        )}

      {openTool === "tmcr" &&
        createPortal(
          <TMCR isVisible={true} onClose={closeAll} />,
          document.body
        )}

      {openTool === "match" &&
        createPortal(
          <MatchingReport isVisible={true} onClose={closeAll} />,
          document.body
        )}

      {openTool === "consolidate" &&
        createPortal(
          <Consolidate onClose={closeAll} />,
          document.body
        )}

      {openTool === "subdivide" &&
        createPortal(
          <Subdivide visible={true} onClose={closeAll} />,
          document.body
        )}

      {infoOrEditActive &&
        createPortal(
          <InfoTool
            visible={true}
            onClose={closeAll}
            data={infoProps}
            editable={activeTool === "edit"}
            position={infoProps?.fromSearch ? "right" : "left"}
          />,
          document.body
        )}

      <ParcelClickHandler
        activeTool={openTool}
        setInfoProps={setInfoProps}
        setInfoVisible={() => {}}
        setAttributeEditMode={() => {}}
        setEditHeader={setEditHeader}
      />
    </>
  );
};

export default TBMainToolbar;
