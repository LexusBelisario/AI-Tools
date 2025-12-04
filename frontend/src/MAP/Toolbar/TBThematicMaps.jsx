import React from "react";
import "./toolbar.css";
import Thematic from "../Thematic/Thematic.jsx";

const TBThematicMaps = () => {
  return (
    <>
      {/* === Thematic Map Buttons === */}
      <button
        className="tool-button"
        id="btnLandClass"
        onClick={() => window.toggleLandClassLayer?.()}
        title="Toggle thematic map for Land Classification"
      >
        <img src="/icons/landclassmap.png" alt="Land Class" />
        <span>Land Class</span>
      </button>

      <button
        className="tool-button"
        id="btnLandUse"
        onClick={() => window.toggleLandUseLayer?.()}
        title="Toggle thematic map for Land Use"
      >
        <img src="/icons/landusemap.png" alt="CLUP" />
        <span>CLUP</span>
      </button>

      <button
        className="tool-button"
        id="btnLandCover"
        onClick={() => window.toggleLandCoverLayer?.()}
        title="Toggle thematic map based on Land Cover"
      >
        <img src="/icons/landcovermap.png" alt="Land Cover" />
        <span>Land Cover</span>
      </button>

      <button
        className="tool-button"
        id="btnElevationMap"
        onClick={() => window.toggleElevationLayer?.()}
        title="Toggle thematic map based on Elevation"
      >
        <img src="/icons/elevationmap.png" alt="Elevation" />
        <span>Elevation Map</span>
      </button>

      <button
        className="tool-button"
        id="btnSlopeMap"
        onClick={() => window.toggleSlopeLayer?.()}
        title="Toggle thematic map based on Slope"
      >
        <img src="/icons/slopemap.png" alt="Slope" />
        <span>Slope Map</span>
      </button>

      <button
        className="tool-button"
        id="btnRoadNetwork"
        onClick={() => window.toggleRoadNetwork?.()}
        title="Toggle thematic map based on Road Network"
      >
        <img src="/icons/roadnetworkmap.png" alt="Road Network" />
        <span>Road Network</span>
      </button>

      <button
        className="tool-button"
        id="btnSurfaceWater"
        onClick={() => window.toggleSurfaceWater?.()}
        title="Toggle thematic map based on Surface Water"
      >
        <img src="/icons/surfacewatermap.png" alt="Surface Water" />
        <span>Surface Water</span>
      </button>

      <button
        className="tool-button"
        id="btnSoilType"
        onClick={() => window.toggleSoilType?.()}
        title="Toggle thematic map based on Soil Type"
      >
        <img src="/icons/soilmap.png" alt="Soil Type" />
        <span>Soil Type Map</span>
      </button>

      <button
        className="tool-button"
        id="btnLandslide"
        onClick={() => window.toggleLandslide?.()}
        title="Toggle on and off Landslide Hazard Map"
      >
        <img src="/icons/landslidehazardmap.png" alt="Landslide Hazard" />
        <span>Landslide Hazard</span>
      </button>

      <button
        className="tool-button"
        id="btnLandslideRisk"
        onClick={() => window.toggleLandslideRisk?.()}
        title="Toggle on and off Landslide Risk Map"
      >
        <img src="/icons/landslideriskmap.png" alt="Landslide Risk" />
        <span>Landslide Risk</span>
      </button>

      <button
        className="tool-button"
        id="btnFloodHazard"
        onClick={() => window.toggleFloodHazard?.()}
        title="Toggle on and off Flood Hazard Map"
      >
        <img src="/icons/floodhazardmap.png" alt="Flood Hazard" />
        <span>Flood Hazard</span>
      </button>

      <button
        className="tool-button"
        id="btnFloodRisk"
        onClick={() => window.toggleFloodRisk?.()}
        title="Toggle on and off Flood Risk Map"
      >
        <img src="/icons/floodriskmap.png" alt="Flood Risk" />
        <span>Flood Risk</span>
      </button>

      {/* === Logic Mount === */}
      {/* This ensures that SoilType, Elevation, etc. are mounted in the background,
          so their window.toggleX functions exist when buttons are clicked. */}
      <Thematic map={window.map} />
    </>
  );
};

export default TBThematicMaps;
