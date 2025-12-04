// src/components/Thematic/Thematic.jsx
import React, { useEffect } from "react";
import { useSchema } from "../SchemaContext";
import CLUP from "./CLUP.jsx";
import LandClass from "./LandClass.jsx";
import ElevationMap from "./ElevationMap.jsx";
import SlopeMap from "./SlopeMap";
import LandCover from "./LandCover";
import RoadNetwork from "./RoadNetwork.jsx";
import SurfaceWater from "./SurfaceWater.jsx";
import SoilType from "./SoilType.jsx";
import LandslideHazard from "./LandslideHazard.jsx";
import LandslideRisk from "./LandslideRisk.jsx";
import FloodHazard from "./FloodHazard.jsx";
import FloodRisk from "./FloodRisk.jsx";
import ThematicLegend from "./ThematicLegend.jsx";
import "./thematic.css";

const Thematic = ({ map }) => {
  const { schema } = useSchema();

  // Debug: watch schema changes
  useEffect(() => {
    console.log("🌐 Thematic: current schema =", schema);
  }, [schema]);

  if (!schema) {
    console.log("⚠️ Thematic: schema not set, skipping layer mounts");
    return null;
  }

  return (
    <>
      {/* Each child thematic component registers its toggle in useEffect */}
      <ElevationMap map={map} schema={schema} />
      <LandClass map={map} schema={schema} />
      <SlopeMap map={map} schema={schema} />
      <LandCover map={map} schema={schema} />
      <RoadNetwork map={map} schema={schema} />
      <SoilType map={map} />
      <SurfaceWater map={map} schema={schema} />
      <FloodHazard map={map} schema={schema} />
      <LandslideHazard map={map} schema={schema} />
      <LandslideRisk map={map} schema={schema} />
      <FloodRisk map={map} schema={schema} />
      <CLUP /> 

      {/* Unified legend container */}
      <ThematicLegend />
    </>
  );
};

export default Thematic;
