// src/components/TBLandmarkTools.jsx
import React, { useState } from "react";
import "./toolbar.css";

import Landmarks from "../LandmarkTools/Landmarks.jsx"; // ✅ Landmarks manager (now uses context)

const TBLandmarkTools = () => {
  const [activeTool, setActiveTool] = useState(null);
  const [landmarksVisible, setLandmarksVisible] = useState(false);

  return (
    <>
      {/* === Toolbar Buttons === */}
      <button
        className={`tool-button ${activeTool === "showLandmarks" ? "active" : ""}`}
        id="btnShowLandmark"
        onClick={() => {
          setActiveTool(activeTool === "showLandmarks" ? null : "showLandmarks");
        }}
        title="Click to show all landmarks."
      >
        <img src="/icons/show_landmark_icon.png" alt="Show Landmarks" />
        <span>Show Landmarks</span>
      </button>

      <button
        className={`tool-button ${activeTool === "addLandmark" ? "active" : ""}`}
        id="btnAddLandmark"
        onClick={() =>
          setActiveTool(activeTool === "addLandmark" ? null : "addLandmark")
        }
        title="Click on any part of the map to add landmark"
      >
        <img src="/icons/add_landmark_icon.png" alt="Add Landmark" />
        <span>Add Landmark</span>
      </button>

      <button
        className={`tool-button ${activeTool === "landmarkInfo" ? "active" : ""}`}
        id="btnLandmarkInfo"
        onClick={() => {
          if (activeTool !== "landmarkInfo") setActiveTool("landmarkInfo");
          else setActiveTool(null);
        }}
        title="Click on any landmark and see its information"
      >
        <img src="/icons/info_landmark_icon.png" alt="Landmark Info" />
        <span>Landmark Info Tool</span>
      </button>

      <button
        className={`tool-button ${activeTool === "updateLandmark" ? "active" : ""}`}
        id="btnUpdateLandmark"
        onClick={() =>
          setActiveTool(activeTool === "updateLandmark" ? null : "updateLandmark")
        }
        title="Updates the attributes' values of a Landmark"
      >
        <img src="/icons/update_landmark_icon.png" alt="Update Landmark" />
        <span>Update Landmark</span>
      </button>

      <button
        className={`tool-button ${activeTool === "removeLandmark" ? "active" : ""}`}
        id="btnRemoveLandmark"
        onClick={() =>
          setActiveTool(activeTool === "removeLandmark" ? null : "removeLandmark")
        }
        title="Mark landmarks for deletion by clicking them"
      >
        <img src="/icons/remove_landmark_icon.png" alt="Remove Landmark" />
        <span>Remove Landmark</span>
      </button>

      {/* === Landmarks Manager === */}
      <Landmarks
        activeTool={activeTool}
        setActiveTool={setActiveTool}
        setLandmarksVisible={setLandmarksVisible}
      />
    </>
  );
};

export default TBLandmarkTools;
