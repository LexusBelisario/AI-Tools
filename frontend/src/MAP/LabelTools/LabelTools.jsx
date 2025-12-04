// src/components/Labels/LabelTools.jsx
import React, { useEffect, useState } from "react";
import StreetLabel from "./StreetLabel.jsx";
import Multilabel from "./Multilabel.jsx";
import LandmarkLabel from "./LandmarkLabel.jsx";
import { useMap } from "react-leaflet";
import { useSchema } from "../SchemaContext.jsx";

const LabelTools = () => {
  const map = useMap();
  const { schema } = useSchema();

  const [showStreetLabel, setShowStreetLabel] = useState(false);
  const [showMultiLabel, setShowMultiLabel] = useState(false);
  const [showLandmarkLabel, setShowLandmarkLabel] = useState(false);

  useEffect(() => {
    console.log("🧭 LabelTools mounted — registering global toggles");

    // 🛣️ Street label toggle
    window.toggleStreetLabel = () => {
      setShowStreetLabel((prev) => {
        const next = !prev;
        document.getElementById("btnStreetLabel")?.classList.toggle("active", next);
        console.log(`🚦 Street labels ${next ? "ON" : "OFF"}`);
        return next;
      });
    };

    // 🏷️ Multi-label toggle
    window.toggleBrgyLabel = () => {
      setShowMultiLabel((prev) => {
        const next = !prev;
        document.getElementById("btnBrgyLabel")?.classList.toggle("active", next);
        console.log(`🚦 Barangay labels ${next ? "ON" : "OFF"}`);
        return next;
      });
    };

    // 📍 Landmark label toggle
    window.toggleLandmarkLabel = () => {
      setShowLandmarkLabel((prev) => {
        const next = !prev;
        document.getElementById("btnLandmarkLabel")?.classList.toggle("active", next);
        console.log(`🚦 Landmark labels ${next ? "ON" : "OFF"}`);
        return next;
      });
    };

    return () => {
      delete window.toggleStreetLabel;
      delete window.toggleBrgyLabel;
      delete window.toggleLandmarkLabel;
      console.log("🧹 LabelTools unmounted — toggles cleared");
    };
  }, []);

  // 🧠 Debug log to verify mount
  useEffect(() => {
    console.log("🧩 LabelTools active", { mapReady: !!map, schema });
  }, [map, schema]);

  return (
    <>
      {showStreetLabel && map && <StreetLabel map={map} />}
      {showMultiLabel && map && <Multilabel map={map} />}
      {showLandmarkLabel && map && <LandmarkLabel map={map} />}
    </>
  );
};

export default LabelTools;
