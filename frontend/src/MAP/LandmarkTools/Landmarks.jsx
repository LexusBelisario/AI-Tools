// src/components/Landmarks.jsx
import React, { useState, useMemo, useCallback, useEffect } from "react";
import API from "../../api.js";
import ShowLandmarks from "./ShowLandmarks.jsx";
import RemoveLandmark from "./RemoveLandmark.jsx";
import LandmarkInfotool from "./LandmarkInfotool.jsx";
import AddLandmark from "./AddLandmarks.jsx";
import { useSchema } from "../SchemaContext";

const Landmarks = ({ activeTool, setActiveTool, setLandmarksVisible }) => {
  const { schema } = useSchema();

  const showLandmarksVisible =
    activeTool === "showLandmarks" ||
    activeTool === "removeLandmark" ||
    activeTool === "landmarkInfo" ||
    activeTool === "updateLandmark" ||
    activeTool === "addLandmark";

  const [landmarksOnScreen, setLandmarksOnScreen] = useState(false);
  const [removalList, setRemovalList] = useState([]);
  const [selectedLandmark, setSelectedLandmark] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

  // Keep external state in sync
  useEffect(() => {
    setLandmarksVisible?.(landmarksOnScreen);
  }, [landmarksOnScreen, setLandmarksVisible]);

  // === REMOVE LANDMARK TOOL ===
  const handleMark = useCallback((feature) => {
    const { id, name, type, barangay } = feature.properties || {};
    if (!id) {
      console.warn("⚠️ Missing landmark ID, cannot mark for removal.");
      return;
    }

    setRemovalList((prev) => {
      if (prev.some((item) => item.id === id)) return prev;
      return [...prev, { id, name, type, barangay }];
    });
  }, []);

  const handleUnmark = (id) => {
    setRemovalList((prev) => prev.filter((item) => item.id !== id));
  };

  const handleCancel = () => {
    setRemovalList([]);
    if (activeTool === "removeLandmark") {
      setActiveTool(null);
    }
  };

  const handleRemoveSelected = async () => {
    const payload = {
      schema,
      ids: removalList.map((l) => l.id),
    };

    if (!payload.ids.length) return;

    try {
      const token =
        localStorage.getItem("access_token") ||
        localStorage.getItem("accessToken");
      const res = await fetch(`${API}/landmarks/remove`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token && { Authorization: `Bearer ${token}` }),
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        if (res.status === 401 || res.status === 403) {
          console.error("❌ Authentication error");
          localStorage.removeItem("access_token");
          localStorage.removeItem("accessToken");
          window.location.href = "/login";
          return;
        }
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      console.log("✅ Deletion result:", data);

      setRemovalList([]);
      setRefreshKey((k) => k + 1); // 🔄 refresh landmarks
    } catch (err) {
      console.error("❌ Failed to remove landmarks:", err);
    }
  };

  // === INFO / UPDATE TOOL ===
  const onClickFeature = useMemo(() => {
    if (activeTool === "removeLandmark") return handleMark;

    // ✅ Allow both Info and Update modes to open popup
    if (activeTool === "landmarkInfo" || activeTool === "updateLandmark") {
      return (feature) => setSelectedLandmark(feature);
    }

    return null;
  }, [activeTool, handleMark]);

  // ✅ Trigger refresh after update
  const handleLandmarkUpdated = () => {
    setRefreshKey((k) => k + 1);
  };

  return (
    <>
      <ShowLandmarks
        schema={schema}
        visible={showLandmarksVisible}
        onClickFeature={onClickFeature}
        onVisibleChange={setLandmarksOnScreen}
        refreshKey={refreshKey}
      />

      {/* === Add Landmark Panel === */}
      <AddLandmark
        visible={activeTool === "addLandmark"}
        onClose={() => setActiveTool(null)}
        onAdded={() => setRefreshKey((k) => k + 1)}
      />

      <RemoveLandmark
        visible={activeTool === "removeLandmark"}
        removalList={removalList}
        onUnmark={handleUnmark}
        onCancel={handleCancel}
        onRemoveSelected={handleRemoveSelected}
      />

      {/* === Info / Update Popup === */}
      <LandmarkInfotool
        key={activeTool}
        visible={
          activeTool === "landmarkInfo" || activeTool === "updateLandmark"
        }
        data={selectedLandmark}
        schema={schema}
        startEditable={activeTool === "updateLandmark"}
        onClose={() => setActiveTool(null)}
        onUpdated={handleLandmarkUpdated} // ✅ refresh map after save
      />
    </>
  );
};

export default Landmarks;
