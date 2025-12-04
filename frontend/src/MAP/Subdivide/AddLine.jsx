import React, { useEffect, useState, useRef } from "react";
import L from "leaflet";
import API from "../../api.js";
import { useSchema } from "../SchemaContext";
import SubdividePinEditor from "./SubdividePinEditor";

const AddLine = ({ map }) => {
  const { schema } = useSchema();
  const [selectedParcel, setSelectedParcel] = useState(null);
  const [locked, setLocked] = useState(false);
  const lockedRef = useRef(false);
  const [hasValidLine, setHasValidLine] = useState(false);
  const [previewData, setPreviewData] = useState(null);
  const [showInstructions, setShowInstructions] = useState(true);

  const activePoints = useRef([]);
  const currentLine = useRef(null);
  const pointMarkers = useRef([]);
  const allLines = useRef([]);
  const clickHandlers = useRef({});

  useEffect(() => {
    lockedRef.current = locked;
  }, [locked]);

  useEffect(() => {
    window.setSelectedParcelForSubdivide = (parcel) => {
      if (!lockedRef.current) setSelectedParcel(parcel);
    };
  }, []);

  const handleChooseParcel = () => {
    if (!selectedParcel) return;
    setLocked(true);
    lockedRef.current = true;
    window.subdivideLocked = true;
    enableDrawing();
  };

  const enableDrawing = () => {
    if (!map) return;

    if (clickHandlers.current.left)
      map.off("click", clickHandlers.current.left);
    if (clickHandlers.current.right)
      map.off("contextmenu", clickHandlers.current.right);

    activePoints.current = [];
    pointMarkers.current.forEach((m) => map.removeLayer(m));
    pointMarkers.current = [];

    if (currentLine.current) {
      map.removeLayer(currentLine.current);
      currentLine.current = null;
    }

    currentLine.current = L.polyline([], {
      color: "blue",
      weight: 2,
      dashArray: "5, 5",
    }).addTo(map);

    const handleLeftClick = (e) => {
      if (!lockedRef.current) return;
      const { lat, lng } = e.latlng;
      activePoints.current.push([lng, lat]);
      const marker = L.circleMarker([lat, lng], {
        radius: 4,
        color: "red",
        fillColor: "red",
        fillOpacity: 1,
      }).addTo(map);
      pointMarkers.current.push(marker);
      const latlngs = activePoints.current.map(([lng, lat]) => [lat, lng]);
      currentLine.current.setLatLngs(latlngs);
    };

    const handleRightClick = (e) => {
      if (!lockedRef.current) return;
      e.originalEvent.preventDefault();
      if (activePoints.current.length > 1) {
        allLines.current.push({
          points: [...activePoints.current],
          layer: currentLine.current,
          markers: [...pointMarkers.current],
        });
        setHasValidLine(true);
      } else {
        map.removeLayer(currentLine.current);
      }

      activePoints.current = [];
      pointMarkers.current = [];
      currentLine.current = L.polyline([], {
        color: "blue",
        weight: 2,
        dashArray: "5, 5",
      }).addTo(map);
    };

    map.on("click", handleLeftClick);
    map.on("contextmenu", handleRightClick);
    clickHandlers.current.left = handleLeftClick;
    clickHandlers.current.right = handleRightClick;
  };

  const handleUndo = () => {
    const last = allLines.current.pop();
    if (last) {
      map.removeLayer(last.layer);
      last.markers?.forEach((m) => map.removeLayer(m));
      setHasValidLine(allLines.current.length > 0);
    }
  };

  const clearDrawings = () => {
    allLines.current.forEach((l) => {
      map.removeLayer(l.layer);
      l.markers?.forEach((m) => map.removeLayer(m));
    });
    pointMarkers.current.forEach((m) => map.removeLayer(m));
    if (currentLine.current) map.removeLayer(currentLine.current);
    allLines.current = [];
    activePoints.current = [];
    pointMarkers.current = [];
    currentLine.current = null;
    setHasValidLine(false);
  };

  const handleCancel = () => {
    clearDrawings();
    setPreviewData(null);
    setLocked(false);
    lockedRef.current = false;
    window.subdivideLocked = false;
    setSelectedParcel(null);
  };

  const handlePreview = async () => {
    if (!lockedRef.current || !selectedParcel || !schema) return;
    if (allLines.current.length === 0) return;

    const splitLines = allLines.current.map((l) => l.points);
    const payload = {
      schema,
      table: selectedParcel.source_table || "LandParcels",
      pin: selectedParcel.pin,
      split_lines: splitLines,
    };

    try {
      const res = await fetch(`${API}/subdivide-preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (data.status === "success" && data.parts?.length) {
        clearDrawings();
        setPreviewData({
          schema,
          table: selectedParcel.source_table || "LandParcels",
          basePin: selectedParcel.pin,
          splitLines,
          parts: data.parts,
          suggestedPins: data.suggested_pins,
        });
      }
    } catch (err) {
      console.error("Preview error:", err);
    }
  };

  // === Highlight all new parcels after Subdivide(Save)
  const highlightNewParcels = (pins, schema, table) => {
    if (!window.parcelLayers?.length) {
      setTimeout(() => highlightNewParcels(pins, schema, table), 600);
      return;
    }

    // Reset all styles
    window.parcelLayers.forEach(({ layer }) => {
      layer.setStyle?.({
        color: "black",
        weight: 1,
        fillColor: "white",
        fillOpacity: 0.1,
      });
    });

    // Highlight new parcels
    pins.forEach((pin) => {
      const match = window.parcelLayers.find(
        (p) =>
          p.feature?.properties?.pin === pin &&
          p.feature?.properties?.source_schema === schema &&
          p.feature?.properties?.source_table === table
      );
      if (match?.layer) {
        match.layer.setStyle?.({
          color: "black",
          weight: 2,
          fillColor: "yellow",
          fillOpacity: 0.4,
        });
      }
    });

    // Zoom to include all highlighted parcels
    const layersToFit = window.parcelLayers.filter((p) =>
      pins.includes(p.feature?.properties?.pin)
    );
    if (layersToFit.length > 0) {
      const group = L.featureGroup(layersToFit.map((l) => l.layer));
      map.fitBounds(group.getBounds(), { maxZoom: 18 });
    }
  };

  useEffect(() => {
    return () => {
      if (map) {
        if (clickHandlers.current.left)
          map.off("click", clickHandlers.current.left);
        if (clickHandlers.current.right)
          map.off("contextmenu", clickHandlers.current.right);
      }
      clearDrawings();
    };
  }, [map]);

  return (
    <>
      {/* === Collapsible Instructions === */}
      <div className="collapsible-header">
        <button
          className="toggle-instructions-btn"
          onClick={() => setShowInstructions(!showInstructions)}
        >
          {showInstructions ? "Hide Instructions ▲" : "Show Instructions ▼"}
        </button>
      </div>

      {showInstructions && (
        <div className="subdivide-instructions">
          <p><strong>1.</strong> Click a parcel to select it.</p>
          <p><strong>2.</strong> Click <strong>Choose This Parcel</strong> to lock it for subdividing.</p>
          <p><strong>3.</strong> Left-click adds points. Right-click finishes a line.</p>
          <p><strong>4.</strong> Click <strong>Subdivide (Preview)</strong> to see result before saving.</p>
          <p><strong>5.</strong> Assign <strong>PINs</strong> for new parcels.</p>
          <p><strong>6.</strong> Click <strong>Subdivide (Save)</strong> to save and highlight new parcels.</p>
        </div>
      )}

      {/* === Drawing Mode === */}
      {!previewData && (
        <div className="parcel-selection-controls">
          {!selectedParcel && (
            <p style={{ fontSize: "13px", color: "#aaa" }}>
              Click a parcel on the map to enable selection.
            </p>
          )}

          {selectedParcel && !locked && (
            <button className="subdivide-btn" onClick={handleChooseParcel}>
              Choose This Parcel
            </button>
          )}

          {locked && (
            <>
              <div style={{ marginBottom: "6px", fontSize: "13px" }}>
                Locked Parcel: <strong>{selectedParcel.pin}</strong>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                <button
                  className="subdivide-btn"
                  onClick={handlePreview}
                  disabled={!hasValidLine}
                >
                  Subdivide (Preview)
                </button>

                <button
                  className="subdivide-btn"
                  style={{ backgroundColor: "#777" }}
                  onClick={handleUndo}
                >
                  Undo Last Line
                </button>

                <button
                  className="subdivide-btn"
                  style={{ backgroundColor: "#b33", color: "white" }}
                  onClick={handleCancel}
                >
                  Cancel
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* === Preview Mode === */}
      {previewData && (
        <SubdividePinEditor
          map={map}
          {...previewData}
          onCancel={handleCancel}
          onDone={(createdPins) => {
            setPreviewData(null);
            setLocked(false);
            window.subdivideLocked = false;

            // ✅ Highlight newly created parcels after Subdivide(Save)
            if (Array.isArray(createdPins) && createdPins.length > 0) {
              highlightNewParcels(
                createdPins,
                previewData.schema,
                previewData.table
              );
            }
          }}
        />
      )}
    </>
  );
};

export default AddLine;
