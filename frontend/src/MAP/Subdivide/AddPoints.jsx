import React, {
  useRef,
  useState,
  forwardRef,
  useImperativeHandle,
} from "react";
import L from "leaflet";

const AddPoints = forwardRef(({
  map,
  setLines,
  manualPoints,
  setManualPoints,
  locked,
  selectedParcel,
  renderParcelControls,
  handleSubdivide,
  handleCancel,
  suggestedPins,
  setSuggestedPins
}, ref) => {
  const [manualLat, setManualLat] = useState("");
  const [error, setError] = useState("");
  const [showExportOptions, setShowExportOptions] = useState(false);
  const manualMarkers = useRef([]);
  const drawnLayers = useRef([]);

  useImperativeHandle(ref, () => ({
    clearManualPoints: () => {
      manualMarkers.current.forEach(m => m.remove());
      drawnLayers.current.forEach(l => l.remove());
      manualMarkers.current = [];
      drawnLayers.current = [];
      setManualPoints([]);
      setLines([]);
    }
  }));

  const addManualPoint = () => {
    const lines = manualLat.split("\n").map(line => line.trim()).filter(Boolean);
    const newPoints = [];
    const markers = [];

    for (let line of lines) {
      const parts = line.split(",").map(p => p.trim());
      if (parts.length !== 2 || isNaN(parts[0]) || isNaN(parts[1])) {
        setError(`Invalid format: "${line}". Use: lat,lng`);
        return;
      }

      const lat = parseFloat(parts[0]);
      const lng = parseFloat(parts[1]);
      if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
        setError(`Coordinates out of range: "${line}"`);
        return;
      }

      newPoints.push([lng, lat]);

      const marker = L.circleMarker([lat, lng], {
        radius: 5,
        color: "red",
        fillColor: "red",
        fillOpacity: 0.9
      }).addTo(map);
      markers.push(marker);
    }

    const updated = [...manualPoints, ...newPoints];
    setManualPoints(updated);
    setManualLat("");
    setError("");
    manualMarkers.current.push(...markers);

    if (updated.length >= 2) {
      drawnLayers.current.forEach(l => l.remove());
      drawnLayers.current = [];
      const poly = L.polyline(updated.map(([lng, lat]) => [lat, lng]), {
        color: "blue", weight: 2, dashArray: "4, 6"
      }).addTo(map);
      drawnLayers.current.push(poly);
      setLines([updated]);
    }
  };

  const removeManualPoint = (index) => {
    const updatedPoints = [...manualPoints];
    updatedPoints.splice(index, 1);
    setManualPoints(updatedPoints);

    const marker = manualMarkers.current[index];
    if (marker) marker.remove();
    manualMarkers.current.splice(index, 1);

    drawnLayers.current.forEach(l => l.remove());
    drawnLayers.current = [];

    if (updatedPoints.length >= 2) {
      const poly = L.polyline(updatedPoints.map(([lng, lat]) => [lat, lng]), {
        color: "blue", weight: 2, dashArray: "4, 6"
      }).addTo(map);
      drawnLayers.current.push(poly);
      setLines([updatedPoints]);
    } else {
      setLines([]);
    }
  };

  const handleClearPoints = () => {
    setManualPoints([]);
    setManualLat("");
    setError("");
    manualMarkers.current.forEach(m => m.remove());
    manualMarkers.current = [];
    drawnLayers.current.forEach(l => l.remove());
    drawnLayers.current = [];
    setLines([]);
  };

  const handleImportFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (event) {
      let content = event.target.result;
      let lines = content.split("\n").map(l => l.trim()).filter(Boolean);

      if (lines[0].toLowerCase().includes("latitude")) lines = lines.slice(1);

      const newPoints = [];
      const newMarkers = [];

      for (let line of lines) {
        const parts = line.split(",").map(p => p.trim());
        if (parts.length !== 2 || isNaN(parts[0]) || isNaN(parts[1])) {
          setError(`Invalid line: "${line}". Use lat,lng format.`);
          return;
        }

        const lat = parseFloat(parts[0]);
        const lng = parseFloat(parts[1]);

        if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
          setError(`Coordinates out of range: "${line}"`);
          return;
        }

        newPoints.push([lng, lat]);

        const marker = L.circleMarker([lat, lng], {
          radius: 5,
          color: "red",
          fillColor: "red",
          fillOpacity: 0.9
        }).addTo(map);
        newMarkers.push(marker);
      }

      const updated = [...manualPoints, ...newPoints];
      setManualPoints(updated);
      manualMarkers.current.push(...newMarkers);
      setError("");

      if (updated.length >= 2) {
        drawnLayers.current.forEach(l => l.remove());
        drawnLayers.current = [];
        const poly = L.polyline(updated.map(([lng, lat]) => [lat, lng]), {
          color: "blue", weight: 2, dashArray: "4, 6"
        }).addTo(map);
        drawnLayers.current.push(poly);
        setLines([updated]);
      }
    };

    reader.readAsText(file);
  };

  const handleExportPoints = (format = "txt") => {
    if (manualPoints.length === 0) return;

    let content = "";
    let filename = "";

    if (format === "csv") {
      content = "Latitude,Longitude\n" +
        manualPoints.map(([lng, lat]) => `${lat.toFixed(6)},${lng.toFixed(6)}`).join("\n");
      filename = "subdivide_points.csv";
    } else {
      content = manualPoints.map(([lng, lat]) => `${lat.toFixed(6)},${lng.toFixed(6)}`).join("\n");
      filename = "subdivide_points.txt";
    }

    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setShowExportOptions(false);
  };

  return (
    <>
      <div className="subdivide-instructions">
        <p><strong>1.</strong> Click a parcel on the map to select it.</p>
        <p><strong>2.</strong> Click <strong>Choose This Parcel</strong> to choose the parcel for subdividing.</p>
        <p><strong>3.</strong> Paste coordinates below and press <strong>Add</strong>.</p>
        <p><strong>4.</strong> You may also import from file or remove any points manually.</p>
      </div>

      {error && <div style={{ color: "red", fontSize: "13px" }}>{error}</div>}
      {renderParcelControls()}

      <textarea
        value={manualLat}
        onChange={(e) => setManualLat(e.target.value)}
        placeholder="Paste coordinates (lat,lng)"
        rows={4}
        style={{ width: "100%", marginTop: "10px" }}
      />

      <div className="subdivide-controls-grid">
        <button className="subdivide-btn" onClick={addManualPoint} disabled={!selectedParcel || !locked}>
          Add
        </button>
        <button className="subdivide-btn" onClick={handleClearPoints} disabled={manualPoints.length === 0}>
          Clear All
        </button>
        <label className="subdivide-btn" style={{ textAlign: "center", cursor: "pointer" }}>
          Import File
          <input type="file" accept=".txt,.csv" style={{ display: "none" }} onChange={handleImportFile} />
        </label>
        <button className="subdivide-btn" onClick={handleCancel}>
          Cancel and Discard
        </button>
      </div>

      {manualPoints.length > 0 && (
        <>
          <div style={{ marginTop: "10px" }}>
            <button className="subdivide-btn" onClick={() => handleExportPoints("txt")} style={{ marginBottom: "8px" }}>
              Export Points (.txt)
            </button>
            <button className="subdivide-btn" onClick={() => handleExportPoints("csv")}>
              Export Points (.csv)
            </button>
          </div>

          <ul className="subdivide-pin-list" style={{ marginTop: "12px" }}>
            {manualPoints.map(([lng, lat], idx) => (
              <li key={idx} className="manual-point-item">
                {`Lat: ${lat.toFixed(6)}, Lng: ${lng.toFixed(6)}`}
                <button
                  onClick={() => removeManualPoint(idx)}
                  style={{ marginLeft: "12px", padding: "2px 6px" }}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </>
      )}

      {suggestedPins.length > 0 && (
        <div style={{ marginTop: "16px", padding: "0 12px" }}>
          <h4>Assign PINs to Subdivided Parcels</h4>
          {suggestedPins.map((pin, idx) => (
            <div key={idx} style={{ marginBottom: "8px" }}>
              <label style={{ fontWeight: "bold" }}>PIN #{idx + 1}:</label>
              <input
                type="text"
                value={pin}
                onChange={(e) => {
                  const updated = [...suggestedPins];
                  updated[idx] = e.target.value;
                  setSuggestedPins(updated);
                }}
                style={{ marginLeft: "8px", padding: "4px", width: "75%" }}
              />
            </div>
          ))}
          <button
            className="subdivide-btn"
            onClick={handleSubdivide}
            disabled={suggestedPins.some(pin => !pin)}
            style={{ marginTop: "12px" }}
          >
            Subdivide and Save
          </button>
        </div>
      )}
    </>
  );
});

export default AddPoints;
