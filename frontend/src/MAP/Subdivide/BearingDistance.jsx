import React, {
  useRef,
  useState,
  useEffect,
  forwardRef,
  useImperativeHandle,
} from "react";
import L from "leaflet";

const BearingDistance = forwardRef(({
  map,
  setLines,
  manualPoints,
  setManualPoints,
  renderParcelControls,
  handleSubdivide,
  selectedParcel,
  locked,
  setSuggestedPins
}, ref) => {
  const [bearingStart, setBearingStart] = useState(null);
  const [bearingDistance, setBearingDistance] = useState("");
  const [bearingAngle, setBearingAngle] = useState("");
  const [bearingCaptureMode, setBearingCaptureMode] = useState(false);
  const [error, setError] = useState("");

  const bearingMarker = useRef(null);
  const allLayers = useRef([]);

  useImperativeHandle(ref, () => ({
    clearBearingPoints: () => {
      setManualPoints([]);
      setLines([]);

      allLayers.current.forEach(layer => {
        if (map.hasLayer(layer)) map.removeLayer(layer);
      });
      allLayers.current = [];

      if (bearingMarker.current && map.hasLayer(bearingMarker.current)) {
        map.removeLayer(bearingMarker.current);
        bearingMarker.current = null;
      }

      setBearingStart(null);
      setBearingDistance("");
      setBearingAngle("");
      setError("");
    }
  }));

  useEffect(() => {
    if (!map || !bearingCaptureMode || !locked) return;

    const onMapClick = (e) => {
      const latlng = [e.latlng.lat, e.latlng.lng];
      setBearingStart(latlng);
      setBearingCaptureMode(false);

      if (bearingMarker.current && map.hasLayer(bearingMarker.current)) {
        map.removeLayer(bearingMarker.current);
      }

      bearingMarker.current = L.circleMarker(latlng, {
        radius: 6,
        color: "orange",
        fillColor: "orange",
        fillOpacity: 1
      }).addTo(map);
    };

    map.on("click", onMapClick);
    return () => map.off("click", onMapClick);
  }, [map, bearingCaptureMode, locked]);

  const computeDestination = ([lat, lng], distance, bearing) => {
    const R = 6378137;
    const brng = (bearing * Math.PI) / 180;
    const φ1 = (lat * Math.PI) / 180;
    const λ1 = (lng * Math.PI) / 180;
    const δ = distance / R;

    const φ2 = Math.asin(
      Math.sin(φ1) * Math.cos(δ) +
      Math.cos(φ1) * Math.sin(δ) * Math.cos(brng)
    );

    const λ2 = λ1 + Math.atan2(
      Math.sin(brng) * Math.sin(δ) * Math.cos(φ1),
      Math.cos(δ) - Math.sin(φ1) * Math.sin(φ2)
    );

    return [φ2 * (180 / Math.PI), λ2 * (180 / Math.PI)];
  };

  const handleAddSegment = () => {
    const dist = parseFloat(bearingDistance);
    const angle = parseFloat(bearingAngle);

    if (!bearingStart || isNaN(dist) || isNaN(angle)) {
      setError("Invalid input.");
      return;
    }

    const nextPoint = computeDestination(bearingStart, dist, angle);
    const newSegment = [
      [bearingStart[1], bearingStart[0]],
      [nextPoint[1], nextPoint[0]]
    ];

    const updated = [...manualPoints, ...newSegment];
    setManualPoints(updated);
    setBearingStart(nextPoint);

    const marker = L.circleMarker(nextPoint, {
      radius: 5,
      color: "red",
      fillColor: "red",
      fillOpacity: 0.9
    }).addTo(map);
    allLayers.current.push(marker);

    allLayers.current = allLayers.current.filter(layer => {
      if (layer instanceof L.Polyline) {
        if (map.hasLayer(layer)) map.removeLayer(layer);
        return false;
      }
      return true;
    });

    const polyline = L.polyline(updated.map(([lng, lat]) => [lat, lng]), {
      color: "blue",
      weight: 2,
      dashArray: "4,6"
    }).addTo(map);
    allLayers.current.push(polyline);

    setLines(updated);
    setError("");

    if (selectedParcel?.properties?.pin) {
  const base = selectedParcel.properties.pin.slice(0, -3);
  const suffix = parseInt(selectedParcel.properties.pin.slice(-3), 10) || 0;
  setSuggestedPins([
    base + String(suffix + 1).padStart(3, "0"),
    base + String(suffix + 2).padStart(3, "0")
  ]);
}
  };
  

  const handleClearStart = () => {
    if (bearingMarker.current && map.hasLayer(bearingMarker.current)) {
      map.removeLayer(bearingMarker.current);
      bearingMarker.current = null;
    }
    setBearingStart(null);
    setBearingDistance("");
    setBearingAngle("");
  };

  const handleClearPoints = () => {
    setManualPoints([]);
    setLines([]);

    allLayers.current.forEach(layer => {
      if (map.hasLayer(layer)) {
        map.removeLayer(layer);
      }
    });
    allLayers.current = [];

    if (bearingMarker.current && map.hasLayer(bearingMarker.current)) {
      map.removeLayer(bearingMarker.current);
      bearingMarker.current = null;
    }

    setBearingStart(null);
    setBearingDistance("");
    setBearingAngle("");
    setError("");
  };

  const removeManualPoint = (index) => {
    const updatedPoints = [...manualPoints];
    updatedPoints.splice(index, 1);
    setManualPoints(updatedPoints);

    allLayers.current.forEach(layer => {
      if (map.hasLayer(layer)) map.removeLayer(layer);
    });
    allLayers.current = [];

    if (updatedPoints.length >= 2) {
      const poly = L.polyline(updatedPoints.map(([lng, lat]) => [lat, lng]), {
        color: "blue", weight: 2, dashArray: "4, 6"
      }).addTo(map);
      allLayers.current.push(poly);
      setLines([updatedPoints]);
    } else {
      setLines([]);
    }
  };

  return (
    <>
      <div className="subdivide-instructions">
        <p><strong>1.</strong> Click a parcel on the map to select it.</p>
        <p><strong>2.</strong> Click <strong>Choose This Parcel</strong> to choose the parcel for subdividing.</p>
        <p><strong>3.</strong> Click <strong>Select Start Point</strong> and choose a location on the parcel.</p>
        <p><strong>4.</strong> Enter distance and bearing, then click <strong>Add Segment</strong> to generate a point. You can add multiple points.</p>
        <p><strong>5.</strong> Click <strong>Subdivide and Save</strong> to finish and save.</p>
      </div>

      {error && <div style={{ color: "red", fontSize: "13px" }}>{error}</div>}

      {renderParcelControls()}

      <div className="subdivide-controls-grid">
        <button
          className="subdivide-btn"
          onClick={() => setBearingCaptureMode(true)}
          disabled={!selectedParcel || !locked}
        >
          Select Start Point
        </button>
        <button
          className="subdivide-btn"
          onClick={handleClearStart}
          disabled={!bearingStart}
        >
          Clear Start Point
        </button>
      </div>

      <div className="subdivide-controls-grid">
        <input
          type="number"
          placeholder="Distance (m)"
          value={bearingDistance}
          onChange={(e) => setBearingDistance(e.target.value)}
          className="subdivide-input"
          disabled={!bearingStart}
        />
        <input
          type="number"
          placeholder="Bearing (°)"
          value={bearingAngle}
          onChange={(e) => setBearingAngle(e.target.value)}
          className="subdivide-input"
          disabled={!bearingStart}
        />
      </div>

      <div className="subdivide-controls-grid">
        <button
          className="subdivide-btn"
          onClick={handleAddSegment}
          disabled={!bearingStart || !bearingDistance || !bearingAngle}
        >
          Add Segment
        </button>
        <button
          className="subdivide-btn"
          onClick={handleClearPoints}
          disabled={manualPoints.length === 0}
        >
          Clear All Points
        </button>
      </div>

      <ul className="subdivide-pin-list">
        {manualPoints.map(([lng, lat], i) => (
          <li className="manual-point-item" key={i}>
            {`Lat: ${lat.toFixed(6)}, Lng: ${lng.toFixed(6)}`}
            <button className="remove-point-btn" onClick={() => removeManualPoint(i)}>
              ×
            </button>
          </li>
        ))}
      </ul>

      <button
        className="subdivide-btn"
        style={{ marginTop: "12px" }}
        onClick={handleSubdivide}
        disabled={!selectedParcel || manualPoints.length < 2}
      >
        Subdivide and Save
      </button>
    </>
  );
});

export default BearingDistance;
