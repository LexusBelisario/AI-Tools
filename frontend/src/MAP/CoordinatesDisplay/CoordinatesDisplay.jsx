import { useMapEvents } from "react-leaflet";
import { useState } from "react";

function CoordinatesDisplay() {
  const [position, setPosition] = useState(null);
  const [scale, setScale] = useState(null);

  useMapEvents({
    mousemove: (e) => {
      setPosition(e.latlng);
    },
    mouseout: () => {
      setPosition(null);
    },
    zoomend: (e) => {
      const map = e.target;
      const zoom = map.getZoom();
      const center = map.getCenter();

      const metersPerPixel =
        (40075016.686 * Math.abs(Math.cos((center.lat * Math.PI) / 180))) /
        Math.pow(2, zoom + 8);

      const scaleDenominator = Math.round(metersPerPixel * (96 / 0.0254));
      setScale(`1:${scaleDenominator.toLocaleString()}`);
    },
    moveend: (e) => {
      const map = e.target;
      const zoom = map.getZoom();
      const center = map.getCenter();

      const metersPerPixel =
        (40075016.686 * Math.abs(Math.cos((center.lat * Math.PI) / 180))) /
        Math.pow(2, zoom + 8);

      const scaleDenominator = Math.round(metersPerPixel * (96 / 0.0254));
      setScale(`1:${scaleDenominator.toLocaleString()}`);
    },
  });

  if (!position) return null;

  const latDir = position.lat >= 0 ? "" : "S";
  const lngDir = position.lng >= 0 ? "" : "W";

  return (
    <div
      className="absolute bottom-3 right-32 px-3 py-1.5 rounded shadow-lg z-[1000] text-xs font-mono"
      style={{
        backgroundColor: "rgba(21, 25, 34, 0.90)",
        color: "#f5f5f5",
      }}
    >
      <span className="font-normal">Latitude:</span>{" "}
      {Math.abs(position.lat).toFixed(4)} {latDir}
      <span className="font-normal ml-2">Longitude:</span>{" "}
      {Math.abs(position.lng).toFixed(4)} {lngDir}
      {scale && (
        <>
          <span className="font-normal ml-2">|</span>
          <span className="font-normal ml-2">Scale:</span> {scale}
        </>
      )}
    </div>
  );
}

export default CoordinatesDisplay;
