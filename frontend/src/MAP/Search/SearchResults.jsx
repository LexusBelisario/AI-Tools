// SearchResults.jsx
import API from "../../api.js";
import React, { useEffect } from "react";
import L from "leaflet";

const SearchResults = ({
  pins = [],
  noInput = false,
  selectedPin,
  setSelectedPin,
  onShowParcelInfo,
}) => {
  // ============================================================
  // 🔹 Maintain highlights on zoom / move
  // ============================================================
  useEffect(() => {
    const map = window.map;
    if (!map) return;

    const handleMapChange = () => {
      if (!window.parcelLayers?.length) return;

      window.parcelLayers.forEach(({ feature, layer }) => {
        const pin = feature.properties.pin;
        if (pin === selectedPin) {
          // selected one (yellow)
          layer.setStyle({
            color: "yellow",
            weight: 3,
            fillColor: "yellow",
            fillOpacity: 0.3,
          });
          layer.bringToFront();
        } else if (pins.includes(pin)) {
          // matched search results (lime)
          layer.setStyle({
            color: "black",
            weight: 2,
            fillColor: "lime",
            fillOpacity: 0.2,
          });
        } else {
          // reset non-matching parcels
          layer.setStyle({
            color: "black",
            weight: 1,
            fillColor: "white",
            fillOpacity: 0.1,
          });
        }
      });
    };

    map.on("moveend", handleMapChange);
    return () => {
      map.off("moveend", handleMapChange);
    };
  }, [pins, selectedPin]);

  // ============================================================
  // 🔹 Click result to zoom and highlight
  // ============================================================
  const handleResultClick = async (pin) => {
    const match = window.parcelLayers?.find(
      ({ feature }) => feature.properties.pin === pin
    );
    if (!match) {
      console.warn("⚠️ No parcel layer found for PIN:", pin);
      return;
    }

    // Highlight selected parcel
    setSelectedPin(pin);
    window.parcelLayers?.forEach(({ feature, layer }) => {
      const parcelPin = feature.properties.pin;
      if (parcelPin === pin) {
        layer.setStyle({
          color: "yellow",
          weight: 3,
          fillColor: "yellow",
          fillOpacity: 0.3,
        });
        layer.bringToFront();
      } else if (pins.includes(parcelPin)) {
        layer.setStyle({
          color: "black",
          weight: 2,
          fillColor: "lime",
          fillOpacity: 0.2,
        });
      } else {
        layer.setStyle({
          color: "black",
          weight: 1,
          fillColor: "white",
          fillOpacity: 0.1,
        });
      }
    });

    // Zoom to bounds safely
    try {
      if (window.map && match.feature?.geometry) {
        const geo = L.geoJSON(match.feature.geometry);
        const bounds = geo.getBounds();
        if (bounds.isValid()) {
          window.map.fitBounds(bounds, { maxZoom: 19 });
        }
      }
    } catch (err) {
      console.error("❌ Error zooming to parcel:", err);
    }

    // === Fetch parcel info for InfoTool ===
    const schema = match.feature.properties.source_schema;
    try {
      const url = `${API}/parcel-info?schema=${schema}&pin=${pin}`;
      const token =
        localStorage.getItem("access_token") ||
        localStorage.getItem("accessToken");

      const res = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          ...(token && { Authorization: `Bearer ${token}` }),
        },
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

      const json = await res.json();
      if (json.status === "success" && json.data) {
        if (typeof onShowParcelInfo === "function") {
          onShowParcelInfo(json.data, schema, pin);
        }
      }
    } catch (err) {
      console.error("Error loading parcel info:", err);
    }
  };

  // ============================================================
  // 🔹 UI
  // ============================================================
  if (noInput) {
    return (
      <div className="search-results">
        <p style={{ color: "#000", fontStyle: "italic" }}>
          Please enter at least one search field.
        </p>
      </div>
    );
  }

  return (
    <div className="search-results">
      {pins.length === 0 ? (
        <>
          <p>
            <b>Results:</b> 0
          </p>
          <p style={{ color: "#555", fontStyle: "italic" }}>No matches found.</p>
        </>
      ) : (
        <>
          <p>
            <b>Results:</b> {pins.length}
          </p>
          <ul>
            {pins.map((pin, idx) => (
              <li
                key={idx}
                className={selectedPin === pin ? "selected" : ""}
                onClick={() => handleResultClick(pin)}
              >
                {pin}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
};

export default SearchResults;
