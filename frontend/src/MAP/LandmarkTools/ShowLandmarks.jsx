import React, { useEffect } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import API from "../../api.js";

const ShowLandmarks = ({
  schema,
  visible,
  onClickFeature,
  hiddenLandmarks = [],
  onVisibleChange,
  refreshKey,
}) => {
  const map = useMap();

  useEffect(() => {
    console.log("🔄 ShowLandmarks effect triggered", {
      schema,
      visible,
      hasHandler: !!onClickFeature,
      refreshKey,
    });

    if (!map || !schema) {
      console.warn("⚠️ Missing map or schema", { hasMap: !!map, schema });
      return;
    }

    if (!window.landmarkLayerGroup) {
      window.landmarkLayerGroup = L.layerGroup();
      console.log("🆕 Created landmarkLayerGroup");
    }
    const group = window.landmarkLayerGroup;

    if (!visible) {
      console.log("👁️ Hiding landmarks");
      if (map.hasLayer(group)) map.removeLayer(group);
      group.clearLayers();
      onVisibleChange?.(false);
      return;
    }

    console.log(`🌐 Fetching landmarks for schema=${schema}`);

    // ✅ ADD AUTH HEADERS
    const fetchLandmarks = async () => {
      try {
        const token =
          localStorage.getItem("access_token") ||
          localStorage.getItem("accessToken");
        const res = await fetch(`${API}/landmarks/${schema}`, {
          headers: {
            "Content-Type": "application/json",
            ...(token && { Authorization: `Bearer ${token}` }),
          },
        });

        console.log("📡 Fetch response status:", res.status);

        // ✅ CHECK RESPONSE STATUS
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
        console.log("📦 Landmarks fetched:", data);

        group.clearLayers();

        const features = data?.features || [];
        console.log("🗺️ Number of features:", features.length);

        features.forEach((feature, idx) => {
          const { id, name, type, barangay } = feature.properties || {};

          const isHidden = hiddenLandmarks.some(
            (item) =>
              item.name === name &&
              item.type === type &&
              item.barangay === barangay
          );
          if (isHidden) {
            return;
          }

          const coords = feature.geometry?.coordinates;
          if (!coords || coords.length < 2) {
            return;
          }

          const iconPath = getIconPath(type);
          const icon = L.icon({
            iconUrl: iconPath,
            iconSize: [32, 32],
            iconAnchor: [16, 32],
            popupAnchor: [0, -32],
          });

          const marker = L.marker([coords[1], coords[0]], { icon });

          marker.bindTooltip(name || "Unnamed Landmark", {
            permanent: false,
            direction: "top",
          });

          marker.feature = feature;

          marker.on("click", () => {
            console.log(
              "🟢 Marker clicked — forwarding to handler:",
              onClickFeature ? "yes" : "no"
            );
            onClickFeature?.(feature, marker);
          });

          group.addLayer(marker);
        });

        if (!map.hasLayer(group)) {
          group.addTo(map);
          console.log("✅ Landmark layer group added to map");
        }

        onVisibleChange?.(true);
      } catch (err) {
        console.error("❌ Failed to load landmarks:", err);
        onVisibleChange?.(false);
      }
    };

    fetchLandmarks();

    return () => {
      console.log("🧹 Cleaning up ShowLandmarks effect (schema:", schema, ")");
      if (visible) {
        onVisibleChange?.(false);
      }
    };
  }, [map, visible, schema, onClickFeature, refreshKey]); // ✅ refreshKey added

  return null;
};

function getIconPath(type) {
  const icons = {
    "commercial entities": "Commercial Entities.svg",
    "educational entities": "Educational Entities.svg",
    "financial entities": "Financial Entities.svg",
    "fire station": "Fire Station.svg",
    "gas station": "Gas Station.svg",
    "government entities": "Government Entities.svg",
    "industrial entities": "Industrial Entities.svg",
    "medical entities": "Medical Entities.svg",
    "police station": "Police Station.svg",
    "recreational entities": "Recreational Entities.svg",
    "religious entities": "Religious Entities.svg",
    subdivision: "Subdivision.svg",
    "telecommunication entities": "Telecommunication Entities.svg",
    "transportation entities": "Transportation Entities.svg",
  };

  const file = icons[type?.toLowerCase()] || "default.svg";
  return `/icons/LandmarkIcons/${file}`;
}

export default ShowLandmarks;
