import { useMap } from "react-leaflet";
import { useEffect } from "react";

export default function MapRefRegisterer() {
  const map = useMap();

  useEffect(() => {
    // ✅ Register both compatible globals for different tools
    window.map = map;                     // Legacy compatibility (used by older tools)
    window._leafletMapInstance = map;     // Standard global (used by sync + new UI)
    
    console.log("✅ Leaflet map registered globally (window.map & window._leafletMapInstance)");
  }, [map]);

  return null;
}

// This file was created to make the map reference available globally so the tools that need to reference the map can do map operations
// in use by Searchresults.jsx
