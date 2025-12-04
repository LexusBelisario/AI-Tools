import { useEffect, useState, useRef } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

function BaseMapSelector() {
  const map = useMap();
  const [activeBase, setActiveBase] = useState("terrain");
  const [isExpanded, setIsExpanded] = useState(false);
  const containerRef = useRef(null);

  // ==========================================================
  // 🗺️ Basemap Options
  // ==========================================================
  const basemaps = [
    {
      key: "osm",
      label: "OpenStreetMap",
      thumbnail: "https://a.tile.openstreetmap.org/5/16/10.png",
      layer: L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap contributors",
        maxZoom: 25,
      }),
    },
    {
      key: "satellite",
      label: "Google Satellite",
      thumbnail: "https://mt1.google.com/vt/lyrs=s&x=16&y=10&z=5",
      layer: L.tileLayer("http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", {
        subdomains: ["mt0", "mt1", "mt2", "mt3"],
        attribution: "© Google Satellite",
        maxZoom: 25,
      }),
    },
    {
      key: "terrain",
      label: "Google Terrain",
      thumbnail: "https://mt1.google.com/vt/lyrs=p&x=16&y=10&z=5",
      layer: L.tileLayer("http://{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}", {
        subdomains: ["mt0", "mt1", "mt2", "mt3"],
        attribution: "© Google Terrain",
        maxZoom: 25,
      }),
    },
  ];

  // ==========================================================
  // 🗺️ Initialize Default Basemap
  // ==========================================================
  useEffect(() => {
    if (!map) return;

    console.log("🗺️ Initializing basemap selector...");

    // Store layers globally
    window._basemapLayers = {};
    basemaps.forEach(({ key, layer }) => {
      window._basemapLayers[key] = layer;
    });

    // Add default basemap
    window._basemapLayers[activeBase].addTo(map);

    return () => {
      Object.values(window._basemapLayers).forEach((layer) => {
        if (map.hasLayer(layer)) map.removeLayer(layer);
      });
      delete window._basemapLayers;
    };
  }, [map]);

  // ==========================================================
  // 🔁 Switch Basemap
  // ==========================================================
  const switchBase = (key) => {
    if (!map || !window._basemapLayers) return;

    console.log(`🔄 Switching to ${key}`);

    // Remove current basemap
    if (activeBase && window._basemapLayers[activeBase]) {
      map.removeLayer(window._basemapLayers[activeBase]);
    }

    // Add new basemap
    const newLayer = window._basemapLayers[key];
    newLayer.addTo(map);
    setActiveBase(key);

    // ✅ Keep orthophoto above basemap if it exists
    if (window._orthoLayer && map.hasLayer(window._orthoLayer)) {
      window._orthoLayer.bringToFront();
    }

    // Collapse after selection
    setIsExpanded(false);
  };

  // ==========================================================
  // 🖱️ Click Outside to Close
  // ==========================================================
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsExpanded(false);
      }
    };

    if (isExpanded) {
      setTimeout(() => {
        document.addEventListener("mousedown", handleClickOutside);
      }, 0);
    }

    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isExpanded]);

  // ==========================================================
  // 📝 Get Sorted Basemaps (Active at Bottom)
  // ==========================================================
  const activeBasemap = basemaps.find((b) => b.key === activeBase);
  const inactiveBasemaps = basemaps.filter((b) => b.key !== activeBase);

  // ==========================================================
  // 🧩 Collapsible UI
  // ==========================================================
  return (
    <div
      className="absolute bottom-5 right-3.5 z-[1000] pointer-events-none"
      ref={containerRef}
    >
      <div
        className={`flex flex-col gap-2 bg-[rgba(21,25,34,0.95)] backdrop-blur-md rounded-[10px] p-2 pointer-events-auto transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] [-webkit-backdrop-filter:blur(10px)] ${
          isExpanded
            ? "shadow-[0_8px_24px_rgba(0,0,0,0.4),0_0_0_1px_rgba(255,255,255,0.15)]"
            : "shadow-[0_4px_12px_rgba(0,0,0,0.3),0_0_0_1px_rgba(255,255,255,0.1)]"
        }`}
      >
        {/* Other Cards - Show when Expanded */}
        {isExpanded && (
          <div className="flex flex-col gap-2 animate-[slideDown_0.3s_cubic-bezier(0.4,0,0.2,1)]">
            {inactiveBasemaps.map(({ key, label, thumbnail }) => (
              <button
                key={key}
                className="group bg-white/8 border-2 border-white/12 rounded-lg p-0 cursor-pointer transition-all duration-[250ms] ease-[cubic-bezier(0.4,0,0.2,1)] overflow-hidden w-[90px] relative backdrop-blur-[5px] [-webkit-backdrop-filter:blur(5px)] select-none hover:-translate-y-0.5 hover:scale-[1.02] hover:bg-[rgba(247,200,0,0.15)] hover:border-[#f7c800] hover:shadow-[0_4px_14px_rgba(247,200,0,0.4),0_0_18px_rgba(247,200,0,0.2)] focus:outline-2 focus:outline-[#f7c800] focus:outline-offset-2 focus:not(:focus-visible):outline-none motion-reduce:transition-none motion-reduce:hover:transform-none contrast-more:border-[3px] max-[768px]:w-[75px] max-[480px]:w-[65px]"
                onClick={() => switchBase(key)}
                title={label}
              >
                <div className="w-full h-[70px] relative overflow-hidden bg-black/30 rounded-t-lg max-[768px]:h-[58px] max-[480px]:h-[50px]">
                  <img
                    src={thumbnail}
                    alt={label}
                    className="w-full h-full object-cover block transition-[transform,filter] duration-300 ease-in-out group-hover:scale-[1.08] group-hover:brightness-110 motion-reduce:transition-none motion-reduce:group-hover:transform-none"
                  />
                </div>
                <div className="py-1.5 px-1 text-[10px] font-semibold text-white/85 text-center bg-transparent leading-tight [text-shadow:0_1px_2px_rgba(0,0,0,0.5)] tracking-wide group-hover:text-[#f7c800] group-hover:font-bold group-hover:[text-shadow:0_0_6px_rgba(247,200,0,0.5)] contrast-more:text-white contrast-more:[text-shadow:0_2px_4px_rgba(0,0,0,0.8)] max-[768px]:text-[9px] max-[768px]:py-[5px] max-[768px]:px-[3px] max-[480px]:text-[8px] max-[480px]:py-1 max-[480px]:px-0.5">
                  {label}
                </div>
              </button>
            ))}
          </div>
        )}

        {/* Active Card - Always at Bottom */}
        <button
          className="group bg-[rgba(26,115,232,0.12)] border-2 border-[#1a73e8] rounded-lg p-0 cursor-pointer transition-all duration-[250ms] ease-[cubic-bezier(0.4,0,0.2,1)] overflow-hidden w-[90px] relative backdrop-blur-[5px] [-webkit-backdrop-filter:blur(5px)] select-none hover:-translate-y-0.5 hover:scale-[1.02] hover:border-[#f7c800] hover:bg-[rgba(247,200,0,0.2)] hover:shadow-[0_0_0_2px_rgba(247,200,0,0.5),0_5px_18px_rgba(247,200,0,0.4),0_0_25px_rgba(247,200,0,0.2)] focus:outline-2 focus:outline-[#f7c800] focus:outline-offset-2 focus:not(:focus-visible):outline-none motion-reduce:transition-none motion-reduce:hover:transform-none contrast-more:border-[3px] contrast-more:hover:border-[#f7c800] max-[768px]:w-[75px] max-[480px]:w-[65px]"
          onClick={() => setIsExpanded(!isExpanded)}
          title={isExpanded ? "Collapse" : "Change basemap"}
        >
          <div className="w-full h-[70px] relative overflow-hidden bg-black/30 rounded-t-lg max-[768px]:h-[58px] max-[480px]:h-[50px]">
            <img
              src={activeBasemap.thumbnail}
              alt={activeBasemap.label}
              className="w-full h-full object-cover block transition-[transform,filter] duration-300 ease-in-out group-hover:scale-[1.08] group-hover:brightness-110 motion-reduce:transition-none motion-reduce:group-hover:transform-none"
            />
            <div className="absolute bottom-1 right-1 bg-black/60 rounded-full w-5 h-5 flex items-center justify-center backdrop-blur-[4px] [-webkit-backdrop-filter:blur(4px)] shadow-[0_2px_6px_rgba(0,0,0,0.3)] transition-all duration-300 ease-in-out group-hover:bg-[#f7c800] group-hover:shadow-[0_0_10px_rgba(247,200,0,0.6)] motion-reduce:transition-none max-[768px]:w-[18px] max-[768px]:h-[18px] max-[480px]:w-4 max-[480px]:h-4 max-[480px]:bottom-[3px] max-[480px]:right-[3px]">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="white"
                className={`transition-transform duration-300 ease-in-out group-hover:[filter:drop-shadow(0_0_3px_rgba(0,0,0,0.5))] motion-reduce:transition-none max-[768px]:w-3.5 max-[768px]:h-3.5 max-[480px]:w-3 max-[480px]:h-3 ${
                  isExpanded ? "rotate-180" : "rotate-0"
                }`}
              >
                <path d="M7 14l5-5 5 5z" />
              </svg>
            </div>
          </div>
          <div className="py-1.5 px-1 text-[10px] font-bold text-[#8ab4f8] text-center bg-transparent leading-tight [text-shadow:0_1px_2px_rgba(0,0,0,0.5)] tracking-wide group-hover:text-[#f7c800] group-hover:[text-shadow:0_0_6px_rgba(247,200,0,0.5)] contrast-more:text-white contrast-more:[text-shadow:0_2px_4px_rgba(0,0,0,0.8)] max-[768px]:text-[9px] max-[768px]:py-[5px] max-[768px]:px-[3px] max-[480px]:text-[8px] max-[480px]:py-1 max-[480px]:px-0.5">
            {activeBasemap.label}
          </div>
        </button>
      </div>

      {/* ✅ FIX: remove jsx attribute */}
      <style>
        {`
          @keyframes slideDown {
            from {
              opacity: 0;
              transform: translateY(8px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}
      </style>
    </div>
  );
}

export default BaseMapSelector;
