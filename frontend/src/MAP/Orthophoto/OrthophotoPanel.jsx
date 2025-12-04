import React, { useRef, useEffect, useState } from "react";
import { ChevronRight, Save, ArrowLeft } from "lucide-react";

const OrthophotoPanel = ({ isVisible, onClose, initialData, onSave }) => {
  const containerRef = useRef(null);

  const [gserverUrl, setGserverUrl] = useState("");
  const [layerName, setLayerName] = useState("");
  const [loading, setLoading] = useState(false);
  const [orthoVisible, setOrthoVisible] = useState(false);
  const [isConfiguring, setIsConfiguring] = useState(false);

  // 🧩 Initialize only when panel opens
  useEffect(() => {
    if (isVisible && initialData) {
      setGserverUrl(initialData.Gsrvr_URL || "");
      setLayerName(initialData.Layer_Name || "");
      setOrthoVisible(initialData.orthoVisible || false);
    }
  }, [isVisible, initialData]);

  // 🧭 Prevent map scroll interference
  useEffect(() => {
    if (!containerRef.current) return;
    const stopEvent = (e) => e.stopPropagation();
    const el = containerRef.current;
    el.addEventListener("wheel", stopEvent);
    el.addEventListener("dblclick", stopEvent);
    return () => {
      el.removeEventListener("wheel", stopEvent);
      el.removeEventListener("dblclick", stopEvent);
    };
  }, []);

  // 💾 Save handler
  const handleSave = async () => {
    const urlTrimmed = gserverUrl.trim();
    const layerTrimmed = layerName.trim();

    if (!urlTrimmed || !layerTrimmed) {
      alert("⚠️ Please fill in both fields before saving.");
      return;
    }

    setLoading(true);
    try {
      await onSave({
        Gsrvr_URL: urlTrimmed,
        Layer_Name: layerTrimmed,
      });
      alert("✅ Orthophoto configuration saved successfully.");
    } catch (err) {
      console.error("❌ Save failed:", err);
      alert("❌ Failed to save orthophoto configuration.");
    } finally {
      setLoading(false);
    }
  };

  // 🛰️ Toggle orthophoto visibility
  const handleToggleVisibility = () => {
    if (window._toggleOrthoVisibility) {
      window._toggleOrthoVisibility();
      setOrthoVisible((v) => !v);
    } else {
      setOrthoVisible((v) => !v);
    }
  };

  if (!isVisible) return null;

  return (
    <div
      ref={containerRef}
      className="absolute top-1/2 -translate-y-1/2 right-14 w-[280px] bg-[#151922] text-white rounded-l-lg shadow-xl border border-[#2A2E35] animate-slideIn z-[9999]"
      style={{ pointerEvents: "auto" }}
    >
      {/* Header */}
      <div className="flex justify-between items-center bg-[#F7C800] text-black font-semibold px-4 py-2 rounded-tl-lg">
        <span>Orthophoto Configuration</span>
        <button
          className="text-black hover:opacity-70 transition"
          onClick={onClose}
        >
          <ChevronRight size={20} />
        </button>
      </div>

      {/* Body */}
      <div className="p-4 text-sm space-y-3">
        {/* === ON/OFF SWITCH (Purple Section) === */}
        <div className="bg-[#7700a8] text-white rounded p-3 shadow-sm border border-[#6B21A8]">
          <label className="flex items-center justify-between cursor-pointer">
            <span className="font-semibold text-sm">Show Orthophoto</span>
            <div className="relative">
              <input
                type="checkbox"
                checked={orthoVisible}
                onChange={handleToggleVisibility}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-300 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-[#7C3AED] after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-white" />
            </div>
          </label>
        </div>

        {/* === CONFIGURATION SECTION === */}
        {!isConfiguring ? (
          <div className="flex flex-col items-center justify-center h-[140px] text-gray-400">
            <button
              onClick={() => setIsConfiguring(true)}
              className="w-full bg-[#2C2F36] hover:bg-[#3B3F47] text-white px-3 py-2 rounded text-sm font-medium transition"
            >
              Configure
            </button>
          </div>
        ) : (
          <>
            <div className="border-t border-[#2A2E35] pt-3">
              <h5 className="text-xs text-gray-400 mb-2 font-semibold">
                LAYER CONFIGURATION
              </h5>

              <div className="space-y-1 mb-3">
                <label
                  htmlFor="gserverUrl"
                  className="block text-gray-300 text-xs"
                >
                  GeoServer URL:
                </label>
                <input
                  id="gserverUrl"
                  type="text"
                  value={gserverUrl}
                  onChange={(e) => setGserverUrl(e.target.value)}
                  placeholder="http://your-geoserver/geoserver/gwc/service/wmts"
                  className="w-full bg-[#1E1E1E] text-white border border-[#2A2E35] rounded px-2 py-1.5 text-xs focus:outline-none focus:border-[#F7C800]"
                />
              </div>

              <div className="space-y-1 mb-3">
                <label
                  htmlFor="layerName"
                  className="block text-gray-300 text-xs"
                >
                  Layer Name:
                </label>
                <input
                  id="layerName"
                  type="text"
                  value={layerName}
                  onChange={(e) => setLayerName(e.target.value)}
                  placeholder="workspace:layername"
                  className="w-full bg-[#1E1E1E] text-white border border-[#2A2E35] rounded px-2 py-1.5 text-xs focus:outline-none focus:border-[#F7C800]"
                />
              </div>
            </div>

            {/* Buttons */}
            <div className="flex justify-between mt-4">
              <button
                onClick={() => setIsConfiguring(false)}
                className="flex items-center gap-2 border border-[#3B3F47] text-gray-300 hover:bg-[#2C2F36] px-3 py-1 rounded text-xs font-medium transition"
              >
                <ArrowLeft size={14} /> Back
              </button>
              <button
                onClick={handleSave}
                disabled={loading || !gserverUrl || !layerName}
                className={`flex items-center gap-2 px-3 py-1 rounded text-xs font-semibold transition ${
                  loading || !gserverUrl || !layerName
                    ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                    : "bg-[#F7C800] text-black hover:bg-[#E0B700]"
                }`}
              >
                <Save size={14} />
                {loading ? "Saving..." : "Save"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default OrthophotoPanel;
