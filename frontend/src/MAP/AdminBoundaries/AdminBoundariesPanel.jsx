import React, { useRef, useEffect, useState } from "react";
import { ChevronRight } from "lucide-react";

const AdminBoundariesPanel = ({ isVisible, onClose }) => {
  const containerRef = useRef(null);

  // === Boundary Toggles ===
  const [municipalChecked, setMunicipalChecked] = useState(true);
  const [barangayChecked, setBarangayChecked] = useState(true);
  const [sectionChecked, setSectionChecked] = useState(true); // FIXED
  const [parcelsChecked, setParcelsChecked] = useState(true);
  const [parcelColor, setParcelColor] = useState("black");

  // === Prevent Scroll / Zoom Interference ===
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;

    const stop = (e) => e.stopPropagation();
    el.addEventListener("wheel", stop);
    el.addEventListener("dblclick", stop);

    return () => {
      el.removeEventListener("wheel", stop);
      el.removeEventListener("dblclick", stop);
    };
  }, []);

  // === General Toggle Handler (Municipal, Barangay, Section) ===
  const handleToggle = (type, setter) => (e) => {
    const checked = e.target.checked;
    setter(checked);

    const setStateFn = window[`_setShow${type}`];
    if (setStateFn) setStateFn(checked);

    if (window._updateBoundaryVisibility) {
      window._updateBoundaryVisibility(true);
    }
  };

  // === FIXED Parcel Toggle Handler ===
  const handleParcelToggle = (e) => {
    const checked = e.target.checked;
    setParcelsChecked(checked);

    if (window._setShowParcels) {
      window._setShowParcels(checked);
    }

    if (window._updateBoundaryVisibility) {
      window._updateBoundaryVisibility(true);
    }
  };

  const handleColorChange = (e) => {
    const color = e.target.value;
    setParcelColor(color);
    window.parcelOutlineColor = color;

    if (window._updateBoundaryVisibility) {
      window._updateBoundaryVisibility(true);
    }
  };

  if (!isVisible) return null;

  return (
    <div
      ref={containerRef}
      className="absolute top-1/2 -translate-y-1/2 right-14 w-[260px] bg-[#151922] text-white rounded-l-lg shadow-xl border border-[#2A2E35] animate-slideIn z-[9999]"
      style={{ pointerEvents: "auto" }}
    >
      {/* Header */}
      <div className="flex justify-between items-center bg-[#F7C800] text-black font-semibold px-4 py-2 rounded-tl-lg">
        <span>Admin Boundaries</span>
        <button
          className="text-black hover:opacity-70 transition"
          onClick={onClose}
        >
          <ChevronRight size={20} />
        </button>
      </div>

      {/* Body */}
      <div className="p-4 text-sm space-y-3">

        {/* Municipal Boundary */}
        <label className="flex items-center gap-2 cursor-pointer hover:text-[#F7C800] transition">
          <input
            type="checkbox"
            checked={municipalChecked}
            onChange={handleToggle("Municipal", setMunicipalChecked)}
            className="accent-[#F7C800] w-4 h-4"
          />
          <span>Municipal Boundary</span>
        </label>

        {/* Barangay Boundary */}
        <label className="flex items-center gap-2 cursor-pointer hover:text-[#F7C800] transition">
          <input
            type="checkbox"
            checked={barangayChecked}
            onChange={handleToggle("Barangay", setBarangayChecked)}
            className="accent-[#F7C800] w-4 h-4"
          />
          <span>Barangay Boundary</span>
        </label>

        {/* Section Boundary */}
        <label className="flex items-center gap-2 cursor-pointer hover:text-[#F7C800] transition">
          <input
            type="checkbox"
            checked={sectionChecked}
            onChange={handleToggle("Section", setSectionChecked)}
            className="accent-[#F7C800] w-4 h-4"
          />
          <span>Section Boundary</span>
        </label>

        <hr className="border-[#2A2E35]" />

        {/* Parcels */}
        <label className="flex items-center gap-2 cursor-pointer hover:text-[#F7C800] transition">
          <input
            type="checkbox"
            checked={parcelsChecked}
            onChange={handleParcelToggle}
            className="accent-[#F7C800] w-4 h-4"
          />
          <span>Parcels</span>
        </label>

        {/* Parcel Color Selector */}
        <div className="space-y-2">
          <label htmlFor="parcelColor" className="block text-gray-300 text-xs">
            Parcel Outline Color:
          </label>
          <select
            id="parcelColor"
            value={parcelColor}
            onChange={handleColorChange}
            className="w-full bg-[#1E1E1E] text-white border border-[#2A2E35] rounded px-2 py-1.5 text-sm focus:outline-none focus:border-[#F7C800] cursor-pointer"
          >
            <option value="red">Red</option>
            <option value="orange">Orange</option>
            <option value="yellow">Yellow</option>
            <option value="green">Green</option>
            <option value="blue">Blue</option>
            <option value="violet">Violet</option>
            <option value="black">Black</option>
            <option value="white">White</option>
          </select>
        </div>

      </div>
    </div>
  );
};

export default AdminBoundariesPanel;
