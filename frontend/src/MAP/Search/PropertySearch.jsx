import React, { useState, useEffect } from "react";
import { useSchema } from "../SchemaContext";
import SearchResults from "./SearchResults";
import { clearParcelHighlights } from "./SearchHelpers";
import "./Search.css";

const PropertySearch = () => {
  const { schema, joinedTable, loadingJoinedTable, status } = useSchema();

  const [filters, setFilters] = useState({
    province: "",
    municipal: "",
    barangay: "",
    section: "",
    parcel: "",
    pin: "",
    arpn: "",
    octtct: "",
    td: "",
    cct: "",
    survey: "",
    cad: "",
    name: "",
    block: "",
    lot: "",
  });

  const [dropdowns, setDropdowns] = useState({
    provinces: [],
    municipals: [],
    barangays: [],
    sections: [],
  });

  const [searchResults, setSearchResults] = useState([]);
  const [selectedPin, setSelectedPin] = useState(null);
  const [searchTriggered, setSearchTriggered] = useState(false);
  const [noInputMessage, setNoInputMessage] = useState(false);

  // ============================================================
  // 🧭 Load dropdowns instantly from global joinedTable
  // ============================================================
  useEffect(() => {
    if (!joinedTable || joinedTable.length === 0) {
      setDropdowns({
        provinces: [],
        municipals: [],
        barangays: [],
        sections: [],
      });
      return;
    }

    const provinces = new Set();
    const municipals = new Set();
    const barangays = new Set();
    const sections = new Set();

    joinedTable.forEach((p) => {
      if (p.prov_code) provinces.add(p.prov_code);

      if (!filters.province || p.prov_code === filters.province) {
        if (p.lgu_code) municipals.add(p.lgu_code);
      }

      if (
        (!filters.province || p.prov_code === filters.province) &&
        (!filters.municipal || p.lgu_code === filters.municipal)
      ) {
        if (p.brgy_nm) barangays.add(p.brgy_nm);
      }

      if (
        (!filters.province || p.prov_code === filters.province) &&
        (!filters.municipal || p.lgu_code === filters.municipal) &&
        (!filters.barangay || p.brgy_nm === filters.barangay)
      ) {
        if (p.sect_code) sections.add(p.sect_code);
      }
    });

    setDropdowns({
      provinces: [...provinces].sort(),
      municipals: [...municipals].sort(),
      barangays: [...barangays].sort(),
      sections: [...sections].sort(),
    });
  }, [joinedTable, filters.province, filters.municipal, filters.barangay]);

  // ============================================================
  // ✏️ Helpers
  // ============================================================
  const updateFilter = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const clearForm = () => {
    setFilters(Object.fromEntries(Object.keys(filters).map((k) => [k, ""])));
    setSearchResults([]);
    setSelectedPin(null);
    setSearchTriggered(false);
    setNoInputMessage(false);
    clearParcelHighlights();
  };

  // ============================================================
  // 🔍 Search Logic
  // ============================================================
  const handlePropertySearch = () => {
    if (loadingJoinedTable || !joinedTable || joinedTable.length === 0) {
      alert("⚠️ Please wait for property data to finish loading.");
      return;
    }

    const hasInput = Object.values(filters).some((v) => v.trim() !== "");
    if (!hasInput) {
      setNoInputMessage(true);
      setSearchResults([]);
      setSearchTriggered(true);
      return;
    }

    setSearchTriggered(true);
    setNoInputMessage(false);
    setSelectedPin(null);
    clearParcelHighlights();

    // === Perform local filtering ===
    const results = joinedTable.filter((p) => {
      return (
        (!filters.province || p.prov_code === filters.province) &&
        (!filters.municipal || p.lgu_code === filters.municipal) &&
        (!filters.barangay || p.brgy_nm === filters.barangay) &&
        (!filters.section || p.sect_code === filters.section) &&
        (!filters.parcel ||
          (p.parcel_cod &&
            p.parcel_cod.toLowerCase().includes(filters.parcel.toLowerCase()))) &&
        (!filters.pin ||
          (p.pin && p.pin.toLowerCase().includes(filters.pin.toLowerCase()))) &&
        (!filters.arpn ||
          (p.arpn && p.arpn.toLowerCase().includes(filters.arpn.toLowerCase()))) &&
        (!filters.octtct ||
          (p.octtct &&
            p.octtct.toLowerCase().includes(filters.octtct.toLowerCase()))) &&
        (!filters.td ||
          (p.td && p.td.toLowerCase().includes(filters.td.toLowerCase()))) &&
        (!filters.cct ||
          (p.cct && p.cct.toLowerCase().includes(filters.cct.toLowerCase()))) &&
        (!filters.survey ||
          (p.survey &&
            p.survey.toLowerCase().includes(filters.survey.toLowerCase()))) &&
        (!filters.cad ||
          (p.cad_no &&
            p.cad_no.toLowerCase().includes(filters.cad.toLowerCase()))) &&
        (!filters.name ||
          (p.l_lastname &&
            p.l_lastname.toLowerCase().includes(filters.name.toLowerCase())) ||
          (p.l_frstname &&
            p.l_frstname.toLowerCase().includes(filters.name.toLowerCase()))) &&
        (!filters.block ||
          (p.blk_no &&
            p.blk_no.toLowerCase().includes(filters.block.toLowerCase()))) &&
        (!filters.lot ||
          (p.lot_no &&
            p.lot_no.toLowerCase().includes(filters.lot.toLowerCase())))
      );
    });

    // === Extract unique sorted PINs ===
    const pins = results
      .map((r) => r.pin)
      .filter(Boolean)
      .filter((v, i, arr) => arr.indexOf(v) === i)
      .sort();

    // === Highlight on map ===
    (window.parcelLayers || []).forEach(({ feature, layer }) => {
      if (pins.includes(feature.properties.pin)) {
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
          fillColor: "black",
          fillOpacity: 0.1,
        });
      }
    });

    setSearchResults(pins);
  };

  // ============================================================
  // 🔹 React-based InfoTool bridge
  // ============================================================
  const handleShowParcelInfo = (data, schema, pin) => {
    if (window.setReactInfoToolData) {
      window.setReactInfoToolData({
        ...data,
        source_schema: schema,
        pin: pin,
        fromSearch: true, // ✅ indicates InfoTool should appear on the right
      });
    } else {
      console.warn("⚠️ InfoTool handler not registered globally yet.");
    }
  };

  // ============================================================
  // 🧱 UI
  // ============================================================
  if (loadingJoinedTable) {
    return <div className="tab-content">⏳ Loading property data...</div>;
  }

  if (status === "error") {
    return (
      <div className="tab-content text-red-500">
        ❌ Failed to load property data for {schema}
      </div>
    );
  }

  return (
    <div className="tab-content">
      {/* === Input Fields === */}
      <div className="field-grid">
        {["province", "municipal", "barangay", "section"].map((key) => (
          <div className="field-cell" key={key}>
            <label>{key}</label>
            <select
              value={filters[key]}
              onChange={(e) => updateFilter(key, e.target.value)}
            >
              <option value="">- Select -</option>
              {dropdowns[`${key}s`].map((val) => (
                <option key={val} value={val}>
                  {val}
                </option>
              ))}
            </select>
          </div>
        ))}

        {[
          "parcel",
          "pin",
          "arpn",
          "octtct",
          "td",
          "cct",
          "survey",
          "cad",
          "name",
          "block",
          "lot",
        ].map((key) => (
          <div className="field-cell" key={key}>
            <label>{key.toUpperCase()}</label>
            <input
              value={filters[key]}
              onChange={(e) => updateFilter(key, e.target.value)}
            />
          </div>
        ))}
      </div>

      {/* === Buttons === */}
      <div className="button-row">
        <button className="search-btn" onClick={handlePropertySearch}>
          Search
        </button>
        <button className="clear-btn" onClick={clearForm}>
          Clear
        </button>
      </div>

      {/* === Results === */}
      {searchTriggered && (
        <SearchResults
          pins={searchResults}
          noInput={noInputMessage}
          selectedPin={selectedPin}
          setSelectedPin={setSelectedPin}
          onShowParcelInfo={handleShowParcelInfo}
        />
      )}
    </div>
  );
};

export default PropertySearch;
