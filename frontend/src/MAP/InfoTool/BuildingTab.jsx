// BuildingTab.jsx

import React from "react";

// === Field Definitions ===
// Each group is an array of label-key mappings used to render form fields

const propertyFields = [
  { label: "PIN", key: "bldg_pin" },
  { label: "ARPN", key: "bldg_arpn" },
  { label: "Update Code", key: "updte_code" },
  { label: "CCT", key: "cct" }
];

const ownerFields = [
  { label: "Acct. No.", key: "b_acctno" },
  { label: "Last Name", key: "b_lastname" },
  { label: "First Name", key: "b_frstname" },
  { label: "Middle Name", key: "b_midname" },
  { label: "Address", key: "b_ownadd" },
  { label: "District", key: "b_owndist" },
  { label: "Mun/City", key: "b_ownmuni" },
  { label: "Barangay", key: "b_ownbrgy" },
  { label: "Province", key: "b_ownprov" },
  { label: "Zip Code", key: "b_ownzip" },
  { label: "Tel No.", key: "b_owntel" }
];

const appraisalFields = [
  { label: "Class", key: "bldg_class" },
  { label: "Subclass", key: "bldg_sbcls" },
  { label: "Area (sq m)", key: "bldg_area" },
  { label: "Unit Value", key: "bldg_uv" },
  { label: "Market Value", key: "bldg_mval" },
  { label: "Building Age", key: "bldg_age" },
  { label: "Total Area", key: "bldg_areat" },
  { label: "Dep. Rate", key: "bldg_drate" },
  { label: "Dep. Value", key: "bldg_dval" },
  { label: "MV", key: "bldg_mval2" },
  { label: "Dep. MV", key: "bldg_dmv" }
];

const assessmentFields = [
  { label: "Actual Use", key: "bldg_ause" },
  { label: "Area", key: "bldg_area" }, // Shared with appraisal group
  { label: "Assessed Level", key: "bldg_aslvl" },
  { label: "Assessed Value", key: "bldg_asval" },
  { label: "Prev. ARP", key: "b_prvarp" },
  { label: "Prev. PIN", key: "b_prvpin" },
  { label: "Prev. AV", key: "b_prvav" },
  { label: "Prev. Owner", key: "b_prvowner" },
  { label: "Effectivity", key: "b_effectvt" }
];


// === Main Functional Component ===
// This component renders a structured data input form for building-related attributes

const BuildingTab = ({ form, update, editable, readOnly }) => {
  return (
    <div className="info-content">

      {/* Property Information Section */}
      <h3>Property Information</h3>
      <div className="info-grid">
        {propertyFields.map(({ label, key }) => (
          <label key={key}>
            {label}
            <input
              value={form[key] || ""}
              onChange={(e) => update(key, e.target.value)}
              readOnly={!editable || key !== "pin"}
              className={(editable && key !== "pin") ? "readonly" : ""}

            />
          </label>
        ))}
      </div>

      {/* Owner Information Section */}
      <h3>Owner Info</h3>
      <div className="info-grid">
        {ownerFields.map(({ label, key }) => (
          <label key={key}>
            {label}
            <input
              value={form[key] || ""}
              onChange={(e) => update(key, e.target.value)}
              readOnly={!editable || key !== "pin"}
              className={(editable && key !== "pin") ? "readonly" : ""}

            />
          </label>
        ))}
      </div>

      {/* Property Appraisal Section (2-column layout) */}
      <h3>Property Appraisal</h3>
      <div className="info-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {appraisalFields.map(({ label, key }) => (
          <label key={key}>
            {label}
            <input
              value={form[key] || ""}
              onChange={(e) => update(key, e.target.value)}
              readOnly={!editable || key !== "pin"}
              className={(editable && key !== "pin") ? "readonly" : ""}

            />
          </label>
        ))}
      </div>

      {/* Property Assessment Section (2-column layout) */}
      <h3>Property Assessment</h3>
      <div className="info-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {assessmentFields.map(({ label, key }) => (
          <label key={key}>
            {label}
            <input
              value={form[key] || ""}
              onChange={(e) => update(key, e.target.value)}
              readOnly={!editable || key !== "pin"}
              className={(editable && key !== "pin") ? "readonly" : ""}

            />
          </label>
        ))}
      </div>

    </div>
  );
};

export default BuildingTab;
