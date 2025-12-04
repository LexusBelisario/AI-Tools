// MachineryTab.jsx
import React from "react";

// === Field Groups ===
const propertyFields = [
  { label: "PIN", key: "mach_pin" },
  { label: "ARPN", key: "mach_arpn" },
  { label: "Update Code", key: "updte_code" }
];

const ownerFields = [
  { label: "Acct. No.", key: "m_acctno" },
  { label: "Last Name", key: "m_lastname" },
  { label: "First Name", key: "m_frstname" },
  { label: "Middle Name", key: "m_midname" },
  { label: "Address", key: "m_ownadd" },
  { label: "District", key: "m_owndist" },
  { label: "Mun/City", key: "m_ownmuni" },
  { label: "Barangay", key: "m_ownbrgy" },
  { label: "Province", key: "m_ownprov" },
  { label: "Zip Code", key: "m_ownzip" },
  { label: "Tel No.", key: "m_owntel" }
];

const machineryDescriptionFields = [
  { label: "Description", key: "mach_desc" },
  { label: "No. of Units", key: "mach_units" },
  { label: "Cost/Unit", key: "mach_cost" },
  { label: "Market Value", key: "mach_mval" },
  { label: "Actual Use", key: "mach_ause" },
  { label: "MV", key: "mach_mval2" },
  { label: "Level", key: "mach_level" },
  { label: "Adjusted Market Value", key: "mach_adjmv" },
  { label: "Total AV", key: "mach_totav" }
];

const previousPropertyFields = [
  { label: "Prev. ARP", key: "m_prvarp" },
  { label: "Prev. PIN", key: "m_prvpin" },
  { label: "Prev. Owner", key: "m_prvowner" },
  { label: "Effectivity", key: "m_effectvt" },
  { label: "Prev. AV", key: "m_prvav" }
];

// === Main Component ===
const MachineryTab = ({ form, update, editable, readOnly }) => {
  return (
    <div className="info-content">

      {/* Property Information */}
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

      {/* Owner Information */}
      <h3>Owner Information</h3>
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

      {/* Description */}
      <h3>Machinery Description</h3>
      <div className="info-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {machineryDescriptionFields.map(({ label, key }) => (
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

      {/* Previous Property Information */}
      <h3>Previous Property Information</h3>
      <div className="info-grid">
        {previousPropertyFields.map(({ label, key }) => (
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

export default MachineryTab;
