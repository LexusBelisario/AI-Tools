// ValuationTab.jsx
import React from "react";

// === Field Groups ===
const previousPropertyFields = [
  { label: "Prev. ARP", key: "l_prvarp" },
  { label: "Prev. PIN", key: "l_prvpin" },
  { label: "Prev. Owner", key: "l_prvowner" },
  { label: "Effectivity", key: "effectvty" }, 
  { label: "Prev. AV", key: "l_prvav" }
];

const propertyAppraisalFields = [
  { label: "Classification", key: "land_class" },
  { label: "Subclass", key: "land_sbcls" },
  { label: "Area (Sq m.)", key: "land_area" },
  { label: "Unit Value", key: "land_uv" },
  { label: "Market Value", key: "land_mval" },
  { label: "Total Area", key: "land_areat" },
  { label: "Total Market Value", key: "land_totmv" }
];

const adjustmentFactorFields = [
  { label: "Description", key: "land_desc" },
  { label: "Adjustment Rate", key: "adj_rate" },
  { label: "Adjustment Value", key: "adj_val" }
];

const propertyAssessmentFields = [
  { label: "Actual Use", key: "land_ause" },
  { label: "Area (Sq m.)", key: "land_area" }, // Reuse of "area" key
  { label: "Assessed Level", key: "land_aslvl" },
  { label: "Assessed Value", key: "land_asval" },
  { label: "Total AV", key: "land_totav" }
];

// === Main Component ===
const ValuationTab = ({ form, update, editable, readOnly }) => {
  return (
    <div className="info-content">
      
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

      {/* Property Appraisal */}
      <h3>Property Appraisal</h3>
      <div className="info-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {propertyAppraisalFields.map(({ label, key }) => (
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

      {/* Adjustment Factor */}
      <h3>Adjustment Factor</h3>
      <div className="info-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {adjustmentFactorFields.map(({ label, key }) => (
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

      {/* Property Assessment */}
      <h3>Property Assessment</h3>
      <div className="info-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {propertyAssessmentFields.map(({ label, key }) => (
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

export default ValuationTab;
