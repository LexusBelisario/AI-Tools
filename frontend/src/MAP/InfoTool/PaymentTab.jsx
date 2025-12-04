// PaymentTab.jsx
import React from "react";

// === Field Groups ===
const propertyFields = [
  { label: "PIN", key: "pin" },
  { label: "Actual Use", key: "land_ause" },
  { label: "Market Value", key: "land_mval" },
  { label: "Assessed Value", key: "land_asval" },
  { label: "ARPN", key: "land_arpn" }
];

const ownerFields = [
  { label: "Acct. No.", key: "l_acctno" },
  { label: "Last Name", key: "l_lastname" },
  { label: "First Name", key: "l_frstname" },
  { label: "Middle Name", key: "l_midname" },
  { label: "Address", key: "l_ownadd" },
  { label: "District", key: "l_owndist" },
  { label: "Mun/City", key: "l_ownmuni" },
  { label: "Barangay", key: "l_ownbrgy" },
  { label: "Province", key: "l_ownprov" },
  { label: "Zip Code", key: "l_ownzip" },
  { label: "Tel No.", key: "l_owntel" }
];

const paymentInfoFields = [
  { label: "Tax Year", key: "tax_year" },
  { label: "Payment Type", key: "paymt_type" },
  { label: "OR No.", key: "or_no" },
  { label: "OR Date", key: "or_date" },
  { label: "Payment Period", key: "pay_period" },
  { label: "No. of Quarter", key: "qtr_no" }
];

const basicFields = [
  { label: "Principal", key: "basic_prin" },
  { label: "Interest", key: "basic_int" },
  { label: "Discount", key: "basic_disc" },
  { label: "Total", key: "basictotal" }
];

const sefFields = [
  { label: "Principal", key: "sef_prin" },
  { label: "Interest", key: "sef_int" },
  { label: "Discount", key: "sef_disc" },
  { label: "Total", key: "sef_total" }
];

// === Main Component ===
const PaymentTab = ({ form, update, editable, readOnly }) => {
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

      {/* Payment Information */}
      <h3>Payment Information</h3>
      <div className="info-grid">
        {paymentInfoFields.map(({ label, key }) => (
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

      {/* Basic Breakdown */}
      <h3>Basic</h3>
      <div className="info-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {basicFields.map(({ label, key }) => (
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

      {/* SEF Breakdown */}
      <h3>SEF</h3>
      <div className="info-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {sefFields.map(({ label, key }) => (
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

export default PaymentTab;
