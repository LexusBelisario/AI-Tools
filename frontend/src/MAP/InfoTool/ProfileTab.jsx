// ProfileTab.jsx
import React from "react";

const identificationFields = [
  { label: "Prov Code", key: "prov_code" },
  { label: "Mun Code", key: "mun_code" },
  { label: "Brgy Code", key: "brgy_code" },
  { label: "Sect Code", key: "sect_code" },
  { label: "Parcel Code", key: "parcel_cod" },
];

const propertyFields = [
  { label: "PIN", key: "pin" },
  { label: "ARPN", key: "land_arpn" },
  { label: "TCT No.", key: "tct_no" },
  { label: "Survey No.", key: "survey_no" },
  { label: "Block No.", key: "blk_no" },
  { label: "TD No.", key: "td_no" },
  { label: "Update Code", key: "updte_code" },
  { label: "Lot No.", key: "lot_no" },
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
  { label: "Tel No.", key: "l_owntel" },
];

const boundaryFields = [
  { label: "North", key: "north" },
  { label: "South", key: "south" },
  { label: "East", key: "east" },
  { label: "West", key: "west" },
];

const locationFields = [
  { label: "Barangay", key: "brgy_nm" },
  { label: "Street", key: "street" },
  { label: "Extent", key: "extent" },
];

const ProfileTab = ({ form, update, editable, readOnly }) => {
  return (
    <div className="info-content">

      {/* Property Identification */}
      <h3>Property Identification</h3>
      <div className="info-grid">
        {identificationFields.map(({ label, key }) => (
          <label key={key}>
            {label}
            <input
              value={form[key] || ""}
              onChange={(e) => update(key, e.target.value)}
              readOnly={!editable}
            />
          </label>
        ))}
      </div>

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

      {/* Property Boundaries */}
      <h3>Property Boundaries</h3>
      <div className="info-grid">
        {boundaryFields.map(({ label, key }) => (
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

      {/* Property Location */}
      <h3>Property Location</h3>
      <div className="info-grid">
        {locationFields.map(({ label, key }) => (
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

export default ProfileTab;
