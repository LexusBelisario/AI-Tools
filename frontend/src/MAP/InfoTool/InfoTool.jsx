import React, { useState, useEffect } from "react";
import "./InfoTool.css";

import ProfileTab from "./ProfileTab.jsx";
import ValuationTab from "./ValuationTab.jsx";
import BuildingTab from "./BuildingTab.jsx";
import MachineryTab from "./MachineryTab.jsx";
import PaymentTab from "./PaymentTab.jsx";
import API from "../../api.js";

// === Allowed fields per tab ===
const allowedFieldsByTab = {
  profile: [
    // Property Identification
    "prov_code",
    "mun_code",
    "brgy_code",
    "sect_code",
    "parcel_cod",
    // Property Information
    "pin",
    "land_arpn",
    "tct_no",
    "survey_no",
    "updte_code",
    "blk_no",
    "td_no",
    "lot_no",

    // Owner's Information
    "l_acctno",
    "l_lastname",
    "l_frstname",
    "l_midname",
    "l_ownadd",
    "l_owndist",
    "l_ownmuni",
    "l_ownbrgy",
    "l_ownprov",
    "l_ownzip",
    "l_owntel",

    // Property Boundaries
    "north",
    "south",
    "east",
    "west",

    // Property Location
    "brgy_nm",
    "street",
    "extent",
  ],

  valuation: [
    // Previous Property Information
    "l_prvarp",
    "l_prvpin",
    "l_prvowner",
    "effectvty",
    "l_prvav",

    // Property Appraisal
    "land_class",
    "land_sbcls",
    "land_area",
    "land_uv",
    "land_mval",
    "land_areat",
    "land_totmv",

    // Adjustment Factor
    "land_desc",
    "adj_rate",
    "adj_val",

    // Property Assessment
    "land_ause",
    "land_area",
    "land_aslvl",
    "land_asval",
    "land_totav",
  ],

  building: [
    // Property Information
    "bldg_pin",
    "bldg_arpn",
    "updte_code",
    "cct",

    // Owner's Information
    "b_acctno",
    "b_lastname",
    "b_frstname",
    "b_midname",
    "b_ownadd",
    "b_owndist",
    "b_ownmuni",
    "b_ownbrgy",
    "b_ownprov",
    "b_ownzip",
    "b_owntel",

    // Appraisal Information
    "bldg_class",
    "bldg_sbcls",
    "bldg_area",
    "bldg_uv",
    "bldg_mval",
    "bldg_age",
    "bldg_areat",
    "bldg_drate",
    "bldg_dval",
    "bldg_mval2",
    "bldg_dmv",

    // Assessment Information
    "bldg_ause",
    "bldg_area",
    "bldg_aslvl",
    "bldg_asval",
    "b_prvarp",
    "b_prvpin",
    "b_prvav",
    "b_prvowner",
    "b_effectvt",
  ],

  machinery: [
    // Property Information
    "mach_pin",
    "mach_arpn",
    "updte_code",

    // Owner's Information
    "m_acctno",
    "m_lastname",
    "m_frstname",
    "m_midname",
    "m_ownadd",
    "m_owndist",
    "m_ownmuni",
    "m_ownbrgy",
    "m_ownprov",
    "m_ownzip",
    "m_owntel",

    // Machinery Description
    "mach_desc",
    "mach_units",
    "mach_cost",
    "mach_mval",
    "mach_ause",
    "mach_mval2",
    "mach_level",
    "mach_adjmv",
    "mach_totav",

    // Previous Property Information
    "m_prvarp",
    "m_prvpin",
    "m_prvowner",
    "m_effectvt",
    "m_prvav",
  ],

  payment: [
    // Property Information
    "pin",
    "land_ause",
    "land_mval",
    "land_asval",
    "land_arpn",

    // Owner's Information
    "l_acctno",
    "l_lastname",
    "l_frstname",
    "l_midname",
    "l_ownadd",
    "l_owndist",
    "l_ownmuni",
    "l_ownbrgy",
    "l_ownprov",
    "l_ownzip",
    "l_owntel",

    // Payment Information
    "tax_year",
    "paymt_type",
    "or_no",
    "or_date",
    "pay_period",
    "qtr_no",

    // Basic Breakdown
    "basic_prin",
    "basic_int",
    "basic_disc",
    "basictotal",

    // SEF Breakdown
    "sef_prin",
    "sef_int",
    "sef_disc",
    "sef_total",
  ],
};

// === Main InfoTool component ===
const InfoTool = ({
  visible,
  onClose,
  data = {},
  editable = false,
  position = "left",
}) => {
  const [activeTab, setActiveTab] = useState("profile");
  const [form, setForm] = useState({});
  const [originalData, setOriginalData] = useState({}); // ✅ ADDED: Store original data
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");

  // === Load form data when visible or tab changes ===
  useEffect(() => {
    if (!visible) return;
    console.log("🔥 InfoTool received:", data);

    const allowedFields = [
      "id",
      "source_table",
      "source_schema",
      ...(allowedFieldsByTab[activeTab] || []),
    ];

    const filtered = Object.fromEntries(
      Object.entries(data || {}).filter(([k]) => allowedFields.includes(k))
    );

    setForm(filtered);
    setOriginalData(filtered); // ✅ ADDED: Store the original data
    document.getElementById("toolbarPanel")?.classList.remove("visible");
    setSaveMessage("");
  }, [visible, data, activeTab]);

  // === Handle field update ===
  const update = (field, value) => {
    if (!editable) return;
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  // === Handle Edit Mode Toggle ===
  const handleEditToggle = () => {
    // Use the global window object to communicate with the toolbar
    // This tells the parent to switch from "info" tool to "edit" tool
    if (window.switchToEditMode) {
      window.switchToEditMode();
    }
  };

  // === Handle Save ===
  const handleSave = async () => {
    if (!form || !form.pin || !form.source_table || !form.source_schema) {
      console.error("Missing required form fields");
      return;
    }

    // ✅ FIXED: Use original data for fields, current form data for new values
    const originalFields = { ...originalData }; // Contains original PIN
    const newPin = form.pin; // Current (possibly edited) PIN

    console.log("💾 Save Data:");
    console.log("  📄 Original PIN:", originalFields.pin);
    console.log("  🆕 New PIN:", newPin);
    console.log("  📊 Original fields:", originalFields);

    // Remove metadata fields from the original data
    delete originalFields.source_table;
    delete originalFields.source_schema;

    // ✅ FIXED: Use the actual source table from the parcel data
    // The source_table tells us which spatial table this parcel came from
    const spatialTable = form.source_table; // e.g., "LandParcels", "BuildingParcels", etc.

    console.log(
      `🔍 Using source table: ${spatialTable} for spatial operations`
    );

    const payload = {
      schema: form.source_schema,
      table: spatialTable, // ✅ Use the actual source table
      pin: newPin, // ✅ NEW PIN (what user wants)
      fields: originalFields, // ✅ ORIGINAL data (including original PIN)
    };

    console.log("📤 Sending payload:", payload);

    try {
      setSaving(true);
      const res = await fetch(`${API}/update-parcel`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const json = await res.json();
      console.log("🔍 Update response:", json);

      if (json.status === "success") {
        setSaveMessage("✅ Parcel updated successfully!");
        // Update the original data to reflect the new state
        setOriginalData({ ...form });

        // Optionally close the tool or refresh data
        setTimeout(() => {
          setSaveMessage("");
          // Switch back to info mode after successful save
          if (window.switchToInfoMode) {
            window.switchToInfoMode();
          }
        }, 2000);
      } else {
        setSaveMessage(`❌ Update failed: ${json.message}`);
      }
    } catch (err) {
      console.error("❌ Error saving parcel:", err);
      setSaveMessage("❌ An error occurred during save.");
    } finally {
      setSaving(false);
    }
  };

  // === Handle Cancel Edit ===
  const handleCancelEdit = () => {
    // Restore original data
    setForm({ ...originalData });
    setSaveMessage("");
    // Switch back to info mode
    if (window.switchToInfoMode) {
      window.switchToInfoMode();
    }
  };

  if (!visible) return null;

  const panelClass = `visible ${position === "right" ? "slide-right" : ""}`;
  const readOnly = !editable;

  return (
    <div id="infoPopup" className={panelClass}>
      {/* === Header === */}
      <div className="info-header">
        <h3>
          {editable
            ? "Parcel Attribute Editing Tool"
            : "Land Parcel Information"}
        </h3>
        <div className="info-header-buttons">
          {!editable && (
            <button className="edit-btn" onClick={handleEditToggle}>
              Edit
            </button>
          )}
          <button className="close-btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>

      {/* === Debug Info (only in editable mode) === */}
      {editable && (
        <div
          style={{
            padding: "5px",
            fontSize: "10px",
            backgroundColor: "#f0f0f0",
            color: "#666",
          }}
        >
          <strong>Debug:</strong> Original PIN: "{originalData.pin}" → New PIN:
          "{form.pin}"
          {form.pin !== originalData.pin && (
            <span style={{ color: "red" }}> (CHANGED)</span>
          )}
        </div>
      )}

      {/* === Tabs === */}
      <div className="info-tab-bar">
        {["profile", "valuation", "building", "machinery", "payment"].map(
          (tab) => (
            <div
              key={tab}
              className={`info-tab ${activeTab === tab ? "active" : ""}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </div>
          )
        )}
      </div>

      {/* === Tab Content === */}
      {activeTab === "profile" && (
        <ProfileTab
          form={form}
          update={update}
          editable={editable}
          readOnly={readOnly}
        />
      )}
      {activeTab === "valuation" && (
        <ValuationTab
          form={form}
          update={update}
          editable={editable}
          readOnly={readOnly}
        />
      )}
      {activeTab === "building" && (
        <BuildingTab
          form={form}
          update={update}
          editable={editable}
          readOnly={readOnly}
        />
      )}
      {activeTab === "machinery" && (
        <MachineryTab
          form={form}
          update={update}
          editable={editable}
          readOnly={readOnly}
        />
      )}
      {activeTab === "payment" && (
        <PaymentTab
          form={form}
          update={update}
          editable={editable}
          readOnly={readOnly}
        />
      )}

      {/* === Save Bar === */}
      {editable && (
        <div className="save-bar">
          <button className="save-btn" onClick={handleSave} disabled={saving}>
            {saving ? "Saving..." : "Save Changes"}
          </button>
          {saveMessage && (
            <p
              className={`save-message ${
                saveMessage.includes("✅") ? "success" : "error"
              }`}
            >
              {saveMessage}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default InfoTool;
