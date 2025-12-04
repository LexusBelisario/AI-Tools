import React, { useState, useEffect } from "react";
import "./Form.css";
import API from "../../api";
import { useSchema } from "../SchemaContext";

const Form = ({ embedded = false, onApply, onClose, showApply = false }) => {
  const { schema } = useSchema();  // Only used for fetching barangays

  const [barangay, setBarangay] = useState("");
  const [barangayList, setBarangayList] = useState([]);

  const [scale, setScale] = useState("2500");
  const [paperSize, setPaperSize] = useState("50x35");
  const [preparedBy, setPreparedBy] = useState("");
  const [verifiedBy, setVerifiedBy] = useState("");
  const [assessor, setAssessor] = useState("");
  const [draftsman, setDraftsman] = useState("");

  // ------------------------------------------------------------
  // Load barangay list for the active schema
  // ------------------------------------------------------------
  useEffect(() => {
    async function loadBarangays() {
      if (!schema) {
        console.warn("TaxmapLayout: schema not ready yet");
        return;
      }

      try {
        const url = `${API}/taxmaplayout/barangays?schema=${schema}`;
        console.log("Fetching barangays:", url);

        const res = await fetch(url);
        const data = await res.json();

        console.log("Barangay list:", data);
        setBarangayList(data.barangays || []);
      } catch (err) {
        console.error("Barangay load failed:", err);
      }
    }

    loadBarangays();
  }, [schema]);

  // ------------------------------------------------------------
  // Submit form
  // ------------------------------------------------------------
  const handleSubmit = () => {
    const payload = {
      barangay,
      scale,
      paperSize,
      preparedBy,
      verifiedBy,
      assessor,
      draftsman
    };

    // DO NOT attach schema here — SectionMap adds it
    if (onApply) onApply(payload);
  };

  const wrapperClass = embedded ? "form-embedded" : "form-overlay";

  return (
    <div className={wrapperClass}>
      <h3 className="form-title">
        {embedded ? "Section Map" : "Form"}
      </h3>

      <div className="form-body">

        {/* Barangay */}
        <div className="form-group">
          <label>Barangay</label>
          <select
            className="form-select"
            value={barangay}
            onChange={(e) => setBarangay(e.target.value)}
          >
            <option value="">Select Barangay</option>
            {barangayList.map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
        </div>

        {/* Scale + Paper */}
        <div className="form-row">
          <div className="form-group" style={{ flex: 1 }}>
            <label>Scale</label>
            <input
              type="text"
              className="form-input"
              value={scale}
              onChange={(e) => setScale(e.target.value)}
            />
          </div>

          <div className="form-group" style={{ flex: 1 }}>
            <label>Paper Size</label>
            <select
              className="form-select"
              value={paperSize}
              onChange={(e) => setPaperSize(e.target.value)}
            >
              <option value="50x35">Standard (50 x 35.5)</option>
              <option value="60x40">Large (60 x 40)</option>
            </select>
          </div>
        </div>

        {/* Prepared By */}
        <div className="form-group">
          <label>Prepared By</label>
          <input
            className="form-input"
            type="text"
            value={preparedBy}
            onChange={(e) => setPreparedBy(e.target.value)}
          />
        </div>

        {/* Verified By */}
        <div className="form-group">
          <label>Verified By</label>
          <input
            className="form-input"
            type="text"
            value={verifiedBy}
            onChange={(e) => setVerifiedBy(e.target.value)}
          />
        </div>

        {/* Assessor */}
        <div className="form-group">
          <label>Provincial/City Assessor</label>
          <input
            className="form-input"
            type="text"
            value={assessor}
            onChange={(e) => setAssessor(e.target.value)}
          />
        </div>

        {/* Draftsman */}
        <div className="form-group">
          <label>Draftsman</label>
          <input
            className="form-input"
            type="text"
            value={draftsman}
            onChange={(e) => setDraftsman(e.target.value)}
          />
        </div>
      </div>

      {/* Apply + Close buttons */}
      {(showApply || !embedded) && (
        <div className="form-btn-row">

          {onClose && (
            <button className="form-btn form-btn-close" onClick={onClose}>
              Close
            </button>
          )}

          <button className="form-btn form-btn-apply" onClick={handleSubmit}>
            Apply
          </button>
        </div>
      )}
    </div>
  );
};

export default Form;
