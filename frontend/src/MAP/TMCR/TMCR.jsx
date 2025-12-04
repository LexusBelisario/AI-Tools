import React, { useState, useMemo } from "react";
import { ApiService } from "../../api_service";
import { useSchema } from "../SchemaContext.jsx";
import TMCRPrintPreview from "./TMCRPrintPreview.jsx";
import "./TMCR.css";

function TMCR({ isVisible, onClose }) {
  const { schema, joinedTable, loadingJoinedTable } = useSchema();

  const [selectedBarangay, setSelectedBarangay] = useState("");
  const [selectedSection, setSelectedSection] = useState("");
  const [loading, setLoading] = useState(false);
  const [reportData, setReportData] = useState([]);

  // ============================================================
  // Barangay Options
  // ============================================================
  const barangayOptions = useMemo(() => {
    const set = new Set();
    joinedTable.forEach((row) => {
      if (row.brgy_nm) set.add(row.brgy_nm);
    });
    return [...set];
  }, [joinedTable]);

  // ============================================================
  // Section Options
  // ============================================================
  const sectionOptions = useMemo(() => {
    if (!selectedBarangay) return [];
    const set = new Set();

    joinedTable.forEach((row) => {
      if (
        row.brgy_nm &&
        row.sect_code &&
        row.brgy_nm.toLowerCase() === selectedBarangay.toLowerCase()
      ) {
        set.add(row.sect_code);
      }
    });

    return [...set].sort((a, b) => parseInt(a) - parseInt(b));
  }, [joinedTable, selectedBarangay]);

  // ============================================================
  // Generate Report
  // ============================================================
  const generateReport = async () => {
    if (!schema || !selectedBarangay || !selectedSection) return;

    setLoading(true);
    setReportData([]);

    try {
      const res = await ApiService.get(
        `/tmcr/generate?schema=${schema}&barangay=${encodeURIComponent(
          selectedBarangay
        )}&section=${encodeURIComponent(selectedSection)}`
      );

      if (res?.status === "success") {
        setReportData(res.data);
      }
    } catch (err) {
      console.error("TMCR Error:", err);
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // Highlight on Map
  // ============================================================
  const highlightOnMap = (pin) => {
    if (!window.highlightFeature || !window.parcelLayers) return;

    const match = window.parcelLayers.find(
      (p) => p.feature?.properties?.pin === pin
    );
    if (match) window.highlightFeature(match.feature);
  };

  // ============================================================
  // Extract Header Fields From DB
  // ============================================================
  const header = reportData.length > 0 ? reportData[0] : null;

  const provinceName = header?.province || "";
  const provinceIndex = header?.prov_code || "";

  const municipalityName = header?.municipal || "";
  const municipalityIndex = header?.mun_code || "";

  const barangayName = header?.brgy_nm || selectedBarangay || "";
  const barangayIndex = header?.brgy_code || "";

  const sectionIndex = header?.sect_code || selectedSection || "";

  // ============================================================
  // UI
  // ============================================================
  if (!isVisible) return null;

  return (
    <div className="tmcr-left-panel">
      <div className="tmcr-header-large">
        <div className="tmcr-title-large">TMCR Report</div>
        <button className="tmcr-close" onClick={onClose}>✕</button>
      </div>

      <div className="tmcr-body">

        {/* ================== TOP 2-COLUMN GRID ================== */}
        <div className="tmcr-top-grid">

          {/* Barangay */}
          <div className="tmcr-field">
            <label>Barangay</label>
            <select
              value={selectedBarangay}
              onChange={(e) => {
                setSelectedBarangay(e.target.value);
                setSelectedSection("");
              }}
            >
              <option value="">Select Barangay</option>
              {barangayOptions.map((x) => (
                <option key={x} value={x}>{x}</option>
              ))}
            </select>
          </div>

          {/* Section */}
          <div className="tmcr-field">
            <label>Section</label>
            <select
              disabled={!selectedBarangay}
              value={selectedSection}
              onChange={(e) => setSelectedSection(e.target.value)}
            >
              <option value="">Select Section</option>
              {sectionOptions.map((x) => (
                <option key={x} value={x}>
                  {x.toString().padStart(3, "0")}
                </option>
              ))}
            </select>
          </div>

          {/* Generate Report (YELLOW GOLD) */}
          <button
            className="tmcr-btn-yellow tmcr-col-btn"
            disabled={!selectedBarangay || !selectedSection || loading}
            onClick={generateReport}
          >
            {loading ? "Generating..." : "Generate Report"}
          </button>

          {/* Print Preview (PURPLE) */}
          <TMCRPrintPreview
            provinceName={provinceName}
            provinceIndex={provinceIndex}
            municipalityName={municipalityName}
            municipalityIndex={municipalityIndex}
            barangayName={barangayName}
            barangayIndex={barangayIndex}
            sectionIndex={sectionIndex}
            reportData={reportData}
            buttonClass="tmcr-btn-purple tmcr-col-btn"
          />
        </div>

        {/* ================== TABLE AREA ================== */}
        <div className="tmcr-table-wrapper">
          {loadingJoinedTable ? (
            <div className="tmcr-status">Loading JoinedTable…</div>
          ) : reportData.length === 0 ? (
            <div className="tmcr-status">No results.</div>
          ) : (
            <table className="tmcr-table">
              <thead>
                <tr>
                  <th>Section</th>
                  <th>Parcel</th>
                  <th>Lot</th>
                  <th>Block</th>
                  <th>TCT</th>
                  <th>Area</th>
                  <th>Classification</th>
                  <th>Owner</th>
                  <th>ARP No.</th>
                  <th>BLDG</th>
                </tr>
              </thead>

              <tbody>
                {reportData.map((row, i) => (
                  <tr
                    key={i}
                    onClick={() =>
                      highlightOnMap(row.pin || row.parcel_cod)
                    }
                  >
                    <td>{row.sect_code}</td>
                    <td>{row.parcel_cod}</td>
                    <td>{row.lot_no}</td>
                    <td>{row.blk_no}</td>
                    <td>{row.tct_no}</td>
                    <td>{row.land_area}</td>
                    <td>{row.land_class}</td>
                    <td>{row.owner_name}</td>
                    <td>{row.land_arpn}</td>
                    <td>{row.bldg_class}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

      </div>
    </div>
  );
}

export default TMCR;
