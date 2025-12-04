import React, { useState } from "react";
import { X, RefreshCw, AlertTriangle } from "lucide-react";
import { useSchema } from "../SchemaContext.jsx";
import { ApiService } from "../../api_service.js";
import MatchSummary from "./MatchSummary.jsx";
import GDBMismatch from "./GDBmismatch.jsx";
import RDBMismatch from "./RDBmismatch.jsx";
import MatchSummaryPrint from "./MatchSummaryPrint.jsx";
import "./MatchingReport.css";

function MatchingReport({ isVisible, onClose }) {
  const { schema } = useSchema();

  // ============================================================
  // 🔹 States
  // ============================================================
  const [activeTab, setActiveTab] = useState("summary");
  const [summaryData, setSummaryData] = useState([]);
  const [gdbMismatchData, setGdbMismatchData] = useState([]);
  const [rdbMismatchData, setRdbMismatchData] = useState([]);
  const [rdbColumns, setRdbColumns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!isVisible) return null;

  // ============================================================
  // 🔹 Run Matching Report
  // ============================================================
  const handleRunReport = async () => {
    if (!schema) {
      alert("⚠️ No schema selected.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const [summaryRes, gdbRes, rdbRes] = await Promise.all([
        ApiService.get(`/matching-report?schema=${schema}`),
        ApiService.get(`/matching-report/gdb-mismatch?schema=${schema}`),
        ApiService.get(`/matching-report/rdb-mismatch?schema=${schema}`)
      ]);

      // --- Summary ---
      if (summaryRes?.status === "success" && Array.isArray(summaryRes.data)) {
        const sortedSummary = [...summaryRes.data].sort((a, b) =>
          (a.code || "").localeCompare(b.code || "")
        );
        setSummaryData(sortedSummary);
      } else {
        setError(summaryRes?.message || "Failed to load summary data.");
      }

      // --- GDB ---
      if (gdbRes?.status === "success" && Array.isArray(gdbRes.data)) {
        const sortedMismatch = [...gdbRes.data].sort((a, b) =>
          (a.pin || "").localeCompare(b.pin || "")
        );
        setGdbMismatchData(sortedMismatch);
      } else {
        setError(gdbRes?.message || "Failed to load GDB mismatch data.");
      }

      // --- RDB ---
      if (rdbRes?.status === "success" && Array.isArray(rdbRes.data)) {
        setRdbMismatchData(rdbRes.data);
        setRdbColumns(rdbRes.columns || []);
      } else {
        setError(rdbRes?.message || "Failed to load RDB mismatch data.");
      }

      console.log("✅ Matching Report Data Loaded.");
    } catch (err) {
      console.error("❌ Error running matching report:", err);
      setError("Failed to run matching report.");
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // 🔹 Tabs
  // ============================================================
  const renderTabs = () => (
    <div className="matching-tabs">
      <button
        className={`tab-btn ${activeTab === "summary" ? "active" : ""}`}
        onClick={() => setActiveTab("summary")}
      >
        Summary
      </button>
      <button
        className={`tab-btn ${activeTab === "gdb" ? "active" : ""}`}
        onClick={() => setActiveTab("gdb")}
      >
        GDB Mismatch
      </button>
      <button
        className={`tab-btn ${activeTab === "rdb" ? "active" : ""}`}
        onClick={() => setActiveTab("rdb")}
      >
        RDB Mismatch
      </button>
    </div>
  );

  // ============================================================
  // 🔹 Active Tab Content
  // ============================================================
  const renderActiveTab = () => {
    switch (activeTab) {
      case "summary":
        return (
          <MatchSummary
            schema={schema}
            rows={summaryData}
            loading={loading}
            error={error}
          />
        );
      case "gdb":
        return (
          <GDBMismatch
            schema={schema}
            rows={gdbMismatchData}
            loading={loading}
            error={error}
          />
        );
      case "rdb":
        return (
          <RDBMismatch
            schema={schema}
            rows={rdbMismatchData}
            columns={rdbColumns}
            loading={loading}
            error={error}
          />
        );
      default:
        return null;
    }
  };

  // ============================================================
  // 🔹 Layout
  // ============================================================
  return (
    <div className="matching-panel compact">
      {/* HEADER */}
      <div className="matching-header">
        <h3 className="matching-title">Matching Report</h3>
        <button className="matching-close-btn" onClick={onClose}>
          <X size={16} />
        </button>
      </div>

      {/* BODY */}
      <div className="matching-body">
        {renderTabs()}
        {renderActiveTab()}
      </div>

      {/* FOOTER BUTTONS */}
      <div className="matching-footer">
        <button
          className="match-btn-yellow"
          onClick={handleRunReport}
          disabled={loading}
        >
          <RefreshCw size={14} />
          {loading ? "Checking..." : "Run Matching Report"}
        </button>

        {activeTab === "summary" && (
          <MatchSummaryPrint
            schema={schema}
            summaryData={summaryData}
            buttonClass="match-btn-purple"
          />
        )}
      </div>

      {error && (
        <div className="matching-error">
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      {/* STYLES */}
      <style jsx="true">{`
        .match-btn-yellow {
          background-color: #f7c800;
          color: black;
          padding: 8px 14px;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          margin-right: 10px;
          font-weight: 500;
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .match-btn-purple {
          background-color: #6a0dad;
          color: white;
          padding: 8px 14px;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 500;
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .match-btn-purple:hover {
          background-color: #590b9e;
        }
      `}</style>
    </div>
  );
}

export default MatchingReport;
