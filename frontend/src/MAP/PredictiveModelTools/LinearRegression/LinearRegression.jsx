import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import API from "../../../api.js";
import { useSchema } from "../../SchemaContext.jsx";

function authFetch(url, opts = {}) {
  const token = localStorage.getItem("token");
  const headers = { ...(opts.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  return fetch(url, {
    credentials: "include",
    ...opts,
    headers,
  });
}
function withSchema(url, schema) {
  if (!schema) return url;
  return (
    url +
    (url.includes("?") ? "&" : "?") +
    "schema=" +
    encodeURIComponent(schema)
  );
}

const pickTTests = (result) => {
  if (!result) return null;
  const cand =
    result.t_tests ??
    result.ttests ??
    result.ttest ??
    result.tTest ??
    result.coefficient_tests ??
    result.stats?.t_tests ??
    result.stats?.ttest ??
    null;

  if (!cand) return null;

  if (Array.isArray(cand)) return cand;

  if (typeof cand === "object") {
    return Object.entries(cand).map(([variable, row]) => ({
      variable,
      ...row,
    }));
  }
  return null;
};

function formatCell(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "number")
    return Number.isInteger(v) ? v.toString() : v.toFixed(4);
  return String(v);
}
function num(v) {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  return Number.isFinite(n)
    ? Math.abs(n) >= 1e6 || Math.abs(n) < 1e-4
      ? n.toExponential(3)
      : n.toFixed(6)
    : String(v);
}

const toNum = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
const fmtP = (v) =>
  v == null ? "—" : v < 1e-4 ? v.toExponential(2) : v.toFixed(4);

export default function LinearRegression({ onClose, onShowMap }) {
  const { selectedSchema, schema } = useSchema();

  const [files, setFiles] = useState([]);
  const [fields, setFields] = useState([]);
  const [independentVars, setIndependentVars] = useState([]);

  const [dependentVar, setDependentVar] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [showDbModal, setShowDbModal] = useState(false);
  const [dbTables, setDbTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState(null);

  const [showRunModal, setShowRunModal] = useState(false);
  const [modelFile, setModelFile] = useState(null);
  const [runFiles, setRunFiles] = useState([]);
  const [showRunDbModal, setShowRunDbModal] = useState(false);
  const [runDbTables, setRunDbTables] = useState([]);
  const [selectedRunDbTable, setSelectedRunDbTable] = useState(null);

  const [activeTab, setActiveTab] = useState("inputs");
  const [selectedGraph, setSelectedGraph] = useState(null);
  const [fullscreenGraph, setFullscreenGraph] = useState(null);

  const [previewRows, setPreviewRows] = useState([]);
  const [previewTotal, setPreviewTotal] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewErr, setPreviewErr] = useState("");
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 100;
  const [excludedIndices, setExcludedIndices] = useState([]);
  const [sortColumn, setSortColumn] = useState(null);
  const [sortDirection, setSortDirection] = useState("desc");
  const handleSort = (col) => {
    if (sortColumn === col) {
      // toggle asc/desc
      setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      // new column → default to highest first
      setSortColumn(col);
      setSortDirection("desc");
    }
  };

  const toggleExcludedRow = (globalIndex) => {
    setExcludedIndices((prev) =>
      prev.includes(globalIndex)
        ? prev.filter((i) => i !== globalIndex)
        : [...prev, globalIndex]
    );
  };

  const selectedColumns = dependentVar
    ? [dependentVar, ...independentVars]
    : independentVars;

  const [camaPreview, setCamaPreview] = useState({
    rows: [],
    cols: [],
    error: "",
    loading: false,
  });

  const plotConfig = (filename) => ({
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    scrollZoom: true,
    toImageButtonOptions: { format: "png", filename },
    modeBarButtonsToRemove: ["select2d", "lasso2d"],
  });
  const plotLayoutBase = {
    paper_bgcolor: "#0f172a",
    plot_bgcolor: "#0f172a",
    font: { color: "white" },
    hoverlabel: {
      bgcolor: "#111",
      bordercolor: "#00ff9d",
      font: { color: "white" },
    },
    margin: { l: 60, r: 30, t: 60, b: 60 },
  };

  const resetStateForNewSource = () => {
    setFields([]);
    setIndependentVars([]);
    setDependentVar("");
    setResult(null);
    setPreviewRows([]);
    setPreviewTotal(null);
    setPreviewErr("");
    setPage(1);
    setSelectedTable(null);
    setExcludedIndices([]); // ✅ clear lahat ng na-exclude
  };

  const loadDbPreview = async (tableName, pageNum = 1) => {
    try {
      setPreviewLoading(true);
      setPreviewErr("");
      const offset = (pageNum - 1) * PAGE_SIZE;
      const url = withSchema(
        `${API}/linear-regression/db-preview?table=${encodeURIComponent(tableName)}&limit=${PAGE_SIZE}&offset=${offset}`,
        selectedSchema
      );
      const res = await authFetch(url);
      if (!res.ok) {
        if (res.status === 404) {
          const data = await res.json().catch(() => ({}));
          setPreviewErr(data.error || "Preview not available for this schema.");
        } else {
          setPreviewErr(`Preview not available (HTTP ${res.status}).`);
        }
        setPreviewRows([]);
        setPreviewTotal(null);
        return;
      }
      const data = await res.json();
      if (data.fields && data.fields.length) setFields(data.fields);
      setPreviewRows(Array.isArray(data.rows) ? data.rows : []);
      setPreviewTotal(Number.isFinite(data.total) ? data.total : null);
    } catch (e) {
      setPreviewErr("Preview not available.");
      setPreviewRows([]);
      setPreviewTotal(null);
    } finally {
      setPreviewLoading(false);
    }
  };

  const loadFilePreview = async (fileList, pageNum = 1) => {
    try {
      setPreviewLoading(true);
      setPreviewErr("");
      const formData = new FormData();
      const hasZip = fileList.some((f) =>
        f.name.toLowerCase().endsWith(".zip")
      );
      if (hasZip) {
        formData.append(
          "zip_file",
          fileList.find((f) => f.name.toLowerCase().endsWith(".zip"))
        );
      } else {
        fileList.forEach((f) => formData.append("shapefiles", f));
      }
      formData.append("limit", String(PAGE_SIZE));
      formData.append("offset", String((pageNum - 1) * PAGE_SIZE));

      const res = await authFetch(`${API}/linear-regression/preview`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.fields && data.fields.length) setFields(data.fields);
      setPreviewRows(Array.isArray(data.rows) ? data.rows : []);
      setPreviewTotal(Number.isFinite(data.total) ? data.total : null);
    } catch (e) {
      setPreviewErr("Preview not available.");
      setPreviewRows([]);
      setPreviewTotal(null);
    } finally {
      setPreviewLoading(false);
    }
  };

  function parseCSV(text, maxRows = 100) {
    const out = [];
    let row = [],
      field = "",
      inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (c === '"') {
        if (inQuotes && text[i + 1] === '"') {
          field += '"';
          i++;
        } else inQuotes = !inQuotes;
      } else if (c === "," && !inQuotes) {
        row.push(field);
        field = "";
      } else if ((c === "\n" || c === "\r") && !inQuotes) {
        if (c === "\r" && text[i + 1] === "\n") i++;
        row.push(field);
        field = "";
        if (row.length > 1 || out.length > 0) out.push(row);
        row = [];
        if (out.length >= maxRows + 1) break;
      } else field += c;
    }
    if (field.length || row.length) {
      row.push(field);
      out.push(row);
    }
    const cols = out[0] || [];
    const rows = out.slice(1);
    return { cols, rows };
  }
  async function previewCAMA(url) {
    try {
      setCamaPreview({ rows: [], cols: [], error: "", loading: true });
      const res = await authFetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const { cols, rows } = parseCSV(text, 100);
      setCamaPreview({ rows, cols, error: "", loading: false });
    } catch (e) {
      setCamaPreview({
        rows: [],
        cols: [],
        error: "Failed to preview CSV.",
        loading: false,
      });
    }
  }

  useEffect(() => {
    const url = result?.downloads?.cama_csv;
    if (!url) return;
    previewCAMA(url);
  }, [result?.downloads?.cama_csv]);

  const handleFileChange = async (e) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (!selectedFiles.length) return;
    setFiles(selectedFiles);
    setSelectedTable(null);
    resetStateForNewSource();

    try {
      const hasZip = selectedFiles.some((f) =>
        f.name.toLowerCase().endsWith(".zip")
      );
      const hasParts = selectedFiles.some((f) =>
        [".shp", ".dbf", ".shx", ".prj"].some((ext) =>
          f.name.toLowerCase().endsWith(ext)
        )
      );
      if (hasZip && hasParts) {
        alert(
          "Please select only a single ZIP file or a complete shapefile set."
        );
        return;
      }

      const formData = new FormData();
      let endpoint;
      if (hasZip) {
        if (selectedFiles.length > 1) {
          alert(
            "Multiple ZIP files detected. Please upload only one ZIP file."
          );
          return;
        }
        formData.append("zip_file", selectedFiles[0]);
        endpoint = `${API}/linear-regression/fields-zip`;
      } else {
        selectedFiles.forEach((f) => formData.append("shapefiles", f));
        endpoint = `${API}/linear-regression/fields`;
      }

      const res = await authFetch(endpoint, { method: "POST", body: formData });
      const data = await res.json();
      if (res.ok && data.fields) setFields(data.fields);
      else alert(data.error || "Unable to extract fields.");

      await loadFilePreview(selectedFiles, 1);
    } catch (err) {
      alert("Error reading shapefile fields. See console for details.");
      console.error(err);
    }
  };

  const fetchDbTables = async () => {
    try {
      setLoading(true);

      // ✅ Debug: Log what schema we're using
      console.log("🔍 fetchDbTables called");
      console.log("🔍 selectedSchema:", selectedSchema);
      console.log("🔍 schema:", schema);

      if (!selectedSchema && !schema) {
        alert(
          "No schema selected. Please go back to Map and select a municipality."
        );
        setDbTables([]);
        return;
      }

      const schemaToUse = selectedSchema || schema;
      console.log("🔍 Using schema:", schemaToUse);

      const url = `${API}/linear-regression/db-tables?schema=${encodeURIComponent(schemaToUse)}`;
      console.log("🔍 Fetching URL:", url);

      const res = await authFetch(url);
      console.log("🔍 Response status:", res.status);

      if (res.status === 404) {
        const data = await res.json().catch(() => ({}));
        console.error("❌ 404 Error:", data);
        alert(data.error || "No AI-eligible tables for this schema.");
        setDbTables([]);
        return;
      }

      const data = await res.json();
      console.log("✅ Response data:", data);

      if (res.ok) {
        setDbTables(data.tables || []);
        console.log("✅ Tables loaded:", data.tables);
      } else {
        alert(data.error || "Failed to load database tables.");
      }
    } catch (err) {
      console.error("❌ fetchDbTables error:", err);
      alert("Cannot connect to database.");
    } finally {
      setLoading(false);
    }
  };
  const fetchRunDbTables = async () => {
    try {
      setLoading(true);
      const res = await authFetch(
        withSchema(`${API}/linear-regression/db-tables`, selectedSchema)
      );
      const data = await res.json();
      if (res.ok) setRunDbTables(data.tables || []);
      else alert(data.error || "Failed to load database tables.");
    } catch (err) {
      alert("Cannot connect to database.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchDbFields = async (tableName) => {
    try {
      setLoading(true);
      resetStateForNewSource();

      if (!selectedSchema) {
        alert("Please select a schema first from the Map page.");
        return;
      }

      const res = await authFetch(
        withSchema(
          `${API}/linear-regression/db-fields?table=${encodeURIComponent(tableName)}`,
          selectedSchema
        )
      );

      if (res.status === 404) {
        const data = await res.json().catch(() => ({}));
        alert(
          data.error ||
            "Table not found in this schema. Local upload still available."
        );
        return;
      }

      const data = await res.json();
      if (res.ok && data.fields) {
        setFields(data.fields);
        setSelectedTable({ schema: selectedSchema, table: tableName });
        setFiles([]);
        await loadDbPreview(tableName, 1);
      } else {
        alert(data.error || "Failed to read fields from table.");
      }
    } catch (err) {
      alert("Cannot load table fields.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const toggleIndependentVar = (f) => {
    setIndependentVars((prev) =>
      prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]
    );
  };

  const handleTrainModel = async () => {
    if (!selectedTable && files.length === 0)
      return alert(
        "Please upload a shapefile, ZIP, or select a database table."
      );
    if (independentVars.length === 0)
      return alert("Select at least one independent variable.");
    if (!dependentVar) return alert("Select a dependent variable.");

    setLoading(true);
    setResult(null);

    try {
      const fd = new FormData();
      if (selectedTable) {
        fd.append("schema", selectedTable.schema); // 👈 ADDED
        fd.append("table_name", selectedTable.table); // 👈 CHANGED
        console.log("✅ Sending DB mode:", {
          schema: selectedTable.schema,
          table: selectedTable.table,
        });
      } else if (files.length > 0) {
        const hasZip = files.some((f) => f.name.toLowerCase().endsWith(".zip"));
        if (hasZip) {
          fd.append(
            "zip_file",
            files.find((f) => f.name.toLowerCase().endsWith(".zip"))
          );
        } else {
          files.forEach((f) => fd.append("shapefiles", f));
        }
        console.log("✅ Sending file mode");
      }

      fd.append("independent_vars", JSON.stringify(independentVars));
      fd.append("dependent_var", dependentVar);
      fd.append("excluded_indices", JSON.stringify(excludedIndices));

      const res = await authFetch(
        `${API}/linear-regression/train`, // 👈 removed withSchema wrapper
        { method: "POST", body: fd }
      );
      const data = await res.json();

      if (!res.ok) {
        alert(`❌ Error: ${data.error || res.statusText}`);
        console.error("Training error:", data);
        return;
      }

      console.log("✅ Training successful:", data);
      setResult(data);
      setActiveTab("metrics");
    } catch (err) {
      console.error("❌ Training request failed:", err);
      alert("Failed to connect to backend or process the model.");
    } finally {
      setLoading(false);
    }
  };

  const handleRunModel = async () => {
    if (!modelFile) return alert("Please select a .pkl model file.");
    const usingDatabase = !!selectedRunDbTable;
    const usingFiles = runFiles.length > 0;
    if (!usingDatabase && !usingFiles)
      return alert("Please upload a shapefile or select a database table.");

    const fd = new FormData();
    fd.append("model_file", modelFile);
    let endpoint = `${API}/linear-regression/run-saved-model`;

    if (usingDatabase) {
      fd.append("schema", selectedSchema);
      fd.append("table_name", selectedRunDbTable);
      endpoint = `${API}/linear-regression/run-saved-model`;
    } else {
      const hasZip = runFiles.some((f) =>
        f.name.toLowerCase().endsWith(".zip")
      );
      if (hasZip) fd.append("zip_file", runFiles[0]);
      else runFiles.forEach((f) => fd.append("shapefiles", f));
    }

    setLoading(true);
    setResult(null);

    try {
      const res = await authFetch(endpoint, { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        alert(`Error: ${data.error || res.statusText}`);
        console.error("Run Model Error:", data);
      } else {
        if (data.downloads?.shapefile && !data.downloads.geojson) {
          data.downloads.geojson = `${API}/linear-regression/preview-geojson?file_path=${encodeURIComponent(
            data.downloads.shapefile
          )}`;
        }
        setResult(data);
        setActiveTab("metrics");
        alert("✅ Model run completed successfully!");
      }
    } catch (err) {
      alert("Failed to connect to backend.");
      console.error(err);
    } finally {
      setLoading(false);
      setShowRunModal(false);
      setModelFile(null);
      setRunFiles([]);
    }
  };

  const ttestRaw = result?.t_test?.residuals ?? result?.t_test ?? null;
  const tResidual = toNum(
    ttestRaw?.t ?? ttestRaw?.t_stat ?? ttestRaw?.tstat ?? ttestRaw?.stat
  );
  const pResidual = toNum(
    ttestRaw?.p ?? ttestRaw?.p_value ?? ttestRaw?.pval ?? ttestRaw?.pvalue
  );
  const coefRows =
    (Array.isArray(result?.t_test?.coefficients) &&
      result.t_test.coefficients) ||
    pickTTests(result) ||
    [];

  return (
    <div className="flex flex-col w-full h-full bg-[#0a0a0a] text-white">
      <div className="flex justify-center gap-3 sm:gap-6 py-3 border-b border-gray-700 bg-[#111827]">
        {["inputs", "metrics", "graphs"].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`uppercase tracking-wide font-semibold text-xs sm:text-sm px-3 sm:px-4 py-2 rounded-md transition-all ${
              activeTab === tab
                ? "bg-[#0038A8] text-white shadow-md"
                : "text-gray-300 hover:text-white hover:bg-[#1e293b]"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-4 sm:p-6 bg-[#111827]">
        {/* schema badge */}
        <div className="text-xs text-gray-400 mb-2">
          🔐 Using schema:{" "}
          <span className="text-white">{selectedSchema || "—"}</span>
        </div>

        {activeTab === "inputs" && (
          <div className="space-y-4">
            <h3 className="text-[#F7C800] font-semibold text-lg">
              Upload & Configure
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">
                  Upload Shapefile (.shp/.dbf/.shx/.prj) or ZIP
                </label>
                <div className="flex items-center gap-3">
                  <input
                    id="shpInput"
                    type="file"
                    multiple
                    accept=".shp,.dbf,.shx,.prj,.zip"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                  <button
                    onClick={() => document.getElementById("shpInput").click()}
                    className="bg-[#F7C800] text-black px-4 py-2 rounded-md hover:bg-[#e6b800]"
                  >
                    📂 Local
                  </button>
                  <button
                    className={`px-4 py-2 rounded-md ${
                      selectedSchema
                        ? "bg-[#374151] text-white hover:bg-[#4b5563]"
                        : "bg-gray-700/50 text-gray-400 cursor-not-allowed"
                    }`}
                    onClick={() => {
                      if (!selectedSchema) {
                        alert(
                          "Pick a schema first in SchemaSelector — required for DB mode."
                        );
                        return;
                      }
                      setShowDbModal(true);
                      fetchDbTables();
                    }}
                  >
                    🗄️ Database
                  </button>
                </div>
                <p className="text-xs text-gray-400 truncate">
                  {selectedTable
                    ? `Database Table: ${selectedTable.schema}.${selectedTable.table}` // 👈 CHANGED
                    : files.length > 0
                      ? files.map((f) => f.name).join(", ")
                      : "No data source chosen"}
                </p>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">
                  Dependent Variable
                </label>
                <select
                  value={dependentVar}
                  onChange={(e) => setDependentVar(e.target.value)}
                  className="w-full bg-[#1f2937] border border-gray-600 rounded-md px-3 py-2 text-sm text-white"
                >
                  <option value="">-- Select --</option>
                  {fields.map((f) => (
                    <option key={f} value={f} className="bg-[#1f2937]">
                      {f}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-300">
                Independent Variables
              </label>
              <div className="rounded-lg p-3 max-h-40 overflow-y-auto border border-gray-700 bg-[#0f172a]">
                {fields.length ? (
                  fields.map((f) => (
                    <label
                      key={f}
                      className="flex items-center gap-2 p-2 rounded cursor-pointer hover:bg-[#111827]"
                    >
                      <input
                        type="checkbox"
                        checked={independentVars.includes(f)}
                        onChange={() => toggleIndependentVar(f)}
                        className="w-4 h-4"
                        style={{ accentColor: "#F7C800" }}
                      />
                      <span className="text-sm">{f}</span>
                    </label>
                  ))
                ) : (
                  <p className="text-xs italic text-center py-2 opacity-70">
                    No fields loaded yet.
                  </p>
                )}
              </div>
            </div>

            <div className="rounded-lg border border-gray-700 bg-[#0f172a]">
              <div className="flex items-center justify-between p-3">
                <h4 className="text-sm font-semibold text-[#F7C800]">
                  Attribute Table Preview
                </h4>
                <div className="text-xs text-gray-400">
                  {previewLoading
                    ? "Loading…"
                    : previewTotal != null
                      ? `${previewTotal} rows`
                      : ""}
                </div>
              </div>
              <div className="overflow-auto max-h-72 border-t border-gray-800">
                {previewErr ? (
                  <div className="p-3 text-xs text-gray-400">{previewErr}</div>
                ) : previewLoading ? (
                  <div className="p-3 text-xs text-gray-400">
                    Loading preview…
                  </div>
                ) : previewRows.length === 0 ? (
                  <div className="p-3 text-xs text-gray-400">
                    No preview available.
                  </div>
                ) : independentVars.length === 0 || !dependentVar ? (
                  <div className="p-3 text-sm text-gray-400 text-center italic">
                    Select a dependent and independent variable to preview the
                    training data.
                  </div>
                ) : (
                  <table className="min-w-full text-xs">
                    <thead className="bg-[#1e293b] sticky top-0 z-10">
                      <tr>
                        <th className="px-2 py-2 border-b border-gray-700 text-center text-xs">
                          <input
                            type="checkbox"
                            className="w-4 h-4"
                            style={{ accentColor: "#F7C800" }}
                            onChange={(e) => {
                              if (e.target.checked) {
                                // ✅ Mark ALL rows on this page as "Use" (i.e., remove from excludedIndices)
                                setExcludedIndices((prev) => {
                                  const start = (page - 1) * PAGE_SIZE;
                                  const end = start + previewRows.length;
                                  return prev.filter(
                                    (idx) => idx < start || idx >= end // keep only indices outside this page
                                  );
                                });
                              } else {
                                // ✅ Mark ALL rows on this page as EXCLUDED
                                setExcludedIndices((prev) => {
                                  const newSet = new Set(prev);
                                  previewRows.forEach((_, i) => {
                                    const globalIndex =
                                      (page - 1) * PAGE_SIZE + i;
                                    newSet.add(globalIndex);
                                  });
                                  return Array.from(newSet);
                                });
                              }
                            }}
                            checked={
                              previewRows.length > 0 &&
                              previewRows.every((_, i) => {
                                const globalIndex = (page - 1) * PAGE_SIZE + i;
                                return !excludedIndices.includes(globalIndex);
                              })
                            }
                          />
                        </th>
                        {selectedColumns.map((col) => (
                          <th
                            key={col}
                            onClick={() => handleSort(col)}
                            className="text-left px-2 py-2 border-b border-gray-700 cursor-pointer select-none hover:bg-[#2a3550]"
                            title="Sort column"
                          >
                            {col}

                            {sortColumn === col && (
                              <span className="ml-1 text-[#F7C800]">
                                {sortDirection === "asc" ? "▲" : "▼"}
                              </span>
                            )}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[...previewRows]
                        .sort((a, b) => {
                          if (!sortColumn) return 0;

                          const valA = Number(a[sortColumn]);
                          const valB = Number(b[sortColumn]);

                          if (!isNaN(valA) && !isNaN(valB)) {
                            return sortDirection === "asc"
                              ? valA - valB
                              : valB - valA;
                          }

                          // fallback string compare
                          const strA = String(
                            a[sortColumn] ?? ""
                          ).toLowerCase();
                          const strB = String(
                            b[sortColumn] ?? ""
                          ).toLowerCase();
                          return sortDirection === "asc"
                            ? strA.localeCompare(strB)
                            : strB.localeCompare(strA);
                        })
                        .map((row, i) => {
                          const globalIndex = (page - 1) * PAGE_SIZE + i; // 🔹 index sa buong dataset
                          const isExcluded =
                            excludedIndices.includes(globalIndex);

                          return (
                            <tr
                              key={i}
                              className={
                                "odd:bg-[#0f172a] even:bg-[#0e1426] hover:bg-[#15213a] " +
                                (isExcluded ? "opacity-40 line-through" : "")
                              }
                            >
                              <td className="px-2 border-b border-gray-800 text-center">
                                <input
                                  type="checkbox"
                                  checked={!isExcluded}
                                  onChange={() =>
                                    toggleExcludedRow(globalIndex)
                                  }
                                  className="w-4 h-4"
                                  style={{ accentColor: "#F7C800" }}
                                />
                              </td>
                              {selectedColumns.map((col) => (
                                <td
                                  key={col}
                                  className="px-2 py-1 border-b border-gray-800 whitespace-nowrap"
                                >
                                  {formatCell(row[col])}
                                </td>
                              ))}
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                )}
              </div>
              {previewTotal != null && previewTotal > PAGE_SIZE && (
                <div className="flex items-center justify-between p-3 border-t border-gray-800">
                  <div className="text-xs text-gray-400">Page {page}</div>
                  <div className="flex gap-2">
                    <button
                      className="text-xs bg-[#374151] hover:bg-[#4b5563] px-3 py-1 rounded disabled:opacity-50"
                      disabled={page === 1 || previewLoading}
                      onClick={async () => {
                        const next = page - 1;
                        setPage(next);
                        if (selectedTable)
                          await loadDbPreview(selectedTable.table, next);
                        else if (files.length)
                          await loadFilePreview(files, next);
                      }}
                    >
                      Prev
                    </button>
                    <button
                      className="text-xs bg-[#374151] hover:bg-[#4b5563] px-3 py-1 rounded disabled:opacity-50"
                      disabled={
                        page * PAGE_SIZE >= previewTotal || previewLoading
                      }
                      onClick={async () => {
                        const next = page + 1;
                        setPage(next);
                        if (selectedTable)
                          await loadDbPreview(selectedTable.table, next);
                        else if (files.length)
                          await loadFilePreview(files, next);
                      }}
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div className="flex gap-3 pt-1">
              <button
                onClick={handleTrainModel}
                disabled={loading}
                className="bg-[#0038A8] text-white px-5 py-2 rounded-lg hover:bg-[#1e40af] disabled:opacity-50"
              >
                {loading ? "Training..." : "Train Model"}
              </button>
              <button
                onClick={() => setShowRunModal(true)}
                disabled={loading}
                className="bg-[#374151] text-white px-5 py-2 rounded-lg hover:bg-[#4b5563] disabled:opacity-50"
              >
                Run Saved Model
              </button>
            </div>
          </div>
        )}

        {activeTab === "metrics" && result && (
          <div className="space-y-4">
            {result.downloads && (
              <div className="rounded-lg border border-gray-700 p-3 bg-[#0f172a]">
                <h4 className="text-[#F7C800] font-semibold mb-1">Downloads</h4>
                <ul className="text-sm space-y-1">
                  {result.downloads.model && (
                    <li>
                      <a
                        href={result.downloads.model}
                        target="_blank"
                        rel="noreferrer"
                        className="underline"
                      >
                        📦 Model (.pkl)
                      </a>
                    </li>
                  )}
                  {result.downloads.report && (
                    <li>
                      <a
                        href={result.downloads.report}
                        target="_blank"
                        rel="noreferrer"
                        className="underline"
                      >
                        📄 PDF Report
                      </a>
                    </li>
                  )}
                  {result.downloads.shapefile && (
                    <li>
                      <a
                        href={result.downloads.shapefile}
                        target="_blank"
                        rel="noreferrer"
                        className="underline"
                      >
                        🗺️ Predicted Shapefile (.zip)
                      </a>
                    </li>
                  )}
                  {result.downloads.cama_csv && (
                    <li>
                      <a
                        href={result.downloads.cama_csv}
                        target="_blank"
                        rel="noreferrer"
                        className="underline"
                      >
                        📊 Full CAMA Table (CSV)
                      </a>
                    </li>
                  )}
                  {result.downloads.residuals_csv && (
                    <li>
                      <a
                        href={result.downloads.residuals_csv}
                        target="_blank"
                        rel="noreferrer"
                        className="underline"
                      >
                        🧪 Residuals (CSV)
                      </a>
                    </li>
                  )}
                </ul>
              </div>
            )}

            {(result.metrics || result.coefficients) && (
              <div className="rounded-lg border border-gray-700 p-4 bg-[#0f172a]">
                <h3 className="text-[#F7C800] text-lg font-semibold mb-2">
                  🧠 Model Summary
                </h3>
                {result.dependent_var && (
                  <p className="text-sm text-gray-300 mb-2">
                    Dependent Variable:{" "}
                    <span className="font-semibold text-white">
                      {result.dependent_var}
                    </span>
                  </p>
                )}

                {result.metrics && (
                  <div className="overflow-x-auto mb-4">
                    <table className="w-full text-sm border border-gray-700">
                      <thead className="bg-[#1e293b]">
                        <tr>
                          <th className="text-left p-2 border border-gray-700">
                            Metric
                          </th>
                          <th className="text-right p-2 border border-gray-700">
                            Value
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(result.metrics || {}).map(([k, v]) => (
                          <tr key={k} className="hover:bg-[#111827]">
                            <td className="p-2 border border-gray-700">{k}</td>
                            <td className="p-2 border border-gray-700 text-right font-mono">
                              {typeof v === "number" ? v.toFixed(6) : String(v)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {result.coefficients && (
                  <>
                    <h4 className="text-sm font-semibold mt-3 mb-1 text-[#F7C800]">
                      Regression Coefficients
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm border border-gray-700">
                        <thead className="bg-[#1e293b]">
                          <tr>
                            <th className="text-left p-2 border border-gray-700">
                              Variable
                            </th>
                            <th className="text-right p-2 border border-gray-700">
                              Coefficient
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(result.coefficients).map(([v, c]) => (
                            <tr key={v} className="hover:bg-[#111827]">
                              <td className="p-2 border border-gray-700">
                                {v}
                              </td>
                              <td className="p-2 border border-gray-700 text-right font-mono">
                                {typeof c === "number"
                                  ? c.toFixed(6)
                                  : String(c)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {"intercept" in result && (
                      <p className="text-right text-sm text-gray-300 mt-2">
                        Intercept:{" "}
                        {typeof result.intercept === "number"
                          ? result.intercept.toFixed(6)
                          : "—"}
                      </p>
                    )}
                  </>
                )}
              </div>
            )}
            {/* Coefficient t-tests (normalized) */}
            {/* {Array.isArray(coefRows) && coefRows.length > 0 && (
              <div className="rounded-lg border border-gray-700 p-4 bg-[#0f172a]">
                <h3 className="text-[#F7C800] text-lg font-semibold mb-2">
                  Coefficient T-Tests (α = 0.05)
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border border-gray-700">
                    <thead className="bg-[#1e293b]">
                      <tr>
                        <th className="text-left p-2 border border-gray-700">
                          Variable
                        </th>
                        <th className="text-right p-2 border border-gray-700">
                          Coefficient
                        </th>
                        <th className="text-right p-2 border border-gray-700">
                          Std Error
                        </th>
                        <th className="text-right p-2 border border-gray-700">
                          t-statistic
                        </th>
                        <th className="text-right p-2 border border-gray-700">
                          p-value
                        </th>
                        <th className="text-center p-2 border border-gray-700">
                          Significant?
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {coefRows.map((test, i) => {
                        const coef = toNum(
                          test.coef ?? test.coefficient ?? test.beta
                        );
                        const se = toNum(
                          test.std_err ?? test.se ?? test.stderr
                        );
                        const t = toNum(test.t ?? test.t_stat ?? test.tstat);
                        const p = toNum(
                          test.p ?? test.p_value ?? test.pval ?? test.pvalue
                        );
                        const significant =
                          test.significant ?? (p != null ? p < 0.05 : false);
                        const variable =
                          test.variable ??
                          test.name ??
                          test.term ??
                          `coef_${i}`;
                        return (
                          <tr
                            key={i}
                            className={`hover:bg-[#111827] ${
                              significant ? "bg-green-900/20" : "bg-red-900/20"
                            }`}
                          >
                            <td className="p-2 border border-gray-700 font-semibold">
                              {variable}
                            </td>
                            <td className="p-2 border border-gray-700 text-right font-mono">
                              {coef == null ? "—" : coef.toFixed(6)}
                            </td>
                            <td className="p-2 border border-gray-700 text-right font-mono">
                              {se == null ? "—" : se.toFixed(6)}
                            </td>
                            <td className="p-2 border border-gray-700 text-right font-mono">
                              {t == null ? "—" : t.toFixed(4)}
                            </td>
                            <td className="p-2 border border-gray-700 text-right font-mono">
                              {fmtP(p)}
                            </td>
                            <td className="p-2 border border-gray-700 text-center">
                              {significant ? (
                                <span className="text-green-400 font-bold">
                                  ✓ Yes
                                </span>
                              ) : (
                                <span className="text-red-400">✗ No</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="text-xs text-gray-400 mt-2">
                  * Variables with p-value &lt; 0.05 are considered
                  statistically significant.
                </p>
              </div>
            )} */}
            {ttestRaw && (
              <div className="rounded-lg border border-gray-700 p-4 bg-[#0f172a]">
                <h3 className="text-[#F7C800] text-lg font-semibold mb-2">
                  Residual Diagnostics
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border border-gray-700">
                    <thead className="bg-[#1e293b]">
                      <tr>
                        <th className="text-left p-2 border border-gray-700">
                          Test
                        </th>
                        <th className="text-right p-2 border border-gray-700">
                          Statistic
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="hover:bg-[#111827]">
                        <td className="p-2 border border-gray-700 font-semibold">
                          T-Test (Residuals)
                        </td>
                        <td className="p-2 border border-gray-700 text-right font-mono">
                          {tResidual == null ? "—" : tResidual.toFixed(4)}
                        </td>
                      </tr>
                      <tr>
                        <td className="p-2 border border-gray-700 font-semibold">
                          P-value
                        </td>
                        <td className="p-2 border border-gray-700 text-right font-mono">
                          {fmtP(pResidual)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {(result.residual_tests ||
              result.diagnostics ||
              result.stats?.diagnostics) && (
              <div className="rounded-lg border border-gray-700 p-4 bg-[#0f172a]">
                <h3 className="text-[#F7C800] text-lg font-semibold mb-2">
                  📈 Additional Diagnostics
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border border-gray-700">
                    <thead className="bg-[#1e293b]">
                      <tr>
                        <th className="text-left p-2 border border-gray-700">
                          Test
                        </th>
                        <th className="text-right p-2 border border-gray-700">
                          Statistic
                        </th>
                        <th className="text-right p-2 border border-gray-700">
                          p-value
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(
                        result.residual_tests ||
                          result.diagnostics ||
                          result.stats?.diagnostics ||
                          {}
                      ).map(([k, v]) => (
                        <tr key={k} className="hover:bg-[#111827]">
                          <td className="p-2 border border-gray-700">{k}</td>
                          <td className="p-2 border border-gray-700 text-right font-mono">
                            {num(v.stat || v.value || v)}
                          </td>
                          <td className="p-2 border border-gray-700 text-right font-mono">
                            {fmtP(
                              toNum(v.p ?? v.pvalue ?? v.p_value ?? v.pval)
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {(result.residual_tests ||
              result.diagnostics ||
              result.stats?.diagnostics) && (
              <div className="rounded-lg border border-gray-700 p-4 bg-[#0f172a]">
                <h3 className="text-[#F7C800] text-lg font-semibold mb-2">
                  Residual Diagnostics
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border border-gray-700">
                    <thead className="bg-[#1e293b]">
                      <tr>
                        <th className="text-left p-2 border border-gray-700">
                          Test
                        </th>
                        <th className="text-right p-2 border border-gray-700">
                          Statistic
                        </th>
                        <th className="text-right p-2 border border-gray-700">
                          p-value
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(
                        result.residual_tests ||
                          result.diagnostics ||
                          result.stats?.diagnostics ||
                          {}
                      ).map(([k, v]) => (
                        <tr key={k} className="hover:bg-[#111827]">
                          <td className="p-2 border border-gray-700">{k}</td>
                          <td className="p-2 border border-gray-700 text-right font-mono">
                            {num(v.stat || v.value || v)}
                          </td>
                          <td className="p-2 border border-gray-700 text-right font-mono">
                            {fmtP(
                              toNum(v.p ?? v.pvalue ?? v.p_value ?? v.pval)
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {result.downloads?.cama_csv && (
              <div className="rounded-lg border border-gray-700 p-4 bg-[#0f172a]">
                <h3 className="text-[#F7C800] text-lg font-semibold mb-2">
                  CAMA Table
                </h3>
                {camaPreview.loading && (
                  <p className="text-sm text-gray-400">Loading…</p>
                )}
                {camaPreview.error && (
                  <p className="text-sm text-red-300">{camaPreview.error}</p>
                )}
                {!camaPreview.loading && camaPreview.rows.length > 0 && (
                  <div className="overflow-auto max-h-72">
                    <table className="min-w-full text-xs">
                      <thead className="bg-[#1e293b] sticky top-0 z-10">
                        <tr>
                          {camaPreview.cols.map((c) => (
                            <th
                              key={c}
                              className="text-left px-2 py-2 border-b border-gray-700"
                            >
                              {c}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {camaPreview.rows.map((r, i) => (
                          <tr
                            key={i}
                            className="odd:bg-[#0f172a] even:bg-[#0e1426] hover:bg-[#15213a]"
                          >
                            {camaPreview.cols.map((c, j) => (
                              <td
                                key={j}
                                className="px-2 py-1 border-b border-gray-800 whitespace-nowrap"
                              >
                                {String(r[j] ?? "")}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {!camaPreview.loading &&
                  camaPreview.rows.length === 0 &&
                  !camaPreview.error && (
                    <p className="text-sm text-gray-400">
                      No rows found in CAMA CSV.
                    </p>
                  )}
              </div>
            )}
          </div>
        )}

        {activeTab === "graphs" && result?.interactive_data && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div
              className="rounded-lg border border-gray-700 p-3 bg-[#0f172a] cursor-pointer"
              onClick={() => setFullscreenGraph("importance")}
            >
              <h4 className="text-sm font-semibold mb-2 text-[#F7C800]">
                Feature Importance
              </h4>
              <Plot
                data={[
                  {
                    x: Object.keys(result.interactive_data.importance || {}),
                    y: Object.values(result.interactive_data.importance || {}),
                    type: "bar",
                    marker: { color: "#F7C800" },
                    showlegend: false,
                  },
                ]}
                layout={{
                  ...plotLayoutBase,
                  margin: { l: 40, r: 20, t: 20, b: 30 },
                  xaxis: { showticklabels: false },
                }}
                config={plotConfig("importance")}
                useResizeHandler
                style={{ width: "100%", height: 250 }}
              />
            </div>

            <div
              className="rounded-lg border border-gray-700 p-3 bg-[#0f172a] cursor-pointer"
              onClick={() => setFullscreenGraph("residuals")}
            >
              <h4 className="text-sm font-semibold mb-2 text-[#F7C800]">
                Residual Distribution
              </h4>
              <Plot
                data={[
                  {
                    type: "bar",
                    x: result.interactive_data.residual_bins || [],
                    y: result.interactive_data.residual_counts || [],
                    marker: {
                      color: "#F7C800",
                      line: { color: "#0038A8", width: 1 },
                    },
                    showlegend: false,
                  },
                ]}
                layout={{
                  ...plotLayoutBase,
                  margin: { l: 40, r: 20, t: 20, b: 30 },
                }}
                config={plotConfig("residual_distribution")}
                useResizeHandler
                style={{ width: "100%", height: 250 }}
              />
            </div>

            <div
              className="rounded-lg border border-gray-700 p-3 bg-[#0f172a] cursor-pointer"
              onClick={() => setFullscreenGraph("actual_pred")}
            >
              <h4 className="text-sm font-semibold mb-2 text-[#F7C800]">
                Actual vs Predicted
              </h4>
              <Plot
                data={[
                  {
                    x: result.interactive_data.y_test || [],
                    y: result.interactive_data.preds || [],
                    mode: "markers",
                    type: "scatter",
                    marker: {
                      color: "#F7C800",
                      size: 6,
                      opacity: 0.8,
                      line: { color: "#0038A8", width: 0.5 },
                    },
                  },
                  {
                    x: result.interactive_data.y_test || [],
                    y: result.interactive_data.y_test || [],
                    mode: "lines",
                    line: { color: "#9ca3af", dash: "dash", width: 2 },
                  },
                ]}
                layout={{
                  ...plotLayoutBase,
                  margin: { l: 40, r: 20, t: 20, b: 30 },
                  showlegend: false,
                }}
                config={plotConfig("actual_pred")}
                useResizeHandler
                style={{ width: "100%", height: 250 }}
              />
            </div>

            <div
              className="rounded-lg border border-gray-700 p-3 bg-[#0f172a] cursor-pointer"
              onClick={() => setFullscreenGraph("resid_pred")}
            >
              <h4 className="text-sm font-semibold mb-2 text-[#F7C800]">
                Residuals vs Predicted
              </h4>
              <Plot
                data={[
                  {
                    x: result.interactive_data.preds || [],
                    y: result.interactive_data.residuals || [],
                    mode: "markers",
                    type: "scatter",
                    marker: { color: "#0038A8", size: 6, opacity: 0.8 },
                  },
                  {
                    x: result.interactive_data.preds || [],
                    y: Array(result.interactive_data.preds?.length || 0).fill(
                      0
                    ),
                    mode: "lines",
                    line: { color: "#9ca3af", dash: "dash", width: 2 },
                  },
                ]}
                layout={{
                  ...plotLayoutBase,
                  margin: { l: 40, r: 20, t: 20, b: 30 },
                  showlegend: false,
                }}
                config={plotConfig("resid_pred")}
                useResizeHandler
                style={{ width: "100%", height: 250 }}
              />
            </div>
          </div>
        )}

        {activeTab !== "inputs" && !result && (
          <p className="text-sm text-gray-400">
            No results yet — train or run a model first.
          </p>
        )}
      </div>

      <div className="flex flex-wrap gap-3 justify-end border-t border-gray-700 p-3 sm:p-4 bg-[#0f172a]">
        <button
          onClick={async () => {
            if (!result?.downloads) return alert("No predictions yet.");
            const geojsonLink =
              result.downloads.geojson ||
              (result.downloads.shapefile
                ? `${API}/linear-regression/preview-geojson?file_path=${encodeURIComponent(
                    result.downloads.shapefile
                  )}`
                : null);
            if (!geojsonLink) return alert("No predicted map data available.");
            if (onShowMap)
              onShowMap({
                url: geojsonLink,
                label: "Linear Regression",
                field: "prediction",
              });
            else alert("Map viewer is not available on this page.");
          }}
          className="bg-[#0038A8] hover:bg-[#1e40af] text-white px-4 py-2 rounded-md"
        >
          🗺️ Show Predicted Values on Map
        </button>

        <button
          onClick={async () => {
            if (!result?.downloads?.shapefile)
              return alert("No shapefile to save.");
            try {
              const res = await authFetch(
                `${API}/linear-regression/save-to-db`,
                {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    shapefile_url: result.downloads.shapefile,
                    table_name: "Predicted_Output",
                  }),
                }
              );
              const data = await res.json();
              if (res.ok) alert(`✅ Saved to database table: ${data.table}`);
              else alert(`❌ Error: ${data.error || "Save failed"}`);
            } catch (err) {
              console.error(err);
              alert("Failed to save to database.");
            }
          }}
          className="bg-[#F7C800] hover:bg-[#e6b800] text-black px-4 py-2 rounded-md"
        >
          💾 Save to Database
        </button>

        <button
          onClick={() => setShowRunModal(true)}
          className="bg-[#374151] hover:bg-[#4b5563] text-white px-4 py-2 rounded-md"
        >
          ▶ Run Saved Model
        </button>
      </div>

      {/* DB Table Picker */}
      {showDbModal && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[10000]"
          onClick={() => setShowDbModal(false)}
        >
          <div
            className="bg-[#0f172a] border border-gray-700 rounded-xl w-[420px] max-h-[80vh] overflow-y-auto p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h4 className="text-[#F7C800] text-lg font-semibold mb-2">
              Select Database Table
            </h4>
            {loading && (
              <p className="text-sm text-gray-400">Loading tables...</p>
            )}
            {!loading && dbTables.length === 0 && (
              <p className="text-sm text-gray-400">
                No tables found or you don’t have access. Ensure this schema has{" "}
                <b>CAMA_Table</b>.
              </p>
            )}
            {!loading && dbTables.length > 0 && (
              <ul className="divide-y divide-gray-800">
                {dbTables.map((t) => (
                  <li key={t}>
                    <button
                      className="w-full text-left px-3 py-2 hover:bg-[#111827] rounded-md"
                      title={t}
                      onClick={() => {
                        fetchDbFields(t);
                        setShowDbModal(false);
                      }}
                    >
                      {t}
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <div className="flex justify-end mt-3">
              <button
                className="bg-[#374151] hover:bg-[#4b5563] text-white px-4 py-2 rounded-md"
                onClick={() => setShowDbModal(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Run Saved Model modal */}
      {showRunModal && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[10000]"
          onClick={() => setShowRunModal(false)}
        >
          <div
            className="bg-[#0f172a] border border-gray-700 rounded-xl w-[420px] p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h4 className="text-[#F7C800] text-lg font-semibold mb-3">
              Run Saved Model
            </h4>
            <label className="text-sm">Upload Model (.pkl)</label>
            <input
              type="file"
              accept=".pkl"
              onChange={(e) => setModelFile(e.target.files?.[0] || null)}
              className="w-full mt-1 bg-[#111827] border border-gray-700 rounded-md px-3 py-2 text-sm text-white"
            />
            <div className="mt-4 space-y-2">
              <label className="text-sm">
                Upload Shapefile (.zip or .shp/.dbf/.shx/.prj)
              </label>
              <input
                id="runShpInput"
                type="file"
                multiple
                accept=".zip,.shp,.dbf,.shx,.prj"
                className="hidden"
                onChange={(e) => setRunFiles(Array.from(e.target.files || []))}
              />
              <div className="flex gap-2">
                <button
                  onClick={() => document.getElementById("runShpInput").click()}
                  className="bg-[#F7C800] hover:bg-[#e6b800] text-black px-4 py-2 rounded-md"
                >
                  📂 Local
                </button>
                <button
                  className={`px-4 py-2 rounded-md ${
                    selectedSchema
                      ? "bg-[#374151] text-white hover:bg-[#4b5563]"
                      : "bg-gray-700/50 text-gray-400 cursor-not-allowed"
                  }`}
                  onClick={() => {
                    if (!selectedSchema) {
                      alert(
                        "Pick a schema first in SchemaSelector — required for DB mode."
                      );
                      return;
                    }
                    setShowRunDbModal(true);
                    fetchRunDbTables();
                  }}
                >
                  🗄️ Database
                </button>
              </div>
              <p className="text-xs text-gray-400 truncate">
                {selectedRunDbTable
                  ? `Database Table: ${selectedRunDbTable}`
                  : runFiles.length > 0
                    ? runFiles.map((f) => f.name).join(", ")
                    : "No data source chosen"}
              </p>
            </div>
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={handleRunModel}
                disabled={loading}
                className="bg-[#0038A8] hover:bg-[#1e40af] text-white px-4 py-2 rounded-md disabled:opacity-50"
              >
                {loading ? "Running..." : "Run"}
              </button>
              <button
                className="bg-[#374151] hover:bg-[#4b5563] text-white px-4 py-2 rounded-md"
                onClick={() => setShowRunModal(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Run DB table picker */}
      {showRunDbModal && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[10000]"
          onClick={() => setShowRunDbModal(false)}
        >
          <div
            className="bg-[#0f172a] border border-gray-700 rounded-xl w-[420px] max-h-[80vh] overflow-y-auto p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h4 className="text-[#F7C800] text-lg font-semibold mb-2">
              Select Database Table
            </h4>
            {loading && (
              <p className="text-sm text-gray-400">Loading tables...</p>
            )}
            {!loading && runDbTables.length === 0 && (
              <p className="text-sm text-gray-400">No tables found.</p>
            )}
            {!loading && runDbTables.length > 0 && (
              <ul className="divide-y divide-gray-800">
                {runDbTables.map((t) => (
                  <li key={t}>
                    <button
                      className="w-full text-left px-3 py-2 hover:bg-[#111827] rounded-md"
                      title={t}
                      onClick={() => {
                        setSelectedRunDbTable(t);
                        setShowRunDbModal(false);
                      }}
                    >
                      {t}
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <div className="flex justify-end mt-3">
              <button
                className="bg-[#374151] hover:bg-[#4b5563] text-white px-4 py-2 rounded-md"
                onClick={() => setShowRunDbModal(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Fullscreen graphs */}
      {fullscreenGraph && result && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[10001] p-4"
          onClick={() => setFullscreenGraph(null)}
        >
          <div
            className="rounded-2xl w-full max-w-6xl overflow-hidden border border-gray-700"
            onClick={(e) => e.stopPropagation()}
            style={{ background: "#0f172a", color: "white" }}
          >
            <div className="px-6 py-4 flex justify-between items-center border-b border-gray-700">
              <h3 className="text-xl font-semibold">
                {fullscreenGraph === "importance" && "Feature Importance"}
                {fullscreenGraph === "residuals" && "Residual Distribution"}
                {fullscreenGraph === "actual_pred" && "Actual vs Predicted"}
                {fullscreenGraph === "resid_pred" && "Residuals vs Predicted"}
              </h3>
              <button
                className="text-2xl rounded-md px-2 hover:bg-[#111827]"
                onClick={() => setFullscreenGraph(null)}
              >
                ✕
              </button>
            </div>
            <div className="p-6">
              <Plot
                data={
                  fullscreenGraph === "importance"
                    ? [
                        {
                          x: Object.keys(
                            result.interactive_data.importance || {}
                          ),
                          y: Object.values(
                            result.interactive_data.importance || {}
                          ),
                          type: "bar",
                          marker: { color: "#F7C800" },
                          name: "Importance",
                        },
                      ]
                    : fullscreenGraph === "residuals"
                      ? [
                          {
                            type: "bar",
                            x: result.interactive_data.residual_bins || [],
                            y: result.interactive_data.residual_counts || [],
                            marker: {
                              color: "#F7C800",
                              line: { color: "#0038A8", width: 1 },
                            },
                            name: "Frequency",
                          },
                        ]
                      : fullscreenGraph === "actual_pred"
                        ? [
                            {
                              x: result.interactive_data.y_test || [],
                              y: result.interactive_data.preds || [],
                              mode: "markers",
                              type: "scatter",
                              marker: {
                                color: "#F7C800",
                                size: 10,
                                opacity: 0.85,
                                line: { color: "#0038A8", width: 0.6 },
                              },
                              name: "Predicted",
                            },
                            {
                              x: result.interactive_data.y_test || [],
                              y: result.interactive_data.y_test || [],
                              mode: "lines",
                              line: {
                                color: "#9ca3af",
                                dash: "dash",
                                width: 3,
                              },
                              name: "y = x",
                            },
                          ]
                        : [
                            {
                              x: result.interactive_data.preds || [],
                              y: result.interactive_data.residuals || [],
                              mode: "markers",
                              type: "scatter",
                              marker: {
                                color: "#0038A8",
                                size: 10,
                                opacity: 0.85,
                              },
                              name: "Residuals",
                            },
                            {
                              x: result.interactive_data.preds || [],
                              y: Array(
                                result.interactive_data.preds?.length || 0
                              ).fill(0),
                              mode: "lines",
                              line: {
                                color: "#9ca3af",
                                dash: "dash",
                                width: 3,
                              },
                              name: "Zero Line",
                            },
                          ]
                }
                layout={{
                  ...plotLayoutBase,
                  xaxis: {
                    title:
                      fullscreenGraph === "importance"
                        ? "Features"
                        : fullscreenGraph === "residuals"
                          ? "Residual"
                          : fullscreenGraph === "actual_pred"
                            ? "Actual"
                            : "Predicted",
                    color: "#d1d5db",
                    gridcolor: "#374151",
                  },
                  yaxis: {
                    title:
                      fullscreenGraph === "importance"
                        ? "Importance (Gain)"
                        : fullscreenGraph === "residuals"
                          ? "Frequency"
                          : fullscreenGraph === "actual_pred"
                            ? "Predicted"
                            : "Residuals",
                    color: "#d1d5db",
                    gridcolor: "#374151",
                  },
                  legend: {
                    x: 0.05,
                    y: 0.95,
                    xanchor: "left",
                    yanchor: "top",
                    bgcolor: "rgba(15,23,42,0.9)",
                    bordercolor: "#374151",
                    borderwidth: 1,
                    font: { color: "white", size: 14 },
                  },
                }}
                config={plotConfig(`${fullscreenGraph}_full`)}
                useResizeHandler
                style={{ width: "100%", height: "75vh" }}
              />
            </div>
          </div>
        </div>
      )}

      {selectedGraph && (
        <div
          className="fixed inset-0 bg-black/70 flex items-center justify-center z-[10005]"
          onClick={() => setSelectedGraph(null)}
        >
          <div
            className="bg-[#0f172a] border border-gray-700 rounded-xl w-[90%] max-w-[900px] max-h-[85vh] overflow-y-auto p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-[#F7C800]">{selectedGraph.title}</h3>
              <button
                className="text-xl hover:bg-[#111827] px-2 rounded"
                onClick={() => setSelectedGraph(null)}
              >
                ✕
              </button>
            </div>
            <Plot
              data={[
                {
                  x: selectedGraph.values,
                  type: "histogram",
                  histnorm: "probability density",
                  marker: { color: "#F7C800", opacity: 0.75 },
                },
              ]}
              layout={{
                paper_bgcolor: "#0f172a",
                plot_bgcolor: "#0f172a",
                font: { color: "white" },
                hoverlabel: {
                  bgcolor: "#111",
                  bordercolor: "#00ff9d",
                  font: { color: "white" },
                },
                margin: { l: 50, r: 30, t: 60, b: 60 },
                xaxis: { title: selectedGraph.column, color: "#d1d5db" },
                yaxis: { title: "Density", color: "#d1d5db" },
                bargap: 0.3,
              }}
              config={{
                responsive: true,
                displaylogo: false,
                scrollZoom: true,
                toImageButtonOptions: {
                  format: "png",
                  filename: selectedGraph.column,
                },
              }}
              useResizeHandler
              style={{ width: "100%", height: "70vh" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
