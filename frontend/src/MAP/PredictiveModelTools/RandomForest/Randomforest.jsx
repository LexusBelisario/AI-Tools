import React, { useState } from "react";
import Plot from "react-plotly.js";
import API from "../../../api.js";
import { useSchema } from "../../SchemaContext.jsx";

export default function RandomForest({ onShowMap }) {
  const { schema: activeSchema } = useSchema();

  const [files, setFiles] = useState([]);
  const [fields, setFields] = useState([]);
  const [independentVars, setIndependentVars] = useState([]);
  const [dependentVar, setDependentVar] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // DB selection
  const [showDbModal, setShowDbModal] = useState(false);
  const [dbTables, setDbTables] = useState([]);
  const [selectedSchema, setSelectedSchema] = useState(null);
  const [selectedTable, setSelectedTable] = useState(null);

  // Run saved model
  const [showRunModal, setShowRunModal] = useState(false);
  const [modelFile, setModelFile] = useState(null);
  const [runFiles, setRunFiles] = useState([]);
  const [showRunDbModal, setShowRunDbModal] = useState(false);
  const [runDbTables, setRunDbTables] = useState([]);
  const [selectedRunSchema, setSelectedRunSchema] = useState(null);
  const [selectedRunDbTable, setSelectedRunDbTable] = useState(null);

  // UI
  const [activeTab, setActiveTab] = useState("inputs");
  const [fullscreenGraph, setFullscreenGraph] = useState(null);

  // Attribute table preview
  const [previewRows, setPreviewRows] = useState([]);
  const [previewTotal, setPreviewTotal] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewErr, setPreviewErr] = useState("");
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 100;

  // Excluded rows (by global index)
  const [excludedIndices, setExcludedIndices] = useState([]);
  const selectedColumns = dependentVar
    ? [dependentVar, ...independentVars]
    : independentVars;

  // CAMA CSV preview
  const [camaPreview, setCamaPreview] = useState({
    rows: [],
    cols: [],
    error: "",
    loading: false,
  });

  // ---------- Plot helpers ----------
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
      bordercolor: "#F7C800",
      font: { color: "white" },
    },
    margin: { l: 60, r: 30, t: 60, b: 60 },
  };

  // 🔗 RF downloads backend
  const normalizeDownloadUrl = (path) => {
    if (!path) return null;
    if (
      path.startsWith("http://") ||
      path.startsWith("https://") ||
      path.startsWith("/api/")
    ) {
      return path;
    }
    const enc = encodeURIComponent(path);
    return `${API}/rf/download?file=${enc}`;
  };

  const resetForNewSource = () => {
    setFields([]);
    setIndependentVars([]);
    setDependentVar("");
    setResult(null);
    setPreviewRows([]);
    setPreviewTotal(null);
    setPreviewErr("");
    setPage(1);
    setExcludedIndices([]);
  };

  // ---------- Attribute preview helpers (DB uses XGB preview) ----------
  const loadDbPreview = async (schema, tableName, pageNum = 1) => {
    try {
      setPreviewLoading(true);
      setPreviewErr("");
      const offset = (pageNum - 1) * PAGE_SIZE;
      const res = await fetch(
        `${API}/xgb/db-preview?schema=${encodeURIComponent(
          schema
        )}&table=${encodeURIComponent(
          tableName
        )}&limit=${PAGE_SIZE}&offset=${offset}`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.fields?.length) setFields(data.fields);
      setPreviewRows(Array.isArray(data.rows) ? data.rows : []);
      setPreviewTotal(Number.isFinite(data.total) ? data.total : null);
    } catch (e) {
      console.error(e);
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

      // reuse LR preview endpoint
      const res = await fetch(`${API}/linear-regression/preview`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data.fields?.length) setFields(data.fields);
      setPreviewRows(Array.isArray(data.rows) ? data.rows : []);
      setPreviewTotal(Number.isFinite(data.total) ? data.total : null);
    } catch (e) {
      console.error(e);
      setPreviewErr("Preview not available.");
      setPreviewRows([]);
      setPreviewTotal(null);
    } finally {
      setPreviewLoading(false);
    }
  };

  // ---------- CSV Preview for CAMA ----------
  function parseCSV(text, maxRows = 100) {
    const rows = [];
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
        if (row.length > 1 || rows.length > 0) rows.push(row);
        row = [];
        if (rows.length >= maxRows + 1) break;
      } else field += c;
    }
    if (field.length || row.length) {
      row.push(field);
      rows.push(row);
    }
    const cols = rows[0] || [];
    const data = rows.slice(1);
    return { cols, rows: data };
  }

  async function previewCAMA(url) {
    try {
      setCamaPreview({ rows: [], cols: [], error: "", loading: true });
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const { cols, rows } = parseCSV(text, 100);
      setCamaPreview({ rows, cols, error: "", loading: false });
    } catch (e) {
      console.error(e);
      setCamaPreview({
        rows: [],
        cols: [],
        error: "Failed to preview CSV.",
        loading: false,
      });
    }
  }

  // ---------- File Input ----------
  const handleFileChange = async (e) => {
    const selected = Array.from(e.target.files || []);
    if (!selected.length) return;
    setFiles(selected);
    setSelectedTable(null);
    setSelectedSchema(null);
    resetForNewSource();

    try {
      const fd = new FormData();
      const hasZip = selected.some((f) =>
        f.name.toLowerCase().endsWith(".zip")
      );

      if (hasZip) {
        if (selected.length > 1) {
          alert("Please upload only one ZIP file.");
          return;
        }
        fd.append(
          "zip_file",
          selected.find((f) => f.name.toLowerCase().endsWith(".zip"))
        );
      } else {
        selected.forEach((f) => fd.append("shapefiles", f));
      }

      // Reuse LR fields endpoint
      const res = await fetch(`${API}/rf/fields`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      if (res.ok && data.fields) setFields(data.fields);
      else alert(data.error || "Unable to extract fields.");

      await loadFilePreview(selected, 1);
    } catch (err) {
      console.error(err);
      alert("Error reading shapefile/zip fields.");
    }
  };

  // ---------- Database Handlers ----------
  const fetchDbTables = async () => {
    try {
      if (!activeSchema) {
        alert("No active schema. Please select a schema first.");
        return;
      }
      setLoading(true);
      // reuse XGB db-tables
      const res = await fetch(
        `${API}/xgb/db-tables?schema=${encodeURIComponent(activeSchema)}`
      );
      const data = await res.json();
      if (res.ok) setDbTables(data.tables || []);
      else alert(data.error || "Failed to load database tables.");
    } catch (err) {
      console.error(err);
      alert("Cannot connect to database.");
    } finally {
      setLoading(false);
    }
  };

  const fetchRunDbTables = async () => {
    try {
      if (!activeSchema) {
        alert("No active schema. Please select a schema first.");
        return;
      }
      setLoading(true);
      // reuse LR db-tables
      const res = await fetch(
        `${API}/linear-regression/db-tables?schema=${encodeURIComponent(
          activeSchema
        )}`
      );
      const data = await res.json();
      if (res.ok) setRunDbTables(data.tables || []);
      else alert(data.error || "Failed to load database tables.");
    } catch (err) {
      console.error(err);
      alert("Cannot connect to database.");
    } finally {
      setLoading(false);
    }
  };

  const fetchDbFields = async (tableNameRaw) => {
    try {
      if (!activeSchema) {
        alert("No active schema. Please select a schema first.");
        return;
      }
      setLoading(true);
      resetForNewSource();

      const schema = activeSchema;
      // allow "schema.table" or just "table"
      const parts = tableNameRaw.split(".");
      const table = parts.length > 1 ? parts[parts.length - 1] : tableNameRaw;

      const res = await fetch(
        `${API}/linear-regression/db-fields?schema=${encodeURIComponent(
          schema
        )}&table=${encodeURIComponent(table)}`
      );
      const data = await res.json();

      if (res.ok && data.fields) {
        setFields(data.fields);
        setSelectedSchema(schema);
        setSelectedTable(table);
        setFiles([]);
        await loadDbPreview(schema, table, 1);
      } else {
        alert(data.error || "Failed to read fields from table.");
      }
    } catch (err) {
      console.error(err);
      alert("Cannot load table fields.");
    } finally {
      setLoading(false);
    }
  };

  // ---------- Exclude rows ----------
  const toggleExcludedRow = (globalIndex) => {
    setExcludedIndices((prev) =>
      prev.includes(globalIndex)
        ? prev.filter((i) => i !== globalIndex)
        : [...prev, globalIndex]
    );
  };

  // ---------- Train Model ----------
  const handleTrainModel = async () => {
    const usingDatabase = !!(selectedSchema && selectedTable);
    const usingFiles = files.length > 0;

    if (!usingDatabase && !usingFiles) {
      return alert("Please upload a shapefile/ZIP or select a database table.");
    }
    if (independentVars.length === 0) {
      return alert("Select independent variables.");
    }
    if (!dependentVar) {
      return alert("Select dependent variable.");
    }

    setLoading(true);
    setResult(null);

    try {
      const fd = new FormData();

      if (usingDatabase) {
        fd.append("schema", selectedSchema);
        fd.append("table_name", selectedTable);
      } else {
        const hasZip = files.some((f) => f.name.toLowerCase().endsWith(".zip"));
        if (hasZip) {
          fd.append(
            "zip_file",
            files.find((f) => f.name.toLowerCase().endsWith(".zip"))
          );
        } else {
          files.forEach((f) => fd.append("shapefiles", f));
        }
      }

      fd.append("independent_vars", JSON.stringify(independentVars));
      fd.append("dependent_var", dependentVar);
      fd.append("excluded_indices", JSON.stringify(excludedIndices));

      const res = await fetch(`${API}/rf/train`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json();

      if (!res.ok) {
        console.error("RF train error:", data);
        alert(`Error: ${data.error || res.statusText}`);
      } else {
        if (data.downloads?.cama_csv) {
          previewCAMA(normalizeDownloadUrl(data.downloads.cama_csv));
        }
        setResult(data);
        setActiveTab("metrics");
        alert("✅ Random Forest training completed!");
      }
    } catch (err) {
      console.error(err);
      alert("Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  // ---------- Run Saved Model ----------
  const handleRunModel = async () => {
    if (!modelFile) return alert("Please select a .pkl model file.");

    const usingDatabase = !!selectedRunSchema && !!selectedRunDbTable;
    const usingFiles = runFiles.length > 0;

    if (!usingDatabase && !usingFiles) {
      return alert("Please upload a shapefile or select a database table.");
    }

    setLoading(true);
    setResult(null);

    try {
      const fd = new FormData();
      fd.append("model_file", modelFile);

      if (usingDatabase) {
        fd.append("schema", selectedRunSchema);
        fd.append("table_name", selectedRunDbTable);
      } else {
        const hasZip = runFiles.some((f) =>
          f.name.toLowerCase().endsWith(".zip")
        );
        if (hasZip) {
          fd.append(
            "zip_file",
            runFiles.find((f) => f.name.toLowerCase().endsWith(".zip"))
          );
        } else {
          runFiles.forEach((f) => fd.append("shapefiles", f));
        }
      }

      const res = await fetch(`${API}/rf/run-saved-model`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json();

      if (!res.ok) {
        console.error("RF run error:", data);
        alert(`Error: ${data.error || res.statusText}`);
      } else {
        if (data.downloads?.shapefile && !data.downloads.geojson) {
          const enc = encodeURIComponent(data.downloads.shapefile);
          data.downloads.geojson = `${API}/rf/preview-geojson?file=${enc}`;
        }
        if (data.downloads?.cama_csv) {
          previewCAMA(normalizeDownloadUrl(data.downloads.cama_csv));
        }
        setResult(data);
        setActiveTab("metrics");
        alert("✅ Prediction completed!");
      }
    } catch (err) {
      console.error(err);
      alert("Failed to connect to backend.");
    } finally {
      setLoading(false);
      setShowRunModal(false);
      setModelFile(null);
      setRunFiles([]);
      setSelectedRunSchema(null);
      setSelectedRunDbTable(null);
    }
  };

  const toggleIndependentVar = (f) =>
    setIndependentVars((prev) =>
      prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]
    );

  const interactive = result && (result.interactive_data || result);
  const ia = interactive ? result.interactive_data || result : null;

  return (
    <div className="flex flex-col w-full h-full bg-[#0a0a0a] text-white">
      {/* Tabs */}
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
        {/* INPUTS */}
        {activeTab === "inputs" && (
          <div className="space-y-4">
            <h3 className="text-[#F7C800] font-semibold text-lg">
              Upload & Configure (Random Forest)
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">
                  Upload Shapefile (.shp/.dbf/.shx/.prj) or ZIP
                </label>
                <div className="flex items-center gap-3">
                  <input
                    id="rfShpInput"
                    type="file"
                    multiple
                    accept=".shp,.dbf,.shx,.prj,.zip"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                  <button
                    onClick={() =>
                      document.getElementById("rfShpInput").click()
                    }
                    className="bg-[#F7C800] text-black px-4 py-2 rounded-md hover:bg-[#e6b800]"
                  >
                    📂 Local
                  </button>
                  <button
                    className="bg-[#374151] text-white px-4 py-2 rounded-md hover:bg-[#4b5563]"
                    onClick={() => {
                      setShowDbModal(true);
                      fetchDbTables();
                    }}
                  >
                    🗄️ Database
                  </button>
                </div>
                <p className="text-xs text-gray-400 truncate">
                  {selectedTable
                    ? `Database Table: ${selectedTable}`
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

            {/* Independent vars */}
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

            {/* Attribute Table Preview with Exclude */}
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
                ) : independentVars.length === 0 || !dependentVar ? (
                  <div className="p-3 text-sm text-gray-400 text-center italic">
                    Select a dependent and independent variable to preview the
                    training data.
                  </div>
                ) : previewRows.length === 0 ? (
                  <div className="p-3 text-xs text-gray-400">
                    No preview available.
                  </div>
                ) : (
                  <table className="min-w-full text-xs">
                    <thead className="bg-[#1e293b] sticky top-0 z-10">
                      <tr>
                        <th className="px-2 py-2 border-b border-gray-700 text-center text-xs">
                          Use
                        </th>
                        {selectedColumns.map((col) => (
                          <th
                            key={col}
                            className="text-left px-2 py-2 border-b border-gray-700"
                          >
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>

                    <tbody>
                      {previewRows.map((row, i) => {
                        const globalIndex = (page - 1) * PAGE_SIZE + i;
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
                            <td className="px-2 py-1 border-b border-gray-800 text-center">
                              <input
                                type="checkbox"
                                checked={!isExcluded}
                                onChange={() => toggleExcludedRow(globalIndex)}
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
                          await loadDbPreview(
                            selectedSchema,
                            selectedTable,
                            next
                          );
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
                          await loadDbPreview(
                            selectedSchema,
                            selectedTable,
                            next
                          );
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

            {/* Actions */}
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

        {/* METRICS */}
        {activeTab === "metrics" && result && (
          <div className="space-y-4">
            {result.downloads && (
              <div className="rounded-lg border border-gray-700 p-3 bg-[#0f172a]">
                <h4 className="text-[#F7C800] font-semibold mb-1">Downloads</h4>
                <ul className="text-sm space-y-1">
                  {result.downloads.model && (
                    <li>
                      <a
                        href={normalizeDownloadUrl(result.downloads.model)}
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
                        href={normalizeDownloadUrl(result.downloads.report)}
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
                        href={normalizeDownloadUrl(result.downloads.shapefile)}
                        target="_blank"
                        rel="noreferrer"
                        className="underline"
                      >
                        🗺️ Predicted Shapefile (.zip)
                      </a>
                    </li>
                  )}
                  {result.downloads.cama_csv && (
                    <li className="flex items-center gap-3">
                      <a
                        href={normalizeDownloadUrl(result.downloads.cama_csv)}
                        target="_blank"
                        rel="noreferrer"
                        className="underline"
                      >
                        📊 Full CAMA Table (CSV)
                      </a>
                    </li>
                  )}
                </ul>
              </div>
            )}

            {(result.metrics || result.features) && (
              <div className="rounded-lg border border-gray-700 p-4 bg-[#0f172a]">
                <h3 className="text-[#F7C800] text-lg font-semibold mb-2">
                  🧠 Random Forest Model Summary
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

                {result.features && result.importance && (
                  <>
                    <h4 className="text-sm font-semibold mt-3 mb-1 text-[#F7C800]">
                      Feature Importance
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm border border-gray-700">
                        <thead className="bg-[#1e293b]">
                          <tr>
                            <th className="text-left p-2 border border-gray-700">
                              Feature
                            </th>
                            <th className="text-right p-2 border border-gray-700">
                              Importance
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.features.map((feat, i) => (
                            <tr key={feat} className="hover:bg-[#111827]">
                              <td className="p-2 border border-gray-700">
                                {feat}
                              </td>
                              <td className="p-2 border border-gray-700 text-right font-mono">
                                {Number.isFinite(result.importance[i])
                                  ? result.importance[i].toFixed(6)
                                  : "—"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* CAMA preview */}
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

        {/* GRAPHS */}
        {activeTab === "graphs" && result && ia && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Importance */}
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
                    x: result.features || [],
                    y: result.importance || [],
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
                config={plotConfig("rf_importance")}
                useResizeHandler
                style={{ width: "100%", height: 250 }}
              />
            </div>

            {/* Residuals histogram */}
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
                    x: ia.residual_bins || [],
                    y: ia.residual_counts || [],
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
                config={plotConfig("rf_residual_distribution")}
                useResizeHandler
                style={{ width: "100%", height: 250 }}
              />
            </div>

            {/* Actual vs Predicted */}
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
                    x: ia.y_test || [],
                    y: ia.preds || [],
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
                    x: ia.y_test || [],
                    y: ia.y_test || [],
                    mode: "lines",
                    line: { color: "#9ca3af", dash: "dash", width: 2 },
                  },
                ]}
                layout={{
                  ...plotLayoutBase,
                  margin: { l: 40, r: 20, t: 20, b: 30 },
                  showlegend: false,
                }}
                config={plotConfig("rf_actual_vs_predicted")}
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

      {/* Bottom actions */}
      <div className="flex flex-wrap gap-3 justify-end border-t border-gray-700 p-3 sm:p-4 bg-[#0f172a]">
        <button
          onClick={async () => {
            if (!result?.downloads) return alert("No predictions yet.");
            let geojsonLink = result.downloads.geojson;
            if (!geojsonLink && result.downloads.shapefile) {
              const enc = encodeURIComponent(result.downloads.shapefile);
              geojsonLink = `${API}/rf/preview-geojson?file=${enc}`;
            }
            if (!geojsonLink) return alert("No predicted map data available.");
            onShowMap?.({
              url: geojsonLink,
              label: "Random Forest",
              field: "prediction",
            });
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
              const payload = {
                shapefile_url: result.downloads.shapefile,
                table_name: "Predicted_Output_RF",
              };
              if (activeSchema) {
                payload.schema = activeSchema;
              }
              const res = await fetch(`${API}/rf/save-to-db`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
              });
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
              <p className="text-sm text-gray-400">No tables found.</p>
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

      {/* Run saved model modal */}
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
              Run Saved Random Forest Model
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
                id="rfRunInput"
                type="file"
                multiple
                accept=".zip,.shp,.dbf,.shx,.prj"
                className="hidden"
                onChange={(e) => setRunFiles(Array.from(e.target.files || []))}
              />
              <div className="flex gap-2">
                <button
                  onClick={() => document.getElementById("rfRunInput").click()}
                  className="bg-[#F7C800] hover:bg-[#e6b800] text-black px-4 py-2 rounded-md"
                >
                  📂 Local
                </button>
                <button
                  className="bg-[#374151] hover:bg-[#4b5563] text-white px-4 py-2 rounded-md"
                  onClick={() => {
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

      {/* Run DB select modal */}
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
                        const parts = t.split(".");
                        const schema = parts.length > 1 ? parts[0] : "";
                        const table = parts.length > 1 ? parts[1] : t;
                        setSelectedRunSchema(schema || activeSchema || "");
                        setSelectedRunDbTable(table);
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

      {/* Fullscreen chart */}
      {fullscreenGraph && result && ia && (
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
                          x: result.features || [],
                          y: result.importance || [],
                          type: "bar",
                          marker: { color: "#F7C800" },
                          name: "Importance",
                        },
                      ]
                    : fullscreenGraph === "residuals"
                      ? [
                          {
                            type: "bar",
                            x: ia.residual_bins || [],
                            y: ia.residual_counts || [],
                            marker: {
                              color: "#F7C800",
                              line: { color: "#0038A8", width: 1 },
                            },
                            name: "Frequency",
                          },
                        ]
                      : [
                          {
                            x: ia.y_test || [],
                            y: ia.preds || [],
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
                            x: ia.y_test || [],
                            y: ia.y_test || [],
                            mode: "lines",
                            line: { color: "#9ca3af", dash: "dash", width: 3 },
                            name: "y = x",
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
                          : "Actual",
                    color: "#d1d5db",
                    gridcolor: "#374151",
                  },
                  yaxis: {
                    title:
                      fullscreenGraph === "importance"
                        ? "Importance"
                        : fullscreenGraph === "residuals"
                          ? "Frequency"
                          : "Predicted",
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
                config={plotConfig(`${fullscreenGraph}_full_rf`)}
                useResizeHandler
                style={{ width: "100%", height: "75vh" }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function formatCell(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "number")
    return Number.isInteger(v) ? v.toString() : v.toFixed(4);
  return String(v);
}
