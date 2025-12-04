import { useMap } from "react-leaflet";
import { useState, useEffect } from "react";
import {
  Plus,
  Minus,
  Crosshair,
  Folder,
  Satellite,
  MapPinned,
  Paintbrush,
  ArrowUpDown,
} from "lucide-react";
import { useSchema } from "./SchemaContext.jsx";
import AdminBoundariesPanel from "./AdminBoundaries/AdminBoundariesPanel.jsx";
import SchemaSelectorPanel from "./SchemaSelector/SchemaSelectorPanel.jsx";
import OrthophotoPanel from "./Orthophoto/OrthophotoPanel.jsx";
import JoinedTableSyncPanel from "./JoinedTableSync/JoinedTableSyncPanel.jsx"; // 🔹 New import

function RightControls({ activeTool, setActiveTool }) {
  const map = useMap();
  const { selectedSchemaBounds } = useSchema();

  // Schema selector data
  const [schemaData, setSchemaData] = useState({
    schemas: [],
    selectedSchema: "",
    loading: true,
    error: null,
    userAccess: null,
  });

  // Orthophoto data
  const [orthophotoData, setOrthophotoData] = useState({
    Gsrvr_URL: "",
    Layer_Name: "",
    loading: false,
    message: "",
    schema: null,
  });

  // ✅ Sync schema selector data
  useEffect(() => {
    const syncData = () => {
      if (window._schemaSelectorData) {
        setSchemaData({ ...window._schemaSelectorData });
      }
    };
    syncData();
    const interval = setInterval(syncData, 100);
    return () => clearInterval(interval);
  }, []);

  // ✅ Sync orthophoto data
  useEffect(() => {
    const syncOrthoData = () => {
      if (window._orthophotoData) {
        setOrthophotoData({ ...window._orthophotoData });
      }
    };
    syncOrthoData();
    const interval = setInterval(syncOrthoData, 100);
    return () => clearInterval(interval);
  }, []);

  // === MAP CONTROLS ===
  const handleZoomIn = () => map.zoomIn();
  const handleZoomOut = () => map.zoomOut();
  const handleCenter = () => {
    if (selectedSchemaBounds) {
      const [[minx, miny, maxx, maxy]] = [selectedSchemaBounds];
      map.fitBounds([
        [miny, minx],
        [maxy, maxx],
      ]);
    } else {
      map.setView([12.8797, 121.774], 6);
    }
  };

  // === TOOL HANDLERS ===
  const toggleTool = (toolName) => {
    setActiveTool((prev) => (prev === toolName ? null : toolName));
  };

  const handleSchemaChange = (schema) => {
    if (window._handleSchemaChange) {
      window._handleSchemaChange(schema);
    }
  };

  const handleOrthophotoSave = async (data) => {
    if (window._handleOrthophotoSave) {
      return await window._handleOrthophotoSave(data);
    }
    return { success: false, message: "Save handler not available" };
  };

  // === STYLING ===
  const panelClass =
    "bg-[#151922E6] rounded-l-lg py-2 w-10 shadow-md backdrop-blur-sm flex flex-col items-center";
  const buttonBase =
    "relative w-8 h-8 flex items-center justify-center rounded transition text-white group";

  const Tooltip = ({ text }) => (
    <span
      className="absolute right-[115%] top-1/2 -translate-y-1/2 bg-[#1E1E1EE6] text-white text-[11px]
                 px-2 py-[3px] rounded-md shadow-md opacity-0 translate-x-1
                 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200 ease-out
                 pointer-events-none whitespace-nowrap font-[Inter] tracking-wide
                 after:content-[''] after:absolute after:right-[-5px] after:top-1/2 after:-translate-y-1/2
                 after:border-4 after:border-transparent after:border-l-[#1E1E1EE6]"
    >
      {text}
    </span>
  );

  const hoverColor = "hover:bg-[#F7C80033]";
  const activeColor = "bg-[#F7C800]";

  return (
    <>
      {/* 🎛️ Right Side Button Stack */}
      <div className="absolute top-1/2 right-0 -translate-y-1/2 flex flex-col items-center gap-3 z-[1000] select-none">

        {/* 🔍 Zoom + Center */}
        <div className={panelClass}>
          <button onClick={handleZoomIn} className={`${buttonBase} ${hoverColor}`}>
            <Plus size={18} />
            <Tooltip text="Zoom In" />
          </button>
          <button onClick={handleZoomOut} className={`${buttonBase} ${hoverColor}`}>
            <Minus size={18} />
            <Tooltip text="Zoom Out" />
          </button>
          <button onClick={handleCenter} className={`${buttonBase} ${hoverColor}`}>
            <Crosshair size={18} />
            <Tooltip text="Center Map" />
          </button>
        </div>

        {/* 🧩 Tool Buttons */}
        <div className={panelClass}>
          <button
            onClick={() => toggleTool("schema")}
            className={`${buttonBase} ${activeTool === "schema" ? activeColor : hoverColor}`}
          >
            <Folder size={18} />
            <Tooltip text="Municipality/City" />
          </button>

          <button
            onClick={() => toggleTool("ortho")}
            className={`${buttonBase} ${activeTool === "ortho" ? activeColor : hoverColor}`}
          >
            <Satellite size={18} />
            <Tooltip text="Orthophoto" />
          </button>

          <button
            onClick={() => toggleTool("admin")}
            className={`${buttonBase} ${activeTool === "admin" ? activeColor : hoverColor}`}
          >
            <MapPinned size={18} />
            <Tooltip text="Admin Boundaries" />
          </button>

          <button
            onClick={() => toggleTool("sync")}
            className={`${buttonBase} ${activeTool === "sync" ? activeColor : hoverColor}`}
          >
            <ArrowUpDown size={18} />
            <Tooltip text="RPT-GIS Sync Tool" />
          </button>

          <button
            onClick={() => toggleTool("parcel")}
            className={`${buttonBase} ${activeTool === "parcel" ? activeColor : hoverColor}`}
          >
            <Paintbrush size={18} />
            <Tooltip text="Parcel Styling" />
          </button>
        </div>
      </div>

      {/* 🧭 Tool Panels */}
      <SchemaSelectorPanel
        isVisible={activeTool === "schema"}
        onClose={() => setActiveTool(null)}
        schemas={schemaData.schemas}
        selectedSchema={schemaData.selectedSchema}
        onSchemaChange={handleSchemaChange}
        loading={schemaData.loading}
        error={schemaData.error}
        userAccess={schemaData.userAccess}
      />

      <OrthophotoPanel
        isVisible={activeTool === "ortho"}
        onClose={() => setActiveTool(null)}
        initialData={orthophotoData}
        onSave={handleOrthophotoSave}
      />

      <AdminBoundariesPanel
        isVisible={activeTool === "admin"}
        onClose={() => setActiveTool(null)}
      />

      {/* 🔄 JoinedTable Sync Panel */}
      <JoinedTableSyncPanel
        isVisible={activeTool === "sync"}
        onClose={() => setActiveTool(null)}
      />
    </>
  );
}

export default RightControls;
