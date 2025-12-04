import React, { useRef, useEffect } from "react";
import { ChevronRight } from "lucide-react";

const SchemaSelectorPanel = ({
  isVisible,
  onClose,
  schemas,
  selectedSchema,
  onSchemaChange,
  loading,
  error,
  userAccess,
}) => {
  const containerRef = useRef(null);
  const scrollableRef = useRef(null);

  // 🧩 Prevent map zoom & drag interference - but allow scrolling inside the list
  useEffect(() => {
    if (!containerRef.current) return;

    const handleWheel = (e) => {
      if (scrollableRef.current && scrollableRef.current.contains(e.target)) {
        e.stopPropagation();
      } else {
        e.stopPropagation();
        e.preventDefault();
      }
    };

    const stopEvent = (e) => e.stopPropagation();
    const el = containerRef.current;

    el.addEventListener("wheel", handleWheel, { passive: false });
    el.addEventListener("dblclick", stopEvent);

    return () => {
      el.removeEventListener("wheel", handleWheel);
      el.removeEventListener("dblclick", stopEvent);
    };
  }, []);

  if (!isVisible) return null;

  return (
    <div
      ref={containerRef}
      className="absolute top-1/2 -translate-y-1/2 right-14 w-[260px] bg-[#151922] text-white rounded-l-lg shadow-xl border border-[#2A2E35] animate-slideIn z-[9999]"
      style={{ pointerEvents: "auto" }}
    >
      {/* Header */}
      <div className="flex justify-between items-center bg-[#F7C800] text-black font-semibold px-4 py-2 rounded-tl-lg">
        <span>Select Municipality / City</span>
        <button
          className="text-black hover:opacity-70 transition"
          onClick={onClose}
        >
          <ChevronRight size={20} />
        </button>
      </div>

      {/* Body */}
      <div className="p-4 text-sm space-y-3">
        {loading && <p className="text-gray-400">Loading schemas...</p>}
        {error && <p className="text-red-400">{error}</p>}

        {!loading && !error && schemas.length > 0 && (
          <>
            {userAccess && (
              <div className="text-xs text-gray-300 mb-2">
                Access Level:{" "}
                <span className="text-white font-semibold">
                  {userAccess.description || "Full Municipal Access"}
                </span>
              </div>
            )}

            <ul
              ref={scrollableRef}
              className="space-y-2 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar"
            >
              {schemas.map((schema) => (
                <li
                  key={schema}
                  className="hover:text-[#F7C800] transition cursor-pointer"
                >
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      value={schema}
                      checked={selectedSchema === schema}
                      onChange={(e) => onSchemaChange(e.target.value)}
                      className="accent-[#F7C800]"
                    />
                    <span className="schema-name">{schema}</span>
                  </label>
                </li>
              ))}
            </ul>

            {selectedSchema && (
              <div className="text-xs text-gray-300 border-t border-[#2A2E35] pt-2 mt-2">
                Current:{" "}
                <span className="font-semibold text-[#F7C800]">
                  {selectedSchema}
                </span>
              </div>
            )}
          </>
        )}

        {!loading && !error && schemas.length === 0 && (
          <div className="text-gray-400 text-center py-4">
            <p>No schemas available</p>
          </div>
        )}
      </div>

      {/* ✅ FIXED: remove 'jsx' attribute */}
      <style>
        {`
          .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
          }

          .custom-scrollbar::-webkit-scrollbar-track {
            background: #1a1d26;
            border-radius: 3px;
          }

          .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #2a2e35;
            border-radius: 3px;
          }

          .custom-scrollbar::-webkit-scrollbar-thumb:hover {
            background: #f7c800;
          }

          /* Firefox */
          .custom-scrollbar {
            scrollbar-width: thin;
            scrollbar-color: #2a2e35 #1a1d26;
          }
        `}
      </style>
    </div>
  );
};

export default SchemaSelectorPanel;
