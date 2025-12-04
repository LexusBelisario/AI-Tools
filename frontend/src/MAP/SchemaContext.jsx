import React, { createContext, useContext, useState, useEffect } from "react";
import { API_BASE } from "../config";

const SchemaContext = createContext();

export const SchemaProvider = ({ children }) => {
  // ============================================================
  // 🔹 Core State
  // ============================================================
  const [schema, setSchema] = useState(null);
  const [joinedTable, setJoinedTable] = useState([]);
  const [loadingJoinedTable, setLoadingJoinedTable] = useState(false);
  const [status, setStatus] = useState("idle");

  // ============================================================
  // 🧠 Auto-fetch JoinedTable when schema changes
  // ============================================================
  useEffect(() => {
    if (!schema) {
      setJoinedTable([]);
      setStatus("idle");
      return;
    }

    const fetchJoinedTable = async () => {
      console.log(`📡 Fetching JoinedTable for schema: ${schema}`);
      setLoadingJoinedTable(true);
      setStatus("loading");

      try {
        const res = await fetch(
          `${API_BASE}/search/attribute-table?schema=${schema}`
        );
        if (!res.ok) throw new Error(`Server responded with ${res.status}`);

        const json = await res.json();
        if (json.status === "success" && Array.isArray(json.data)) {
          setJoinedTable(json.data);
          setStatus("success");
          console.log(
            `✅ JoinedTable loaded successfully (${json.count || json.data.length} records)`
          );
        } else {
          setJoinedTable([]);
          setStatus("error");
          console.warn(
            `⚠️ Failed to load JoinedTable for ${schema}: ${json.message || "Unknown error"}`
          );
        }
      } catch (err) {
        console.error(`❌ Error fetching JoinedTable for ${schema}:`, err);
        setJoinedTable([]);
        setStatus("error");
      } finally {
        setLoadingJoinedTable(false);
      }
    };

    fetchJoinedTable();
  }, [schema]);

  // ============================================================
  // 🔁 Manual reload function
  // ============================================================
  const reloadJoinedTable = async () => {
    if (!schema) return;
    console.log(`🔁 Reloading JoinedTable for schema: ${schema}`);
    setLoadingJoinedTable(true);
    setStatus("loading");

    try {
      const res = await fetch(
        `${API_BASE}/search/attribute-table?schema=${schema}`
      );
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);

      const json = await res.json();
      if (json.status === "success" && Array.isArray(json.data)) {
        setJoinedTable(json.data);
        setStatus("success");
        console.log(
          `✅ JoinedTable reloaded (${json.count || json.data.length} records)`
        );
      } else {
        setJoinedTable([]);
        setStatus("error");
      }
    } catch (err) {
      console.error("❌ Reload error:", err);
      setJoinedTable([]);
      setStatus("error");
    } finally {
      setLoadingJoinedTable(false);
    }
  };

  // ============================================================
  // 🌐 Provide globally
  // ============================================================
  return (
    <SchemaContext.Provider
      value={{
        schema,
        setSchema,
        joinedTable,
        loadingJoinedTable,
        status,
        reloadJoinedTable,
        selectedSchema: schema,
      }}
    >
      {children}
    </SchemaContext.Provider>
  );
};

// Hook for any component to access schema & data
export const useSchema = () => useContext(SchemaContext);
