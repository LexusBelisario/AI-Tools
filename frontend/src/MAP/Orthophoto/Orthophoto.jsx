import { useEffect, useState } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";
import "./LeafletWMTS";
import { ApiService } from "../../api_service";
import { useSchema } from "../SchemaContext";

function Orthophoto() {
  const map = useMap();
  const { schema } = useSchema();

  const [orthoData, setOrthoData] = useState({ Gsrvr_URL: "", Layer_Name: "" });
  const [orthoLayer, setOrthoLayer] = useState(null);
  const [orthoVisible, setOrthoVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  // ==========================================================
  // 🧭 Load configuration when schema changes
  // ==========================================================
  useEffect(() => {
    if (!schema) {
      setOrthoData({ Gsrvr_URL: "", Layer_Name: "" });
      setMessage("");
      // Hide orthophoto if schema cleared
      if (orthoLayer && map.hasLayer(orthoLayer)) {
        map.removeLayer(orthoLayer);
        setOrthoVisible(false);
      }
      return;
    }

    const fetchConfig = async () => {
      setLoading(true);
      try {
        const res = await ApiService.get(`/orthophoto-config?schema=${schema}`);
        if (res.status === "success") {
          setOrthoData({
            Gsrvr_URL: res.Gsrvr_URL || "",
            Layer_Name: res.Layer_Name || "",
          });
          setMessage("Loaded orthophoto configuration.");
          console.log(`✅ Loaded orthophoto config for ${schema}`);
        } else {
          setOrthoData({ Gsrvr_URL: "", Layer_Name: "" });
          setMessage("No configuration found. Please set one.");
          console.warn(`⚠️ No orthophoto config found for ${schema}`);
        }
      } catch (err) {
        console.error("❌ Failed to load orthophoto config:", err);
        setMessage("Failed to load configuration.");
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, [schema, map, orthoLayer]);

  // ==========================================================
  // 🛰️ Create/Update Orthophoto Layer when config changes
  // ==========================================================
  useEffect(() => {
    if (!map || !orthoData.Gsrvr_URL || !orthoData.Layer_Name) {
      // Remove layer if config is cleared
      if (orthoLayer && map.hasLayer(orthoLayer)) {
        map.removeLayer(orthoLayer);
        setOrthoVisible(false);
      }
      return;
    }

    // Remove old layer if exists
    if (orthoLayer) {
      if (map.hasLayer(orthoLayer)) {
        map.removeLayer(orthoLayer);
      }
    }

    // Create new WMTS layer
    const wmtsLayer = L.tileLayer.wmts(orthoData.Gsrvr_URL, {
      layer: orthoData.Layer_Name,
      tilematrixSet: "EPSG:900913",
      format: "image/png",
      style: "",
      maxZoom: 24,
    });

    setOrthoLayer(wmtsLayer);
    window._orthoLayer = wmtsLayer; // Store globally for basemap switcher

    // If was visible before, show new layer
    if (orthoVisible) {
      wmtsLayer.addTo(map).bringToFront();
    }

    console.log(`🛰️ Orthophoto layer ready: ${orthoData.Layer_Name}`);

    return () => {
      if (wmtsLayer && map.hasLayer(wmtsLayer)) {
        map.removeLayer(wmtsLayer);
      }
    };
  }, [map, orthoData.Gsrvr_URL, orthoData.Layer_Name]);

  // ==========================================================
  // 🌍 Toggle Orthophoto Visibility
  // ==========================================================
  const toggleOrthoVisibility = () => {
    if (!map || !orthoLayer) {
      setMessage("Orthophoto layer not available.");
      return;
    }

    if (orthoVisible) {
      map.removeLayer(orthoLayer);
      setOrthoVisible(false);
      console.log("🛰️ Orthophoto hidden");
    } else {
      orthoLayer.addTo(map).bringToFront();
      setOrthoVisible(true);
      console.log("🛰️ Orthophoto shown");
    }
  };

  // ==========================================================
  // 💾 Save configuration to backend
  // ==========================================================
  const handleSave = async (data) => {
    if (!schema || !data.Gsrvr_URL || !data.Layer_Name) {
      setMessage("Please fill in both fields before saving.");
      return { success: false, message: "Please fill in both fields." };
    }

    try {
      setLoading(true);
      setMessage("Saving...");
      const payload = { schema, ...data };
      const res = await ApiService.post("/orthophoto-config", payload);
      
      if (res.status === "success") {
        setMessage("Configuration saved successfully!");
        setOrthoData(data);
        return { success: true, message: "Configuration saved successfully!" };
      } else {
        setMessage("Error saving configuration.");
        return { success: false, message: "Error saving configuration." };
      }
    } catch (err) {
      console.error("Save failed:", err);
      setMessage("Save failed.");
      return { success: false, message: "Save failed." };
    } finally {
      setLoading(false);
    }
  };

  // ==========================================================
  // ✅ Expose state and handlers to window for UI panel
  // ==========================================================
  useEffect(() => {
    window._orthophotoData = {
      Gsrvr_URL: orthoData.Gsrvr_URL,
      Layer_Name: orthoData.Layer_Name,
      orthoVisible,
      hasConfig: !!(orthoData.Gsrvr_URL && orthoData.Layer_Name),
      loading,
      message,
      schema,
    };
  }, [orthoData, orthoVisible, loading, message, schema]);

  useEffect(() => {
    window._handleOrthophotoSave = handleSave;
    window._toggleOrthoVisibility = toggleOrthoVisibility;
    
    return () => {
      delete window._handleOrthophotoSave;
      delete window._toggleOrthoVisibility;
      delete window._orthophotoData;
      delete window._orthoLayer;
    };
  }, [schema, orthoLayer, orthoVisible]);

  return null; // No UI, logic only
}

export default Orthophoto;