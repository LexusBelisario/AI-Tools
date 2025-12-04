import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom";
import L from "leaflet";
import API from "../../api.js";
import "./AddLandmarks.css";
import { useSchema } from "../SchemaContext";
import { useMap } from "react-leaflet";

const AddLandmark = ({ visible, onClose, onAdded }) => {
  const { schema } = useSchema();
  const map = useMap(); // ✅ use react-leaflet map instance
  const [marker, setMarker] = useState(null);
  const [form, setForm] = useState({
    name: "",
    type: "",
    barangay: "",
    descr: "",
    latlng: null,
  });

  useEffect(() => {
    if (!visible || !map) return;

    const handleClick = async (e) => {
      const latlng = e.latlng;
      console.log("🟢 Map clicked at:", latlng);

      if (marker) {
        map.removeLayer(marker);
        setMarker(null);
      }

      const barangay = await getBarangayFromBoundary(latlng, schema);
      console.log("📍 Barangay found:", barangay);

      const newMarker = L.marker(latlng).addTo(map);
      setMarker(newMarker);

      setForm((prev) => ({
        ...prev,
        latlng,
        barangay,
      }));
    };

    console.log("🔗 Attaching map click listener for AddLandmark");
    map.on("click", handleClick);

    return () => {
      console.log("🔗 Removing map click listener for AddLandmark");
      map.off("click", handleClick);
    };
  }, [marker, schema, visible, map]);

  const getBarangayFromBoundary = async (latlng, schema) => {
    try {
      const res = await fetch(`${API}/find-barangay`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ schema, lat: latlng.lat, lng: latlng.lng }),
      });
      console.log("📡 Barangay response status:", res.status);
      const data = await res.json();
      console.log("📦 Barangay response JSON:", data);
      return data?.barangay || "";
    } catch (err) {
      console.error("❌ Error fetching barangay:", err);
      return "";
    }
  };

  const handleSave = async () => {
    if (!form.latlng || !form.name || !form.type || !form.barangay) {
      alert("Please fill in all required fields and click on the map.");
      return;
    }

    const payload = {
      schema,
      name: form.name,
      type: form.type,
      barangay: form.barangay,
      descr: form.descr || null,
      geom: {
        type: "Point",
        coordinates: [form.latlng.lng, form.latlng.lat],
      },
    };

    console.log("📤 Saving landmark with payload:", payload);

    try {
      const res = await fetch(`${API}/landmarks/insert`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      console.log("📡 Save response status:", res.status);
      const result = await res.json();
      console.log("📦 Save response JSON:", result);

      if (res.ok && result.status === "success") {
        alert("✅ Landmark saved.");
        if (marker) {
          map.removeLayer(marker);
        }
        setMarker(null);
        setForm({ name: "", type: "", barangay: "", descr: "", latlng: null });

        onAdded?.(); // 🔄 trigger refresh
        onClose();
      } else {
        alert("❌ Failed to save landmark.");
      }
    } catch (err) {
      alert("Error saving landmark.");
      console.error("❌ Save error:", err);
    }
  };

  const closePopup = () => {
    if (marker) {
      map.removeLayer(marker);
    }
    setMarker(null);
    setForm({ name: "", type: "", barangay: "", descr: "", latlng: null });
    onClose();
  };

  if (!visible) return null;

  const panel = (
    <div className="add-landmark-panel leaflet-top leaflet-right">
      <div className="add-landmark-header">
        <h3 className="add-landmark-title">Add Landmark</h3>
        <button onClick={closePopup} className="add-landmark-close">
          ×
        </button>
      </div>

      <div className="add-landmark-instructions">
        <ol>
          <li>Click on any point on the map to add a landmark.</li>
          <li>Fill in the required details.</li>
          <li>
            Click <b>Save</b> to store it.
          </li>
        </ol>
      </div>

      <label className="add-landmark-label">
        Name <span className="required">*</span>:
        <input
          type="text"
          value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })}
          className="add-landmark-input"
        />
      </label>

      <label className="add-landmark-label">
        Type <span className="required">*</span>:
        <select
          value={form.type}
          onChange={(e) => setForm({ ...form, type: e.target.value })}
          className="add-landmark-input"
        >
          <option value="">-- Select --</option>
          <option value="Commercial Entities">Commercial Entities</option>
          <option value="Educational Entities">Educational Entities</option>
          <option value="Medical Entities">Medical Entities</option>
          <option value="Government Entities">Government Entities</option>
          <option value="Police Station">Police Station</option>
          <option value="Fire Station">Fire Station</option>
          <option value="Gas Station">Gas Station</option>
          <option value="Religious Entities">Religious Entities</option>
          <option value="Recreational Entities">Recreational Entities</option>
          <option value="Telecommunication Entities">
            Telecommunication Entities
          </option>
          <option value="Transportation Entities">
            Transportation Entities
          </option>
          <option value="Financial Entities">Financial Entities</option>
          <option value="Industrial Entities">Industrial Entities</option>
          <option value="Subdivision">Subdivision</option>
        </select>
      </label>

      <label className="add-landmark-label">
        Barangay <span className="required">*</span>:
        <input
          type="text"
          value={form.barangay}
          onChange={(e) => setForm({ ...form, barangay: e.target.value })}
          className="add-landmark-input"
        />
      </label>

      <label className="add-landmark-label">
        Description:
        <textarea
          value={form.descr}
          onChange={(e) => setForm({ ...form, descr: e.target.value })}
          rows={3}
          className="add-landmark-input"
        />
      </label>

      <div className="add-landmark-footer">
        <button onClick={handleSave} className="add-landmark-save">
          Save
        </button>
        <button onClick={closePopup} className="add-landmark-cancel">
          Cancel
        </button>
      </div>
    </div>
  );

  return ReactDOM.createPortal(panel, document.body);
};

export default AddLandmark;
