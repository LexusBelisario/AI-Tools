import React, { useState } from "react";
import ReactDOM from "react-dom";
import "./RemoveLandmark.css";

const RemoveLandmark = ({
  visible,
  removalList,
  onCancel,
  onRemoveSelected,
  onUnmark
}) => {
  const [step, setStep] = useState("default"); // "default", "confirm", "success"

  if (!visible) return null;

  const handleInitialClick = () => {
    if (removalList.length > 0) {
      setStep("confirm");
    }
  };

  const handleConfirm = () => {
    onRemoveSelected();
    setStep("success");
  };

  const handleContinue = () => {
    setStep("default");
  };

  const panel = (
    <div className="remove-landmark-panel leaflet-top leaflet-right">
      <div className="remove-landmark-header">
        <h3 className="remove-landmark-title">Remove Landmark</h3>
        <button onClick={onCancel} className="remove-landmark-close">X</button>
      </div>

      {step === "success" ? (
        <div className="remove-landmark-confirm">
          <p className="remove-landmark-success">✅ Landmarks successfully deleted.</p>
          <button className="remove-landmark-btn" onClick={handleContinue}>OK</button>
        </div>
      ) : step === "confirm" ? (
        <div className="remove-landmark-confirm">
          <p className="remove-landmark-warning">
            Are you sure you want to delete {removalList.length} landmark(s)?
          </p>
          <div className="remove-landmark-footer">
            <button className="remove-landmark-btn" onClick={handleConfirm}>Yes, Delete</button>
            <button className="remove-landmark-btn" onClick={() => setStep("default")}>Cancel</button>
          </div>
        </div>
      ) : (
        <>
          <div className="remove-landmark-instructions">
            Click landmarks to mark them for removal. They’ll appear in the list below.
          </div>

          <div className="remove-landmark-list">
            {removalList.length === 0 ? (
              <div className="remove-landmark-empty">No landmarks selected.</div>
            ) : (
              removalList.map((item, index) => (
                <div key={item.id || index} className="remove-landmark-item">
                  <img
                    src={getIconPath(item.type)}
                    alt={item.type}
                    className="remove-landmark-icon"
                    onError={(e) => {
                      e.currentTarget.src = "/icons/LandmarkIcons/default.svg";
                    }}
                  />
                  <div className="remove-landmark-labels">
                    <div className="remove-landmark-name">{item.name}</div>
                    <div className="remove-landmark-sub">{item.barangay}</div>
                  </div>
                  <button
                    onClick={() => onUnmark(item.id)}
                    className="remove-landmark-unmark"
                  >
                    ✕
                  </button>
                </div>
              ))
            )}
          </div>

          <div className="remove-landmark-footer">
            <button onClick={handleInitialClick} className="remove-landmark-btn">Remove Selected</button>
            <button onClick={onCancel} className="remove-landmark-btn">Cancel</button>
          </div>
        </>
      )}
    </div>
  );

  return ReactDOM.createPortal(panel, document.body); // ✅ portal outside toolbar
};

// Icon Mapper
function getIconPath(type) {
  const icons = {
    "commercial entities": "Commercial Entities.svg",
    "educational entities": "Educational Entities.svg",
    "financial entities": "Financial Entities.svg",
    "fire station": "Fire Station.svg",
    "gas station": "Gas Station.svg",
    "government entities": "Government Entities.svg",
    "industrial entities": "Industrial Entities.svg",
    "medical entities": "Medical Entities.svg",
    "police station": "Police Station.svg",
    "recreational entities": "Recreational Entities.svg",
    "religious entities": "Religious Entities.svg",
    "subdivision": "Subdivision.svg",
    "telecommunication entities": "Telecommunication Entities.svg",
    "transportation entities": "Transportation Entities.svg",
  };
  const file = icons[type?.toLowerCase()] || "default.svg";
  return `/icons/LandmarkIcons/${file}`;
}

export default RemoveLandmark;
