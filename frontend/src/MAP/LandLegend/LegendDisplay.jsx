// LegendDisplay.jsx
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import "./LandLegendTools.css";

const LegendDisplay = () => {
  const [legends, setLegends] = useState({});

  useEffect(() => {
    window.addLandInfoLegend = (id, jsx) => {
      setLegends(prev => ({ ...prev, [id]: jsx }));
    };
    window.removeLandInfoLegend = (id) => {
      setLegends(prev => {
        const updated = { ...prev };
        delete updated[id];
        return updated;
      });
    };
    return () => {
      window.addLandInfoLegend = null;
      window.removeLandInfoLegend = null;
    };
  }, []);

  const activeLegends = Object.values(legends);
  if (activeLegends.length === 0) return null;

  return createPortal(
    <div className="landinfo-legend-container">
      {activeLegends.map((LegendBlock, idx) => (
        <div className="landinfo-legend-box" key={idx}>
          {LegendBlock} {/* ✅ directly rendered, no white wrapper */}
        </div>
      ))}
    </div>,
    document.body
  );
};

export default LegendDisplay;
