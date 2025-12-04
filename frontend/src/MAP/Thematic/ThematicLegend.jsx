import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import "./thematic.css";

const ThematicLegend = () => {
  const [legends, setLegends] = useState({});

  useEffect(() => {
    window.addThematicLegend = (id, jsx) => {
      setLegends(prev => ({ ...prev, [id]: jsx }));
    };

    window.removeThematicLegend = (id) => {
      setLegends(prev => {
        const updated = { ...prev };
        delete updated[id];
        return updated;
      });
    };
  }, []);

  const activeLegends = Object.values(legends);
  if (activeLegends.length === 0) return null;

  // ✅ Render into body (outside toolbar)
  return createPortal(
    <div className="thematic-legend-container">
      {activeLegends.map((LegendBlock, idx) => (
        <div className="thematic-legend-box" key={idx}>
          {LegendBlock}
        </div>
      ))}
    </div>,
    document.body
  );
};

export default ThematicLegend;
