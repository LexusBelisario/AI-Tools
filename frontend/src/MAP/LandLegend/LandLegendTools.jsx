// LandLegendTools.jsx
import React from "react";
import LandValuation from "./LandValuation.jsx";
import LandOwnership from "./LandOwnership.jsx";
import LandActualUse from "./LandActualUse.jsx";
import LandTaxability from "./LandTaxability.jsx";
import DelinquencyAmount from "./DelinquencyAmount.jsx";
import LegendDisplay from "./LegendDisplay.jsx";

const LandLegendTools = () => {
  return (
    <>
      <LandValuation />
      <LandOwnership />
      <LandActualUse />   {/* ✅ no map/schema props */}
      <LandTaxability />
      <DelinquencyAmount />
      <LegendDisplay />
    </>
  );
};

export default LandLegendTools;
