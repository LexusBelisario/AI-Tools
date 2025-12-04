// src/components/TBLandLegendTools.jsx
import React from "react";
import "./toolbar.css";
import LandLegendTools from "../LandLegend/LandLegendTools.jsx"; // ✅ logic container

const TBLandLegendTools = ({ activeTool, setActiveTool }) => {
  const toggleTool = (tool, callback) => {
    const newTool = activeTool === tool ? null : tool;
    setActiveTool(newTool);
    callback?.();
  };

  return (
    <>
      {/* === Land Legend Buttons === */}
      <button
        className={`tool-button ${activeTool === "landvaluation" ? "active" : ""}`}
        id="btnLandValuation"
        onClick={() => toggleTool("landvaluation", () => window.toggleLandValuationLayer?.())}
        title="Toggle thematic map for Land Valuation"
      >
        <img src="/icons/land_parcel_unitvalue_icon.png" alt="Valuation" />
        <span>Land Parcel: Unit Value</span>
      </button>

      <button
        className={`tool-button ${activeTool === "landownership" ? "active" : ""}`}
        id="btnLandOwnership"
        onClick={() => toggleTool("landownership", () => window.toggleLandOwnershipLayer?.())}
        title="Toggle thematic map for Land Ownership"
      >
        <img src="/icons/landownershipmap.png" alt="Ownership" />
        <span>Land Parcel: Ownership</span>
      </button>

      <button
        className={`tool-button ${activeTool === "taxability" ? "active" : ""}`}
        id="btnTaxability"
        onClick={() => toggleTool("taxability", () => window.toggleTaxabilityLayer?.())}
        title="Toggle layer for Land Parcel Taxability"
      >
        <img src="/icons/land_parcel_taxability_icon.png" alt="Taxability" />
        <span>Land Parcel: Taxability</span>
      </button>

      <button
        className={`tool-button ${activeTool === "actualuse" ? "active" : ""}`}
        id="btnActualUse"
        onClick={() => toggleTool("actualuse", () => window.toggleActualUseLayer?.())}
        title="Toggle layer for Land Parcel Actual Use"
      >
        <img src="/icons/land_parcel_actualuse_icon.png" alt="Actual Use" />
        <span>Land Parcel: Actual Use</span>
      </button>

      <button
        className="tool-button"
        id="btnPayment"
        onClick={() => window.togglePaymentLayer?.()}
        title="Toggle layer for Land Parcel Payment"
      >
        <img src="/icons/land_parcel_payment_icon.png" alt="Payment" />
        <span>Land Parcel: Payment</span>
      </button>

      <button
        className={`tool-button ${activeTool === "delinqamount" ? "active" : ""}`}
        id="btnDelinqAmount"
        onClick={() => toggleTool("delinqamount", () => window.toggleDelinqAmountLayer?.())}
        title="Toggle layer for Delinquency Amount"
      >
        <img src="/icons/delinquency_amount_icon.png" alt="Delinquency Amount" />
        <span>Delinquency: Amount</span>
      </button>

      <button
        className="tool-button"
        id="btnDelinqAging"
        onClick={() => window.toggleDelinqAgingLayer?.()}
        title="Toggle layer for Delinquency Aging"
      >
        <img src="/icons/delinquency_aging_icon.png" alt="Delinquency Aging" />
        <span>Delinquency: Aging</span>
      </button>

      <button
        className="tool-button"
        id="btnDelinqUse"
        onClick={() => window.toggleDelinqUseLayer?.()}
        title="Toggle layer for Delinquency Actual Use"
      >
        <img src="/icons/delinquency_actualuse_icon.png" alt="Delinquency Actual Use" />
        <span>Delinquency: Actual Use</span>
      </button>

      {/* === Logic container mounted in the background === */}
      <LandLegendTools />
    </>
  );
};

export default TBLandLegendTools;
