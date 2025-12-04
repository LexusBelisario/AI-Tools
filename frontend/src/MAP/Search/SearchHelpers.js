// searchHelpers.js

// Reset all parcels to default black style
export const clearParcelHighlights = () => {
  window.parcelLayers?.forEach(({ layer }) => {
    layer.setStyle({ color: "black", weight: 1, fillColor: "black", fillOpacity: 0.1 });
  });
  window.currentParcelLayer = null;
};

// Restore original click events after search closes
export const resetParcelLayerClicks = () => {
  window.parcelLayers?.forEach(({ layer, feature }) => {
    layer.off("click");
    layer.on("click", () => {
      if (window.populateParcelInfo) {
        window.populateParcelInfo(feature.properties);
      }
    });
  });
};
