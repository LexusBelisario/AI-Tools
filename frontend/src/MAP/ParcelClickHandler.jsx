import { useEffect } from "react";
import API from "../api_service";
import { useSchema } from "./SchemaContext";

const ParcelClickHandler = ({
  activeTool,
  setInfoProps,
  setInfoVisible,
  setAttributeEditMode,
  setEditHeader,
  onConsolidateSelect,
}) => {
  const { schema } = useSchema();

  useEffect(() => {
    const map = window.map;
    if (!map) return;

    // === Clear bindings and highlights when inactive ===
    if (!["info", "edit", "consolidate", "subdivide"].includes(activeTool)) {
      console.log("⛔ ParcelClickHandler inactive:", activeTool);

      const zoom = map?.getZoom?.() ?? 0;
      const visible = zoom >= 16;

      if (window.parcelLayers?.length) {
        window.parcelLayers.forEach(({ layer }) => {
          layer.off("click");
          layer.setStyle?.({
            stroke: true,
            color: "black",
            weight: 1,
            opacity: visible ? 1 : 0,
            fill: true,
            fillColor: "white",
            fillOpacity: visible ? 0.1 : 0,
          });
        });
      }

      if (window.clearHighlight) window.clearHighlight();
      if (window.clearConsolidateHighlights) window.clearConsolidateHighlights();

      if (window.onParcelsLoaded) window.onParcelsLoaded();
      if (window.enforceLayerOrder) window.enforceLayerOrder();
      return;
    }

    // ============================================================
    // ✅ Bind Clicks for Active Tools
    // ============================================================
    const bindClicks = () => {
      if (!window.parcelLayers?.length) {
        console.log("⏳ Waiting for parcel layers...");
        setTimeout(bindClicks, 500);
        return;
      }

      console.log(`✅ Binding clicks for tool: ${activeTool}`);

      window.parcelLayers.forEach(({ feature, layer }) => {
        layer.off("click");

        layer.on("click", async () => {
          const pin = feature.properties?.pin;
          if (!pin || !schema) return;

          // === Consolidate Mode ===
          if (activeTool === "consolidate") {
            console.log("🔵 Consolidate click:", pin);

            const isAlreadySelected =
              window.consolidateLayer &&
              window.consolidateLayer
                .getLayers()
                .some((l) => l.feature?.properties?.pin === pin);

            if (isAlreadySelected) {
              if (window.removeConsolidateFeature)
                window.removeConsolidateFeature(pin);
              if (onConsolidateSelect)
                onConsolidateSelect(pin, feature, false);
            } else {
              if (window.addConsolidateFeature)
                window.addConsolidateFeature(feature);
              if (onConsolidateSelect)
                onConsolidateSelect(pin, feature, true);
            }
            return;
          }

          // === Subdivide Mode ===
          if (activeTool === "subdivide") {
            if (window.subdivideLocked) {
              console.log("🔒 Subdivide locked – ignoring clicks.");
              return;
            }

            console.log("🟢 Subdivide select:", pin);

            // Reset all parcel styles
            window.parcelLayers.forEach(({ layer }) => {
              layer.setStyle?.({
                stroke: true,
                color: "black",
                weight: 1,
                opacity: 1,
                fill: true,
                fillColor: "white",
                fillOpacity: 0.1,
              });
            });

            // Highlight selected parcel
            layer.setStyle?.({
              stroke: true,
              color: "black",
              weight: 2,
              opacity: 1,
              fill: true,
              fillColor: "green",
              fillOpacity: 0.4,
            });
            layer.bringToFront();

            const selectedParcel = {
              ...feature.properties,
              pin,
              source_table: feature.properties?.source_table || "LandParcels",
              source_schema: schema,
            };

            window.selectedParcelForSubdivide = selectedParcel;
            if (window.setSelectedParcelForSubdivide)
              window.setSelectedParcelForSubdivide(selectedParcel);

            console.log("✅ Selected parcel for subdivide:", selectedParcel);
            return;
          }

          // === Info / Edit Mode ===
          if (activeTool === "info" || activeTool === "edit") {
            console.log("🟡 Info/Edit click:", pin);
            try {
              const res = await fetch(
                `${API}/parcel-info?pin=${encodeURIComponent(pin)}&schema=${schema}`
              );
              const json = await res.json();

              if (json.status === "success") {
                if (window.highlightFeature) {
                  window.highlightFeature(feature);
                }

                const parcelData = {
                  ...json.data,
                  pin,
                  source_table: feature.properties?.source_table || "LandParcels",
                  source_schema: schema,
                };

                setInfoProps(parcelData);
                setInfoVisible(true);
                setAttributeEditMode(activeTool === "edit");
                setEditHeader(
                  activeTool === "edit"
                    ? "Parcel Attribute Editing Tool"
                    : "Land Parcel Information"
                );

                window.currentParcelLayer = layer;
              } else {
                console.error("❌ Parcel not found:", json.message);
              }
            } catch (err) {
              console.error("❌ Fetch error:", err);
            }
          }
        });
      });
    };

    bindClicks();

    // ============================================================
    // 🧭 Persist Subdivide Highlight after Zoom / Move
    // ============================================================
    const handleMapChange = () => {
      if (activeTool !== "subdivide") return;
      const selected = window.selectedParcelForSubdivide;
      if (!selected) return;

      const match = window.parcelLayers?.find(
        (p) => p.feature?.properties?.pin === selected.pin
      );
      if (match?.layer) {
        match.layer.setStyle?.({
          stroke: true,
          color: "black",
          weight: 2,
          opacity: 1,
          fill: true,
          fillColor: "green",
          fillOpacity: 0.4,
        });
        match.layer.bringToFront();
      }
    };

    map.on("moveend", handleMapChange);

    return () => {
      map.off("moveend", handleMapChange);
    };
  }, [activeTool, schema]);

  return null;
};

export default ParcelClickHandler;
