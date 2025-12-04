import L from "leaflet";

// Lightweight WMTS support for Leaflet
L.TileLayer.WMTS = L.TileLayer.extend({
  defaultWmtsParams: {
    service: "WMTS",
    request: "GetTile",
    version: "1.0.0",
    layer: "",
    style: "",
    tilematrixSet: "",
    format: "image/png"
  },

  initialize: function (url, options) {
    this._url = url;
    const wmtsParams = L.extend({}, this.defaultWmtsParams);

    for (const i in options) {
      if (!(i in this.options)) {
        wmtsParams[i] = options[i];
      }
    }

    this.wmtsParams = wmtsParams;
    L.setOptions(this, options);
  },

  getTileUrl: function (coords) {
    return (
      this._url +
      "?service=WMTS" +
      "&request=GetTile" +
      "&version=" + this.wmtsParams.version +
      "&layer=" + this.wmtsParams.layer +
      "&style=" + this.wmtsParams.style +
      "&tilematrixSet=" + this.wmtsParams.tilematrixSet +
      "&format=" + this.wmtsParams.format +
      "&tilematrix=" + this.wmtsParams.tilematrixSet + ":" + coords.z +
      "&tilerow=" + coords.y +
      "&tilecol=" + coords.x
    );
  }
});

L.tileLayer.wmts = function (url, options) {
  return new L.TileLayer.WMTS(url, options);
};
