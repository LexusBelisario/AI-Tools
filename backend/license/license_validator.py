import json
import os
from shapely.geometry import shape, box
from shapely.ops import transform
import pyproj
from fastapi import HTTPException

LICENSE_PATH = os.path.join(os.path.dirname(__file__), "license.json")


class LicenseValidator:
    """Central validator for bounding-box-based LGU pseudo-license."""

    def __init__(self):
        if not os.path.exists(LICENSE_PATH):
            raise FileNotFoundError(f"❌ license.json not found at {LICENSE_PATH}")

        with open(LICENSE_PATH, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # --- Parse basic info ---
        self.lgu_code = self.data.get("LGU_CODE")
        self.prov_code = self.data.get("PROV_CODE")
        self.license_name = self.data.get("LICENSE_NAME")
        self.buffer_km = float(self.data.get("BUFFER_KM", 5))
        self.bounds = [float(x) for x in self.data.get("ALLOWED_BOUNDS", [])]

        if len(self.bounds) != 4:
            raise ValueError("Invalid ALLOWED_BOUNDS format in license.json")

        # --- Build buffered polygon ---
        xmin, ymin, xmax, ymax = self.bounds
        poly = box(xmin, ymin, xmax, ymax)
        to_m = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
        to_deg = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
        poly_m = transform(to_m, poly)
        buffered_m = poly_m.buffer(self.buffer_km * 1000)
        self.buffered_poly = transform(to_deg, buffered_m)

        print(f"✅ LicenseValidator initialized for {self.license_name}")
        print(f"   LGU_CODE: {self.lgu_code}")
        print(f"   Allowed Bounds: {self.bounds}")
        print(f"   Buffered (km): {self.buffer_km}")

    # --------------------------------------------------------------
    # 🔹 Check if a GeoJSON geometry or [lon, lat] point is allowed
    # --------------------------------------------------------------
    def is_within_bounds(self, geometry):
        """
        Accepts GeoJSON geometry dict or simple [lon, lat] coordinate list.
        Returns True if geometry is fully inside buffered license polygon.
        """
        if isinstance(geometry, (list, tuple)) and len(geometry) == 2:
            lon, lat = geometry
            geom = shape({"type": "Point", "coordinates": [lon, lat]})
        elif isinstance(geometry, dict):
            geom = shape(geometry)
        else:
            raise ValueError("Unsupported geometry format for license validation.")

        return self.buffered_poly.contains(geom)

    # --------------------------------------------------------------
    # 🔹 SQL helper to add WHERE clause for filtering
    # --------------------------------------------------------------
    def sql_filter_clause(self, geom_column="geom"):
        """Return SQL fragment restricting geometries to license bounds."""
        xmin, ymin, xmax, ymax = self.bounds
        return (
            f"ST_Intersects({geom_column}, "
            f"ST_Buffer(ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, 4326)::geography, {self.buffer_km * 1000})::geometry)"
        )

    # --------------------------------------------------------------
    # 🔹 FastAPI-friendly enforcement
    # --------------------------------------------------------------
    def enforce_geometry(self, geometry):
        """Raise HTTPException(403) if geometry is outside licensed area."""
        if not self.is_within_bounds(geometry):
            raise HTTPException(
                status_code=403,
                detail=f"Operation outside licensed LGU bounds ({self.license_name})."
            )


# ==============================================================
# Create global singleton (loads once at startup)
# ==============================================================
try:
    license_validator = LicenseValidator()
except Exception as e:
    print(f"⚠️ LicenseValidator failed to initialize: {e}")
    license_validator = None
