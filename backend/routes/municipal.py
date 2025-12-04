from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from shapely.geometry import box
from auth.dependencies import get_user_main_db
from license.license_validator import license_validator

router = APIRouter()

# ==========================================================
# 🔍 Helper Function — Get extent from any geometry table
# ==========================================================
def get_schema_extent(schema: str, db: Session):
    """Return (box, table_name) from the first table that contains geometry."""
    # 1️⃣ Find all tables with a 'geom' column
    tables_query = text("""
        SELECT table_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND column_name = 'geom'
    """)
    tables = [row[0] for row in db.execute(tables_query, {"schema": schema})]
    print(f"🔍 Geometry tables detected in {schema}: {tables}")

    if not tables:
        return None, None

    # 2️⃣ Compute extent from first valid table
    for table in tables:
        try:
            result = db.execute(
                text(f'SELECT ST_Extent(geom)::text AS bbox FROM "{schema}"."{table}"')
            ).mappings().first()
            if result and result["bbox"]:
                bbox_text = result["bbox"].replace("BOX(", "").replace(")", "")
                xmin, ymin, xmax, ymax = map(float, bbox_text.replace(",", " ").split())
                extent_box = box(xmin, ymin, xmax, ymax)
                print(f"✅ Found extent from {table}: {extent_box.bounds}")
                return extent_box, table
        except Exception as e:
            print(f"⚠️ Failed to read extent from {table}: {e}")

    return None, None


# ==========================================================
# 🗺️ MUNICIPAL BOUNDS (auto-detect any geometry source)
# ==========================================================
@router.get("/municipal-bounds")
def get_municipal_bounds(schema: str, db: Session = Depends(get_user_main_db)):
    """
    Dynamically computes bounding box from the first available table
    that contains a 'geom' column (Barangay, Section, or Parcels).
    Performs license validation using buffered license polygon.
    """
    try:
        current_db = db.execute(text("SELECT current_database()")).scalar()
        print(f"\n📌 Connected to DB={current_db}, schema={schema}")

        extent_box, used_table = get_schema_extent(schema, db)
        if not extent_box:
            return {
                "status": "invalid_bounds",
                "message": f"No geometry found in schema '{schema}' for extent computation."
            }

        # 🔍 License intersection check
        print(f"🔍 LICENSE BOUNDS: {license_validator.buffered_poly.bounds}")
        intersects = extent_box.intersects(license_validator.buffered_poly)
        print(f"🔍 INTERSECTS? {intersects}")

        if not intersects:
            return {
                "status": "invalid_bounds",
                "message": (
                    f"Invalid location — schema '{schema}' is outside "
                    f"licensed LGU area ({license_validator.license_name})."
                ),
                "license_name": license_validator.license_name
            }

        print(f"✅ License validated for {schema} (LGU: {license_validator.license_name})")
        return {
            "status": "success",
            "bounds": list(extent_box.bounds),
            "source_table": used_table,
            "license_applied": True,
            "license_name": license_validator.license_name
        }

    except Exception as e:
        print(f"❌ Error computing municipal bounds for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# 🧭 MUNICIPAL BOUNDARIES (with license check)
# ==========================================================
@router.get("/municipal-boundaries")
def get_municipal_boundaries(schema: str, db: Session = Depends(get_user_main_db)):
    """
    Fetch Barangay and Section boundaries directly from the schema,
    but automatically validates license coverage using any geometry table.
    """
    try:
        current_db = db.execute(text("SELECT current_database()")).scalar()
        print(f"\n📌 Connected to DB={current_db}, schema={schema} (GET municipal-boundaries)")

        extent_box, used_table = get_schema_extent(schema, db)
        if not extent_box:
            return {
                "status": "invalid_bounds",
                "message": f"No geometry found in schema '{schema}' for license check."
            }

        intersects = extent_box.intersects(license_validator.buffered_poly)
        print(f"🔍 License intersection check: {intersects}")

        if not intersects:
            return {
                "status": "invalid_bounds",
                "message": (
                    f"Invalid location — schema '{schema}' is outside "
                    f"licensed LGU area ({license_validator.license_name})."
                ),
                "license_name": license_validator.license_name
            }

        print(f"✅ License validated for {schema}. Loading boundaries...")

        results = {"barangay": None, "section": None}

        # --- Fetch Barangay boundaries ---
        try:
            barangay_query = text(f'''
                SELECT *, ST_AsGeoJSON(geom)::json AS geometry
                FROM "{schema}"."BarangayBoundary"
            ''')
            barangay_rows = db.execute(barangay_query).mappings().all()
            results["barangay"] = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": row["geometry"],
                        "properties": {k: v for k, v in row.items() if k not in ("geom", "geometry")}
                    }
                    for row in barangay_rows
                ]
            }
            print(f"✅ Loaded {len(barangay_rows)} barangay features.")
        except Exception as e:
            print(f"⚠️ BarangayBoundary fetch failed: {e}")

        # --- Fetch Section boundaries ---
        try:
            section_query = text(f'''
                SELECT *, ST_AsGeoJSON(geom)::json AS geometry
                FROM "{schema}"."SectionBoundary"
            ''')
            section_rows = db.execute(section_query).mappings().all()
            results["section"] = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": row["geometry"],
                        "properties": {k: v for k, v in row.items() if k not in ("geom", "geometry")}
                    }
                    for row in section_rows
                ]
            }
            print(f"✅ Loaded {len(section_rows)} section features.")
        except Exception as e:
            print(f"⚠️ SectionBoundary fetch failed: {e}")

        return {
            "status": "success",
            **results,
            "license_applied": True,
            "license_name": license_validator.license_name,
            "extent_source": used_table
        }

    except Exception as e:
        print(f"❌ Error fetching municipal boundaries for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
