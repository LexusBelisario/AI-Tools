from fastapi import APIRouter, HTTPException, Query, Depends
from auth.dependencies import get_current_user, get_user_main_db
from auth.models import User
from auth.access_control import AccessControl
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
from license.license_validator import license_validator  # ✅ Added

router = APIRouter()

# ==========================================================
# 🗺️ Load all parcel/road features from selected schemas
# ==========================================================
@router.get("/all-barangays")
def get_all_geom_tables(
    schemas: List[str] = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_user_main_db)
):
    """
    Loads all parcel-like tables (with geom + pin) from the given schemas.
    Adds license-based spatial filtering via license_validator.sql_filter_clause().
    """

    # ✅ Verify user access
    access_info = AccessControl.check_user_access(current_user)
    if access_info["status"] == "pending_approval":
        raise HTTPException(status_code=403, detail=access_info["message"])

    # ✅ Validate which schemas user can access
    invalid_schemas, valid_schemas = [], []
    for schema in schemas:
        if AccessControl.validate_schema_access(schema, current_user):
            valid_schemas.append(schema)
        else:
            invalid_schemas.append(schema)

    if invalid_schemas:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied to schemas: {', '.join(invalid_schemas)}"
        )
    if not valid_schemas:
        raise HTTPException(status_code=403, detail="No valid schemas to query")

    all_features = []
    license_filter = license_validator.sql_filter_clause("t.geom")  # ✅ License bounding box filter

    # ✅ Query tables within each valid schema
    for schema in valid_schemas:
        try:
            result = db.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                      AND column_name IN ('geom', 'pin')
                      AND table_name NOT ILIKE :pattern1
                      AND table_name NOT ILIKE :pattern2
                      AND table_name NOT ILIKE :pattern3
                      AND table_name NOT ILIKE :pattern4
                      AND table_name NOT ILIKE :pattern5
                    GROUP BY table_name
                    HAVING COUNT(DISTINCT column_name) = 2
                """),
                {
                    "schema": schema,
                    "pattern1": "%transaction_log%",
                    "pattern2": "%JoinedTable%",
                    "pattern3": "%CAMA-Table%",
                    "pattern4": "%RunSavedModel1%",
                    "pattern5": "%RunSavedModel2%"
                }
            )
            tables = [row[0] for row in result]
        except Exception as e:
            print(f"❌ Error listing tables in schema '{schema}': {e}")
            continue

        # ✅ Build and execute queries per table
        for table in tables:
            try:
                sql = text(f'''
                    SELECT t."pin", ST_AsGeoJSON(
                        ST_SimplifyPreserveTopology(t.geom, 0.00005)
                    )::json AS geometry
                    FROM "{schema}"."{table}" t
                    WHERE {license_filter}
                ''')
                result = db.execute(sql)
                rows = result.fetchall()

                for row in rows:
                    geom = row[1]
                    if not geom:
                        continue

                    all_features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "pin": row[0],
                            "source_schema": schema,
                            "source_table": table
                        }
                    })
                print(f"✅ Loaded {len(rows)} from {schema}.{table} (license filter applied)")

            except Exception as e:
                print(f"⚠️ Query failed on {schema}.{table}: {e}")
                continue

    return {
        "status": "success",
        "license_name": license_validator.license_name,
        "license_applied": True,
        "type": "FeatureCollection",
        "features": all_features
    }


# ==========================================================
# 📦 Load features from a single table (Landmarks, Roads, Parcels, etc.)
# ==========================================================
@router.get("/single-table")
def get_single_table(
    schema: str,
    table: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_user_main_db)
):
    """
    Loads all features from a single specified table (e.g., Landmarks, Roads, Parcels).
    Applies license bounding-box filtering.
    """

    try:
        license_filter = license_validator.sql_filter_clause("t.geom")  # ✅ License filter
        sql = text(f'''
            SELECT t.*, ST_AsGeoJSON(
                ST_SimplifyPreserveTopology(t.geom, 0.00005)
            )::json AS geometry
            FROM "{schema}"."{table}" t
            WHERE {license_filter}
        ''')
        rows = db.execute(sql).mappings().all()

        features = []
        for row in rows:
            geometry = row.get("geometry")
            if not geometry:
                continue
            props = {k: v for k, v in row.items() if k not in ("geom", "geometry")}
            props["source_table"] = table
            props["source_schema"] = schema

            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": props
            })

        print(f"✅ Loaded {len(features)} from {schema}.{table} (license filter applied)")
        return {
            "status": "success",
            "license_name": license_validator.license_name,
            "license_applied": True,
            "type": "FeatureCollection",
            "features": features
        }

    except Exception as e:
        print(f"❌ Error loading single table {schema}.{table}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
