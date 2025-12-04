from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user, get_user_main_db
from auth.models import User
from auth.access_control import AccessControl
from license.license_validator import license_validator

router = APIRouter(prefix="/taxmaplayout", tags=["Taxmap Layout"])


# ===================================================================
#  GET: /taxmaplayout/barangays?schema=SCHEMA
# ===================================================================
@router.get("/barangays")
def list_barangays(
    schema: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_user_main_db),
):
    if not AccessControl.validate_schema_access(schema, current_user):
        raise HTTPException(403, f"Access denied for schema {schema}")

    license_filter = license_validator.sql_filter_clause("t.geom")

    sql = text(f"""
        SELECT DISTINCT t.brgy_nm
        FROM "{schema}"."JoinedTable" t
        WHERE {license_filter}
        ORDER BY t.brgy_nm ASC
    """)

    rows = db.execute(sql).fetchall()
    barangays = [r[0] for r in rows if r[0]]

    return {
        "status": "success",
        "barangays": barangays,
        "schema": schema
    }


# ===================================================================
#  GET: /taxmaplayout/sections?schema=SCHEMA&barangay=NAME
# ===================================================================
@router.get("/sections")
def list_sections(
    schema: str,
    barangay: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_user_main_db),
):
    if not AccessControl.validate_schema_access(schema, current_user):
        raise HTTPException(403, f"Access denied for schema {schema}")

    license_filter = license_validator.sql_filter_clause("s.geom")

    sql = text(f"""
        SELECT 
            s.section,
            ST_AsGeoJSON(s.geom)::json AS geometry
        FROM "{schema}"."SectionBoundary" s
        WHERE s.barangay = :barangay
          AND {license_filter}
        ORDER BY s.section ASC
    """)

    rows = db.execute(sql, {"barangay": barangay}).mappings().all()

    sections = [
        {
            "section": r["section"],
            "geometry": r["geometry"],
        }
        for r in rows
    ]

    return {
        "status": "success",
        "schema": schema,
        "barangay": barangay,
        "count": len(sections),
        "sections": sections,
    }


# ===================================================================
#  GET: /taxmaplayout/section-boundary?schema=SCHEMA&barangay=NAME
# ===================================================================
@router.get("/section-boundary")
def get_section_boundary(
    schema: str,
    barangay: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_user_main_db),
):
    if not AccessControl.validate_schema_access(schema, current_user):
        raise HTTPException(403, f"Access denied for schema {schema}")

    license_filter = license_validator.sql_filter_clause("s.geom")

    sql = text(f"""
        SELECT 
            s.section,
            ST_AsGeoJSON(s.geom)::json AS geometry
        FROM "{schema}"."SectionBoundary" s
        WHERE s.barangay = :barangay
          AND {license_filter}
    """)

    rows = db.execute(sql, {"barangay": barangay}).mappings().all()

    if not rows:
        return {"status": "empty", "features": [], "bounds": None}

    features = [
        {
            "type": "Feature",
            "geometry": r["geometry"],
            "properties": {"section": r["section"]},
        }
        for r in rows
    ]

    bbox_sql = text(f"""
        SELECT
            ST_XMin(ST_Extent(s.geom)) AS xmin,
            ST_YMin(ST_Extent(s.geom)) AS ymin,
            ST_XMax(ST_Extent(s.geom)) AS xmax,
            ST_YMax(ST_Extent(s.geom)) AS ymax
        FROM "{schema}"."SectionBoundary" s
        WHERE s.barangay = :barangay
          AND {license_filter}
    """)

    bbox = db.execute(bbox_sql, {"barangay": barangay}).fetchone()

    bounds = None
    if bbox and bbox.xmin is not None:
        bounds = [
            float(bbox.xmin),
            float(bbox.ymin),
            float(bbox.xmax),
            float(bbox.ymax),
        ]

    return {
        "status": "success",
        "schema": schema,
        "barangay": barangay,
        "features": features,
        "bounds": bounds,
    }


# ===================================================================
#  NEW ENDPOINT:
#  GET: /taxmaplayout/other-barangays?schema=SCHEMA&exclude=NAME
# ===================================================================
@router.get("/other-barangays")
def get_other_barangays(
    schema: str,
    exclude: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_user_main_db),
):
    """
    Returns all barangays EXCEPT the selected one,
    from BarangayBoundary table.
    Uses column `barangay`.
    """

    if not AccessControl.validate_schema_access(schema, current_user):
        raise HTTPException(403, f"Access denied for schema {schema}")

    license_filter = license_validator.sql_filter_clause("b.geom")

    sql = text(f"""
        SELECT 
            b.barangay,
            ST_AsGeoJSON(b.geom)::json AS geometry
        FROM "{schema}"."BarangayBoundary" b
        WHERE b.barangay <> :exclude
          AND {license_filter}
        ORDER BY b.barangay ASC
    """)

    rows = db.execute(sql, {"exclude": exclude}).mappings().all()

    features = [
        {
            "type": "Feature",
            "geometry": r["geometry"],
            "properties": {"barangay": r["barangay"]},
        }
        for r in rows
    ]

    return {
        "status": "success",
        "exclude": exclude,
        "count": len(features),
        "features": features,
    }
