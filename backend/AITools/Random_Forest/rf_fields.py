# AITools/rf_fields.py
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import pandas as pd
from sqlalchemy import text

from AITools.ai_utils import gdf_from_zip_or_parts, get_provincial_code_from_schema
from db import get_user_database_session

router = APIRouter()

GEOM_NAMES = {"geom", "geometry", "wkb_geometry", "the_geom"}


@router.post("/fields")
async def extract_fields(shapefiles: List[UploadFile]):
    """
    Get field names from uploaded shapefile parts (.shp/.dbf/.shx/.prj).
    Geometry columns are dropped.
    """
    try:
        gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=None)
        df = gdf.drop(
            columns=[c for c in gdf.columns if c.lower() in GEOM_NAMES],
            errors="ignore",
        )
        fields = df.columns.tolist()
        return {"fields": fields}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.post("/fields-zip")
async def extract_fields_zip(zip_file: UploadFile):
    """
    Get field names from uploaded ZIP file containing a shapefile.
    Geometry columns are dropped.
    """
    try:
        gdf = gdf_from_zip_or_parts(shapefiles=None, zip_file=zip_file)
        df = gdf.drop(
            columns=[c for c in gdf.columns if c.lower() in GEOM_NAMES],
            errors="ignore",
        )
        fields = df.columns.tolist()
        return {"fields": fields}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.get("/db-fields")
async def get_db_fields_for_table(table: str, schema: str):
    """
    Get field names from a DB table, excluding geometry columns.
    """
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)
        try:
            q = text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ordinal_position
                """
            )
            rows = db_session.execute(
                q, {"schema": schema, "table": table}
            ).fetchall()
            cols_all = [r[0] for r in rows]
            fields = [c for c in cols_all if c.lower() not in GEOM_NAMES]
            if not fields:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": f"Table '{table}' not found or has no non-geometry columns in schema '{schema}'"
                    },
                )
            return {"fields": fields, "table": table, "schema": schema}
        finally:
            db_session.close()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/db-tables")
async def get_db_tables_for_schema(schema: str):
    """
    Return candidate AI tables for this schema.
    For now: only CAMA_Table, same as LR/XGB.
    """
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)
        try:
            q = text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema
                  AND table_type = 'BASE TABLE'
                  AND LOWER(table_name) = 'cama_table'
                ORDER BY table_name
                """
            )
            rows = db_session.execute(q, {"schema": schema}).fetchall()
            tables = [r[0] for r in rows]
            return {"tables": tables, "schema": schema}
        finally:
            db_session.close()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
