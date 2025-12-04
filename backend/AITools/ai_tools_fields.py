from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from sqlalchemy import text
from typing import List
import pandas as pd

from AITools.ai_utils import (
    gdf_from_zip_or_parts,
    get_provincial_code_from_schema,
    GEOM_NAMES,
)
from db import get_user_database_session

router = APIRouter()

@router.post("/fields")
async def ai_extract_fields(shapefiles: List[UploadFile]):
    try:
        gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=None)

        df = gdf.drop(
            columns=[c for c in gdf.columns if c.lower() in GEOM_NAMES],
            errors="ignore",
        )

        return {"fields": df.columns.tolist()}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ------------------------------------------------------------
# 🔹 2. Extract fields from ZIP file
# ------------------------------------------------------------
@router.post("/fields-zip")
async def ai_extract_fields_zip(zip_file: UploadFile):
    try:
        gdf = gdf_from_zip_or_parts(shapefiles=None, zip_file=zip_file)

        df = gdf.drop(
            columns=[c for c in gdf.columns if c.lower() in GEOM_NAMES],
            errors="ignore",
        )

        return {"fields": df.columns.tolist()}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ------------------------------------------------------------
# 🔹 3. Get list of fields from a DB table (excluding geometry)
# ------------------------------------------------------------
@router.post("/fields-db")  # ✅ Changed to POST to accept JSON body
async def ai_db_fields(schema: str = Form(...), table_name: str = Form(...)):
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)

        try:
            q = text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND table_name = :table
                ORDER BY ordinal_position
                """
            )

            rows = db_session.execute(
                q, {"schema": schema, "table": table_name}
            ).fetchall()

            if not rows:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Table '{table_name}' not found in schema '{schema}'"},
                )

            fields = [
                r[0] for r in rows
                if r[0].lower() not in GEOM_NAMES
            ]

            return {
                "fields": fields,
                "schema": schema,
                "table": table_name,
            }

        finally:
            db_session.close()

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------------
# 🔹 4. Get AI training tables (Training_Table only)
# ------------------------------------------------------------
@router.post("/list-tables")  # ✅ Match the frontend endpoint name
async def ai_list_training_tables(schema: str = Form(...)):
    """List all Training_Table variants in a schema"""
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
                  AND LOWER(table_name) LIKE '%training_table%'
                ORDER BY table_name
                """
            )

            rows = db_session.execute(q, {"schema": schema}).fetchall()
            tables = [r[0] for r in rows]

            return {
                "tables": tables,
                "schema": schema,
            }

        finally:
            db_session.close()

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})