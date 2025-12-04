from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sqlalchemy import text
from db import get_user_database_session
from .lr_train import gdf_from_zip_or_parts, get_provincial_code_from_schema

router = APIRouter()

GEOM_NAMES = {"geom", "geometry", "wkb_geometry", "the_geom"}

@router.post("/preview")
async def file_preview(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    limit: int = Form(100),
    offset: int = Form(0),
):
    try:
        gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
        if gdf.geometry is not None:
            gdf = gdf.drop(columns=[gdf.geometry.name], errors="ignore")
        total = len(gdf)
        page_df = gdf.iloc[int(offset): int(offset) + int(limit)].copy()
        fields = list(page_df.columns)
        def _py(v):
            if pd.isna(v):
                return None
            try:
                return v.item()
            except Exception:
                return v
        rows = [{k: _py(v) for k, v in rec.items()} for rec in page_df.to_dict(orient="records")]
        return {"rows": rows, "total": int(total), "fields": fields}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    
@router.get("/db-preview")
async def preview_db_table(
    table: str,
    schema: str,
    limit: int = 100,
    offset: int = 0
):
    """Single, non-duplicated /db-preview."""
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)
        try:
            # fields
            fields_rows = db_session.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :table
                    ORDER BY ordinal_position
                """),
                {"schema": schema, "table": table},
            ).fetchall()
            fields_all = [r[0] for r in fields_rows]
            fields = [c for c in fields_all if c.lower() not in GEOM_NAMES]
            if not fields:
                return JSONResponse(status_code=404, content={"error": "No non-geometry columns to preview."})

            # total
            total = db_session.execute(
                text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
            ).scalar()

            # rows
            col_sql = ", ".join([f'"{c}"' for c in fields])  # ✅ build safely outside the f-string
            data_query = text(
                f'SELECT {col_sql} FROM "{schema}"."{table}" LIMIT :limit OFFSET :offset'
            )
            res = db_session.execute(data_query, {"limit": limit, "offset": offset})
            rows = [dict(zip(fields, r)) for r in res]
            return {"fields": fields, "rows": rows, "total": total, "schema": schema, "table": table}
        finally:
            db_session.close()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})