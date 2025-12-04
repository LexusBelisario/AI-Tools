# AITools/rf_preview.py
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
from sqlalchemy import text

from db import get_user_database_session
from AITools.ai_utils import gdf_from_zip_or_parts, get_provincial_code_from_schema

router = APIRouter()

GEOM_NAMES = {"geom", "geometry", "wkb_geometry", "the_geom"}


@router.post("/preview")
async def file_preview(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    limit: int = Form(100),
    offset: int = Form(0),
):
    """
    Preview uploaded file (shapefile parts or ZIP).
    """
    try:
        gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
        df = gdf.drop(
            columns=[c for c in gdf.columns if c.lower() in GEOM_NAMES],
            errors="ignore",
        )

        total = len(df)
        page_df = df.iloc[int(offset): int(offset) + int(limit)].copy()
        fields = list(page_df.columns)

        def _py(v):
            if pd.isna(v):
                return None
            try:
                return v.item()
            except Exception:
                return v

        rows = [
            {k: _py(v) for k, v in rec.items()}
            for rec in page_df.to_dict(orient="records")
        ]
        return {"rows": rows, "total": int(total), "fields": fields}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/db-preview")
async def preview_db_table(
    table: str,
    schema: str,
    limit: int = 100,
    offset: int = 0,
):
    """
    Preview rows from a DB table (no geometry).
    """
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)
        try:
            # fields
            rows = db_session.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :table
                    ORDER BY ordinal_position
                    """
                ),
                {"schema": schema, "table": table},
            ).fetchall()
            all_fields = [r[0] for r in rows]
            fields = [c for c in all_fields if c.lower() not in GEOM_NAMES]
            if not fields:
                return JSONResponse(
                    status_code=404,
                    content={"error": "No non-geometry columns to preview."},
                )

            # total
            total = db_session.execute(
                text(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
            ).scalar()

            # data
            col_sql = ", ".join(f'"{c}"' for c in fields)
            data_q = text(
                f'SELECT {col_sql} FROM "{schema}"."{table}" LIMIT :limit OFFSET :offset'
            )
            res = db_session.execute(
                data_q, {"limit": limit, "offset": offset}
            )
            data_rows = [dict(zip(fields, r)) for r in res]

            return {
                "fields": fields,
                "rows": data_rows,
                "total": total,
                "schema": schema,
                "table": table,
            }
        finally:
            db_session.close()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
