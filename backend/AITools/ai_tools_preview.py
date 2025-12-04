
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
from sqlalchemy import text

from AITools.ai_utils import (
    gdf_from_zip_or_parts,
    get_provincial_code_from_schema,
    GEOM_NAMES,
)
from db import get_user_database_session

router = APIRouter()


# ------------------------------------------------------
# 🔵 Helper for Python-safe values (numpy → normal types)
# ------------------------------------------------------
def _py(v):
    if pd.isna(v):
        return None
    try:
        return v.item()
    except Exception:
        return v


# ------------------------------------------------------
# 🔷 1. FILE PREVIEW (shapefile parts / ZIP)
# ------------------------------------------------------
@router.post("/preview")
async def ai_file_preview(
    shapefiles: Optional[List[UploadFile]] = None,
    zip_file: Optional[UploadFile] = None,
    limit: int = Form(100),
    offset: int = Form(0),
):
    try:
        print(f"📄 FILE PREVIEW: limit={limit}, offset={offset}")
        
        # Load GeoDataFrame
        gdf = gdf_from_zip_or_parts(shapefiles=shapefiles, zip_file=zip_file)
        print(f"   Loaded GeoDataFrame with {len(gdf)} rows")

        # Drop geometry-like fields
        df = gdf.drop(
            columns=[c for c in gdf.columns if c.lower() in GEOM_NAMES],
            errors="ignore"
        )

        total = len(df)
        page_df = df.iloc[int(offset): int(offset) + int(limit)].copy()
        fields = list(page_df.columns)

        rows = [
            {k: _py(v) for k, v in rec.items()}
            for rec in page_df.to_dict(orient="records")
        ]

        print(f"   ✅ Returning {len(rows)} rows, {total} total")
        return {"rows": rows, "total": int(total), "fields": fields}

    except Exception as e:
        import traceback
        print(f"❌ FILE PREVIEW ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------
# 🔷 2. DB PREVIEW (Preview any table → no geometry)
# ------------------------------------------------------
@router.post("/preview-db")  # ✅ Changed to POST and matching frontend URL
async def ai_db_preview(
    schema: str = Form(...),
    table_name: str = Form(...),
    limit: int = Form(100),
    offset: int = Form(0),
):
    try:
        print(f"📊 DB PREVIEW: {schema}.{table_name}, limit={limit}, offset={offset}")
        
        if not schema or not table_name:
            return JSONResponse(
                status_code=400,
                content={"error": "Schema and table_name are required"}
            )
        
        provincial_code = get_provincial_code_from_schema(schema)
        print(f"   Provincial code: {provincial_code}")
        
        if not provincial_code:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid schema: {schema}"}
            )
        
        db_session = get_user_database_session(provincial_code)

        try:
            # 🔹 Fetch all columns
            col_rows = db_session.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :table_name
                    ORDER BY ordinal_position
                """),
                {"schema": schema, "table_name": table_name},
            ).fetchall()

            all_cols = [r[0] for r in col_rows]
            print(f"   Found {len(all_cols)} columns: {all_cols}")

            # 🔹 Drop geometry columns
            fields = [c for c in all_cols if c.lower() not in GEOM_NAMES]
            print(f"   Non-geometry fields: {fields}")

            if not fields:
                return JSONResponse(
                    status_code=404,
                    content={"error": "No non-geometry columns to preview."}
                )

            # 🔹 Count rows
            total = db_session.execute(
                text(f'SELECT COUNT(*) FROM "{schema}"."{table_name}"')
            ).scalar()
            print(f"   Total rows in table: {total}")

            # 🔹 Data query
            col_sql = ", ".join(f'"{c}"' for c in fields)
            data_query = text(
                f'SELECT {col_sql} FROM "{schema}"."{table_name}" '
                f'LIMIT :limit OFFSET :offset'
            )
            res = db_session.execute(
                data_query, {"limit": limit, "offset": offset}
            )

            rows = [dict(zip(fields, r)) for r in res]
            print(f"   ✅ Returning {len(rows)} rows")

            return {
                "fields": fields,
                "rows": rows,
                "total": total,
                "schema": schema,
                "table": table_name,
            }

        finally:
            db_session.close()

    except Exception as e:
        import traceback
        print(f"❌ DB PREVIEW ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})