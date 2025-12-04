from fastapi import APIRouter, Query, HTTPException
from db import get_connection

router = APIRouter()

@router.get("/tables-in-schema")
def get_tables_in_schema(schema: str = Query(...)):
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                ORDER BY table_name
            """, (schema,))
            rows = cur.fetchall()
            tables = [list(r.values())[0] if isinstance(r, dict) else r[0] for r in rows]
        return {"status": "success", "data": tables}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/distinct-values")
def get_distinct_values(
    schema: str = Query(...),
    table: str = Query(...),
    column: str = Query(...)
):
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            sql = f'SELECT DISTINCT "{column}" FROM "{schema}"."{table}" WHERE "{column}" IS NOT NULL'
            cur.execute(sql)
            rows = cur.fetchall()
            values = [list(r.values())[0] if isinstance(r, dict) else r[0] for r in rows]
        return {"status": "success", "data": values}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
