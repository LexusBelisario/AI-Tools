# ============================================================
#  🔍 SEARCH ROUTES
#  Handles all search-related endpoints for property, road,
#  landmark, and attribute (JoinedTable) data.
# ============================================================

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from auth.dependencies import get_user_main_db

router = APIRouter(prefix="/search", tags=["Search Tools"])

# ============================================================
# 🧩 1. ATTRIBUTE TABLE (JoinedTable Loader)
# ============================================================

@router.get("/attribute-table")
def get_attribute_table(
    schema: str = Query(..., description="Municipal schema name, e.g., PH0403406"),
    db: Session = Depends(get_user_main_db)
):
    try:
        print(f"📂 Fetching JoinedTable for schema: {schema}")

        check_sql = text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = 'JoinedTable'
            );
        """)
        exists = db.execute(check_sql, {"schema": schema}).scalar()

        if not exists:
            print(f"⚠️ No JoinedTable found in schema '{schema}'")
            return {
                "status": "error",
                "message": f"No JoinedTable found in schema '{schema}'",
                "data": []
            }

        query = text(f'SELECT * FROM "{schema}"."JoinedTable"')
        result = db.execute(query)
        rows = [dict(row._mapping) for row in result]

        print(f"✅ Retrieved {len(rows)} records from {schema}.JoinedTable")
        return {"status": "success", "data": rows, "count": len(rows)}

    except Exception as e:
        print(f"❌ Error fetching JoinedTable for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🏠 2. PROPERTY SEARCH
# ============================================================

@router.post("/property-search")
async def property_search(request: Request, db: Session = Depends(get_user_main_db)):
    data = await request.json()
    schema = data.get("schema")
    filters = data.get("filters", {})

    if not schema:
        raise HTTPException(status_code=400, detail="Schema is required for property search.")

    try:
        where_clauses = []
        params = {}

        for i, (field, value) in enumerate(filters.items()):
            if not value:
                continue
            key = f"v{i}"
            where_clauses.append(f'LOWER("{field}") LIKE :{key}')
            params[key] = f"%{value.lower()}%"

        sql = f'SELECT * FROM "{schema}"."JoinedTable"'
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        query = text(sql)
        result = db.execute(query, params)
        rows = [dict(row._mapping) for row in result]

        print(f"✅ Property search in {schema}: {len(rows)} result(s)")
        return {"status": "success", "data": rows, "count": len(rows)}

    except Exception as e:
        print(f"❌ Property search error for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🛣️ 3. ROAD SEARCH (Adaptive)
# ============================================================

@router.post("/road-search")
async def road_search(request: Request, db: Session = Depends(get_user_main_db)):
    """
    Searches for roads by name, type, or classification.
    Works for both 'RoadNetwork' and 'RoadInfo' tables.
    """
    data = await request.json()
    schema = data.get("schema")
    filters = data.get("filters", {})

    if not schema:
        raise HTTPException(status_code=400, detail="Schema is required for road search.")

    try:
        # 🔍 Check which road table exists
        table_check = text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = :schema AND table_name IN ('RoadNetwork', 'RoadInfo')
            LIMIT 1;
        """)
        result = db.execute(table_check, {"schema": schema}).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail=f"No road table found in schema '{schema}'")

        table_name = result[0]
        print(f"🛣️ Using road table: {table_name}")

        # 🧩 Build WHERE clause dynamically
        where_clauses = []
        params = {}

        for i, (field, value) in enumerate(filters.items()):
            if not value:
                continue
            key = f"v{i}"
            where_clauses.append(f'(LOWER("{field}") LIKE :{key})')
            params[key] = f"%{value.lower()}%"

        # 🧭 If only road_name provided, search across name, road_type, and classification
        if not where_clauses and "road_name" in filters:
            val = filters.get("road_name")
            if val:
                params["val"] = f"%{val.lower()}%"
                where_clauses = [
                    '(LOWER("road_name") LIKE :val OR LOWER("road_type") LIKE :val '
                    'OR LOWER("type") LIKE :val OR LOWER("classification") LIKE :val)'
                ]

        sql = f'SELECT * FROM "{schema}"."{table_name}"'
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        query = text(sql)
        result = db.execute(query, params)
        rows = [dict(row._mapping) for row in result]

        print(f"✅ Road search in {schema}.{table_name}: {len(rows)} result(s)")
        # ✅ Log first record for debugging column names
        if rows:
            print(f"📋 Sample fields: {list(rows[0].keys())}")

        return {"status": "success", "data": rows, "count": len(rows)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Road search error for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 📍 4. LANDMARK SEARCH
# ============================================================

@router.post("/landmark-search")
async def landmark_search(request: Request, db: Session = Depends(get_user_main_db)):
    data = await request.json()
    schema = data.get("schema")
    filters = data.get("filters", {})

    if not schema:
        raise HTTPException(status_code=400, detail="Schema is required for landmark search.")

    try:
        where_clauses = []
        params = {}

        for i, (field, value) in enumerate(filters.items()):
            if not value:
                continue
            key = f"v{i}"
            where_clauses.append(f'LOWER("{field}") LIKE :{key}')
            params[key] = f"%{value.lower()}%"

        sql = f'SELECT * FROM "{schema}"."Landmarks"'
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        query = text(sql)
        result = db.execute(query, params)
        rows = [dict(row._mapping) for row in result]

        print(f"✅ Landmark search in {schema}: {len(rows)} result(s)")
        return {"status": "success", "data": rows, "count": len(rows)}

    except Exception as e:
        print(f"❌ Landmark search error for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
