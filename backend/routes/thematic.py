# routes/thematic.py
from fastapi import APIRouter, HTTPException
from db import get_connection

router = APIRouter()

@router.get("/thematic-layers")
def get_thematic_layers():
    try:
        conn = get_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    thematic_features = []

    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name
            FROM information_schema.columns
            WHERE column_name IN ('geom', 'elevation')
            GROUP BY table_name
            HAVING COUNT(DISTINCT column_name) = 2
        """)
        tables = [row["table_name"] for row in cur.fetchall()]

    for table in tables:
        with conn.cursor() as cur:
            try:
                cur.execute(f'''
                    SELECT *, ST_AsGeoJSON(geom)::json AS geometry
                    FROM "{table}"
                ''')
                rows = cur.fetchall()
            except Exception as e:
                print(f"Skipping {table}: {e}")
                continue

            for row in rows:
                props = row.copy()
                geometry = props.pop("geometry")
                thematic_features.append({
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": props
                })

    return {
        "type": "FeatureCollection",
        "features": thematic_features
    }
