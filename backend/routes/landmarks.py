from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import json

from auth.dependencies import get_user_main_db, get_current_user
from auth.models import User

router = APIRouter()


# ============================================================
# 📦 Pydantic Models
# ============================================================

class LandmarkById(BaseModel):
    db_schema: str = Field(alias="schema")
    id: int

class LandmarkUpdateByFields(BaseModel):
    db_schema: str = Field(alias="schema")
    id: int
    updated: Dict[str, Any]

class LandmarkInsert(BaseModel):
    db_schema: str = Field(alias="schema")
    name: str
    type: str
    barangay: Optional[str] = None
    descr: Optional[str] = None
    geom: dict

class LandmarkRemove(BaseModel):
    db_schema: str = Field(alias="schema")
    ids: list[int]

class BarangayQuery(BaseModel):
    db_schema: str = Field(alias="schema")
    lat: float
    lng: float


# ============================================================
# 📍 1. GET LANDMARKS
# ============================================================

@router.get("/landmarks/{schema}")
async def get_landmarks(
    schema: str,
    db: Session = Depends(get_user_main_db),
    current_user: User = Depends(get_current_user)
):
    """Fetch all landmarks for a given schema."""
    print(f"🔍 Fetching landmarks for schema={schema} by user={current_user.user_name}")
    conn = db.connection().connection

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f'''
                SELECT id, name, type, barangay, descr, ST_AsGeoJSON(geom)::json AS geometry
                FROM "{schema}"."Landmarks"
            ''')
            rows = cur.fetchall()

        features = [
            {
                "type": "Feature",
                "geometry": row["geometry"],
                "properties": {
                    "id": row["id"],
                    "name": row["name"],
                    "type": row["type"],
                    "barangay": row["barangay"],
                    "descr": row["descr"],
                },
            }
            for row in rows
        ]

        print(f"✅ Returned {len(features)} landmarks from {schema}")
        return {"type": "FeatureCollection", "features": features}

    except Exception as e:
        print(f"❌ Landmark query failed for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ➕ 2. INSERT LANDMARK
# ============================================================

@router.post("/landmarks/insert")
async def insert_landmark(
    body: LandmarkInsert,
    db: Session = Depends(get_user_main_db),
    current_user: User = Depends(get_current_user)
):
    """Insert a new landmark into the schema's Landmarks table."""
    print(f"📝 Insert landmark request by {current_user.user_name} in schema={body.db_schema}")
    conn = db.connection().connection

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                INSERT INTO "{body.db_schema}"."Landmarks" (name, type, barangay, descr, geom)
                VALUES (%s, %s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
                RETURNING id
                """,
                (
                    body.name,
                    body.type,
                    body.barangay,
                    body.descr,
                    json.dumps(body.geom),
                ),
            )
            row = cur.fetchone()
            conn.commit()

        new_id = row["id"] if row else None
        print(f"✅ Inserted landmark id={new_id} by {current_user.user_name}")
        return {"status": "success", "id": new_id}

    except Exception as e:
        conn.rollback()
        print(f"❌ Landmark insert failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ✏️ 3. UPDATE LANDMARK
# ============================================================

@router.put("/landmarks/update-by-fields")
async def update_landmark(
    body: LandmarkUpdateByFields,
    db: Session = Depends(get_user_main_db),
    current_user: User = Depends(get_current_user)
):
    """Update landmark attributes by ID."""
    print(f"📝 Update landmark request by {current_user.user_name}: id={body.id}, schema={body.db_schema}")
    conn = db.connection().connection

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            set_clauses = []
            values = []
            for col, val in body.updated.items():
                set_clauses.append(f'"{col}" = %s')  # Quote column names
                values.append(val)

            if not set_clauses:
                raise HTTPException(status_code=400, detail="No fields to update")

        values.append(body.id)
        sql = f'''
            UPDATE "{body.db_schema}"."Landmarks"
            SET {', '.join(set_clauses)}
            WHERE id = %s
        '''

        with conn.cursor() as cur:
            cur.execute(sql, values)
            conn.commit()

        print(f"✅ Updated landmark id={body.id} by {current_user.user_name}")
        return {"status": "success", "updated_id": body.id}

    except Exception as e:
        conn.rollback()
        print(f"❌ Landmark update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🗑️ 4. REMOVE LANDMARKS
# ============================================================

@router.post("/landmarks/remove")
async def remove_landmarks(
    body: LandmarkRemove,
    db: Session = Depends(get_user_main_db),
    current_user: User = Depends(get_current_user)
):
    """Delete multiple landmarks by IDs."""
    print(f"🗑️ Remove landmarks by {current_user.user_name} from schema={body.db_schema}, ids={body.ids}")
    conn = db.connection().connection

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if not body.ids:
                raise HTTPException(status_code=400, detail="No IDs provided")

        with conn.cursor() as cur:
            cur.execute(
                f'DELETE FROM "{body.db_schema}"."Landmarks" WHERE id = ANY(%s)',
                (body.ids,),
            )
            conn.commit()

        print(f"✅ Removed landmarks {body.ids} by {current_user.user_name}")
        return {"status": "success", "removed_ids": body.ids}

    except Exception as e:
        conn.rollback()
        print(f"❌ Landmark removal failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🧭 5. FIND BARANGAY
# ============================================================

@router.post("/find-barangay")
async def find_barangay(
    body: BarangayQuery,
    db: Session = Depends(get_user_main_db),
    current_user: User = Depends(get_current_user)
):
    """Find which barangay boundary polygon contains a given lat/lng point."""
    print(f"📍 Find barangay request by {current_user.user_name} in schema={body.db_schema} at lat={body.lat}, lng={body.lng}")
    conn = db.connection().connection

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT barangay
                FROM "{body.db_schema}"."BarangayBoundary"
                WHERE ST_Contains(geom, ST_SetSRID(ST_Point(%s, %s), 4326))
                LIMIT 1
                """,
                (float(body.lng), float(body.lat)),
            )
            row = cur.fetchone()

        if not row:
            print(f"⚠️ No barangay found for this point ({body.lat}, {body.lng})")
            return {"barangay": None}

        print(f"✅ Found barangay '{row['barangay']}' for user={current_user.user_name}")
        return {"barangay": row["barangay"]}

    except Exception as e:
        print(f"❌ Barangay lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
