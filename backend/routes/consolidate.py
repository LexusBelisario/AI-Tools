from fastapi import APIRouter, Request, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import json
from psycopg2.extras import RealDictCursor

from auth.dependencies import get_user_main_db, get_current_user
from auth.models import User

router = APIRouter()

@router.post("/merge-parcels-postgis")
async def merge_parcels_postgis(
    request: Request,
    db: Session = Depends(get_user_main_db),
    current_user: User = Depends(get_current_user)
):
    data = await request.json()

    schema = data.get("schema")
    table = data.get("table")
    base_props = data.get("base_props")
    original_pins = data.get("original_pins")
    geometries = data.get("geometries")

    if not schema or not table or not base_props or not original_pins or not geometries:
        return {"status": "error", "message": "Missing required data."}

    full_table = f'"{schema}"."{table}"'
    log_table = f'"{schema}"."parcel_transaction_log"'
    attr_table = f'"{schema}"."JoinedTable"'

    try:
        conn = db.connection().connection
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # ==========================================================
            # STEP 1: Column detection
            # ==========================================================
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
            """, (schema, table))
            all_columns = [r["column_name"] for r in cur.fetchall()]
            allowed_columns = set(all_columns) - {"geom"}

            # ==========================================================
            # STEP 2: Merge geometries
            # ==========================================================
            geojson_strings = [json.dumps(g) for g in geometries]
            union_args = ', '.join(['ST_GeomFromGeoJSON(%s)'] * len(geojson_strings))
            cur.execute(f"""
                SELECT ST_AsGeoJSON(ST_Union(ARRAY[{union_args}])) AS merged;
            """, geojson_strings)
            merged_geom = cur.fetchone()["merged"]
            if not merged_geom:
                return {"status": "error", "message": "Geometry union failed."}

            # ==========================================================
            # STEP 3: Inherit barangay info
            # ==========================================================
            cur.execute(f"""
                SELECT DISTINCT brgy_code, brgy_nm
                FROM {attr_table}
                WHERE pin = ANY(%s)
            """, (original_pins,))
            brgy_data = cur.fetchone() or {}
            brgy_code = brgy_data.get("brgy_code")
            brgy_nm = brgy_data.get("brgy_nm")

            # ==========================================================
            # STEP 4: Generate new PIN
            # ==========================================================
            prefix = original_pins[0].rsplit("-", 1)[0]
            cur.execute(f'SELECT pin FROM {full_table} WHERE pin ~ %s', (r'.*\d{3}$',))
            existing_pins = [r["pin"] for r in cur.fetchall()]
            suffixes = [int(p[-3:]) for p in existing_pins if p[-3:].isdigit()]
            new_pin = f"{prefix}-{str(max(suffixes or [0]) + 1).zfill(3)}"

            # ==========================================================
            # STEP 5: Insert new parcel (with brgy inheritance)
            # ==========================================================
            base_props.update({
                "pin": new_pin,
                "parcel": "",
                "section": "",
                "brgy_code": brgy_code,
                "brgy_nm": brgy_nm
            })
            base_props.pop("id", None)

            clean_props = {k: v for k, v in base_props.items() if k in allowed_columns}
            cols = ', '.join(f'"{c}"' for c in clean_props)
            vals = list(clean_props.values())
            ph = ', '.join(['%s'] * len(clean_props))

            cur.execute(f"""
                INSERT INTO {full_table} ({cols}, geom)
                VALUES ({ph}, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
            """, vals + [merged_geom])

            # ==========================================================
            # STEP 6: Insert new JoinedTable record
            # ==========================================================
            cur.execute(f"""
                INSERT INTO {attr_table} ("pin", "brgy_code", "brgy_nm")
                VALUES (%s, %s, %s)
            """, (new_pin, brgy_code, brgy_nm))

            # ==========================================================
            # STEP 7: Log new parcel (with brgy info)
            # ==========================================================
            new_date = datetime.now()
            log_fields = ['"table_name"', '"transaction_type"', '"transaction_date"', '"pin"', '"brgy_code"', '"brgy_nm"', '"geom"']
            log_placeholders = ['%s', '%s', '%s', '%s', '%s', '%s', 'ST_GeomFromGeoJSON(%s)']
            log_values = [table, "new (consolidate)", new_date, new_pin, brgy_code, brgy_nm, merged_geom]

            cur.execute(f"""
                INSERT INTO {log_table} ({', '.join(log_fields)})
                VALUES ({', '.join(log_placeholders)})
            """, log_values)

            # ==========================================================
            # STEP 8: Log & delete old parcels
            # ==========================================================
            for pin in original_pins:
                cur.execute(f"""
                    SELECT *, ST_AsGeoJSON(geom)::json AS geometry
                    FROM {full_table}
                    WHERE pin = %s
                """, (pin,))
                parcel = cur.fetchone()
                if not parcel:
                    continue

                cur.execute(f'SELECT * FROM {attr_table} WHERE pin = %s', (pin,))
                attr = cur.fetchone() or {}

                merged_data = {**parcel, **attr}
                geom = merged_data.pop("geometry", None)
                merged_data.pop("geom", None)
                merged_data.pop("id", None)

                old_fields = list(merged_data.keys())
                lf = ['"table_name"', '"transaction_type"', '"transaction_date"'] + [f'"{f}"' for f in old_fields] + ['"geom"']
                lp = ['%s'] * (3 + len(old_fields)) + ['ST_GeomFromGeoJSON(%s)']
                lv = [table, "consolidated", datetime.now()] + list(merged_data.values()) + [json.dumps(geom)]

                cur.execute(f'INSERT INTO {log_table} ({", ".join(lf)}) VALUES ({", ".join(lp)})', lv)
                cur.execute(f'DELETE FROM {full_table} WHERE pin = %s', (pin,))
                cur.execute(f'DELETE FROM {attr_table} WHERE pin = %s', (pin,))

            conn.commit()
            print(f"✅ Consolidation successful for user {current_user.user_name}: {new_pin}")
            return {"status": "success", "new_pin": new_pin}

    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"❌ Consolidation error: {e}")
        return {"status": "error", "message": str(e)}
