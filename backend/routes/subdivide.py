from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from psycopg2.extras import RealDictCursor
import json

from auth.dependencies import get_user_main_db, get_current_user
from auth.models import User

router = APIRouter()


# =========================================================
# 🔹 1. PREVIEW: Split parcel geometrically only (no save)
# =========================================================
@router.post("/subdivide-preview")
async def subdivide_preview(
    request: Request,
    db: Session = Depends(get_user_main_db),
    current_user: User = Depends(get_current_user)
):
    """
    Run ST_Split() on a parcel using provided lines, return split polygons and
    suggested PINs — no database modification.
    """
    data = await request.json()
    pin = data.get("pin")
    table = data.get("table")
    schema = data.get("schema")
    split_lines = data.get("split_lines")

    if not schema or not table or not split_lines:
        return {"status": "error", "message": "Missing required input (schema, table, or split lines)."}

    full_table = f'"{schema}"."{table}"'

    try:
        conn = db.connection().connection
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            print(f"🧮 Subdivide PREVIEW by {current_user.user_name}: schema={schema}, table={table}, pin={pin}")

            # === 1. Get original parcel geometry ===
            cur.execute(f'''
                SELECT pin, ST_AsGeoJSON(geom)::json AS geometry
                FROM {full_table}
                WHERE pin = %s
                LIMIT 1
            ''', (pin,))
            row = cur.fetchone()
            if not row or not row.get("geometry"):
                return {"status": "error", "message": "Parcel not found or geometry missing."}

            parcel_geom = json.dumps(row["geometry"])

            # === 2. Build MultiLineString and run ST_Split() ===
            line_geoms = [
                f"ST_SetSRID(ST_GeomFromGeoJSON('{json.dumps({'type': 'LineString', 'coordinates': line})}'), 4326)"
                for line in split_lines
            ]
            multiline_sql = f"ST_SetSRID(ST_Union(ARRAY[{','.join(line_geoms)}]), 4326)"
            split_sql = f'''
                SELECT ST_AsGeoJSON((ST_Dump(ST_CollectionExtract(
                    ST_Split(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), {multiline_sql}), 3
                ))).geom)::json AS geom
            '''
            cur.execute(split_sql, (parcel_geom,))
            parts = cur.fetchall()

            if not parts or len(parts) < 2:
                return {"status": "error", "message": "Split operation produced less than 2 parts."}

            print(f"📐 Preview split success: {len(parts)} parts generated.")

            # === 3. Generate suggested PINs ===
            pin_parts = pin.split("-")
            if len(pin_parts) != 5:
                prefix = pin.rsplit("-", 1)[0]
            else:
                prefix = "-".join(pin_parts[:4])

            cur.execute(f'SELECT pin FROM {full_table} WHERE pin LIKE %s', (f"{prefix}%",))
            existing_pins = [r["pin"] for r in cur.fetchall()]
            suffixes = [int(p.split("-")[-1]) for p in existing_pins if p.split("-")[-1].isdigit()]
            next_suffix = max(suffixes or [0]) + 1

            suggested_pins = [
                f"{prefix}-{str(next_suffix + i).zfill(3)}" for i in range(len(parts))
            ]

            print(f"🔢 Suggested preview PINs: {suggested_pins}")

            return {
                "status": "success",
                "message": f"Preview successful with {len(parts)} parts.",
                "parts": parts,
                "suggested_pins": suggested_pins
            }

    except Exception as e:
        print(f"❌ Preview error: {str(e)}")
        return {"status": "error", "message": str(e)}


# =========================================================
# 🔹 2. SAVE: Commit subdivision to the database
# =========================================================
@router.post("/subdivide")
async def subdivide_parcel(
    request: Request,
    db: Session = Depends(get_user_main_db),
    current_user: User = Depends(get_current_user)
):
    """
    Commit the subdivision results to the database. Deletes original parcel,
    inserts new parts with given or suggested PINs, logs transactions.
    """
    data = await request.json()
    pin = data.get("pin")
    table = data.get("table")
    schema = data.get("schema")
    split_lines = data.get("split_lines")
    new_pins = data.get("new_pins")

    if not schema or not table or not split_lines:
        return {"status": "error", "message": "Missing required input (schema, table, or split lines)."}

    full_table = f'"{schema}"."{table}"'
    attr_table = f'"{schema}"."JoinedTable"'
    log_table = f'"{schema}"."parcel_transaction_log"'

    try:
        conn = db.connection().connection
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            print(f"🧩 Subdivide SAVE by {current_user.user_name}: schema={schema}, table={table}, pin={pin}")

            # === Detect actual log table columns ===
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
            """, (schema, "parcel_transaction_log"))
            log_columns = [r["column_name"] for r in cur.fetchall()]

            # === 1. Get original parcel geometry ===
            cur.execute(f'''
                SELECT pin, ST_AsGeoJSON(geom)::json AS geometry
                FROM {full_table}
                WHERE pin = %s
                LIMIT 1
            ''', (pin,))
            geo_row = cur.fetchone()
            if not geo_row or not geo_row.get("geometry"):
                return {"status": "error", "message": "Parcel not found or geometry missing."}

            parcel_geom = json.dumps(geo_row["geometry"])

            # === 1.1 Get attributes from JoinedTable ===
            cur.execute(f'''SELECT * FROM {attr_table} WHERE pin = %s''', (pin,))
            attr_row = cur.fetchone() or {}
            merged_props = attr_row.copy()

            # === 2. Perform split ===
            line_geoms = [
                f"ST_SetSRID(ST_GeomFromGeoJSON('{json.dumps({'type': 'LineString', 'coordinates': line})}'), 4326)"
                for line in split_lines
            ]
            multiline_sql = f"ST_SetSRID(ST_Union(ARRAY[{','.join(line_geoms)}]), 4326)"
            split_sql = f'''
                SELECT (ST_Dump(ST_CollectionExtract(
                    ST_Split(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), {multiline_sql}), 3
                ))).geom::geometry
            '''
            cur.execute(split_sql, (parcel_geom,))
            parts = cur.fetchall()

            if not parts or len(parts) < 2:
                return {"status": "error", "message": "Subdivision failed or created less than 2 parts."}

            transaction_date = datetime.now()

            # === 3. Log original parcel safely ===
            attr_fields = [f for f in merged_props.keys() if f.lower() not in ("id", "geom") and f in log_columns]
            log_fields = ', '.join(['"table_name"', '"transaction_type"', '"transaction_date"']
                                   + [f'"{f}"' for f in attr_fields] + ['"geom"'])
            log_placeholders = ', '.join(['%s'] * (3 + len(attr_fields)) + ['ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)'])
            log_values = [table, "subdivided", transaction_date] + [merged_props.get(f) for f in attr_fields] + [parcel_geom]

            cur.execute(f'''
                INSERT INTO {log_table} ({log_fields})
                VALUES ({log_placeholders})
            ''', log_values)
            print("🗂️ Logged original parcel as subdivided.")

            # === 4. Delete original parcel ===
            cur.execute(f'DELETE FROM {full_table} WHERE pin = %s', (pin,))
            cur.execute(f'DELETE FROM {attr_table} WHERE pin = %s', (pin,))
            print("🧹 Original parcel removed.")

            # === 5. Compute suggested PINs ===
            pin_parts = pin.split("-")
            prefix = "-".join(pin_parts[:4]) if len(pin_parts) == 5 else pin.rsplit("-", 1)[0]
            cur.execute(f'SELECT pin FROM {full_table} WHERE pin LIKE %s', (f"{prefix}%",))
            existing_pins = [r["pin"] for r in cur.fetchall()]
            suffixes = [int(p.split("-")[-1]) for p in existing_pins if p.split("-")[-1].isdigit()]
            next_suffix = max(suffixes or [0]) + 1

            suggested_pins = [f"{prefix}-{str(next_suffix + i).zfill(3)}" for i in range(len(parts))]

            # === 6. Insert and log new parts safely ===
            for idx, geom in enumerate(parts):
                final_pin = (new_pins[idx] if new_pins and idx < len(new_pins)
                             else suggested_pins[idx])
                raw_geom = geom["geom"]
                new_props = {k: None for k in merged_props.keys() if k.lower() not in ("id", "geom")}
                new_props["pin"] = final_pin

                # geometry insert
                cur.execute(f'''
                    INSERT INTO {full_table} ("pin", geom)
                    VALUES (%s, ST_SetSRID(%s::geometry, 4326))
                ''', [final_pin, raw_geom])

                # attribute insert
                attr_cols = ', '.join(f'"{k}"' for k in new_props.keys())
                attr_vals = ', '.join(['%s'] * len(new_props))
                cur.execute(f'''
                    INSERT INTO {attr_table} ({attr_cols})
                    VALUES ({attr_vals})
                ''', list(new_props.values()))

                # log insert safely filtered
                loggable_props = {k: v for k, v in new_props.items() if k in log_columns}
                log_cols = ', '.join(f'"{k}"' for k in loggable_props.keys())
                log_vals = list(loggable_props.values())

                cur.execute(f'''
                    INSERT INTO {log_table}
                    (table_name, transaction_type, transaction_date, {log_cols}, geom)
                    VALUES (%s, %s, %s, {', '.join(['%s'] * len(loggable_props))}, ST_SetSRID(%s::geometry, 4326))
                ''', [table, "new (subdivide)", transaction_date] + log_vals + [raw_geom])

            conn.commit()
            print(f"✅ Subdivision saved successfully ({len(parts)} parts).")

            return {
                "status": "success",
                "message": f"Created {len(parts)} subdivisions.",
                "suggested_pins": suggested_pins
            }

    except Exception as e:
        try:
            conn.rollback()
        except:
            pass
        print(f"❌ Subdivide SAVE error: {str(e)}")
        return {"status": "error", "message": str(e)}
