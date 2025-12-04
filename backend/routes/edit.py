from fastapi import APIRouter, Request, Depends
from auth.dependencies import get_current_user, get_user_main_db
from auth.models import User
from sqlalchemy.orm import Session
from datetime import datetime
import json
from psycopg2.extras import RealDictCursor

router = APIRouter()

@router.post("/update-parcel")
async def update_parcel(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_user_main_db)
):
    data = await request.json()
    schema = data.get("schema")
    geom_table_name = data.get("table")
    new_pin = data.get("pin")
    fields = data.get("fields", {})

    old_pin = fields.get("pin")

    print("🔍 Incoming request:")
    print(f"  schema: {schema}")
    print(f"  table: {geom_table_name}")
    print(f"  old_pin: {old_pin}")
    print(f"  new_pin: {new_pin}")

    if not schema or not geom_table_name or not old_pin or not new_pin:
        return {"status": "error", "message": "Missing required data."}

    if old_pin == new_pin:
        return {"status": "error", "message": "Old and new PIN are the same."}

    parcel_table = f'"{schema}"."{geom_table_name}"'
    attr_table = f'"{schema}"."JoinedTable"'
    log_table = f'"{schema}"."parcel_transaction_log"'

    try:
        # Get raw psycopg2 connection
        conn = db.connection().connection
        
        # Use RealDictCursor for dictionary results
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Fetch full attribute record
            cur.execute(f'''
                SELECT *
                FROM {attr_table}
                WHERE pin = %s
            ''', (old_pin,))
            attr_row = cur.fetchone()
            if not attr_row:
                print("❌ Attribute data not found.")
                return {"status": "error", "message": "Attribute data not found."}
            print("📦 Loaded full attribute record.")

            # 2. Fetch geometry
            cur.execute(f'''
                SELECT ST_AsGeoJSON(geom)::json AS geometry
                FROM {parcel_table}
                WHERE pin = %s
            ''', (old_pin,))
            geo_row = cur.fetchone()
            if not geo_row:
                print("❌ Geometry not found.")
                return {"status": "error", "message": "Geometry not found."}
            parcel_geom = json.dumps(geo_row["geometry"])
            print("📦 Loaded geometry.")

            # 3. Merge data
            base_data = dict(attr_row)
            base_data.pop("id", None)
            base_data.pop("geom", None)  # ✅ Prevent duplicate 'geom' column

            # 4. Create log records
            timestamp = datetime.now()

            # Determine which fields changed (exclude 'id')
            edited_fields = [
                k for k in fields
                if k in base_data and base_data[k] != fields[k] and k.lower() != "id"
            ]

            # Determine if pin changed separately
            pin_changed = old_pin != new_pin

            # Build field list for transaction type
            if pin_changed and not edited_fields:
                field_list = "pin"
            elif pin_changed and edited_fields:
                field_list = ", ".join(edited_fields + ["pin"])
            elif not pin_changed and edited_fields:
                field_list = ", ".join(edited_fields)
            else:
                field_list = "unknown"

            transaction_type_old = f"attr. edit (original)({field_list})"
            transaction_type_new = f"attr. edit (new)({field_list})"

            # --- OLD version log
            log_old = base_data.copy()
            log_old["pin"] = old_pin
            log_fields_old = ['"table_name"', '"transaction_type"', '"transaction_date"'] + \
                             [f'"{k}"' for k in log_old] + ['"geom"']
            values_old = [geom_table_name, transaction_type_old, timestamp] + list(log_old.values()) + [parcel_geom]
            placeholders_old = ['%s'] * (3 + len(log_old)) + ['ST_GeomFromGeoJSON(%s)']

            cur.execute(f'''
                INSERT INTO {log_table} ({', '.join(log_fields_old)})
                VALUES ({', '.join(placeholders_old)})
            ''', values_old)
            print("📝 Logged old version.")

            # --- NEW version log
            log_new = base_data.copy()
            log_new["pin"] = new_pin
            log_fields_new = ['"table_name"', '"transaction_type"', '"transaction_date"'] + \
                             [f'"{k}"' for k in log_new] + ['"geom"']
            values_new = [geom_table_name, transaction_type_new, timestamp] + list(log_new.values()) + [parcel_geom]
            placeholders_new = ['%s'] * (3 + len(log_new)) + ['ST_GeomFromGeoJSON(%s)']

            cur.execute(f'''
                INSERT INTO {log_table} ({', '.join(log_fields_new)})
                VALUES ({', '.join(placeholders_new)})
            ''', values_new)
            print("📝 Logged new version.")

            # 5. Update geometry table pin
            cur.execute(f'''
                UPDATE {parcel_table}
                SET pin = %s
                WHERE pin = %s
            ''', (new_pin, old_pin))
            print(f"🔄 Updated pin in geometry table: {old_pin} → {new_pin}")

            # 6. Update JoinedTable pin
            cur.execute(f'''
                UPDATE {attr_table}
                SET pin = %s
                WHERE pin = %s
            ''', (new_pin, old_pin))
            print(f"🔄 Updated pin in JoinedTable: {old_pin} → {new_pin}")

        conn.commit()
        print("✅ Parcel edit completed.")
        return {"status": "success", "message": "Parcel edited and logged successfully."}

    except Exception as e:
        try:
            conn.rollback()
        except:
            pass
        print("❌ Error during update:", str(e))
        return {"status": "error", "message": str(e)}