from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from auth.dependencies import get_user_main_db

router = APIRouter()

# ==========================================================
# 🛰️ ORTHOPHOTO CONFIGURATION (Get & Save)
# ==========================================================

@router.get("/orthophoto-config")
def get_orthophoto_config(schema: str, db: Session = Depends(get_user_main_db)):
    """
    Retrieve the stored GeoServer WMTS URL and Layer Name for orthophotos
    within the selected schema's "Orthophotos" table.
    Automatically creates the table if missing.
    """
    try:
        current_db = db.execute(text("SELECT current_database()")).scalar()
        print(f"📌 Connected to DB={current_db}, schema={schema} (GET orthophoto-config)")

        create_table_if_missing(db, schema)

        row = db.execute(text(f"""
            SELECT gsrvr_url, layer_name
            FROM "{schema}"."Orthophotos"
            ORDER BY id DESC
            LIMIT 1
        """)).mappings().first()

        if not row:
            print(f"ℹ️ No orthophoto config found for schema={schema}")
            return {
                "status": "empty",
                "message": f"No orthophoto configuration found for schema {schema}."
            }

        print(f"✅ Loaded orthophoto config for {schema}: {row}")
        return {
            "status": "success",
            "Gsrvr_URL": row["gsrvr_url"],
            "Layer_Name": row["layer_name"]
        }

    except Exception as e:
        print(f"❌ Error retrieving orthophoto config for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orthophoto-config")
async def save_orthophoto_config(request: Request, db: Session = Depends(get_user_main_db)):
    """
    Save or update the orthophoto configuration (GeoServer URL + Layer Name)
    for a given schema. If an entry exists, overwrite it; otherwise insert a new one.
    """
    data = await request.json()
    schema = data.get("schema")
    url = data.get("Gsrvr_URL")
    layer = data.get("Layer_Name")

    if not schema or not url or not layer:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: schema, Gsrvr_URL, or Layer_Name."
        )

    try:
        current_db = db.execute(text("SELECT current_database()")).scalar()
        print(f"📌 Connected to DB={current_db}, schema={schema} (POST orthophoto-config)")

        create_table_if_missing(db, schema)

        # --- Check if a record already exists
        result = db.execute(
            text(f'SELECT id FROM "{schema}"."Orthophotos" ORDER BY id DESC LIMIT 1')
        ).mappings().first()

        if result and "id" in result:
            # --- Update existing
            update_query = text(f'''
                UPDATE "{schema}"."Orthophotos"
                SET gsrvr_url = :url,
                    layer_name = :layer
                WHERE id = :id
            ''')
            db.execute(update_query, {"url": url, "layer": layer, "id": result["id"]})
            db.commit()
            print(f"✅ Orthophoto config updated for {schema}: URL={url}, Layer={layer}")
            return {
                "status": "success",
                "message": "Orthophoto configuration updated successfully.",
                "Gsrvr_URL": url,
                "Layer_Name": layer
            }

        else:
            # --- Insert new
            insert_query = text(f'''
                INSERT INTO "{schema}"."Orthophotos" (gsrvr_url, layer_name)
                VALUES (:url, :layer)
            ''')
            db.execute(insert_query, {"url": url, "layer": layer})
            db.commit()
            print(f"✅ Orthophoto config inserted for {schema}: URL={url}, Layer={layer}")
            return {
                "status": "success",
                "message": "Orthophoto configuration inserted successfully.",
                "Gsrvr_URL": url,
                "Layer_Name": layer
            }

    except Exception as e:
        db.rollback()
        print(f"❌ Error saving orthophoto config for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# 🛠️ Utility: Create table if missing
# ==========================================================
def create_table_if_missing(db: Session, schema: str):
    """Ensure the 'Orthophotos' table exists for this schema."""
    try:
        db.execute(text(f"""
            CREATE TABLE IF NOT EXISTS "{schema}"."Orthophotos" (
                id SERIAL PRIMARY KEY,
                gsrvr_url TEXT NOT NULL,
                layer_name TEXT NOT NULL
            );
        """))
        db.commit()
        print(f"🧱 Ensured {schema}.Orthophotos exists.")
    except Exception as e:
        db.rollback()
        print(f"⚠️ Failed to ensure Orthophotos table for {schema}: {e}")
