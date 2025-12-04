# routes/sync.py
from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
import psycopg
from auth.dependencies import get_user_main_db

router = APIRouter()

# ============================================================
# 🔹 1. GET — Retrieve SyncCreds
# ============================================================
@router.get("/sync-config")
def get_sync_config(schema: str, db: Session = Depends(get_user_main_db)):
    """Retrieve host, port, username, password from SyncCreds table."""
    try:
        current_db = db.execute(text("SELECT current_database()")).scalar()
        print(f"📡 GET /sync-config — DB={current_db}, schema={schema}")

        row = db.execute(text(f"""
            SELECT host, port, username, password
            FROM "{schema}"."SyncCreds"
            ORDER BY id DESC LIMIT 1
        """)).mappings().first()

        if not row:
            return {"status": "empty", "message": f"No SyncCreds found in {schema}"}

        return {
            "status": "success",
            "host": row["host"],
            "port": row["port"],
            "username": row["username"],
            "password": row["password"] or ""
        }

    except Exception as e:
        print(f"❌ Error in get_sync_config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🔹 2. POST — Save SyncCreds
# ============================================================
@router.post("/sync-config")
async def save_sync_config(request: Request, db: Session = Depends(get_user_main_db)):
    """Save or update SyncCreds with password."""
    data = await request.json()
    schema = data.get("schema")
    host = data.get("host")
    port = data.get("port")
    username = data.get("username")
    password = data.get("password")

    if not schema or not host or not port or not username:
        raise HTTPException(status_code=400, detail="Missing required fields")

    try:
        db.execute(text(f"""
            CREATE TABLE IF NOT EXISTS "{schema}"."SyncCreds" (
                id SERIAL PRIMARY KEY,
                host TEXT,
                port TEXT,
                username TEXT,
                password TEXT
            )
        """))
        db.execute(text(f"""
            INSERT INTO "{schema}"."SyncCreds" (host, port, username, password)
            VALUES (:host, :port, :username, :password)
        """), {"host": host, "port": port, "username": username, "password": password})
        db.commit()
        print(f"✅ SyncCreds saved for {schema}: {username}@{host}:{port}")
        return {"status": "success", "message": "Credentials saved successfully."}
    except Exception as e:
        db.rollback()
        print(f"❌ Error saving SyncCreds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🔹 3. PUSH — Match by ID (fixes PIN rename problem)
# ============================================================
@router.post("/sync-push")
async def sync_push(request: Request, db: Session = Depends(get_user_main_db)):
    """
    Push 'id', 'pin', 'bounds', 'computed_area' from GIS → RPIS.
    Uses 'id' as the matching key to handle renamed PINs safely.
    """
    data = await request.json()
    schema = data.get("schema")
    if not schema:
        raise HTTPException(status_code=400, detail="Schema is required.")

    try:
        current_db = db.execute(text("SELECT current_database()")).scalar()
        print(f"🚀 PUSH triggered from {current_db}.{schema}")

        creds = db.execute(text(f"""
            SELECT host, port, username, password
            FROM "{schema}"."SyncCreds"
            ORDER BY id DESC LIMIT 1
        """)).mappings().first()
        if not creds:
            raise HTTPException(status_code=400, detail=f"No SyncCreds found for {schema}")

        target_host = creds["host"]
        target_port = creds["port"]
        target_user = creds["username"]
        target_pass = creds["password"] or ""
        target_dbname = current_db
        target_schema = schema

        print(f"🔗 Target: {target_user}@{target_host}:{target_port}/{target_dbname} → {target_schema}.JoinedTable")

        # === Fetch GIS data (now includes ID)
        rows = db.execute(text(f"""
            SELECT id, pin, bounds, computed_area
            FROM "{schema}"."JoinedTable"
            WHERE pin IS NOT NULL
        """)).fetchall()
        if not rows:
            return {"status": "empty", "message": "No data found in JoinedTable"}

        print(f"📦 Retrieved {len(rows)} rows from GIS.{schema}.JoinedTable")

        # === Connect to RPIS
        with psycopg.connect(
            dbname=target_dbname,
            user=target_user,
            password=target_pass,
            host=target_host,
            port=target_port
        ) as conn:
            with conn.cursor() as cur:
                # 1️⃣ Drop and recreate staging table
                cur.execute(f'DROP TABLE IF EXISTS "{target_schema}"."_push_staging";')
                cur.execute(f"""
                    CREATE TABLE "{target_schema}"."_push_staging" (
                        id INTEGER,
                        pin TEXT,
                        bounds DOUBLE PRECISION,
                        computed_area DOUBLE PRECISION
                    );
                """)

                # 2️⃣ Bulk insert GIS data
                placeholders = ", ".join(["(%s, %s, %s, %s)"] * len(rows))
                flat_values = [v for r in rows for v in (r[0], r[1], r[2], r[3])]
                cur.execute(f"""
                    INSERT INTO "{target_schema}"."_push_staging" (id, pin, bounds, computed_area)
                    VALUES {placeholders};
                """, flat_values)

                # 3️⃣ Delete RPIS rows that no longer exist in GIS
                cur.execute(f"""
                    DELETE FROM "{target_schema}"."JoinedTable"
                    WHERE id NOT IN (SELECT id FROM "{target_schema}"."_push_staging");
                """)

                # 4️⃣ Update existing RPIS rows (match by ID)
                cur.execute(f"""
                    UPDATE "{target_schema}"."JoinedTable" AS rpis
                    SET
                        pin = staging.pin,
                        bounds = staging.bounds,
                        computed_area = staging.computed_area
                    FROM "{target_schema}"."_push_staging" AS staging
                    WHERE rpis.id = staging.id;
                """)

                # 5️⃣ Insert new rows (new parcels)
                cur.execute(f"""
                    INSERT INTO "{target_schema}"."JoinedTable" (id, pin, bounds, computed_area)
                    SELECT s.id, s.pin, s.bounds, s.computed_area
                    FROM "{target_schema}"."_push_staging" s
                    WHERE NOT EXISTS (
                        SELECT 1 FROM "{target_schema}"."JoinedTable" r WHERE r.id = s.id
                    );
                """)

                # 6️⃣ Drop staging
                cur.execute(f'DROP TABLE IF EXISTS "{target_schema}"."_push_staging";')
                conn.commit()

        print(f"✅ Push complete — {len(rows)} records synced using ID match safely.")

        return {
            "status": "success",
            "message": f"Pushed {len(rows)} records successfully using ID match.",
            "count": len(rows)
        }

    except Exception as e:
        print(f"❌ Error in /sync-push: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🔹 4. PULL — Also use ID for updates
# ============================================================
@router.post("/sync-pull")
async def sync_pull(request: Request, db: Session = Depends(get_user_main_db)):
    """
    Pull data from RPIS JoinedTable and update GIS JoinedTable.
    Uses 'id' for matching so renamed PINs sync correctly.
    """
    data = await request.json()
    schema = data.get("schema")
    if not schema:
        raise HTTPException(status_code=400, detail="Schema is required.")

    try:
        current_db = db.execute(text("SELECT current_database()")).scalar()
        print(f"⬇️ PULL triggered for {current_db}.{schema}")

        creds = db.execute(text(f"""
            SELECT host, port, username, password
            FROM "{schema}"."SyncCreds"
            ORDER BY id DESC LIMIT 1
        """)).mappings().first()
        if not creds:
            raise HTTPException(status_code=400, detail=f"No SyncCreds found for {schema}")

        rpis_host, rpis_port = creds["host"], creds["port"]
        rpis_user, rpis_pass = creds["username"], creds["password"] or ""
        rpis_db = current_db

        with psycopg.connect(
            dbname=rpis_db,
            user=rpis_user,
            password=rpis_pass,
            host=rpis_host,
            port=rpis_port
        ) as conn_remote:
            with conn_remote.cursor() as cur:
                cur.execute(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = 'JoinedTable'
                """, (schema,))
                all_cols = [r[0] for r in cur.fetchall()]
                excluded = {"id", "pin", "bounds", "computed_area", "geom"}
                cols_to_update = [c for c in all_cols if c not in excluded]
                col_list = ", ".join(f'"{c}"' for c in all_cols)

                cur.execute(f'SELECT {col_list} FROM "{schema}"."JoinedTable" WHERE id IS NOT NULL')
                rows = cur.fetchall()
                colnames = [desc[0] for desc in cur.description]

        if not rows:
            return {"status": "empty", "message": "No rows found in RPIS JoinedTable"}

        print(f"📦 Retrieved {len(rows)} rows from RPIS.{schema}.JoinedTable")

        db.execute(text(f"""
            CREATE TABLE IF NOT EXISTS "{schema}"."_rpis_staging"
            (LIKE "{schema}"."JoinedTable" INCLUDING ALL);
        """))
        db.execute(text(f'TRUNCATE TABLE "{schema}"."_rpis_staging";'))
        db.commit()

        conn = db.connection().connection
        with conn.cursor() as cur:
            col_list_str = ", ".join(f'"{c}"' for c in colnames)
            placeholders = ", ".join(["(" + ",".join(["%s"] * len(colnames)) + ")"] * len(rows))
            flat_values = [v for row in rows for v in row]
            cur.execute(f'INSERT INTO "{schema}"."_rpis_staging" ({col_list_str}) VALUES {placeholders}', flat_values)
            conn.commit()

        set_clause = ", ".join([f'"{c}" = staging."{c}"' for c in cols_to_update])
        db.execute(text(f"""
            UPDATE "{schema}"."JoinedTable" AS gis
            SET {set_clause}
            FROM "{schema}"."_rpis_staging" AS staging
            WHERE gis.id = staging.id;
        """))
        db.commit()

        db.execute(text(f'TRUNCATE TABLE "{schema}"."_rpis_staging";'))
        db.commit()

        print(f"✅ Pull complete — updated {len(rows)} GIS records using ID match.")

        return {
            "status": "success",
            "message": f"Pulled {len(rows)} records successfully using ID match.",
            "updated": len(rows)
        }

    except Exception as e:
        db.rollback()
        print(f"❌ Error in /sync-pull: {e}")
        raise HTTPException(status_code=500, detail=str(e))
