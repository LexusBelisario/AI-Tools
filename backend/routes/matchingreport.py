# routes/matchingreport.py
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
import psycopg
from auth.dependencies import get_user_main_db

router = APIRouter()

# ============================================================
# 📊 MATCHING REPORT — Barangay-level Summary
# ============================================================
@router.get("/matching-report")
def matching_report(schema: str, db: Session = Depends(get_user_main_db)):
    """Compares GIS vs RPIS by barangay and returns counts and ratios."""
    try:
        current_db = db.execute(text("SELECT current_database()")).scalar()
        print(f"📊 Running Matching Report for {current_db}.{schema}")

        creds = db.execute(text(f"""
            SELECT host, port, username, password
            FROM "{schema}"."SyncCreds"
            ORDER BY id DESC LIMIT 1
        """)).mappings().first()
        if not creds:
            raise HTTPException(status_code=400, detail=f"No SyncCreds found for schema {schema}")

        host, port, user, password = creds["host"], creds["port"], creds["username"], creds["password"] or ""

        # --- Fetch GIS barangay counts
        gis_data = db.execute(text(f"""
            SELECT brgy_code, brgy_nm, COUNT(pin) AS gdb_count
            FROM "{schema}"."JoinedTable"
            WHERE pin IS NOT NULL
            GROUP BY brgy_code, brgy_nm
        """)).mappings().all()
        gis_dict = {row["brgy_code"]: row for row in gis_data}

        # --- Fetch RPIS barangay counts (remote)
        with psycopg.connect(
            dbname=current_db, user=user, password=password, host=host, port=port
        ) as conn_remote:
            with conn_remote.cursor() as cur:
                cur.execute(f"""
                    SELECT brgy_code, brgy_nm, COUNT(pin) AS rdb_count
                    FROM "{schema}"."JoinedTable"
                    WHERE pin IS NOT NULL
                    GROUP BY brgy_code, brgy_nm
                """)
                rpis_rows = cur.fetchall()
                rpis_dict = {r[0]: {"brgy_code": r[0], "brgy_nm": r[1], "rdb_count": r[2]} for r in rpis_rows}

        # --- Combine comparison
        matches = []
        for brgy_code in set(gis_dict.keys()) | set(rpis_dict.keys()):
            brgy_nm = (gis_dict.get(brgy_code) or rpis_dict.get(brgy_code) or {}).get("brgy_nm", "")
            gdb_count = gis_dict.get(brgy_code, {}).get("gdb_count", 0)
            rdb_count = rpis_dict.get(brgy_code, {}).get("rdb_count", 0)

            gis_pins = [
                row[0]
                for row in db.execute(text(f'''
                    SELECT pin FROM "{schema}"."JoinedTable"
                    WHERE brgy_code = :brgy_code AND pin IS NOT NULL
                '''), {"brgy_code": brgy_code})
            ]

            with psycopg.connect(
                dbname=current_db, user=user, password=password, host=host, port=port
            ) as conn_remote:
                with conn_remote.cursor() as cur:
                    cur.execute(f'''
                        SELECT pin FROM "{schema}"."JoinedTable"
                        WHERE brgy_code = %s AND pin IS NOT NULL
                    ''', (brgy_code,))
                    rdb_pins = [r[0] for r in cur.fetchall()]

            total_match = len(set(gis_pins).intersection(set(rdb_pins)))
            diff = rdb_count - gdb_count
            rdb_over_gdb = (rdb_count / gdb_count * 100) if gdb_count else 0
            match_over_gdb = (total_match / gdb_count * 100) if gdb_count else 0
            match_over_rdb = (total_match / rdb_count * 100) if rdb_count else 0

            matches.append({
                "code": brgy_code,
                "barangay": brgy_nm,
                "gdb": gdb_count,
                "rdb": rdb_count,
                "rdb_minus_gdb": diff,
                "total_match": total_match,
                "rdb_over_gdb": round(rdb_over_gdb, 2),
                "match_over_gdb": round(match_over_gdb, 2),
                "match_over_rdb": round(match_over_rdb, 2)
            })

        return {"status": "success", "count": len(matches), "data": matches}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating matching report for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 📋 GDB MISMATCH — GIS PINs Missing in RPIS (with barangay info)
# ============================================================
@router.get("/matching-report/gdb-mismatch")
def gdb_mismatch(schema: str, db: Session = Depends(get_user_main_db)):
    """Returns all GIS PINs missing from RPIS, including barangay info if present."""
    try:
        current_db = db.execute(text("SELECT current_database()")).scalar()
        print(f"📍 Running GDB Mismatch check for {current_db}.{schema}")

        creds = db.execute(text(f"""
            SELECT host, port, username, password
            FROM "{schema}"."SyncCreds"
            ORDER BY id DESC LIMIT 1
        """)).mappings().first()
        if not creds:
            raise HTTPException(status_code=400, detail=f"No SyncCreds found for schema {schema}")

        host, port, user, password = creds["host"], creds["port"], creds["username"], creds["password"] or ""

        # --- Detect barangay columns dynamically
        cols = db.execute(text(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = 'JoinedTable'
        """), {"schema": schema}).scalars().all()

        available_cols = ["pin"]
        for col in ["brgy_nm", "brgy_code"]:
            if col in cols:
                available_cols.append(col)

        select_cols = ", ".join(f'"{c}"' for c in available_cols)
        print(f"✅ Using columns for GDB mismatch: {available_cols}")

        # --- Fetch GIS records
        gis_rows = db.execute(text(f"""
            SELECT {select_cols}
            FROM "{schema}"."JoinedTable"
            WHERE pin IS NOT NULL
        """)).mappings().all()

        # --- Fetch RPIS PINs
        with psycopg.connect(
            dbname=current_db, user=user, password=password, host=host, port=port
        ) as conn_remote:
            with conn_remote.cursor() as cur:
                cur.execute(f"""
                    SELECT pin FROM "{schema}"."JoinedTable"
                    WHERE pin IS NOT NULL
                """)
                rdb_pins = [r[0] for r in cur.fetchall()]
        rdb_pin_set = set(rdb_pins)

        # --- Compare sets
        mismatches = [row for row in gis_rows if row["pin"] not in rdb_pin_set]
        print(f"🚫 Found {len(mismatches)} GIS records missing in RPIS")

        return {"status": "success", "count": len(mismatches), "data": mismatches}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching GDB mismatches for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 📋 RDB MISMATCH — RPIS PINs Missing in GIS (Exclude geom/id)
# ============================================================
@router.get("/matching-report/rdb-mismatch")
def rdb_mismatch(schema: str, db: Session = Depends(get_user_main_db)):
    """Returns RPIS records missing in GIS, excluding geometry and ID columns."""
    try:
        current_db = db.execute(text("SELECT current_database()")).scalar()
        print(f"📍 Running RDB Mismatch check for {current_db}.{schema}")

        creds = db.execute(text(f"""
            SELECT host, port, username, password
            FROM "{schema}"."SyncCreds"
            ORDER BY id DESC LIMIT 1
        """)).mappings().first()
        if not creds:
            raise HTTPException(status_code=400, detail=f"No SyncCreds found for schema {schema}")

        host, port, user, password = creds["host"], creds["port"], creds["username"], creds["password"] or ""

        # --- Fetch GIS PINs locally
        gis_pins = db.execute(text(f"""
            SELECT pin FROM "{schema}"."JoinedTable"
            WHERE pin IS NOT NULL
        """)).scalars().all()
        gis_pin_set = set(gis_pins)
        print(f"🗺️ Retrieved {len(gis_pins)} GIS PINs")

        # --- Connect to RPIS and fetch non-geom columns
        with psycopg.connect(
            dbname=current_db, user=user, password=password, host=host, port=port
        ) as conn_remote:
            with conn_remote.cursor() as cur:
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = %s
                    AND table_name = 'JoinedTable'
                    AND column_name NOT ILIKE '%%geom%%'
                    AND column_name NOT ILIKE '%%id%%'
                    ORDER BY ordinal_position
                """, (schema,))
                colnames = [r[0] for r in cur.fetchall()]
                print(f"✅ Retrieved {len(colnames)} non-geometry columns from RPIS schema {schema}")

                select_cols = ", ".join(f'"{c}"' for c in colnames)
                cur.execute(f"""
                    SELECT {select_cols}
                    FROM "{schema}"."JoinedTable"
                    WHERE pin IS NOT NULL
                """)
                rows = cur.fetchall()

                # --- Filter records missing in GIS
                data = []
                for row in rows:
                    record = dict(zip(colnames, row))
                    if record.get("pin") not in gis_pin_set:
                        data.append(record)

        print(f"🚫 Found {len(data)} RPIS records missing in GIS")
        return {"status": "success", "count": len(data), "columns": colnames, "data": data}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching RDB mismatches for {schema}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
