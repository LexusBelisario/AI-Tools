from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from auth.models import User
from auth.access_control import AccessControl
from auth.dependencies import get_current_user, get_user_main_db

router = APIRouter()


# ============================================================
# 📜 List all accessible schemas for the logged-in user
# ============================================================
@router.get("/list-schemas")
def list_schemas(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_user_main_db)
):
    """
    Retrieve all allowed schemas from the provincial database for the logged-in user.
    Includes the actual connected database name (e.g., PH04034_Laguna).
    """

    print("=== 🧩 LIST SCHEMAS REQUEST ===")
    print(f"User: {current_user.user_name}")
    print(f"Provincial Access (Code): {current_user.provincial_access}")
    print(f"Municipal Access: {current_user.municipal_access}")

    # ✅ Check access validity
    access_info = AccessControl.check_user_access(current_user)
    if access_info["status"] == "pending_approval":
        raise HTTPException(status_code=403, detail=access_info["message"])

    try:
        # ✅ Query all non-system schemas
        query = text("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN (
                'information_schema', 'pg_catalog', 'pg_toast', 'public',
                'credentials_login', 'auth', 'storage', 'vault',
                'graphql', 'graphql_public', 'realtime', 'extensions',
                'pgbouncer', 'postgres', 'credentials_users_schema'
            )
            AND schema_name NOT LIKE 'pg_%'
            AND schema_name NOT LIKE '%credential%'
            ORDER BY schema_name;
        """)
        result = db.execute(query)
        all_schemas = [row[0] for row in result]

        print(f"📦 All Schemas Found: {len(all_schemas)}")
        print(f"All Schemas: {all_schemas}")

        # ✅ Filter by user access rules
        allowed_schemas = AccessControl.filter_schemas_by_access(all_schemas, current_user)
        allowed_schemas = [s.replace(", ", "_").replace(",", "_") for s in allowed_schemas]

        print(f"✅ Allowed Schemas: {allowed_schemas}")

        # ✅ Access description
        access_description = AccessControl.get_access_description(current_user)

        # ✅ Determine actual connected database name
        try:
            db_result = db.execute(text("SELECT current_database()"))
            actual_dbname = db_result.scalar() or getattr(current_user, "actual_dbname", None)
        except Exception:
            actual_dbname = getattr(current_user, "actual_dbname", None)

        # ✅ Construct final response
        response = {
            "schemas": allowed_schemas,
            "total_accessible": len(allowed_schemas),
            "user_access": {
                "provincial": current_user.provincial_access,
                "municipal": current_user.municipal_access,
                "description": access_description,
                "actual_dbname": actual_dbname,
            },
        }

        print(f"🌐 Actual Connected Database: {actual_dbname}")
        print(f"=== ✅ LIST SCHEMAS SUCCESS ===")

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error listing schemas: {e}")
        raise HTTPException(status_code=500, detail=f"Schema listing error: {str(e)}")
