from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from auth.dependencies import get_current_user, get_user_main_db
from auth.models import User

router = APIRouter()

# ==========================================================
# 🧾 Get parcel information by PIN
# ==========================================================
@router.get("/parcel-info")
def get_parcel_info(
    pin: str,
    schema: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_user_main_db)
):
    try:
        # ✅ Query parcel data from JoinedTable
        result = db.execute(
            text(f'''
                SELECT * FROM "{schema}"."JoinedTable"
                WHERE pin = :pin
                LIMIT 1
            '''),
            {"pin": pin}
        )
        row = result.fetchone()

        if not row:
            return {"status": "error", "message": "Parcel not found."}

        # ✅ Convert row to dictionary
        data = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
        return {"status": "success", "data": data}

    except Exception as e:
        print(f"❌ Error retrieving parcel info for {pin} in {schema}: {e}")
        return {"status": "error", "message": str(e)}
