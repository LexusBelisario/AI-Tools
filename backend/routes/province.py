from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from auth.dependencies import get_user_main_db   # ✅ correct source

router = APIRouter(prefix="/province", tags=["Province Data"])

@router.get("/provincial-bounds")
def get_provincial_bounds(db: Session = Depends(get_user_main_db)):
    """
    Fetch the bounding box for the current provincial database.
    Uses the 'bounds' column from public."PH_ProvincialMap",
    matching the current database's provincial code.
    """
    try:
        # 🔹 Step 1: Identify current database (e.g. "PH04034_Laguna")
        result = db.execute(text("SELECT current_database()"))
        db_name = result.scalar()
        if not db_name:
            raise HTTPException(status_code=500, detail="Failed to detect current database name.")

        # 🔹 Step 2: Extract provincial code (e.g. "PH04034" from "PH04034_Laguna")
        prov_code = db_name.split("_")[0]

        # 🔹 Step 3: Query PH_ProvincialMap for bounds
        result = db.execute(
            text("""
                SELECT prov_code, bounds
                FROM public."PH_ProvincialMap"
                WHERE prov_code = :prov_code
                LIMIT 1
            """),
            {"prov_code": prov_code}
        )
        row = result.fetchone()

        if not row or not row.bounds:
            raise HTTPException(
                status_code=404,
                detail=f"No bounding box found for province {prov_code}."
            )

        # 🔹 Step 4: Parse the bounds string into float list
        try:
            bounds_values = [float(x.strip()) for x in row.bounds.split(",")]
            if len(bounds_values) != 4:
                raise ValueError
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid bounds format for province {prov_code}: {row.bounds}"
            )

        # 🔹 Step 5: Return formatted response
        return {
            "status": "success",
            "prov_code": row.prov_code,
            "bounds": bounds_values
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in get_provincial_bounds: {e}")
        raise HTTPException(status_code=500, detail=str(e))
