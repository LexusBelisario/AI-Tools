from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from auth.dependencies import get_user_main_db, get_current_user
from auth.models import User

router = APIRouter(prefix="/tmcr", tags=["TMCR Report"])

@router.get("/generate")
def generate_tmcr_report(
    schema: str,
    barangay: str,
    section: str,
    db: Session = Depends(get_user_main_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate TMCR report using COMPLETE JoinedTable fields.
    """

    try:
        full_table = f'"{schema}"."JoinedTable"'

        sql = text(f"""
            SELECT 
                -- HEADER FIELDS (IMPORTANT!)
                province,
                prov_code,
                municipal,
                mun_code,
                brgy_nm,
                brgy_code,

                -- SECTION FIELD
                sect_code,

                -- TABLE FIELDS
                parcel_cod,
                lot_no,
                blk_no,
                tct_no,
                land_area,
                land_class,

                (COALESCE(l_lastname,'') || ', ' || COALESCE(l_frstname,'') || ' ' || COALESCE(l_midname,'')) AS owner_name,

                land_arpn,
                bldg_class

            FROM {full_table}
            WHERE lower(brgy_nm) = lower(:barangay)
              AND lower(sect_code) = lower(:section)
            ORDER BY parcel_cod::int NULLS LAST;
        """)

        rows = db.execute(sql, {
            "barangay": barangay,
            "section": section
        }).mappings().all()

        return {
            "status": "success",
            "count": len(rows),
            "data": rows
        }

    except Exception as e:
        print("TMCR ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
