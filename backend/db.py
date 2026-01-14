from typing import Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from common_db_runtime import get_request_session


def get_user_database_session(provincial_code: Optional[str] = None) -> Session:
    """
    Seamless Phase 1:
    - Ignore provincial_code (legacy argument)
    - Use the db/schema resolved from the GIS token (set by middleware in main.py)
    """
    try:
        return get_request_session()
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"DB error: {e}",
        )
