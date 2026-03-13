from typing import Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from common_db_runtime import get_request_session


def get_user_database_session(provincial_code: Optional[str] = None) -> Session:
    """
    Returns a database session based on the token context
    resolved by the middleware in main.py.
    The provincial_code argument is kept for backward compatibility but ignored.
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