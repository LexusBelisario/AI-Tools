from typing import Optional
from sqlalchemy.orm import Session

from common_db_runtime import connect_common_db, get_common_db_session

COMMON_KEY = "COMMON"


def get_user_database_session(provincial_code: Optional[str] = None) -> Session:
    session_pair = get_common_db_session(COMMON_KEY)
    if session_pair is None:
        connect_common_db(COMMON_KEY)
        session_pair = get_common_db_session(COMMON_KEY)

    if session_pair is None:
        raise RuntimeError("Common DB is not connected.")

    session, _meta = session_pair
    return session
