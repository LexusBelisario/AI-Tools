from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
from typing import Generator
import jwt
import os

from db import get_auth_db, get_user_database_session
from auth.models import User, Admin
from auth.access_control import AccessControl

# ============================================================
# 🔐 JWT and Security Setup
# ============================================================
security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY", "secret_ngani")
ALGORITHM = "HS256"


# ============================================================
# 🧠 AUTH HELPERS
# ============================================================

async def get_current_user_or_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_db: Session = Depends(get_auth_db)
):
    """Return the current authenticated user (regular or admin)."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_type = payload.get("user_type")
        user_id = payload.get("user_id")

        if not user_id or not user_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token."
            )

        # ✅ Handle Admin login
        if user_type == "admin":
            admin = auth_db.query(Admin).filter(Admin.id == user_id).first()
            if not admin:
                raise HTTPException(status_code=401, detail="Admin not found.")
            admin.user_type = "admin"
            admin.provincial_access = "postgres"  # default for admin
            return admin

        # ✅ Handle Regular User login
        user = auth_db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found.")
        user.user_type = "user"
        return user

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_db: Session = Depends(get_auth_db)
) -> User:
    """Return current user (non-admin)."""
    user_or_admin = await get_current_user_or_admin(credentials, auth_db)
    if getattr(user_or_admin, "user_type", None) == "admin":
        raise HTTPException(status_code=403, detail="Admins not allowed for this route.")
    return user_or_admin


async def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_db: Session = Depends(get_auth_db)
) -> Admin:
    """Return current admin (non-user)."""
    user_or_admin = await get_current_user_or_admin(credentials, auth_db)
    if getattr(user_or_admin, "user_type", None) != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user_or_admin


# ============================================================
# 🗺️ DATABASE CONNECTIONS
# ============================================================

def get_user_main_db(current_user: User = Depends(get_current_user)) -> Generator[Session, None, None]:
    """
    Route user to their provincial database.
    Automatically detects and attaches the actual connected DB name
    (e.g. 'PH04034_Laguna') for use in frontend and logs.
    """
    access_info = AccessControl.check_user_access(current_user)

    if not current_user.provincial_access:
        raise HTTPException(
            status_code=403,
            detail="No provincial access assigned. Please contact administrator."
        )

    db = None
    try:
        # ✅ Step 1. Connect using the provincial PSA code (e.g. PH04034)
        db = get_user_database_session(current_user.provincial_access)

        # ✅ Step 2. Detect the actual DB name
        result = db.execute(text("SELECT current_database()"))
        actual_dbname = result.scalar() or "unknown"

        # ✅ Step 3. Attach to current_user for use in /list-schemas
        setattr(current_user, "actual_dbname", actual_dbname)

        # ✅ Step 4. Logging
        print("=== DATABASE CONNECTION ===")
        print(f"User: {current_user.user_name}")
        print(f"Provincial Access (PSA): {current_user.provincial_access}")
        print(f"Connected Database: {actual_dbname}")
        print(f"Municipal Access: {current_user.municipal_access}")
        print(f"Access Status: {access_info['status']}")
        print("============================")

        yield db

    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database connection error: {str(e)}"
        )
    finally:
        if db:
            db.close()


def get_user_or_admin_db(
    current_user_or_admin=Depends(get_current_user_or_admin)
) -> Generator[Session, None, None]:
    """
    Allow both user and admin to connect to a provincial database.
    """
    provincial_access = getattr(current_user_or_admin, "provincial_access", None)
    if not provincial_access:
        raise HTTPException(status_code=403, detail="No provincial access assigned.")

    db = None
    try:
        db = get_user_database_session(provincial_access)
        result = db.execute(text("SELECT current_database()"))
        current_db = result.scalar()
        username = getattr(current_user_or_admin, "user_name", "Unknown")

        print(f"✅ {username} connected to database: {current_db}")
        yield db

    except Exception as e:
        print(f"❌ Database connection error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database connection error: {str(e)}"
        )
    finally:
        if db:
            db.close()


# ============================================================
# ✅ ACCESS VALIDATION
# ============================================================

def require_approved_access(current_user: User = Depends(get_current_user)) -> User:
    """Ensure user has approved access before allowing route usage."""
    access_info = AccessControl.check_user_access(current_user)
    if access_info["status"] != "approved":
        raise HTTPException(status_code=403, detail=access_info["message"])
    return current_user
