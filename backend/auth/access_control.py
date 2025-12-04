from typing import List, Dict, Optional
from auth.models import User
import re


class AccessControl:
    @staticmethod
    def normalize_code(code: str) -> str:
        """
        Normalize PSA code by stripping non-alphanumeric chars and removing suffixes.
        Example: "PH0403419_Pagsanjan" -> "PH0403419"
        """
        if not code:
            return ""
        match = re.match(r"^(PH\\d+)", code)
        return match.group(1) if match else code.strip()

    # =====================================================
    # ✅ MAIN ACCESS CHECK
    # =====================================================
    @staticmethod
    def check_user_access(user: User) -> Dict:
        if not user.provincial_access and not user.municipal_access:
            return {
                "status": "pending_approval",
                "message": "Your account needs admin approval. Please contact the administrator.",
                "allowed_schemas": []
            }

        if user.provincial_access and not user.municipal_access:
            return {
                "status": "pending_approval",
                "message": f"You have provincial access to {user.provincial_access}, but no municipal access assigned.",
                "allowed_schemas": []
            }

        return {
            "status": "approved",
            "message": "Access granted",
            "allowed_schemas": AccessControl.get_allowed_schemas(user)
        }

    # =====================================================
    # ✅ FILTER SCHEMAS BASED ON USER ACCESS
    # =====================================================
    @staticmethod
    def filter_schemas_by_access(all_schemas: List[str], user: User) -> List[str]:
        if not user.provincial_access or not user.municipal_access:
            return []

        prov_code = AccessControl.normalize_code(user.provincial_access)
        mun_code = AccessControl.normalize_code(user.municipal_access)

        # Allow all municipalities under province
        if user.municipal_access.strip().lower() == "all":
            return sorted([s for s in all_schemas if s.startswith(prov_code)])

        # Otherwise match by municipal prefix (ignoring suffix)
        return sorted([s for s in all_schemas if s.startswith(mun_code)])

    # =====================================================
    # ✅ VALIDATE SPECIFIC SCHEMA
    # =====================================================
    @staticmethod
    def validate_schema_access(schema: str, user: User) -> bool:
        if not user.provincial_access or not user.municipal_access:
            return False

        schema_code = AccessControl.normalize_code(schema)
        prov_code = AccessControl.normalize_code(user.provincial_access)
        mun_code = AccessControl.normalize_code(user.municipal_access)

        if user.municipal_access.strip().lower() == "all":
            return schema_code.startswith(prov_code)
        return schema_code.startswith(mun_code)

    # =====================================================
    # ✅ HUMAN DESCRIPTION (UPDATED)
    # =====================================================
    @staticmethod
    def get_access_description(user: User) -> str:
        """
        Generates a human-readable description of the user's access level.
        Dynamically fetches the actual provincial database name if access is 'All'.
        """
        if not user.provincial_access and not user.municipal_access:
            return "No access assigned"

        if user.provincial_access and not user.municipal_access:
            return f"Provincial access {user.provincial_access} only (no municipal access)"

        if user.municipal_access.strip().lower() == "all":
            # ✅ Dynamically get the connected provincial DB name
            try:
                from sqlalchemy import text
                # Delayed import to avoid circular dependency
                from auth.dependencies import get_user_database_session

                db = get_user_database_session(user.provincial_access)
                result = db.execute(text("SELECT current_database()")).scalar()
                db_name = result or user.provincial_access
            except Exception:
                db_name = user.provincial_access
            finally:
                if "db" in locals():
                    db.close()

            return f"Full access to all municipalities under {db_name}"

        return f"Access limited to municipality {user.municipal_access}"

    # =====================================================
    # ✅ RETURN ALLOWED SCHEMAS
    # =====================================================
    @staticmethod
    def get_allowed_schemas(user: User, all_schemas: Optional[List[str]] = None) -> List[str]:
        if not user.provincial_access or not user.municipal_access:
            return []

        prov_code = AccessControl.normalize_code(user.provincial_access)
        mun_code = AccessControl.normalize_code(user.municipal_access)

        if user.municipal_access.strip().lower() == "all":
            if all_schemas is not None:
                return sorted([s for s in all_schemas if s.startswith(prov_code)])
            return ["*"]

        if all_schemas is not None:
            return sorted([s for s in all_schemas if s.startswith(mun_code)])

        return [user.municipal_access]
