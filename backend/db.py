from sqlalchemy import create_engine, text, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Dict
import os
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

# Import models for credentials lookup
from auth.models import Credentials

load_dotenv()

# Base for main DB models
Base = declarative_base()

# Supabase connection details (auth DB only)
DB_HOST = 'aws-1-ap-southeast-1.pooler.supabase.com'
DB_PORT = '6543'
DB_USER = 'postgres.ljmlswhybxcmwzdzuxhl'
DB_PASSWORD = '#IGDIwebapp'

# Auth database connection - pointing to credentials_login database
AUTH_DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/credentials_login"

auth_engine = create_engine(AUTH_DB_URL, pool_pre_ping=True)
AuthSessionLocal = sessionmaker(bind=auth_engine)

# Cache for database engines
_db_engines: Dict[str, any] = {}


def get_auth_db():
    """Get session for auth database (credentials_login)"""
    db = AuthSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_database_engine_from_credentials(creds: Credentials):
    """Create or get cached engine for given credentials row"""
    key = f"{creds.host}:{creds.port}/{creds.dbname}"
    if key not in _db_engines:
        db_url = f"postgresql://{creds.user}:{creds.password}@{creds.host}:{creds.port}/{creds.dbname}"
        _db_engines[key] = create_engine(db_url, pool_pre_ping=True)
        print(f"✅ Created engine for {key}")
    return _db_engines[key]


def get_user_database_session(provincial_access: str) -> Session:
    """
    Get database session based on user's provincial access.
    provincial_access will be a PSA code only (e.g. "PH04034"),
    but the actual DB is named like "PH04034_Laguna".
    """
    if not provincial_access:
        raise ValueError("User has no provincial access assigned")

    auth_db = AuthSessionLocal()
    try:
        # Match dbname by prefix (PSA code)
        creds = auth_db.query(Credentials).filter(
            Credentials.dbname.like(f"{provincial_access}%")
        ).first()

        if not creds:
            raise ValueError(
                f"No credentials found for provincial_access (PSA code): {provincial_access}"
            )

        # Step 2: Create/get engine for that DB
        engine = get_database_engine_from_credentials(creds)

        # Step 3: Return SQLAlchemy session
        SessionLocal = sessionmaker(bind=engine)
        return SessionLocal()
    finally:
        auth_db.close()


# Legacy functions (kept for backward compatibility but should be phased out)
def get_main_db_connection():
    """Legacy function - returns postgres database connection"""
    creds = Credentials(
        id=0,
        host=DB_HOST,
        port=DB_PORT,
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD
    )
    engine = get_database_engine_from_credentials(creds)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_main_engine():
    """Legacy function - returns postgres engine"""
    creds = Credentials(
        id=0,
        host=DB_HOST,
        port=DB_PORT,
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD
    )
    return get_database_engine_from_credentials(creds)


def get_engine():
    """Legacy function - alias for postgres engine"""
    return get_main_engine()


def get_connection(database_name: str = 'postgres'):
    """
    Legacy connection method - returns psycopg connection with dict_row factory
    This ensures compatibility with consolidate.py
    """
    return psycopg.connect(
        dbname=database_name,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        row_factory=dict_row
    )


# Test auth database connection on startup
try:
    with auth_engine.connect() as conn:
        result = conn.execute(text("SELECT current_database()"))
        current_db = result.scalar()
        print(f"✅ Auth database connection successful to: {current_db}")
        
        # Verify tables exist
        result = conn.execute(text("""
            SELECT table_schema, table_name 
            FROM information_schema.tables 
            WHERE table_name IN ('users_table', 'admin_login', 'credentials')
            AND table_schema = 'credentials_users_schema'
            ORDER BY table_name;
        """))
        tables = result.fetchall()
        if tables:
            print("✅ Found authentication tables:")
            for schema, table in tables:
                print(f"  - {schema}.{table}")
        else:
            print("⚠️ Warning: Authentication tables not found in credentials_users_schema")
            
except Exception as e:
    print(f"⚠️ Failed to connect to auth database: {e}")
