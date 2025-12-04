# create_admin.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import bcrypt
import os
import socket
import time
from dotenv import load_dotenv

load_dotenv()

# Admin details
admin_data = {
    "first_name": "John",
    "last_name": "Lex",
    "email": "johnlloyd@example.com",
    "contact_number": "09090909090",
    "user_name": "admin1",
    "password": "#admin1"  # Change this!
}

# Try to resolve the host first
host = os.getenv('AUTH_DB_HOST')
print(f"🔍 Attempting to resolve host: {host}")

try:
    ip = socket.gethostbyname(host)
    print(f"✅ Host resolved to: {ip}")
except socket.gaierror as e:
    print(f"❌ DNS resolution failed: {e}")
    exit(1)

# Build connection URL
DB_URL = f"postgresql://{os.getenv('AUTH_DB_USER')}:{os.getenv('AUTH_DB_PASSWORD')}@{host}:{os.getenv('AUTH_DB_PORT')}/{os.getenv('AUTH_DB_NAME')}"

print(f"🔗 Connecting to database...")

try:
    engine = create_engine(DB_URL, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Test connection
    session.execute(text("SELECT 1"))
    print("✅ Database connection successful!")
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    exit(1)

# Hash the password
print(f"\n🔐 Hashing password...")
salt = bcrypt.gensalt()
hashed_password = bcrypt.hashpw(admin_data["password"].encode('utf-8'), salt).decode('utf-8')

# Create admin
try:
    print(f"👤 Creating admin user '{admin_data['user_name']}'...")
    
    # First check if user already exists
    existing = session.execute(
        text("SELECT user_name FROM credentials_users_schema.admin_login WHERE user_name = :username"),
        {"username": admin_data["user_name"]}
    ).fetchone()
    
    if existing:
        print(f"⚠️ Admin user '{admin_data['user_name']}' already exists!")
    else:
        # Get the next ID
        result = session.execute(
            text("SELECT COALESCE(MAX(id), 0) + 1 FROM credentials_users_schema.admin_login")
        ).fetchone()
        next_id = result[0]
        
        print(f"📝 Using ID: {next_id}")
        
        session.execute(
            text("""
            INSERT INTO credentials_users_schema.admin_login 
            (id, first_name, last_name, email, contact_number, user_name, password)
            VALUES (:id, :first_name, :last_name, :email, :contact_number, :user_name, :password)
            """),
            {
                "id": next_id,
                **admin_data,
                "password": hashed_password
            }
        )
        session.commit()
        print(f"\n✅ Admin user created successfully!")
        print(f"   ID: {next_id}")
        print(f"   Username: {admin_data['user_name']}")
        print(f"   Password: {admin_data['password']}")
        print("   ⚠️  Please change the password after first login!")
        
except Exception as e:
    print(f"❌ Error creating admin: {e}")
    session.rollback()
finally:
    session.close()