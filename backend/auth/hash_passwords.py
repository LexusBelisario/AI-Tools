import bcrypt
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

def hash_existing_passwords():
    conn = psycopg2.connect(
        dbname=os.getenv("AUTH_DB_NAME"),
        user=os.getenv("AUTH_DB_USER"),
        password=os.getenv("AUTH_DB_PASSWORD"),
        host=os.getenv("AUTH_DB_HOST"),
        port=os.getenv("AUTH_DB_PORT")
    )
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT user_name, password FROM credentials_users_schema.users_table")
            users = cursor.fetchall()
            
            for user in users:
                if not user['password'].startswith('$2b$'):
                    # Hash the password
                    hashed = bcrypt.hashpw(
                        user['password'].encode('utf-8'), 
                        bcrypt.gensalt()
                    ).decode('utf-8')
                    
                    # Update the password
                    cursor.execute(
                        "UPDATE credentials_users_schema.users_table SET password = %s WHERE user_name = %s",
                        (hashed, user['user_name'])
                    )
                    print(f"Updated password for user: {user['user_name']}")
            
            conn.commit()
            print("All passwords updated successfully!")
            
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    hash_existing_passwords()