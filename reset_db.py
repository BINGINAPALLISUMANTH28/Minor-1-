import os
import sqlite3
from database import init_db, get_connection

def reset_and_clean_database():
    """Wipes all sample/mock students and preloaded records, creating a clean production schema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Delete all preloaded/mock students, encodings, sessions, attendance, observations
    tables = ['students', 'face_encodings', 'sessions', 'session_roster', 'attendance', 'observations', 'audit_logs', 'recognition_logs', 'unknown_visitors']
    for t in tables:
        try:
            cursor.execute(f"DELETE FROM {t}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

    init_db()
    print("Database purged of all mock/sample data. Clean production database initialized.")

if __name__ == '__main__':
    reset_and_clean_database()
