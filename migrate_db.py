import os
import sqlite3
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "classroom.db")

def verify_and_dump_sqlite():
    """Verifies SQLite database integrity and prints summary stats."""
    if not os.path.exists(DB_PATH):
        print("SQLite Database does not exist yet. It will be initialized on server start.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    tables = ['students', 'face_encodings', 'sessions', 'session_roster', 'attendance', 'observations', 'audit_logs']
    print("\n--- DATABASE VERIFICATION SUMMARY ---")
    for t in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {t}")
            cnt = cursor.fetchone()[0]
            print(f"Table '{t}': {cnt} records")
        except sqlite3.OperationalError:
            print(f"Table '{t}': Missing")

    conn.close()

if __name__ == '__main__':
    print("Verifying Smart Classroom Monitoring Database...")
    verify_and_dump_sqlite()
