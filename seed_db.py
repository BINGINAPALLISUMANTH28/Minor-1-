import sqlite3
from database import init_db, get_connection

def seed_database():
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    default_students = [
        {'id': 'S001', 'name': 'Ravi Kumar', 'class': 'CSE - III Year'},
        {'id': 'S002', 'name': 'Sumanth', 'class': 'CSE - III Year'},
        {'id': 'S003', 'name': 'Kiran', 'class': 'CSE - III Year'},
        {'id': 'S004', 'name': 'Priya Sharma', 'class': 'CSE - III Year'},
        {'id': 'S005', 'name': 'Ananya Roy', 'class': 'CSE - III Year'}
    ]

    for s in default_students:
        cursor.execute('''
            INSERT INTO students (student_id, name, class_name, is_biometric_enrolled)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(student_id) DO UPDATE SET
                name = excluded.name,
                class_name = excluded.class_name
        ''', (s['id'], s['name'], s['class']))

    conn.commit()
    conn.close()

    print("Sample student profiles added without face templates. Capture each person's real face from the Students page before using recognition.")

if __name__ == "__main__":
    seed_database()
