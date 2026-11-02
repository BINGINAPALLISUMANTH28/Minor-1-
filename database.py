import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "classroom.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(reset=False):
    if reset and os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except Exception:
            pass

    conn = get_connection()
    cursor = conn.cursor()

    # 1. Students Table (FR-01: ID, Name, Department, Section, Class, Status, Consent)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL DEFAULT 'Computer Science & Engineering',
            section TEXT NOT NULL DEFAULT 'Sec-A',
            class_name TEXT NOT NULL DEFAULT 'CSE - III Year',
            status TEXT DEFAULT 'ACTIVE',
            consent_status TEXT DEFAULT 'GRANTED',
            is_biometric_enrolled INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Face Encodings Table (Biometric Templates with Pose & Quality Scores)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            encoding_json TEXT NOT NULL,
            quality_score REAL DEFAULT 0.90,
            pose_type TEXT DEFAULT 'FRONTAL',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
        )
    ''')

    # 3. Sessions Table (Subject Period Schedule)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            faculty_name TEXT NOT NULL DEFAULT 'Dr. Faculty In-Charge',
            subject TEXT NOT NULL,
            slot_number TEXT NOT NULL DEFAULT 'Slot 1 (Period 1)',
            department TEXT NOT NULL DEFAULT 'Computer Science & Engineering',
            class_name TEXT NOT NULL DEFAULT 'CSE III Year - Sec A',
            date TEXT NOT NULL,
            scheduled_start TEXT NOT NULL DEFAULT '09:30 AM',
            scheduled_end TEXT NOT NULL DEFAULT '10:30 AM',
            actual_start TEXT,
            actual_end TEXT,
            min_duration_req_mins INTEGER DEFAULT 15,
            timezone TEXT DEFAULT 'IST (+05:30)',
            attendance_policy TEXT DEFAULT 'MIN_DURATION_15MIN',
            status TEXT DEFAULT 'ACTIVE'
        )
    ''')

    # 4. Session Roster Table (Roster eligibility)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session_roster (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            student_id TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE,
            UNIQUE(session_id, student_id)
        )
    ''')

    # 5. Attendance Table (Verified Duration, First/Last Sighting, Qualification)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            student_id TEXT NOT NULL,
            verified_duration_secs INTEGER DEFAULT 0,
            first_seen TEXT,
            last_seen TEXT,
            status TEXT DEFAULT 'Not yet verified',
            qualification_result TEXT DEFAULT 'PENDING',
            review_reason TEXT,
            marked_by TEXT DEFAULT 'SYSTEM',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE,
            UNIQUE(session_id, student_id)
        )
    ''')

    # 6. Observable Cue Logs (Replacing mental "attention" with observable cues)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            track_id INTEGER NOT NULL,
            student_id TEXT,
            confidence REAL DEFAULT 0.0,
            image_quality REAL DEFAULT 0.90,
            observable_cue TEXT NOT NULL DEFAULT 'FACE_VISIBLE_FORWARD',
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
        )
    ''')

    # 8. Users & Role-Based Auth Table (ADMIN & FACULTY)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'FACULTY',
            full_name TEXT NOT NULL,
            department TEXT DEFAULT 'Computer Science & Engineering',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Seed Default Administrator & Faculty Accounts if not present
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO users (username, password_hash, role, full_name, department)
            VALUES 
            ('admin', 'admin123', 'ADMIN', 'System Administrator', 'Information Technology'),
            ('faculty', 'faculty123', 'FACULTY', 'Dr. Faculty In-Charge', 'Computer Science & Engineering')
        ''')

    # 9. Recognition Logs Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            track_id INTEGER NOT NULL,
            student_id TEXT,
            confidence REAL DEFAULT 0.0,
            match_status TEXT NOT NULL DEFAULT 'UNKNOWN',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 10. Unknown Visitors Log Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS unknown_visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            track_id INTEGER NOT NULL,
            confidence REAL DEFAULT 0.0,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 11. System Configuration Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def wipe_database():
    """Resets database schema cleanly."""
    init_db(reset=True)

if __name__ == "__main__":
    init_db()
    print("Database initialized with RBAC users, security logs, and clean schema.")
