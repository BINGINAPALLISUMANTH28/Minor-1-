import sqlite3
import os
import secrets
from werkzeug.security import generate_password_hash

DB_PATH = os.environ.get('CLASSROOM_DB_PATH') or os.path.join(os.path.dirname(__file__), "data", "classroom.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
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
            is_biometric_enrolled INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Optional consent-backed frontal thumbnail for the student profile UI.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS student_face_photos (
            student_id TEXT PRIMARY KEY,
            image_data BLOB NOT NULL,
            mime_type TEXT NOT NULL DEFAULT 'image/jpeg',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
        )
    ''')

    # 2. Face Encodings Table (Biometric Templates with Pose & Quality Scores)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            encoding_json TEXT NOT NULL,
            model_version TEXT NOT NULL DEFAULT 'legacy_hog',
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
            section TEXT NOT NULL DEFAULT 'Sec-A',
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
            last_observation_epoch REAL,
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

    # Audit trail used by session lifecycle, overrides, and the timeline endpoint.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            action TEXT NOT NULL,
            user TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Add fields to existing SQLite databases without dropping attendance history.
    attendance_columns = {row['name'] for row in cursor.execute("PRAGMA table_info(attendance)")}
    if 'last_observation_epoch' not in attendance_columns:
        cursor.execute("ALTER TABLE attendance ADD COLUMN last_observation_epoch REAL")

    session_columns = {row['name'] for row in cursor.execute("PRAGMA table_info(sessions)")}
    if 'section' not in session_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN section TEXT NOT NULL DEFAULT 'Sec-A'")

    face_encoding_columns = {row['name'] for row in cursor.execute("PRAGMA table_info(face_encodings)")}
    if 'model_version' not in face_encoding_columns:
        cursor.execute("ALTER TABLE face_encodings ADD COLUMN model_version TEXT NOT NULL DEFAULT 'legacy_hog'")

    observation_columns = {row['name'] for row in cursor.execute("PRAGMA table_info(observations)")}
    if 'obs_id' not in observation_columns:
        cursor.execute("ALTER TABLE observations ADD COLUMN obs_id INTEGER")
        if 'id' in observation_columns:
            cursor.execute("UPDATE observations SET obs_id = id WHERE obs_id IS NULL")
    if 'image_quality' not in observation_columns:
        cursor.execute("ALTER TABLE observations ADD COLUMN image_quality REAL DEFAULT 0.90")
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS observations_fill_obs_id
        AFTER INSERT ON observations
        WHEN NEW.obs_id IS NULL
        BEGIN
            UPDATE observations SET obs_id = NEW.rowid WHERE rowid = NEW.rowid;
        END
    ''')

    audit_columns = {row['name'] for row in cursor.execute("PRAGMA table_info(audit_logs)")}
    if not {'audit_id', 'session_id', 'action', 'user', 'details', 'timestamp'}.issubset(audit_columns):
        cursor.execute("ALTER TABLE audit_logs RENAME TO audit_logs_legacy")
        cursor.execute('''
            CREATE TABLE audit_logs (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                action TEXT NOT NULL,
                user TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        legacy_columns = {row['name'] for row in cursor.execute("PRAGMA table_info(audit_logs_legacy)")}
        def legacy_column(*names, fallback='NULL'):
            for name in names:
                if name in legacy_columns:
                    return f'"{name}"'
            return fallback
        cursor.execute(f'''
            INSERT INTO audit_logs (audit_id, session_id, action, user, details, timestamp)
            SELECT {legacy_column('audit_id', 'id', fallback='NULL')},
                   {legacy_column('session_id')},
                   {legacy_column('action', fallback="'LEGACY'" )},
                   COALESCE({legacy_column('user', 'user_name', fallback="'SYSTEM'")}, 'SYSTEM'),
                   {legacy_column('details')},
                   COALESCE({legacy_column('timestamp', 'created_at', fallback='CURRENT_TIMESTAMP')}, CURRENT_TIMESTAMP)
            FROM audit_logs_legacy
        ''')
        cursor.execute("DROP TABLE audit_logs_legacy")

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

    # Create secure initial accounts only when no users exist. Their one-time
    # passwords are printed to the server console; deployments can set env vars.
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        admin_password = os.environ.get('ADMIN_INITIAL_PASSWORD') or secrets.token_urlsafe(18)
        faculty_password = os.environ.get('FACULTY_INITIAL_PASSWORD') or secrets.token_urlsafe(18)
        cursor.execute('''
            INSERT INTO users (username, password_hash, role, full_name, department)
            VALUES 
            (?, ?, 'ADMIN', 'System Administrator', 'Information Technology'),
            (?, ?, 'FACULTY', 'Dr. Faculty In-Charge', 'Computer Science & Engineering')
        ''', ('admin', generate_password_hash(admin_password), 'faculty', generate_password_hash(faculty_password)))
        print(f"Initial admin login: admin / {admin_password}")
        print(f"Initial faculty login: faculty / {faculty_password}")

    # Upgrade legacy plaintext passwords. Replace the old known prototype
    # defaults with random one-time credentials instead of preserving them.
    for user in cursor.execute("SELECT user_id, username, password_hash FROM users").fetchall():
        stored = user['password_hash'] or ''
        if stored.startswith(('scrypt:', 'pbkdf2:')):
            continue
        password = stored
        if user['username'] == 'admin' and stored == 'admin123':
            password = os.environ.get('ADMIN_INITIAL_PASSWORD') or secrets.token_urlsafe(18)
            print(f"Reset legacy admin password: admin / {password}")
        elif user['username'] == 'faculty' and stored == 'faculty123':
            password = os.environ.get('FACULTY_INITIAL_PASSWORD') or secrets.token_urlsafe(18)
            print(f"Reset legacy faculty password: faculty / {password}")
        cursor.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (generate_password_hash(password), user['user_id']))

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
