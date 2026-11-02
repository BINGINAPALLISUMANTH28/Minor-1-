-- PostgreSQL Schema Migration for AI Smart Classroom Monitoring System
-- Production Enterprise Database Definition

CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(100) DEFAULT 'Computer Science & Engineering',
    section VARCHAR(50) DEFAULT 'Sec-A',
    class_name VARCHAR(100) DEFAULT 'CSE III Year',
    consent_status VARCHAR(20) DEFAULT 'GRANTED',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS face_encodings (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR(50) REFERENCES students(student_id) ON DELETE CASCADE,
    encoding_json TEXT NOT NULL,
    quality_score NUMERIC(4, 2) DEFAULT 0.90,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    faculty_name VARCHAR(100) NOT NULL,
    subject VARCHAR(100) NOT NULL,
    slot_number VARCHAR(50) NOT NULL,
    department VARCHAR(100) NOT NULL,
    class_name VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    scheduled_start VARCHAR(20) NOT NULL,
    scheduled_end VARCHAR(20) NOT NULL,
    actual_start VARCHAR(20),
    actual_end VARCHAR(20),
    min_duration_req_mins INTEGER DEFAULT 15,
    status VARCHAR(20) DEFAULT 'ACTIVE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_roster (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES sessions(session_id) ON DELETE CASCADE,
    student_id VARCHAR(50) REFERENCES students(student_id) ON DELETE CASCADE,
    enrolled_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, student_id)
);

CREATE TABLE IF NOT EXISTS attendance (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES sessions(session_id) ON DELETE CASCADE,
    student_id VARCHAR(50) REFERENCES students(student_id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'Not yet verified',
    verified_duration_secs INTEGER DEFAULT 0,
    first_seen VARCHAR(20),
    last_seen VARCHAR(20),
    qualification_result VARCHAR(100) DEFAULT 'PENDING',
    review_reason TEXT,
    marked_by VARCHAR(50) DEFAULT 'SYSTEM_BIOMETRIC',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, student_id)
);

CREATE TABLE IF NOT EXISTS observations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES sessions(session_id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,
    student_id VARCHAR(50) REFERENCES students(student_id) ON DELETE SET NULL,
    confidence NUMERIC(5, 4) DEFAULT 0.0,
    observable_cue VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50),
    action VARCHAR(50) NOT NULL,
    user_name VARCHAR(100) NOT NULL,
    details TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_obs_session_student ON observations(session_id, student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance(session_id);
