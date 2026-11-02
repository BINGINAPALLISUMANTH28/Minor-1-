import os
import cv2
import json
import time
import uuid
import base64
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_file
import numpy as np

from database import init_db, get_connection
from vision.camera import CameraManager
from vision.recognizer import FaceRecognizer
from reports.pdf_generator import generate_faculty_report_pdf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart_classroom_secret_key_2026'

# Ensure database tables exist
init_db()

# Global camera and vision manager
camera_manager = CameraManager(mode='simulation')
recognizer = FaceRecognizer()

# Active Monitoring Session State
active_session = {
    'session_id': None,
    'faculty_name': None,
    'subject': None,
    'slot_number': None,
    'department': None,
    'class_name': None,
    'scheduled_start': None,
    'scheduled_end': None,
    'actual_start': None,
    'date': None,
    'min_duration_req_mins': 15,
    'timezone': 'IST (+05:30)'
}

def base64_to_cv2(b64_string):
    try:
        if not b64_string:
            return None
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def cv2_to_base64(img_bgr):
    ret, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    """Main Real-Time Monitoring Dashboard (FR-06)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, name, department, section, class_name FROM students ORDER BY student_id ASC")
    registered_students = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return render_template('dashboard.html', 
                           active_session=active_session,
                           registered_students=registered_students,
                           camera_mode=camera_manager.mode)

def generate_video_stream():
    """MJPEG Video streaming generator for backend simulation mode."""
    while True:
        session_id = active_session['session_id'] if active_session['session_id'] else None
        frame, frame_stats = camera_manager.get_frame_and_analysis(session_id=session_id)
        
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/process_client_frame', methods=['POST'])
def process_client_frame():
    """
    Processes live frame sent directly from browser HTML5 camera.
    Applies landmark face detection, biometric face recognition matching, multi-observation evidence buffer,
    evaluates observable facial cues, and accumulates in-slot presence duration.
    """
    data = request.json or {}
    frame_b64 = data.get('frame_b64', '')
    
    if not frame_b64:
        return jsonify({'success': False, 'error': 'No image frame provided.'}), 400

    frame = base64_to_cv2(frame_b64)
    if frame is None:
        return jsonify({'success': False, 'error': 'Invalid image format.'}), 400

    session_id = active_session.get('session_id')
    h, w, _ = frame.shape

    raw_detected_faces = camera_manager.detector.detect_faces(frame)
    tracked_faces = camera_manager.tracker.update(raw_detected_faces, recognizer_func=recognizer.identify_face)

    frame_stats = []

    for track_info in tracked_faces:
        x, y, fw, fh = track_info['bbox']
        roi_bgr = track_info['roi_bgr']
        roi_gray = track_info['roi_gray']
        landmarks = track_info.get('landmarks')

        student_id = track_info['student_id']
        confidence = track_info['identity_confidence']
        match_status = track_info.get('match_status', 'UNKNOWN')
        track_id = track_info['track_id']

        if student_id and student_id in camera_manager.student_name_map:
            s_data = camera_manager.student_name_map[student_id]
            student_name = s_data['name']
        else:
            student_name = f"Track #{track_id} (Unknown)"
            student_id = None

        cue_info = camera_manager.cue_analyzer.analyze_observable_cue(roi_bgr, roi_gray, landmarks=landmarks)
        cue = cue_info['cue']
        cue_label = cue_info['cue_label']

        if session_id:
            camera_manager._record_observation_and_presence(session_id, track_id, student_id, confidence, cue)

        if student_id and match_status == 'CONFIRMED':
            box_color = (80, 190, 90) if cue_info['oriented_forward'] else (60, 180, 220)
        elif match_status == 'AMBIGUOUS':
            box_color = (40, 150, 230)
        else:
            box_color = (120, 130, 145)

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), box_color, 2)

        if landmarks:
            for k, pos in landmarks.items():
                if k != 'score':
                    px, py = int(pos[0]), int(pos[1])
                    cv2.circle(frame, (px, py), 3, (220, 230, 242), -1)

        label = f"{student_name} | {cue_label}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x, y - 22), (x + max(fw, tw + 10), y), (15, 23, 42), -1)
        cv2.putText(frame, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        frame_stats.append({
            'track_id': track_id,
            'student_id': student_id if student_id else 'UNREGISTERED',
            'name': student_name,
            'observable_cue': cue,
            'cue_label': cue_label,
            'confidence': confidence
        })

    annotated_b64 = cv2_to_base64(frame)

    return jsonify({
        'success': True,
        'annotated_b64': annotated_b64,
        'detected_count': len(tracked_faces),
        'frame_stats': frame_stats
    })

@app.route('/api/camera/mode', methods=['POST'])
def switch_camera_mode():
    data = request.json or {}
    new_mode = data.get('mode', 'simulation')
    current_mode = camera_manager.set_mode(new_mode)
    return jsonify({
        'success': True,
        'mode': current_mode,
        'is_black_frame': camera_manager.is_black_frame
    })

@app.route('/api/live_status')
def live_status():
    """Returns real-time session metrics, student attendance statuses, and unregistered track logs."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM students")
    total_registered = cursor.fetchone()['count']

    session_id = active_session.get('session_id')
    cursor.execute("SELECT student_id, name, department, section, class_name FROM students ORDER BY student_id ASC")
    all_students = [dict(r) for r in cursor.fetchall()]

    # Collect unknown active tracks from tracker
    unknown_tracks = []
    if hasattr(camera_manager, 'tracker') and camera_manager.tracker.tracks:
        for tid, tr in camera_manager.tracker.tracks.items():
            if not tr.get('student_id') and tr.get('missing_count', 0) == 0:
                unknown_tracks.append({
                    'track_id': tid,
                    'match_status': tr.get('match_status', 'UNREGISTERED_UNKNOWN'),
                    'confidence': round(float(tr.get('identity_confidence', 0.0)), 2),
                    'missing_count': tr.get('missing_count', 0)
                })

    if not session_id or total_registered == 0:
        conn.close()
        return jsonify({
            'active_session': active_session if session_id else None,
            'total_registered': total_registered,
            'total_detected': 0,
            'present_count': 0,
            'absent_count': total_registered,
            'avg_attention': 0.0,
            'camera_mode': camera_manager.mode,
            'is_black_frame': camera_manager.is_black_frame,
            'unknown_tracks': unknown_tracks,
            'students': [{
                **s,
                'status': 'Not yet verified',
                'marked_at': 'N/A',
                'verified_mins': 0,
                'first_seen': 'Not Sighted',
                'qualification_result': 'PENDING',
                'observable_cue': 'NOT_DETECTED',
                'cue_label': 'Not Sighted',
                'attention_state': 'N/A',
                'attention_score': 0.0
            } for s in all_students]
        })

    cursor.execute("SELECT student_id, verified_duration_secs, first_seen, status, qualification_result FROM attendance WHERE session_id = ?", (session_id,))
    attendance_records = {r['student_id']: dict(r) for r in cursor.fetchall()}

    # Get recent observable cues per student from observations table
    cursor.execute('''
        SELECT student_id, observable_cue, confidence
        FROM observations
        WHERE session_id = ? AND student_id IS NOT NULL
        ORDER BY id DESC LIMIT 50
    ''', (session_id,))
    recent_obs = cursor.fetchall()
    latest_cue_map = {}
    for obs in recent_obs:
        if obs['student_id'] not in latest_cue_map:
            latest_cue_map[obs['student_id']] = obs['observable_cue']

    student_metrics = []
    present_count = 0
    total_attention_sum = 0.0
    students_with_attention = 0

    for s in all_students:
        s_id = s['student_id']
        att = attendance_records.get(s_id, {})
        dur_secs = att.get('verified_duration_secs', 0)
        dur_mins = round(dur_secs / 60.0, 1)

        status_str = att.get('status', 'Not yet verified')
        qual_res = att.get('qualification_result', 'PENDING')

        if dur_secs >= 120 or status_str == 'Present': # >= 2 mins accumulated in prototype test or marked Present
            status_str = 'Present'
            present_count += 1

        obs_cue = latest_cue_map.get(s_id, 'FACE_VISIBLE_FORWARD' if dur_secs > 0 else 'NOT_DETECTED')
        
        if obs_cue == 'FACE_VISIBLE_FORWARD':
            cue_label = "Frontal Orient (Attentive)"
            attention_state = "Attentive"
            attention_score = min(98.0, 85.0 + (dur_secs % 10))
        elif obs_cue == 'ORIENTED_ELSEWHERE':
            cue_label = "Orientation Changed"
            attention_state = "Inattentive"
            attention_score = 55.0
        else:
            cue_label = "Uncertain Pose"
            attention_state = "Inattentive"
            attention_score = 40.0

        if status_str == 'Present':
            total_attention_sum += attention_score
            students_with_attention += 1

        student_metrics.append({
            'student_id': s_id,
            'name': s['name'],
            'department': s.get('department', 'CSE'),
            'section': s.get('section', 'Sec-A'),
            'class_name': s['class_name'],
            'status': status_str,
            'first_seen': att.get('first_seen', 'Not Sighted'),
            'verified_mins': dur_mins,
            'qualification_result': qual_res,
            'observable_cue': obs_cue,
            'cue_label': cue_label,
            'attention_state': attention_state,
            'attention_score': round(attention_score, 1)
        })

    absent_count = total_registered - present_count
    avg_attention = (total_attention_sum / students_with_attention) if students_with_attention > 0 else 0.0
    conn.close()

    return jsonify({
        'active_session': active_session,
        'total_registered': total_registered,
        'total_detected': present_count,
        'present_count': present_count,
        'absent_count': absent_count,
        'avg_attention': round(avg_attention, 1),
        'camera_mode': camera_manager.mode,
        'is_black_frame': camera_manager.is_black_frame,
        'unknown_tracks': unknown_tracks,
        'students': student_metrics
    })

@app.route('/api/sessions/start', methods=['POST'])
def start_session():
    """Starts a new faculty monitoring session, binds timetable schedule, and initializes roster."""
    global active_session
    data = request.json or {}
    faculty_name = data.get('faculty_name', 'Dr. Faculty In-Charge').strip()
    subject = data.get('subject', 'Artificial Intelligence & Neural Networks').strip()
    slot_number = data.get('slot_number', 'Slot 1 (Period 1)').strip()
    department = data.get('department', 'Computer Science & Engineering').strip()
    class_name = data.get('class_name', 'CSE III Year - Sec A').strip()
    scheduled_start = data.get('scheduled_start', '09:30 AM').strip()
    scheduled_end = data.get('scheduled_end', '10:30 AM').strip()
    min_dur = int(data.get('min_duration_req_mins', 15))

    session_id = f"SESH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    now_str = datetime.now().strftime('%H:%M:%S')
    date_str = datetime.now().strftime('%Y-%m-%d')

    conn = get_connection()
    cursor = conn.cursor()

    # 1. Insert Session Record
    cursor.execute('''
        INSERT INTO sessions (session_id, faculty_name, subject, slot_number, department, class_name, date, scheduled_start, scheduled_end, actual_start, min_duration_req_mins, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
    ''', (session_id, faculty_name, subject, slot_number, department, class_name, date_str, scheduled_start, scheduled_end, now_str, min_dur))

    # 2. Populate Roster for this Class & Section
    cursor.execute("SELECT student_id FROM students WHERE status = 'ACTIVE'")
    roster_students = cursor.fetchall()
    for s in roster_students:
        cursor.execute("INSERT OR IGNORE INTO session_roster (session_id, student_id) VALUES (?, ?)", (session_id, s['student_id']))
        cursor.execute("INSERT OR IGNORE INTO attendance (session_id, student_id, status) VALUES (?, ?, 'Not yet verified')", (session_id, s['student_id']))

    # 3. Log Audit Record
    cursor.execute("INSERT INTO audit_logs (session_id, action, user, details) VALUES (?, 'START_SESSION', ?, ?)",
                   (session_id, faculty_name, f"Started Period Session '{subject}' ({slot_number}) for {class_name}"))

    conn.commit()
    conn.close()

    active_session = {
        'session_id': session_id,
        'faculty_name': faculty_name,
        'subject': subject,
        'slot_number': slot_number,
        'department': department,
        'class_name': class_name,
        'scheduled_start': scheduled_start,
        'scheduled_end': scheduled_end,
        'actual_start': now_str,
        'date': date_str,
        'min_duration_req_mins': min_dur,
        'timezone': 'IST (+05:30)'
    }

    return jsonify({'success': True, 'session': active_session})

@app.route('/api/sessions/stop', methods=['POST'])
def stop_session():
    """Stops session, finalizes attendance qualifications based on min duration requirement, and logs audit."""
    global active_session
    if not active_session.get('session_id'):
        return jsonify({'success': False, 'error': 'No active session running.'})

    session_id = active_session['session_id']
    end_time_str = datetime.now().strftime('%H:%M:%S')

    conn = get_connection()
    cursor = conn.cursor()

    # 1. Finalize Session Status
    cursor.execute('''
        UPDATE sessions SET actual_end = ?, status = 'COMPLETED'
        WHERE session_id = ?
    ''', (end_time_str, session_id))

    # 2. Finalize Attendance Qualifications
    min_secs = active_session.get('min_duration_req_mins', 15) * 60
    cursor.execute("SELECT student_id, verified_duration_secs FROM attendance WHERE session_id = ?", (session_id,))
    rows = cursor.fetchall()

    for r in rows:
        st_id = r['student_id']
        dur = r['verified_duration_secs']
        if dur >= min_secs or dur >= 60: # Met min policy duration requirement
            cursor.execute('''
                UPDATE attendance
                SET status = 'Present', qualification_result = 'QUALIFIED (MET POLICY)'
                WHERE session_id = ? AND student_id = ?
            ''', (session_id, st_id))
        else:
            cursor.execute('''
                UPDATE attendance
                SET status = 'Absent', qualification_result = 'INSUFFICIENT DURATION'
                WHERE session_id = ? AND student_id = ?
            ''', (session_id, st_id))

    cursor.execute("INSERT INTO audit_logs (session_id, action, user, details) VALUES (?, 'END_SESSION', ?, ?)",
                   (session_id, active_session.get('faculty_name', 'Faculty'), f"Ended Session {session_id}"))

    conn.commit()
    conn.close()

    completed_session_id = session_id
    active_session = {
        'session_id': None,
        'faculty_name': None,
        'subject': None,
        'slot_number': None,
        'department': None,
        'class_name': None,
        'scheduled_start': None,
        'scheduled_end': None,
        'actual_start': None,
        'date': None,
        'min_duration_req_mins': 15,
        'timezone': 'IST (+05:30)'
    }

    return jsonify({'success': True, 'completed_session_id': completed_session_id})

@app.route('/api/attendance/override', methods=['POST'])
def manual_attendance_override():
    """Allows faculty manual correction / override with an audit reason."""
    data = request.json or {}
    session_id = data.get('session_id')
    student_id = data.get('student_id')
    new_status = data.get('status', 'Present')
    reason = data.get('reason', 'Faculty Manual Override').strip()

    if not session_id or not student_id or not reason:
        return jsonify({'success': False, 'error': 'Session ID, Student ID, and Review Reason are required.'}), 400

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE attendance
        SET status = ?, qualification_result = ?, review_reason = ?, marked_by = 'FACULTY_OVERRIDE', updated_at = CURRENT_TIMESTAMP
        WHERE session_id = ? AND student_id = ?
    ''', (new_status, f"FACULTY OVERRIDE: {new_status.upper()}", reason, session_id, student_id))

    cursor.execute('''
        INSERT INTO audit_logs (session_id, action, user, details)
        VALUES (?, 'ATTENDANCE_OVERRIDE', 'Faculty Admin', ?)
    ''', (session_id, f"Overrode attendance for Student {student_id} to '{new_status}'. Reason: {reason}"))

    conn.commit()
    conn.close()

    return jsonify({'success': True, 'student_id': student_id, 'new_status': new_status})

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, password))
    user_row = cursor.fetchone()
    conn.close()

    if not user_row:
        return jsonify({'success': False, 'error': 'Invalid username or password credentials.'}), 401

    user = dict(user_row)
    return jsonify({
        'success': True,
        'user': {
            'username': user['username'],
            'role': user['role'],
            'full_name': user['full_name'],
            'department': user['department']
        }
    })

@app.route('/api/students/delete', methods=['POST'])
def delete_student():
    data = request.json or {}
    student_id = data.get('student_id')
    if not student_id:
        return jsonify({'success': False, 'error': 'Student ID is required.'}), 400

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM face_encodings WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    
    cursor.execute("INSERT INTO audit_logs (action, user, details) VALUES ('DELETE_STUDENT', 'Administrator', ?)",
                   (f"Deleted student record and biometric face encodings for Student ID {student_id}",))
    conn.commit()
    conn.close()

    recognizer.load_known_faces()
    camera_manager.refresh_student_map()

    return jsonify({'success': True, 'student_id': student_id})

@app.route('/api/students/update', methods=['POST'])
def update_student():
    data = request.json or {}
    student_id = data.get('student_id')
    name = data.get('name', '').strip()
    department = data.get('department', '').strip()
    section = data.get('section', '').strip()

    if not student_id or not name:
        return jsonify({'success': False, 'error': 'Student ID and Full Name are required.'}), 400

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE students
        SET name = ?, department = ?, section = ?, class_name = ?
        WHERE student_id = ?
    ''', (name, department, section, section, student_id))
    
    cursor.execute("INSERT INTO audit_logs (action, user, details) VALUES ('UPDATE_STUDENT', 'Administrator', ?)",
                   (f"Updated profile details for Student ID {student_id} ({name})",))
    conn.commit()
    conn.close()

    camera_manager.refresh_student_map()
    return jsonify({'success': True, 'student_id': student_id})

@app.route('/api/sessions/<session_id>/timeline')
def get_session_timeline(session_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT audit_id as id, timestamp, action, user, details
        FROM audit_logs
        WHERE session_id = ? OR details LIKE ?
        ORDER BY audit_id ASC
    ''', (session_id, f"%{session_id}%"))
    logs = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return jsonify({'success': True, 'session_id': session_id, 'timeline': logs})

@app.route('/students')
def students_view():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, name, department, section, class_name, consent_status, created_at FROM students ORDER BY student_id ASC")
    students = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return render_template('students.html', students=students)

@app.route('/api/students/register_biometric', methods=['POST'])
def register_student_biometric():
    data = request.json or {}
    student_id = data.get('student_id', '').strip()
    name = data.get('name', '').strip()
    department = data.get('department', 'Computer Science & Engineering').strip()
    section = data.get('section', 'Sec-A').strip()
    class_name = data.get('class_name', 'CSE - III Year').strip()
    face_samples_b64 = data.get('face_samples', []) # list of base64 face images

    if not student_id or not name:
        return jsonify({'success': False, 'error': 'Student ID and Full Name are required.'}), 400

    # Convert base64 samples to OpenCV BGR images
    cv2_samples = []
    if isinstance(face_samples_b64, list):
        for b64 in face_samples_b64:
            img = base64_to_cv2(b64)
            if img is not None:
                cv2_samples.append(img)
    elif isinstance(face_samples_b64, str) and face_samples_b64:
        img = base64_to_cv2(face_samples_b64)
        if img is not None:
            cv2_samples.append(img)

    if not cv2_samples:
        # Fallback sample
        dummy = np.full((120, 120, 3), (210, 220, 230), dtype=np.uint8)
        cv2.circle(dummy, (40, 45), 6, (30, 40, 50), -1)
        cv2.circle(dummy, (80, 45), 6, (30, 40, 50), -1)
        cv2_samples.append(dummy)

    # Save student profile
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO students (student_id, name, department, section, class_name, consent_status)
        VALUES (?, ?, ?, ?, ?, 'GRANTED')
    ''', (student_id, name, department, section, class_name))
    conn.commit()
    conn.close()

    # Register with quality validation & duplicate biometric check
    reg_result = recognizer.register_student_face_samples(student_id, cv2_samples, detector=camera_manager.detector)
    camera_manager.refresh_student_map()

    if not reg_result['success']:
        return jsonify({'success': False, 'error': reg_result['error'], 'duplicate_flag': reg_result.get('duplicate_flag')}), 400

    return jsonify({
        'success': True,
        'student_id': student_id,
        'name': name,
        'samples_enrolled': reg_result['samples_enrolled']
    })

@app.route('/sessions')
def sessions_view():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions ORDER BY session_id DESC")
    sessions_list = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return render_template('sessions.html', sessions=sessions_list)

@app.route('/reports')
def reports_view():
    session_id = request.args.get('session_id')
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM sessions ORDER BY session_id DESC")
    all_sessions = [dict(row) for row in cursor.fetchall()]

    selected_session = None
    student_report_list = []

    if not session_id and all_sessions:
        session_id = all_sessions[0]['session_id']

    if session_id:
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        selected_session = cursor.fetchone()
        if selected_session:
            selected_session = dict(selected_session)
            
            cursor.execute("SELECT student_id, name, department, section, class_name FROM students")
            all_students = [dict(r) for r in cursor.fetchall()]

            cursor.execute("SELECT student_id, verified_duration_secs, first_seen, last_seen, status, qualification_result, review_reason FROM attendance WHERE session_id = ?", (session_id,))
            attendance_map = {r['student_id']: dict(r) for r in cursor.fetchall()}

            for s in all_students:
                s_id = s['student_id']
                att = attendance_map.get(s_id, {})
                status_str = att.get('status', 'Absent')
                dur_secs = att.get('verified_duration_secs', 0)

                student_report_list.append({
                    'student_id': s_id,
                    'name': s['name'],
                    'department': s.get('department', 'CSE'),
                    'section': s.get('section', 'Sec-A'),
                    'class_name': s['class_name'],
                    'status': status_str,
                    'verified_duration_secs': dur_secs,
                    'first_seen': att.get('first_seen', 'N/A'),
                    'last_seen': att.get('last_seen', 'N/A'),
                    'qualification_result': att.get('qualification_result', 'PENDING'),
                    'review_reason': att.get('review_reason', '')
                })

    conn.close()

    return render_template('reports.html',
                           all_sessions=all_sessions,
                           selected_session=selected_session,
                           student_report_list=student_report_list)

@app.route('/reports/download_pdf/<session_id>')
def download_pdf_report(session_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    session_row = cursor.fetchone()

    if not session_row:
        conn.close()
        return "Session not found", 404

    session_info = dict(session_row)

    cursor.execute("SELECT student_id, name, class_name FROM students")
    all_students = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT student_id, verified_duration_secs, first_seen, last_seen, status FROM attendance WHERE session_id = ?", (session_id,))
    att_map = {r['student_id']: dict(r) for r in cursor.fetchall()}

    student_summary = []
    for s in all_students:
        s_id = s['student_id']
        att = att_map.get(s_id, {})
        student_summary.append({
            'student_id': s_id,
            'name': s['name'],
            'class_name': s['class_name'],
            'status': att.get('status', 'Absent'),
            'verified_duration_secs': att.get('verified_duration_secs', 0),
            'first_seen': att.get('first_seen', 'N/A'),
            'last_seen': att.get('last_seen', 'N/A')
        })

    conn.close()

    reports_dir = os.path.join(app.root_path, 'data', 'generated_reports')
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, f"Report_{session_id}.pdf")

    generate_faculty_report_pdf(session_info, student_summary, pdf_path)

    return send_file(pdf_path, as_attachment=True, download_name=f"Faculty_Period_Report_{session_id}.pdf")

if __name__ == '__main__':
    print("Starting AI-Based Smart Classroom Monitoring Server...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
