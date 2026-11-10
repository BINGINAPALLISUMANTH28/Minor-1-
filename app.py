import os
import cv2
import json
import time
import uuid
import base64
import secrets
from functools import wraps
from urllib.parse import urlparse
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_file, session, redirect, url_for
import numpy as np
from werkzeug.security import check_password_hash

from database import init_db, get_connection
from vision.camera import CameraManager
from vision.recognizer import FaceRecognizer
from reports.pdf_generator import generate_faculty_report_pdf

app = Flask(__name__)

# Ensure database tables exist
init_db()

def load_secret_key():
    configured = os.environ.get('FLASK_SECRET_KEY')
    if configured:
        return configured
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM system_config WHERE key = 'flask_secret_key'")
    row = cursor.fetchone()
    if row:
        conn.close()
        return row['value']
    generated = secrets.token_hex(32)
    cursor.execute("INSERT INTO system_config (key, value) VALUES ('flask_secret_key', ?)", (generated,))
    conn.commit()
    conn.close()
    return generated

app.config['SECRET_KEY'] = load_secret_key()
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

def refresh_biometric_enrollment_flags():
    """Only report enrollment when current SFace templates are available."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE students
        SET is_biometric_enrolled = CASE
            WHEN EXISTS (
                SELECT 1 FROM face_encodings f
                WHERE f.student_id = students.student_id AND f.model_version = ?
            ) THEN 1
            ELSE 0
        END
    ''', (FaceRecognizer.MODEL_VERSION,))
    conn.commit()
    conn.close()

# Load database identity data before starting either camera pipeline.
recognizer = FaceRecognizer()
recognizer.load_known_faces()
refresh_biometric_enrollment_flags()
camera_manager = CameraManager(mode='simulation', recognizer=recognizer)

def login_required(*roles):
    def decorate(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = session.get('user')
            if not user:
                if request.path.startswith('/api/') or request.path == '/video_feed':
                    return jsonify({'success': False, 'error': 'Authentication required.'}), 401
                return redirect(url_for('login_page', next=request.path))
            if roles and user.get('role') not in roles:
                if request.path.startswith('/api/'):
                    return jsonify({'success': False, 'error': 'You do not have permission for this action.'}), 403
                return 'Forbidden', 403
            return view(*args, **kwargs)
        return wrapped
    return decorate

@app.context_processor
def inject_current_user():
    return {'current_user': session.get('user')}

@app.route('/login')
def login_page():
    next_url = request.args.get('next', '/')
    parsed = urlparse(next_url)
    if parsed.netloc or not next_url.startswith('/'):
        next_url = '/'
    return render_template('login.html', next_url=next_url)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# Active Monitoring Session State
active_session = {
    'session_id': None,
    'faculty_name': None,
    'subject': None,
    'slot_number': None,
    'department': None,
    'class_name': None,
    'section': None,
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
@login_required('ADMIN', 'FACULTY')
def index():
    """Main Real-Time Monitoring Dashboard (FR-06)."""
    conn = get_connection()
    cursor = conn.cursor()
    if active_session.get('session_id'):
        cursor.execute('''
            SELECT s.student_id, s.name, s.department, s.section, s.class_name
            FROM session_roster r JOIN students s ON s.student_id = r.student_id
            WHERE r.session_id = ? AND s.status = 'ACTIVE'
            ORDER BY s.student_id ASC
        ''', (active_session['session_id'],))
    else:
        cursor.execute("SELECT student_id, name, department, section, class_name FROM students WHERE status = 'ACTIVE' ORDER BY student_id ASC")
    registered_students = [dict(row) for row in cursor.fetchall()]
    cursor.execute("SELECT DISTINCT class_name FROM students WHERE status = 'ACTIVE' ORDER BY class_name")
    class_options = [row['class_name'] for row in cursor.fetchall()]
    cursor.execute("SELECT DISTINCT section FROM students WHERE status = 'ACTIVE' ORDER BY section")
    section_options = [row['section'] for row in cursor.fetchall()]
    conn.close()

    return render_template('dashboard.html', 
                           active_session=active_session,
                           registered_students=registered_students,
                           class_options=class_options,
                           section_options=section_options,
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
@login_required('ADMIN', 'FACULTY')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/process_client_frame', methods=['POST'])
@login_required('ADMIN', 'FACULTY')
def process_client_frame():
    """
    Processes live frame sent directly from browser HTML5 camera.
    Applies landmark face detection, biometric face recognition matching, multi-observation evidence buffer,
    evaluates observable facial cues, and accumulates in-slot presence duration.
    """
    data = request.json or {}
    if camera_manager.mode != 'browser':
        return jsonify({'success': False, 'error': 'Select Browser Live Camera before sending live frames.'}), 409
    frame_b64 = data.get('frame_b64', '')
    
    if not frame_b64:
        return jsonify({'success': False, 'error': 'No image frame provided.'}), 400

    frame = base64_to_cv2(frame_b64)
    if frame is None:
        return jsonify({'success': False, 'error': 'Invalid image format.'}), 400

    session_id = active_session.get('session_id')
    h, w, _ = frame.shape

    raw_detected_faces = camera_manager.detector.detect_faces(frame)
    tracked_faces = camera_manager.client_tracker.update(raw_detected_faces, recognizer_func=recognizer.identify_face)

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

        if session_id and student_id and match_status == 'CONFIRMED':
            camera_manager._record_observation_and_presence(session_id, track_id, student_id, confidence, cue)

        frame_stats.append({
            'track_id': track_id,
            'student_id': student_id if student_id else 'UNREGISTERED',
            'name': student_name,
            'bbox': [int(x), int(y), int(fw), int(fh)],
            'match_status': match_status,
            'observable_cue': cue,
            'cue_label': cue_label,
            'confidence': confidence
        })

    return jsonify({
        'success': True,
        'frame_width': int(w),
        'frame_height': int(h),
        'detected_count': len(tracked_faces),
        'frame_stats': frame_stats
    })

@app.route('/api/camera/mode', methods=['POST'])
@login_required('ADMIN', 'FACULTY')
def switch_camera_mode():
    data = request.json or {}
    new_mode = data.get('mode', 'simulation')
    if new_mode not in ('simulation', 'browser', 'auto', 'webcam'):
        return jsonify({'success': False, 'error': 'Unsupported camera mode.'}), 400
    if active_session.get('session_id') and new_mode == 'simulation':
        return jsonify({'success': False, 'error': 'Cannot switch to the prototype stream during an active attendance session.'}), 409
    current_mode = camera_manager.set_mode(new_mode)
    return jsonify({
        'success': True,
        'mode': current_mode,
        'is_black_frame': camera_manager.is_black_frame
    })

@app.route('/api/live_status')
@login_required('ADMIN', 'FACULTY')
def live_status():
    """Returns real-time session metrics, student attendance statuses, and unregistered track logs."""
    conn = get_connection()
    cursor = conn.cursor()

    session_id = active_session.get('session_id')
    if session_id:
        cursor.execute('''
            SELECT s.student_id, s.name, s.department, s.section, s.class_name
            FROM session_roster r JOIN students s ON s.student_id = r.student_id
            WHERE r.session_id = ? AND s.status = 'ACTIVE' ORDER BY s.student_id ASC
        ''', (session_id,))
    else:
        cursor.execute("SELECT student_id, name, department, section, class_name FROM students WHERE status = 'ACTIVE' ORDER BY student_id ASC")
    all_students = [dict(r) for r in cursor.fetchall()]
    total_registered = len(all_students)

    # Collect unknown active tracks from tracker
    unknown_tracks = []
    active_tracker = camera_manager.client_tracker if camera_manager.mode == 'browser' else camera_manager.tracker
    if active_tracker.tracks:
        for tid, tr in active_tracker.tracks.items():
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

    cursor.execute("SELECT student_id, verified_duration_secs, first_seen, status, qualification_result, last_observation_epoch FROM attendance WHERE session_id = ?", (session_id,))
    attendance_records = {r['student_id']: dict(r) for r in cursor.fetchall()}

    # Get recent observable cues per student from observations table
    cursor.execute('''
        SELECT student_id, observable_cue, confidence
        FROM observations
        WHERE session_id = ? AND student_id IS NOT NULL
        ORDER BY obs_id DESC LIMIT 50
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
    now_epoch = time.time()

    for s in all_students:
        s_id = s['student_id']
        att = attendance_records.get(s_id, {})
        dur_secs = att.get('verified_duration_secs') or 0
        dur_mins = round(dur_secs / 60.0, 1)

        status_str = att.get('status', 'Not yet verified')
        qual_res = att.get('qualification_result', 'PENDING')

        required_secs = int(active_session.get('min_duration_req_mins', 15)) * 60
        if dur_secs >= required_secs:
            status_str = 'Present'
            qual_res = 'QUALIFIED (MET POLICY)'
        elif active_session.get('session_id'):
            status_str = 'Presence Accumulating' if dur_secs > 0 else 'Waiting for Match'
        if status_str == 'Present':
            present_count += 1

        last_observation_epoch = att.get('last_observation_epoch')
        attention_available = bool(
            last_observation_epoch and 0 <= now_epoch - last_observation_epoch <= 5.0
        )
        obs_cue = latest_cue_map.get(s_id, 'NOT_DETECTED') if attention_available else 'NOT_DETECTED'
        
        if obs_cue == 'FACE_VISIBLE_FORWARD':
            cue_label = "Frontal Orient (Attentive)"
            attention_state = "Attentive"
            attention_score = 85.0
        elif obs_cue in ('ORIENTED_ELSEWHERE', 'FACE_VISIBLE_ELSEWHERE'):
            cue_label = "Orientation Changed"
            attention_state = "Inattentive"
            attention_score = 55.0
        elif obs_cue == 'UNCERTAIN':
            cue_label = "Uncertain Pose"
            attention_state = "Inattentive"
            attention_score = 40.0
        else:
            cue_label = "Not Sighted"
            attention_state = "N/A"
            attention_score = 0.0

        if attention_available:
            total_attention_sum += attention_score
            students_with_attention += 1

        student_metrics.append({
            'student_id': s_id,
            'name': s['name'],
            'department': s.get('department', 'CSE'),
            'section': s.get('section', 'Sec-A'),
            'class_name': s['class_name'],
            'status': status_str,
            'first_seen': att.get('first_seen') or 'Not Sighted',
            'verified_mins': dur_mins,
            'qualification_result': qual_res,
            'observable_cue': obs_cue,
            'cue_label': cue_label,
            'attention_state': attention_state,
            'attention_score': round(attention_score, 1),
            'attention_available': attention_available
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
@login_required('ADMIN', 'FACULTY')
def start_session():
    """Starts a new faculty monitoring session, binds timetable schedule, and initializes roster."""
    global active_session
    data = request.json or {}
    faculty_name = str(data.get('faculty_name') or 'Dr. Faculty In-Charge').strip()
    subject = str(data.get('subject') or 'Artificial Intelligence & Neural Networks').strip()
    slot_number = str(data.get('slot_number') or 'Slot 1 (Period 1)').strip()
    department = str(data.get('department') or 'Computer Science & Engineering').strip()
    class_name = str(data.get('class_name') or '').strip()
    section = str(data.get('section') or '').strip()
    scheduled_start = str(data.get('scheduled_start') or '09:30 AM').strip()
    scheduled_end = str(data.get('scheduled_end') or '10:30 AM').strip()
    try:
        min_dur = max(1, min(1440, int(data.get('min_duration_req_mins', 15))))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Minimum attendance duration must be a valid number of minutes.'}), 400

    if active_session.get('session_id'):
        return jsonify({'success': False, 'error': 'A monitoring session is already active.'}), 409
    if camera_manager.mode == 'simulation':
        return jsonify({'success': False, 'error': 'The prototype camera stream cannot record attendance. Select Browser Live Camera before starting the session.'}), 409

    session_id = f"SESH-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    now_str = datetime.now().strftime('%H:%M:%S')
    date_str = datetime.now().strftime('%Y-%m-%d')

    conn = get_connection()
    cursor = conn.cursor()

    # Select active students in the exact class before creating any session data.
    cursor.execute("SELECT student_id FROM students WHERE status = 'ACTIVE' AND class_name = ? AND section = ?", (class_name, section))
    roster_students = cursor.fetchall()
    if not roster_students:
        conn.close()
        return jsonify({'success': False, 'error': 'No active students are enrolled in the selected class.'}), 400

    cursor.execute('''
        INSERT INTO sessions (session_id, faculty_name, subject, slot_number, department, class_name, section, date, scheduled_start, scheduled_end, actual_start, min_duration_req_mins, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
    ''', (session_id, faculty_name, subject, slot_number, department, class_name, section, date_str, scheduled_start, scheduled_end, now_str, min_dur))

    # 2. Create roster and attendance rows.
    for s in roster_students:
        cursor.execute("INSERT OR IGNORE INTO session_roster (session_id, student_id) VALUES (?, ?)", (session_id, s['student_id']))
        cursor.execute("INSERT OR IGNORE INTO attendance (session_id, student_id, status) VALUES (?, ?, 'Not yet verified')", (session_id, s['student_id']))

    # 3. Log Audit Record
    user_name = session.get('user', {}).get('full_name', faculty_name)
    cursor.execute("INSERT INTO audit_logs (session_id, action, user, details) VALUES (?, 'START_SESSION', ?, ?)",
                   (session_id, user_name, f"Started Period Session '{subject}' ({slot_number}) for {class_name}"))

    conn.commit()
    conn.close()

    camera_manager.reset_trackers()

    active_session = {
        'session_id': session_id,
        'faculty_name': faculty_name,
        'subject': subject,
        'slot_number': slot_number,
        'department': department,
        'class_name': class_name,
        'section': section,
        'scheduled_start': scheduled_start,
        'scheduled_end': scheduled_end,
        'actual_start': now_str,
        'date': date_str,
        'min_duration_req_mins': min_dur,
        'timezone': 'IST (+05:30)'
    }

    return jsonify({'success': True, 'session': active_session})

@app.route('/api/sessions/stop', methods=['POST'])
@login_required('ADMIN', 'FACULTY')
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
    cursor.execute("SELECT student_id, verified_duration_secs, marked_by FROM attendance WHERE session_id = ?", (session_id,))
    rows = cursor.fetchall()

    for r in rows:
        st_id = r['student_id']
        dur = r['verified_duration_secs'] or 0
        if r['marked_by'] == 'FACULTY_OVERRIDE':
            continue
        if dur >= min_secs:
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

    user_name = session.get('user', {}).get('full_name', active_session.get('faculty_name', 'Faculty'))
    cursor.execute("INSERT INTO audit_logs (session_id, action, user, details) VALUES (?, 'END_SESSION', ?, ?)",
                   (session_id, user_name, f"Ended Session {session_id}"))

    conn.commit()
    conn.close()

    camera_manager.reset_trackers()

    completed_session_id = session_id
    active_session = {
        'session_id': None,
        'faculty_name': None,
        'subject': None,
        'slot_number': None,
        'department': None,
        'class_name': None,
        'section': None,
        'scheduled_start': None,
        'scheduled_end': None,
        'actual_start': None,
        'date': None,
        'min_duration_req_mins': 15,
        'timezone': 'IST (+05:30)'
    }

    return jsonify({'success': True, 'completed_session_id': completed_session_id})

@app.route('/api/attendance/override', methods=['POST'])
@login_required('ADMIN', 'FACULTY')
def manual_attendance_override():
    """Allows faculty manual correction / override with an audit reason."""
    data = request.json or {}
    session_id = data.get('session_id')
    student_id = data.get('student_id')
    new_status = data.get('status', 'Present')
    reason = data.get('reason', 'Faculty Manual Override').strip()

    if not session_id or not student_id or not reason or new_status not in ('Present', 'Absent'):
        return jsonify({'success': False, 'error': 'Session ID, Student ID, and Review Reason are required.'}), 400

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE attendance
        SET status = ?, qualification_result = ?, review_reason = ?, marked_by = 'FACULTY_OVERRIDE', updated_at = CURRENT_TIMESTAMP
        WHERE session_id = ? AND student_id = ?
    ''', (new_status, f"FACULTY OVERRIDE: {new_status.upper()}", reason, session_id, student_id))

    if cursor.rowcount == 0:
        conn.close()
        return jsonify({'success': False, 'error': 'No attendance record exists for that student and session.'}), 404

    cursor.execute('''
        INSERT INTO audit_logs (session_id, action, user, details)
        VALUES (?, 'ATTENDANCE_OVERRIDE', ?, ?)
    ''', (session_id, session.get('user', {}).get('full_name', 'Faculty'), f"Overrode attendance for Student {student_id} to '{new_status}'. Reason: {reason}"))

    conn.commit()
    conn.close()

    return jsonify({'success': True, 'student_id': student_id, 'new_status': new_status})

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.json or {}
    username = str(data.get('username') or '').strip()
    password = str(data.get('password') or '')

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user_row = cursor.fetchone()
    conn.close()

    if not user_row or not check_password_hash(user_row['password_hash'], password):
        return jsonify({'success': False, 'error': 'Invalid username or password credentials.'}), 401

    user = dict(user_row)
    session.clear()
    session['user'] = {
        'username': user['username'],
        'role': user['role'],
        'full_name': user['full_name'],
        'department': user['department']
    }
    return jsonify({
        'success': True,
        'user': session['user']
    })

@app.route('/api/students/delete', methods=['POST'])
@login_required('ADMIN')
def delete_student():
    data = request.json or {}
    student_id = data.get('student_id')
    if not student_id:
        return jsonify({'success': False, 'error': 'Student ID is required.'}), 400

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM face_encodings WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM student_face_photos WHERE student_id = ?", (student_id,))
    cursor.execute("UPDATE students SET status = 'INACTIVE', is_biometric_enrolled = 0 WHERE student_id = ?", (student_id,))
    
    cursor.execute("INSERT INTO audit_logs (action, user, details) VALUES ('DELETE_STUDENT', ?, ?)",
                   (session.get('user', {}).get('full_name', 'Administrator'), f"Archived Student ID {student_id} and removed its biometric face encodings"))
    conn.commit()
    conn.close()

    recognizer.load_known_faces()
    camera_manager.recognizer.load_known_faces()
    camera_manager.refresh_student_map()
    camera_manager.reset_trackers()

    return jsonify({'success': True, 'student_id': student_id})

@app.route('/api/students/update', methods=['POST'])
@login_required('ADMIN')
def update_student():
    data = request.json or {}
    student_id = data.get('student_id')
    name = data.get('name', '').strip()
    department = data.get('department', '').strip()
    section = data.get('section', '').strip()
    class_name = data.get('class_name', '').strip()

    if not student_id or not name or not department or not section or not class_name:
        return jsonify({'success': False, 'error': 'Student ID, name, department, class, and section are required.'}), 400

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE students
        SET name = ?, department = ?, section = ?, class_name = ?
        WHERE student_id = ?
    ''', (name, department, section, class_name, student_id))
    
    cursor.execute("INSERT INTO audit_logs (action, user, details) VALUES ('UPDATE_STUDENT', ?, ?)",
                   (session.get('user', {}).get('full_name', 'Administrator'), f"Updated profile details for Student ID {student_id} ({name})"))
    conn.commit()
    conn.close()

    camera_manager.refresh_student_map()
    return jsonify({'success': True, 'student_id': student_id})

@app.route('/api/sessions/<session_id>/timeline')
@login_required('ADMIN', 'FACULTY')
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
@login_required('ADMIN')
def students_view():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.student_id, s.name, s.department, s.section, s.class_name,
               s.consent_status, s.created_at,
               (SELECT COUNT(*) FROM face_encodings f
                WHERE f.student_id = s.student_id AND f.model_version = ?) AS template_count
        FROM students s WHERE s.status = 'ACTIVE' ORDER BY s.student_id ASC
    ''', (FaceRecognizer.MODEL_VERSION,))
    students = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return render_template('students.html', students=students)

@app.route('/api/students/<student_id>/thumbnail')
@login_required('ADMIN', 'FACULTY')
def student_face_thumbnail(student_id):
    conn = get_connection()
    row = conn.execute('''
        SELECT p.image_data, p.mime_type
        FROM student_face_photos p
        JOIN students s ON s.student_id = p.student_id
        WHERE p.student_id = ? AND s.status = 'ACTIVE' AND s.consent_status = 'GRANTED'
    ''', (student_id,)).fetchone()
    conn.close()
    if not row:
        return '', 404
    response = Response(row['image_data'], mimetype=row['mime_type'])
    response.headers['Cache-Control'] = 'private, no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/api/students/register_biometric', methods=['POST'])
@login_required('ADMIN')
def register_student_biometric():
    data = request.json or {}
    student_id = str(data.get('student_id') or '').strip()
    name = str(data.get('name') or '').strip()
    department = str(data.get('department') or 'Computer Science & Engineering').strip()
    section = str(data.get('section') or '').strip()
    class_name = str(data.get('class_name') or '').strip()
    face_samples_b64 = data.get('face_samples', []) # list of base64 face images

    if not student_id or not name or not department or not section or not class_name:
        return jsonify({'success': False, 'error': 'Student ID, name, department, class, and section are required.'}), 400
    if data.get('consent_granted') is not True:
        return jsonify({'success': False, 'error': 'Student consent is required to save biometric templates and a frontal profile thumbnail.'}), 400

    frontal_sample = base64_to_cv2(data.get('frontal_sample'))
    if frontal_sample is None:
        return jsonify({'success': False, 'error': 'A clear frontal face sample is required for the profile thumbnail.'}), 400

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
        return jsonify({'success': False, 'error': 'No usable face images were received. Capture face samples and try again.'}), 400

    # Validate first, then atomically update the profile and replace old templates.
    try:
        reg_result = recognizer.register_student_face_samples(
            student_id,
            cv2_samples,
            detector=camera_manager.detector,
            profile={'name': name, 'department': department, 'section': section, 'class_name': class_name},
            frontal_sample=frontal_sample
        )
    except Exception as exc:
        app.logger.exception("Biometric enrollment persistence failed")
        return jsonify({'success': False, 'error': f'Could not save biometric enrollment: {exc}'}), 500

    camera_manager.recognizer.load_known_faces()
    camera_manager.refresh_student_map()
    camera_manager.reset_trackers()

    if not reg_result['success']:
        return jsonify({'success': False, 'error': reg_result['error'], 'duplicate_flag': reg_result.get('duplicate_flag')}), 400

    return jsonify({
        'success': True,
        'student_id': student_id,
        'name': name,
        'samples_enrolled': reg_result['samples_enrolled']
    })

@app.route('/sessions')
@login_required('ADMIN', 'FACULTY')
def sessions_view():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions ORDER BY session_id DESC")
    sessions_list = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return render_template('sessions.html', sessions=sessions_list)

@app.route('/reports')
@login_required('ADMIN', 'FACULTY')
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
            
            cursor.execute('''
                SELECT s.student_id, s.name, s.department, s.section, s.class_name,
                       a.verified_duration_secs, a.first_seen, a.last_seen, a.status,
                       a.qualification_result, a.review_reason
                FROM session_roster r
                JOIN students s ON s.student_id = r.student_id
                LEFT JOIN attendance a ON a.session_id = r.session_id AND a.student_id = r.student_id
                WHERE r.session_id = ? ORDER BY s.student_id
            ''', (session_id,))
            roster_rows = [dict(r) for r in cursor.fetchall()]

            for s in roster_rows:
                s_id = s['student_id']
                status_str = s.get('status') or 'Absent'
                dur_secs = s.get('verified_duration_secs') or 0

                student_report_list.append({
                    'student_id': s_id,
                    'name': s['name'],
                    'department': s.get('department', 'CSE'),
                    'section': s.get('section', 'Sec-A'),
                    'class_name': s['class_name'],
                    'status': status_str,
                    'verified_duration_secs': dur_secs,
                    'first_seen': s.get('first_seen') or 'N/A',
                    'last_seen': s.get('last_seen') or 'N/A',
                    'qualification_result': s.get('qualification_result') or 'PENDING',
                    'review_reason': s.get('review_reason') or ''
                })

    conn.close()

    return render_template('reports.html',
                           all_sessions=all_sessions,
                           selected_session=selected_session,
                           student_report_list=student_report_list)

@app.route('/reports/download_pdf/<session_id>')
@login_required('ADMIN', 'FACULTY')
def download_pdf_report(session_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    session_row = cursor.fetchone()

    if not session_row:
        conn.close()
        return "Session not found", 404

    session_info = dict(session_row)

    cursor.execute('''
        SELECT s.student_id, s.name, s.class_name,
               a.verified_duration_secs, a.first_seen, a.last_seen, a.status
        FROM session_roster r
        JOIN students s ON s.student_id = r.student_id
        LEFT JOIN attendance a ON a.session_id = r.session_id AND a.student_id = r.student_id
        WHERE r.session_id = ? ORDER BY s.student_id
    ''', (session_id,))
    roster_students = [dict(r) for r in cursor.fetchall()]

    student_summary = []
    for s in roster_students:
        student_summary.append({
            'student_id': s['student_id'],
            'name': s['name'],
            'class_name': s['class_name'],
            'status': s.get('status') or 'Absent',
            'verified_duration_secs': s.get('verified_duration_secs') or 0,
            'first_seen': s.get('first_seen') or 'N/A',
            'last_seen': s.get('last_seen') or 'N/A'
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
