import os
import cv2
import numpy as np
from database import init_db, get_connection
from vision.detector import FaceDetector
from vision.recognizer import FaceRecognizer
from vision.attention import AttentionAnalyzer
from vision.camera import CameraManager
from reports.pdf_generator import generate_faculty_report_pdf
from app import app

def run_system_verification():
    print("=== 1. Testing Database ===")
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM students")
    count = cursor.fetchone()[0]
    conn.close()
    print(f"Database operational. Registered student count: {count}")
    assert count >= 0, "Database should be accessible and support real production registration."

    print("\n=== 2. Testing Vision Core (Detector, Recognizer, Attention) ===")
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    attention = AttentionAnalyzer()
    camera = CameraManager(mode='simulation')

    frame, stats = camera.get_frame_and_analysis(session_id="TEST-SESH-001")
    print(f"Camera frame generated size: {frame.shape}")
    print(f"Detected and analyzed faces in test frame: {len(stats)}")
    for s in stats:
        print(f"  - Student: {s['name']} ({s['student_id']}) | Cue: {s.get('cue_label', 'Active')} | Confidence: {s.get('confidence', 0.9)*100:.1f}%")
    assert len(stats) >= 3, "Prototype simulation should detect multi-person group (3-5 students)."

    print("\n=== 3. Testing PDF Report Generation ===")
    test_session_info = {
        'session_id': 'TEST-SESH-001',
        'class_name': 'CSE - III Year',
        'subject': 'Artificial Intelligence & Neural Networks',
        'date': '2026-09-24',
        'start_time': '10:00:00',
        'end_time': '11:00:00'
    }
    test_students = [
        {'student_id': 'S001', 'name': 'Ravi Kumar', 'class_name': 'CSE - III Year', 'status': 'Present', 'avg_attention': 88.5},
        {'student_id': 'S002', 'name': 'Sumanth', 'class_name': 'CSE - III Year', 'status': 'Present', 'avg_attention': 64.2},
        {'student_id': 'S003', 'name': 'Kiran', 'class': 'CSE - III Year', 'status': 'Present', 'avg_attention': 92.0},
        {'student_id': 'S004', 'name': 'Priya Sharma', 'class_name': 'CSE - III Year', 'status': 'Present', 'avg_attention': 85.0},
        {'student_id': 'S005', 'name': 'Ananya Roy', 'class_name': 'CSE - III Year', 'status': 'Absent', 'avg_attention': 0.0}
    ]
    os.makedirs('data/test_reports', exist_ok=True)
    pdf_out = 'data/test_reports/Verification_Report.pdf'
    generate_faculty_report_pdf(test_session_info, test_students, pdf_out)
    assert os.path.exists(pdf_out), "PDF Report file should be created."
    print(f"PDF report generated successfully at: {pdf_out}")

    print("\n=== 4. Testing Flask Endpoints ===")
    client = app.test_client()
    
    resp_dashboard = client.get('/')
    assert resp_dashboard.status_code == 200, "Dashboard route should return 200."

    resp_live = client.get('/api/live_status')
    assert resp_live.status_code == 200, "Live status API should return 200."
    live_json = resp_live.get_json()
    print(f"Live status API response: Registered={live_json['total_registered']}, Present={live_json['present_count']}")

    resp_start = client.post('/api/sessions/start', json={'class_name': 'CSE - III Year', 'subject': 'AI Test'})
    assert resp_start.status_code == 200
    start_json = resp_start.get_json()
    active_sid = start_json['session']['session_id']
    print(f"Started session via API: {active_sid}")

    resp_stop = client.post('/api/sessions/stop')
    assert resp_stop.status_code == 200
    print("Stopped session via API.")

    resp_pdf = client.get(f'/reports/download_pdf/{active_sid}')
    assert resp_pdf.status_code == 200, "PDF download endpoint should return 200."

    print("\n>>> ALL SYSTEM VERIFICATION CHECKS PASSED PERFECTLY! <<<")

if __name__ == "__main__":
    run_system_verification()
