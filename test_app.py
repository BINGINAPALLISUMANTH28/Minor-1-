"""Isolated integration checks for authentication, onboarding, and attendance APIs.

Run with ``python test_app.py``. All records and generated reports stay in a
temporary directory; the configured classroom database is never opened.
"""
import os
import tempfile
import time
import unittest


_test_data = tempfile.TemporaryDirectory(
    prefix=".minor_attendance_check_", dir=os.path.dirname(__file__)
)
os.environ["CLASSROOM_DB_PATH"] = os.path.join(_test_data.name, "classroom.db")
os.environ["ADMIN_INITIAL_PASSWORD"] = "test-admin-password"
os.environ["FACULTY_INITIAL_PASSWORD"] = "test-faculty-password"

from app import app, camera_manager  # noqa: E402
from database import get_connection  # noqa: E402


class AttendanceFlowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)

    def test_auth_onboarding_and_live_attendance_flow(self):
        client = app.test_client()

        self.assertEqual(client.get("/api/live_status").status_code, 401)
        login = client.post("/api/auth/login", json={
            "username": "admin", "password": "test-admin-password"
        })
        self.assertEqual(login.status_code, 200, login.get_json())
        self.assertEqual(client.get("/").status_code, 200)

        # An invalid/empty enrollment must not create a student profile.
        bad_enrollment = client.post("/api/students/register_biometric", json={
            "student_id": "TEST-001", "name": "Test Student", "section": "A",
            "class_name": "CSE Test", "face_samples": []
        })
        self.assertEqual(bad_enrollment.status_code, 400)
        conn = get_connection()
        self.assertIsNone(conn.execute(
            "SELECT student_id FROM students WHERE student_id = 'TEST-001'"
        ).fetchone())
        conn.execute("""
            INSERT INTO students (student_id, name, department, section, class_name)
            VALUES ('TEST-001', 'Test Student', 'CSE', 'A', 'CSE Test')
        """)
        conn.commit()
        conn.close()

        # Prototype frames are deliberately prohibited from marking attendance.
        blocked = client.post("/api/sessions/start", json={
            "class_name": "CSE Test", "section": "A"
        })
        self.assertEqual(blocked.status_code, 409)

        self.assertEqual(client.post("/api/camera/mode", json={
            "mode": "browser"
        }).status_code, 200)
        started = client.post("/api/sessions/start", json={
            "class_name": "CSE Test", "section": "A", "subject": "Verification",
            "min_duration_req_mins": 1
        })
        self.assertEqual(started.status_code, 200, started.get_json())
        session_id = started.get_json()["session"]["session_id"]

        status = client.get("/api/live_status")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.get_json()["total_registered"], 1)
        self.assertEqual(status.get_json()["students"][0]["student_id"], "TEST-001")

        # Presence only accrues for a confirmed identity in the session roster.
        camera_manager._record_observation_and_presence(
            session_id, 1, "TEST-001", 0.9, "FACE_VISIBLE_FORWARD"
        )
        time.sleep(0.15)
        camera_manager._record_observation_and_presence(
            session_id, 1, "TEST-001", 0.9, "FACE_VISIBLE_FORWARD"
        )
        conn = get_connection()
        duration = conn.execute(
            "SELECT verified_duration_secs FROM attendance WHERE session_id = ? AND student_id = ?",
            (session_id, "TEST-001")
        ).fetchone()[0]
        conn.close()
        self.assertGreater(duration, 0)
        self.assertLess(duration, 1.0)

        stopped = client.post("/api/sessions/stop")
        self.assertEqual(stopped.status_code, 200, stopped.get_json())
        report = client.get(f"/reports/download_pdf/{session_id}")
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.mimetype, "application/pdf")


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        _test_data.cleanup()
