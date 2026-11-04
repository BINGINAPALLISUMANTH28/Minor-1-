# AI-Based Smart Classroom Monitoring System

An enterprise-grade, real-time AI solution for multi-person face detection, student identification, automatic attendance recording, individual attention monitoring, and automated faculty report generation.

---

## Architecture & System Overview

```
                      ┌───────────────────────────────┐
                      │    HD Webcam / Camera Feed    │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │ OpenCV YuNet AI Face Detector │
                      │  (Multi-Person Landmarking)   │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │  Spatial Feature Recognizer   │
                      │  (Student Identification)     │
                      └───────────────┬───────────────┘
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
       ┌──────────────────────────────┐  ┌──────────────────────────────┐
       │   Automatic Attendance Engine│  │ Attention Geometry Analyzer  │
       │   (Session Attendance Log)   │  │ (Yaw, Pitch & Eye Focus)     │
       └──────────────┬───────────────┘  └──────────────┬───────────────┘
                      │                                 │
                      └───────────────┬─────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    SQLite Central Database    │
                      │  (Telemetry & Audit Records)  │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │  Faculty Dashboard & Reports  │
                      │ (Live Telemetry & PDF Export) │
                      └───────────────────────────────┘
```

---

## Core System Modules

1. **Student Registration (`FR-01`)**:
   - Registers student metadata (Student ID, Name, Class/Section).
   - Generates and stores normalized facial feature descriptors in SQLite.

2. **Multi-Person Face Detection (`FR-02`)**:
   - Uses OpenCV YuNet AI ONNX deep learning face detector (`cv2.FaceDetectorYN`).
   - Supports 3–5 students simultaneously in real-time camera view.

3. **Face Recognition (`FR-03`)**:
   - Aligns detected faces with YuNet's five landmarks and creates normalized 128-value OpenCV SFace embeddings.
   - Compares embeddings by cosine similarity; older HOG templates are retained in the database but are not used for SFace matching.
   - Requires at least three quality-checked face samples for enrollment. Existing students must enroll again after upgrading to SFace.
   - With student consent, stores a small frontal thumbnail separately for the profile card; the thumbnail is not used for recognition.

4. **Automatic Attendance (`FR-04`)**:
   - Accumulates presence time only for confirmed identities on the session's class-and-section roster.
   - Qualifies attendance against the configured minimum duration when the session ends.
   - Prototype/simulation frames cannot start an attendance session.

5. **Individual Attention Monitoring (`FR-05`)**:
   - Computes dynamic head pose geometry (Yaw, Pitch, Roll) and 5 facial landmark positions.
   - Classifies status as **Attentive** or **Inattentive** per student.
   - Fully immune to background room lighting variations (**NFR-02**).

6. **Real-Time Monitoring Dashboard (`FR-06`)**:
   - Built with a clean chromatic Dark Obsidian & Slate UI.
   - Live MJPEG stream with dynamic bounding box overlays and attention indicators.
   - Real-time telemetry table updated asynchronously via REST APIs (`/api/live_status`).

7. **Data Management (`FR-07` & `FR-08`)**:
   - SQLite backend (`classroom.db`) managing `students`, `face_encodings`, `sessions`, `attendance`, and `monitoring_logs`.

8. **Faculty Report Generation (`FR-09`)**:
   - Summarizes total registered, present count, absent count, and classroom focus score.
   - Generates downloadable official PDF reports using ReportLab (`/reports/download_pdf/<session_id>`).

---

## Setup & Execution Instructions

### Prerequisites
- Python 3.10+
- Webcam (Optional; includes automated multi-person classroom camera stream fallback)

### Installation
1. Install required dependencies:
   ```bash
   python -m pip install opencv-python numpy pillow flask reportlab
   ```

2. Add optional sample student profiles (this does not create fake face templates):
   ```bash
   python seed_db.py
   ```
   Then sign in as an administrator, open **Student Directory**, and enroll each real student with their camera samples.

3. Launch the application server:
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000` in your web browser.

On first initialization, the server prints one-time randomly generated `admin` and `faculty` passwords. Set `ADMIN_INITIAL_PASSWORD` and `FACULTY_INITIAL_PASSWORD` before first startup to choose them yourself. Legacy prototype passwords are replaced and the new credentials are printed once. Set `FLASK_SECRET_KEY` to a long random value for deployments; otherwise the app stores a generated key in its local SQLite configuration table.

---

## Database Schema Overview

- **`students`**: `student_id`, `name`, `class_name`, `face_image_path`, `created_at`
- **`face_encodings`**: `id`, `student_id`, `encoding_json`
- **`sessions`**: `session_id`, `class_name`, `subject`, `date`, `start_time`, `end_time`, `status`
- **`attendance`**: `attendance_id`, `session_id`, `student_id`, `status`, `marked_at`
- **`monitoring_logs`**: `log_id`, `session_id`, `student_id`, `timestamp`, `attention_status`, `attention_score`

---
© 2026 AI Smart Classroom Monitoring System. All rights reserved.
