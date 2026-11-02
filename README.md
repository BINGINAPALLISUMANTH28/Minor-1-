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
   - Extracts spatial HOG, LBP, and intensity histograms.
   - Computes cosine distance similarity against registered student embeddings.

4. **Automatic Attendance (`FR-04`)**:
   - Marks recognized students as **Present** once per active monitoring session.
   - Prevents duplicate counting and maintains accurate presence logs.

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

2. Seed the database with initial prototype student encodings:
   ```bash
   python seed_db.py
   ```

3. Run system verification tests:
   ```bash
   python test_app.py
   ```

4. Launch the application server:
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000` in your web browser.

---

## Database Schema Overview

- **`students`**: `student_id`, `name`, `class_name`, `face_image_path`, `created_at`
- **`face_encodings`**: `id`, `student_id`, `encoding_json`
- **`sessions`**: `session_id`, `class_name`, `subject`, `date`, `start_time`, `end_time`, `status`
- **`attendance`**: `attendance_id`, `session_id`, `student_id`, `status`, `marked_at`
- **`monitoring_logs`**: `log_id`, `session_id`, `student_id`, `timestamp`, `attention_status`, `attention_score`

---
© 2026 AI Smart Classroom Monitoring System. All rights reserved.
