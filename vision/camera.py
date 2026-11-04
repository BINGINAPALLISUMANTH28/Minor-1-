import cv2
import numpy as np
import time
import math
import os
from vision.detector import FaceDetector
from vision.recognizer import FaceRecognizer
from vision.observations import ObservableCueAnalyzer
from vision.tracker import BiometricFaceTracker
from database import get_connection

class CameraManager:
    def __init__(self, mode='simulation', recognizer=None):
        self.mode = mode
        self.cap = None
        self.camera_index = 0
        self.detector = FaceDetector()
        self.recognizer = recognizer or FaceRecognizer()
        self.cue_analyzer = ObservableCueAnalyzer()
        self.tracker = BiometricFaceTracker(alpha=0.20, iou_threshold=0.35)
        # Browser-camera frames and the server video stream are independent
        # sources; keeping separate trackers prevents interleaved frame IDs.
        self.client_tracker = BiometricFaceTracker(alpha=0.20, iou_threshold=0.35)
        self.is_black_frame = False
        
        self.student_name_map = {}
        self.refresh_student_map()
        self._init_camera()

    def set_mode(self, new_mode):
        if self.mode != new_mode:
            self.mode = new_mode
            self.reset_trackers()
            self._init_camera()
        return self.mode

    def reset_trackers(self):
        self.tracker.tracks.clear()
        self.tracker.next_track_id = 1
        self.client_tracker.tracks.clear()
        self.client_tracker.next_track_id = 1

    def refresh_student_map(self):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, name, department, section, class_name FROM students WHERE status = 'ACTIVE'")
        rows = cursor.fetchall()
        self.student_name_map = {}
        for r in rows:
            self.student_name_map[r['student_id']] = {
                'name': r['name'],
                'department': r['department'],
                'section': r['section'],
                'class_name': r['class_name']
            }
        conn.close()

    def _init_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        if self.mode in ['auto', 'webcam']:
            for idx in [0, 1]:
                for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                    try:
                        cap = cv2.VideoCapture(idx, backend)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                mean_brightness = float(np.mean(frame))
                                if mean_brightness > 5.0:
                                    self.cap = cap
                                    self.camera_index = idx
                                    self.mode = 'webcam'
                                    self.is_black_frame = False
                                    return
                                else:
                                    self.cap = cap
                                    self.camera_index = idx
                                    self.mode = 'webcam'
                                    self.is_black_frame = True
                    except Exception:
                        pass

        if self.mode == 'auto':
            self.mode = 'simulation'

    def get_frame_and_analysis(self, session_id=None):
        """
        Processes frame, applies IoU tracking and multi-observation evidence buffer,
        identifies enrolled students cleanly, evaluates observable facial cues,
        accumulates in-slot presence duration, and returns annotated frame + stats.
        """
        self.refresh_student_map()

        # In browser mode the browser-frame endpoint is the sole source of
        # detections. The hidden MJPEG element must not process synthetic frames.
        if self.mode == 'browser':
            frame = self._generate_blank_or_warning_frame("Browser camera is active")
            self.is_black_frame = False
            return frame, []
        
        if self.mode == 'webcam':
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    mean_val = float(np.mean(frame))
                    self.is_black_frame = (mean_val < 5.0)
                else:
                    self.is_black_frame = True
                    frame = self._generate_blank_or_warning_frame("Webcam Hardware Read Error")
            else:
                self.is_black_frame = True
                frame = self._generate_blank_or_warning_frame("Webcam Disconnected")
        else:
            frame = self._generate_simulated_classroom_frame()
            self.is_black_frame = False

        h, w, c = frame.shape
        
        if self.is_black_frame and self.mode == 'webcam':
            cv2.rectangle(frame, (40, h//2 - 60), (w - 40, h//2 + 60), (15, 23, 42), -1)
            cv2.rectangle(frame, (40, h//2 - 60), (w - 40, h//2 + 60), (220, 70, 70), 2)
            cv2.putText(frame, "WEBCAM HARDWARE RETURNED BLACK FRAME", (60, h//2 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(frame, "Check camera physical shutter or switch source to 'Browser Live Camera' or 'Prototype Stream'",
                        (60, h//2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, cv2.LINE_AA)

        raw_detected_faces = self.detector.detect_faces(frame)
        tracked_faces = self.tracker.update(raw_detected_faces, recognizer_func=self.recognizer.identify_face)

        frame_stats = []

        for track_info in tracked_faces:
            x, y, fw, fh = track_info['bbox']
            roi_bgr = track_info['roi_bgr']
            roi_gray = track_info['roi_gray']
            landmarks = track_info.get('landmarks')
            track_id = track_info['track_id']
            student_id = track_info['student_id']
            confidence = track_info['identity_confidence']
            match_status = track_info.get('match_status', 'UNKNOWN')

            if student_id and student_id in self.student_name_map:
                student_name = self.student_name_map[student_id]['name']
            else:
                student_name = f"Track #{track_id} (Unknown)"
                student_id = None

            # Observable Cue Analysis (Replacing mental attention claims)
            cue_info = self.cue_analyzer.analyze_observable_cue(roi_bgr, roi_gray, landmarks=landmarks)
            cue = cue_info['cue']
            cue_label = cue_info['cue_label']

            # Record observation & accumulate presence duration in SQLite if session is ACTIVE
            if session_id and self.mode != 'simulation' and student_id and match_status == 'CONFIRMED':
                self._record_observation_and_presence(session_id, track_id, student_id, confidence, cue)

            # Render Bounding Box Overlay
            if student_id and match_status == 'CONFIRMED':
                box_color = (80, 190, 90) if cue_info['oriented_forward'] else (60, 180, 220)
            elif match_status == 'AMBIGUOUS':
                box_color = (40, 150, 230) # Orange/Amber for ambiguous match
            else:
                box_color = (120, 130, 145) # Muted Gray for unknown visitors

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), box_color, 2)

            if landmarks:
                for k, pos in landmarks.items():
                    if k != 'score':
                        px, py = int(pos[0]), int(pos[1])
                        cv2.circle(frame, (px, py), 3, (220, 230, 242), -1)

            # Display Label: Student Name or Track # (Unknown)
            if student_id:
                label = f"{student_name} | {cue_info['cue_label']}"
            elif match_status == 'AMBIGUOUS':
                label = f"Track #{track_id} | Ambiguous Identity"
            else:
                label = f"Track #{track_id} | Unregistered"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x, y - 22), (x + max(fw, tw + 10), y), (15, 23, 42), -1)
            cv2.putText(frame, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

            frame_stats.append({
                'track_id': track_id,
                'student_id': student_id if student_id else 'UNREGISTERED',
                'name': student_name,
                'bbox': [x, y, fw, fh],
                'observable_cue': cue,
                'cue_label': cue_label,
                'confidence': confidence
            })

        return frame, frame_stats

    def _record_observation_and_presence(self, session_id, track_id, student_id, confidence, observable_cue):
        """Records raw observation log and accumulates verified in-slot presence duration for confirmed students."""
        conn = get_connection()
        cursor = conn.cursor()
        now_time_only = time.strftime('%H:%M:%S')
        now_epoch = time.time()

        # 1. Log observation event
        cursor.execute('''
            INSERT INTO observations (session_id, track_id, student_id, confidence, observable_cue)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, track_id, student_id, float(confidence), observable_cue))

        # 2. Accumulate presence duration for confirmed student in session roster
        if student_id:
            cursor.execute('''
            SELECT verified_duration_secs, first_seen, last_observation_epoch FROM attendance
                WHERE session_id = ? AND student_id = ?
            ''', (session_id, student_id))
            row = cursor.fetchone()

            if row:
                previous_epoch = row['last_observation_epoch']
                elapsed = 0.0 if previous_epoch is None else min(max(now_epoch - previous_epoch, 0.0), 2.0)
                new_duration = float(row['verified_duration_secs'] or 0) + elapsed
                first_seen = row['first_seen'] or now_time_only
                
                # Update qualification status if min duration (e.g. 15 mins = 900s or 1 min for prototype test) is met
                cursor.execute('''
                    UPDATE attendance
                    SET verified_duration_secs = ?, first_seen = ?, last_seen = ?,
                        last_observation_epoch = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ? AND student_id = ?
                ''', (new_duration, first_seen, now_time_only, now_epoch, session_id, student_id))

        conn.commit()
        conn.close()

    def _generate_blank_or_warning_frame(self, message):
        canvas = np.full((720, 1280, 3), (15, 23, 42), dtype=np.uint8)
        cv2.putText(canvas, message, (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 80, 80), 2, cv2.LINE_AA)
        return canvas

    def _generate_simulated_classroom_frame(self):
        canvas = np.full((720, 1280, 3), (235, 240, 245), dtype=np.uint8)
        
        cv2.rectangle(canvas, (0, 0), (1280, 70), (15, 23, 42), -1)
        cv2.putText(canvas, "CLASSROOM CAM-01 (PROTOTYPE STREAM) | DEPT OF CSE | ROOM 302", (24, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (226, 232, 240), 2, cv2.LINE_AA)

        demo_positions = [(140, 200), (380, 200), (620, 200), (260, 420), (500, 420)]
        t = time.time()

        for idx, pos in enumerate(demo_positions):
            bx, by = pos
            sway_x = int(math.sin(t * 1.5 + idx) * 3)
            sway_y = int(math.cos(t * 1.2 + idx) * 2)
            
            cx = bx + sway_x
            cy = by + sway_y
            
            cv2.ellipse(canvas, (cx, cy + 90), (75, 50), 0, 180, 360, (70, 80, 95), -1)

            face_w, face_h = 60, 75
            cv2.ellipse(canvas, (cx, cy), (face_w, face_h), 0, 0, 360, (215, 225, 235), -1)
            cv2.ellipse(canvas, (cx, cy), (face_w, face_h), 0, 0, 360, (90, 100, 115), 2)

            is_distracted = (idx == 1 and int(t) % 10 > 5) or (idx == 4 and int(t) % 8 > 3)

            if not is_distracted:
                cv2.circle(canvas, (cx - 20, cy - 10), 7, (20, 30, 40), -1)
                cv2.circle(canvas, (cx + 20, cy - 10), 7, (20, 30, 40), -1)
                cv2.circle(canvas, (cx - 20, cy - 10), 2, (255, 255, 255), -1)
                cv2.circle(canvas, (cx + 20, cy - 10), 2, (255, 255, 255), -1)
                cv2.line(canvas, (cx, cy), (cx, cy + 12), (60, 70, 85), 2)
                cv2.line(canvas, (cx - 14, cy + 28), (cx + 14, cy + 28), (50, 60, 75), 2)
            else:
                cv2.line(canvas, (cx - 25, cy - 5), (cx - 15, cy - 5), (20, 30, 40), 2)
                cv2.line(canvas, (cx + 15, cy - 5), (cx + 25, cy - 5), (20, 30, 40), 2)
                cv2.line(canvas, (cx, cy + 5), (cx, cy + 15), (60, 70, 85), 2)
                cv2.line(canvas, (cx - 12, cy + 32), (cx + 12, cy + 32), (80, 90, 100), 2)

            cv2.rectangle(canvas, (cx - 85, cy + 115), (cx + 85, cy + 135), (175, 185, 200), -1)

        return canvas

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
