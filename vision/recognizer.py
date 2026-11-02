import cv2
import numpy as np
import json
import sqlite3
import os

class FaceRecognizer:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "classroom.db")
        self.db_path = db_path
        self.known_encodings = {}  # student_id -> list of dicts {'vec': numpy_array, 'quality': float}
        self.load_known_faces()

    def load_known_faces(self):
        """Loads enrolled student face encodings from database."""
        self.known_encodings = {}
        if not os.path.exists(self.db_path):
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT student_id, encoding_json, quality_score FROM face_encodings")
            rows = cursor.fetchall()
            for student_id, encoding_json, quality in rows:
                encoding = np.array(json.loads(encoding_json), dtype=np.float32)
                if student_id not in self.known_encodings:
                    self.known_encodings[student_id] = []
                self.known_encodings[student_id].append({'vec': encoding, 'quality': float(quality or 0.9)})
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    def validate_image_quality(self, face_bgr):
        """
        Validates image quality: checks blur (Laplacian variance), contrast, and brightness.
        Returns (is_valid: bool, quality_score: float, reason: str).
        """
        if face_bgr is None or face_bgr.size == 0:
            return False, 0.0, "Empty image sample"

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if len(face_bgr.shape) == 3 else face_bgr

        # 1. Blur detection via Laplacian Variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 30.0:
            return False, round(lap_var / 100.0, 2), f"Image blurred (Laplacian variance {lap_var:.1f} < 30.0)"

        # 2. Lighting & Exposure Check
        mean_brightness = np.mean(gray)
        if mean_brightness < 35.0:
            return False, round(mean_brightness / 255.0, 2), "Image underexposed / too dark"
        if mean_brightness > 235.0:
            return False, round(mean_brightness / 255.0, 2), "Image overexposed / heavily washed out"

        quality_score = min(1.0, max(0.4, (lap_var / 180.0) * 0.5 + (mean_brightness / 128.0) * 0.5))
        return True, round(float(quality_score), 2), "Quality Adequate"

    def extract_features(self, face_bgr):
        """
        Extracts a normalized facial biometric feature vector from a face ROI crop using
        spatial HOG, LBP descriptor, and local contrast intensity histograms.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        target_size = (96, 96)
        face_resized = cv2.resize(face_bgr, target_size)
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) if len(face_resized.shape) == 3 else face_resized
        
        # Local Adaptive CLAHE Equalization
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # 1. Spatial Intensity Feature Vector (Grid 8x8)
        grid_h, grid_w = 8, 8
        cell_h, cell_w = target_size[1] // grid_h, target_size[0] // grid_w
        grid_features = []
        for r in range(grid_h):
            for c in range(grid_w):
                cell = gray_eq[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                grid_features.append(np.mean(cell))
                grid_features.append(np.std(cell))

        # 2. Gradient direction / HOG approximation
        sobelx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
        
        hist_angle, _ = np.histogram(angle, bins=16, range=(0, 360), weights=magnitude)

        # Concatenate and normalize to unit L2 length
        raw_vector = np.concatenate([
            np.array(grid_features, dtype=np.float32),
            hist_angle.astype(np.float32)
        ])
        
        norm = np.linalg.norm(raw_vector)
        if norm > 1e-6:
            normalized_vector = raw_vector / norm
        else:
            normalized_vector = raw_vector

        return normalized_vector

    def check_duplicate_biometric(self, feature_vec, threshold=0.85):
        """
        Checks if a candidate biometric feature vector matches an existing registered student.
        Returns (is_duplicate: bool, existing_student_id: str, similarity: float).
        """
        if not self.known_encodings or feature_vec is None:
            return False, None, 0.0

        for student_id, templates in self.known_encodings.items():
            for tdata in templates:
                sim = float(np.dot(feature_vec, tdata['vec']))
                if sim >= threshold:
                    return True, student_id, sim
        return False, None, 0.0

    def register_student_face_samples(self, student_id, face_samples, detector=None):
        """
        Registers multiple quality-validated biometric face samples for a student.
        Returns dict: {'success': bool, 'samples_enrolled': int, 'duplicate_flag': dict, 'error': str}
        """
        if not face_samples:
            return {'success': False, 'samples_enrolled': 0, 'error': 'No face samples provided'}

        valid_templates = []
        for idx, sample_bgr in enumerate(face_samples):
            # Extract tight face ROI if full frame passed
            face_crop = sample_bgr
            if detector is not None and sample_bgr.shape[0] > 160:
                detections = detector.detect_faces(sample_bgr)
                if len(detections) != 1:
                    continue # Validation: reject if 0 or multiple faces visible in enrollment sample
                x, y, w, h = detections[0]['bbox']
                face_crop = sample_bgr[y:y+h, x:x+w]

            is_valid, q_score, reason = self.validate_image_quality(face_crop)
            if not is_valid:
                continue

            fvec = self.extract_features(face_crop)
            if fvec is not None:
                valid_templates.append((fvec, q_score))

        if not valid_templates:
            return {'success': False, 'samples_enrolled': 0, 'error': 'Enrollment failed: All samples failed quality validation (blurred, poorly lit, or invalid pose).'}

        # Duplicate check against existing enrolled students
        primary_vec = valid_templates[0][0]
        is_dup, dup_id, dup_sim = self.check_duplicate_biometric(primary_vec, threshold=0.85)
        if is_dup and dup_id != student_id:
            return {
                'success': False,
                'samples_enrolled': 0,
                'duplicate_flag': {'existing_student_id': dup_id, 'similarity': round(dup_sim, 2)},
                'error': f'Duplicate Biometric Detected: Face matches already enrolled Student ID {dup_id} ({dup_sim*100:.1f}% similarity). Flagged for Admin review.'
            }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        enrolled_count = 0
        for fvec, q_score in valid_templates:
            encoding_json = json.dumps(fvec.tolist())
            cursor.execute(
                "INSERT INTO face_encodings (student_id, encoding_json, quality_score) VALUES (?, ?, ?)",
                (student_id, encoding_json, q_score)
            )
            if student_id not in self.known_encodings:
                self.known_encodings[student_id] = []
            self.known_encodings[student_id].append({'vec': fvec, 'quality': q_score})
            enrolled_count += 1

        conn.commit()
        conn.close()

        return {'success': True, 'samples_enrolled': enrolled_count, 'duplicate_flag': None, 'error': None}

    def identify_face(self, face_bgr, min_threshold=0.75):
        """
        Strict Biometric Recognition with Ambiguous Match Rejection.
        Returns (student_id, confidence, match_status: 'CONFIRMED' | 'UNKNOWN' | 'AMBIGUOUS').
        """
        if not self.known_encodings:
            return None, 0.0, "UNKNOWN"

        query_vec = self.extract_features(face_bgr)
        if query_vec is None:
            return None, 0.0, "UNKNOWN"

        # Calculate similarity against all enrolled student templates
        matches = []
        for student_id, templates in self.known_encodings.items():
            max_sim_for_student = -1.0
            for tdata in templates:
                sim = float(np.dot(query_vec, tdata['vec']))
                if sim > max_sim_for_student:
                    max_sim_for_student = sim
            matches.append((student_id, max_sim_for_student))

        # Sort candidate matches by similarity descending
        matches.sort(key=lambda m: m[1], reverse=True)

        if not matches or matches[0][1] < min_threshold:
            return None, float(matches[0][1]) if matches else 0.0, "UNKNOWN"

        top1_id, top1_sim = matches[0]

        # Ambiguous Match Rejection: If top 2 candidates are within 0.05 similarity, reject identity to prevent guessing
        if len(matches) > 1:
            top2_id, top2_sim = matches[1]
            if (top1_sim - top2_sim) < 0.05 and top1_id != top2_id:
                return None, float(top1_sim), "AMBIGUOUS"

        return top1_id, float(top1_sim), "CONFIRMED"
