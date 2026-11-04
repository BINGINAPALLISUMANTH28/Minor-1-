import cv2
import json
import numpy as np
import os
import sqlite3

from database import DB_PATH


class FaceRecognizer:
    """SFace embeddings with YuNet five-landmark alignment."""

    MODEL_VERSION = 'opencv_sface_2021dec_v1'
    COSINE_THRESHOLD = 0.363
    AMBIGUITY_MARGIN = 0.08

    def __init__(self, db_path=None, model_path=None):
        self.db_path = db_path or DB_PATH
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models',
            'face_recognition_sface_2021dec.onnx'
        )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f'SFace model not found at {self.model_path}. Restore the model file and restart the app.'
            )
        if not hasattr(cv2, 'FaceRecognizerSF'):
            raise RuntimeError('This OpenCV build does not include FaceRecognizerSF; upgrade opencv-python.')
        self.model = cv2.FaceRecognizerSF.create(self.model_path, '')
        self.known_encodings = {}
        self.load_known_faces()

    def load_known_faces(self):
        """Load only embeddings created by the current SFace model."""
        self.known_encodings = {}
        if not os.path.exists(self.db_path):
            return

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT student_id, encoding_json, quality_score
                FROM face_encodings WHERE model_version = ?
            ''', (self.MODEL_VERSION,))
            for student_id, encoding_json, quality in cursor.fetchall():
                try:
                    encoding = np.asarray(json.loads(encoding_json), dtype=np.float32).reshape(1, -1)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if encoding.shape[1] != 128 or not np.isfinite(encoding).all():
                    continue
                self.known_encodings.setdefault(student_id, []).append({
                    'vec': encoding,
                    'quality': float(quality or 0.9)
                })
        except sqlite3.OperationalError:
            # init_db adds model_version to pre-existing databases on startup.
            pass
        finally:
            conn.close()

    def validate_image_quality(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return False, 0.0, 'Empty image sample'
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 30.0:
            return False, round(lap_var / 100.0, 2), f'Image blurred (Laplacian variance {lap_var:.1f} < 30.0)'
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 35.0:
            return False, round(mean_brightness / 255.0, 2), 'Image underexposed / too dark'
        if mean_brightness > 235.0:
            return False, round(mean_brightness / 255.0, 2), 'Image overexposed / heavily washed out'
        quality = min(1.0, max(0.4, (lap_var / 180.0) * 0.5 + (mean_brightness / 128.0) * 0.5))
        return True, round(float(quality), 2), 'Quality adequate'

    def extract_features(self, face_info, detector=None):
        """Return an L2-normalized 128-value SFace descriptor for one detected face."""
        if isinstance(face_info, dict) and 'frame_bgr' in face_info and 'face_vector' in face_info:
            source = face_info['frame_bgr']
            detection = face_info['face_vector']
        elif detector is not None and isinstance(face_info, np.ndarray):
            detected = detector.detect_faces(face_info)
            if len(detected) != 1:
                return None
            source = detected[0]['frame_bgr']
            detection = detected[0]['face_vector']
        else:
            return None

        try:
            aligned_face = self.model.alignCrop(source, detection)
            feature = self.model.feature(aligned_face)
        except cv2.error:
            return None
        if feature is None or feature.size != 128:
            return None
        feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(feature))
        if norm <= 1e-8:
            return None
        return feature / norm

    @staticmethod
    def _cosine_similarity(feature_a, feature_b):
        return float(np.dot(feature_a.reshape(-1), feature_b.reshape(-1)))

    def check_duplicate_biometric(self, feature_vec, threshold=COSINE_THRESHOLD, exclude_student_id=None):
        if feature_vec is None:
            return False, None, 0.0
        best = (None, -1.0)
        for student_id, templates in self.known_encodings.items():
            if student_id == exclude_student_id:
                continue
            for template in templates:
                similarity = self._cosine_similarity(feature_vec, template['vec'])
                if similarity > best[1]:
                    best = (student_id, similarity)
        if best[0] is not None and best[1] >= threshold:
            return True, best[0], best[1]
        return False, None, max(0.0, best[1])

    def register_student_face_samples(self, student_id, face_samples, detector=None, profile=None, frontal_sample=None):
        if not face_samples:
            return {'success': False, 'samples_enrolled': 0, 'error': 'No face samples provided'}
        if detector is None:
            return {'success': False, 'samples_enrolled': 0, 'error': 'Face detector is required to align enrollment images.'}
        if frontal_sample is None:
            return {'success': False, 'samples_enrolled': 0, 'error': 'Capture a frontal sample to save the consented profile thumbnail.'}

        frontal_detections = detector.detect_faces(frontal_sample)
        if len(frontal_detections) != 1:
            return {'success': False, 'samples_enrolled': 0, 'error': 'The first sample must show exactly one clear frontal face for the profile thumbnail.'}
        frontal_roi = frontal_detections[0]['roi_bgr']
        frontal_quality_ok, _, _ = self.validate_image_quality(frontal_roi)
        if not frontal_quality_ok:
            return {'success': False, 'samples_enrolled': 0, 'error': 'The frontal sample is too dark or blurry to use as a profile thumbnail. Capture it again in clear lighting.'}
        face_h, face_w = frontal_roi.shape[:2]
        resize_scale = min(1.0, 320.0 / max(face_h, face_w))
        thumbnail = cv2.resize(
            frontal_roi,
            (max(1, int(face_w * resize_scale)), max(1, int(face_h * resize_scale))),
            interpolation=cv2.INTER_AREA if resize_scale < 1.0 else cv2.INTER_LINEAR
        )
        encoded_ok, thumbnail_jpeg = cv2.imencode('.jpg', thumbnail, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not encoded_ok:
            return {'success': False, 'samples_enrolled': 0, 'error': 'Could not prepare the frontal profile thumbnail. Please capture it again.'}

        valid_templates = []
        for sample in face_samples:
            detections = detector.detect_faces(sample)
            if len(detections) != 1:
                continue
            detection = detections[0]
            is_valid, quality, _ = self.validate_image_quality(detection['roi_bgr'])
            if not is_valid:
                continue
            feature = self.extract_features(detection)
            if feature is not None:
                valid_templates.append((feature, quality))

        if len(valid_templates) < 3:
            return {
                'success': False,
                'samples_enrolled': len(valid_templates),
                'error': f'Enrollment needs at least 3 quality-validated face samples; only {len(valid_templates)} passed. Capture more samples and try again.'
            }

        # Reject enrollment if any captured view strongly matches another student.
        duplicate_matches = []
        for feature, _ in valid_templates:
            is_duplicate, duplicate_id, similarity = self.check_duplicate_biometric(
                feature, threshold=self.COSINE_THRESHOLD, exclude_student_id=student_id
            )
            if is_duplicate:
                duplicate_matches.append((duplicate_id, similarity))
        if duplicate_matches:
            duplicate_id, similarity = max(duplicate_matches, key=lambda item: item[1])
            return {
                'success': False,
                'samples_enrolled': 0,
                'duplicate_flag': {'existing_student_id': duplicate_id, 'similarity': round(similarity, 3)},
                'error': f'This face matches existing student {duplicate_id} (cosine {similarity:.3f}); enrollment was not saved.'
            }

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            if profile:
                cursor.execute('''
                    INSERT INTO students (student_id, name, department, section, class_name, consent_status, is_biometric_enrolled)
                    VALUES (?, ?, ?, ?, ?, 'GRANTED', 1)
                    ON CONFLICT(student_id) DO UPDATE SET
                        name = excluded.name,
                        department = excluded.department,
                        section = excluded.section,
                        class_name = excluded.class_name,
                        consent_status = 'GRANTED',
                        status = 'ACTIVE',
                        is_biometric_enrolled = 1
                ''', (student_id, profile['name'], profile['department'], profile['section'], profile['class_name']))

            cursor.execute("DELETE FROM face_encodings WHERE student_id = ? AND model_version = ?", (student_id, self.MODEL_VERSION))
            cursor.executemany('''
                INSERT INTO face_encodings (student_id, encoding_json, model_version, quality_score)
                VALUES (?, ?, ?, ?)
            ''', [
                (student_id, json.dumps(feature.reshape(-1).tolist()), self.MODEL_VERSION, quality)
                for feature, quality in valid_templates
            ])
            cursor.execute('''
                INSERT INTO student_face_photos (student_id, image_data, mime_type, updated_at)
                VALUES (?, ?, 'image/jpeg', CURRENT_TIMESTAMP)
                ON CONFLICT(student_id) DO UPDATE SET
                    image_data = excluded.image_data,
                    mime_type = excluded.mime_type,
                    updated_at = CURRENT_TIMESTAMP
            ''', (student_id, sqlite3.Binary(thumbnail_jpeg.tobytes())))
            conn.commit()
        except Exception:
            conn.rollback()
            conn.close()
            raise
        conn.close()

        self.known_encodings[student_id] = [
            {'vec': feature, 'quality': quality} for feature, quality in valid_templates
        ]
        return {'success': True, 'samples_enrolled': len(valid_templates), 'duplicate_flag': None, 'error': None}

    def identify_face(self, face_info):
        if not self.known_encodings:
            return None, 0.0, 'UNKNOWN'
        query_feature = self.extract_features(face_info)
        if query_feature is None:
            return None, 0.0, 'UNKNOWN'

        scores = []
        for student_id, templates in self.known_encodings.items():
            # Use the best template for each person, then compare the top two.
            score = max(self._cosine_similarity(query_feature, template['vec']) for template in templates)
            scores.append((student_id, score))
        scores.sort(key=lambda entry: entry[1], reverse=True)

        top_id, top_score = scores[0]
        if top_score < self.COSINE_THRESHOLD:
            return None, max(0.0, top_score), 'UNKNOWN'
        if len(scores) > 1 and top_score - scores[1][1] < self.AMBIGUITY_MARGIN:
            return None, top_score, 'AMBIGUOUS'
        return top_id, top_score, 'CONFIRMED'
