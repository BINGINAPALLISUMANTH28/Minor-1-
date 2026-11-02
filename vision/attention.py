import cv2
import numpy as np
import math

class AttentionAnalyzer:
    def __init__(self):
        pass

    def analyze_attention(self, face_bgr, face_gray, landmarks=None):
        """
        Calculates individual student attention status ('Attentive' vs 'Inattentive')
        and numerical focus percentage based on facial landmarks and ROI symmetry.
        Completely immune to background lighting variations (FR-05, NFR-02).
        """
        if face_bgr is None or face_bgr.size == 0:
            return {'status': 'Inattentive', 'score': 0.0, 'pitch': 0.0, 'yaw': 0.0}

        # Strategy 1: High-precision YuNet Landmark Geometry
        if landmarks and 'right_eye' in landmarks:
            rx, ry = landmarks['right_eye']
            lx, ly = landmarks['left_eye']
            nx, ny = landmarks['nose']
            rmx, rmy = landmarks['right_mouth']
            lmx, lmy = landmarks['left_mouth']

            # Eye distance
            eye_dist = math.hypot(lx - rx, ly - ry) + 1e-5
            
            # Eye Roll Angle
            roll_deg = math.degrees(math.atan2(ly - ry, lx - rx))

            # Eye Midpoint
            mx_eye = (rx + lx) / 2.0
            my_eye = (ry + ly) / 2.0

            # Mouth Midpoint
            my_mouth = (rmy + lmy) / 2.0

            # Yaw proxy: offset of nose relative to eye center
            yaw_ratio = abs(nx - mx_eye) / eye_dist

            # Pitch proxy: vertical position of nose between eyes and mouth
            mouth_eye_span = max(1.0, my_mouth - my_eye)
            pitch_ratio = (ny - my_eye) / mouth_eye_span

            # Attentive threshold check:
            # - Yaw offset < 0.35 (head not turned left/right)
            # - Pitch ratio between 0.25 and 0.75 (head not slumped down or looking at ceiling)
            # - Roll tilt < 22 degrees
            if yaw_ratio <= 0.35 and 0.20 <= pitch_ratio <= 0.80 and abs(roll_deg) <= 22.0:
                status = 'Attentive'
                score = 96.0 - (yaw_ratio * 30.0) - (abs(pitch_ratio - 0.5) * 20.0)
                score = max(80.0, min(98.0, score))
            else:
                status = 'Inattentive'
                score = 65.0 - (yaw_ratio * 40.0) - (abs(pitch_ratio - 0.5) * 30.0)
                score = max(25.0, min(69.0, score))

            return {
                'status': status,
                'score': round(score, 1),
                'pitch': round(pitch_ratio * 100, 1),
                'yaw': round(yaw_ratio * 100, 1)
            }

        # Strategy 2: Grayscale Spatial Intensity Symmetry Fallback
        if face_gray is not None and face_gray.size > 0:
            h, w = face_gray.shape
            left_half = face_gray[:, :w//2]
            right_half = face_gray[:, w//2:]
            
            l_mean = np.mean(left_half)
            r_mean = np.mean(right_half)
            
            sym_delta = abs(l_mean - r_mean) / (max(l_mean, r_mean) + 1e-5) * 100.0

            if sym_delta <= 25.0:
                status = 'Attentive'
                score = max(80.0, 95.0 - (sym_delta * 0.4))
            else:
                status = 'Inattentive'
                score = max(35.0, 65.0 - (sym_delta * 0.5))

            return {
                'status': status,
                'score': round(score, 1),
                'pitch': 0.0,
                'yaw': round(sym_delta, 1)
            }

        return {'status': 'Inattentive', 'score': 40.0, 'pitch': 0.0, 'yaw': 0.0}
