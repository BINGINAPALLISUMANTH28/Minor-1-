import cv2
import numpy as np
import math

class ObservableCueAnalyzer:
    """
    Analyzes observable face cues (visibility, estimated head orientation toward reference direction,
    and observation quality) without claiming definitive mental attention or internal cognitive state.
    """
    def __init__(self):
        pass

    def analyze_observable_cue(self, face_bgr, face_gray, landmarks=None, quality_score=1.0):
        """
        Returns objective observable cue dict:
            {
                'cue': 'FACE_VISIBLE_FORWARD' | 'FACE_VISIBLE_ELSEWHERE' | 'UNCERTAIN' | 'NOT_DETECTED',
                'cue_label': str,
                'oriented_forward': bool,
                'quality_score': float,
                'yaw_proxy': float,
                'pitch_proxy': float
            }
        """
        if face_bgr is None or face_bgr.size == 0 or quality_score < 0.35:
            return {
                'cue': 'UNCERTAIN',
                'cue_label': 'Uncertain (Low Quality / Pose)',
                'oriented_forward': False,
                'quality_score': round(float(quality_score), 2),
                'yaw_proxy': 0.0,
                'pitch_proxy': 0.0
            }

        if landmarks and 'right_eye' in landmarks:
            rx, ry = landmarks['right_eye']
            lx, ly = landmarks['left_eye']
            nx, ny = landmarks['nose']
            rmx, rmy = landmarks['right_mouth']
            lmx, lmy = landmarks['left_mouth']

            eye_dist = math.hypot(lx - rx, ly - ry) + 1e-5
            roll_deg = math.degrees(math.atan2(ly - ry, lx - rx))

            mx_eye = (rx + lx) / 2.0
            my_eye = (ry + ly) / 2.0
            my_mouth = (rmy + lmy) / 2.0

            yaw_ratio = abs(nx - mx_eye) / eye_dist
            mouth_eye_span = max(1.0, my_mouth - my_eye)
            pitch_ratio = (ny - my_eye) / mouth_eye_span

            # Objective Reference Direction Thresholds:
            # Face oriented toward camera reference direction if yaw offset < 0.32, pitch ratio within normal bounds, roll < 20 deg
            if yaw_ratio <= 0.32 and 0.20 <= pitch_ratio <= 0.78 and abs(roll_deg) <= 20.0:
                cue = 'FACE_VISIBLE_FORWARD'
                cue_label = 'Face Visible (Oriented Forward)'
                oriented_forward = True
            elif yaw_ratio > 0.45 or pitch_ratio > 0.85 or pitch_ratio < 0.15 or abs(roll_deg) > 30.0:
                cue = 'FACE_VISIBLE_ELSEWHERE'
                cue_label = 'Face Visible (Oriented Elsewhere)'
                oriented_forward = False
            else:
                cue = 'UNCERTAIN'
                cue_label = 'Uncertain Orientation'
                oriented_forward = False

            return {
                'cue': cue,
                'cue_label': cue_label,
                'oriented_forward': oriented_forward,
                'quality_score': round(float(quality_score), 2),
                'yaw_proxy': round(float(yaw_ratio * 100), 1),
                'pitch_proxy': round(float(pitch_ratio * 100), 1)
            }

        return {
            'cue': 'FACE_VISIBLE_FORWARD',
            'cue_label': 'Face Visible',
            'oriented_forward': True,
            'quality_score': round(float(quality_score), 2),
            'yaw_proxy': 0.0,
            'pitch_proxy': 0.0
        }
