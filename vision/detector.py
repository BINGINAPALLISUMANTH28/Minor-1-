import cv2
import numpy as np
import os
import urllib.request

class FaceDetector:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yunet.onnx")
        self.yunet = None
        self._load_yunet_model()

    def _load_yunet_model(self):
        """Loads OpenCV YuNet Deep Learning AI Face Detector."""
        if not os.path.exists(self.model_path):
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                urllib.request.urlretrieve(url, self.model_path)
            except Exception as e:
                print(f"Warning: Failed to download YuNet model: {e}")

        if os.path.exists(self.model_path):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    self.model_path,
                    "",
                    (640, 480),
                    score_threshold=0.50, # High score threshold for true facial features
                    nms_threshold=0.3,
                    top_k=5000
                )
            except Exception as e:
                print(f"Warning: Failed to initialize YuNet FaceDetectorYN: {e}")
                self.yunet = None

    def detect_faces(self, frame):
        """
        Detects faces strictly using facial feature landmarks (Eyes, Nose, Mouth).
        Bounding boxes are mathematically anchored directly to facial feature positions.
        Skin color masking is completely removed to prevent box sliding onto neck/chest.
        """
        h_frame, w_frame, _ = frame.shape
        detected_faces = []

        if self.yunet is None:
            return []

        try:
            self.yunet.setInputSize((w_frame, h_frame))
            _, faces = self.yunet.detect(frame)
            
            if faces is not None and len(faces) > 0:
                for face in faces:
                    score = float(face[14])
                    if score < 0.50:
                        continue

                    # 5 Facial Landmark Coordinates
                    r_eye = (float(face[4]), float(face[5]))
                    l_eye = (float(face[6]), float(face[7]))
                    nose = (float(face[8]), float(face[9]))
                    r_mouth = (float(face[10]), float(face[11]))
                    l_mouth = (float(face[12]), float(face[13]))

                    # Calculate box boundaries strictly anchored to Facial Feature Landmarks
                    eye_dx = abs(l_eye[0] - r_eye[0])
                    eye_dy = abs(l_eye[1] - r_eye[1])
                    eye_span = max(20.0, np.hypot(eye_dx, eye_dy))

                    min_x = min(r_eye[0], l_eye[0], nose[0], r_mouth[0], l_mouth[0])
                    max_x = max(r_eye[0], l_eye[0], nose[0], r_mouth[0], l_mouth[0])
                    min_y = min(r_eye[1], l_eye[1])
                    max_y = max(r_mouth[1], l_mouth[1])

                    # Tight facial boundaries around forehead, cheeks, and chin
                    margin_x = int(eye_span * 0.65)
                    top_margin = int(eye_span * 0.85)
                    bottom_margin = int(eye_span * 0.75)

                    x1 = max(0, int(min_x - margin_x))
                    y1 = max(0, int(min_y - top_margin))
                    x2 = min(w_frame, int(max_x + margin_x))
                    y2 = min(h_frame, int(max_y + bottom_margin))

                    w_box = x2 - x1
                    h_box = y2 - y1

                    if w_box < 30 or h_box < 30:
                        continue

                    roi_bgr = frame[y1:y2, x1:x2]
                    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.size > 0 else None

                    landmarks = {
                        'right_eye': r_eye,
                        'left_eye': l_eye,
                        'nose': nose,
                        'right_mouth': r_mouth,
                        'left_mouth': l_mouth,
                        'score': score
                    }

                    detected_faces.append({
                        'bbox': (x1, y1, w_box, h_box),
                        'landmarks': landmarks,
                        'frame_bgr': frame,
                        'face_vector': face.copy(),
                        'roi_bgr': roi_bgr,
                        'roi_gray': roi_gray
                    })
        except Exception as e:
            print(f"Detection exception: {e}")
            pass

        return detected_faces
