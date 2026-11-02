import numpy as np
import time

class BiometricFaceTracker:
    def __init__(self, alpha=0.20, iou_threshold=0.35, max_missing_frames=25):
        self.alpha = alpha  # EMA smoothing factor
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.tracks = {}  # track_id -> track_data dict
        self.next_track_id = 1

    @staticmethod
    def _compute_iou(boxA, boxB):
        """Calculates Intersection over Union (IoU) between two bounding boxes (x, y, w, h)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, detected_faces, recognizer_func=None):
        """
        Updates multi-person tracks:
        1. Assigns temporary track ID (Track #1, Track #2).
        2. Requires 3 consistent multi-observation evidence matches before confirming student identity.
        3. Revalidates identity when tracks cross (high IoU overlap).
        4. Applies EMA smoothing & 4px deadband filter for stable bounding boxes.
        """
        updated_tracks = {}
        unmatched_detections = list(range(len(detected_faces)))

        # 1. Match new detections against existing active tracks using IoU
        matched_pairs = []
        for track_id, track_data in list(self.tracks.items()):
            old_box = track_data['bbox']
            best_iou = -1.0
            best_det_idx = -1

            for det_idx in unmatched_detections:
                det_box = detected_faces[det_idx]['bbox']
                iou = self._compute_iou(old_box, det_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx != -1:
                matched_pairs.append((track_id, best_det_idx))
                unmatched_detections.remove(best_det_idx)
            else:
                track_data['missing_count'] += 1
                if track_data['missing_count'] <= self.max_missing_frames:
                    updated_tracks[track_id] = track_data

        # Check for crossing tracks (high IoU overlap between active tracks)
        track_boxes = [t['bbox'] for t in updated_tracks.values()]
        is_crossing_event = False
        for i in range(len(track_boxes)):
            for j in range(i + 1, len(track_boxes)):
                if self._compute_iou(track_boxes[i], track_boxes[j]) > 0.20:
                    is_crossing_event = True
                    break

        # 2. Update matched tracks with EMA coordinate smoothing and multi-observation evidence buffer
        for track_id, det_idx in matched_pairs:
            det = detected_faces[det_idx]
            raw_box = det['bbox']
            track_data = self.tracks[track_id]
            old_box = track_data['bbox']

            # EMA Coordinate Smoothing
            sx = int(self.alpha * raw_box[0] + (1.0 - self.alpha) * old_box[0])
            sy = int(self.alpha * raw_box[1] + (1.0 - self.alpha) * old_box[1])
            sw = int(self.alpha * raw_box[2] + (1.0 - self.alpha) * old_box[2])
            sh = int(self.alpha * raw_box[3] + (1.0 - self.alpha) * old_box[3])

            # Deadband filter (< 4px micro jitter ignored)
            if abs(sx - old_box[0]) < 4: sx = old_box[0]
            if abs(sy - old_box[1]) < 4: sy = old_box[1]
            if abs(sw - old_box[2]) < 4: sw = old_box[2]
            if abs(sh - old_box[3]) < 4: sh = old_box[3]

            track_data['bbox'] = (sx, sy, sw, sh)
            track_data['roi_bgr'] = det['roi_bgr']
            track_data['roi_gray'] = det['roi_gray']
            track_data['landmarks'] = det.get('landmarks')
            track_data['missing_count'] = 0

            # Identity Multi-Observation Confirmation Buffer
            # Revalidate if crossing tracks occurred or identity not yet confirmed
            needs_revalidation = is_crossing_event or (track_data['confirmation_count'] < 3)
            
            if recognizer_func and needs_revalidation:
                st_id, conf, match_status = recognizer_func(det['roi_bgr'])
                
                if match_status == 'CONFIRMED' and st_id:
                    if track_data['candidate_id'] == st_id:
                        track_data['confirmation_count'] += 1
                    else:
                        track_data['candidate_id'] = st_id
                        track_data['confirmation_count'] = 1
                    
                    track_data['identity_confidence'] = conf

                    # Require 3 consistent observations to lock confirmed student ID
                    if track_data['confirmation_count'] >= 3:
                        track_data['student_id'] = st_id
                        track_data['match_status'] = 'CONFIRMED'
                elif match_status == 'AMBIGUOUS':
                    track_data['match_status'] = 'AMBIGUOUS'
                    track_data['confirmation_count'] = max(0, track_data['confirmation_count'] - 1)
                else:
                    # Unknown / Unregistered
                    track_data['candidate_id'] = None
                    track_data['confirmation_count'] = 0
                    track_data['match_status'] = 'UNKNOWN'
                    track_data['student_id'] = None

            updated_tracks[track_id] = track_data

        # 3. Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detected_faces[det_idx]
            st_id = None
            conf = 0.0
            match_status = "UNKNOWN"

            if recognizer_func:
                st_id, conf, match_status = recognizer_func(det['roi_bgr'])

            track_id = self.next_track_id
            self.next_track_id += 1

            updated_tracks[track_id] = {
                'track_id': track_id,
                'bbox': det['bbox'],
                'roi_bgr': det['roi_bgr'],
                'roi_gray': det['roi_gray'],
                'landmarks': det.get('landmarks'),
                'candidate_id': st_id if match_status == 'CONFIRMED' else None,
                'student_id': None, # Unconfirmed initially; requires 3 consistent frames
                'identity_confidence': conf,
                'match_status': match_status,
                'confirmation_count': 1 if match_status == 'CONFIRMED' else 0,
                'missing_count': 0
            }

        self.tracks = updated_tracks

        # 4. Enforce 1-to-1 Uniqueness Constraint per Frame
        # A registered student ID cannot be assigned to multiple faces in the same frame.
        # Higher confidence tracks retain identity; duplicates are demoted to AMBIGUOUS.
        assigned_students = set()
        active_results = []
        
        # Process tracks ordered by highest confidence first
        sorted_track_ids = sorted(self.tracks.keys(), key=lambda tid: self.tracks[tid].get('identity_confidence', 0.0), reverse=True)
        for tid in sorted_track_ids:
            tdata = self.tracks[tid]
            if tdata['missing_count'] == 0:
                sid = tdata.get('student_id')
                if sid:
                    if sid in assigned_students:
                        # Duplicate student ID detected in same frame -> demote to AMBIGUOUS
                        tdata['student_id'] = None
                        tdata['match_status'] = 'AMBIGUOUS'
                    else:
                        assigned_students.add(sid)
                active_results.append(tdata)

        return active_results

