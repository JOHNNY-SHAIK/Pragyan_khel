import cv2
import numpy as np
from ultralytics import YOLO
import threading
import logging
import os
from collections import deque

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalmanTracker:
    """Kalman Filter for predicting object position during fast motion"""
    
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.frames_since_seen = 0
        self.hit_streak = 0
        
    def predict(self):
        pred = self.kf.predict()
        cx, cy = pred[0, 0], pred[1, 0]
        x1 = int(cx - self.width / 2)
        y1 = int(cy - self.height / 2)
        x2 = int(cx + self.width / 2)
        y2 = int(cy + self.height / 2)
        return (x1, y1, x2, y2)
    
    def update(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.kf.correct(np.array([[cx], [cy]], np.float32))
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.frames_since_seen = 0
        self.hit_streak += 1


class StreamingDetector:
    def __init__(self, model_size='n'):
        logger.info(f"Initializing ULTRA-FAST Detector with YOLOv8-{model_size}...")
        
        self.yolo = YOLO(f'yolov8{model_size}.pt')
        
        self.cricket_classes = {
            0: 'player',
            32: 'ball',
            35: 'bat',
            56: 'chair',
            39: 'bottle'
        }
        
        self.focused_track_id = None
        self.blur_intensity = 35
        
        self.video_source = None
        self.cap = None
        self.current_frame = None
        self.processed_frame = None
        self.is_playing = False
        self.frame_count = 0
        
        # Advanced tracking with Kalman
        self.trackers = {}
        self.track_history = {}
        self.track_appearances = {}
        self.next_track_id = 0
        
        # FIXED: Much higher tolerance for missing frames
        self.focused_frames_missing = 0
        self.max_missing_frames = 150  # ~6 seconds at 25fps
        
        # Store last known good bbox for focused object
        self.last_focused_bbox = None
        
        # PERFORMANCE: Cached mask
        self.cached_mask = None
        self.cached_mask_bbox = None
        
        # Low-light enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.low_light_threshold = 60
        
        self.lock = threading.Lock()
        
        # Frame skip for speed
        self.process_every_n_frames = 2
        self.skip_counter = 0
        self.cached_detections = []
        
        logger.info("✅ ULTRA-FAST detector initialized with PERSISTENT FOCUS!")

    def load_video(self, video_path):
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.video_source = video_path
        self.is_playing = False
        self.frame_count = 0
        
        # Reset tracking
        self.trackers = {}
        self.track_history = {}
        self.track_appearances = {}
        self.next_track_id = 0
        self.focused_track_id = None
        self.last_focused_bbox = None
        self.cached_mask = None
        self.cached_detections = []
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Loaded: {fps} FPS, {total_frames} frames")
        return fps, total_frames

    def start_processing(self):
        self.is_playing = True
        self.frame_count = 0
        logger.info("▶️ Processing started")

    def stop_processing(self):
        self.is_playing = False
        logger.info("⏸️ Processing stopped")

    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def _extract_appearance(self, frame, bbox):
        """Extract color histogram for re-identification"""
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x2 = min(x2, frame.shape[1])
        y2 = min(y2, frame.shape[0])
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _compare_appearance(self, hist1, hist2):
        """Compare two appearance histograms"""
        if hist1 is None or hist2 is None:
            return 0
        return cv2.compareHist(hist1.reshape(8, 8).astype(np.float32), 
                               hist2.reshape(8, 8).astype(np.float32), 
                               cv2.HISTCMP_CORREL)

    def _assign_tracks(self, detections, frame):
        """Advanced tracking with Kalman + Re-ID"""
        # Predict positions for all existing trackers
        predictions = {}
        for track_id, tracker in self.trackers.items():
            predictions[track_id] = tracker.predict()
            tracker.frames_since_seen += 1
        
        matched_tracks = set()
        
        # Match detections to existing tracks
        for detection in detections:
            bbox = detection['bbox']
            best_match_id = None
            best_score = 0.15  # LOWERED threshold for better matching
            
            for track_id, pred_bbox in predictions.items():
                if track_id in matched_tracks:
                    continue
                    
                iou = self._calculate_iou(bbox, pred_bbox)
                
                # Combine IoU with appearance similarity
                appearance_score = 0
                if track_id in self.track_appearances:
                    new_appearance = self._extract_appearance(frame, bbox)
                    appearance_score = self._compare_appearance(
                        self.track_appearances.get(track_id), new_appearance
                    ) * 0.3
                
                total_score = iou + appearance_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_match_id = track_id
            
            if best_match_id is not None:
                detection['track_id'] = best_match_id
                self.trackers[best_match_id].update(bbox)
                self.track_history[best_match_id] = bbox
                
                # Update appearance
                new_appearance = self._extract_appearance(frame, bbox)
                if new_appearance is not None:
                    self.track_appearances[best_match_id] = new_appearance
                
                matched_tracks.add(best_match_id)
                
                # Update last known bbox for focused object
                if best_match_id == self.focused_track_id:
                    self.last_focused_bbox = bbox
            else:
                # New track
                new_id = self.next_track_id
                self.next_track_id += 1
                detection['track_id'] = new_id
                self.trackers[new_id] = KalmanTracker(bbox)
                self.track_history[new_id] = bbox
                self.track_appearances[new_id] = self._extract_appearance(frame, bbox)
        
        # Remove very old trackers (but not the focused one!)
        tracks_to_remove = []
        for track_id, tracker in self.trackers.items():
            if track_id != self.focused_track_id and tracker.frames_since_seen > self.max_missing_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.trackers[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
            if track_id in self.track_appearances:
                del self.track_appearances[track_id]

    def _enhance_low_light(self, frame):
        """Apply CLAHE for low-light enhancement"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness < self.low_light_threshold:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        return frame

    def detect_and_track(self, frame):
        # Apply low-light enhancement if needed
        enhanced_frame = self._enhance_low_light(frame)
        
        results = self.yolo.predict(
            enhanced_frame,
            conf=0.20,  # Lower confidence for better detection
            iou=0.5,
            imgsz=416,
            classes=list(self.cricket_classes.keys()),
            verbose=False,
            device='cpu'
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detection = {
                    'class_id': class_id,
                    'class_name': self.cricket_classes.get(class_id, 'unknown'),
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'track_id': None
                }
                detections.append(detection)
        
        self._assign_tracks(detections, frame)
        
        # IMPROVED FOCUS PERSISTENCE
        if self.focused_track_id is not None:
            focused_found = any(d['track_id'] == self.focused_track_id for d in detections)
            
            if not focused_found:
                self.focused_frames_missing += 1
                
                # USE KALMAN PREDICTION to keep tracking!
                if self.focused_track_id in self.trackers:
                    predicted_bbox = self.trackers[self.focused_track_id].predict()
                    
                    # Validate prediction is within frame bounds
                    h, w = frame.shape[:2]
                    px1, py1, px2, py2 = predicted_bbox
                    if 0 <= px1 < w and 0 <= py1 < h and px2 > 0 and py2 > 0:
                        ghost_detection = {
                            'class_id': 0,
                            'class_name': 'player (predicted)',
                            'confidence': 0.5,
                            'bbox': predicted_bbox,
                            'center': ((px1 + px2) // 2, (py1 + py2) // 2),
                            'track_id': self.focused_track_id
                        }
                        detections.append(ghost_detection)
                        self.last_focused_bbox = predicted_bbox
                
                # OR use last known bbox
                elif self.last_focused_bbox is not None:
                    ghost_detection = {
                        'class_id': 0,
                        'class_name': 'player (last known)',
                        'confidence': 0.3,
                        'bbox': self.last_focused_bbox,
                        'center': ((self.last_focused_bbox[0] + self.last_focused_bbox[2]) // 2,
                                  (self.last_focused_bbox[1] + self.last_focused_bbox[3]) // 2),
                        'track_id': self.focused_track_id
                    }
                    detections.append(ghost_detection)
                
                # Only clear after VERY long time
                if self.focused_frames_missing > self.max_missing_frames:
                    logger.info(f"Auto-clearing focus after {self.focused_frames_missing} frames")
                    self.focused_track_id = None
                    self.focused_frames_missing = 0
                    self.last_focused_bbox = None
                    self.cached_mask = None
            else:
                self.focused_frames_missing = 0
        
        return detections

    def apply_fast_blur(self, frame, detections):
        """ULTRA-FAST ROI-based blur."""
        focused_detection = None
        if self.focused_track_id is not None:
            for det in detections:
                if det['track_id'] == self.focused_track_id:
                    focused_detection = det
                    break
        
        # Fallback to last known bbox
        if focused_detection is None and self.last_focused_bbox is not None:
            focused_detection = {
                'bbox': self.last_focused_bbox,
                'track_id': self.focused_track_id
            }
        
        if focused_detection is None:
            return frame

        x1, y1, x2, y2 = focused_detection['bbox']
        h, w = frame.shape[:2]

        # Validate bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return frame

        pad = 50
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)

        # FAST BLUR at 1/4 resolution
        small = cv2.resize(frame, (w // 4, h // 4))
        k = self.blur_intensity if self.blur_intensity % 2 != 0 else self.blur_intensity + 1
        blurred_small = cv2.GaussianBlur(small, (15, 15), 0)
        background_blurred = cv2.resize(blurred_small, (w, h))

        roi_sharp = frame[ry1:ry2, rx1:rx2].copy()
        roi_h, roi_w = roi_sharp.shape[:2]
        
        if roi_h <= 0 or roi_w <= 0:
            return background_blurred
            
        local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        center_x, center_y = roi_w//2, roi_h//2
        axes_x, axes_y = max(1, roi_w//2-5), max(1, roi_h//2-5)
        
        cv2.ellipse(local_mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
        local_mask = cv2.GaussianBlur(local_mask, (21, 21), 0).astype(np.float32) / 255.0
        local_mask = local_mask[:, :, np.newaxis]

        blended_roi = (roi_sharp * local_mask + 
                       background_blurred[ry1:ry2, rx1:rx2] * (1 - local_mask)).astype(np.uint8)

        background_blurred[ry1:ry2, rx1:rx2] = blended_roi

        return background_blurred

    def draw_annotations(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection['track_id']
            
            if track_id == self.focused_track_id:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (0, 255, 255)
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{detection['class_name']} #{track_id}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Status display
        if self.focused_track_id is not None:
            status_text = f"FOCUSED: #{self.focused_track_id}"
            if self.focused_frames_missing > 0:
                status_text += f" (predicting...)"
            status_color = (0, 255, 0)
        else:
            status_text = "Click on object to focus"
            status_color = (255, 255, 255)
            
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame

    def get_next_frame(self):
        if not self.cap or not self.is_playing:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Click PLAY to start", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return blank
        
        success, frame = self.cap.read()
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            success, frame = self.cap.read()
            if not success:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Video ended", (200, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return blank
        
        self.frame_count += 1
        
        # Frame skip for detection
        self.skip_counter += 1
        if self.skip_counter >= self.process_every_n_frames:
            self.skip_counter = 0
            detections = self.detect_and_track(frame)
            with self.lock:
                self.cached_detections = detections
        else:
            with self.lock:
                detections = self.cached_detections.copy() if self.cached_detections else []
        
        # Apply blur
        if self.focused_track_id is not None:
            frame = self.apply_fast_blur(frame, detections)
        
        # Draw annotations
        if detections:
            frame = self.draw_annotations(frame, detections)
        
        with self.lock:
            self.current_frame = frame
            self.processed_frame = frame
        
        return frame

    def set_focus_by_click(self, x, y):
        """Click to focus on object"""
        with self.lock:
            detections = self.cached_detections.copy() if self.cached_detections else []
        
        if not detections:
            logger.warning("No detections available")
            return None
        
        # Sort by area (smallest first)
        sorted_dets = sorted(detections,
                            key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))
        
        for det in sorted_dets:
            x1, y1, x2, y2 = det['bbox']
            padding = 10
            if (x1 - padding) <= x <= (x2 + padding) and (y1 - padding) <= y <= (y2 + padding):
                self.focused_track_id = det['track_id']
                self.focused_frames_missing = 0
                self.last_focused_bbox = det['bbox']
                self.cached_mask = None
                logger.info(f"✓ FOCUS LOCKED: {det['class_name']} Track #{det['track_id']}")
                return det
        
        logger.warning(f"No object at click ({x}, {y})")
        return None

    def clear_focus(self):
        self.focused_track_id = None
        self.focused_frames_missing = 0
        self.last_focused_bbox = None
        self.cached_mask = None
        logger.info("✓ Focus cleared")

    def set_blur_intensity(self, value):
        self.blur_intensity = max(11, min(51, value))
        if self.blur_intensity % 2 == 0:
            self.blur_intensity += 1
        logger.info(f"✓ Blur: {self.blur_intensity}")


_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = StreamingDetector(model_size='n')
    return _detector