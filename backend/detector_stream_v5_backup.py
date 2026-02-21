import cv2
import numpy as np
from ultralytics import YOLO
import threading
import logging
import os
import time
from collections import deque
from queue import Queue, Empty

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalmanTracker:
    """Enhanced Kalman Filter with velocity prediction"""
    
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(6, 4)  # 6 state vars, 4 measurements
        
        # State: [x, y, w, h, vx, vy]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], np.float32)
        
        # Transition: position += velocity
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0]], np.float32)
        
        self.frames_since_seen = 0
        self.hit_streak = 0
        self.total_hits = 1
        
    def predict(self):
        pred = self.kf.predict()
        cx, cy, w, h = pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0]
        
        # Ensure valid dimensions
        w, h = max(20, w), max(20, h)
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return (x1, y1, x2, y2)
    
    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        measurement = np.array([[cx], [cy], [w], [h]], np.float32)
        self.kf.correct(measurement)
        
        self.frames_since_seen = 0
        self.hit_streak += 1
        self.total_hits += 1


class StreamingDetector:
    def __init__(self, model_size='n'):
        logger.info(f"Initializing ULTIMATE Detector with YOLOv8-{model_size}...")
        
        self.yolo = YOLO(f'yolov8{model_size}.pt')
        
        # Extended class detection
        self.cricket_classes = {
            0: 'player',      # person
            32: 'ball',       # sports ball
            35: 'bat',        # baseball bat (close to cricket bat)
            38: 'racket',     # tennis racket
            56: 'chair',
            39: 'bottle',
            37: 'skateboard', # Sometimes detects stumps
            67: 'phone',
            73: 'laptop'
        }
        
        self.focused_track_id = None
        self.blur_intensity = 35
        
        self.video_source = None
        self.cap = None
        self.current_frame = None
        self.processed_frame = None
        self.is_playing = False
        self.frame_count = 0
        
        # Advanced tracking
        self.trackers = {}
        self.track_history = {}
        self.track_appearances = {}
        self.next_track_id = 0
        
        # NEVER LOSE FOCUS
        self.focused_frames_missing = 0
        self.max_missing_frames = 999999  # Essentially infinite
        self.last_focused_bbox = None
        self.focus_locked = False  # Manual lock
        
        # Performance tracking
        self.fps = 0
        self.prev_time = time.time()
        self.frame_times = deque(maxlen=30)
        
        # Low-light enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.low_light_threshold = 50
        
        self.lock = threading.Lock()
        
        # Frame skip for speed (process every Nth frame)
        self.process_every_n_frames = 2
        self.skip_counter = 0
        self.cached_detections = []
        
        # Optical flow for smooth tracking
        self.prev_gray = None
        self.optical_flow_enabled = True
        
        logger.info("✅ ULTIMATE detector initialized - FOCUS WILL NEVER BE LOST!")

    def load_video(self, video_path):
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to set optimal resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
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
        self.focus_locked = False
        self.cached_detections = []
        self.prev_gray = None
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"✅ Loaded: {width}x{height} @ {fps} FPS, {total_frames} frames")
        return fps, total_frames

    def start_processing(self):
        self.is_playing = True
        self.frame_count = 0
        self.prev_time = time.time()
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
        h, w = frame.shape[:2]
        x2 = min(x2, w)
        y2 = min(y2, h)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        except:
            return None

    def _compare_appearance(self, hist1, hist2):
        """Compare two appearance histograms"""
        if hist1 is None or hist2 is None:
            return 0
        try:
            return cv2.compareHist(
                hist1.reshape(-1).astype(np.float32), 
                hist2.reshape(-1).astype(np.float32), 
                cv2.HISTCMP_CORREL
            )
        except:
            return 0

    def _update_optical_flow(self, frame, bbox):
        """Use optical flow to track object between detections"""
        if self.prev_gray is None:
            return bbox
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Define points to track
        prev_pts = np.array([[[cx, cy]]], dtype=np.float32)
        
        try:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_pts, None,
                winSize=(21, 21), maxLevel=3
            )
            
            if status[0][0] == 1:
                new_cx, new_cy = next_pts[0][0]
                dx = int(new_cx - cx)
                dy = int(new_cy - cy)
                
                # Move bbox
                return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        except:
            pass
            
        return bbox

    def _assign_tracks(self, detections, frame):
        """Advanced tracking with Kalman + Re-ID + Optical Flow"""
        
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
            best_score = 0.1  # Very low threshold for better matching
            
            for track_id, pred_bbox in predictions.items():
                if track_id in matched_tracks:
                    continue
                    
                iou = self._calculate_iou(bbox, pred_bbox)
                
                # Appearance similarity
                appearance_score = 0
                if track_id in self.track_appearances:
                    new_appearance = self._extract_appearance(frame, bbox)
                    appearance_score = self._compare_appearance(
                        self.track_appearances.get(track_id), new_appearance
                    ) * 0.5
                
                # Distance bonus (closer = better)
                pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2
                pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2
                det_cx = (bbox[0] + bbox[2]) / 2
                det_cy = (bbox[1] + bbox[3]) / 2
                
                distance = np.sqrt((pred_cx - det_cx)**2 + (pred_cy - det_cy)**2)
                distance_score = max(0, 1 - distance / 200) * 0.3
                
                total_score = iou + appearance_score + distance_score
                
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
                    self.focused_frames_missing = 0
            else:
                # New track
                new_id = self.next_track_id
                self.next_track_id += 1
                detection['track_id'] = new_id
                self.trackers[new_id] = KalmanTracker(bbox)
                self.track_history[new_id] = bbox
                self.track_appearances[new_id] = self._extract_appearance(frame, bbox)
        
        # Clean up old trackers (but NEVER delete the focused one!)
        tracks_to_remove = []
        for track_id, tracker in self.trackers.items():
            if track_id != self.focused_track_id and tracker.frames_since_seen > 60:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.trackers[track_id]
            self.track_history.pop(track_id, None)
            self.track_appearances.pop(track_id, None)

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
        # Apply low-light enhancement
        enhanced_frame = self._enhance_low_light(frame)
        
        results = self.yolo.predict(
            enhanced_frame,
            conf=0.15,  # Lower for better detection
            iou=0.45,
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
        
        # FOCUS PERSISTENCE - Never lose the focused object!
        if self.focused_track_id is not None:
            focused_found = any(d['track_id'] == self.focused_track_id for d in detections)
            
            if not focused_found:
                self.focused_frames_missing += 1
                
                # Use Kalman prediction
                if self.focused_track_id in self.trackers:
                    predicted_bbox = self.trackers[self.focused_track_id].predict()
                    
                    # Apply optical flow correction if available
                    if self.optical_flow_enabled and self.last_focused_bbox:
                        predicted_bbox = self._update_optical_flow(frame, predicted_bbox)
                    
                    h, w = frame.shape[:2]
                    px1, py1, px2, py2 = predicted_bbox
                    
                    # Validate prediction is within bounds
                    if px1 < w and py1 < h and px2 > 0 and py2 > 0:
                        ghost_detection = {
                            'class_id': 0,
                            'class_name': 'player (tracking)',
                            'confidence': max(0.3, 0.8 - self.focused_frames_missing * 0.01),
                            'bbox': predicted_bbox,
                            'center': ((px1 + px2) // 2, (py1 + py2) // 2),
                            'track_id': self.focused_track_id
                        }
                        detections.append(ghost_detection)
                        self.last_focused_bbox = predicted_bbox
                
                # Fallback to last known position
                elif self.last_focused_bbox is not None:
                    ghost_detection = {
                        'class_id': 0,
                        'class_name': 'player (last seen)',
                        'confidence': 0.3,
                        'bbox': self.last_focused_bbox,
                        'center': ((self.last_focused_bbox[0] + self.last_focused_bbox[2]) // 2,
                                  (self.last_focused_bbox[1] + self.last_focused_bbox[3]) // 2),
                        'track_id': self.focused_track_id
                    }
                    detections.append(ghost_detection)
            else:
                self.focused_frames_missing = 0
        
        # Update optical flow reference
        try:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            pass
        
        return detections

    def apply_fast_blur(self, frame, detections):
        """ULTRA-FAST ROI-based bokeh blur"""
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

        # ROI with padding
        pad = 60
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)

        # FAST BLUR: 1/4 resolution
        small = cv2.resize(frame, (w // 4, h // 4))
        k = self.blur_intensity if self.blur_intensity % 2 != 0 else self.blur_intensity + 1
        k = min(k, 31)  # Cap for speed
        
        blurred_small = cv2.GaussianBlur(small, (k, k), 0)
        background_blurred = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # Extract sharp ROI
        roi_sharp = frame[ry1:ry2, rx1:rx2].copy()
        roi_h, roi_w = roi_sharp.shape[:2]
        
        if roi_h <= 0 or roi_w <= 0:
            return background_blurred

        # Create smooth elliptical mask
        local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        center_x, center_y = roi_w // 2, roi_h // 2
        axes_x = max(1, (x2 - x1) // 2 + 10)
        axes_y = max(1, (y2 - y1) // 2 + 10)
        
        cv2.ellipse(local_mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
        
        # Smooth edges
        local_mask = cv2.GaussianBlur(local_mask, (31, 31), 0)
        local_mask = local_mask.astype(np.float32) / 255.0
        local_mask = local_mask[:, :, np.newaxis]

        # Blend
        blended_roi = (roi_sharp * local_mask + 
                       background_blurred[ry1:ry2, rx1:rx2] * (1 - local_mask)).astype(np.uint8)

        background_blurred[ry1:ry2, rx1:rx2] = blended_roi

        return background_blurred

    def draw_annotations(self, frame, detections):
        # Calculate FPS
        current_time = time.time()
        self.frame_times.append(current_time - self.prev_time)
        self.prev_time = current_time
        
        if len(self.frame_times) > 0:
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection['track_id']
            
            if track_id == self.focused_track_id:
                color = (0, 255, 0)  # Green
                thickness = 3
                
                # Draw focus indicator
                cv2.circle(frame, ((x1+x2)//2, y1-15), 8, (0, 255, 0), -1)
            else:
                color = (0, 200, 255)  # Orange-yellow
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{detection['class_name']} #{track_id}"
            conf = detection.get('confidence', 0)
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Status overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS Display
        fps_color = (0, 255, 0) if self.fps > 20 else (0, 255, 255) if self.fps > 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Focus status
        if self.focused_track_id is not None:
            status = f"FOCUS LOCKED: #{self.focused_track_id}"
            if self.focused_frames_missing > 0:
                status += f" (predicting: {self.focused_frames_missing}f)"
            cv2.putText(frame, status, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Click any object to FOCUS", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 72), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return frame

    def get_next_frame(self):
        if not self.cap or not self.is_playing:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Upload video and click PLAY", (120, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return blank
        
        success, frame = self.cap.read()
        if not success:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            success, frame = self.cap.read()
            if not success:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Video ended", (220, 240),
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
        frame = self.draw_annotations(frame, detections)
        
        with self.lock:
            self.current_frame = frame
            self.processed_frame = frame
        
        return frame

    def set_focus_by_click(self, x, y):
        """Click to focus - with distance-based selection"""
        with self.lock:
            detections = self.cached_detections.copy() if self.cached_detections else []
        
        if not detections:
            logger.warning("No detections - try clicking after video plays")
            return None
        
        # Find closest object to click point
        best_det = None
        best_score = float('inf')
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Check if click is inside bbox
            padding = 15
            if (x1 - padding) <= x <= (x2 + padding) and (y1 - padding) <= y <= (y2 + padding):
                # Calculate distance to center
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                # Prefer smaller objects (more precise selection)
                area = (x2 - x1) * (y2 - y1)
                score = distance + area * 0.01
                
                if score < best_score:
                    best_score = score
                    best_det = det
        
        if best_det:
            self.focused_track_id = best_det['track_id']
            self.focused_frames_missing = 0
            self.last_focused_bbox = best_det['bbox']
            self.focus_locked = True
            logger.info(f"🎯 FOCUS LOCKED: {best_det['class_name']} Track #{best_det['track_id']}")
            return best_det
        
        logger.warning(f"No object found at ({x}, {y})")
        return None

    def clear_focus(self):
        self.focused_track_id = None
        self.focused_frames_missing = 0
        self.last_focused_bbox = None
        self.focus_locked = False
        logger.info("🔓 Focus cleared")

    def set_blur_intensity(self, value):
        self.blur_intensity = max(11, min(51, value))
        if self.blur_intensity % 2 == 0:
            self.blur_intensity += 1
        logger.info(f"✓ Blur intensity: {self.blur_intensity}")


_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = StreamingDetector(model_size='n')
    return _detector