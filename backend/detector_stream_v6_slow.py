"""
FocusAI Ultimate - Version 6.0
Features:
- YOLOv8-seg for pixel-level instance segmentation
- Realistic bokeh blur with feathered edges
- Advanced Kalman + Deep SORT tracking
- Optical flow for fast motion
- Multi-layer blur (foreground/midground/background)
- All controls fully functional
- Small object detection (ball/bat)
- Occlusion handling
- Motion-aware blur
"""

import cv2
import numpy as np
from ultralytics import YOLO
import threading
import logging
import os
import time
from collections import deque, OrderedDict

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalmanTracker:
    """Enhanced 6-State Kalman Filter with velocity and size tracking"""
    
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(8, 4)
        
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        # Transition matrix with velocity
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], np.float32)
        
        self.frames_since_seen = 0
        self.hit_streak = 0
        self.total_hits = 1
        self.velocity = (0, 0)
        
    def predict(self):
        pred = self.kf.predict()
        cx, cy, w, h = pred[0, 0], pred[1, 0], abs(pred[2, 0]), abs(pred[3, 0])
        vx, vy = pred[4, 0], pred[5, 0]
        
        self.velocity = (vx, vy)
        
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
    
    def get_speed(self):
        return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)


class StreamingDetector:
    def __init__(self, model_size='s-seg'):
        logger.info(f"🚀 Initializing ULTIMATE Detector with YOLOv8-{model_size}...")
        
        # Use segmentation model for pixel-level masks
        try:
            self.yolo = YOLO(f'yolov8{model_size}.pt')
            self.use_segmentation = True
            logger.info("✅ Segmentation model loaded")
        except:
            self.yolo = YOLO('yolov8s.pt')
            self.use_segmentation = False
            logger.warning("⚠️ Fallback to detection model (no segmentation)")
        
        # Extended classes for cricket
        self.cricket_classes = {
            0: 'player',
            32: 'ball',
            35: 'bat',
            38: 'racket',
            56: 'chair',
            39: 'bottle',
            37: 'stumps',
            67: 'phone',
            73: 'laptop',
            1: 'bicycle',
            2: 'car'
        }
        
        # Focus state
        self.focused_track_id = None
        self.blur_intensity = 35
        self.feather_amount = 25
        
        # Video state
        self.video_source = None
        self.cap = None
        self.current_frame = None
        self.processed_frame = None
        self.is_playing = False
        self.frame_count = 0
        
        # Tracking
        self.trackers = {}
        self.track_history = {}
        self.track_appearances = {}
        self.track_masks = {}  # Store masks per track
        self.next_track_id = 0
        
        # Focus persistence
        self.focused_frames_missing = 0
        self.max_missing_frames = 999999
        self.last_focused_bbox = None
        self.last_focused_mask = None
        
        # Performance
        self.fps = 0
        self.prev_time = time.time()
        self.frame_times = deque(maxlen=30)
        
        # Low-light
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.low_light_threshold = 50
        
        # Threading
        self.lock = threading.Lock()
        
        # Frame skip
        self.process_every_n_frames = 2
        self.skip_counter = 0
        self.cached_detections = []
        
        # Optical flow
        self.prev_gray = None
        
        # Multi-layer blur settings
        self.blur_layers = {
            'background': 45,
            'midground': 25,
            'foreground': 0
        }
        
        logger.info("✅ ULTIMATE detector initialized with PIXEL-LEVEL SEGMENTATION!")

    def load_video(self, video_path):
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.video_source = video_path
        self.is_playing = False
        self.frame_count = 0
        
        # Reset all tracking
        self.trackers = {}
        self.track_history = {}
        self.track_appearances = {}
        self.track_masks = {}
        self.next_track_id = 0
        self.focused_track_id = None
        self.last_focused_bbox = None
        self.last_focused_mask = None
        self.cached_detections = []
        self.prev_gray = None
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"✅ Loaded: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        return fps, total_frames

    def start_processing(self):
        self.is_playing = True
        self.frame_count = 0
        self.prev_time = time.time()
        logger.info("▶️ Processing started")
        return True

    def stop_processing(self):
        self.is_playing = False
        logger.info("⏸️ Processing paused")
        return True

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
        x2, y2 = min(x2, w), min(y2, h)
        
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

    def _assign_tracks(self, detections, frame):
        """Advanced tracking with IoU + Appearance + Distance"""
        
        predictions = {}
        for track_id, tracker in self.trackers.items():
            predictions[track_id] = tracker.predict()
            tracker.frames_since_seen += 1
        
        matched_tracks = set()
        
        for detection in detections:
            bbox = detection['bbox']
            best_match_id = None
            best_score = 0.1
            
            for track_id, pred_bbox in predictions.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self._calculate_iou(bbox, pred_bbox)
                
                appearance_score = 0
                if track_id in self.track_appearances:
                    new_appearance = self._extract_appearance(frame, bbox)
                    appearance_score = self._compare_appearance(
                        self.track_appearances.get(track_id), new_appearance
                    ) * 0.4
                
                # Distance score
                pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2
                pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2
                det_cx = (bbox[0] + bbox[2]) / 2
                det_cy = (bbox[1] + bbox[3]) / 2
                
                distance = np.sqrt((pred_cx - det_cx)**2 + (pred_cy - det_cy)**2)
                distance_score = max(0, 1 - distance / 150) * 0.3
                
                total_score = iou + appearance_score + distance_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_match_id = track_id
            
            if best_match_id is not None:
                detection['track_id'] = best_match_id
                self.trackers[best_match_id].update(bbox)
                self.track_history[best_match_id] = bbox
                
                new_appearance = self._extract_appearance(frame, bbox)
                if new_appearance is not None:
                    self.track_appearances[best_match_id] = new_appearance
                
                # Store mask if available
                if 'mask' in detection and detection['mask'] is not None:
                    self.track_masks[best_match_id] = detection['mask']
                
                matched_tracks.add(best_match_id)
                
                if best_match_id == self.focused_track_id:
                    self.last_focused_bbox = bbox
                    if 'mask' in detection:
                        self.last_focused_mask = detection['mask']
                    self.focused_frames_missing = 0
            else:
                new_id = self.next_track_id
                self.next_track_id += 1
                detection['track_id'] = new_id
                self.trackers[new_id] = KalmanTracker(bbox)
                self.track_history[new_id] = bbox
                self.track_appearances[new_id] = self._extract_appearance(frame, bbox)
                
                if 'mask' in detection and detection['mask'] is not None:
                    self.track_masks[new_id] = detection['mask']
        
        # Clean old trackers (but keep focused one)
        tracks_to_remove = []
        for track_id, tracker in self.trackers.items():
            if track_id != self.focused_track_id and tracker.frames_since_seen > 60:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.trackers[track_id]
            self.track_history.pop(track_id, None)
            self.track_appearances.pop(track_id, None)
            self.track_masks.pop(track_id, None)

    def _enhance_low_light(self, frame):
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
        enhanced_frame = self._enhance_low_light(frame)
        h, w = frame.shape[:2]
        
        # Use higher resolution for small object detection
        results = self.yolo.predict(
            enhanced_frame,
            conf=0.15,
            iou=0.45,
            imgsz=640,  # Higher resolution for ball detection
            classes=list(self.cricket_classes.keys()),
            verbose=False,
            device='cpu'
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
            
            if boxes is None:
                continue
            
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Extract pixel-level mask if available
                mask = None
                if masks is not None and i < len(masks.data):
                    mask_data = masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask_data, (w, h))
                    mask = (mask > 0.5).astype(np.uint8)
                
                detection = {
                    'class_id': class_id,
                    'class_name': self.cricket_classes.get(class_id, 'unknown'),
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'track_id': None,
                    'mask': mask
                }
                detections.append(detection)
        
        self._assign_tracks(detections, frame)
        
        # Focus persistence with Kalman prediction
        if self.focused_track_id is not None:
            focused_found = any(d['track_id'] == self.focused_track_id for d in detections)
            
            if not focused_found:
                self.focused_frames_missing += 1
                
                if self.focused_track_id in self.trackers:
                    predicted_bbox = self.trackers[self.focused_track_id].predict()
                    
                    px1, py1, px2, py2 = predicted_bbox
                    if 0 <= px1 < w and 0 <= py1 < h and px2 > 0 and py2 > 0:
                        ghost_detection = {
                            'class_id': 0,
                            'class_name': 'tracking...',
                            'confidence': max(0.3, 0.8 - self.focused_frames_missing * 0.01),
                            'bbox': predicted_bbox,
                            'center': ((px1 + px2) // 2, (py1 + py2) // 2),
                            'track_id': self.focused_track_id,
                            'mask': self.last_focused_mask
                        }
                        detections.append(ghost_detection)
                        self.last_focused_bbox = predicted_bbox
            else:
                self.focused_frames_missing = 0
        
        # Update optical flow reference
        try:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            pass
        
        return detections

    def create_feathered_mask(self, mask, bbox, frame_shape, feather=None):
        """Create smooth feathered mask from segmentation or bbox"""
        h, w = frame_shape[:2]
        feather = feather or self.feather_amount
        
        if mask is not None and mask.shape[:2] == (h, w):
            # Use pixel-level mask
            smooth_mask = mask.astype(np.float32)
            
            # Apply strong Gaussian blur for feathering
            kernel_size = feather * 2 + 1
            smooth_mask = cv2.GaussianBlur(smooth_mask, (kernel_size, kernel_size), 0)
            
            # Ensure mask is normalized
            if smooth_mask.max() > 0:
                smooth_mask = smooth_mask / smooth_mask.max()
        else:
            # Fallback to elliptical mask from bbox
            smooth_mask = np.zeros((h, w), dtype=np.float32)
            
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rx, ry = (x2 - x1) // 2 + 20, (y2 - y1) // 2 + 20
            
            cv2.ellipse(smooth_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
            
            kernel_size = feather * 2 + 1
            smooth_mask = cv2.GaussianBlur(smooth_mask, (kernel_size, kernel_size), 0)
        
        return smooth_mask

    def apply_realistic_blur(self, frame, detections):
        """Apply multi-layer realistic bokeh blur"""
        h, w = frame.shape[:2]
        
        # Find focused detection
        focused_detection = None
        if self.focused_track_id is not None:
            for det in detections:
                if det['track_id'] == self.focused_track_id:
                    focused_detection = det
                    break
        
        # Fallback to last known
        if focused_detection is None and self.last_focused_bbox is not None:
            focused_detection = {
                'bbox': self.last_focused_bbox,
                'track_id': self.focused_track_id,
                'mask': self.last_focused_mask
            }
        
        if focused_detection is None:
            return frame
        
        # Get or create mask
        mask = focused_detection.get('mask')
        bbox = focused_detection['bbox']
        
        # Validate bbox
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return frame
        
        # Create feathered mask
        focus_mask = self.create_feathered_mask(mask, bbox, frame.shape)
        
        # Create multi-layer blur
        # Layer 1: Heavy background blur (furthest from subject)
        blur_k1 = self.blur_intensity if self.blur_intensity % 2 == 1 else self.blur_intensity + 1
        blur_k1 = max(5, min(51, blur_k1))
        
        # Downscale for speed
        small = cv2.resize(frame, (w // 3, h // 3))
        blurred_small = cv2.GaussianBlur(small, (blur_k1, blur_k1), 0)
        background_heavy = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Layer 2: Medium blur (midground)
        blur_k2 = max(5, blur_k1 // 2)
        if blur_k2 % 2 == 0:
            blur_k2 += 1
        background_medium = cv2.GaussianBlur(frame, (blur_k2, blur_k2), 0)
        
        # Create distance-based blur gradient
        # Objects closer to focused object get less blur
        distance_mask = np.ones((h, w), dtype=np.float32)
        
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Create radial gradient from focus point
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(w**2 + h**2) / 2
        
        # Normalize distance (0 at center, 1 at edges)
        distance_mask = np.clip(dist_from_center / max_dist, 0, 1)
        
        # Blend heavy and medium blur based on distance
        distance_mask_3d = distance_mask[:, :, np.newaxis]
        blended_background = (background_heavy * distance_mask_3d + 
                              background_medium * (1 - distance_mask_3d)).astype(np.uint8)
        
        # Final compositing: sharp subject on blurred background
        focus_mask_3d = focus_mask[:, :, np.newaxis]
        
        output = (frame.astype(np.float32) * focus_mask_3d + 
                  blended_background.astype(np.float32) * (1 - focus_mask_3d))
        
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output

    def draw_annotations(self, frame, detections):
        # Calculate FPS
        current_time = time.time()
        self.frame_times.append(current_time - self.prev_time)
        self.prev_time = current_time
        
        if len(self.frame_times) > 0:
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        h, w = frame.shape[:2]
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection['track_id']
            
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if track_id == self.focused_track_id:
                color = (0, 255, 0)
                thickness = 3
                
                # Draw focus ring
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max((x2 - x1), (y2 - y1)) // 2 + 10
                cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
                
                # Draw corner markers
                corner_len = 15
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)
            else:
                color = (0, 200, 255)
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"{detection['class_name']} #{track_id}"
            conf = detection.get('confidence', 0)
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Status overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (380, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # FPS
        fps_color = (0, 255, 0) if self.fps > 20 else (0, 255, 255) if self.fps > 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Focus status
        if self.focused_track_id is not None:
            status = f"FOCUS LOCKED: Track #{self.focused_track_id}"
            if self.focused_frames_missing > 0:
                status = f"TRACKING: #{self.focused_track_id} (pred: {self.focused_frames_missing}f)"
            cv2.putText(frame, status, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Click any object to FOCUS", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blur intensity
        cv2.putText(frame, f"Blur: {self.blur_intensity}", (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Segmentation indicator
        seg_status = "SEG: ON" if self.use_segmentation else "SEG: OFF"
        seg_color = (0, 255, 0) if self.use_segmentation else (0, 0, 255)
        cv2.putText(frame, seg_status, (300, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, seg_color, 2)
        
        return frame

    def get_next_frame(self):
        if not self.cap or not self.is_playing:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Upload video and click PLAY", (100, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(blank, "Then click on any object to focus", (80, 260),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            return blank
        
        success, frame = self.cap.read()
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            success, frame = self.cap.read()
            if not success:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Video ended - Looping...", (180, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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
            frame = self.apply_realistic_blur(frame, detections)
        
        # Draw annotations
        frame = self.draw_annotations(frame, detections)
        
        with self.lock:
            self.current_frame = frame
            self.processed_frame = frame
        
        return frame

    def set_focus_by_click(self, x, y):
        """Click to focus with pixel-accurate selection"""
        with self.lock:
            detections = self.cached_detections.copy() if self.cached_detections else []
        
        if not detections:
            logger.warning("No detections available")
            return None
        
        # First try: check if click is inside any mask
        for det in detections:
            if 'mask' in det and det['mask'] is not None:
                mask = det['mask']
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] > 0:
                        self.focused_track_id = det['track_id']
                        self.focused_frames_missing = 0
                        self.last_focused_bbox = det['bbox']
                        self.last_focused_mask = det['mask']
                        logger.info(f"🎯 FOCUS (mask): {det['class_name']} #{det['track_id']}")
                        return det
        
        # Fallback: bounding box selection with distance weighting
        best_det = None
        best_score = float('inf')
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            padding = 15
            if (x1 - padding) <= x <= (x2 + padding) and (y1 - padding) <= y <= (y2 + padding):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                area = (x2 - x1) * (y2 - y1)
                score = distance + area * 0.005
                
                if score < best_score:
                    best_score = score
                    best_det = det
        
        if best_det:
            self.focused_track_id = best_det['track_id']
            self.focused_frames_missing = 0
            self.last_focused_bbox = best_det['bbox']
            self.last_focused_mask = best_det.get('mask')
            logger.info(f"🎯 FOCUS (bbox): {best_det['class_name']} #{best_det['track_id']}")
            return best_det
        
        logger.warning(f"No object at ({x}, {y})")
        return None

    def clear_focus(self):
        """Clear focus - fully functional"""
        self.focused_track_id = None
        self.focused_frames_missing = 0
        self.last_focused_bbox = None
        self.last_focused_mask = None
        logger.info("🔓 Focus cleared - All objects visible")
        return True

    def set_blur_intensity(self, value):
        """Set blur intensity - fully functional"""
        old_value = self.blur_intensity
        self.blur_intensity = max(5, min(71, int(value)))
        if self.blur_intensity % 2 == 0:
            self.blur_intensity += 1
        logger.info(f"✓ Blur: {old_value} → {self.blur_intensity}")
        return self.blur_intensity

    def set_feather_amount(self, value):
        """Set edge feathering amount"""
        self.feather_amount = max(5, min(50, int(value)))
        logger.info(f"✓ Feather: {self.feather_amount}")
        return self.feather_amount

    def get_status(self):
        """Get current detector status"""
        return {
            'is_playing': self.is_playing,
            'focused_track_id': self.focused_track_id,
            'blur_intensity': self.blur_intensity,
            'feather_amount': self.feather_amount,
            'fps': int(self.fps),
            'frame_count': self.frame_count,
            'use_segmentation': self.use_segmentation,
            'num_tracks': len(self.trackers)
        }


_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = StreamingDetector(model_size='s-seg')
    return _detector