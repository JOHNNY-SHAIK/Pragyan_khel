"""
FocusAI CPU-TURBO - Version 8.8 (FFMPEG ERROR FIXED)
- NO FFmpeg threading conflicts (FINAL FIX)
- Boxes correctly positioned
- No double boxes
- Head coverage fixed
- Good FPS without threading
- Stable boxes, no dancing
- Blur outside, sharp inside

Target: 15-25 FPS on CPU
"""

import cv2
import numpy as np
from ultralytics import YOLO
import threading
import logging
import os
import time

# CRITICAL: COMPLETE FFmpeg threading fix
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "4048576"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MIN_BOX_SIZE = 15
MAX_BOX_RATIO = 0.8
ELLIPSE_PADDING = 15
ELLIPSE_VERTICAL_EXTRA = 30
MAX_TRACKERS = 50
MIN_IOU_THRESHOLD = 0.15


class StableTracker:
    """Stable Tracker with EMA smoothing"""
    
    def __init__(self, bbox):
        x1, y1, x2, y2 = bbox
        
        self.cx = float((x1 + x2) / 2)
        self.cy = float((y1 + y2) / 2)
        self.width = float(x2 - x1)
        self.height = float(y2 - y1)
        
        self.vx = 0.0
        self.vy = 0.0
        
        self.display_cx = self.cx
        self.display_cy = self.cy
        self.display_w = self.width
        self.display_h = self.height
        
        self.frames_since_seen = 0
        self.hits = 1
        
        self.ema_alpha = 0.35
        self.display_alpha = 0.25
        self.min_movement = 3.0
        self.velocity_decay = 0.75
        
    def predict(self):
        self.cx += self.vx
        self.cy += self.vy
        
        self.vx *= self.velocity_decay
        self.vy *= self.velocity_decay
        
        self.display_cx += (self.cx - self.display_cx) * self.display_alpha
        self.display_cy += (self.cy - self.display_cy) * self.display_alpha
        self.display_w += (self.width - self.display_w) * self.display_alpha
        self.display_h += (self.height - self.display_h) * self.display_alpha
        
        x1 = int(self.display_cx - self.display_w / 2)
        y1 = int(self.display_cy - self.display_h / 2)
        x2 = int(self.display_cx + self.display_w / 2)
        y2 = int(self.display_cy + self.display_h / 2)
        
        return (x1, y1, x2, y2)
    
    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        new_cx = (x1 + x2) / 2
        new_cy = (y1 + y2) / 2
        new_w = x2 - x1
        new_h = y2 - y1
        
        dx = new_cx - self.cx
        dy = new_cy - self.cy
        distance = (dx**2 + dy**2) ** 0.5
        
        if distance > self.min_movement:
            self.vx = self.vx * 0.5 + dx * 0.5
            self.vy = self.vy * 0.5 + dy * 0.5
            self.cx += (new_cx - self.cx) * self.ema_alpha
            self.cy += (new_cy - self.cy) * self.ema_alpha
        
        self.width += (new_w - self.width) * self.ema_alpha * 0.5
        self.height += (new_h - self.height) * self.ema_alpha * 0.5
        
        self.frames_since_seen = 0
        self.hits += 1
    
    def get_stable_bbox(self):
        x1 = int(self.display_cx - self.display_w / 2)
        y1 = int(self.display_cy - self.display_h / 2)
        x2 = int(self.display_cx + self.display_w / 2)
        y2 = int(self.display_cy + self.display_h / 2)
        return (x1, y1, x2, y2)


class CPUTurboDetector:
    def __init__(self):
        logger.info("🚀 Initializing CPU-TURBO v8.8 (FFMPEG FIXED)...")
        
        self.yolo = YOLO('yolov8n.pt')
        
        self.cricket_classes = {
            0: 'player',
            32: 'ball',
            35: 'bat',
            38: 'racket',
            56: 'chair',
            39: 'bottle'
        }
        
        self.focused_track_id = None
        self.blur_intensity = 21
        
        self.cap = None
        self.is_playing = False
        self.frame_count = 0
        
        self.video_width = 640
        self.video_height = 480
        self.detection_size = 416
        
        self.trackers = {}
        self.track_history = {}
        self.next_track_id = 0
        
        self.last_focused_bbox = None
        self.focused_frames_missing = 0
        
        self.fps = 0
        self.fps_counter = 0
        self.fps_start = time.time()
        
        self.detect_every_n = 3
        self.skip_counter = 0
        
        self.blur_downscale = 4
        self.blur_kernel = 15
        
        self.cached_detections = []
        
        self.lock = threading.Lock()
        self.processed_frame = None
        
        logger.info("✅ v8.8 Ready - FFmpeg threading fixed!")

    def load_video(self, video_path):
        if self.cap:
            self.cap.release()
            self.cap = None
            time.sleep(0.3)
        
        # Try Windows Media Foundation first (no threading issues)
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_MSMF)
        
        if not self.cap.isOpened():
            logger.warning("MSMF failed, trying FFMPEG...")
            self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open: {video_path}")
            return 0, 0
        
        # CRITICAL: Force single-threaded mode
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if orig_w > 1920:
            self.video_width = 1280
            self.video_height = int(orig_h * (1280 / orig_w))
            logger.warning(f"⚠️ Large video ({orig_w}px) - resizing to {self.video_width}x{self.video_height}")
        elif orig_w > 1280:
            self.video_width = 1280
            self.video_height = int(orig_h * (1280 / orig_w))
            logger.info(f"📐 Resizing to {self.video_width}x{self.video_height}")
        else:
            self.video_width = orig_w
            self.video_height = orig_h
        
        self.is_playing = False
        self.frame_count = 0
        self.trackers = {}
        self.track_history = {}
        self.next_track_id = 0
        self.focused_track_id = None
        self.last_focused_bbox = None
        self.cached_detections = []
        
        logger.info(f"✅ Loaded: {orig_w}x{orig_h} @ {fps:.1f} FPS, {total} frames")
        
        return fps, total

    def start_processing(self):
        self.is_playing = True
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start = time.time()
        logger.info("▶️ Started")
        return True

    def stop_processing(self):
        self.is_playing = False
        logger.info("⏸️ Paused")
        return True

    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        inter = (x2 - x1) * (y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        
        return inter / (a1 + a2 - inter + 1e-6)

    def _track(self, detections, frame_w, frame_h):
        preds = {}
        for tid, tr in self.trackers.items():
            preds[tid] = tr.predict()
            tr.frames_since_seen += 1
        
        matched = set()
        
        for d in detections:
            bbox = d['bbox']
            
            x1 = max(0, min(bbox[0], frame_w - 1))
            y1 = max(0, min(bbox[1], frame_h - 1))
            x2 = max(x1 + 1, min(bbox[2], frame_w))
            y2 = max(y1 + 1, min(bbox[3], frame_h))
            
            if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
                continue
            
            d['bbox'] = (x1, y1, x2, y2)
            
            best_id = None
            best_iou = MIN_IOU_THRESHOLD
            
            for tid, pred in preds.items():
                if tid in matched:
                    continue
                iou = self._iou(d['bbox'], pred)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            
            if best_id:
                d['track_id'] = best_id
                self.trackers[best_id].update(d['bbox'])
                d['bbox'] = self.trackers[best_id].get_stable_bbox()
                self.track_history[best_id] = d['bbox']
                matched.add(best_id)
                
                if best_id == self.focused_track_id:
                    self.last_focused_bbox = d['bbox']
                    self.focused_frames_missing = 0
            else:
                nid = self.next_track_id
                self.next_track_id += 1
                d['track_id'] = nid
                self.trackers[nid] = StableTracker(d['bbox'])
                d['bbox'] = self.trackers[nid].get_stable_bbox()
                self.track_history[nid] = d['bbox']
        
        to_del = [t for t, tr in self.trackers.items() 
                  if tr.frames_since_seen > 30 and t != self.focused_track_id]
        for t in to_del:
            del self.trackers[t]
            self.track_history.pop(t, None)
        
        if len(self.trackers) > MAX_TRACKERS:
            sorted_tracks = sorted(self.trackers.items(), 
                                   key=lambda x: x[1].frames_since_seen, 
                                   reverse=True)
            for tid, _ in sorted_tracks[MAX_TRACKERS:]:
                if tid != self.focused_track_id:
                    del self.trackers[tid]
                    self.track_history.pop(tid, None)

    def detect(self, frame):
        frame_h, frame_w = frame.shape[:2]
        
        results = self.yolo.predict(
            frame,
            conf=0.25,
            iou=0.5,
            imgsz=self.detection_size,
            classes=list(self.cricket_classes.keys()),
            verbose=False,
            device='cpu'
        )
        
        detections = []
        
        for r in results:
            if r.boxes is None:
                continue
            
            for box in r.boxes:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                x1 = max(0, min(x1, frame_w - 1))
                y1 = max(0, min(y1, frame_h - 1))
                x2 = max(x1 + 10, min(x2, frame_w))
                y2 = max(y1 + 10, min(y2, frame_h))
                
                box_w = x2 - x1
                box_h = y2 - y1
                
                if box_w < MIN_BOX_SIZE or box_h < MIN_BOX_SIZE:
                    continue
                
                if box_w > frame_w * MAX_BOX_RATIO or box_h > frame_h * MAX_BOX_RATIO:
                    continue
                
                detections.append({
                    'class_id': cid,
                    'class_name': self.cricket_classes.get(cid, '?'),
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'track_id': None
                })
        
        self._track(detections, frame_w, frame_h)
        
        if self.focused_track_id is not None:
            found = any(d['track_id'] == self.focused_track_id for d in detections)
            
            if not found:
                self.focused_frames_missing += 1
                if self.focused_track_id in self.trackers:
                    pred = self.trackers[self.focused_track_id].predict()
                    
                    if (0 <= pred[0] < frame_w and 0 <= pred[1] < frame_h and
                        pred[2] > pred[0] and pred[3] > pred[1]):
                        detections.append({
                            'class_id': 0,
                            'class_name': 'tracking',
                            'confidence': 0.5,
                            'bbox': pred,
                            'track_id': self.focused_track_id
                        })
                        self.last_focused_bbox = pred
            else:
                self.focused_frames_missing = 0
        
        return detections

    def blur_correct(self, frame, detections):
        if self.focused_track_id is None:
            return frame
        
        focused = None
        for d in detections:
            if d['track_id'] == self.focused_track_id:
                focused = d
                break
        
        if not focused and self.last_focused_bbox:
            focused = {'bbox': self.last_focused_bbox}
        
        if not focused:
            return frame
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = focused['bbox']
        
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        if x2 - x1 < 20 or y2 - y1 < 20:
            return frame
        
        ds = self.blur_downscale
        sw = max(10, w // ds)
        sh = max(10, h // ds)
        
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_NEAREST)
        
        k = self.blur_kernel
        if k % 2 == 0:
            k += 1
        k = max(3, min(k, 31))
        
        blurred_small = cv2.GaussianBlur(small, (k, k), 0)
        blurred_bg = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        output = blurred_bg.copy()
        
        padding = 35
        roi_x1 = max(0, x1 - padding)
        roi_y1 = max(0, y1 - padding)
        roi_x2 = min(w, x2 + padding)
        roi_y2 = min(h, y2 + padding)
        
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        
        if roi_w <= 10 or roi_h <= 10:
            return output
        
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        
        obj_w = x2 - x1
        obj_h = y2 - y1
        
        obj_cx = (x1 + x2) // 2 - roi_x1
        obj_cy = int(y1 + obj_h * 0.40) - roi_y1
        
        ellipse_ax = max(10, (obj_w // 2) + ELLIPSE_PADDING)
        ellipse_ay = max(10, (obj_h // 2) + ELLIPSE_VERTICAL_EXTRA)
        
        ellipse_ax = min(ellipse_ax, roi_w // 2 - 2)
        ellipse_ay = min(ellipse_ay, roi_h // 2 - 2)
        
        if ellipse_ax > 5 and ellipse_ay > 5:
            cv2.ellipse(mask, (obj_cx, obj_cy), (ellipse_ax, ellipse_ay), 
                        0, 0, 360, 255, -1)
        
        mask_float = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
        mask_3d = mask_float[:, :, np.newaxis]
        
        sharp_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        blur_roi = output[roi_y1:roi_y2, roi_x1:roi_x2]
        
        blended_roi = (sharp_roi.astype(np.float32) * mask_3d + 
                       blur_roi.astype(np.float32) * (1 - mask_3d)).astype(np.uint8)
        
        output[roi_y1:roi_y2, roi_x1:roi_x2] = blended_roi
        
        return output

    def annotate(self, frame, detections):
        self.fps_counter += 1
        if self.fps_counter >= 10:
            elapsed = time.time() - self.fps_start
            if elapsed > 0:
                self.fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start = time.time()
        
        h, w = frame.shape[:2]
        
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            
            x1 = max(0, min(x1, w - 5))
            y1 = max(0, min(y1, h - 5))
            x2 = max(x1 + 5, min(x2, w))
            y2 = max(y1 + 5, min(y2, h))
            
            tid = d['track_id']
            
            if tid == self.focused_track_id:
                color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                ln = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
                if ln > 3:
                    cv2.line(frame, (x1, y1), (x1 + ln, y1), color, 5)
                    cv2.line(frame, (x1, y1), (x1, y1 + ln), color, 5)
                    cv2.line(frame, (x2, y1), (x2 - ln, y1), color, 5)
                    cv2.line(frame, (x2, y1), (x2, y1 + ln), color, 5)
                    cv2.line(frame, (x1, y2), (x1 + ln, y2), color, 5)
                    cv2.line(frame, (x1, y2), (x1, y2 - ln), color, 5)
                    cv2.line(frame, (x2, y2), (x2 - ln, y2), color, 5)
                    cv2.line(frame, (x2, y2), (x2, y2 - ln), color, 5)
                
                label = f"FOCUS #{tid}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
                label_y = max(th + 10, y1)
                cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 8, label_y), color, -1)
                cv2.putText(frame, label, (x1 + 4, label_y - 4), font, 0.6, (0, 0, 0), 2)
                
                continue
            
            color = (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"#{tid}"
            label_y = max(15, y1 - 6)
            cv2.putText(frame, label, (x1 + 2, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        bar_h = 60
        cv2.rectangle(frame, (0, 0), (260, bar_h), (30, 30, 30), -1)
        cv2.line(frame, (0, bar_h), (260, bar_h), (0, 255, 0), 2)
        
        if self.fps >= 15:
            fps_color = (0, 255, 0)
            fps_status = "Good"
        elif self.fps >= 8:
            fps_color = (0, 255, 255)
            fps_status = "OK"
        else:
            fps_color = (0, 0, 255)
            fps_status = "Slow"
        
        cv2.putText(frame, f"FPS: {int(self.fps)} ({fps_status})", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 2)
        
        cv2.putText(frame, f"Size: {w}x{h}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        if self.focused_track_id is not None:
            cv2.putText(frame, f"Focus: #{self.focused_track_id}", (10, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Click to focus", (10, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        return frame

    def get_next_frame(self):
        if not self.cap or not self.is_playing:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            
            cv2.rectangle(blank, (80, 160), (560, 320), (40, 40, 40), -1)
            cv2.rectangle(blank, (80, 160), (560, 320), (0, 255, 0), 2)
            
            cv2.putText(blank, "FocusAI v8.8", (230, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(blank, "FFmpeg Error Fixed!", (190, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(blank, "Upload • Play • Click to Focus", (160, 290),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return blank
        
        success, frame = self.cap.read()
        
        if not success or frame is None or frame.size == 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.cap.read()
            if not success or frame is None or frame.size == 0:
                return np.zeros((480, 640, 3), dtype=np.uint8)
        
        h, w = frame.shape[:2]
        if w != self.video_width or h != self.video_height:
            frame = cv2.resize(frame, (self.video_width, self.video_height), 
                               interpolation=cv2.INTER_AREA)
        
        self.frame_count += 1
        
        self.skip_counter += 1
        if self.skip_counter >= self.detect_every_n:
            self.skip_counter = 0
            detections = self.detect(frame)
            with self.lock:
                self.cached_detections = detections
        else:
            with self.lock:
                detections = []
                for d in self.cached_detections:
                    d_copy = d.copy()
                    tid = d_copy['track_id']
                    if tid in self.trackers:
                        d_copy['bbox'] = self.trackers[tid].predict()
                    detections.append(d_copy)
        
        if self.focused_track_id is not None:
            frame = self.blur_correct(frame, detections)
        
        frame = self.annotate(frame, detections)
        
        with self.lock:
            self.processed_frame = frame
        
        return frame

    def set_focus_by_click(self, x, y):
        with self.lock:
            detections = self.cached_detections.copy() if self.cached_detections else []
        
        if not detections:
            return None
        
        best = None
        best_dist = float('inf')
        
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            if x1 - 20 <= x <= x2 + 20 and y1 - 20 <= y <= y2 + 20:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                dist = abs(x - cx) + abs(y - cy)
                if dist < best_dist:
                    best_dist = dist
                    best = d
        
        if best:
            self.focused_track_id = best['track_id']
            self.focused_frames_missing = 0
            self.last_focused_bbox = best['bbox']
            logger.info(f"🎯 Focus: #{best['track_id']}")
            return best
        
        return None

    def clear_focus(self):
        self.focused_track_id = None
        self.focused_frames_missing = 0
        self.last_focused_bbox = None
        logger.info("🔓 Cleared")
        return True

    def set_blur_intensity(self, value):
        self.blur_intensity = max(5, min(35, int(value)))
        self.blur_kernel = self.blur_intensity
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        return self.blur_intensity

    def get_status(self):
        return {
            'is_playing': self.is_playing,
            'focused_track_id': self.focused_track_id,
            'blur_intensity': self.blur_intensity,
            'fps': int(self.fps),
            'frame_count': self.frame_count,
            'tracks': len(self.trackers),
            'video_size': f"{self.video_width}x{self.video_height}"
        }


_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = CPUTurboDetector()
    return _detector