import cv2
import numpy as np
from ultralytics import YOLO

class CricketDetector:
    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.4
        self.load_model()
        
        # Cricket-relevant COCO classes
        self.cricket_classes = {
            0: 'player',        # person -> player
            32: 'ball',         # sports ball -> cricket ball
            34: 'bat',          # baseball bat -> cricket bat
            37: 'racket',       # sports equipment
            29: 'frisbee',      # disc-like objects
            56: 'chair',        # umpire chair
            67: 'phone',        # for general detection
            73: 'laptop',       # scoreboard
        }
        
        # Colors for each class
        self.class_colors = {
            'player': (0, 255, 0),      # Green
            'ball': (0, 0, 255),        # Red
            'bat': (255, 165, 0),       # Orange
            'racket': (255, 255, 0),    # Yellow
            'frisbee': (128, 0, 128),   # Purple
            'chair': (0, 128, 128),     # Teal
            'phone': (255, 192, 203),   # Pink
            'laptop': (128, 128, 0),    # Olive
            'unknown': (200, 200, 200)  # Gray
        }
    
    def load_model(self):
        """Load YOLOv8 nano model"""
        try:
            self.model = YOLO('yolov8n.pt')  # Auto-downloads if not present
            print("? YOLOv8n model loaded successfully")
        except Exception as e:
            print(f"? Error loading model: {e}")
            # Fallback: try to download
            try:
                self.model = YOLO('yolov8n.pt')
                print("? YOLOv8n downloaded and loaded")
            except Exception as e2:
                print(f"? Fatal error: {e2}")
    
    def detect(self, frame, conf=None):
        """Run detection on a frame"""
        if self.model is None:
            return []
        
        threshold = conf or self.confidence_threshold
        
        try:
            results = self.model(frame, conf=threshold, verbose=False)[0]
            
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Map to cricket-relevant class names
                class_name = self.cricket_classes.get(cls_id, None)
                
                # Skip non-cricket classes
                if class_name is None:
                    class_name = results.names[cls_id]
                
                detection = {
                    'id': len(detections),
                    'class': class_name,
                    'class_id': cls_id,
                    'confidence': round(confidence, 3),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h
                    'bbox_xyxy': [x1, y1, x2, y2],
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def detect_cricket_specific(self, frame):
        """Enhanced detection with cricket-specific post-processing"""
        detections = self.detect(frame)
        
        # Post-processing for cricket context
        enhanced = []
        
        for det in detections:
            # Enhance ball detection - small round objects
            if det['class'] == 'ball':
                w, h = det['bbox'][2], det['bbox'][3]
                aspect_ratio = w / max(h, 1)
                
                # Cricket ball should be roughly circular and small
                if 0.7 < aspect_ratio < 1.4 and w < frame.shape[1] * 0.1:
                    det['cricket_type'] = 'cricket_ball'
                    det['confidence'] = min(det['confidence'] * 1.2, 1.0)
                    enhanced.append(det)
                else:
                    enhanced.append(det)
            
            # Enhance player detection
            elif det['class'] == 'player':
                h = det['bbox'][3]
                # Players should be reasonably sized
                if h > frame.shape[0] * 0.15:
                    det['cricket_type'] = 'batsman' if self._is_batting_pose(det, detections) else 'fielder'
                    enhanced.append(det)
                else:
                    enhanced.append(det)
            
            # Enhance bat detection
            elif det['class'] == 'bat':
                det['cricket_type'] = 'cricket_bat'
                enhanced.append(det)
            
            else:
                enhanced.append(det)
        
        return enhanced if enhanced else detections
    
    def _is_batting_pose(self, player_det, all_detections):
        """Simple heuristic to detect if player is batting"""
        player_center = player_det['center']
        
        for det in all_detections:
            if det['class'] == 'bat':
                bat_center = det['center']
                distance = np.sqrt(
                    (player_center[0] - bat_center[0])**2 + 
                    (player_center[1] - bat_center[1])**2
                )
                if distance < player_det['bbox'][3]:  # Within player height
                    return True
        return False
    
    def draw_detections(self, frame, detections, tracked_objects=None):
        """Draw detection boxes on frame"""
        output = frame.copy()
        
        for det in detections:
            x, y, w, h = det['bbox']
            class_name = det.get('cricket_type', det['class'])
            confidence = det['confidence']
            
            color = self.class_colors.get(det['class'], self.class_colors['unknown'])
            
            # Check if this object is being tracked
            is_tracked = False
            if tracked_objects:
                for tracked in tracked_objects:
                    if tracked.get('detection_id') == det['id']:
                        is_tracked = True
                        color = (0, 229, 255)  # Cyan for tracked
                        break
            
            # Draw bounding box
            thickness = 3 if is_tracked else 2
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
            
            # Draw corner accents for tracked objects
            if is_tracked:
                corner_len = 20
                # Top-left
                cv2.line(output, (x, y), (x + corner_len, y), (0, 87, 255), 4)
                cv2.line(output, (x, y), (x, y + corner_len), (0, 87, 255), 4)
                # Top-right
                cv2.line(output, (x + w, y), (x + w - corner_len, y), (0, 87, 255), 4)
                cv2.line(output, (x + w, y), (x + w, y + corner_len), (0, 87, 255), 4)
                # Bottom-left
                cv2.line(output, (x, y + h), (x + corner_len, y + h), (0, 87, 255), 4)
                cv2.line(output, (x, y + h), (x, y + h - corner_len), (0, 87, 255), 4)
                # Bottom-right
                cv2.line(output, (x + w, y + h), (x + w - corner_len, y + h), (0, 87, 255), 4)
                cv2.line(output, (x + w, y + h), (x + w, y + h - corner_len), (0, 87, 255), 4)
            
            # Label background
            label = f"{class_name} {confidence:.0%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(output, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), color, -1)
            cv2.putText(output, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output
    
    def get_detectable_classes(self):
        """Return list of detectable cricket classes"""
        return list(self.cricket_classes.values())
