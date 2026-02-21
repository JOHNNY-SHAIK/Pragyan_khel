import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter, binary_dilation
from deep_sort_realtime.deepsort_tracker import DeepSort
from shapely.geometry import Point, Polygon
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedObjectDetector:
    def __init__(self, model_size='s'):
        logger.info(f"Initializing YOLOv8-{model_size}-seg (segmentation)...")
        
        # Use segmentation model instead of detection
        self.yolo = YOLO(f'yolov8{model_size}-seg.pt')
        
        # DeepSORT for persistent tracking
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            embedder="mobilenet",
            half=True,
            embedder_gpu=False
        )
        
        self.cricket_classes = {
            0: 'player',
            32: 'ball',
            35: 'bat',
            56: 'chair',
            39: 'bottle'
        }
        
        self.current_focus_track_id = None
        self.focus_mask = None
        self.blur_intensity = 35
        self.feather_pixels = 15
        
        logger.info("✅ Advanced detector initialized with DeepSORT")
        
    def detect_objects(self, frame, conf_threshold=0.25):
        """
        Detect objects with instance segmentation and persistent tracking
        """
        results = self.yolo.predict(
            frame,
            conf=conf_threshold,
            iou=0.4,
            imgsz=1280,
            classes=list(self.cricket_classes.keys()),
            verbose=False
        )
        
        detections = []
        raw_detections = []  # For DeepSORT
        
        for result in results:
            boxes = result.boxes
            masks = result.masks
            
            if boxes is None:
                continue
            
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get segmentation mask
                mask = None
                polygon = None
                if masks is not None and i < len(masks):
                    mask_data = masks[i].data.cpu().numpy()[0]
                    mask = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
                    
                    # Extract polygon from mask
                    contours, _ = cv2.findContours(
                        (mask > 0.5).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        polygon = contours[0].squeeze()
                
                # Prepare for DeepSORT
                bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
                raw_detections.append((bbox, confidence, class_id))
                
                detection = {
                    'class_id': class_id,
                    'class_name': self.cricket_classes.get(class_id, 'unknown'),
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'mask': mask,
                    'polygon': polygon,
                    'track_id': None  # Will be assigned by DeepSORT
                }
                
                detections.append(detection)
        
        # Update tracks with DeepSORT
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        
        # Assign track IDs to detections
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()  # [x1, y1, x2, y2]
            
            # Find matching detection
            for detection in detections:
                det_bbox = detection['bbox']
                if self._iou(bbox, det_bbox) > 0.5:
                    detection['track_id'] = track_id
                    break
        
        return detections
    
    def _iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def point_in_polygon(self, point, polygon):
        """Check if click point is inside polygon mask"""
        if polygon is None or len(polygon) < 3:
            return False
        
        try:
            poly = Polygon(polygon)
            pt = Point(point)
            return poly.contains(pt)
        except:
            return False
    
    def set_focus(self, frame, click_pos, detections):
        """
        Set focus using point-in-polygon selection
        """
        x_click, y_click = click_pos
        
        # Sort by smallest area first (to prioritize smaller overlapping objects)
        sorted_detections = sorted(
            detections,
            key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1])
        )
        
        for detection in sorted_detections:
            # First check polygon (most accurate)
            if detection['polygon'] is not None:
                if self.point_in_polygon((x_click, y_click), detection['polygon']):
                    self.current_focus_track_id = detection['track_id']
                    self.focus_mask = self.create_advanced_mask(frame, detection)
                    logger.info(f"✓ Focus: {detection['class_name']} (Track ID: {detection['track_id']})")
                    return detection
            
            # Fallback to bounding box
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            if x1 <= x_click <= x2 and y1 <= y_click <= y2:
                self.current_focus_track_id = detection['track_id']
                self.focus_mask = self.create_advanced_mask(frame, detection)
                logger.info(f"✓ Focus: {detection['class_name']} (Track ID: {detection['track_id']})")
                return detection
        
        logger.warning("No object at click position")
        return None
    
    def create_advanced_mask(self, frame, detection):
        """
        Create advanced mask with dilation and feathering
        """
        h, w = frame.shape[:2]
        
        if detection['mask'] is not None:
            # Use segmentation mask
            mask = detection['mask'].astype(np.uint8)
            
            # Dilate to prevent edge bleeding
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
        else:
            # Fallback to elliptical mask
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = max(x2 - x1, 1)
            height = max(y2 - y1, 1)
            cv2.ellipse(mask, (center_x, center_y),
                       (width // 2 + 20, height // 2 + 20),
                       0, 0, 360, 1, -1)
        
        # Apply feathering
        mask = mask.astype(np.float32)
        mask = gaussian_filter(mask, sigma=self.feather_pixels)
        
        # Normalize
        if mask.max() > 0:
            mask = mask / mask.max()
        
        return mask
    
    def apply_depth_aware_blur(self, frame, mask):
        """
        Apply graduated blur based on distance from focus
        """
        if mask is None:
            return frame
        
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Resize if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        # Create distance map
        distance_map = cv2.distanceTransform(
            (1 - mask).astype(np.uint8),
            cv2.DIST_L2,
            5
        )
        
        # Normalize distance
        if distance_map.max() > 0:
            distance_map = distance_map / distance_map.max()
        
        # Create multiple blur layers
        blur_layers = {
            'near': cv2.GaussianBlur(frame, (15, 15), 0),
            'mid': cv2.GaussianBlur(frame, (35, 35), 0),
            'far': cv2.GaussianBlur(frame, (self.blur_intensity | 1, self.blur_intensity | 1), 0)
        }
        
        # Composite based on distance
        output = np.zeros_like(frame, dtype=np.float32)
        
        # Zone 1: Very close (sharp transition)
        zone1 = distance_map < 0.2
        output[zone1] = frame[zone1]
        
        # Zone 2: Near (light blur)
        zone2 = (distance_map >= 0.2) & (distance_map < 0.5)
        output[zone2] = blur_layers['near'][zone2]
        
        # Zone 3: Mid (medium blur)
        zone3 = (distance_map >= 0.5) & (distance_map < 0.8)
        output[zone3] = blur_layers['mid'][zone3]
        
        # Zone 4: Far (strong blur)
        zone4 = distance_map >= 0.8
        output[zone4] = blur_layers['far'][zone4]
        
        # Blend with mask
        mask_3d = np.expand_dims(mask, axis=2)
        final = mask_3d * frame.astype(np.float32) + (1 - mask_3d) * output
        
        return np.clip(final, 0, 255).astype(np.uint8)
    
    def update_focus_mask(self, frame, detections):
        """
        Update mask for currently focused track
        """
        if self.current_focus_track_id is None:
            return None
        
        # Find detection with matching track ID
        for detection in detections:
            if detection['track_id'] == self.current_focus_track_id:
                self.focus_mask = self.create_advanced_mask(frame, detection)
                return self.focus_mask
        
        logger.warning(f"Track ID {self.current_focus_track_id} lost")
        return None
    
    def clear_focus(self):
        """Clear current focus"""
        self.current_focus_track_id = None
        self.focus_mask = None
        logger.info("✓ Focus cleared")
    
    def set_blur_intensity(self, intensity):
        """Set blur intensity"""
        self.blur_intensity = intensity if intensity % 2 == 1 else intensity + 1
        logger.info(f"✓ Blur: {self.blur_intensity}")
    
    def set_feather_amount(self, pixels):
        """Set mask feathering amount"""
        self.feather_pixels = max(5, min(50, pixels))
        logger.info(f"✓ Feather: {self.feather_pixels}px")


_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = AdvancedObjectDetector(model_size='s')
    return _detector
