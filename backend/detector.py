import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter
from filterpy.kalman import KalmanFilter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedObjectDetector:
    def __init__(self, model_size='s'):
        logger.info(f"Initializing YOLOv8-{model_size}...")
        self.yolo = YOLO(f'yolov8{model_size}.pt')
        self.cricket_classes = {0: 'player', 32: 'ball', 35: 'bat', 56: 'chair', 39: 'bottle'}
        self.trackers = {}
        self.current_focus = None
        self.focus_mask = None
        self.blur_intensity = 35
        self.sam_available = False
        logger.info("Detector initialized")
        
    def create_kalman_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([0., 0., 0., 0.])
        kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        kf.H = np.array([[1,0,0,0],[0,1,0,0]])
        kf.P *= 1000
        kf.R = np.array([[5,0],[0,5]])
        kf.Q = np.eye(4) * 0.1
        return kf
        
    def detect_objects(self, frame, conf_threshold=0.25):
        results = self.yolo.predict(frame, conf=conf_threshold, iou=0.4, imgsz=1280, classes=list(self.cricket_classes.keys()), verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            masks = result.masks
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                mask = None
                if masks is not None and i < len(masks):
                    mask = masks[i].data.cpu().numpy()[0]
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                detection = {'class_id': class_id, 'class_name': self.cricket_classes.get(class_id, 'unknown'), 'confidence': confidence, 'bbox': (x1,y1,x2,y2), 'center': ((x1+x2)//2, (y1+y2)//2), 'mask': mask, 'id': f"{class_id}_{i}"}
                detections.append(detection)
                if class_id == 32:
                    if detection['id'] not in self.trackers:
                        self.trackers[detection['id']] = self.create_kalman_filter()
                    kf = self.trackers[detection['id']]
                    kf.predict()
                    kf.update(np.array(detection['center']))
        return detections
    
    def create_elliptical_mask(self, shape, bbox, feather_amount=30):
        mask = np.zeros(shape, dtype=np.float32)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        cv2.ellipse(mask, (center_x, center_y), (width//2+feather_amount, height//2+feather_amount), 0, 0, 360, 1, -1)
        mask = gaussian_filter(mask, sigma=feather_amount//2)
        if mask.max() > 0:
            mask = mask / mask.max()
        return mask
    
    def apply_selective_blur(self, frame, mask, blur_intensity=None):
        if blur_intensity is None:
            blur_intensity = self.blur_intensity
        if blur_intensity % 2 == 0:
            blur_intensity += 1
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        mask = mask.astype(np.float32)
        if mask.max() > 0:
            mask = mask / mask.max()
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        distance_map = cv2.distanceTransform((1-mask).astype(np.uint8), cv2.DIST_L2, 5)
        if distance_map.max() > 0:
            distance_map = distance_map / distance_map.max()
        blur_weak = cv2.GaussianBlur(frame, (21,21), 0)
        blur_medium = cv2.GaussianBlur(frame, (51,51), 0)
        blur_strong = cv2.GaussianBlur(frame, (blur_intensity, blur_intensity), 0)
        blurred = np.zeros_like(frame, dtype=np.float32)
        dist_weak = distance_map < 0.3
        dist_medium = (distance_map >= 0.3) & (distance_map < 0.6)
        dist_strong = distance_map >= 0.6
        blurred[dist_weak] = blur_weak[dist_weak]
        blurred[dist_medium] = blur_medium[dist_medium]
        blurred[dist_strong] = blur_strong[dist_strong]
        mask_3d = np.expand_dims(mask, axis=2)
        output = mask_3d * frame.astype(np.float32) + (1-mask_3d) * blurred
        return np.clip(output, 0, 255).astype(np.uint8)
    
    def set_focus(self, frame, click_pos, detections):
        x_click, y_click = click_pos
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            if x1 <= x_click <= x2 and y1 <= y_click <= y2:
                if detection['mask'] is not None:
                    mask_value = detection['mask'][y_click, x_click]
                    if mask_value > 0.5:
                        self.current_focus = detection['id']
                        self.focus_mask = self.create_elliptical_mask(frame.shape[:2], bbox)
                        logger.info(f"Focus: {detection['class_name']}")
                        return detection
                else:
                    self.current_focus = detection['id']
                    self.focus_mask = self.create_elliptical_mask(frame.shape[:2], bbox)
                    logger.info(f"Focus: {detection['class_name']}")
                    return detection
        return None
    
    def update_focus_mask(self, frame, detections):
        if self.current_focus is None:
            return None
        for detection in detections:
            if detection['id'] == self.current_focus:
                self.focus_mask = self.create_elliptical_mask(frame.shape[:2], detection['bbox'])
                return self.focus_mask
        if self.current_focus in self.trackers:
            kf = self.trackers[self.current_focus]
            predicted_pos = kf.x[:2]
            estimated_bbox = (int(predicted_pos[0]-20), int(predicted_pos[1]-20), int(predicted_pos[0]+20), int(predicted_pos[1]+20))
            self.focus_mask = self.create_elliptical_mask(frame.shape[:2], estimated_bbox)
            return self.focus_mask
        return None
    
    def clear_focus(self):
        self.current_focus = None
        self.focus_mask = None
        logger.info("Focus cleared")
    
    def set_blur_intensity(self, intensity):
        self.blur_intensity = intensity if intensity % 2 == 1 else intensity + 1
        logger.info(f"Blur: {self.blur_intensity}")

_detector = None
def get_detector():
    global _detector
    if _detector is None:
        _detector = AdvancedObjectDetector(model_size='s')
    return _detector
