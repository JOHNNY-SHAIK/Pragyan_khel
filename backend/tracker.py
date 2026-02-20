import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class Track:
    """Single object track"""
    track_count = 0
    
    def __init__(self, detection):
        self.id = Track.track_count
        Track.track_count += 1
        
        self.class_name = detection['class']
        self.confidence = detection['confidence']
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.detection_id = detection['id']
        
        # Kalman Filter for smooth tracking
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State: [x, y, s, r, vx, vy, vs]
        # x, y = center, s = scale(area), r = aspect ratio
        x, y, w, h = detection['bbox']
        self.kf.x[:4] = np.array([
            [x + w/2], [y + h/2], [w * h], [w / max(h, 1)]
        ])
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Covariance matrices
        self.kf.R *= 10.0   # Measurement noise
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty for velocities
        self.kf.Q[4:, 4:] *= 0.01
        
        # Track management
        self.age = 0
        self.hits = 1
        self.misses = 0
        self.max_misses = 15
        
        # Speed tracking
        self.positions = [(x + w/2, y + h/2)]
        self.speed = 0
    
    def predict(self):
        """Predict next state"""
        self.kf.predict()
        self.age += 1
        return self.get_bbox()
    
    def update(self, detection):
        """Update track with new detection"""
        x, y, w, h = detection['bbox']
        
        measurement = np.array([
            [x + w/2], [y + h/2], [w * h], [w / max(h, 1)]
        ])
        
        self.kf.update(measurement)
        
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = detection['confidence']
        self.class_name = detection['class']
        self.detection_id = detection['id']
        self.hits += 1
        self.misses = 0
        
        # Update speed
        new_pos = (x + w/2, y + h/2)
        if self.positions:
            last_pos = self.positions[-1]
            self.speed = np.sqrt(
                (new_pos[0] - last_pos[0])**2 + 
                (new_pos[1] - last_pos[1])**2
            )
        self.positions.append(new_pos)
        if len(self.positions) > 30:
            self.positions.pop(0)
    
    def mark_missed(self):
        """Mark as missed in current frame"""
        self.misses += 1
    
    def is_dead(self):
        """Check if track should be removed"""
        return self.misses > self.max_misses
    
    def get_bbox(self):
        """Get current bounding box from Kalman state"""
        state = self.kf.x
        cx, cy, s, r = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        
        w = np.sqrt(max(s * r, 1))
        h = max(s / max(w, 1), 1)
        
        return [int(cx - w/2), int(cy - h/2), int(w), int(h)]
    
    def to_dict(self):
        """Convert to dictionary for JSON"""
        return {
            'track_id': self.id,
            'class': self.class_name,
            'confidence': round(self.confidence, 3),
            'bbox': self.get_bbox(),
            'center': [int(self.kf.x[0, 0]), int(self.kf.x[1, 0])],
            'speed': round(self.speed, 2),
            'age': self.age,
            'hits': self.hits,
            'detection_id': self.detection_id
        }


class ObjectTracker:
    """Multi-object tracker using IOU + Kalman Filter (SORT-based)"""
    
    def __init__(self):
        self.tracks = []
        self.iou_threshold = 0.3
        self.min_hits = 3  # Minimum hits to consider confirmed
    
    def update(self, detections, frame=None):
        """Update tracks with new detections"""
        
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        if not detections:
            # Mark all as missed
            for track in self.tracks:
                track.mark_missed()
            self.tracks = [t for t in self.tracks if not t.is_dead()]
            return [t.to_dict() for t in self.tracks]
        
        if not self.tracks:
            # Initialize tracks for all detections
            for det in detections:
                self.tracks.append(Track(det))
            return [t.to_dict() for t in self.tracks]
        
        # Calculate IOU cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                cost_matrix[t, d] = 1 - self._calculate_iou(
                    track.get_bbox(), det['bbox']
                )
        
        # Hungarian algorithm for optimal assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Track matched and unmatched
        matched = set()
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))
        
        for t, d in zip(track_indices, det_indices):
            if cost_matrix[t, d] < (1 - self.iou_threshold):
                self.tracks[t].update(detections[d])
                matched.add(t)
                unmatched_tracks.discard(t)
                unmatched_dets.discard(d)
        
        # Mark unmatched tracks as missed
        for t in unmatched_tracks:
            self.tracks[t].mark_missed()
        
        # Create new tracks for unmatched detections
        for d in unmatched_dets:
            self.tracks.append(Track(detections[d]))
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead()]
        
        return [t.to_dict() for t in self.tracks]
    
    def _calculate_iou(self, box1, box2):
        """Calculate IOU between two bounding boxes [x, y, w, h]"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to x1, y1, x2, y2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        
        intersection = max(0, xb - xa) * max(0, yb - ya)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)
    
    def get_track_by_class(self, class_name):
        """Get tracks of specific class"""
        return [t for t in self.tracks if t.class_name == class_name]
    
    def get_fastest_object(self):
        """Get the fastest moving tracked object"""
        if not self.tracks:
            return None
        return max(self.tracks, key=lambda t: t.speed).to_dict()
    
    def reset(self):
        """Reset all tracks"""
        self.tracks = []
        Track.track_count = 0
