import cv2
import numpy as np

class BlurEngine:
    """Advanced blur effects engine for cinematic focus"""
    
    def __init__(self):
        self.blur_cache = {}
    
    def apply_focus_blur(self, frame, tracked_objects, focus_target, blur_intensity=15):
        """Apply blur to everything except the focused object"""
        
        if not tracked_objects or not focus_target:
            return frame
        
        # Find the target object
        target = None
        for obj in tracked_objects:
            if focus_target.get('track_id') is not None:
                if obj['track_id'] == focus_target['track_id']:
                    target = obj
                    break
            elif focus_target.get('class'):
                if obj['class'] == focus_target['class']:
                    target = obj
                    break
        
        if target is None:
            # Auto-focus on largest object
            if tracked_objects:
                target = max(tracked_objects, key=lambda o: o['bbox'][2] * o['bbox'][3])
            else:
                return frame
        
        return self._apply_bokeh_blur(frame, target, blur_intensity)
    
    def _apply_bokeh_blur(self, frame, target, blur_intensity):
        """Apply professional bokeh-style blur"""
        h, w = frame.shape[:2]
        
        # Create blurred version
        blur_kernel = max(3, blur_intensity * 2 + 1)  # Must be odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        
        # Create mask for the focused object
        mask = np.zeros((h, w), dtype=np.float32)
        
        x, y, bw, bh = target['bbox']
        cx, cy = x + bw // 2, y + bh // 2
        
        # Create elliptical mask with soft edges
        padding = int(max(bw, bh) * 0.15)
        rx = bw // 2 + padding
        ry = bh // 2 + padding
        
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        
        # Apply Gaussian blur to mask for smooth transition
        mask = cv2.GaussianBlur(mask, (51, 51), 20)
        
        # Ensure mask values are 0-1
        mask = np.clip(mask, 0, 1)
        
        # Blend: sharp * mask + blurred * (1 - mask)
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = (frame.astype(np.float32) * mask_3ch + 
                  blurred.astype(np.float32) * (1 - mask_3ch))
        
        # Add subtle vignette
        result = self._add_vignette(result.astype(np.uint8), cx, cy, max(rx, ry))
        
        return result
    
    def apply_tilt_shift(self, frame, target, blur_intensity=15):
        """Apply tilt-shift miniature effect"""
        h, w = frame.shape[:2]
        
        blur_kernel = max(3, blur_intensity * 2 + 1)
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        
        # Create horizontal band mask
        _, cy, _, bh = target['bbox']
        center_y = cy + bh // 2
        band_height = bh * 2
        
        mask = np.zeros((h, w), dtype=np.float32)
        y_start = max(0, center_y - band_height // 2)
        y_end = min(h, center_y + band_height // 2)
        mask[y_start:y_end, :] = 1.0
        
        mask = cv2.GaussianBlur(mask, (1, 101), 30)
        mask_3ch = np.stack([mask] * 3, axis=-1)
        
        result = (frame.astype(np.float32) * mask_3ch + 
                  blurred.astype(np.float32) * (1 - mask_3ch))
        
        return result.astype(np.uint8)
    
    def apply_radial_blur(self, frame, target, blur_intensity=15):
        """Apply radial blur from focus point"""
        h, w = frame.shape[:2]
        x, y, bw, bh = target['bbox']
        cx, cy = x + bw // 2, y + bh // 2
        
        result = frame.copy()
        
        # Create distance map from center
        Y, X = np.mgrid[0:h, 0:w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(w**2 + h**2) / 2
        
        # Normalize distance
        norm_dist = dist / max_dist
        
        # Apply progressive blur based on distance
        for i in range(1, 5):
            factor = i / 4.0
            kernel = max(3, int(blur_intensity * factor * 2 + 1))
            if kernel % 2 == 0:
                kernel += 1
            
            layer_blur = cv2.GaussianBlur(frame, (kernel, kernel), 0)
            
            mask = np.clip((norm_dist - (i - 1) * 0.25) / 0.25, 0, 1)
            mask_3ch = np.stack([mask] * 3, axis=-1)
            
            result = (result.astype(np.float32) * (1 - mask_3ch) + 
                      layer_blur.astype(np.float32) * mask_3ch).astype(np.uint8)
        
        return result
    
    def _add_vignette(self, frame, cx, cy, radius):
        """Add subtle vignette effect"""
        h, w = frame.shape[:2]
        
        # Create vignette mask
        Y, X = np.mgrid[0:h, 0:w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        vignette = 1 - np.clip((dist - radius * 2) / (max(w, h) * 0.5), 0, 0.3)
        vignette_3ch = np.stack([vignette] * 3, axis=-1)
        
        result = (frame.astype(np.float32) * vignette_3ch).astype(np.uint8)
        return result
    
    def enhance_low_light(self, frame, intensity=1.5):
        """Enhance low-light frames"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Adjust gamma if needed
        avg_brightness = np.mean(l)
        if avg_brightness < 80:
            gamma = intensity
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255 
                for i in np.arange(256)
            ]).astype(np.uint8)
            enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
