import cv2
import time
import json
import os
import numpy as np

class VideoProcessor:
    """Complete video processing pipeline"""
    
    def __init__(self, detector, tracker, blur_engine):
        self.detector = detector
        self.tracker = tracker
        self.blur_engine = blur_engine
    
    def process_video(self, input_path, output_path, focus_target=None, 
                      blur_intensity=15, show_detections=True):
        """Process entire video with focus effect"""
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            return {'error': 'Could not open video'}
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing stats
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'detections_per_frame': [],
            'processing_times': [],
            'objects_tracked': set()
        }
        
        # Reset tracker
        self.tracker.reset()
        
        frame_count = 0
        auto_focus_target = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Detect objects
            detections = self.detector.detect_cricket_specific(frame)
            
            # Update tracker
            tracked = self.tracker.update(detections, frame)
            
            # Track unique objects
            for t in tracked:
                stats['objects_tracked'].add(f"{t['class']}_{t['track_id']}")
            
            # Determine focus target
            current_target = focus_target
            
            if not current_target and tracked:
                # Auto-focus: prioritize ball > bat > largest player
                ball_tracks = [t for t in tracked if t['class'] == 'ball']
                bat_tracks = [t for t in tracked if t['class'] == 'bat']
                player_tracks = [t for t in tracked if t['class'] == 'player']
                
                if ball_tracks:
                    current_target = {'track_id': ball_tracks[0]['track_id']}
                elif bat_tracks:
                    current_target = {'track_id': bat_tracks[0]['track_id']}
                elif player_tracks:
                    largest = max(player_tracks, key=lambda t: t['bbox'][2] * t['bbox'][3])
                    current_target = {'track_id': largest['track_id']}
            
            # Apply blur effect
            if current_target and tracked:
                frame = self.blur_engine.apply_focus_blur(
                    frame, tracked, current_target, blur_intensity
                )
            
            # Draw detections overlay
            if show_detections:
                frame = self.detector.draw_detections(frame, detections, tracked)
                
                # Draw analytics overlay
                frame = self._draw_analytics(frame, tracked, frame_count, fps, stats)
            
            # Write frame
            writer.write(frame)
            
            # Update stats
            processing_time = time.time() - start_time
            stats['processed_frames'] = frame_count + 1
            stats['detections_per_frame'].append(len(detections))
            stats['processing_times'].append(processing_time)
            
            frame_count += 1
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_time = np.mean(stats['processing_times'][-30:])
                print(f"  Processing: {progress:.1f}% | FPS: {1/avg_time:.1f}")
        
        cap.release()
        writer.release()
        
        # Final stats
        result = {
            'success': True,
            'output_path': output_path,
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'avg_fps': round(1 / np.mean(stats['processing_times']), 1),
            'avg_detections': round(np.mean(stats['detections_per_frame']), 1),
            'unique_objects': len(stats['objects_tracked']),
            'processing_time': round(sum(stats['processing_times']), 2)
        }
        
        print(f"\n? Processing complete!")
        print(f"   Frames: {frame_count}")
        print(f"   Avg FPS: {result['avg_fps']}")
        print(f"   Objects tracked: {result['unique_objects']}")
        
        return result
    
    def process_frame(self, frame, focus_target=None, blur_intensity=15):
        """Process a single frame"""
        detections = self.detector.detect_cricket_specific(frame)
        tracked = self.tracker.update(detections, frame)
        
        if focus_target and tracked:
            frame = self.blur_engine.apply_focus_blur(
                frame, tracked, focus_target, blur_intensity
            )
        
        frame = self.detector.draw_detections(frame, detections, tracked)
        
        return frame, detections, tracked
    
    def _draw_analytics(self, frame, tracked, frame_num, fps, stats):
        """Draw analytics overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Text info
        y_offset = 35
        
        # FPS
        if stats['processing_times']:
            current_fps = 1 / stats['processing_times'][-1] if stats['processing_times'][-1] > 0 else 0
            color = (0, 255, 0) if current_fps > 20 else (0, 255, 255) if current_fps > 10 else (0, 0, 255)
            cv2.putText(frame, f"FPS: {current_fps:.0f}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Objects count
        y_offset += 30
        cv2.putText(frame, f"Objects: {len(tracked)}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 229, 255), 2)
        
        # Frame number
        y_offset += 30
        cv2.putText(frame, f"Frame: {frame_num}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Tracked objects list
        y_offset += 30
        for t in tracked[:3]:  # Show max 3
            label = f"  {t['class']}: {t['confidence']:.0%}"
            cv2.putText(frame, label, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_offset += 20
        
        return frame
