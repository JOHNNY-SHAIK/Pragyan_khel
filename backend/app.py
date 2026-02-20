import os
import cv2
import json
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from detector import CricketDetector
from tracker import ObjectTracker
from blur_engine import BlurEngine
from processor import VideoProcessor

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:5173"])
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize components
detector = CricketDetector()
tracker = ObjectTracker()
blur_engine = BlurEngine()
processor = VideoProcessor(detector, tracker, blur_engine)

# Store state
app_state = {
    'focus_target': None,
    'blur_intensity': 15,
    'tracking_mode': 'auto',
    'show_detections': True
}

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': detector.model is not None,
        'version': '2.0 - Cricket Edition'
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
    file.save(filepath)
    
    # Get video info
    cap = cv2.VideoCapture(filepath)
    info = {
        'filename': file.filename,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    cap.release()
    
    return jsonify({'success': True, 'video_info': info})

@app.route('/api/detect_frame', methods=['POST'])
def detect_frame():
    """Detect objects in a single frame"""
    data = request.json
    
    if 'frame' in data:
        # Decode base64 frame
        frame_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif 'video_path' in data and 'frame_number' in data:
        filepath = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
        cap = cv2.VideoCapture(filepath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, data['frame_number'])
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({'error': 'Could not read frame'}), 400
    else:
        return jsonify({'error': 'No frame data provided'}), 400
    
    # Run detection
    detections = detector.detect(frame)
    
    # Update tracker
    tracked = tracker.update(detections, frame)
    
    return jsonify({
        'detections': detections,
        'tracked_objects': tracked,
        'frame_shape': list(frame.shape)
    })

@app.route('/api/set_focus', methods=['POST'])
def set_focus():
    """Set the focus target"""
    data = request.json
    app_state['focus_target'] = data.get('target')
    app_state['blur_intensity'] = data.get('blur_intensity', 15)
    
    return jsonify({'success': True, 'focus_target': app_state['focus_target']})

@app.route('/api/process_video', methods=['POST'])
def process_video():
    """Process entire video with focus effect"""
    data = request.json
    
    input_path = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
    output_path = os.path.join(OUTPUT_FOLDER, 'processed_video.mp4')
    
    if not os.path.exists(input_path):
        return jsonify({'error': 'No video uploaded'}), 400
    
    focus_target = data.get('focus_target', app_state['focus_target'])
    blur_intensity = data.get('blur_intensity', app_state['blur_intensity'])
    
    # Process video
    result = processor.process_video(
        input_path, 
        output_path, 
        focus_target=focus_target,
        blur_intensity=blur_intensity
    )
    
    return jsonify(result)

@app.route('/api/stream_processed')
def stream_processed():
    """Stream processed video frames"""
    input_path = os.path.join(UPLOAD_FOLDER, 'input_video.mp4')
    
    if not os.path.exists(input_path):
        return jsonify({'error': 'No video uploaded'}), 400
    
    def generate_frames():
        cap = cv2.VideoCapture(input_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            detections = detector.detect(frame)
            
            # Track
            tracked = tracker.update(detections, frame)
            
            # Apply blur if focus target set
            if app_state['focus_target']:
                frame = blur_engine.apply_focus_blur(
                    frame, tracked, 
                    app_state['focus_target'],
                    app_state['blur_intensity']
                )
            
            # Draw detections
            if app_state['show_detections']:
                frame = detector.draw_detections(frame, detections, tracked)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/get_processed_video')
def get_processed_video():
    """Download processed video"""
    output_path = os.path.join(OUTPUT_FOLDER, 'processed_video.mp4')
    if os.path.exists(output_path):
        return send_file(output_path, mimetype='video/mp4')
    return jsonify({'error': 'No processed video available'}), 404

# ========== WebSocket Events ==========

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to FocusAI Backend'})

@socketio.on('frame')
def handle_frame(data):
    """Process frame in real-time via WebSocket"""
    try:
        frame_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect
        detections = detector.detect(frame)
        
        # Track
        tracked = tracker.update(detections, frame)
        
        # Apply blur
        if app_state['focus_target']:
            frame = blur_engine.apply_focus_blur(
                frame, tracked,
                app_state['focus_target'],
                app_state['blur_intensity']
            )
        
        # Draw detections
        frame = detector.draw_detections(frame, detections, tracked)
        
        # Encode and send back
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        emit('processed_frame', {
            'frame': f'data:image/jpeg;base64,{frame_b64}',
            'detections': detections,
            'tracked': tracked
        })
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('set_focus')
def handle_set_focus(data):
    app_state['focus_target'] = data.get('target')
    app_state['blur_intensity'] = data.get('blur_intensity', 15)
    emit('focus_updated', app_state)

@socketio.on('update_settings')
def handle_settings(data):
    app_state.update(data)
    emit('settings_updated', app_state)

if __name__ == '__main__':
    print("\n?? FocusAI Backend - Cricket Edition")
    print("=" * 40)
    print(f"Server: http://localhost:5000")
    print(f"Model: YOLOv8n")
    print("=" * 40 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
