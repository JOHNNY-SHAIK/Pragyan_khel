from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import logging
from detector_stream import get_detector
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detector = get_detector()


def create_blank_frame():
    '''Create a blank frame with message'''
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, 'Waiting for video...', (150, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return blank


def generate_frames():
    '''Generator function for MJPEG streaming'''
    logger.info("Video stream started")
    
    while True:
        try:
            frame = detector.get_next_frame()
            
            if frame is None:
                frame = create_blank_frame()
            
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay
            time.sleep(0.01)
            
        except GeneratorExit:
            logger.info("Stream closed by client")
            break
        except Exception as e:
            logger.error(f"Stream error: {e}")
            continue


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'detector': 'loaded',
        'model': 'YOLOv8-s-seg',
        'version': '3.5-streaming'
    })


@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400
        
        file = request.files['video']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        fps, total_frames = detector.load_video(filepath)
        
        logger.info(f'Uploaded: {file.filename}')
        
        return jsonify({
            'message': 'Video uploaded successfully',
            'filepath': filepath,
            'metadata': {
                'fps': fps,
                'frames': total_frames
            }
        })
    except Exception as e:
        logger.error(f'Upload error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/video_feed')
def video_feed():
    '''MJPEG streaming endpoint'''
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/play', methods=['POST'])
def play_video():
    detector.start_processing()
    return jsonify({'success': True, 'message': 'Processing started'})


@app.route('/api/pause', methods=['POST'])
def pause_video():
    detector.stop_processing()
    return jsonify({'success': True, 'message': 'Processing paused'})


@app.route('/api/set_focus', methods=['POST'])
def set_focus():
    try:
        data = request.get_json()
        x = int(data.get('x'))
        y = int(data.get('y'))
        
        focused = detector.set_focus_by_click(x, y)
        
        if focused:
            return jsonify({
                'success': True,
                'focused_object': {
                    'track_id': focused['track_id'],
                    'class': focused['class_name'],
                    'confidence': focused['confidence']
                }
            })
        
        return jsonify({'success': False, 'message': 'No object at position'}), 404
    
    except Exception as e:
        logger.error(f'Set focus error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear_focus', methods=['POST'])
def clear_focus():
    detector.clear_focus()
    return jsonify({'success': True})


@app.route('/api/set_blur', methods=['POST'])
def set_blur():
    try:
        data = request.get_json()
        intensity = int(data.get('intensity', 35))
        detector.set_blur_intensity(intensity)
        return jsonify({'success': True, 'blur_intensity': detector.blur_intensity})
    except Exception as e:
        logger.error(f'Set blur error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify({
        'blur_intensity': detector.blur_intensity,
        'focused_track_id': detector.focused_track_id,
        'is_playing': detector.is_playing
    })


if __name__ == '__main__':
    logger.info('╔════════════════════════════════════════════╗')
    logger.info('║   FOCUSAI V3.5 STREAMING BACKEND          ║')
    logger.info('╚════════════════════════════════════════════╝')
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
