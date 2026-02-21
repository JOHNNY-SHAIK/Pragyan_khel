"""
FocusAI CPU-TURBO Backend v8.0
Maximum CPU Performance
"""

from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2
import os
import logging
from detector_stream import get_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detector = get_detector()


def generate_frames():
    """Ultra-fast streaming"""
    while True:
        try:
            frame = detector.get_next_frame()
            if frame is None:
                continue
            
            # SPEED: Low quality JPEG
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        except GeneratorExit:
            break
        except:
            continue


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'version': '8.0-cpu-turbo'})


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    f = request.files['video']
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    
    fps, frames = detector.load_video(path)
    return jsonify({'success': True, 'fps': fps, 'frames': frames})


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/play', methods=['POST'])
def play():
    detector.start_processing()
    return jsonify({'success': True})


@app.route('/api/pause', methods=['POST'])
def pause():
    detector.stop_processing()
    return jsonify({'success': True})


@app.route('/api/set_focus', methods=['POST'])
def set_focus():
    data = request.get_json()
    x, y = int(data.get('x', 0)), int(data.get('y', 0))
    
    result = detector.set_focus_by_click(x, y)
    if result:
        return jsonify({'success': True, 'track_id': result['track_id']})
    return jsonify({'success': False}), 404


@app.route('/api/clear_focus', methods=['POST'])
def clear_focus():
    detector.clear_focus()
    return jsonify({'success': True})


@app.route('/api/set_blur', methods=['POST'])
def set_blur():
    data = request.get_json()
    intensity = int(data.get('intensity', 21))
    result = detector.set_blur_intensity(intensity)
    return jsonify({'success': True, 'blur': result})


@app.route('/api/status')
@app.route('/api/settings')
def status():
    return jsonify(detector.get_status())


if __name__ == '__main__':
    logger.info('╔═══════════════════════════════════════════════╗')
    logger.info('║   FOCUSAI CPU-TURBO v8.0 - NO GPU REQUIRED   ║')
    logger.info('║   Target: 20-30 FPS on CPU                    ║')
    logger.info('║   http://localhost:5000                       ║')
    logger.info('╚═══════════════════════════════════════════════╝')
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)