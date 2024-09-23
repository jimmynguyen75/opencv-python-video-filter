from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Xử lý từng khung hình từ webcam với tính năng chỉnh sửa ảnh
@socketio.on('video_frame')
def handle_video_frame(data):
    # Decode khung hình từ base64
    frame_data = base64.b64decode(data['frame'].split(',')[1])
    np_frame = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    # Áp dụng các tính năng chỉnh sửa ảnh dựa trên lựa chọn của người dùng
    if data['action'] == 'filter':
        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Lọc ảnh
    elif data['action'] == 'denoise':
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)  # Khử nhiễu
    elif data['action'] == 'brightness':
        brightness = int(data['brightness'])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, brightness)
        v = np.clip(v, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)  # Chỉnh độ sáng
    elif data['action'] == 'detect_bound':
        frame = cv2.Canny(frame, 100, 200)  # Phát hiện biên

    # Encode lại khung hình đã chỉnh sửa thành base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')
    frame_data_url = 'data:image/jpeg;base64,' + frame_encoded

    # Gửi khung hình đã xử lý lại cho client
    emit('processed_frame', frame_data_url)

if __name__ == '__main__':
    eventlet.monkey_patch()  # Eventlet yêu cầu phải có monkey patch
    socketio.run(app, host='0.0.0.0', port=5000)