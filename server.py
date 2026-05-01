from flask import Flask
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import numpy as np
import cv2
import logging

# ---------------------------------
# إيقاف اللوجات المزعجة
# ---------------------------------
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ---------------------------------
# إنشاء السيرفر
# ---------------------------------
app = Flask(__name__)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"   # 🔥 بدون eventlet
)

# ---------------------------------
# تحميل YOLO
# ---------------------------------
model = YOLO("yolov8n.pt")
model.fuse()

print("🔥 YOLO Server Running (Threading Mode)")

# ---------------------------------
# استقبال الفريم من Unity
# ---------------------------------
@socketio.on('frame')
def handle_frame(data):
    try:
        # تأكد من نوع البيانات
        if isinstance(data, list):
            data = bytes(data)

        if not isinstance(data, (bytes, bytearray)):
            print("❌ Invalid data type:", type(data))
            return

        img_array = np.frombuffer(data, np.uint8)

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ Failed to decode image")
            return

        results = model(img)

        annotated = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated)

        socketio.emit('response', buffer.tobytes())

    except Exception as e:
        print("🔥 ERROR INSIDE handle_frame:", e)

# ---------------------------------
# تشغيل السيرفر
# ---------------------------------
if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000
    )