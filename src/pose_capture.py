import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from action_detector import ActionDetector

# OPTIMIZATION 1: Offload resizing and format conversion to Jetson GPU via nvvidconv
# Changing output format directly to BGRx saves significant CPU cycles
gstreamer_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=true max-buffers=1 emit-signals=true sync=false"
)


# OPTIMIZATION 2: Load the compiled TensorRT Engine (.engine) instead of PyTorch (.pt)
model = YOLO("models/yolov8n-pose.pt", task="pose")

cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
detector = ActionDetector()

window_size = 15
status_history = deque(maxlen=window_size)

# Mapping colors cleanly via a dictionary instead of long if-elif blocks
COLOR_MAP = {
    "Dozing": (0, 0, 255),
    "Stretching": (255, 165, 0),
    "T-POSE DOMINANCE!": (0, 255, 255),
    "DAB ON EM!": (0, 255, 255),
    "114514": (255, 0, 255),
    "KILLER QUEEN!": (203, 150, 255)
}

if not cap.isOpened():
    print("Error: Unable to open camera feed via GStreamer.")
    exit()

print("Pipeline initialized successfully using TensorRT acceleration!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OPTIMIZATION 3: Extract fast 3-channel BGR from the fast BGRx matrix efficiently
    frame = frame[:, :, :3]
    
    # OPTIMIZATION 4: Force execution on GPU (device=0) and pass half precision tensor configurations
    results = model.predict(frame, device=0, half=True, verbose=False)

    current_status = "Unknown"
    if results[0].keypoints is not None:
        # OPTIMIZATION 5: Keep data on GPU as long as possible before sending keypoints to CPU
        keypoints = results[0].keypoints.data.cpu().numpy()
        current_status = detector.detect(keypoints)

    status_history.append(current_status)
    smoothed_status = max(set(status_history), key=status_history.count) if status_history else current_status

    # Annotation and Rendering
    annotated_frame = results[0].plot()
    color = COLOR_MAP.get(smoothed_status, (0, 255, 0))

    cv2.putText(
        annotated_frame,
        f"Status: {smoothed_status}",
        (30, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        color,
        3,
    )

    cv2.imshow("Combined Action & Meme Tracker", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()