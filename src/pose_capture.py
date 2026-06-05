import cv2
from ultralytics import YOLO
from collections import deque
from action_detector import ActionDetector

model = YOLO("models/yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

# 初始化動作總管
detector = ActionDetector()

# 結合時序平滑化，避免單一幀的雜訊導致狀態閃爍
window_size = 15
status_history = deque(maxlen=window_size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = model.predict(frame, verbose=False)

    # 提取特徵與判斷
    current_status = "Unknown"
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()
        # 透過總管來進行判斷！
        current_status = detector.detect(keypoints)

    # 時序平滑化處理
    status_history.append(current_status)
    if len(status_history) > 0:
        smoothed_status = max(set(status_history), key=status_history.count)
    else:
        smoothed_status = current_status

    # 視覺化輸出
    annotated_frame = results[0].plot()

    # 根據不同狀態設定顏色
    if smoothed_status == "Dozing":
        color = (0, 0, 255)  
    elif smoothed_status == "Stretching":
        color = (255, 165, 0)  
    elif smoothed_status in ["T-POSE DOMINANCE!", "DAB ON EM!"]:
        color = (0, 255, 255) 
    elif smoothed_status == "YAJUSENPAI!!!":
        color = (255, 0, 255)
    elif smoothed_status == "KILLER QUEEN!":
        color = (203, 150, 255)
    else:
        color = (0, 255, 0)  

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