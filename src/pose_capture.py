import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import deque

# ==========================================
# 1. 初始化模型與設定
# ==========================================
model = YOLO("models/yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

# 結合時序平滑化，避免單一幀的雜訊導致狀態閃爍
window_size = 15
status_history = deque(maxlen=window_size)

# 定義骨架關鍵點的索引
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10


# ==========================================
# 2. 統整姿態與迷因判斷邏輯
# ==========================================
def detect_action(keypoints):
    """
    結合迷因與辦公室姿態的統一判斷函數。
    依序判斷優先權：迷因動作 > 伸懶腰 > 打瞌睡 > 正常
    """
    if len(keypoints) == 0:
        return "No Person"

    kp = keypoints[0]
    nose = kp[NOSE]
    l_sh, r_sh = kp[L_SHOULDER], kp[R_SHOULDER]
    l_wr, r_wr = kp[L_WRIST], kp[R_WRIST]

    # 確保所有需要的關鍵點都有被準確偵測 (> 0.5 置信度)
    if any(pt[2] < 0.5 for pt in [nose, l_sh, r_sh, l_wr, r_wr]):
        return "Working / Normal"

    # --- 優先級 1: T-Pose (統治之姿) ---
    y_aligned = abs(l_wr[1] - l_sh[1]) < 60 and abs(r_wr[1] - r_sh[1]) < 60
    x_spread = l_wr[0] > l_sh[0] + 50 and r_wr[0] < r_sh[0] - 50
    if y_aligned and x_spread:
        return "T-POSE DOMINANCE!"

    # --- 優先級 2: Dab (嘻哈動作) ---
    dist_l_nose = math.hypot(l_wr[0] - nose[0], l_wr[1] - nose[1])
    dist_r_nose = math.hypot(r_wr[0] - nose[0], r_wr[1] - nose[1])

    if (dist_r_nose < 80 and l_wr[1] < l_sh[1] and l_wr[0] > l_sh[0] + 80) or (
        dist_l_nose < 80 and r_wr[1] < r_sh[1] and r_wr[0] < r_sh[0] - 80
    ):
        return "DAB ON EM!"

    # --- 優先級 3: 伸懶腰 (Stretching) ---
    if (l_wr[1] < nose[1] - 50) or (r_wr[1] < nose[1] - 50):
        return "Stretching"

    # --- 優先級 4: 打瞌睡 (Dozing) ---
    shoulder_avg_y = (l_sh[1] + r_sh[1]) / 2
    if nose[1] > shoulder_avg_y - 20:
        return "Dozing"

    # --- 預設狀態 ---
    return "Working / Normal"


# ==========================================
# 3. 主循環：即時影像處理
# ==========================================
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
        current_status = detect_action(keypoints)

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
        color = (0, 0, 255)  # 紅色
    elif smoothed_status == "Stretching":
        color = (255, 165, 0)  # 橘色
    elif smoothed_status in ["T-POSE DOMINANCE!", "DAB ON EM!"]:
        color = (0, 255, 255)  # 黃色
    else:
        color = (0, 255, 0)  # 綠色

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
