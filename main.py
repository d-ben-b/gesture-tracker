import cv2

# 1. Print and search the build configuration
build_info = cv2.getBuildInformation()
# 測試一個最簡單的 GStreamer 管線
test_str = "videotestsrc ! videoconvert ! appsink"
cap = cv2.VideoCapture(test_str, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("最簡單的管線都開不了，請檢查系統 GStreamer 安裝")
else:
    print("管線正常！問題出在你的原始 gstream_str 設定")
if "GStreamer" in build_info:
    print("--- GStreamer Build Block Found ---")
    # Extract only the relevant lines for scannability
    lines = build_info.split('\n')
    for line in lines:
        if "GStreamer" in line or "videoio" in line:
            print(line)
else:
    print("GStreamer reference not found in build info.")
