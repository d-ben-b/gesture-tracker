import cv2

# 1. Print and search the build configuration
build_info = cv2.getBuildInformation()

if "GStreamer" in build_info:
    print("--- GStreamer Build Block Found ---")
    # Extract only the relevant lines for scannability
    lines = build_info.split('\n')
    for line in lines:
        if "GStreamer" in line or "videoio" in line:
            print(line)
else:
    print("GStreamer reference not found in build info.")
