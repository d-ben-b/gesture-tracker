import sys
import cv2
import torch
import time
from ultralytics import YOLO

class VisionDebugger:
    def __init__(self, model_path="yolov8n-pose.pt"):
        self.model_path = model_path
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.cap = None

    def run_environment_diagnostics(self):
        """第一步：全面診斷系統環境"""
        print("\n" + "="*50)
        print("🔍 [系統環境診斷報告]")
        print("="*50)
        
        # 1. 檢查 Python 資訊
        print(f"▶ Python 版本: {sys.version.split(' ')[0]}")
        
        # 2. 檢查 OpenCV 與 GStreamer
        build_info = cv2.getBuildInformation()
        gst_support = "YES" if "GStreamer" in build_info and "YES" in build_info.split("GStreamer")[1].split("\n")[0] else "NO"
        print(f"▶ OpenCV 版本: {cv2.__version__}")
        print(f"▶ OpenCV GStreamer 支援: {gst_support}")
        
        if gst_support == "NO":
            print("❌ 錯誤：目前的 cv2.so 不支援 GStreamer，請確認虛擬環境設定。")
            sys.exit(1)

        # 3. 檢查 PyTorch 與 CUDA
        print(f"▶ PyTorch 版本: {torch.__version__}")
        print(f"▶ CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"▶ CUDA 裝置名稱: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ 警告：CUDA 不可用。模型將使用 CPU 執行，速度會非常慢。")
        print("="*50 + "\n")

    def build_gstreamer_pipeline(self):
        """建構專為 Jetson + Arducam 設計的 CSI 相機管線"""
        # 根據你之前的日誌，相機支援 2304x1296 @ 55fps
        capture_width = 2304
        capture_height = 1296
        display_width = 640
        display_height = 480
        framerate = 30
        flip_method = 2 # 0:無翻轉, 2:翻轉180度 (依相機安裝方向調整)

        return (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
        )

    def initialize_hardware(self):
        """第二步：初始化相機與 AI 模型"""
        print("⏳ 正在啟動 CSI 相機與 GStreamer 管線...")
        gst_str = self.build_gstreamer_pipeline()
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("❌ 錯誤：無法透過 GStreamer 開啟相機。請檢查相機排線或 nvargus-daemon 狀態。")
            sys.exit(1)
        print("✅ 相機初始化成功！")

        print("📸 正在擷取測試影像 (debug_frame.jpg)...")
        ret, frame = self.cap.read()
        if ret:
            # 儲存一張測試圖檔，幫助你免開視窗就能確認「紫色色偏」是否已解決
            cv2.imwrite("debug_frame.jpg", frame)
            print("✅ 測試影像已儲存。請檢查專案目錄下的 debug_frame.jpg 以確認色彩是否正常。")
        else:
            print("⚠️ 警告：無法讀取相機畫面。")

        print(f"🧠 正在將 YOLO 模型載入至 {self.device}...")
        try:
            self.model = YOLO(self.model_path)
            # 預熱模型以避免第一幀卡頓
            dummy_frame = torch.zeros((1, 3, 480, 640)).to(self.device) if self.device != 'cpu' else None
            print("✅ 模型載入完成！")
        except Exception as e:
            print(f"❌ 載入模型時發生錯誤: {e}")
            sys.exit(1)

    def start_stream(self):
        """第三步：執行主迴圈"""
        print("\n🚀 開始即時推論！(按 'q' 結束)")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 無法獲取影像幀，退出迴圈。")
                break

            # 進行姿態辨識推論
            results = self.model.predict(frame, device=self.device, half=(self.device != 'cpu'), verbose=False)
            
            # 將辨識結果繪製到畫面上
            annotated_frame = results[0].plot()

            # 顯示畫面
            cv2.imshow("Gesture Tracker Debugger", annotated_frame)

            # 按下 'q' 鍵退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        """第四步：安全釋放資源"""
        print("\n🧹 正在清理硬體資源...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ 程式已安全結束。")


def main():
    # 實例化我們的除錯器，並指定模型路徑
    # 請確保 'models/yolov8n-pose.pt' 存在，或者改成你實際的路徑
    debugger = VisionDebugger(model_path="models/yolov8n-pose.pt")
    
    try:
        # 1. 發散檢測：印出所有系統參數
        debugger.run_environment_diagnostics()
        
        # 2. 設備初始化：啟動相機並載入模型
        debugger.initialize_hardware()
        
        # 3. 收斂執行：進入影像串流與 AI 推論迴圈
        debugger.start_stream()
        
    except KeyboardInterrupt:
        print("\n🛑 接收到中斷指令 (Ctrl+C)，準備退出程式...")
        debugger.cleanup()
    except Exception as e:
        print(f"\n❌ 發生未預期的崩潰: {e}")
        debugger.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()