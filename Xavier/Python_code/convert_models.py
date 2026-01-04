from ultralytics import YOLO
import os

# 경로를 명확하게 지정합니다
# 파일명 / 모델명 확인(l,m,s,n)
MODEL_PATH = "/home/risexavier01/Downloads/models/yolov8l_fp32.pt"

def convert():
    # 파일이 존재하는지 먼저 확인 (에러 방지)
    if not os.path.exists(MODEL_PATH):
        print(f"에러: {MODEL_PATH} 파일을 찾을 수 없습니다.")
        return

    print("--- 변환 시작: PT to TensorRT (FP16) ---")
    model = YOLO(MODEL_PATH)

    # export 실행
    model.export(format="engine", device=0, half=True)
    print("--- 변환 완료 ---")

if __name__ == "__main__":
    convert()
