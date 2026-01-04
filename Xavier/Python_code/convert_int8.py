from ultralytics import YOLO

# 1. 원본 PT 모델 로드
# 모델명 꼭 확인
model = YOLO("/home/risexavier01/Downloads/models/yolov8n_fp32.pt")

# 2. INT8 변환 실행
# data: 캘리브레이션에 사용할 데이터셋 설정 파일
# int8: True로 설정
# imgsz: 학습 시와 동일한 크기
model.export(
    format="engine",
    int8=True,
    data="data.yaml",  # val 경로가 포함된 yaml 파일
    device=0,
    simplify=True             # ONNX 그래프 단순화 (에러 방지)
)
# 3. [추가] 이름 변경 로직
# 기본 생성된 파일명(..fp32.engine)을 원하는 이름(..int8.engine)으로 변경
new_path = export_path.replace("_fp32.engine", "_int8.engine")
os.rename(export_path, new_path)

print(f"✅ 변환 및 이름 변경 완료: {new_path}")
