import os
import sys
import json
import time


from ultralytics import YOLO

# 1. 환경 설정
MODEL_PATH = '/home/risexavier01/Downloads/yolov8n_fp32.engine' # 생성된 엔진 파일 경로
DATA_YAML = '/home/risexavier01/Downloads/data.yaml'        # 클래스 정보가 담긴 yaml 파일
TEST_IMAGES = '/home/risexaiver01/Downloads/test/images'      # 테스트 이미지 폴더 경로
SAVE_PATH = '/home/risexavier01/Downloads/test_results.json'

def evaluate_engine():
    # 2. 모델 로드
    # task="detect"를 명시해야 엔진 파일을 올바르게 인식합니다.
    model = YOLO(MODEL_PATH, task="detect")

    print(f"--- {MODEL_PATH} 성능 측정 시작 ---")

    # 3. 검증 실행 (Validation)
    # split='test'로 설정하여 테스트 폴더의 데이터 사용
    results = model.val(
        data=DATA_YAML,
        split='test',
        device=0,      # Xavier GPU 사용
        half=True      # FP16 엔진이므로 True
    )

    # 4. 성능 지표 추출
    # mAP50, mAP50-95, F1-score 등
    metrics = {
        "mAP50": round(results.results_dict['metrics/mAP50(B)'], 4),
        "mAP50-95": round(results.results_dict['metrics/mAP50-95(B)'], 4),
        "F1_score": round(float(results.box.f1.mean()), 4), # fitness는 보통 mAP와 F1의 가중 합
        "Precision": round(results.results_dict['metrics/precision(B)'], 4),
        "Recall": round(results.results_dict['metrics/recall(B)'], 4)
    }

    # 5. FPS 측정
    # 전처리 + 추론 + 후처리 시간을 합산하여 계산
    speed = results.speed # {'preprocess': ms, 'inference': ms, 'postprocess': ms}
    total_time_per_img = sum(speed.values()) # ms
    fps = 1000 / total_time_per_img

    metrics["FPS"] = round(fps, 2)
    metrics["inference_ms"] = round(speed['inference'], 2)

    # 6. JSON 저장
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(f"--- 측정 완료! 결과 저장됨: {SAVE_PATH} ---")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    evaluate_engine()
