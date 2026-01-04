import os
import json
import time
from ultralytics import YOLO
from datetime import datetime

def run_master_benchmark():
    # 1. 경로 설정
    MODEL_DIR = '/home/risexavier01/Downloads/models'
    DATA_PATH = 'data.yaml'  # mAP 측정용
    IMAGE_DIR = '/home/risexavier01/Downloads/test/images' # FPS 측정용 실제 이미지 폴더

    model_files = [
        'yolov8n_fp16.engine', 'yolov8s_fp16.engine',
        'yolov8m_fp16.engine', 'yolov8l_fp16.engine',
        'yolov8n_int8.engine', 'yolov8s_int8.engine',
        'yolov8m_int8.engine', 'yolov8l_int8.engine'
    ]

    # 이미지 리스트 준비
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_list = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                  if f.lower().endswith(valid_extensions)]

    if not image_list:
        print(f"❌ '{IMAGE_DIR}'에 이미지가 없습니다. FPS 측정이 불가능합니다.")
        return

    final_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"--- [지능 + 속도] 통합 벤치마크 시작 (대상: {len(model_files)}종) ---")

    for filename in model_files:
        model_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(model_path):
            continue

        print(f"\n🔍 [분석 중] {filename}...")

        try:
            model = YOLO(model_path, task="detect")

            # --- PART 1. 지능 지표 측정 (mAP, P, R, F1) ---
            val_results = model.val(
                data=DATA_PATH, split='test', device=0,
                plots=False, save_json=False, verbose=False
            )

            res_dict = val_results.results_dict
            p = res_dict.get('metrics/precision(B)', 0)
            r = res_dict.get('metrics/recall(B)', 0)
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

            # --- PART 2. 속도 지표 측정 (FPS) ---
            # Warm-up
            model.predict(image_list[0], verbose=False)

            start_time = time.time()
            for img_path in image_list:
                model.predict(img_path, verbose=False, device=0)
            end_time = time.time()

            total_duration = end_time - start_time
            fps = len(image_list) / total_duration

            # --- PART 3. 데이터 통합 (총 6개 항목) ---
            final_results[filename] = {
                "FPS": round(fps, 2),
                "mAP50": round(res_dict.get('metrics/mAP50(B)', 0), 4),
                "mAP50-95": round(res_dict.get('metrics/mAP50-95(B)', 0), 4),
                "Precision": round(p, 4),
                "Recall": round(r, 4),
                "F1-Score": round(f1, 4)
            }

            print(f"✅ {filename} 완료 | FPS: {fps:.2f} | mAP50: {final_results[filename]['mAP50']} | F1: {f1:.4f}")

        except Exception as e:
            print(f"❌ {filename} 분석 중 오류 발생: {e}")

    # 2. 결과 저장 및 출력
    save_path = f'master_benchmark_{timestamp}.json'
    with open(save_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    print("\n" + "="*60)
    print(f"📊 벤치마크 종료! 결과 저장 완료: {save_path}")
    print("="*60)
    print(json.dumps(final_results, indent=4))

if __name__ == "__main__":
    run_master_benchmark()
