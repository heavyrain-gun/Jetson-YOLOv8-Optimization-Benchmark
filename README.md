# 🛰️ Jetson-YOLOv8-Optimization-Benchmark

**NVIDIA Jetson AGX Xavier**와 **Orin Nano**를 활용한 YOLOv8 모델의 성능 최적화 및 벤치마크 프로젝트입니다.

## 📖 소개
본 프로젝트는 제한된 자원을 가진 Edge AI 환경에서 모델의 추론 속도를 극대화하기 위해 **TensorRT** 최적화 과정을 수행했습니다. PyTorch 모델을 TensorRT로 변환하고, 정밀도(FP16, INT8)에 따른 하드웨어별 성능 데이터를 비교 분석했습니다.

## 💻 실험 환경
- **Target HW**: NVIDIA Jetson AGX Xavier, Jetson Orin Nano
- **Target SW**: JetPack 5.1.5(AGX Xavier), JetPack 6.x.x(Orin Nano)
- **AI Model**: YOLOv8 (Detection)
- **Library**: JetPack SDK, TensorRT, PyTorch, ONNX

## 📊 벤치마크 결과 (Preview)
| Device | Precision | FPS | Latency (ms) |
| :--- | :--- | :--- | :--- |
| **Orin Nano** | FP16 | (입력예정) | (입력예정) |
| **AGX Xavier** | INT8 | (입력예정) | (입력예정) |

## 🛠️ 주요 수행 내용
1. **Model Export**: `.pt` → `.onnx` → `.engine` 변환 파이프라인 구축
2. **Quantization**: FP16 및 INT8 양자화 적용 및 성능 비교
3. **Environment**: 하드웨어별 최적의 JetPack 환경 구성 및 라이브러리 연동
