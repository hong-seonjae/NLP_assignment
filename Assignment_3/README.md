# BERT Fine-Tuning vs Prefix Tuning (SST-2 실험 비교)

이 프로젝트는 `bert-base-uncased` 모델을 기반으로 SST-2 감정 분류 태스크에 대해 Full Fine-Tuning과 Prefix Tuning (Soft Prompt Tuning)을 비교한 실험입니다. 성능, 파라미터 효율성, GPU 자원 사용량 등을 종합적으로 분석합니다.

---

## 프로젝트 개요

- 모델: `bert-base-uncased`
- 데이터셋: GLUE - SST-2
- 실험 방식:
  - Full Fine-Tuning: 전체 파라미터 학습
  - Prefix Tuning: 일부 prefix token만 학습 (`peft` 라이브러리 사용)
- 공통 설정:
  - Epochs: 10
  - Batch size: 16
  - Learning rate: 1e-6

---

## 코드 구성

| 파일명              | 설명 |
|---------------------|------|
| `config.py`         | 모델명, 학습률, 배치 사이즈 등의 하이퍼파라미터 설정 |
| `data_loader.py`    | SST-2 데이터셋 로딩 및 전처리 함수 정의 |
| `utils.py`          | accuracy metric 계산 함수 |
| `train_finetune.py` | Full Fine-Tuning 방식 학습 스크립트 |
| `train_prefix.py`   | Prefix Tuning 방식 학습 스크립트 (`peft` 활용) |

---

## 실험 결과 요약

| 항목               | Full Fine-Tuning | Prefix Tuning |
|--------------------|------------------|----------------|
| Accuracy           | 88.11%           | 88.07%         |
| Train Loss         | 0.0717           | 0.2704         |
| Trainable Params   | 109.48M          | 14.78M         |
| Max GPU Memory     | 2526 MB          | 1495 MB        |
| Training Time      | 8632s            | 6864s          |

---

## 성능 비교 시각화

### Accuracy, Train Loss, Trainable Params  
![image](https://github.com/user-attachments/assets/2d01bcbc-70b4-4464-9d84-99be222ce594)


> Accuracy는 거의 동일하지만, Train Loss는 Full Fine-Tuning이 빠르게 수렴하며 안정적인 성능을 보입니다. 반면 Prefix Tuning은 매우 적은 파라미터만 학습하며 효율적인 구조를 보입니다.

---

### GPU Memory & Training Time  
![image](https://github.com/user-attachments/assets/997a3186-4004-4353-85dc-2338e02cb3f1)


> Prefix Tuning은 Full Fine-Tuning 대비 약 40% 적은 메모리 사용량과 20% 더 빠른 학습 시간을 기록하였습니다. 리소스가 제한된 환경에 특히 적합합니다.

---

## 느낀 점

- 단순히 정확도만 보는 것이 아니라, 자원 효율성과 실용성을 함께 고려해야 함을 체감했습니다.
- Prefix Tuning은 적은 파라미터로도 성능을 유지하며, 실제 환경에 적용하기 적합한 전략임을 확인했습니다.
- 추후 Prompt Tuning, Adapter, BitFit 등 다양한 방식도 실험하여 상황별로 적절한 튜닝 전략을 찾아보고 싶습니다.

---

## 사용 라이브러리

- [Transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [peft](https://github.com/huggingface/peft)

---

## 작성자

20203164 홍선재  
국민대학교 소프트웨어학부  
자연어처리 과제 (2025년 1학기)
