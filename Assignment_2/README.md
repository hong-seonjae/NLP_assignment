# LoRA vs Full Fine-Tuning: SST-2 감정 분류 성능 비교

20203164 홍선재 / 자연어처리 과제 (2025년 1학기)

이 프로젝트는 SST-2 감정 분석(Sentiment Classification) 태스크를 대상으로, **LoRA(Low-Rank Adaptation)** 기반 파인튜닝과 **Full Fine-Tuning** 방식의 성능과 효율성을 정량적으로 비교하는 실험입니다.
실험은 Hugging Face Transformers 라이브러리를 기반으로 구성되었으며, 학습 로그 기록, 시각화, 추론 예시 등을 포함합니다.

---

## 프로젝트 구성

```
.
├── train_full.py                    # Full Fine-Tuning 학습 스크립트
├── train_lora.py                    # LoRA Fine-Tuning 학습 스크립트
├── inference.py                     # 학습된 LoRA 모델로 감정 예측 수행
├── config.py                        # 모델 및 학습 관련 하이퍼파라미터 설정
├── utils.py                         # SST-2 데이터 로딩 및 전처리 함수
├── logger_callback.py               # 학습 중 loss/accuracy 기록용 콜백 함수
├── training_log_full.txt            # Full Fine-Tuning 학습 로그
├── training_log_lora.txt            # LoRA 학습 로그
├── requirements.txt                 # 의존성 패키지 목록
├── lora_vs_full_finetuning_comparison.png   # 학습/평가 성능 비교 그래프
└── lora_vs_full_time_memory.png             # 자원 사용량 비교 그래프
```

---

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 학습 실행

```bash
# Full Fine-Tuning 실행
python train_full.py

# LoRA Fine-Tuning 실행
python train_lora.py
```

학습 결과는 `./results/full`, `./results/lora` 디렉토리에 저장됩니다.

### 3. 모델 추론 테스트

```bash
python inference.py
```

`inference.py`는 LoRA 모델을 불러와 예시 문장에 대해 감정 예측을 수행합니다.

---

## 실험 개요

* 모델: DistilBERT (base-uncased)
* 데이터: GLUE SST-2 (binary classification)
* 학습 epoch: 10
* 평가 지표: Accuracy, Training/Eval Loss
* 학습 시간 및 GPU 메모리 사용량은 `torch.cuda`를 통해 측정

---

## 시각화 결과

### 1. 학습 성능 비교 (Train/Eval Loss, Eval Accuracy)

![image](https://github.com/user-attachments/assets/f226a0ae-e3f2-4f05-8daa-640f5733ec20)


* Full Fine-Tuning은 빠르게 수렴하지만, Eval Loss가 점점 증가하여 과적합 양상을 보임
* LoRA는 상대적으로 완만하게 수렴하나 Eval Loss가 지속적으로 감소하며 일반화 성능이 더 안정적임

### 2. 학습 시간 및 GPU 메모리 사용량 비교

![image](https://github.com/user-attachments/assets/bcf81058-4c62-47d4-9b12-ff2532618adf)


| 방법               | 학습 시간 (sec) | 최대 메모리 사용량 (MB) |
| ---------------- | ----------- | --------------- |
| Full Fine-Tuning | 2565.11     | 2086.94         |
| LoRA             | 2275.25     | 1295.13         |

* LoRA는 Full Fine-Tuning 대비 약 13% 빠른 학습 시간과 38% 적은 GPU 메모리 사용량을 기록함

---

## 로그 분석 요약

### Full Fine-Tuning (`training_log_full.txt`)

* Train Loss는 0.03 이하까지 빠르게 감소
* Eval Loss는 0.28 → 0.56으로 증가하는 경향
* Eval Accuracy는 89% \~ 91%로 높지만 일관성이 부족
* 전형적인 과적합 현상상이며 표현력은 우수하나 일반화는 불안정

### LoRA (`training_log_lora.txt`)

* Train Loss는 천천히 감소하며 0.27 수준에서 수렴
* Eval Loss는 학습 내내 안정적으로 감소
* Eval Accuracy는 84.6% → 88.0%까지 점진적 향상
* 수렴 속도는 느리지만 일반화 성능이 우수하며, 안정적인 학습 패턴을 보임

---

## 하이퍼파라미터 설정 (`config.py`)

```python
model_name = "distilbert-base-uncased"
batch_size = 32
epochs = 10
lr = 1e-5
max_len = 128
```

---

## 결론 요약

| 비교 항목        | Full Fine-Tuning | LoRA            |
| ------------ | ---------------- | --------------- |
| 수렴 속도        | 빠름               | 느림              |
| 성능(Accuracy) | 높음               | 점진적 향상, 안정적     |
| Eval Loss    | 증가 (과적합)         | 지속적 감소          |
| GPU 메모리 사용량  | 높음               | 낮음 (약 38% 절감)   |
| 학습 시간        | 다소 길음            | 더 빠름 (약 13% 단축) |

* Full Fine-Tuning은 성능 면에서 우수하지만, 리소스를 많이 요구하고 과적합 가능성이 있음
* LoRA는 약간의 성능 손해를 감수하면서도 자원 효율성과 일반화 성능에서 매우 우수한 결과를 보임

---

## 작성자

* 이름: 홍선재
* 소속: 국민대학교 소프트웨어학부
* 과목: 자연어처리 (NLP)
