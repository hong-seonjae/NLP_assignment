# Sentiment Analysis with RNN, LSTM, BERT, GPT-2

**20203164 홍선재 - 자연어처리 과제 (NLP Assignment)**

이 프로젝트는 다양한 딥러닝 모델(RNN, LSTM, BERT, GPT-2)을 활용하여 감정 분석(Sentiment Classification) 성능을 비교한 실험입니다.
PyTorch와 Hugging Face Transformers 기반으로 구성되어 있으며 학습·평가·시각화·예측까지 모두 자동화되어 있습니다..

---

## 📁 프로젝트 구조

```
.
├── main.py                     # 전체 실험 실행 스크립트 (entry point)
├── datasets.py                 # 모델별 Dataset 정의 (SentimentDataset, VocabDataset)
├── bert_classifier.py          # BERT 모델 정의
├── gpt_classifier.py           # GPT-2 모델 정의
├── lstm_classifier.py          # LSTM 모델 정의
├── rnn_classifier.py           # RNN 모델 정의
├── train.csv / test.csv        # Kaggle 감정 분석 데이터셋
├── plot/                       # 학습 결과 시각화 이미지 저장
├── nlp.ipynb                   # 본격적인 실험 전에 연습용 ipynb file
└── NLP_assignmnet_Analysis.pdf # 상세한 정보를 포함한 분석 보고서

```

---

## 실행 방법

```bash
python3 main.py

```

- `main.py`만 실행하면 전체 실험이 자동으로 시작됩니다.
- 다음 과정을 자동으로 수행합니다:
    1. 데이터 로딩 및 전처리
    2. 모델 학습 (RNN, LSTM, BERT, GPT-2) — epoch : 10
    3. 성능 평가 및 시각화 이미지 저장
    4. 테스트 문장에 대한 감정 예측 출력

**Dataset 클래스**는 모두 `datasets.py`에 정의되어 있으며 사용자 개입 없이 각 모델에 맞는 전처리가 자동 적용됩니다.

---

## 📊 데이터셋

- 출처: [Kaggle - Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
- 구성: `train.csv`, `test.csv`
- 레이블: `positive`, `negative` (（neutral은 제외）
- 전처리 방식:
    - **SentimentDataset**: BERT, GPT-2용 — Hugging Face Tokenizer 기반
    - **VocabDataset**: RNN, LSTM용 — 사용자 정의 Vocab 기반

---

## 사용된 모델

| 모델명 | 설명 |
| --- | --- |
| **RNN** | 가장 기본적인 순환시간량 |
| **LSTM** | 장기 의존성 문제를 해결한 RNN |
| **BERT** | 양방향 Transformer 인코더 (`kykim/bert-kor-base`) |
| **GPT-2** | 단방향향 Transformer 디코더 (`gpt2`) |

---

## 📄 문서 및 참고 자료

- [`NLP_assignmnet_Analysis.pdf`](./NLP_assignment/Assignment_1/NLP_assignmnet_Analysis.pdf):  
  보다 상세한 실험 결과와 분석 내용을 확인하고자 할 경우, 첨부된 PDF 문서를 클릭하여 확인할 수 있습니다.


---

## 📈 실험 결과 예시

`plot/` 폴더에 저장되는 그래프 예시:

- `bert_training_plot.png`
- `gpt_training_plot.png`
- `lstm_training_plot.png`
- `rnn_training_plot.png`

실행 시 출력되는 감정 예측 예시:

```
문장: 이 영화는 정말 감독적이었어요.
예측: 그정 (확률: 0.92)

문장: 오늘 너무 피곤하고 짜증나.
예측: 부정 (확률: 0.87)

```

---

## 참고 사항

- GPT-2는 pad token이 없어 `eos_token`을 대신 사용합니다.
- 모든 모델은 동일한 train/validation split 환경에서 비교됩니다.
- 평가 지표는 Accuracy와 Training Loss입니다.

---

## 작성자

| 이름 | 소속 |
| --- | --- |
| 홍선재 | 국민대학교 소프트웨어학부 4학년 |
- 지도교수: 김장호 교수님
- 과반: 자연어처리 (NLP)
- 학번: 20203164
