import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import torch.nn.functional as F
from tqdm import tqdm

from datasets import SentimentDataset, VocabDataset
from bert_classifier import CustomBertClassifier
from lstm_classifier import LSTMClassifier
from rnn_classifier import RNNClassifier
from gpt_classifier import get_gpt_model
from transformers import BertTokenizer

# 데이터 로드 및 전처리
df = pd.read_csv("train.csv", encoding="ISO-8859-1")[["text", "sentiment"]]
df = df[df["sentiment"] != "neutral"]
df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 함수
def train(model, loader, optimizer, epoch, total_epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch [{epoch+1}/{total_epoch}]"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

# 평가 함수
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    print(classification_report(all_labels, all_preds, target_names=["부정", "긍정"]))
    return sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

# 전체 실행 함수
def run(name, model, train_loader, val_loader, tokenizer=None, total_epoch=10):
    print(f"\n===== [{name}] 학습 시작 =====")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    train_losses, val_accuracies = [], []

    for epoch in range(total_epoch):
        loss = train(model, train_loader, optimizer, epoch, total_epoch)
        acc = evaluate(model, val_loader)
        train_losses.append(loss)
        val_accuracies.append(acc)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    # 시각화 저장
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker='o')
    plt.title(f"{name} - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, marker='o')
    plt.title(f"{name} - Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(f"{name.lower()}_training_plot.png")
    print(f"[시각화 저장 완료] → {name.lower()}_training_plot.png")

    # 문장 예측
    test_sentences = [
        "이 영화는 정말 감동적이었어요.",
        "오늘 너무 피곤하고 짜증나.",
        "차 앞에다 대놨으니까 빨리 나오라했잖아",
        "아 나 그냥 집갈래"
    ]
    labels = ["부정", "긍정"]

    if tokenizer is not None:
        print("\n[테스트 문장 예측 결과]")
        for text in test_sentences:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
            with torch.no_grad():
                logits = model(**tokens).logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            print(f"문장: {text}\n예측: {labels[pred]} (확률: {probs[0][pred]:.4f})\n")
    elif name in ["LSTM", "RNN"]:
        print("\n[테스트 문장 예측 결과]")
        vocab = train_loader.dataset.vocab
        tokenizer_fn = lambda x: x.split()

        def encode(text, vocab, max_len=128):
            tokens = tokenizer_fn(text)
            ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
            return ids[:max_len] + [vocab["<pad>"]] * (max_len - len(ids))

        for text in test_sentences:
            input_ids = torch.tensor([encode(text, vocab)], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            print(f"문장: {text}\n예측: {labels[pred]} (확률: {probs[0][pred]:.4f})\n")

# 모델 정의 및 실행
bert_tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")
bert_model = CustomBertClassifier()
gpt_model, gpt_tokenizer = get_gpt_model()
lstm_data = VocabDataset(train_texts, train_labels)
lstm_model = LSTMClassifier(vocab_size=len(lstm_data.vocab))
rnn_model = RNNClassifier(vocab_size=len(lstm_data.vocab))

run("BERT", bert_model,
    DataLoader(SentimentDataset(train_texts, train_labels, bert_tokenizer), batch_size=16, shuffle=True),
    DataLoader(SentimentDataset(val_texts, val_labels, bert_tokenizer), batch_size=16),
    tokenizer=bert_tokenizer)

run("GPT", gpt_model,
    DataLoader(SentimentDataset(train_texts, train_labels, gpt_tokenizer), batch_size=16, shuffle=True),
    DataLoader(SentimentDataset(val_texts, val_labels, gpt_tokenizer), batch_size=16),
    tokenizer=gpt_tokenizer)

run("LSTM", lstm_model,
    DataLoader(lstm_data, batch_size=16, shuffle=True),
    DataLoader(VocabDataset(val_texts, val_labels, vocab=lstm_data.vocab), batch_size=16))

run("RNN", rnn_model,
    DataLoader(VocabDataset(train_texts, train_labels, vocab=lstm_data.vocab), batch_size=16, shuffle=True),
    DataLoader(VocabDataset(val_texts, val_labels, vocab=lstm_data.vocab), batch_size=16))
