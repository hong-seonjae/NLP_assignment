# datasets.py
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):  # for BERT, GPT
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(list(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

class VocabDataset(Dataset):  # for RNN, LSTM
    def __init__(self, texts, labels, vocab=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer or (lambda x: x.split())
        self.max_len = max_len

        if vocab is None:
            self.vocab = self.build_vocab(texts)
        else:
            self.vocab = vocab

    def build_vocab(self, texts):
        token_set = set()
        for text in texts:
            token_set.update(self.tokenizer(text))
        vocab = {"<pad>": 0, "<unk>": 1}
        for i, token in enumerate(sorted(token_set), start=2):
            vocab[token] = i
        return vocab

    def encode(self, text):
        tokens = self.tokenizer(text)
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        return ids[:self.max_len] + [self.vocab["<pad>"]] * (self.max_len - len(ids))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encode(self.texts.iloc[idx]), dtype=torch.long)
        label = torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long)
        return {"input_ids": input_ids, "labels": label}

