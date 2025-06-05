#  lstm_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_labels=2):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        logits = self.fc(hidden[-1])
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return type('Output', (), {'loss': loss, 'logits': logits})()