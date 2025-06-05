#  bert_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class CustomBertClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_size=768, num_labels=2):
        super(CustomBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("kykim/bert-kor-base")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.to(logits.device))
        return type('Output', (), {'loss': loss, 'logits': logits})()
