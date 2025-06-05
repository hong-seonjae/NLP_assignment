# gpt_classifier.py
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

def get_gpt_model(num_labels=2):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT2는 pad_token이 없음 → eos_token 사용

    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id
    )
    return model, tokenizer