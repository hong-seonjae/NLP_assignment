import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from utils import load_sst2_dataset
from config import *
from logger_callback import CustomLoggingCallback

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_ds, val_ds = load_sst2_dataset(tokenizer, max_len)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./results/full",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    logging_dir="./logs/full_tb",
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[CustomLoggingCallback(log_file="training_log_full.txt", tag="full")]
)

# 측정
torch.cuda.reset_peak_memory_stats()
start = time.time()
trainer.train()
end = time.time()

max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"[Full Fine-Tuning] Training time: {end - start:.2f} sec")
print(f"[Full Fine-Tuning] Max memory allocated: {max_mem:.2f} MB")
