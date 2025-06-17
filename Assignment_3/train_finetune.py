import os
import time
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from config import *
from data_loader import get_sst2_dataset
from utils import compute_metrics

os.environ.pop("HF_ENDPOINT", None)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
train_dataset, eval_dataset = get_sst2_dataset()

args = TrainingArguments(
    output_dir=f"{output_dir}/finetune",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    save_strategy="epoch",
    learning_rate=lr,
    logging_dir=f"{log_dir}/finetune",
    report_to="none",
    logging_steps=10,
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

start = time.time()
trainer.train()
end = time.time()
print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
print(f"Training Time: {end - start:.2f}s")
if torch.cuda.is_available():
    print(f"Max GPU Memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")