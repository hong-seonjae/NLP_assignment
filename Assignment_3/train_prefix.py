import os
import time
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, PrefixTuningConfig, TaskType
from config import *
from data_loader import get_sst2_dataset
from utils import compute_metrics

os.environ.pop("HF_ENDPOINT", None)

base_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
peft_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_CLS,
    num_virtual_tokens=20,
    encoder_hidden_size=base_model.config.hidden_size,
    prefix_projection=True
)
model = get_peft_model(base_model, peft_config)
train_dataset, eval_dataset = get_sst2_dataset()

args = TrainingArguments(
    output_dir=f"{output_dir}/prefix",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    save_strategy="epoch",
    learning_rate=lr,
    logging_dir=f"{log_dir}/prefix",
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
print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.4f}M")
print(f"Training Time: {end - start:.2f}s")
if torch.cuda.is_available():
    print(f"Max GPU Memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
