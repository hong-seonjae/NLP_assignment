from datasets import load_dataset
from transformers import AutoTokenizer
from config import model_name, max_len

def get_sst2_dataset():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("glue", "sst2")

    def preprocess(ex):
        return tokenizer(ex["sentence"], padding="max_length", truncation=True, max_length=max_len)

    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset["train"], dataset["validation"]
