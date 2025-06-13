from datasets import load_dataset
from transformers import AutoTokenizer

def load_sst2_dataset(tokenizer, max_len=128):
    dataset = load_dataset("glue", "sst2")

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset["train"], dataset["validation"]