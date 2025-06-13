from transformers import pipeline

classifier = pipeline("text-classification", model="./results/lora/checkpoint-best")

examples = [
    "This movie is amazing!",
    "I didn't like this film at all.",
    "Absolutely fantastic acting.",
    "Waste of time.",
    "It's okay, not great but not terrible."
]

for i, text in enumerate(examples):
    result = classifier(text)[0]
    print(f"Example {i+1}: {text}\n → Predicted: {result['label']} (Confidence: {result['score']:.4f})\n")
