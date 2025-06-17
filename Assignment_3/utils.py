from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    preds = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    return {"accuracy": accuracy_score(labels, preds)}