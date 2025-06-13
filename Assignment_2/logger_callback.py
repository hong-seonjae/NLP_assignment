from transformers import TrainerCallback
import math

class CustomLoggingCallback(TrainerCallback):
    def __init__(self, log_file="training_log.txt", tag="full"):
        self.losses = []
        self.train_losses = []
        self.eval_losses = []
        self.eval_accs = []
        self.current_epoch = 0
        self.log_file = log_file
        self.tag = tag

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if "loss" in logs:
            self.losses.append(logs["loss"])
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
        if "eval_accuracy" in logs:
            self.eval_accs.append(logs["eval_accuracy"])

        epoch = math.floor(state.epoch or 0)
        if epoch > self.current_epoch:
            avg_loss = sum(self.losses) / len(self.losses)
            print(f"Epoch [{self.current_epoch + 1}/{int(args.num_train_epochs)}] | Avg Loss: {avg_loss:.4f}")
            with open(self.log_file, "a") as f:
                f.write(f"Epoch {self.current_epoch + 1}: Train Loss = {avg_loss:.4f}\n")
            self.losses = []
            self.current_epoch = epoch

    def on_train_end(self, args, state, control, **kwargs):
        min_len = min(len(self.train_losses), len(self.eval_losses))
        epochs = list(range(1, min_len + 1))
        with open(f"training_log_{self.tag}.txt", "a") as f:
            for i in range(min_len):
                train_l = self.train_losses[i]
                eval_l = self.eval_losses[i] if i < len(self.eval_losses) else None
                acc = self.eval_accs[i] if i < len(self.eval_accs) else None
                f.write(f"Epoch {i+1}: Train Loss={train_l:.4f}, Eval Loss={eval_l:.4f}, Eval Acc={acc:.4f}\n")
