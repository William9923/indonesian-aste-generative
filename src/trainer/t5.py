import os

from transformers import (
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
    AdamW,
)
import torch
from tqdm import tqdm
import numpy as np

from src.utility import get_device
from src.constant import Path


class T5Trainer:
    def __init__(self, tokenizer, train_loader, val_loader, prefix, configs):
        self.device = get_device()
        self.configs = configs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer

        model_name = self.configs.get("main").get("pretrained")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        _ = self.model.to(self.device)

        self.learning_rate = self.configs.get("trainer").get("learning_rate")
        self.n_gpu = self.configs.get("trainer").get("n_gpu")
        self.epochs = self.configs.get("trainer").get("epochs")
        self.batch_size = self.configs.get("trainer").get("batch_size")
        self.eval_batch_size = self.configs.get("trainer").get("eval_batch_size")

        self.eps = self.configs.get("trainer").get("adam_epsilon")

        self.__configure_optimizer()
        self.__create_folder(prefix)

        # define states...
        self.is_trained = False

    def fit(self):
        self.model.train()
        torch.set_grad_enabled(True)

        for epoch in range(self.epochs):
            print(f"Current Epochs: {epoch+1}/{self.epochs}")

            train_loss, val_loss = [], []
            with tqdm(self.train_loader, unit="batch") as tepoch:
                train_loss_per_batch = []
                for batch in tepoch:
                    lm_labels = batch["target_ids"]
                    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
                    outputs = self.model(
                        input_ids=batch["source_ids"].to(self.device),
                        attention_mask=batch["source_mask"].to(self.device),
                        labels=lm_labels.to(self.device),
                        decoder_attention_mask=batch["target_mask"].to(self.device),
                    )

                    # -- [Part : Backward Propagation] --
                    loss, logits = outputs[:2]

                    # -- end forward propagation --

                    # -- training step --
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    # -- end training step --

                    train_loss_per_batch.append(loss.item())

                    # -- [Part : Logging Related Information] --
                    lr = self.scheduler.get_last_lr()[-1]
                    tepoch.set_description("Training")
                    tepoch.set_postfix(loss=loss.item(), lr=lr)

                # -- [Validation per epoch] --
                # -- start validation epoch end --
                with torch.no_grad():
                    val_loss_per_batch = []
                    for batch in self.val_loader:
                        lm_labels = batch["target_ids"]
                        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
                        outputs = self.model(
                            input_ids=batch["source_ids"].to(self.device),
                            attention_mask=batch["source_mask"].to(self.device),
                            labels=lm_labels.to(self.device),
                            decoder_attention_mask=batch["target_mask"].to(self.device),
                        )
                        loss, _ = outputs[:2]
                        val_loss_per_batch.append(loss.item())
                # -- return validation epoch end...

                # -- [Logging] --
                epoch_train_loss = np.array(train_loss_per_batch).mean()
                epoch_val_loss = np.array(val_loss_per_batch).mean()

                train_loss.append(epoch_train_loss)
                val_loss.append(epoch_val_loss)
                print(
                    f"Avg training Loss : {epoch_train_loss} | Val Loss : {epoch_val_loss}"
                )

                # -- [Checkpoint] --
                if best_val_loss > epoch_val_loss:
                    print("Removing Previous Best ...")

                    try:
                        path = os.path.join(
                            self.ckpt_path, f"model-state-{best_epoch+1}.pt"
                        )
                        os.remove(path)
                    except FileNotFoundError:
                        print("File Not Found...")

                    print("Saving Best Model...")
                    path = os.path.join(self.ckpt_path, f"model-state-{epoch+1}.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": epoch_train_loss,
                        },
                        path,
                    )

                    # Renew data
                    best_epoch = epoch
                    best_val_loss = epoch_val_loss
                    best_train_loss = epoch_train_loss

        # -- Logging best training status --
        print("Best Training Params:")
        print("---------------------")
        print(f"Epoch           : {best_epoch+1}")
        print(f"Training Loss   : {best_train_loss}")
        print(f"Validation Loss : {best_val_loss}")
        print(f"Batch Size      : {self.train_loader.batch_size}")
        print(f"Ckpt Path       : {path}")

        self.is_trained = True

    def save(self):
        if not self.is_trained:
            print("Warning: Model not trained!")
            return
        path = os.path.join(self.ckpt_path, f"model-state-last.pt")
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        device = get_device()
        _ = self.model.to(device)

        self.is_trained = True

    def get_model(self):
        return self.model

    def __configure_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        weight_decay = self.configs.get("trainer").get("weight_decay")
        gradient_accumulation_steps = self.configs.get("trainer").get(
            "gradient_accumulation_steps"
        )
        warmup_steps = self.configs.get("trainer").get("warm_up_step")

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate, eps=self.eps
        )

        t_total = (
            (len(self.train_loader.dataset) // (self.batch_size))
            // gradient_accumulation_steps
            * float(self.epochs)
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

    def __create_folder(self, prefix):
        self.folder_path = os.path.join(Path.MODEL, prefix)
        self.ckpt_path = os.path.join(self.folder_path, Path.CHECKPOINT)
        os.mkdir(self.folder_path)
        os.mkdir(self.ckpt_path)
