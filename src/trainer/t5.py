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

    def forward(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        return self.model(
            input_ids=batch["source_ids"].to(self.device),
            attention_mask=batch["source_mask"].to(self.device),
            labels=lm_labels.to(self.device),
            decoder_attention_mask=batch["target_mask"].to(self.device),
        )

    def training_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def validation_epoch_end(self):
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
        return val_loss_per_batch

    def fit(self):
        self.model.train()
        torch.set_grad_enabled(True)

        best_val_loss = 999  # init dummy variable

        for epoch in range(self.epochs):
            train_loss, val_loss = [], []

            with tqdm(self.train_loader, unit="batch") as tepoch:
                train_loss_per_batch = []
                for batch in tepoch:
                    outputs = self.forward(batch)  # Forward Propagation
                    loss, _ = outputs[:2]
                    self.training_step(loss)  # Backward Propagation (per step...)
                    train_loss_per_batch.append(loss.item())

                    tepoch.set_description(f"Epochs: {epoch+1}/{self.epochs}")
                    tepoch.set_postfix(
                        loss=loss.item(), lr=self.scheduler.get_last_lr()[-1]
                    )

                val_loss_per_batch = self.validation_epoch_end()

                epoch_train_loss = np.array(train_loss_per_batch).mean()
                epoch_val_loss = np.array(val_loss_per_batch).mean()

                train_loss.append(epoch_train_loss)
                val_loss.append(epoch_val_loss)
                print(
                    f"Avg training Loss : {epoch_train_loss} | Val Loss : {epoch_val_loss}"
                )

                # -- Checkpoint --
                if best_val_loss > epoch_val_loss:
                    self.save()
                    best_val_loss = epoch_val_loss

        self.train_loss = train_loss
        self.val_loss = val_loss

        print("\n".join(self.__build_report(train_loss, val_loss)))  # Reporting
        self.is_trained = True

    def save(self):
        if not self.is_trained:
            print("Warning: Model not trained!")
            return
        path = os.path.join(self.folder_path, f"model-best.pt")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        torch.save(
            {
                "epoch": self.epochs,
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
        os.makedirs(self.folder_path)

    def training_report(self):
        if not self.is_trained:
            print("Warning: Model not trained!")
            return
        return self.__build_report(self.train_loss, self.val_loss)

    def __build_report(self, train_loss, val_loss):
        assert len(train_loss) == len(val_loss)
        report_accumulator = []
        report_accumulator.append(
            "----------------------------------------------------------------------"
        )
        report_accumulator.append(
            "Epochs               Avg Training Loss     Avg Val Loss "
        )
        report_accumulator.append(
            "======================================================================"
        )

        for i in range(len(train_loss)):
            report_accumulator.append(
                "%-20s %-21s %-20s"
                % (
                    str(i + 1) + " / " + str(len(train_loss)),  # Epoch
                    train_loss[i],
                    val_loss[i],
                )
            )
            report_accumulator.append(
                "----------------------------------------------------------------------"
            )
        return report_accumulator
