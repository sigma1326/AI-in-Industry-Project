from __future__ import annotations

import collections
import pprint
import os
from functools import partial

import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
    MultilabelAUROC,
)
import wandb

pp = pprint.PrettyPrinter(indent=4)


class BaseModel(nn.Module):
    def __init__(
        self,
        target_classes_len: int = 6,
        optimizer_type: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        maximum_tokens: int = 50,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.optimizer = None
        self.loss_fn = None

        self.target_classes_len = target_classes_len
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_tokens = maximum_tokens

        self.op = {
            "adamW": torch.optim.AdamW,
            "adadelta": torch.optim.Adadelta,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        if optimizer_type not in self.op:
            raise ValueError("Invalid value")

        self.log_values = {}
        self.scaler = torch.amp.GradScaler("cuda")  # For Mixed Precision Speedup

    def _prepare_model(self, class_weights: list[int] | None = None):
        self.optimizer = self.op[self.optimizer_type](
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if class_weights is not None:
            device = next(self.parameters()).device
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            class_weights = class_weights.to(device)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def train_model(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        num_of_epochs: int,
        device: torch.device,
        name: str,
        architecture: str,
        class_weights: list[int],
        threshold: float = 0.5,
        average_mode: str = "macro",
        init_wandb: bool = True,
        wandb_run_: wandb.Run | None = None,
        print_logs: bool = False,
    ):
        self._prepare_model(class_weights)

        # --- BEST MODEL TRACKING SETUP ---
        best_val_loss = float("inf")
        best_model_epoch = 1
        checkpoint_dir = "model_weights"
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = f"{checkpoint_dir}/{name}_best.pth"

        if init_wandb:
            wandb_run = wandb.init(
                entity="zagros-devs",
                project="Toxic Comment Detection",
                name=name,
                config={
                    "architecture": architecture,
                    "optimizer": self.optimizer_type,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "threshold": threshold,
                    "epochs": num_of_epochs,
                    "type": "train",
                },
                anonymous="allow",
            )
        else:
            wandb_run = None

        if wandb_run_:
            wandb_run = wandb_run_

        for epoch_num in range(num_of_epochs):
            self.train()
            total_loss = 0

            # Initialize all metrics for this epoch
            accuracy = MultilabelAccuracy(
                num_labels=self.target_classes_len,
                average=average_mode,
                threshold=threshold,
            ).to(device)
            precision = MultilabelPrecision(
                num_labels=self.target_classes_len,
                average=average_mode,
                threshold=threshold,
            ).to(device)
            recall = MultilabelRecall(
                num_labels=self.target_classes_len,
                average=average_mode,
                threshold=threshold,
            ).to(device)
            f1_score = MultilabelF1Score(
                num_labels=self.target_classes_len,
                average=average_mode,
                threshold=threshold,
            ).to(device)
            auroc = MultilabelAUROC(
                num_labels=self.target_classes_len,
                average=average_mode,
            ).to(device)

            loop = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch_num + 1}/{num_of_epochs}",
                leave=False,
            )

            for text_batch, label_batch in loop:
                text_batch = text_batch.to(device)
                label_batch = label_batch.to(device)
                self.optimizer.zero_grad()

                # Mixed Precision Forward Pass
                with torch.autocast(device_type=device.type):
                    pred = self(text_batch)
                    loss = self.loss_fn(pred, label_batch)

                self.scaler.scale(loss).backward()
                # Unscale the gradients back to their normal mathematical size
                self.scaler.unscale_(self.optimizer)
                # Clip the true gradients so they don't explode
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

                # Apply Sigmoid to logits before updating metrics
                pred_probs = torch.sigmoid(pred.detach())
                accuracy.update(pred_probs, label_batch)
                precision.update(pred_probs, label_batch)
                recall.update(pred_probs, label_batch)
                f1_score.update(pred_probs, label_batch)
                auroc.update(pred_probs, label_batch.to(torch.long))

            # Validation Phase
            val_logs = self.validate_model(
                dataloader=validation_dataloader,
                device=device,
                threshold=threshold,
                average_mode=average_mode,
            )

            # --- CHECKPOINT LOGIC ---
            current_val_loss = val_logs["Validation loss"]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_epoch = epoch_num + 1
                torch.save(self.state_dict(), best_model_path)

            all_logs = {
                "Training loss": total_loss / len(train_dataloader),
                "Training accuracy": accuracy.compute().item(),
                "Training precision": precision.compute().item(),
                "Training recall": recall.compute().item(),
                "Training F1_score": f1_score.compute().item(),
                "Training AUROC": auroc.compute().item(),
                **val_logs,
            }

            if wandb_run:
                wandb.log(all_logs)

            if print_logs:
                print(f"Epoch {epoch_num + 1}:")
                pp.pprint(all_logs)
                print("=" * 50)

            # File Logging Logic
            log_entry = all_logs.copy()
            log_entry.update({"Epoch": int(epoch_num + 1)})
            for key in log_entry.keys():
                if isinstance(log_entry[key], float):
                    log_entry[key] = float(format(log_entry[key], ".3f"))
            self.log_values[epoch_num + 1] = log_entry

        if wandb_run:
            wandb.finish()

        print(f"Best model @ epoch {best_model_epoch}")
        self.write_logs_to_file(name)
        self.save_model_to_disk(f"{name}_last_epoch")

    def validate_model(
        self,
        dataloader: DataLoader,
        device: torch.device,
        threshold: float = 0.5,
        average_mode: str = "macro",
    ):
        self.eval()
        total_loss = 0

        accuracy = MultilabelAccuracy(
            num_labels=self.target_classes_len,
            average=average_mode,
            threshold=threshold,
        ).to(device)
        precision = MultilabelPrecision(
            num_labels=self.target_classes_len,
            average=average_mode,
            threshold=threshold,
        ).to(device)
        recall = MultilabelRecall(
            num_labels=self.target_classes_len,
            average=average_mode,
            threshold=threshold,
        ).to(device)
        f1_score = MultilabelF1Score(
            num_labels=self.target_classes_len,
            average=average_mode,
            threshold=threshold,
        ).to(device)
        auroc = MultilabelAUROC(
            num_labels=self.target_classes_len,
            average=average_mode,
        ).to(device)

        with torch.no_grad():
            for text_batch, label_batch in dataloader:
                text_batch = text_batch.to(device)
                label_batch = label_batch.to(device)

                with torch.autocast(device_type=device.type):
                    pred = self(text_batch)
                    loss = self.loss_fn(pred, label_batch)

                pred_probs = torch.sigmoid(pred)
                accuracy.update(pred_probs, label_batch)
                precision.update(pred_probs, label_batch)
                recall.update(pred_probs, label_batch)
                f1_score.update(pred_probs, label_batch)
                auroc.update(pred_probs, label_batch.to(torch.long))

                total_loss += loss.item()

        return {
            "Validation loss": total_loss / len(dataloader),
            "Validation accuracy": accuracy.compute().item(),
            "Validation precision": precision.compute().item(),
            "Validation recall": recall.compute().item(),
            "Validation F1_score": f1_score.compute().item(),
            "Validation AUROC": auroc.compute().item(),
        }

    def test_model(
        self,
        test_dataloader: DataLoader,
        device: torch.device,
        target_classes: list[str],
        threshold: float,
    ):
        print(f"\n{'=' * 50}")
        print(f"INITIATING FINAL TEST EVALUATION")
        print(f"Threshold: {threshold}")
        print(f"{'=' * 50}")

        self.eval()

        # Initialize MACRO metrics (The overall score)
        macro_f1 = MultilabelF1Score(
            num_labels=self.target_classes_len, average="macro", threshold=threshold
        ).to(device)
        macro_auc = MultilabelAUROC(
            num_labels=self.target_classes_len, average="macro"
        ).to(device)

        # Initialize PER-CLASS metrics (average="none" returns an array of 6 values)
        class_precision = MultilabelPrecision(
            num_labels=self.target_classes_len, average="none", threshold=threshold
        ).to(device)
        class_recall = MultilabelRecall(
            num_labels=self.target_classes_len, average="none", threshold=threshold
        ).to(device)
        class_f1 = MultilabelF1Score(
            num_labels=self.target_classes_len, average="none", threshold=threshold
        ).to(device)
        class_auc = MultilabelAUROC(
            num_labels=self.target_classes_len, average="none"
        ).to(device)

        # Run the Test Set
        with torch.no_grad():
            for text_batch, label_batch in tqdm(
                test_dataloader, desc="Testing", leave=False
            ):
                text_batch = text_batch.to(device)
                label_batch = label_batch.to(device)

                with torch.autocast(device_type=device.type):
                    pred = self(text_batch)

                pred_probs = torch.sigmoid(pred)

                # Update all metrics
                macro_f1.update(pred_probs, label_batch)
                macro_auc.update(pred_probs, label_batch.to(torch.long))
                class_precision.update(pred_probs, label_batch)
                class_recall.update(pred_probs, label_batch)
                class_f1.update(pred_probs, label_batch)
                class_auc.update(pred_probs, label_batch.to(torch.long))

        # Compute Final Values
        final_macro_f1 = macro_f1.compute().item()
        final_macro_auc = macro_auc.compute().item()

        c_prec = class_precision.compute().cpu().tolist()
        c_rec = class_recall.compute().cpu().tolist()
        c_f1 = class_f1.compute().cpu().tolist()
        c_auc = class_auc.compute().cpu().tolist()

        # Print the Report
        print(f"\nOVERALL PERFORMANCE")
        print(f"Macro AUROC: {final_macro_auc:.4f}")
        print(f"Macro F1:    {final_macro_f1:.4f}\n")

        print(f"PER-CLASS BREAKDOWN")
        print(
            f"{'Class':<15} | {'AUROC':<7} | {'Precision':<9} | {'Recall':<7} | {'F1-Score':<7}"
        )
        print("-" * 55)

        per_class_results = {}
        for i, class_name in enumerate(target_classes):
            print(
                f"{class_name:<15} | {c_auc[i]:.4f}  | {c_prec[i]:.4f}    | {c_rec[i]:.4f}  | {c_f1[i]:.4f}"
            )
            per_class_results[class_name] = {
                "AUROC": c_auc[i],
                "Precision": c_prec[i],
                "Recall": c_rec[i],
                "F1": c_f1[i],
            }

        print("=" * 55)

        return {
            "macro_f1": final_macro_f1,
            "macro_auroc": final_macro_auc,
            "per_class": per_class_results,
        }

    @classmethod
    def _train_sweep(
        cls,
        architecture: str,
        embedding_dim: int,
        train_dl: DataLoader,
        val_dl: DataLoader,
        maximum_tokens: int,
        class_weights: list[int],
        device: torch.device,
        num_of_epochs: int = 20,
    ):
        # Initialize the W&B run
        run = wandb.init()

        # Define how W&B should summarize the metric
        wandb.define_metric("Validation AUROC", summary="max")

        # wandb.config holds the specific combination for THIS run
        config = wandb.config
        name = f"{architecture}_embed{embedding_dim}_{maximum_tokens}t_sweep_{config.optimizer_type}_{config.learning_rate}_{config.weight_decay}"
        if getattr(config, "num_channels", None):
            name += f"_{config.num_channels}_{config.kernel_sizes}"
        run.name = name
        config["architecture"] = architecture
        config["type"] = "sweep"
        config["embedding_dimension"] = embedding_dim
        config["maximum_tokens"] = maximum_tokens

        # Initialize your model using the sweep's choices
        if getattr(config, "num_channels", None):
            model = cls(
                embed_dim=embedding_dim,  # Keep this static for the sweep
                conv_config={
                    "num_channels": config.num_channels,
                    "kernel_sizes": config.kernel_sizes,
                },
                optimizer_type=config.optimizer_type,
                learning_rate=config.learning_rate,
                maximum_tokens=maximum_tokens,
            ).to(device)
        else:
            model = cls(
                embed_dim=embedding_dim,  # Keep this static for the sweep
                optimizer_type=config.optimizer_type,
                learning_rate=config.learning_rate,
                maximum_tokens=maximum_tokens,
            ).to(device)

        # Train the model
        model.train_model(
            train_dataloader=train_dl,
            validation_dataloader=val_dl,
            num_of_epochs=num_of_epochs,
            device=device,
            name=name,
            architecture=architecture,
            class_weights=class_weights,
            threshold=0.5,
            average_mode="macro",
            init_wandb=False,  # The train_model shouldn't call wandb.init() again if a run is active
            wandb_run_=run,
        )

    @classmethod
    def run_sweep(
        cls,
        sweep_config: dict,
        architecture: str,
        embedding_dim: int,
        train_dl: DataLoader,
        val_dl: DataLoader,
        maximum_tokens: int,
        class_weights: list[int],
        device: torch.device,
        num_of_epochs: int = 20,
        num_of_sweeps: int = 15,
    ):
        # Create the sweep on the W&B servers
        sweep_id = wandb.sweep(
            sweep_config,
            entity="zagros-devs",
            project="Toxic Comment Detection",
        )

        # try the 15 smartest combinations and then stop
        wandb.agent(
            sweep_id,
            partial(
                cls._train_sweep,
                **{
                    "architecture": architecture,
                    "embedding_dim": embedding_dim,
                    "train_dl": train_dl,
                    "val_dl": val_dl,
                    "maximum_tokens": maximum_tokens,
                    "class_weights": class_weights,
                    "device": device,
                    "num_of_epochs": num_of_epochs,
                },
            ),
            count=num_of_sweeps,
        )

        return sweep_id

    def find_the_best_threshold(
        self,
        val_dl: DataLoader,
        device: torch.device,
    ):
        # Gather all the raw predictions from the validation set
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for text_batch, label_batch in val_dl:  # Use your validation dataloader
                text_batch = text_batch.to(device)

                # Get raw logits and apply sigmoid to get probabilities (0.0 to 1.0)
                logits = self(text_batch)
                probs = torch.sigmoid(logits)

                all_preds.append(probs.cpu())
                all_labels.append(label_batch.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Sweep the thresholds from 0.1 to 0.9!
        print("Testing Thresholds for Peak F1...")
        best_threshold = 0.5
        peak_f1 = 0.0

        for t in torch.arange(0.5, 1.0, 0.02):
            # Create a temporary F1 metric with the current threshold
            temp_f1_metric = MultilabelF1Score(
                num_labels=6,
                average="macro",
                threshold=float(t),
            )

            # Calculate the score
            current_f1 = temp_f1_metric(all_preds, all_labels).item()
            print(f"Threshold: {t:.2f} | Macro F1: {current_f1:.4f}")

            if current_f1 > peak_f1:
                peak_f1 = current_f1
                best_threshold = t

        print("=" * 40)
        print(f"Best MODEL FOUND")
        print(f"Optimal Threshold: {best_threshold:.2f}")
        print(f"Peak Macro F1: {peak_f1:.4f}")

        return float(best_threshold), float(peak_f1)

    def save_model_to_disk(self, name: str):
        print(f"Saving {name} weights...")
        torch.save(self.state_dict(), f"model_weights/{name}.pth")

    def load_model_from_disk(self, path: str):
        print(f"Loading weights from {path}...")
        self.load_state_dict(torch.load(path), strict=False)
        self.eval()

    def write_logs_to_file(self, name: str):
        all_series = collections.deque()
        for key, log_dict in self.log_values.items():
            all_series.append(pd.Series(log_dict))
        df = pd.DataFrame(all_series)
        os.makedirs("metric_logs", exist_ok=True)
        df.to_csv(f"metric_logs/{name}.csv", index=False)
