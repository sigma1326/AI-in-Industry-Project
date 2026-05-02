import collections
import logging
import os

from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
    MultilabelAUROC,
)

os.environ["WANDB_SILENT"] = "true"

import warnings

warnings.filterwarnings("ignore", message=".*cudnnException.*", category=UserWarning)

warnings.filterwarnings("ignore", module="torch.nn.modules.conv")

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

print(f"Using device: {device}")

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
CONFIG = {
    "model_name": "bert-base-cased",
    "max_tokens": 120,
    "batch_size": 128,
    "epochs": 3,
    "learning_rate": 2e-5,  # BERT needs a tiny learning rate
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,  # 10% of total steps used for warmup
    "threshold": 0.5,
    "device": device,
}

TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# ==========================================
# DATASET CLASS
# ==========================================
class ToxicBertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_tokens):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # MODERN UPGRADE: Let HuggingFace handle all the padding and truncation automatically!
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels = torch.tensor(self.labels[idx], dtype=torch.float)

        # .squeeze() removes the batch dimension that return_tensors='pt' adds
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


# ==========================================
# CLEAN MODEL ARCHITECTURE
# ==========================================
class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = (
            outputs.pooler_output
        )  # HuggingFace's official way to get the CLS token
        logits = self.classifier(cls_output)
        return logits  # NO SIGMOID HERE!


# ==========================================
# TRAINING LOOP WITH AMP
# ==========================================
def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    num_classes: int = 6,
    average_mode: str = "macro",
    threshold: float = 0.5,
):
    accuracy = MultilabelAccuracy(
        num_labels=num_classes,
        average=average_mode,
        threshold=threshold,
    ).to(device=device)
    precision = MultilabelPrecision(
        num_labels=num_classes,
        average=average_mode,
        threshold=threshold,
    ).to(device=device)
    recall = MultilabelRecall(
        num_labels=num_classes,
        average=average_mode,
        threshold=threshold,
    ).to(device=device)
    f1_score = MultilabelF1Score(
        num_labels=num_classes,
        average=average_mode,
        threshold=threshold,
    ).to(device=device)
    auroc = MultilabelAUROC(
        num_labels=num_classes,
        average=average_mode,
    ).to(device=device)

    model.train()
    total_loss = 0

    loop = tqdm(dataloader, desc="Training")
    for batch in loop:
        input_ids = batch["input_ids"].to(CONFIG["device"])
        attention_mask = batch["attention_mask"].to(CONFIG["device"])
        labels = batch["labels"].to(CONFIG["device"])

        optimizer.zero_grad()

        # MIXED PRECISION FORWARD PASS
        with torch.autocast(device_type=CONFIG["device"].type):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        # Apply Sigmoid to logits before updating metrics
        pred_probs = torch.sigmoid(logits.detach())
        accuracy.update(pred_probs, labels)
        precision.update(pred_probs, labels)
        recall.update(pred_probs, labels)
        f1_score.update(pred_probs, labels)
        auroc.update(pred_probs, labels.to(torch.long))

        # SAFE SCALING & CLIPPING
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return {
        "Training loss": total_loss / len(dataloader),
        "Training accuracy": accuracy.compute().item(),
        "Training precision": precision.compute().item(),
        "Training recall": recall.compute().item(),
        "Training F1_score": f1_score.compute().item(),
        "Training AUROC": auroc.compute().item(),
    }


# ==========================================
# EVALUATION LOOP
# ==========================================
def eval_epoch(
    model,
    dataloader,
    criterion,
    num_classes: int = 6,
    average_mode: str = "macro",
    threshold: float = 0.5,
):
    accuracy = MultilabelAccuracy(
        num_labels=num_classes,
        average=average_mode,
        threshold=threshold,
    ).to(device=device)
    precision = MultilabelPrecision(
        num_labels=num_classes,
        average=average_mode,
        threshold=threshold,
    ).to(device=device)
    recall = MultilabelRecall(
        num_labels=num_classes,
        average=average_mode,
        threshold=threshold,
    ).to(device=device)
    f1_score = MultilabelF1Score(
        num_labels=num_classes,
        average=average_mode,
        threshold=threshold,
    ).to(device=device)
    auroc = MultilabelAUROC(
        num_labels=num_classes,
        average=average_mode,
    ).to(device=device)

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(CONFIG["device"])
            attention_mask = batch["attention_mask"].to(CONFIG["device"])
            labels = batch["labels"].to(CONFIG["device"])

            with torch.autocast(CONFIG["device"].type):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()

            # Apply Sigmoid to logits before updating metrics
            pred_probs = torch.sigmoid(logits.detach())
            accuracy.update(pred_probs, labels)
            precision.update(pred_probs, labels)
            recall.update(pred_probs, labels)
            f1_score.update(pred_probs, labels)
            auroc.update(pred_probs, labels.to(torch.long))

    return {
        "Validation loss": total_loss / len(dataloader),
        "Validation accuracy": accuracy.compute().item(),
        "Validation precision": precision.compute().item(),
        "Validation recall": recall.compute().item(),
        "Validation F1_score": f1_score.compute().item(),
        "Validation AUROC": auroc.compute().item(),
    }


def final_test_evaluation(
    model,
    test_dataloader: DataLoader,
    device: torch.device,
    target_classes: list,
    threshold: float,
):
    print(f"\n{'=' * 50}")
    print(f"INITIATING FINAL TEST EVALUATION")
    print(f"Threshold: {threshold}")
    print(f"{'=' * 50}")

    model.eval()
    num_classes = len(target_classes)

    # Initialize MACRO metrics
    macro_f1 = MultilabelF1Score(
        num_labels=num_classes,
        average="macro",
        threshold=threshold,
    ).to(device)
    macro_auc = MultilabelAUROC(num_labels=num_classes, average="macro").to(device)

    # Initialize PER-CLASS metrics
    class_precision = MultilabelPrecision(
        num_labels=num_classes,
        average="none",
        threshold=threshold,
    ).to(device)
    class_recall = MultilabelRecall(
        num_labels=num_classes,
        average="none",
        threshold=threshold,
    ).to(device)
    class_f1 = MultilabelF1Score(
        num_labels=num_classes,
        average="none",
        threshold=threshold,
    ).to(device)
    class_auc = MultilabelAUROC(
        num_labels=num_classes,
        average="none",
    ).to(device)

    # Run the Test Set
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            # BERT requires input_ids and attention_mask, not just 'text_batch'
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(CONFIG["device"].type):
                logits = model(input_ids, attention_mask)

            pred_probs = torch.sigmoid(logits)

            # Update all metrics
            macro_f1.update(pred_probs, labels)
            macro_auc.update(pred_probs, labels.to(torch.long))
            class_precision.update(pred_probs, labels)
            class_recall.update(pred_probs, labels)
            class_f1.update(pred_probs, labels)
            class_auc.update(pred_probs, labels.to(torch.long))

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


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    checkpoint_dir = "model_weights"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = f"{checkpoint_dir}/bert_best.pth"

    # Initialize W&B
    wandb.init(
        entity="zagros-devs",
        project="Toxic-Comment-Detection",
        name="BERT-Base-Modern",
        config=CONFIG,
    )

    # Load Training Data
    print("Loading data...")
    df = pd.read_csv("data/train.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])

    train_dataset = ToxicBertDataset(
        texts=train_df["comment_text"].values,
        labels=train_df[TARGET_COLS].values,
        tokenizer=tokenizer,
        max_tokens=CONFIG["max_tokens"],
    )
    val_dataset = ToxicBertDataset(
        texts=val_df["comment_text"].values,
        labels=val_df[TARGET_COLS].values,
        tokenizer=tokenizer,
        max_tokens=CONFIG["max_tokens"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # Setup Model
    print("Initializing Model...")
    model = BertClassifier(
        CONFIG["model_name"],
        len(TARGET_COLS),
    ).to(CONFIG["device"])

    # No Decay for Biases and LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": CONFIG["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=CONFIG["learning_rate"],
    )

    # Mathematical stabilization combining Sigmoid + BCE
    criterion = nn.BCEWithLogitsLoss()

    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * CONFIG["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda")  # For Mixed Precision Speedup

    best_val_loss = float("inf")
    log_values = {}

    # Run Training
    print("Starting Training Loop...")
    for epoch_num in range(CONFIG["epochs"]):
        train_logs = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
        )
        val_logs = eval_epoch(model, val_loader, criterion)

        current_val_loss = val_logs["Validation loss"]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_epoch = epoch_num + 1
            torch.save(model.state_dict(), best_model_path)

        all_logs = {
            **train_logs,
            **val_logs,
        }

        wandb.log(all_logs)
        log_values[epoch_num + 1] = all_logs

    # Save last epoch model
    torch.save(model.state_dict(), f"{checkpoint_dir}/BERT_last.pth")

    # Save the metric logs to a CSV file
    all_series = collections.deque()
    for key, log_dict in log_values.items():
        all_series.append(pd.Series(log_dict))

    df = pd.DataFrame(all_series)
    df.to_csv(f"metric_logs/BERT.csv", index=False)
    print("\nTraining Phase Complete. Logs saved to CSV.")

    # ==========================================
    # FINAL TEST EVALUATION PHASE
    # ==========================================
    print("\n" + "=" * 50)
    print("INITIATING FINAL TEST EVALUATION")
    print("=" * 50)

    # Load and Merge Test Data
    test_df = pd.read_csv("data/test.csv")
    test_labels_df = pd.read_csv("data/test_labels.csv")
    test_merged = pd.merge(test_df, test_labels_df, on="id")

    # Filter out the Kaggle Ignore Rows (-1)
    test_merged = test_merged[test_merged["toxic"] != -1]

    # Create Test Dataset and Loader
    test_dataset = ToxicBertDataset(
        texts=test_merged["comment_text"].values,
        labels=test_merged[TARGET_COLS].values,
        tokenizer=tokenizer,
        max_tokens=CONFIG["max_tokens"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # Load the BEST weights we saved during training
    print(f"Loading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path))

    # Run test evaluation
    test_results = final_test_evaluation(
        model=model,
        test_dataloader=test_loader,
        device=CONFIG["device"],
        target_classes=TARGET_COLS,
        threshold=CONFIG["threshold"],
    )

    # Log the final macro scores to W&B
    # wandb.log({
    #     "Test Macro AUROC": test_results["macro_auroc"],
    #     "Test Macro F1": test_results["macro_f1"]
    # })
    wandb.finish()
    print("\nSUCCESSFULLY COMPLETED: BERT_base_cased pipeline")
