import os

os.environ["WANDB_SILENT"] = "true"

import warnings

warnings.filterwarnings("ignore", message=".*cudnnException.*", category=UserWarning)

warnings.filterwarnings("ignore", module="torch.nn.modules.conv")

import logging

import torchtext

torchtext.disable_torchtext_deprecation_warning()

import pandas as pd
import torch

import wandb
import traceback

from models import CLSTM, CNN, MLP, MCBiGRU, CNNBiGRU
from preprocessing import get_training_dataloaders, get_test_dataloader


logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

print(f"Using device: {device}")


train_df = pd.read_csv("data/train.csv")


target_classes = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Count how many 0s and 1s are in each column of the training data
# Formula: number_of_negatives / number_of_positives
class_weights = []
for col in target_classes:
    num_positives = train_df[col].sum()
    num_negatives = len(train_df) - num_positives
    weight = num_negatives / num_positives
    class_weights.append(weight)

print(f"Class Weights: {class_weights}")


#  Initialize the W&B API for querying results
api = wandb.Api()

# Define the testing matrix
model_registry = {
    "CLSTM": CLSTM,
    "CNN": CNN,
    "CNN-BiGRU": CNNBiGRU,
    "MCBiGRU": MCBiGRU,
    "MLP": MLP,
}

token_configs = [25, 50, 100]
embedding_configs = [
    {"name": "glove_100", "dim": 100},
    {"name": "glove_300", "dim": 300},
]

# Hyperparameters for the master loop
BATCH_SIZE = 256
NUM_SWEEPS = 20
SWEEP_EPOCHS = 20
FINAL_EPOCHS = 50

sweep_config = {
    "method": "bayes",  # Bayesian optimization
    "metric": {"name": "Validation AUROC", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [0.005, 0.001, 0.0005, 0.0001]},
        "weight_decay": {"values": [1e-3, 1e-4, 1e-5, 0.0]},
        "optimizer_type": {"values": ["adamW", "rmsprop"]},
        "num_channels": {"values": [16, 32, 50, 64, 100, 128, 256, 512]},
        "kernel_sizes": {
            "values": [
                [1, 2, 3],
                [1, 2, 3, 5],
                [2, 3, 4],
                [1, 3, 5],
                [1, 3, 5, 7],
                [1, 3, 5, 7, 11],
                [3, 4, 5],
                [3, 5, 7],
            ]
        },
    },
}

for model_name, ModelClass in model_registry.items():
    for max_tokens in token_configs:
        for emb in embedding_configs:
            emb_name = str(emb["name"])
            emb_dim = int(emb["dim"])

            run_signature = f"{model_name}_{max_tokens}t_{emb_name}"
            print(f"\n{'=' * 60}")
            print(f"STARTING PIPELINE: {run_signature}")
            print(f"{'=' * 60}")

            try:
                # ==========================================
                # PHASE 1: THE SWEEP
                # ==========================================
                sweep_train_dl, sweep_val_dl = get_training_dataloaders(
                    target_convertor=emb_name,
                    maximum_tokens=max_tokens,
                    batch_size=BATCH_SIZE,
                    dataset_fraction=0.3,
                )

                print(f"Phase 1: Running Sweep for {run_signature}...")
                sweep_id = ModelClass.run_sweep(
                    sweep_config=sweep_config,
                    architecture=model_name,
                    embedding_dim=emb_dim,
                    train_dl=sweep_train_dl,
                    val_dl=sweep_val_dl,
                    maximum_tokens=max_tokens,
                    class_weights=class_weights,
                    device=device,
                    num_of_epochs=SWEEP_EPOCHS,
                    num_of_sweeps=NUM_SWEEPS,
                )

                # ==========================================
                # PHASE 2: EXTRACT THE CHAMPION
                # ==========================================
                print(f"Phase 2: Fetching winning configs from W&B...")
                sweep = api.sweep(f"zagros-devs/Toxic Comment Detection/{sweep_id}")

                # W&B automatically finds the best run based on your sweep_config goal!
                best_run = sweep.best_run()
                best_config = best_run.config
                if best_config.get("num_channels", None):
                    print(
                        f"--> Best Optimizer: {best_config['optimizer_type']} | LR: {best_config['learning_rate']} | Num Channels: {best_config['num_channels']} | Kernel Sizes: {best_config['kernel_sizes']}"
                    )
                else:
                    print(
                        f"--> Best Optimizer: {best_config['optimizer_type']} | LR: {best_config['learning_rate']}"
                    )
                # ==========================================
                # PHASE 3: FIND BEST THRESHOLD
                # ==========================================
                print(f"Phase 3: Finding optimal threshold...")

                if best_config.get("num_channels", None):
                    temp_model = ModelClass(
                        embed_dim=emb_dim,
                        conv_config={
                            "num_channels": best_config.get("num_channels", 50),
                            "kernel_sizes": best_config.get("kernel_sizes", [1, 2, 3]),
                        },
                        optimizer_type=best_config["optimizer_type"],
                        learning_rate=best_config["learning_rate"],
                        weight_decay=best_config["weight_decay"],
                        maximum_tokens=max_tokens,
                    ).to(device)
                else:
                    temp_model = ModelClass(
                        embed_dim=emb_dim,
                        optimizer_type=best_config["optimizer_type"],
                        learning_rate=best_config["learning_rate"],
                        weight_decay=best_config["weight_decay"],
                        maximum_tokens=max_tokens,
                    ).to(device)

                # Load weights saved during the sweep
                weight_path = f"model_weights/{best_run.name}_best.pth"
                temp_model.load_model_from_disk(weight_path)

                best_threshold, peak_f1 = temp_model.find_the_best_threshold(
                    sweep_val_dl, device=device
                )
                print(
                    f"--> Optimal Threshold: {best_threshold:.2f} (Peak F1: {peak_f1:.4f})"
                )

                # ==========================================
                # PHASE 4: FINAL FULL-DATASET TRAINING
                # ==========================================
                print(f"Phase 4: Training Final Model on 100% Data...")
                full_train_dl, full_val_dl = get_training_dataloaders(
                    target_convertor=emb_name,
                    maximum_tokens=max_tokens,
                    batch_size=BATCH_SIZE,
                    dataset_fraction=1.0,  # 100% Data
                )

                if best_config.get("num_channels", None):
                    final_model = ModelClass(
                        embed_dim=emb_dim,
                        conv_config={
                            "num_channels": best_config.get("num_channels", 50),
                            "kernel_sizes": best_config.get("kernel_sizes", [1, 2, 3]),
                        },
                        optimizer_type=best_config["optimizer_type"],
                        learning_rate=best_config["learning_rate"],
                        maximum_tokens=max_tokens,
                    ).to(device)
                else:
                    final_model = ModelClass(
                        embed_dim=emb_dim,
                        optimizer_type=best_config["optimizer_type"],
                        learning_rate=best_config["learning_rate"],
                        maximum_tokens=max_tokens,
                    ).to(device)

                final_model.train_model(
                    train_dataloader=full_train_dl,
                    validation_dataloader=full_val_dl,
                    num_of_epochs=FINAL_EPOCHS,
                    device=device,
                    name=f"FINAL_{run_signature}",
                    architecture=model_name,
                    class_weights=class_weights,
                    threshold=best_threshold,
                    average_mode="macro",
                    init_wandb=True,
                )

                # ==========================================
                # Phase 5: FINAL TEST EVALUATION
                # ==========================================
                print(f"Phase 5: Evaluating Final Model on Test Set...")

                test_dl = get_test_dataloader(
                    target_convertor=emb_name,
                    maximum_tokens=max_tokens,
                    batch_size=BATCH_SIZE,
                    dataset_fraction=1.0,
                )

                final_model.eval()
                test_results = final_model.test_model(
                    test_dataloader=test_dl,
                    device=device,
                    target_classes=target_classes,
                    threshold=best_threshold,
                )

                print(f"SUCCESSFULLY COMPLETED: {run_signature}")

            except Exception as e:
                print(f"ERROR ON {run_signature}: {e}")
                traceback.print_exc()
                continue
