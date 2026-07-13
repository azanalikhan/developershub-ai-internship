"""
Trains the multimodal (image + tabular) housing price regressor.

Usage:
    python generate_synthetic_data.py   # if you don't have a real dataset yet
    python train.py --epochs 15
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from dataset import HousingMultimodalDataset, load_and_split
from model import FusionRegressor

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--pretrained_cnn", action="store_true", default=True)
    p.add_argument("--no_pretrained_cnn", dest="pretrained_cnn", action="store_false")
    return p.parse_args()


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            log_price = batch["log_price"].to(device)

            preds = model(images, tabular)
            loss = criterion(preds, log_price)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            n += images.size(0)
            all_preds.extend(torch.expm1(preds).detach().cpu().numpy())
            all_targets.extend(torch.expm1(log_price).detach().cpu().numpy())

    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return total_loss / n, mae, rmse


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    csv_path = Path(__file__).parent / "data" / "houses.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python generate_synthetic_data.py` first, "
            "or provide your own data/houses.csv (see README)."
        )

    train_df, test_df, location_categories = load_and_split()

    train_ds = HousingMultimodalDataset(train_df, location_categories, img_size=args.img_size, train=True)
    test_ds = HousingMultimodalDataset(
        test_df, location_categories, img_size=args.img_size, train=False,
        tabular_mean=train_ds.tabular_mean, tabular_std=train_ds.tabular_std,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FusionRegressor(
        tabular_input_dim=train_ds.tabular_dim,
        pretrained_cnn=args.pretrained_cnn,
        freeze_backbone=True,
    ).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    best_mae = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae, train_rmse = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_mae, val_rmse = run_epoch(model, test_loader, optimizer, criterion, device, train=False)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_MAE=${train_mae:,.0f} | "
            f"val_loss={val_loss:.4f} val_MAE=${val_mae:,.0f} val_RMSE=${val_rmse:,.0f}"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_mae": train_mae,
             "val_loss": val_loss, "val_mae": val_mae, "val_rmse": val_rmse}
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tabular_dim": train_ds.tabular_dim,
                    "location_categories": location_categories,
                    "tabular_mean": train_ds.tabular_mean,
                    "tabular_std": train_ds.tabular_std,
                    "img_size": args.img_size,
                },
                MODELS_DIR / "fusion_model.pt",
            )

    with open(MODELS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val MAE: ${best_mae:,.0f}")
    print(f"Saved best checkpoint to {MODELS_DIR / 'fusion_model.pt'}")


if __name__ == "__main__":
    main()
