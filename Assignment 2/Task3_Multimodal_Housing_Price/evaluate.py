"""
Reload the best saved fusion checkpoint and report MAE / RMSE on the test
split, plus a predicted-vs-actual scatter plot.

Usage:
    python evaluate.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from dataset import HousingMultimodalDataset, load_and_split
from model import FusionRegressor

MODELS_DIR = Path(__file__).parent / "models"


def main():
    ckpt_path = MODELS_DIR / "fusion_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found. Run `python train.py` first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    _, test_df, _ = load_and_split()
    test_ds = HousingMultimodalDataset(
        test_df, ckpt["location_categories"], img_size=ckpt["img_size"], train=False,
        tabular_mean=ckpt["tabular_mean"], tabular_std=ckpt["tabular_std"],
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = FusionRegressor(tabular_input_dim=ckpt["tabular_dim"], pretrained_cnn=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            preds = torch.expm1(model(images, tabular)).cpu().numpy()
            targets = batch["price"].numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    print(f"Test MAE:  ${mae:,.2f}")
    print(f"Test RMSE: ${rmse:,.2f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    lims = [min(all_targets.min(), all_preds.min()), max(all_targets.max(), all_preds.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual price ($)")
    plt.ylabel("Predicted price ($)")
    plt.title(f"Predicted vs Actual (MAE=${mae:,.0f}, RMSE=${rmse:,.0f})")
    plt.legend()
    plt.tight_layout()
    out_path = MODELS_DIR / "pred_vs_actual.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
