"""
Reload a fine-tuned checkpoint and report accuracy / F1 / confusion matrix
on the AG News test split.

Usage:
    python evaluate.py --model_dir models/bert-agnews
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="models/bert-agnews")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_eval_samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_dir} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    ds = load_dataset("fancyzhx/ag_news", split="test")
    if args.max_eval_samples:
        ds = ds.select(range(args.max_eval_samples))

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(ds), args.batch_size):
            batch = ds[i : i + args.batch_size]
            enc = tokenizer(batch["text"], truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"])

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1w = f1_score(all_labels, all_preds, average="weighted")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Weighted F1: {f1w:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        all_labels, all_preds, display_labels=LABEL_NAMES, cmap="Blues", ax=ax
    )
    ax.set_title("AG News - Confusion Matrix")
    plt.tight_layout()
    out_path = model_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
