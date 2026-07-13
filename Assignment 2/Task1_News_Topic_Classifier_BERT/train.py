"""
Task 1: News Topic Classifier Using BERT
==========================================
Fine-tunes bert-base-uncased on the AG News dataset for 4-way topic
classification (World, Sports, Business, Sci/Tech).

Usage:
    python train.py --epochs 3 --batch_size 16
    python train.py --max_train_samples 2000 --max_eval_samples 500  # quick smoke test
"""
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = Path(__file__).parent / "models" / "bert-agnews"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--max_train_samples", type=int, default=None,
                    help="Subset the training set (useful for a CPU smoke test)")
    p.add_argument("--max_eval_samples", type=int, default=None,
                    help="Subset the eval set (useful for a CPU smoke test)")
    return p.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    args = parse_args()

    print("Loading AG News dataset...")
    raw_datasets = load_dataset("fancyzhx/ag_news")

    if args.max_train_samples:
        raw_datasets["train"] = raw_datasets["train"].select(range(args.max_train_samples))
    if args.max_eval_samples:
        raw_datasets["test"] = raw_datasets["test"].select(range(args.max_eval_samples))

    print("Loading tokenizer + base model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES)
    )
    model.config.id2label = dict(enumerate(LABEL_NAMES))
    model.config.label2id = {name: i for i, name in enumerate(LABEL_NAMES)}

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    print("Tokenizing...")
    tokenized = raw_datasets.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        report_to=[],
        fp16=False,  # set True automatically-detected GPUs can flip this on
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Final evaluation on test split:")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving model + tokenizer to {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
