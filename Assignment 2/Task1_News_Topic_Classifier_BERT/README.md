# Task 1: News Topic Classifier Using BERT

## Objective
Fine-tune `bert-base-uncased` to classify news headlines into one of four AG
News topic categories — **World, Sports, Business, Sci/Tech** — and serve the
model live through a Streamlit app.

---

## Files in this folder — what each one does, and the order to use them

There are 4 files that matter. Use them **in this order**:

| Order | File | What it does | When to run it |
|---|---|---|---|
| 1 | `requirements.txt` | Lists every Python package this task needs (transformers, datasets, torch, streamlit, etc.) | Once, before anything else |
| 2 | `train.py` | Downloads AG News + `bert-base-uncased`, fine-tunes the model, evaluates it, and **saves the trained model** to `models/bert-agnews/` | Run this first — nothing else works until a model exists |
| 3 | `evaluate.py` | Reloads the saved model from `models/bert-agnews/` and reports detailed metrics (accuracy, per-class precision/recall/F1, confusion matrix image) | Run after training, whenever you want a full report |
| 4 | `app.py` | Reloads the saved model and launches a Streamlit web page where you can type a headline and see a live prediction | Run any time after training — this is the "demo" |

**In short: `train.py` -> `evaluate.py` -> `app.py`.** You cannot run
`evaluate.py` or `app.py` before `train.py` has produced a saved model — both
will show a clear error telling you to run `train.py` first if the model
isn't found.

---

## Step-by-step: how to actually run it

### 0. Install dependencies (once)
```powershell
pip install -r requirements.txt
```

### 1. Train the model (`train.py`)
```powershell
python train.py --max_train_samples 2000 --max_eval_samples 500 --epochs 1
```
- This is the **smoke-test** version — trains fast (well under an hour on
  CPU) on a small subset of the data, just to prove the pipeline works.
- For a stronger model, drop the subset flags and increase epochs, e.g.
  `python train.py --epochs 3` — but be aware this trains on the full
  120,000-example dataset and can take many hours on CPU alone.
- What it produces: a saved model + tokenizer in `models/bert-agnews/`, plus
  printed accuracy/F1 at the end.
- Useful flags:
  - `--max_train_samples N` / `--max_eval_samples N` — use a subset (faster, for testing)
  - `--epochs N` — how many passes over the training data
  - `--batch_size N` — lower this (e.g. 8) if you run out of memory

### 2. Get a full evaluation report (`evaluate.py`)
```powershell
python evaluate.py --model_dir models/bert-agnews --batch_size 8
```
- Reloads the model saved by `train.py` and runs it over the full AG News
  test set (7,600 examples), independent of whatever subset you trained on.
- Prints accuracy, weighted F1, and a full per-class precision/recall/F1
  breakdown.
- Saves a confusion matrix image to `models/bert-agnews/confusion_matrix.png`
  — open this file to see visually which topics the model confuses most.
- **Note:** this loop doesn't print progress by default, so it can look
  "stuck" for several minutes while it's actually working — that's normal on
  CPU. Lowering `--batch_size` (e.g. to 8) reduces memory pressure if it's
  crashing silently on your machine.

### 3. Try the live demo (`app.py`)
```powershell
streamlit run app.py
```
- Opens a browser tab (usually `http://localhost:8501`) with a text box.
- Type any headline, click **Classify**, and see the predicted topic with a
  confidence breakdown across all 4 classes.
- If Streamlit asks for an email on first run, just press Enter to skip it —
  it's an optional onboarding prompt, unrelated to the app itself.

---

## Troubleshooting notes (things that came up while setting this up)

- **`ModuleNotFoundError: No module named 'datasets'`** — `requirements.txt`
  wasn't installed into the Python environment actually running the script.
  Use `python -m pip install -r requirements.txt` to guarantee it installs
  into the same interpreter you run scripts with.
- **`HfUriError` / dataset `"ag_news"` not found** — Hugging Face renamed the
  dataset repo; the code already uses the correct current name,
  `fancyzhx/ag_news`.
- **`Trainer.__init__() got an unexpected keyword argument 'tokenizer'`** —
  newer `transformers` versions renamed this argument to `processing_class`;
  already fixed in `train.py`.
- **`ModuleNotFoundError: No module named 'torchvision'`** — some
  `transformers` versions try to import unrelated image-model code
  (`ZoeDepth`) on load. Fix: `pip install torchvision`. Not actually used for
  this text-classification task, just a dependency the library expects to
  find.
- **Evaluation hangs after "Using the latest cached version..."** — it's not
  frozen, it's silently doing CPU inference over thousands of examples. Add
  `$env:HF_HUB_OFFLINE = "1"` before running to skip an unnecessary network
  check, and lower `--batch_size` if it crashes without an error (this is
  usually Windows silently killing the process under memory pressure).

---

## Methodology / Approach
1. **Dataset** — [AG News](https://huggingface.co/datasets/fancyzhx/ag_news)
   via the Hugging Face `datasets` library (120k train / 7.6k test headlines,
   4 balanced classes).
2. **Preprocessing** — headlines tokenized with `BertTokenizerFast`
   (`bert-base-uncased` vocab), truncated/padded to 128 tokens.
3. **Fine-tuning** — `BertForSequenceClassification` fine-tuned with the
   Hugging Face `Trainer` API (AdamW, linear LR decay).
4. **Evaluation** — accuracy, weighted F1, macro F1, and a full
   classification report + confusion matrix on the AG News test split.
5. **Deployment** — `app.py` is a Streamlit app that loads the fine-tuned
   model and classifies any headline typed in live.

## Key Results / Observations
Trained on a 2,000-example subset for 1 epoch (CPU-only run):
- **Accuracy: 86.96%**, **Weighted F1: 0.868**
- Per-class F1: World 0.88, **Sports 0.96** (easiest — distinct vocabulary),
  Business 0.79, Sci/Tech 0.84
- **Business and Sci/Tech are the most confused pair** — Business recall was
  only 0.71, meaning a meaningful share of true Business headlines get
  predicted as Sci/Tech (or vice versa). This makes sense: headlines about
  tech companies (earnings, product launches, stock moves) share vocabulary
  with both categories.
- Training on the full 120k-example dataset for more epochs would likely
  raise these numbers further (published benchmarks for fully fine-tuned
  BERT-base on AG News are typically ~94-95% accuracy), at the cost of a much
  longer CPU training time.
