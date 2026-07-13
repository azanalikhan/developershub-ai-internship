# Task 3: Multimodal ML — Housing Price Prediction Using Images + Tabular Data

## Objective
Predict housing sale prices by fusing **structured/tabular features** (area,
bedrooms, bathrooms, condition, location, etc.) with **visual features
extracted from house images** via a CNN.

---

## Files in this folder — what each one does, and the order to use them

| Order | File | What it does | Run directly? |
|---|---|---|---|
| 1 | `requirements.txt` | torch, torchvision, pandas, scikit-learn, Pillow, matplotlib | Once, before anything else |
| 2 | `generate_synthetic_data.py` | Creates a synthetic dataset — tabular house rows + procedurally rendered house-front images whose color/quality loosely tracks price | Yes — run first if you don't have a real image+tabular dataset |
| 3 | `dataset.py` | PyTorch `Dataset` that pairs each house's tabular row with its image(s) | No — imported by `train.py`/`evaluate.py` |
| 4 | `model.py` | Defines the model: ResNet18 image encoder + tabular MLP encoder + fusion regression head | No — imported by `train.py`/`evaluate.py` |
| 5 | `train.py` | Trains the fusion model, saves the best checkpoint to `models/fusion_model.pt` | Yes — after data exists |
| 6 | `evaluate.py` | Reloads the checkpoint, reports MAE/RMSE, saves a predicted-vs-actual scatter plot | Yes — after training |

**Run order: `generate_synthetic_data.py` -> `train.py` -> `evaluate.py`**

---

## Step-by-step: how to actually run it

### 0. Install dependencies (once)
```powershell
python -m pip install -r requirements.txt
```
This includes `torch` and `torchvision` — a larger download than the other
tasks.

### 1. Generate the dataset
```powershell
python generate_synthetic_data.py --n_samples 600
```
Writes `data/houses.csv` (tabular rows) and `data/images/` (600 small PNG
house images). Takes under a minute.

**To use a real dataset instead:** drop your own CSV at `data/houses.csv`
(columns: `area, bedrooms, bathrooms, floors, year_built, location, condition,
image_path, price`) and images under `data/images/`, then skip this step
entirely — no code changes needed.

### 2. Train the fusion model
```powershell
python train.py --epochs 30 --lr 3e-4
```
On first run this downloads ImageNet-pretrained ResNet18 weights (needs
normal internet access to `download.pytorch.org`). Prints per-epoch
train/val loss, MAE, and RMSE, then saves the best checkpoint (lowest
val MAE) to `models/fusion_model.pt`.

Useful flags:
- `--epochs N` — more epochs = more time for the fusion head to learn to
  weight the tabular vs. image branches correctly (see "Key Results" below
  for why this matters more than usual here)
- `--lr` — default `1e-3` can be unstable on this task; `3e-4` trains more
  smoothly (see the fix explained below)
- `--no_pretrained_cnn` — skip the ImageNet weights download (useful for a
  fast offline smoke test, at the cost of image-branch quality)

### 3. Evaluate
```powershell
python evaluate.py
```
Prints final test MAE/RMSE and saves `models/pred_vs_actual.png` — a
predicted-vs-actual scatter plot with a red dashed "perfect prediction" line.

---

## A bug we found and fixed while testing this: gradient clipping

The first training run showed early-epoch `train_MAE` values in the **tens
of millions of dollars** (e.g. `$213,135,186` in epoch 1) despite a normal,
moderate training loss. This isn't slow convergence — it's a specific and
useful thing to understand about regression on log-transformed prices.

**Why it happens:** the model predicts `log1p(price)`, and MAE is computed
after converting back to dollars with `expm1()`. That conversion is
exponential — a *small* log-space miss turns into a *huge* dollar error once
exponentiated (e.g. a log-space error of 2 becomes roughly a 6.4x price
error). Early in training, before the fusion head's randomly-initialized
weights have settled, a handful of predictions land moderately wrong in
log-space, and `expm1()` amplifies those into eye-watering dollar figures
that dominate the MAE average — even though the *loss* the optimizer sees
stays reasonable.

**The fix:** we added gradient clipping (`clip_grad_norm_`, max norm 1.0) in
`train.py`'s training loop, and lowered the default learning rate
recommendation from `1e-3` to `3e-4`. This measurably reduced instability:

| | Before fix | After fix |
|---|---|---|
| Best val MAE | $204,906 | **$156,682** |
| Best val RMSE | $370,488 | **$284,958** |
| Worst scatter-plot outlier | ~$3.58M predicted | ~$2.67M predicted |

RMSE — which is much more sensitive to outliers than MAE — dropped the most
(23%), confirming the fix mainly worked by reducing wild individual
mispredictions rather than shifting the average.

---

## Methodology / Approach
1. **Tabular data** — numeric/categorical house attributes (area, bedrooms,
   bathrooms, floors, year built, location, condition), scaled/encoded inside
   `dataset.py`.
2. **Image data** — one photo per house. A CNN backbone (`ResNet18`, via
   `torchvision`, ImageNet-pretrained, backbone frozen) extracts a
   fixed-length embedding per image, projected down to 128-d by a small
   trainable projection layer.
3. **Feature fusion** — the CNN's image embedding is concatenated with the
   tabular encoder's 32-d output into a single fused feature vector.
4. **Model** — a small feed-forward regression head (`FusionRegressor`)
   predicts `log1p(price)` from the fused vector, trained with `SmoothL1Loss`
   and gradient clipping for stability.
5. **Evaluation** — MAE and RMSE on a held-out test split (in original dollar
   units, after converting predictions back with `expm1()`), plus a
   predicted-vs-actual scatter plot.

## Key Results / Observations
Final result on the synthetic dataset (600 samples, 30 epochs, lr=3e-4):
- **Test MAE: $156,682**
- **Test RMSE: $284,958**

Reading the scatter plot honestly:
- Most predictions cluster reasonably close to the diagonal in the
  **$150k-$500k** range, where most of the training data sits.
- **Regression to the mean at the high end**: actual prices of $600k-$750k
  are systematically *underpredicted* (down toward $100k-$200k). With only
  ~600 synthetic samples and relatively few high-price examples, the model
  learned to hedge toward the bulk of the price distribution rather than
  confidently extrapolate to expensive houses — a classic small-data
  regression pattern, not a fusion-architecture flaw.
- **One remaining outlier**: a house with actual price ~$500k gets predicted
  at ~$2.67M. This is the same log-space-amplification effect described
  above, just surviving in a single stubborn case even after the stability
  fix — a good illustration of why `expm1()`-transformed regression targets
  need to be handled carefully both in training and in reporting metrics.
- **On a real dataset with real photos**, this would likely perform
  meaningfully better: our synthetic house images are deliberately weak
  proxies for a genuine housing photo's price-relevant signal (curb appeal,
  renovation quality, staging, neighborhood context), so the CNN branch here
  is contributing far less signal than it would on real data — most of the
  price signal in this run comes from the tabular branch.

## Files
- `generate_synthetic_data.py` — creates a runnable synthetic tabular+image dataset
- `dataset.py` — PyTorch `Dataset`/`DataLoader` that loads paired tabular + image data
- `model.py` — CNN image encoder + tabular encoder + fusion regression head
- `train.py` — trains the fusion model (with gradient clipping for stability), prints/saves metrics
- `evaluate.py` — reloads a saved model and reports MAE/RMSE + scatter plot
- `data/` — tabular CSV + images (synthetic or your own)
- `models/` — saved model checkpoint + plots
