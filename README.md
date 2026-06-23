# DevelopersHub Corporation — AI/ML Engineering Internship

**Intern:** Aazan Ali Khan  
**Program:** AI/ML Engineering Internship — DevelopersHub Corporation  
**GitHub:** [azanalikhan/developershub-ai-internship](https://github.com/azanalikhan/developershub-ai-internship)

---

## Repository Structure

```
developershub-ai-internship/
├── task1_iris/
│   └── Task1_Iris_EDA.ipynb
├── task2_stocks/
│   └── Task2_Stock_Price_Prediction.ipynb
├── task3_heart/
│   └── Task3_Heart_Disease_Prediction.ipynb
├── task4_health_chatbot/
│   └── Task4_Health_Chatbot.ipynb
├── task5_mental_health/
│   └── Task5_Mental_Health_Chatbot.ipynb
├── task6_house_prices/
│   └── Task6_House_Price_Prediction.ipynb
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Task Summaries

### Task 1 — Exploring and Visualizing the Iris Dataset

| Item | Detail |
|------|--------|
| **Objective** | Load, inspect, and visualize the Iris dataset to understand data trends and distributions |
| **Dataset** | Iris Dataset (loaded via `seaborn`) |
| **Libraries** | pandas, matplotlib, seaborn |
| **Key Results** | Petal features are the most discriminative; Setosa is linearly separable; 0.96 correlation between petal length and width |

**Visualizations:** Scatter plots · Histograms · Box plots · Pair plot · Correlation heatmap

---

### Task 2 — Stock Price Prediction (Short-Term)

| Item | Detail |
|------|--------|
| **Objective** | Predict next day's closing price using historical OHLCV data |
| **Dataset** | Apple Inc. (AAPL) via `yfinance` (2020–2024) |
| **Models** | Linear Regression · Random Forest Regressor |
| **Key Results** | Lag features dominate importance; Random Forest achieves R² > 0.98 on test data |

**Features engineered:** 5-day & 20-day moving averages, daily range, % change, 3-day lag prices

---

### Task 3 — Heart Disease Prediction

| Item | Detail |
|------|--------|
| **Objective** | Classify whether a patient is at risk of heart disease |
| **Dataset** | Heart Disease UCI Dataset (Cleveland) via OpenML |
| **Models** | Logistic Regression · Decision Tree Classifier |
| **Metrics** | Accuracy · ROC-AUC · Confusion Matrix · Feature Importance |
| **Key Results** | Top features: `thalach`, `cp`, `oldpeak`, `ca` — ROC-AUC > 0.88 for both models |

---

### Task 4 — General Health Query Chatbot

| Item | Detail |
|------|--------|
| **Objective** | Build a chatbot answering general health questions with safety filters |
| **Model** | Llama 3 (Meta) via Groq API — free, no credit card required |
| **Skills** | Prompt engineering, API integration, safety handling, multi-turn conversation |
| **Safety Design** | Two-layer system: regex pre-screening + LLM system prompt rules + automatic disclaimer |

**Example queries handled:** What causes a sore throat? · Is paracetamol safe for children? · Early signs of diabetes · Fever management

> **Requires:** Free Groq API key — see Setup Instructions below

---

### Task 5 — Mental Health Support Chatbot

| Item | Detail |
|------|--------|
| **Objective** | Build an empathetic chatbot for stress, anxiety, and emotional wellness |
| **Fine-Tuning Dataset** | EmpatheticDialogues (Facebook AI) — 32 emotion categories |
| **Base Model** | DistilGPT-2 (fine-tuning demo via Hugging Face Trainer API) |
| **Production Chatbot** | Llama 3 via Groq API with empathetic persona prompt |
| **Crisis Protocol** | Detects self-harm keywords → provides Pakistan Umang helpline (0317-4288665) |

> **Requires:** Free Groq API key — see Setup Instructions below

---

### Task 6 — House Price Prediction

| Item | Detail |
|------|--------|
| **Objective** | Predict median house values using property and demographic features |
| **Dataset** | California Housing Dataset (sklearn built-in) |
| **Models** | Linear Regression · Gradient Boosting Regressor |
| **Metrics** | MAE · RMSE · R² |
| **Key Results** | Median income is the dominant predictor (~50% importance); Gradient Boosting reduces MAE by ~40% vs Linear Regression |

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/azanalikhan/developershub-ai-internship.git
cd developershub-ai-internship
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. API Key Setup (Tasks 4 & 5 only)

Tasks 4 and 5 use the **Groq API** which is completely free — no credit card required.

**Get your free key:**
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with Google or GitHub
3. Click **API Keys** → **Create API Key** → copy it

**Add it to the notebooks:**

In `Task4_Health_Chatbot.ipynb` — Cell 2:
```python
GROQ_API_KEY = "gsk_your_key_here"
```

In `Task5_Mental_Health_Chatbot.ipynb` — find the `mindease_respond` function cell:
```python
GROQ_API_KEY = "gsk_your_key_here"
```

> ⚠️ **Never commit your actual API key to GitHub.** Use `.env.example` as a reference and keep your real key only in the notebook locally.

### 5. Run the Notebooks
```bash
jupyter notebook
```
Then open any notebook from its task folder and run **Kernel → Restart & Run All**.

---

## Technologies Used

| Category | Libraries / Tools |
|----------|-------------------|
| **Data** | pandas, numpy, yfinance, scikit-learn datasets |
| **Visualization** | matplotlib, seaborn |
| **Machine Learning** | scikit-learn (LinearRegression, RandomForest, GradientBoosting, DecisionTree, LogisticRegression) |
| **Deep Learning / NLP** | transformers, torch (DistilGPT-2 fine-tuning), datasets (Hugging Face) |
| **LLM API** | Groq (Llama 3.3 70B) |
| **Notebook** | Jupyter |

---

## Key Learnings

- **EDA** with pandas and seaborn reveals feature importance before any model is trained
- **Feature engineering** (lag features, moving averages, engineered ratios) consistently improves model performance
- **Stratified splitting** is essential for classification tasks with imbalanced classes
- **Prompt engineering** is the single most impactful lever for LLM response quality and safety
- **Two-layer safety systems** (regex + LLM system prompt) provide robust protection in health chatbots
- **Gradient Boosting** outperforms Linear Regression significantly on real-world tabular data

---


