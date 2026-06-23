# DevelopersHub Corporation — AI/ML Engineering Internship Tasks

**Intern:** Aazan Ali Khan  
**Program:** AI/ML Engineering Internship — DevelopersHub Corporation  
**Due Date:** 26th June, 2026  
**Completed Tasks:** All 6 / 6

---

## Repository Structure

```
├── task1_iris/                    ← Task 1: Iris Dataset EDA
│   └── Task1_Iris_EDA.ipynb
├── task2_stocks/                  ← Task 2: Stock Price Prediction
│   └── Task2_Stock_Price_Prediction.ipynb
├── task3_heart/                   ← Task 3: Heart Disease Prediction
│   └── Task3_Heart_Disease_Prediction.ipynb
├── task4_health_chatbot/          ← Task 4: Health Query Chatbot
│   └── Task4_Health_Chatbot.ipynb
├── task5_mental_health/           ← Task 5: Mental Health Chatbot
│   └── Task5_Mental_Health_Chatbot.ipynb
├── task6_house_prices/            ← Task 6: House Price Prediction
│   └── Task6_House_Price_Prediction.ipynb
└── README.md
```

---

## Task Summaries

### Task 1 — Exploring and Visualizing the Iris Dataset

| Item | Detail |
|------|--------|
| **Objective** | Load, inspect, and visualize the Iris dataset to understand data trends and distributions |
| **Dataset** | Iris Dataset (loaded via `seaborn`) |
| **Skills** | pandas, matplotlib, seaborn, descriptive statistics |
| **Key Findings** | Petal features are the most discriminative; Setosa is linearly separable; strong 0.96 correlation between petal length and width |

**Visualizations produced:** Scatter plots · Histograms · Box plots · Pair plot · Correlation heatmap

---

### Task 2 — Predict Future Stock Prices (Short-Term)

| Item | Detail |
|------|--------|
| **Objective** | Predict the next day's closing price using historical OHLCV data |
| **Dataset** | Apple Inc. (AAPL) historical data via `yfinance` (2020–2024) |
| **Models** | Linear Regression · Random Forest Regressor |
| **Key Findings** | Lag features (previous close) dominate importance; Random Forest achieves R² > 0.98 on test data |

**Features engineered:** 5-day & 20-day moving averages, daily range, % change, 3-day lag prices

---

### Task 3 — Heart Disease Prediction

| Item | Detail |
|------|--------|
| **Objective** | Classify whether a patient is at risk of heart disease |
| **Dataset** | Heart Disease UCI Dataset (Cleveland) via OpenML |
| **Models** | Logistic Regression · Decision Tree Classifier |
| **Metrics** | Accuracy · ROC-AUC · Confusion Matrix · Feature Importance |
| **Key Findings** | Top features: max heart rate (`thalach`), chest pain type (`cp`), ST depression (`oldpeak`). ROC-AUC > 0.88 for both models |

---

### Task 4 — General Health Query Chatbot (Prompt Engineering)

| Item | Detail |
|------|--------|
| **Objective** | Build a chatbot answering general health questions with safety filters |
| **Model** | Claude (claude-sonnet-4-6) via Anthropic API |
| **Skills** | Prompt engineering, API integration, safety handling, multi-turn conversation |
| **Safety Design** | Regex pre-screening layer + LLM system prompt rules; automatic medical disclaimer on all responses |

**Example queries handled:** "What causes a sore throat?" · "Is paracetamol safe for children?" · fever management · blood pressure advice

> **Setup:** Set `ANTHROPIC_API_KEY` as an environment variable before running.

---

### Task 5 — Mental Health Support Chatbot

| Item | Detail |
|------|--------|
| **Objective** | Build an empathetic chatbot for stress, anxiety, and emotional wellness |
| **Fine-Tuning Dataset** | EmpatheticDialogues (Facebook AI) — 32 emotion categories, 24,850 conversations |
| **Base Model** | DistilGPT-2 (fine-tuning demo) + Claude API (production persona) |
| **Skills** | Hugging Face Transformers, Trainer API, prompt engineering, crisis detection |
| **Crisis Protocol** | Detects self-harm/suicidal ideation → immediately provides Pakistan Umang helpline (0317-4288665) |

> **Setup:** Set `ANTHROPIC_API_KEY` as an environment variable before running the chatbot demo cells.

---

### Task 6 — House Price Prediction

| Item | Detail |
|------|--------|
| **Objective** | Predict median house values using property and demographic features |
| **Dataset** | California Housing Dataset (sklearn built-in — equivalent structure to Kaggle house price datasets) |
| **Models** | Linear Regression · Gradient Boosting Regressor |
| **Metrics** | MAE · RMSE · R² |
| **Key Findings** | Median income is the dominant predictor (~50% importance); Gradient Boosting reduces MAE by ~40% vs Linear Regression; engineered features improve accuracy |

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/developershub-aiml-internship.git
cd developershub-aiml-internship
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn yfinance \
            anthropic datasets transformers torch nbformat jupyter
```

### 3. API Key (Tasks 4 & 5)
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 4. Run Notebooks
```bash
jupyter notebook
```
Then open any notebook from the task folders.

---

## Technologies Used

| Category | Libraries |
|----------|-----------|
| Data | pandas, numpy, yfinance, scikit-learn datasets, Hugging Face datasets |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn (LinearRegression, RandomForest, GradientBoosting, DecisionTree, LogisticRegression) |
| Deep Learning / NLP | transformers, torch (DistilGPT-2 fine-tuning) |
| LLM API | anthropic (claude-sonnet-4-6) |
| Notebook | Jupyter |

---

## Submission

Submitted via Google Classroom as required.  
GitHub Repository: `https://github.com/YOUR_USERNAME/developershub-aiml-internship`
