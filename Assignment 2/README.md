# AI/ML Engineering — Advanced Internship Tasks
**DevelopersHub Corporation | Due: 21st July, 2026**

All 5 advanced tasks are complete, tested, and working end-to-end. Each task
lives in its own self-contained directory with its own code,
`requirements.txt`, and `README.md` covering objective, methodology, file
usage order, troubleshooting notes, and real results.

| # | Task | Directory | Result |
|---|------|-----------|--------|
| 1 | News Topic Classifier Using BERT | [`Task1_News_Topic_Classifier_BERT/`](Task1_News_Topic_Classifier_BERT) | 86.96% accuracy, 0.868 weighted F1 |
| 2 | End-to-End ML Pipeline (Scikit-learn) | [`Task2_ML_Pipeline_Churn/`](Task2_ML_Pipeline_Churn) | Random Forest, ROC-AUC 0.843, exported production pipeline |
| 3 | Multimodal ML — Housing Price Prediction | [`Task3_Multimodal_Housing_Price/`](Task3_Multimodal_Housing_Price) | MAE $156,682, RMSE $284,958 |
| 4 | Context-Aware Chatbot (LangChain/RAG) | [`Task4_RAG_Chatbot/`](Task4_RAG_Chatbot) | Verified grounded retrieval + working multi-turn memory |
| 5 | Auto Tagging Support Tickets Using LLM | [`Task5_Auto_Tagging_LLM/`](Task5_Auto_Tagging_LLM) | Few-shot beat zero-shot on precision, recall, and tag accuracy |

The internship only requires **3 of 5** — all 5 are complete here for a
stronger portfolio.

## Quick start (per task)
Each task directory is independent — `cd` into it, install its
`requirements.txt`, and follow the "Files in this folder" + "Step-by-step"
sections at the top of that task's README:

```bash
cd Task2_ML_Pipeline_Churn
pip install -r requirements.txt
python train_pipeline.py
```

## LLM backend note (Tasks 4 and 5)
Tasks 4 and 5 call an LLM API. Both support three interchangeable backends
via the `LLM_BACKEND` environment variable — use whichever you have a key
for:

| Backend | `LLM_BACKEND` value | API key env var |
|---|---|---|
| Groq | `groq` | `GROQ_API_KEY` |
| Anthropic | `anthropic` (default) | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |

```powershell
$env:LLM_BACKEND = "groq"
$env:GROQ_API_KEY = "gsk_..."
```

Task 1 and Task 3 also reach out to Hugging Face Hub / `download.pytorch.org`
for models and datasets — normal internet access is required, no API key.
Task 2 is fully self-contained and needs neither.

## Repo structure
```
AIML_Advanced_Internship/
├── README.md                              <- you are here
├── Task1_News_Topic_Classifier_BERT/
│   ├── README.md
│   ├── requirements.txt
│   ├── train.py
│   ├── evaluate.py
│   ├── app.py                             <- Streamlit deployment
│   └── models/                            <- fine-tuned model + confusion matrix
├── Task2_ML_Pipeline_Churn/
│   ├── README.md
│   ├── requirements.txt
│   ├── train_pipeline.py
│   ├── predict.py
│   ├── data/Telco-Customer-Churn.csv
│   └── models/                            <- churn_pipeline.joblib, metrics, plots
├── Task3_Multimodal_Housing_Price/
│   ├── README.md
│   ├── requirements.txt
│   ├── generate_synthetic_data.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py                           <- includes gradient clipping fix
│   ├── evaluate.py
│   └── data/ , models/
├── Task4_RAG_Chatbot/
│   ├── README.md
│   ├── requirements.txt
│   ├── ingest.py
│   ├── chatbot.py                         <- supports Groq/Anthropic/OpenAI
│   ├── app.py                             <- Streamlit deployment
│   └── data/corpus/                       <- sample knowledge base
└── Task5_Auto_Tagging_LLM/
    ├── README.md
    ├── requirements.txt
    ├── tagging.py                         <- supports Groq/Anthropic/OpenAI
    ├── evaluate.py
    └── data/support_tickets.csv
```

## What each task actually demonstrates

- **Task 1 (BERT)** — fine-tuned `bert-base-uncased` on AG News. Real
  finding: Business and Sci/Tech are the most confused pair (Business recall
  only 0.71), reflecting genuine vocabulary overlap between tech-company and
  finance headlines.
- **Task 2 (ML Pipeline)** — full scikit-learn `Pipeline` with
  `ColumnTransformer` preprocessing, `GridSearchCV`-tuned Logistic Regression
  and Random Forest, exported as one self-contained `.joblib` artifact.
  Random Forest was chosen for higher recall (0.794) despite lower raw
  accuracy — a deliberate, explained trade-off for a churn use case.
- **Task 3 (Multimodal)** — CNN (ResNet18) + tabular fusion regression.
  Includes a real debugging story: an early run showed unstable training
  (train MAE briefly in the tens of millions of dollars due to
  `expm1()`-amplified log-space errors), diagnosed and fixed with gradient
  clipping + a lower learning rate, improving MAE by ~24% and RMSE by ~23%.
- **Task 4 (RAG Chatbot)** — `ConversationalRetrievalChain` with FAISS
  retrieval + `ConversationBufferMemory`. Verified with a real transcript
  showing the bot correctly resolving an ambiguous follow-up question
  ("And what are the core hours?") using conversation history, with every
  answer grounded and source-cited.
- **Task 5 (LLM Tagging)** — zero-shot vs few-shot ticket tagging. Found a
  concrete case (ticket 10) where few-shot prompting corrected a zero-shot
  misclassification (a "permission denied" ticket wrongly tagged as a login
  issue by zero-shot, correctly tagged as a Bug by few-shot).

## Submission checklist mapping
Per the internship instructions, each task directory's `README.md` covers:
- **Objective** of the task
- **Methodology / Approach**
- **Key results or observations** (with real numbers/transcripts, not
  placeholders)

...and each task's code covers: dataset loading & preprocessing, model
development & training, evaluation with relevant metrics, and (where
applicable) a deployment/demo entry point (`app.py`). Push each task
directory to GitHub and submit the link(s) via Google Classroom.
