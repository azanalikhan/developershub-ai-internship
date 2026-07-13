"""
Load the exported churn pipeline and run predictions on new, raw customer
records (no manual preprocessing needed -- it's baked into the pipeline).

Usage:
    python predict.py
"""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "churn_pipeline.joblib"


def sample_new_customers() -> pd.DataFrame:
    """A couple of raw, unprocessed customer rows (as you'd get from a live system)."""
    return pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 171.0,
            },
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 65,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 60.0,
                "TotalCharges": 3900.0,
            },
        ]
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python train_pipeline.py` first to train and export it."
        )
    pipeline = joblib.load(MODEL_PATH)
    new_customers = sample_new_customers()

    preds = pipeline.predict(new_customers)
    probs = pipeline.predict_proba(new_customers)[:, 1]

    for i, (pred, prob) in enumerate(zip(preds, probs)):
        label = "CHURN" if pred == 1 else "STAY"
        print(f"Customer {i}: prediction={label}  churn_probability={prob:.3f}")


if __name__ == "__main__":
    main()
