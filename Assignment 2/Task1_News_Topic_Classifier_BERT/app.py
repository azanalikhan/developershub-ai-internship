"""
Streamlit demo app for the fine-tuned BERT news-topic classifier.

Usage:
    streamlit run app.py
"""
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "models" / "bert-agnews"
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_EMOJI = {"World": "🌍", "Sports": "🏅", "Business": "💼", "Sci/Tech": "🔬"}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


st.set_page_config(page_title="News Topic Classifier (BERT)", page_icon="📰")
st.title("📰 News Topic Classifier")
st.caption("Fine-tuned `bert-base-uncased` on AG News — classifies a headline into "
           "World, Sports, Business, or Sci/Tech.")

if not MODEL_DIR.exists():
    st.error(
        f"No fine-tuned model found at `{MODEL_DIR}`. "
        "Run `python train.py` first to fine-tune and save the model."
    )
    st.stop()

tokenizer, model = load_model()

headline = st.text_area(
    "Enter a news headline (or short paragraph):",
    placeholder="e.g. 'Central bank raises interest rates amid inflation fears'",
    height=100,
)

if st.button("Classify", type="primary") and headline.strip():
    with torch.no_grad():
        inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()

    pred_idx = int(probs.argmax())
    pred_label = LABEL_NAMES[pred_idx]

    st.subheader(f"{LABEL_EMOJI[pred_label]} Predicted topic: **{pred_label}**")
    st.write("Confidence breakdown:")
    for name, prob in sorted(zip(LABEL_NAMES, probs), key=lambda x: -x[1]):
        st.progress(float(prob), text=f"{LABEL_EMOJI[name]} {name}: {prob:.1%}")
