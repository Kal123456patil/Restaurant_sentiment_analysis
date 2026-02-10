# ===== Path Fix (IMPORTANT for src imports) =====
import sys
import os
sys.path.append(os.getcwd())

# ===== Imports =====
import streamlit as st
import joblib
import numpy as np

from src.preprocessing.text_cleaner import clean_text

# ===== Page Configuration =====
st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    layout="centered"
)

# ===== Load Model & Vectorizer =====
@st.cache_resource
def load_model():
    model = joblib.load("artifacts/model.pkl")
    vectorizer = joblib.load("artifacts/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===== App Title =====
st.title("🍽️ Restaurant Review Sentiment Analysis")
st.write(
    "Enter a restaurant review below and the app will predict "
    "whether the sentiment is **Positive** or **Negative**."
)

# ===== Text Input (FIXED with UNIQUE KEY) =====
review = st.text_area(
    "✍️ Enter Restaurant Review",
    height=150,
    placeholder="Example: The food was amazing and service was great!",
    key="review_input"
)

# ===== Analyze Button =====
if st.button("Analyze Sentiment", key="analyze_button"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a restaurant review.")
    else:
        # Clean text
        cleaned_review = clean_text(review)

        # Vectorize
        X = vectorizer.transform([cleaned_review])

        # Prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = np.max(probabilities) * 100

        # ===== Display Result =====
        st.subheader("📊 Prediction Result")

        if prediction == "positive":
            st.success(f"😊 Sentiment: POSITIVE ({confidence:.2f}%)")
        else:
            st.error(f"😞 Sentiment: NEGATIVE ({confidence:.2f}%)")

        # Show cleaned text
        st.write("🧹 **Cleaned Review:**")
        st.code(cleaned_review)

        # Confidence bar
        st.write("📈 **Prediction Confidence**")
        st.progress(confidence / 100)
