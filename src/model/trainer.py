import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Imports from src
from src.ingestion.data_loader import DataLoader
from src.preprocessing.text_cleaner import clean_text
from src.features.vectorizer import build_vectorizer


def train_model():
    # 1. Load dataset
    loader = DataLoader("data/restaurant_reviews.csv")
    df = loader.load_data()

    # 2. Clean text
    df["cleaned_review"] = df["review"].apply(clean_text)

    # 3. TF-IDF vectorization
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(df["cleaned_review"])
    y = df["sentiment"]

    # 4. Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. Evaluate model
    y_pred = model.predict(X_test)
    print("📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # 7. Save model & vectorizer
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    joblib.dump(vectorizer, "artifacts/vectorizer.pkl")

    print("✅ Model training completed and files saved!")


# ✅ Run training
if __name__ == "__main__":
    train_model()
