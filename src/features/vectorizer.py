from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# ✅ IMPORTS FROM SRC (CORRECT WAY)
from src.ingestion.data_loader import DataLoader
from src.preprocessing.text_cleaner import clean_text


def build_vectorizer():
    """
    Create and return a TF-IDF vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    return vectorizer


def fit_transform(vectorizer, text_data):
    """
    Fit TF-IDF on text data and transform it
    """
    X = vectorizer.fit_transform(text_data)
    return X


def save_vectorizer(vectorizer, path="artifacts/vectorizer.pkl"):
    """
    Save the trained vectorizer
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)


# ✅ TEST BLOCK
if __name__ == "__main__":
    # Load data
    loader = DataLoader("data/restaurant_reviews.csv")
    df = loader.load_data()

    # Clean text
    df["cleaned_review"] = df["review"].apply(clean_text)

    # TF-IDF
    vectorizer = build_vectorizer()
    X = fit_transform(vectorizer, df["cleaned_review"])

    # Save vectorizer
    save_vectorizer(vectorizer)

    print("✅ TF-IDF vectorization successful!")
    print("Feature matrix shape:", X.shape)
