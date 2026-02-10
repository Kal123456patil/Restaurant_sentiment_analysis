import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
def clean_text(text):
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # 4. Join words back into sentence
    cleaned_text = " ".join(words)

    return cleaned_text
if __name__ == "__main__":
    sample_text = "The food was AMAZING!!! and the service was excellent 😍"
    print("Original Text:", sample_text)
    print("Cleaned Text:", clean_text(sample_text))
