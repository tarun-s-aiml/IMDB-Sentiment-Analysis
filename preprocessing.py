import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt_tab")

df = pd.read_csv("IMDB Dataset.csv")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove unwanted characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)

df["clean_review"] = df["review"].apply(clean_text)

df.to_csv("cleaned_imdb.csv", index=False)

print(df[["review", "clean_review"]].head())

print("\nPreprocessing Completed Successfully!")
