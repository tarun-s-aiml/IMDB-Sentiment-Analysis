# Sentiment Analysis using TF-IDF and Machine Learning
# Models: Logistic Regression and Multinomial Naive Bayes

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

print("Loading IMDB dataset...")

dataset = load_dataset("stanfordnlp/imdb")

train_data = pd.DataFrame(dataset["train"])
test_data = pd.DataFrame(dataset["test"])

print("\nDataset Information")
print("-------------------")
print("Training samples:", len(train_data))
print("Testing samples :", len(test_data))

print("\nClass Distribution:")
print(train_data["label"].value_counts())

# ---------------------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------------------

train_data["text"] = train_data["text"].fillna("").astype(str)
test_data["text"] = test_data["text"].fillna("").astype(str)

X_train_text = train_data["text"]
X_test_text = test_data["text"]

y_train = train_data["label"]
y_test = test_data["label"]

# ---------------------------------------------------------
# 3. TF-IDF FEATURE EXTRACTION
# ---------------------------------------------------------

print("\nPerforming TF-IDF feature extraction...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Training feature matrix:", X_train.shape)
print("Testing feature matrix :", X_test.shape)

# ---------------------------------------------------------
# 4. LOGISTIC REGRESSION
# ---------------------------------------------------------

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print("\nLogistic Regression Accuracy:",
      round(logistic_accuracy * 100, 2), "%")

print("\nLogistic Regression Classification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=["Negative", "Positive"]
    )
)

# ---------------------------------------------------------
# 5. MULTINOMIAL NAIVE BAYES
# ---------------------------------------------------------

print("\nTraining Multinomial Naive Bayes...")

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)

nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)

print("\nMultinomial Naive Bayes Accuracy:",
      round(nb_accuracy * 100, 2), "%")

print("\nNaive Bayes Classification Report:")
print(
    classification_report(
        y_test,
        nb_predictions,
        target_names=["Negative", "Positive"]
    )
)

# ---------------------------------------------------------
# 6. CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    logistic_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="magma",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("Sentiment Analysis Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

os.makedirs("results", exist_ok=True)

plt.tight_layout()
plt.savefig(
    "results/sentiment_confusion_matrix.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------------
# 7. MODEL COMPARISON
# ---------------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Multinomial Naive Bayes"
    ],
    "Accuracy (%)": [
        logistic_accuracy * 100,
        nb_accuracy * 100
    ]
})

print("\nModel Comparison")
print("----------------")
print(results)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    results["Model"],
    results["Accuracy (%)"]
)

plt.title("Comparison of Sentiment Analysis Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------------
# 8. USER INPUT PREDICTION
# ---------------------------------------------------------

def predict_sentiment(text):
    """
    Predict sentiment of a new text using
    the trained Logistic Regression model.
    """

    text_vector = vectorizer.transform([text])

    prediction = logistic_model.predict(text_vector)[0]

    probability = logistic_model.predict_proba(
        text_vector
    )[0]

    if prediction == 1:
        sentiment = "Positive"
        confidence = probability[1]
    else:
        sentiment = "Negative"
        confidence = probability[0]

    return sentiment, confidence


print("\n----------------------------------------")
print("SENTIMENT ANALYSIS DEMONSTRATION")
print("----------------------------------------")

sample_texts = [
    "The movie was absolutely amazing and I loved every moment.",
    "The movie was boring and completely disappointing.",
    "The acting was excellent and the story was very interesting.",
    "I hated this movie. It was a waste of time."
]

for text in sample_texts:

    sentiment, confidence = predict_sentiment(text)

    print("\nText:", text)
    print("Prediction:", sentiment)
    print("Confidence:", round(confidence * 100, 2), "%")

print("\nSentiment Analysis Completed Successfully.")
