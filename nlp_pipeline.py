# Integrated NLP Pipeline
# Sentiment Analysis + Named Entity Recognition

import spacy

from sentiment_analysis import (
    vectorizer,
    logistic_model
)

# ---------------------------------------------------------
# LOAD NER MODEL
# ---------------------------------------------------------

nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------
# SENTIMENT FUNCTION
# ---------------------------------------------------------

def analyze_sentiment(text):

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


# ---------------------------------------------------------
# NER FUNCTION
# ---------------------------------------------------------

def extract_entities(text):

    doc = nlp(text)

    entities = []

    for entity in doc.ents:

        if entity.label_ in [
            "PERSON",
            "ORG",
            "GPE",
            "LOC"
        ]:

            if entity.label_ == "PERSON":
                entity_type = "PER"

            elif entity.label_ == "ORG":
                entity_type = "ORG"

            else:
                entity_type = "LOC"

            entities.append(
                (entity.text, entity_type)
            )

    return entities


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

print("\n==========================================")
print("        NLP ANALYSIS SYSTEM")
print("==========================================")

print("\nEnter a sentence or review.")

text = input("\nInput Text: ")

# Sentiment
sentiment, confidence = analyze_sentiment(text)

# NER
entities = extract_entities(text)

# ---------------------------------------------------------
# DISPLAY RESULT
# ---------------------------------------------------------

print("\n==========================================")
print("             NLP RESULT")
print("==========================================")

print(
    "\nSentiment:",
    sentiment
)

print(
    "Sentiment Confidence:",
    round(confidence * 100, 2),
    "%"
)

print("\nNamed Entities:")

if entities:

    for entity, entity_type in entities:

        print(
            f"{entity} -> {entity_type}"
        )

else:

    print("No named entities detected.")

print("\n==========================================")
print("Analysis completed successfully.")
print("==========================================")
