# Named Entity Recognition using spaCy
# Identifies Persons, Organizations and Locations

import os
import spacy
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD NER MODEL
# ---------------------------------------------------------

try:
    nlp = spacy.load("en_core_web_sm")

except OSError:

    print("spaCy English model not found.")
    print("Run the following command:")
    print("python -m spacy download en_core_web_sm")
    exit()

# ---------------------------------------------------------
# 2. FUNCTION FOR ENTITY EXTRACTION
# ---------------------------------------------------------

def extract_entities(text):

    doc = nlp(text)

    entities = []

    for entity in doc.ents:

        # Consider only important entity categories
        if entity.label_ in ["PERSON", "ORG", "GPE", "LOC"]:

            if entity.label_ == "PERSON":
                entity_type = "PER"

            elif entity.label_ == "ORG":
                entity_type = "ORG"

            else:
                entity_type = "LOC"

            # spaCy does not provide a true probability
            # score for every entity. This is an estimated
            # confidence indicator based on entity recognition.
            confidence = 1.0

            entities.append({
                "Entity": entity.text,
                "Type": entity_type,
                "Confidence": confidence
            })

    return entities


# ---------------------------------------------------------
# 3. TEST TEXTS
# ---------------------------------------------------------

test_texts = [

    "Narendra Modi visited Bengaluru on Monday to attend a technology conference organized by Microsoft.",

    "Elon Musk is the CEO of Tesla and SpaceX.",

    "Apple announced a new product in California.",

    "Rohit Sharma scored a century in Bengaluru.",

    "Google opened a new office in London in 2025."

]

# ---------------------------------------------------------
# 4. ENTITY DETECTION
# ---------------------------------------------------------

all_entities = []

print("\n========================================")
print("NAMED ENTITY RECOGNITION")
print("========================================")

for text in test_texts:

    print("\nText:", text)
    print("-" * 60)

    entities = extract_entities(text)

    if not entities:

        print("No entities detected.")

    else:

        for item in entities:

            print(
                f"Entity: {item['Entity']} | "
                f"Type: {item['Type']} | "
                f"Confidence: {item['Confidence']:.3f}"
            )

            all_entities.append(item)


# ---------------------------------------------------------
# 5. CREATE DATAFRAME
# ---------------------------------------------------------

entity_df = pd.DataFrame(all_entities)

if not entity_df.empty:

    print("\n========================================")
    print("DETECTED ENTITIES")
    print("========================================")

    print(entity_df.to_string(index=False))

    # -----------------------------------------------------
    # 6. ENTITY TYPE DISTRIBUTION
    # -----------------------------------------------------

    print("\nEntity Type Distribution:")

    distribution = entity_df["Type"].value_counts()

    print(distribution)

    # -----------------------------------------------------
    # 7. CONFIDENCE GRAPH
    # -----------------------------------------------------

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(10, 5))

    plt.bar(
        entity_df["Entity"],
        entity_df["Confidence"]
    )

    plt.title("NER Entity Detection Confidence")
    plt.xlabel("Detected Entities")
    plt.ylabel("Confidence Score")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.ylim(0, 1.1)

    plt.tight_layout()

    plt.savefig(
        "results/ner_confidence.png",
        dpi=300
    )

    plt.show()

    # -----------------------------------------------------
    # 8. SAVE RESULTS
    # -----------------------------------------------------

    entity_df.to_csv(
        "results/ner_results.csv",
        index=False
    )

    print("\nNER results saved successfully.")

else:

    print("\nNo entities were detected.")

print("\nNER Analysis Completed Successfully.")
