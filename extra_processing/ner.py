import os
import pandas as pd
import spacy

# Paths to raw data directories
lapaz_path = "data/raw/la_paz"
quito_path = "data/raw/quito"

# Load Spanish NER model
nlp = spacy.load("es_core_news_sm")

def detect_entities(text):
    """Detect named entities (PER, LOC, ORG) and return them as a set."""
    doc = nlp(text)
    return {ent.text for ent in doc.ents if ent.label_ in {"PER", "LOC", "ORG"}}

def process_folder(folder_path):
    """Process all .txt files in a folder and extract entities."""
    entities = set()
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("I:"):
                        line_entities = detect_entities(line)
                        entities.update(line_entities)
    return entities

# Process both folders
entities_lapaz = process_folder(lapaz_path)
entities_quito = process_folder(quito_path)

all_entities = entities_lapaz.union(entities_quito)

# Keep only entities with at least one uppercase letter
entities_with_caps = [e for e in all_entities if any(c.isupper() for c in e)]

# Save to CSV
out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "ner_entities_list.csv")

df = pd.DataFrame(sorted(entities_with_caps), columns=["entity"])
df.to_csv(out_path, index=False, encoding="utf-8")

print(f"Entities CSV saved as '{out_path}' ({len(entities_with_caps)} entities)")

