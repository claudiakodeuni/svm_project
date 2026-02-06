import stanza
import pandas as pd
import os
from collections import Counter

# -----------------------------
# SETTINGS
# -----------------------------
CLEANED_CSV = "data/clean/cleaned_data.csv"
STANDARD_SPANISH_FILE = "extra_processing/data/lexicons/standard_spanish.dic"
OUTPUT_DIR = "extra_processing/data/lexicons"
FREQ_THRESHOLD = 2

# -----------------------------
# 1. Load Stanza Spanish pipeline
# -----------------------------
stanza_nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')
print("Stanza Spanish lemmatizer loaded.")

# -----------------------------
# 2. Function to lemmatize text
# -----------------------------
def lemmatize_text(text):
    doc = stanza_nlp(text)
    lemmas = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.lemma.isalpha():
                lemmas.append(word.lemma.lower())
    return lemmas

# -----------------------------
# 3. Load cleaned corpus
# -----------------------------
df = pd.read_csv(CLEANED_CSV)
print(f"Loaded {len(df)} lines.")

# -----------------------------
# 4. Load & expand standard Spanish lexicon
# -----------------------------
def expand_forms(word):
    word = word.strip().lower()
    if not word:
        return []
    parts = word.split("/")
    base = parts[0]
    variants = {base}
    for suf in parts[1:]:
        if suf:
            if len(suf) == 1:
                variants.add(base + suf)
            else:
                variants.add(base[:-len(suf)] + suf)
    return variants

expanded_lex = []
with open(STANDARD_SPANISH_FILE, encoding="utf-8") as f:
    for line in f:
        expanded_lex.extend(expand_forms(line))

# LEMMATIZE THE LEXICON USING STANZA
lexicon_lemmas = set()
for w in expanded_lex:
    lem = lemmatize_text(w)
    if lem:
        lexicon_lemmas.add(lem[0])

print(f"Standard Spanish lemma lexicon size: {len(lexicon_lemmas)}")

# -----------------------------
# 5. Lemmatize corpus & build counters
# -----------------------------
lapaz_counter = Counter()
quito_counter = Counter()
corpus_vocab = set()

for text, label in zip(df["line"], df["label"]):
    lemmas = lemmatize_text(text)
    corpus_vocab.update(lemmas)

    if label == "lapaz":
        lapaz_counter.update(lemmas)
    else:
        quito_counter.update(lemmas)

print(f"Lemmatized corpus vocabulary size: {len(corpus_vocab)}")

# -----------------------------
# 6. City-only sets
# -----------------------------
lapaz_only = set(lapaz_counter) - set(quito_counter)
quito_only = set(quito_counter) - set(lapaz_counter)

# -----------------------------
# 7. Remove standard Spanish lemmas
# -----------------------------
lapaz_nonstandard = {w for w in lapaz_only if w not in lexicon_lemmas}
quito_nonstandard = {w for w in quito_only if w not in lexicon_lemmas}

# Frequency filter
lapaz_nonstandard = {w for w in lapaz_nonstandard if lapaz_counter[w] >= FREQ_THRESHOLD}
quito_nonstandard = {w for w in quito_nonstandard if quito_counter[w] >= FREQ_THRESHOLD}

# -----------------------------
# 8. SAVE OUTPUT
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_set(s, name):
    with open(os.path.join(OUTPUT_DIR, name), "w", encoding="utf-8") as f:
        for w in sorted(s):
            f.write(w + "\n")

save_set(corpus_vocab, "corpus_vocab_lemmas.txt")
save_set(lexicon_lemmas, "standard_spanish_lemmas.txt")
save_set(lapaz_nonstandard, "lapaz_nonstandard_lemmas.txt")
save_set(quito_nonstandard, "quito_nonstandard_lemmas.txt")

print("Done! Lexicons saved.")

