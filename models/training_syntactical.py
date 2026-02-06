import pandas as pd
import joblib
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.tokenizer import tokenize_dataframe
from src.preprocessing.pos_tagger import pos_tag_dataframe
from src.features.feature_extraction import build_tfidf_features

def train_syntactic(train_csv_path="data/splits/train.csv", model_dir="models"):
    print("=" * 60)
    print("TRAINING SYNTACTIC MODEL (POS Tags)")
    print("=" * 60)

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Train file not found: {train_csv_path}. Please run splitting.py first.")

    # 1. Load Training Data
    df_train = pd.read_csv(train_csv_path)
    
    # 2. Preprocessing (Tokenize + POS Tag)
    print("Tokenizing and POS tagging training data...")
    df_train = tokenize_dataframe(df_train, text_column="line")
    df_train = pos_tag_dataframe(df_train, tokens_column="tokens")

    # 3. Build TF-IDF Features (POS Unigrams + Bigrams)
    print("Building features...")
    X_train, vectorizer = build_tfidf_features(
        df_train,
        column="pos_sequence",
        ngram_range=(1, 2),
        max_features=100
    )
    y_train = df_train["label"].values

    # 4. Define Model Pipeline
    model = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('svc', SVC(kernel='linear', C=10, probability=True, random_state=42))
    ])

    # 5. Internal Cross-Validation
    print("\nRunning internal 5-Fold Cross-Validation on Training Set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_num = 1
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        print(f"  Fold {fold_num}: train={X_tr.shape}, val={X_val.shape}")

        # Fit on fold train and evaluate on fold validation
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1_mic = f1_score(y_val, y_pred, average='micro')
        f1_mac = f1_score(y_val, y_pred, average='macro')

        print(f"    Accuracy: {acc:.4f}, F1(micro): {f1_mic:.4f}, F1(macro): {f1_mac:.4f}")
        fold_num += 1

    # 6. Train Final Model on All Training Data
    print("\nFitting final model on full training set...")
    model.fit(X_train, y_train)

    # 7. Save Model and Vectorizer
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "svm_syntactic.joblib")
    vect_path = os.path.join(model_dir, "vectorizer_syntactic.joblib")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vect_path)

    print(f"✓ Model saved to {model_path}")
    print(f"✓ Vectorizer saved to {vect_path}")

if __name__ == "__main__":
    train_syntactic()
