import pandas as pd
import joblib
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.tokenizer import tokenize_dataframe
from src.preprocessing.pos_tagger import pos_tag_dataframe

def load_model_and_vectorizer(model_path, vect_path):
    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        print(f"Warning: Model or vectorizer not found at {model_path} / {vect_path}")
        return None, None
    return joblib.load(model_path), joblib.load(vect_path)

def print_top_features(vectorizer, model, name, n=10, ignore_substrings=None):
    """
    Extracts and prints the most discriminative features for a Linear SVM.
    
    Args:
        vectorizer: The fitted vectorizer
        model: The fitted SVM model pipeline
        name: Name of the model for display
        n: Number of top features to display per class
        ignore_substrings: Tuple of substrings to filter out from feature names (e.g., ("entity",))
    """
    if ignore_substrings is None:
        ignore_substrings = ()
    
    # Access the SVC step from the pipeline
    svc = model.named_steps['svc']
    
    if svc.kernel != 'linear':
        print(f"\n[!] Cannot extract features for {name}: Model must use a 'linear' kernel.")
        return

    # Get feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients (weights). For binary, it's a single row
    # .toarray() is used because SVM coefficients can be sparse
    coefs = svc.coef_.toarray()[0] 
    
    # Pair weights with names and sort
    features_coefs = sorted(zip(feature_names, coefs), key=lambda x: x[1])
    
    # Filter out features containing ignored substrings
    filtered_features = [
        (feat, coef) for feat, coef in features_coefs
        if not any(substr in feat for substr in ignore_substrings)
    ]

    print(f"\n--- Top {n} Discriminative Features: {name} ---")
    
    # Class 1 (La Paz) - Lowest/Most Negative weights
    print(f"Top for La Paz (Class 1):")
    for feat, coef in filtered_features[:n]:
        print(f"  {feat:<20} : {coef:.4f}")

    # Class 2 (Quito) - Highest/Most Positive weights
    print(f"\nTop for Quito (Class 2):")
    for feat, coef in filtered_features[:-(n + 1):-1]:
        print(f"  {feat:<20} : {coef:.4f}")

def evaluate_model(name, df_test, model, vectorizer, feature_col, needs_pos=False):
    print(f"\n--- Evaluating {name} Model ---")
    
    df_eval = df_test.copy()
    if needs_pos:
        df_eval = tokenize_dataframe(df_eval, text_column="line")
        df_eval = pos_tag_dataframe(df_eval, tokens_column="tokens")
        text_data = df_eval[feature_col].apply(lambda x: " ".join(x))
    else:
        df_eval = tokenize_dataframe(df_eval, text_column="line")
        text_data = df_eval[feature_col].apply(lambda x: " ".join(x))

    X_test = vectorizer.transform(text_data)
    y_test = df_eval["label"].values
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_mic = f1_score(y_test, y_pred, average='micro')
    f1_mac = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (Micro): {f1_mic:.4f}")
    print(f"  F1 (Macro): {f1_mac:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    return {"accuracy": acc, "f1_micro": f1_mic, "f1_macro": f1_mac, "confusion_matrix": cm}

def run_evaluation(test_csv_path="data/splits/test.csv", model_dir="models"):
    print("=" * 60)
    print("FINAL MODEL EVALUATION (Test Set)")
    print("=" * 60)

    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test file not found: {test_csv_path}.")

    df_test = pd.read_csv(test_csv_path)

    # Load Models
    lex_model, lex_vect = load_model_and_vectorizer(
        os.path.join(model_dir, "svm_lexical.joblib"),
        os.path.join(model_dir, "vectorizer_lexical.joblib")
    )
    
    syn_model, syn_vect = load_model_and_vectorizer(
        os.path.join(model_dir, "svm_syntactic.joblib"),
        os.path.join(model_dir, "vectorizer_syntactic.joblib")
    )

    results = {}

    # 1. Evaluate and Show Lexical Features
    if lex_model and lex_vect:
        results["Lexical"] = evaluate_model(
            "Lexical", df_test, lex_model, lex_vect, feature_col="tokens", needs_pos=False
        )
        print_top_features(lex_vect, lex_model, "Lexical Model", ignore_substrings=("entity",) )

    # 2. Evaluate and Show Syntactic Features
    if syn_model and syn_vect:
        results["Syntactic"] = evaluate_model(
            "Syntactic", df_test, syn_model, syn_vect, feature_col="pos_sequence", needs_pos=True
        )
        print_top_features(syn_vect, syn_model, "Syntactic Model")

    # 3. Comparison
    if "Lexical" in results and "Syntactic" in results:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        lex = results["Lexical"]
        syn = results["Syntactic"]
        wins = {"Lexical": 0, "Syntactic": 0}
        
        for metric in ["accuracy", "f1_micro", "f1_macro"]:
            l_score, s_score = lex[metric], syn[metric]
            winner = "Lexical" if l_score > s_score else "Syntactic" if s_score > l_score else "Tie"
            if winner != "Tie": wins[winner] += 1
            print(f"{metric:<15} {l_score:.4f}       {s_score:.4f}       {winner}")

if __name__ == "__main__":
    run_evaluation()
