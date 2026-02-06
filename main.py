"""
Main entry point for the SVM project.
Runs the full pipeline:
1. Data Splitting (Train/Test)
2. Training Lexical Model
3. Training Syntactic Model
4. Final Evaluation & Comparison
"""

import sys
import os

# Ensure the root directory is in the path so we can import from models/
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from models.splitting import create_splits
    from models.training_lexical import train_lexical
    from models.training_syntactical import train_syntactic
    from models.evaluation import run_evaluation
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure you have created the files: splitting.py, training_lexical.py, "
          "training_syntactical.py, and evaluation.py inside the 'models/' directory.")
    sys.exit(1)


def main():
    print("STARTING FULL PIPELINE")
    print("=" * 60)

    # Step 1: Split Data
    # Creates data/splits/train.csv and data/splits/test.csv
    try:
        create_splits(input_path="data/clean/cleaned_data.csv")
    except Exception as e:
        print(f"Pipeline failed at splitting stage: {e}")
        return

    # Step 2: Train Lexical Model
    # Uses data/splits/train.csv -> Saves models/svm_lexical.joblib
    try:
        train_lexical(train_csv_path="data/splits/train.csv")
    except Exception as e:
        print(f"Pipeline failed at lexical training stage: {e}")
        return

    # Step 3: Train Syntactic Model
    # Uses data/splits/train.csv -> Saves models/svm_syntactic.joblib
    try:
        train_syntactic(train_csv_path="data/splits/train.csv")
    except Exception as e:
        print(f"Pipeline failed at syntactic training stage: {e}")
        return

    # Step 4: Final Evaluation
    # Uses data/splits/test.csv and loaded models -> Prints comparison
    try:
        run_evaluation(test_csv_path="data/splits/test.csv")
    except Exception as e:
        print(f"Pipeline failed at evaluation stage: {e}")
        return

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
