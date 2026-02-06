import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_splits(input_path, output_dir="data/splits", test_size=0.2, random_state=42):
    """
    Loads the cleaned data, maps labels to integers, splits into train/test sets,
    and saves them to CSV files to ensure consistency across training and evaluation.
    """
    print("=" * 60)
    print("DATA SPLITTING")
    print("=" * 60)

    # 1. Load Data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples from {input_path}")

    # 2. Map Labels (Pre-processing step common to all)
    # Mapping: lapaz -> 1, quito -> 2
    label_map = {"lapaz": 1, "quito": 2}
    if df["label"].dtype == object:
        df["label"] = df["label"].map(label_map)
        # Drop rows where label might be NaN if mapping failed
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        print("Mapped labels to integers: {'lapaz': 1, 'quito': 2}")

    # 3. Stratified Split
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )

    # 4. Save Splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"\nSplits saved to '{output_dir}/':")
    print(f"  - Train: {len(df_train)} samples -> {train_path}")
    print(f"  - Test:  {len(df_test)} samples -> {test_path}")

if __name__ == "__main__":
    # Adjust path if necessary
    create_splits("data/clean/cleaned_data.csv")
