"""Data Preparation Script for SuperKart MLOps Pipeline."""

import os
import sys
import numpy as np
import pandas as pd


def clean_and_prepare_data(
    raw_path: str = "data/SuperKart.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data, clean anomalies, engineer features, and split into train/test."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at '{raw_path}'")

    df = pd.read_csv(raw_path)
    print(f"[INFO] Loaded raw dataset: {df.shape}")

    # 1. Standardize inconsistent categorical labels in Product_Sugar_Content
    sugar_before = df["Product_Sugar_Content"].value_counts().to_dict()
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})
    sugar_after = df["Product_Sugar_Content"].value_counts().to_dict()
    print(f"[CLEAN] Product_Sugar_Content before: {sugar_before}")
    print(f"[CLEAN] Product_Sugar_Content after : {sugar_after}")

    # 2. Drop duplicate rows if any
    before_dups = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[CLEAN] Removed {before_dups - len(df)} duplicate rows.")

    # 3. Feature engineering: extract Product_Category_Prefix from Product_Id before dropping it
    df["Product_Category_Prefix"] = df["Product_Id"].astype(str).str[:2]
    df["Store_Age"] = 2025 - df["Store_Establishment_Year"]

    # 4. Drop high-cardinality identifier column (Product_Id)
    df = df.drop(columns=["Product_Id"])
    print(f"[INFO] Dropped 'Product_Id'. Remaining columns ({len(df.columns)}): {list(df.columns)}")

    # 5. Reproducible 80/20 train-test split
    rng = np.random.default_rng(seed=random_state)
    indices = rng.permutation(len(df))
    n_test = int(round(len(df) * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[SAVED] Train split: {train_df.shape} -> {train_path}")
    print(f"[SAVED] Test split : {test_df.shape} -> {test_path}")
    return train_df, test_df


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/SuperKart.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    clean_and_prepare_data(raw, out)
