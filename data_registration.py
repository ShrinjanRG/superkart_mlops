"""Data Registration and Validation Script for SuperKart MLOps Pipeline."""

import os
import sys
import pandas as pd

EXPECTED_COLUMNS = [
    "Product_Id",
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Store_Sales_Total",
]

NUMERIC_COLUMNS = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
    "Product_Store_Sales_Total",
]


def register_and_validate(data_path: str = "data/SuperKart.csv") -> pd.DataFrame:
    """Load SuperKart dataset, validate schema & quality, and print summary."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'")

    df = pd.read_csv(data_path)
    print(f"[INFO] Successfully loaded dataset from: {data_path}")
    print(f"[INFO] Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # 1. Column schema check
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    if extra_cols:
        print(f"[WARN] Unexpected extra columns detected: {extra_cols}")
    print("[PASS] Schema validation passed: all 12 expected columns present.")

    # 2. Numeric type validation
    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric, found {df[col].dtype}")
    print("[PASS] Numeric dtype validation passed.")

    # 3. Missing value and duplicate checks
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    dup_count = int(df.duplicated().sum())
    print(f"[INFO] Total missing values: {total_nulls}")
    print(f"[INFO] Total duplicate rows: {dup_count}")

    # 4. Target range sanity check
    if (df["Product_Store_Sales_Total"] <= 0).any():
        raise ValueError("Target 'Product_Store_Sales_Total' contains non-positive values.")
    print("[PASS] Target sanity check passed (all sales > 0).")

    # 5. Print dataset summary
    print("\n--- Dataset Summary ---")
    print(df.describe(include="all").T[["count", "unique", "top", "mean", "std", "min", "max"]].to_string())
    print("\n[SUCCESS] Data registration and validation complete.")
    return df


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/SuperKart.csv"
    register_and_validate(path)
