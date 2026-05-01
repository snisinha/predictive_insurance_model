"""data_loader.py — load and perform basic cleaning on the raw dataset"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and drop rows with missing values."""
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", list(df.columns))

    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with null values.")

    print(df.info())
    return df