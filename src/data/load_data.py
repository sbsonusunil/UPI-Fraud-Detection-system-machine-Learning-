# src/data/load_data.py

import pandas as pd
import os


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw UPI fraud dataset from CSV file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at path: {path}")

    df = pd.read_csv(path)
    return df
