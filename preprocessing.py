# preprocessing.py — encoding, scaling, balancing, and train/val/test split

from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import config


def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categoricals, binary-encode Yes/No columns,
    scale numeric columns, and drop unused columns.
    Returns a fully numeric copy of the dataframe.
    """
    df = df.copy()

    if "policy_id" in df.columns:
        df.drop("policy_id", axis=1, inplace=True)

    df.replace({"Yes": 1, "No": 0}, inplace=True)

    df = pd.get_dummies(df, columns=config.OHE_COLUMNS)

    # Drop ALL remaining object columns (e.g. transmission_type, steering_type)
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df.drop(obj_cols, axis=1, inplace=True)

    scaler = MinMaxScaler()
    cols_present = [c for c in config.COLS_TO_SCALE if c in df.columns]
    df[cols_present] = scaler.fit_transform(df[cols_present])

    return df


def split_data(df: pd.DataFrame) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Stratified holdout → SMOTE on train
    Returns: X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    # Features and target
    X = df.drop("is_claim", axis=1)
    y = df["is_claim"]

    # Train-Test Split (stratify to maintain class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    X_valid = np.copy(X_test)
    X_test_out = np.copy(X_test)
    y_valid = np.copy(y_test)
    y_test_out = np.copy(y_test)

    dist_train = dict(zip(*np.unique(y_train, return_counts=True)))
    dist_hold = dict(zip(*np.unique(y_test, return_counts=True)))
    print(f"train distribution: {dist_train}")
    print(f"valid/test distribution (same split): {dist_hold}")
    print(f"\ntrain shape : {X_train.shape}")
    print(f"valid shape : {X_valid.shape}")
    print(f"test  shape : {X_test_out.shape}")

    return X_train, X_valid, X_test_out, y_train, y_valid, y_test_out
