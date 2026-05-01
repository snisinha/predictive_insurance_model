"""preprocessing.py - encoding, scaling, balancing, and train/val/test split"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
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


def _balance(df: pd.DataFrame) -> pd.DataFrame:
    """Undersample the majority class to match the minority class size."""
    pos = df[df["is_claim"] == 1]
    neg = df[df["is_claim"] == 0]
    # Always sample the majority DOWN to minority size — no replacement needed
    majority, minority = (neg, pos) if len(neg) >= len(pos) else (pos, neg)
    majority_sample = majority.sample(n=len(minority), random_state=config.RANDOM_STATE)
    balanced = pd.concat([minority, majority_sample], axis=0)
    return balanced.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)


def split_data(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series,   pd.Series,   pd.Series,
]:
    """
    Split into balanced train / validation / test sets.
    Returns: X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    data = df.sample(n=len(df), random_state=config.RANDOM_STATE).reset_index(drop=True)

    valid_test = data.sample(frac=config.VALID_TEST_FRAC, random_state=config.RANDOM_STATE)
    train_all  = data.drop(valid_test.index)

    test  = valid_test.sample(frac=config.TEST_FRAC_OF_VALID_TEST, random_state=config.RANDOM_STATE)
    valid = valid_test.drop(test.index)

    train = _balance(train_all)
    valid = _balance(valid)
    test  = _balance(test)

    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        dist = split.groupby("is_claim").size().to_dict()
        print(f"{name} distribution: {dist}")

    X_train = train.drop("is_claim", axis=1)
    X_valid = valid.drop("is_claim", axis=1)
    X_test  = test.drop("is_claim",  axis=1)
    y_train = train["is_claim"].values
    y_valid = valid["is_claim"].values
    y_test  = test["is_claim"].values

    print(f"\ntrain shape : {X_train.shape}")
    print(f"valid shape : {X_valid.shape}")
    print(f"test  shape : {X_test.shape}")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def split_data_nn(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
]:
    """Simple stratified 80/20 split used by the neural network."""
    X = df.drop("is_claim", axis=1)
    y = df["is_claim"]
    return train_test_split(
        X, y,
        test_size=config.NN_TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )