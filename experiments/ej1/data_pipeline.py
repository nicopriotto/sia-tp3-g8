"""Data loading, splitting and normalization for ej1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

CSV_PATH = "data/fraud_dataset.csv"
TARGET_COL = "big_model_fraud_probability"
FLAG_COL = "flagged_fraud"


@dataclass
class DataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    y_flag_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    y_flag_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_flag_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler


def prepare_data(
    csv_path: str = CSV_PATH,
    *,
    drop_timestamp: bool = False,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> DataBundle:
    df = pd.read_csv(csv_path)

    if drop_timestamp and "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    y = df[TARGET_COL].values.astype(float)
    y_flag = df[FLAG_COL].values.astype(float)
    X_df = df.drop(columns=[TARGET_COL, FLAG_COL])
    feature_names = list(X_df.columns)
    X = X_df.values.astype(float)

    # First split off test set
    X_trainval, X_test, y_trainval, y_test, yf_trainval, yf_test = train_test_split(
        X, y, y_flag, test_size=test_ratio, random_state=seed
    )
    # Then split val from the remaining
    adjusted_val = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val, yf_train, yf_val = train_test_split(
        X_trainval, y_trainval, yf_trainval,
        test_size=adjusted_val, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return DataBundle(
        X_train=X_train, y_train=y_train, y_flag_train=yf_train,
        X_val=X_val, y_val=y_val, y_flag_val=yf_val,
        X_test=X_test, y_test=y_test, y_flag_test=yf_test,
        feature_names=feature_names,
        scaler=scaler,
    )


def make_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    y_flag: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list:
    """Return list of (X_tr, y_tr, yf_tr, X_vl, y_vl, yf_vl) with per-fold scaling."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]
        yf_tr, yf_vl = y_flag[train_idx], y_flag[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_vl = scaler.transform(X_vl)

        splits.append((X_tr, y_tr, yf_tr, X_vl, y_vl, yf_vl))
    return splits
