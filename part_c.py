#!/usr/bin/env python3
"""Part (c): Feature-engineered linear model for heart-rate prediction.

Usage:
    python3 part_c.py train.csv test.csv predictions.txt

Library rules (per assignment statement):
  - Feature creation: Python stdlib, NumPy (no numpy.fft), pandas only.
  - scikit-learn: only for scaling/preprocessing, CV/hyperparameter
    selection, feature selection, and fitting the final permitted
    linear model (OLS / ridge / lasso / elastic net).

This is a starter feature set (per-modality statistical summaries).
Extend build_features() with more informative features (autocorrelation,
polynomial terms, cross-axis interactions, etc.) to improve ranking.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

N_BLOCKS = 10
ACC_LEN = 32
BVP_LEN = 64
EDA_LEN = 4
BLOCK_LEN = 3 * ACC_LEN + BVP_LEN + EDA_LEN  # 164, matches assignment column order


def load_raw(path, has_target):
    df = pd.read_csv(path)
    if has_target:
        y = df.iloc[:, -1].to_numpy(dtype=np.float64)
        X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    else:
        y = None
        X = df.to_numpy(dtype=np.float64)
    return X, y


def split_signals(X):
    """X: (n, 1640) -> per-modality time series concatenated across the 10 blocks."""
    n = X.shape[0]
    blocks = X.reshape(n, N_BLOCKS, BLOCK_LEN)

    acc_x = blocks[:, :, 0:ACC_LEN].reshape(n, N_BLOCKS * ACC_LEN)
    acc_y = blocks[:, :, ACC_LEN:2 * ACC_LEN].reshape(n, N_BLOCKS * ACC_LEN)
    acc_z = blocks[:, :, 2 * ACC_LEN:3 * ACC_LEN].reshape(n, N_BLOCKS * ACC_LEN)
    bvp = blocks[:, :, 3 * ACC_LEN:3 * ACC_LEN + BVP_LEN].reshape(n, N_BLOCKS * BVP_LEN)
    eda = blocks[:, :, 3 * ACC_LEN + BVP_LEN:].reshape(n, N_BLOCKS * EDA_LEN)

    acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    return {
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "acc_mag": acc_mag,
        "bvp": bvp,
        "eda": eda,
    }


def signal_stats(arr, prefix):
    n, T = arr.shape
    t = np.arange(T, dtype=np.float64)
    t_mean = t.mean()
    denom = np.sum((t - t_mean) ** 2)

    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    mn = arr.min(axis=1)
    mx = arr.max(axis=1)
    median = np.median(arr, axis=1)
    q25 = np.quantile(arr, 0.25, axis=1)
    q75 = np.quantile(arr, 0.75, axis=1)
    energy = np.sum(arr ** 2, axis=1)
    diff = np.diff(arr, axis=1)
    diff_mean = diff.mean(axis=1)
    diff_std = diff.std(axis=1)
    centered = arr - mean[:, None]
    slope = np.sum(centered * (t - t_mean)[None, :], axis=1) / denom

    feats = np.stack(
        [mean, std, mn, mx, median, q25, q75, energy, diff_mean, diff_std, slope],
        axis=1,
    )
    names = [
        f"{prefix}_{s}"
        for s in ["mean", "std", "min", "max", "median", "q25", "q75",
                  "energy", "diff_mean", "diff_std", "slope"]
    ]
    return feats, names


def build_features(X):
    signals = split_signals(X)
    blocks, names = [], []
    for name, arr in signals.items():
        f, n = signal_stats(arr, name)
        blocks.append(f)
        names.extend(n)
    return np.hstack(blocks), names


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 part_c.py train.csv test.csv predictions.txt")
        sys.exit(1)

    train_path, test_path, predictions_path = sys.argv[1:4]

    X_train_raw, y_train = load_raw(train_path, has_target=True)
    X_test_raw, _ = load_raw(test_path, has_target=False)

    Z_train, _ = build_features(X_train_raw)
    Z_test, _ = build_features(X_test_raw)

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_test_s = scaler.transform(Z_test)

    model = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
    model.fit(Z_train_s, y_train)

    preds = model.predict(Z_test_s)
    np.savetxt(predictions_path, preds, fmt="%.10f")


if __name__ == "__main__":
    main()
