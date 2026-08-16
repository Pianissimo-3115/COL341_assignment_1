#!/usr/bin/env python3
"""Part (a): Closed-form ordinary least-squares linear regression.

Usage:
    python3 part_a.py train.csv test.csv predictions.txt weights.txt
"""
import sys

import numpy as np
import pandas as pd


TARGET = "hr"


def load_train(path):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"{path} is missing the '{TARGET}' target column")
    y = df[TARGET].to_numpy(dtype=np.float64)
    X = df.drop(columns=[TARGET]).to_numpy(dtype=np.float64)
    return X, y


def load_test(path):
    df = pd.read_csv(path)
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])
    return df.to_numpy(dtype=np.float64)


def augment(X):
    n = X.shape[0]
    return np.hstack([np.ones((n, 1), dtype=np.float64), X])


def main():
    if len(sys.argv) != 5:
        print("Usage: python3 part_a.py train.csv test.csv predictions.txt weights.txt")
        sys.exit(1)

    train_path, test_path, predictions_path, weights_path = sys.argv[1:5]

    X_train, y_train = load_train(train_path)
    X_test = load_test(test_path)

    X_aug = augment(X_train)
    W = np.linalg.inv(X_aug.T @ X_aug) @ (X_aug.T @ y_train)

    X_test_aug = augment(X_test)
    preds = X_test_aug @ W

    np.savetxt(predictions_path, preds, fmt="%.10f")
    np.savetxt(weights_path, W, fmt="%.10f")


if __name__ == "__main__":
    main()
