#!/usr/bin/env python3
"""Part (b): Ridge regression with five-fold cross-validation to select lambda.

Usage:
    python3 part_b.py train.csv test.csv folds.txt regularization.txt \
        predictions.txt weights.txt bestlambda.txt crossvalidation_errors.txt
"""
import sys

import numpy as np
import pandas as pd


def load_features(path, has_target):
    df = pd.read_csv(path)
    if has_target:
        y = df.iloc[:, -1].to_numpy(dtype=np.float64)
        X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    else:
        y = None
        X = df.to_numpy(dtype=np.float64)
    return X, y


def augment(X):
    n = X.shape[0]
    return np.hstack([np.ones((n, 1), dtype=np.float64), X])


def read_folds(path):
    """folds.txt contains the ending index for each fold, one per line.

    NOTE: this assumes folds are contiguous, non-overlapping blocks of
    0-indexed row positions (inclusive end index) that partition the
    training rows in the order given. Cross-check this against the
    reference folds.txt once it is available, and adjust if the actual
    format differs (e.g. exclusive end, or explicit index lists).
    """
    with open(path) as f:
        ends = [int(line.strip()) for line in f if line.strip()]

    folds = []
    start = 0
    for end in ends:
        folds.append(np.arange(start, end + 1))
        start = end + 1
    return folds


def read_lambdas(path):
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]


def ridge_weights(X_aug, y, lam):
    m1 = X_aug.shape[1]
    R = np.eye(m1)
    R[0, 0] = 0.0
    A = X_aug.T @ X_aug + lam * R
    return np.linalg.inv(A) @ (X_aug.T @ y)


def nmse(y_true, y_pred):
    y_bar = np.mean(y_true)
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - y_bar) ** 2)
    return num / den


def main():
    if len(sys.argv) != 9:
        print(
            "Usage: python3 part_b.py train.csv test.csv folds.txt regularization.txt "
            "predictions.txt weights.txt bestlambda.txt crossvalidation_errors.txt"
        )
        sys.exit(1)

    (
        train_path,
        test_path,
        folds_path,
        reg_path,
        predictions_path,
        weights_path,
        bestlambda_path,
        cverrors_path,
    ) = sys.argv[1:9]

    X_train, y_train = load_features(train_path, has_target=True)
    X_test, _ = load_features(test_path, has_target=False)

    folds = read_folds(folds_path)
    lambdas = read_lambdas(reg_path)

    X_aug_full = augment(X_train)

    cv_errors = []
    for lam in lambdas:
        fold_nmses = []
        for k in range(len(folds)):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])

            W = ridge_weights(X_aug_full[train_idx], y_train[train_idx], lam)
            y_pred = X_aug_full[val_idx] @ W
            fold_nmses.append(nmse(y_train[val_idx], y_pred))

        cv_errors.append(float(np.mean(fold_nmses)))

    best_idx = int(np.argmin(cv_errors))  # first occurrence on ties
    best_lambda = lambdas[best_idx]

    W_final = ridge_weights(X_aug_full, y_train, best_lambda)

    X_test_aug = augment(X_test)
    preds = X_test_aug @ W_final

    np.savetxt(predictions_path, preds, fmt="%.10f")
    np.savetxt(weights_path, W_final, fmt="%.10f")

    with open(bestlambda_path, "w") as f:
        f.write(f"{best_lambda}\n")

    with open(cverrors_path, "w") as f:
        for lam, err in zip(lambdas, cv_errors):
            f.write(f"{lam},{err:.6f}\n")


if __name__ == "__main__":
    main()