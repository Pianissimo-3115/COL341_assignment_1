#!/usr/bin/env python3
import re
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


def read_folds(path, n):
    """folds.txt gives the (exclusive) ending row index of each fold.

    Accepts either one integer per line, or the reference format
    `fold_ends = [32161, 64993, 97079, 127934, 158986]` (all integers in
    the file are extracted regardless of surrounding syntax). Fold k
    covers rows [ends[k-1], ends[k]); the folds must partition all n
    training rows in order, i.e. the last end must equal n.
    """
    with open(path) as f:
        text = f.read()
    ends = [int(tok) for tok in re.findall(r"-?\d+", text)]
    if not ends or ends[-1] != n:
        raise ValueError(
            f"{path}: fold ends {ends} do not partition {n} training rows"
        )
    return list(zip([0] + ends[:-1], ends))


def read_lambdas(path):
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]


def fold_normal_equations(X_aug, y, bounds):
    """Per-fold Gram matrix X_k^T X_k and X_k^T y_k, one pass over the data.

    Excluding fold k from training just means summing the OTHER folds'
    precomputed (G, b) pairs -- an O(m^2) addition -- instead of gathering
    and re-multiplying an (n_train x m) matrix from scratch for every
    lambda/fold combination.
    """
    m1 = X_aug.shape[1]
    G, b = [], []
    for start, end in bounds:
        X_k = X_aug[start:end]
        y_k = y[start:end]
        G.append(X_k.T @ X_k)
        b.append(X_k.T @ y_k)
    return G, b


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

    X_train, y_train = load_train(train_path)
    n, m1 = X_train.shape[0], X_train.shape[1] + 1
    X_aug = augment(X_train)
    del X_train

    bounds = read_folds(folds_path, n)
    lambdas = read_lambdas(reg_path)

    G, b = fold_normal_equations(X_aug, y_train, bounds)
    G_total = sum(G)
    b_total = sum(b)

    R = np.eye(m1, dtype=np.float64)
    R[0, 0] = 0.0

    cv_errors = []
    for lam in lambdas:
        fold_nmses = []
        for k, (start, end) in enumerate(bounds):
            A = (G_total - G[k]) + lam * R
            rhs = b_total - b[k]
            W = np.linalg.inv(A) @ rhs
            y_pred = X_aug[start:end] @ W
            fold_nmses.append(nmse(y_train[start:end], y_pred))
        cv_errors.append(float(np.mean(fold_nmses)))

    best_idx = int(np.argmin(cv_errors))  # first occurrence on ties
    best_lambda = lambdas[best_idx]

    W_final = np.linalg.inv(G_total + best_lambda * R) @ b_total

    del X_aug, y_train
    X_test = load_test(test_path)
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