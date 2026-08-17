#!/usr/bin/env python3
"""Part (c) variant v3: part_c_better.py + a thresholded gating activation
applied to every correlation-valued feature, with the threshold lambda
chosen by cross-validation.

Usage:
    python3 part_3_v3.py train.csv test.csv predictions.txt

Same library rules as part_c.py (stdlib/NumPy-no-fft/pandas for feature
creation; sklearn only for scaling, CV/hyperparameter selection, and
fitting a permitted linear model).

Activation function
--------------------
The banked autocorrelation peak (bvp_autocorr_peak) and every sliding
cross-correlation offset (bvp_xcorr_off*) are all normalized correlation
coefficients in [-1, 1]. The idea: a correlation whose *magnitude* clears
some threshold lambda represents a genuine periodic match (a real pulse
cycle) and should be pushed further from zero (amplified); a correlation
below lambda is more likely to be noise and should be pushed toward zero
(diminished). This is implemented as a smooth multiplicative gate

    gate(c; lambda) = 1 + AMP * tanh(STEEPNESS * (|c| - lambda))
    c' = c * gate(c; lambda)

so gate -> 1 - AMP for |c| << lambda (strong shrink), gate -> 1 for
|c| == lambda (unchanged), and gate -> 1 + AMP for |c| >> lambda (strong
amplification). AMP and STEEPNESS are fixed constants (0.6 and 12); lambda
is the single hyperparameter, swept via GridSearchCV with an inner RidgeCV
so both lambda and the ridge penalty are picked from the training data
only. This transform is pure NumPy (implemented by us); scikit-learn's
Pipeline/GridSearchCV only supply the CV plumbing, per the library rules.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_BLOCKS = 10
ACC_LEN = 32
BVP_LEN = 64
EDA_LEN = 4
BLOCK_LEN = 3 * ACC_LEN + BVP_LEN + EDA_LEN  # 164, matches assignment column order

BVP_FS = 64.0
HR_MIN_BPM = 40.0
HR_MAX_BPM = 200.0

XCORR_REF_LEN = 64   # most recent 1-second BVP window, used as the fixed reference
XCORR_STEP = 8        # slide the reference in 8-sample (~0.125s) steps

GATE_AMP = 0.6         # max +/- multiplicative swing around 1.0
GATE_STEEPNESS = 12.0  # how sharply the gate transitions around lambda
LAMBDA_GRID = np.linspace(0.0, 0.9, 19)  # candidate thresholds, tuned via CV

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


def autocorr_features(arr, fs, hr_min, hr_max, prefix):
    """Banked-lag autocorrelation reduced to a single best-matching lag:
    implied heart rate plus how strong that match was. See part_c.py.
    """
    n, T = arr.shape
    lag_min = max(1, int(np.floor(60.0 * fs / hr_max)))
    lag_max = min(T - 1, int(np.ceil(60.0 * fs / hr_min)))

    centered = arr - arr.mean(axis=1, keepdims=True)
    zero_lag_energy = np.sum(centered ** 2, axis=1) + 1e-8

    best_corr = np.full(n, -np.inf, dtype=np.float64)
    best_lag = np.ones(n, dtype=np.float64)
    for lag in range(lag_min, lag_max + 1):
        corr = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / zero_lag_energy
        improved = corr > best_corr
        best_corr = np.where(improved, corr, best_corr)
        best_lag = np.where(improved, lag, best_lag)

    implied_bpm = 60.0 * fs / best_lag

    feats = np.stack([implied_bpm, best_corr], axis=1)
    names = [f"{prefix}_autocorr_bpm", f"{prefix}_autocorr_peak"]
    return feats, names


def sliding_xcorr_features(arr, ref_len, step, prefix):
    """Fixed most-recent `ref_len`-sample window correlated against
    same-length windows slid across the rest of the signal every `step`
    samples. Unlike autocorr_features, every offset's (normalized)
    correlation is kept as its own feature instead of reducing to a
    single best match.
    """
    n, T = arr.shape
    ref = arr[:, T - ref_len:]
    ref_c = ref - ref.mean(axis=1, keepdims=True)
    ref_energy = np.sum(ref_c ** 2, axis=1)

    offsets = list(range(0, T - ref_len, step))  # excludes the trivial self-match offset
    feats = np.empty((n, len(offsets)), dtype=np.float64)
    names = []
    for i, off in enumerate(offsets):
        win = arr[:, off:off + ref_len]
        win_c = win - win.mean(axis=1, keepdims=True)
        win_energy = np.sum(win_c ** 2, axis=1)
        cross = np.sum(ref_c * win_c, axis=1)
        feats[:, i] = cross / np.sqrt(ref_energy * win_energy + 1e-8)
        names.append(f"{prefix}_xcorr_off{off}")
    return feats, names


def build_features(X):
    """Returns (features, names, corr_idx) where corr_idx are the column
    indices of every correlation-coefficient-valued feature (bounded in
    [-1, 1]) -- these are the columns the gating activation applies to.
    """
    signals = split_signals(X)
    blocks, names = [], []
    for name, arr in signals.items():
        f, n = signal_stats(arr, name)
        blocks.append(f)
        names.extend(n)

    corr_idx = []

    f, n = autocorr_features(signals["bvp"], BVP_FS, HR_MIN_BPM, HR_MAX_BPM, "bvp")
    offset = sum(b.shape[1] for b in blocks)
    corr_idx.append(offset + n.index("bvp_autocorr_peak"))
    blocks.append(f)
    names.extend(n)

    f, n = sliding_xcorr_features(signals["bvp"], XCORR_REF_LEN, XCORR_STEP, "bvp")
    offset = sum(b.shape[1] for b in blocks)
    corr_idx.extend(offset + i for i in range(len(n)))
    blocks.append(f)
    names.extend(n)

    return np.hstack(blocks), names, np.array(corr_idx, dtype=np.int64)


class CorrGateTransform(BaseEstimator, TransformerMixin):
    """Applies the amplify-above/diminish-below-lambda gate (see module
    docstring) to a fixed set of correlation-valued columns; every other
    column passes through unchanged.
    """

    def __init__(self, corr_idx=None, lam=0.3, amp=GATE_AMP, steepness=GATE_STEEPNESS):
        self.corr_idx = corr_idx
        self.lam = lam
        self.amp = amp
        self.steepness = steepness

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        idx = self.corr_idx
        c = X[:, idx]
        gate = 1.0 + self.amp * np.tanh(self.steepness * (np.abs(c) - self.lam))
        X[:, idx] = c * gate
        return X


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 part_3_v3.py train.csv test.csv predictions.txt")
        sys.exit(1)

    train_path, test_path, predictions_path = sys.argv[1:4]

    X_train_raw, y_train = load_train(train_path)
    X_test_raw = load_test(test_path)

    Z_train, _, corr_idx = build_features(X_train_raw)
    Z_test, _, _ = build_features(X_test_raw)

    pipeline = Pipeline([
        ("corr_gate", CorrGateTransform(corr_idx=corr_idx)),
        ("scaler", StandardScaler()),
        # cv=None -> efficient built-in leave-one-out (generalized CV) ridge
        # path instead of explicitly refitting per alpha per fold; keeps the
        # outer lambda search (which already does its own 5-fold CV) cheap.
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13), cv=None)),
    ])

    search = GridSearchCV(
        pipeline,
        param_grid={"corr_gate__lam": LAMBDA_GRID},
        cv=5,
        scoring="neg_mean_squared_error",
        refit=True,
        n_jobs=-1,
    )
    search.fit(Z_train, y_train)

    preds = search.predict(Z_test)
    np.savetxt(predictions_path, preds, fmt="%.10f")


if __name__ == "__main__":
    main()
