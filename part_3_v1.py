#!/usr/bin/env python3
"""Part (c) variant v1: part_c_better.py + quadratic (squared) versions of
every autocorrelation/cross-correlation feature.

Usage:
    python3 part_3_v1.py train.csv test.csv predictions.txt

Same library rules as part_c.py (stdlib/NumPy-no-fft/pandas for feature
creation; sklearn only for scaling, CV/hyperparameter selection, and
fitting a permitted linear model).

Difference from part_c_better.py: both the single-best-lag autocorrelation
features (bvp_autocorr_bpm, bvp_autocorr_peak) and the sliding cross-
correlation bank (bvp_xcorr_off*) get a squared companion feature. Squaring
a correlation coefficient turns it into a sign-free match-strength measure
(a strong match at lag L looks the same whether the signal is shifted "up"
or "down" in phase), which a plain linear model on the raw signed value
cannot represent. The bpm-valued feature is squared too, letting the linear
model fit curvature in the bpm -> heart-rate relationship. Model stays
ElasticNetCV, unchanged from part_c_better.py, so the only variable being
tested here is the extra quadratic terms.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
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


def quadratic_expand(feats, names, suffix="_sq"):
    """Elementwise squared companion of every column in `feats`."""
    return feats ** 2, [f"{nm}{suffix}" for nm in names]


def build_features(X):
    signals = split_signals(X)
    blocks, names = [], []
    for name, arr in signals.items():
        f, n = signal_stats(arr, name)
        blocks.append(f)
        names.extend(n)

    ac_feats, ac_names = autocorr_features(
        signals["bvp"], BVP_FS, HR_MIN_BPM, HR_MAX_BPM, "bvp"
    )
    blocks.append(ac_feats)
    names.extend(ac_names)

    xc_feats, xc_names = sliding_xcorr_features(
        signals["bvp"], XCORR_REF_LEN, XCORR_STEP, "bvp"
    )
    blocks.append(xc_feats)
    names.extend(xc_names)

    # Quadratic companions for every autocorrelation / cross-correlation feature.
    ac_sq_feats, ac_sq_names = quadratic_expand(ac_feats, ac_names)
    blocks.append(ac_sq_feats)
    names.extend(ac_sq_names)

    xc_sq_feats, xc_sq_names = quadratic_expand(xc_feats, xc_names)
    blocks.append(xc_sq_feats)
    names.extend(xc_sq_names)

    return np.hstack(blocks), names


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 part_3_v1.py train.csv test.csv predictions.txt")
        sys.exit(1)

    train_path, test_path, predictions_path = sys.argv[1:4]

    X_train_raw, y_train = load_train(train_path)
    X_test_raw = load_test(test_path)

    Z_train, _ = build_features(X_train_raw)
    Z_test, _ = build_features(X_test_raw)

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_test_s = scaler.transform(Z_test)

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        alphas=np.logspace(-2, 2, 50),
        cv=5,
        max_iter=10000,
        random_state=0,
    )
    model.fit(Z_train_s, y_train)

    preds = model.predict(Z_test_s)
    np.savetxt(predictions_path, preds, fmt="%.10f")


if __name__ == "__main__":
    main()
