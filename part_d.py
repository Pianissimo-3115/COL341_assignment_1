#!/usr/bin/env python3
"""Part (d): CGM glucose estimation from wearable physiological signals.

Usage:
    python3 part_d.py train <protocol> <train_dir> <model.pkl>
    python3 part_d.py feature_engineering <protocol> <test_dir> <model.pkl> <features.npy>

<protocol> is one of d1, d2, d3.

This is a starter feature set: per-channel statistical summaries for a
subset of the supplied modalities (BVP, HR, EDA, temperature, E4
accelerometer axes/magnitude, breathing). zephyr_ecg and zephyr_acc are
left out by default (large arrays, ~75000 and 30000x3 samples per
window) -- add extractors to CHANNELS if you want to use them.
Extend/replace features as needed; keep the final model linear
(w0 + w^T z) per the assignment's model restrictions.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

FEATURE_STATS = [
    "mean", "std", "min", "max", "median", "q25", "q75",
    "energy", "diff_mean", "diff_std", "slope",
]

CHANNELS = [
    ("e4_bvp", lambda d: d["e4_bvp"]),
    ("e4_hr", lambda d: d["e4_hr"]),
    ("e4_eda", lambda d: d["e4_eda"]),
    ("e4_temp", lambda d: d["e4_temp"]),
    ("e4_acc_x", lambda d: d["e4_acc"][:, :, 0]),
    ("e4_acc_y", lambda d: d["e4_acc"][:, :, 1]),
    ("e4_acc_z", lambda d: d["e4_acc"][:, :, 2]),
    ("e4_acc_mag", lambda d: np.sqrt(np.sum(np.square(d["e4_acc"]), axis=2))),
    ("zephyr_breathing", lambda d: d["zephyr_breathing"]),
]


def signal_stats(arr):
    """arr: (n, T), may contain NaN. Returns (n, len(FEATURE_STATS))."""
    n, T = arr.shape
    t = np.arange(T, dtype=np.float64)
    t_mean = t.mean()
    denom = np.sum((t - t_mean) ** 2)

    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=1)
        std = np.nanstd(arr, axis=1)
        mn = np.nanmin(arr, axis=1)
        mx = np.nanmax(arr, axis=1)
        median = np.nanmedian(arr, axis=1)
        q25 = np.nanquantile(arr, 0.25, axis=1)
        q75 = np.nanquantile(arr, 0.75, axis=1)
        energy = np.nansum(arr ** 2, axis=1)
        diff = np.diff(arr, axis=1)
        diff_mean = np.nanmean(diff, axis=1)
        diff_std = np.nanstd(diff, axis=1)
        centered = arr - mean[:, None]
        slope = np.nansum(centered * (t - t_mean)[None, :], axis=1) / denom

    return np.stack(
        [mean, std, mn, mx, median, q25, q75, energy, diff_mean, diff_std, slope],
        axis=1,
    )


def build_features_for_file(data):
    blocks, names = [], []
    for chan_name, extractor in CHANNELS:
        arr = np.asarray(extractor(data), dtype=np.float64)
        blocks.append(signal_stats(arr))
        names.extend(f"{chan_name}_{stat}" for stat in FEATURE_STATS)
    return np.hstack(blocks), names


def build_dataset(directory, need_glucose):
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {directory}")

    feature_chunks = []
    target_chunks = [] if need_glucose else None
    feature_names = None

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            feats, names = build_features_for_file(data)
            if feature_names is None:
                feature_names = names
            feature_chunks.append(feats)
            if need_glucose:
                target_chunks.append(np.asarray(data["glucose"], dtype=np.float64))

    Z = np.vstack(feature_chunks)
    y = np.concatenate(target_chunks) if need_glucose else None
    return Z, y, feature_names


def impute(Z, col_means):
    Z = Z.copy()
    bad = np.where(~np.isfinite(Z))
    Z[bad] = np.take(col_means, bad[1])
    return Z


def train(protocol, train_dir, model_path):
    Z_raw, y, feature_names = build_dataset(train_dir, need_glucose=True)

    col_means = np.nanmean(Z_raw, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    Z_imputed = impute(Z_raw, col_means)

    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z_imputed)

    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model.fit(Z_scaled, y)

    # Fold the scaler into a single (intercept, coef) pair on RAW feature
    # space, so predictions are simply w0 + Z_raw @ w -- matching how the
    # evaluator scores features_<protocol>.npy against the saved model.
    coef_scaled = model.coef_
    coef_raw = coef_scaled / scaler.scale_
    intercept_raw = model.intercept_ - np.sum(coef_scaled * scaler.mean_ / scaler.scale_)

    state = {
        "format_version": 1,
        "protocol": protocol,
        "intercept": float(intercept_raw),
        "coef": coef_raw.astype(np.float64),
        "feature_names": feature_names,
        "preprocessing_state": {
            "col_means": col_means,
            "alpha": float(model.alpha_),
        },
    }

    with open(model_path, "wb") as f:
        pickle.dump(state, f)


def feature_engineering(test_dir, model_path, output_path):
    with open(model_path, "rb") as f:
        state = pickle.load(f)

    Z_raw, _, feature_names = build_dataset(test_dir, need_glucose=False)
    assert feature_names == state["feature_names"], "Feature layout mismatch with saved model."

    col_means = state["preprocessing_state"]["col_means"]
    Z_test = impute(Z_raw, col_means)

    np.save(output_path, Z_test)


def main():
    if len(sys.argv) < 5:
        print("Usage:")
        print("  python3 part_d.py train <protocol> <train_dir> <model.pkl>")
        print("  python3 part_d.py feature_engineering <protocol> <test_dir> <model.pkl> <features.npy>")
        sys.exit(1)

    mode = sys.argv[1]
    protocol = sys.argv[2]

    if mode == "train":
        if len(sys.argv) != 5:
            print("Usage: python3 part_d.py train <protocol> <train_dir> <model.pkl>")
            sys.exit(1)
        train_dir, model_path = sys.argv[3:5]
        train(protocol, train_dir, model_path)
    elif mode == "feature_engineering":
        if len(sys.argv) != 6:
            print("Usage: python3 part_d.py feature_engineering <protocol> <test_dir> <model.pkl> <features.npy>")
            sys.exit(1)
        test_dir, model_path, output_path = sys.argv[3:6]
        feature_engineering(test_dir, model_path, output_path)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
