#!/usr/bin/env python3
"""Part (c) variant v7 (own design, no autocorrelation): a fresh feature
family built from pulse morphology, exponential recency-weighting, and
motion/quality indicators -- deliberately avoiding every autocorrelation
and sliding cross-correlation feature used in earlier variants.

Usage:
    python3 part_3_v7.py train.csv test.csv predictions.txt

Same library rules as part_c.py (stdlib/NumPy-no-fft/pandas for feature
creation; sklearn only for scaling, CV/hyperparameter selection, and
fitting a permitted linear model).

Feature families and their motivation
--------------------------------------
- Pulse morphology (bvp): local-maxima/minima detection (no correlation)
  gives, per pulse, an amplitude (peak minus adjacent trough), rise time
  (trough -> peak) and fall time (peak -> trough). Literature on PPG
  waveform analysis treats pulse amplitude, upstroke time and pulse width
  as core morphological descriptors distinct from raw statistical summaries
  (see e.g. Charlton et al., "Photoplethysmography Signal Processing and
  Synthesis"; Perpetuini et al., MDPI Physiologia 2024 on wrist-PPG
  morphological classes). The rise/fall time ratio captures upstroke-vs-
  downstroke asymmetry, and inter-peak-interval mean/std/CV gives a direct
  beats-per-window rate plus an HRV proxy -- all without ever cross-
  correlating the signal against a shifted copy of itself.
- Exponentially weighted recency features (bvp, acc magnitude, eda): an
  EWMA (and matching exponentially weighted variance) computed with a
  short and a long half-life gives a smoothed "current level" and "current
  spread" that weight the samples closest to the prediction time t (the
  end of the 10s window, block s=9) more heavily than the oldest samples
  (block s=0) -- exactly the recency structure of this prediction task.
  EWMA is a standard low-pass/recency filter for physiological time series
  (see "Exponentially Weighted Moving Models", arXiv:2404.08136, and its
  use for PPG/accelerometer aggregation in wearable vital-sign pipelines).
  The "momentum" feature (EWMA level minus the whole-window mean) is a
  simple, explicit trend signal: it is positive when the signal is running
  higher near the end of the window than on average.
- Exponential block-energy trend (bvp, acc magnitude): splitting the
  window back into its ten 1-second blocks and fitting log(block energy)
  linearly against the block index is equivalent to fitting an exponential
  a*exp(b*t) to the (always positive) block energies -- the fitted rate b
  is a single number summarizing whether motion/pulse energy is ramping up
  or settling down across the window.
- Recency-contrast (bvp, acc magnitude): the plain difference between the
  last block's and first block's mean/std -- a cheap, literal "how did the
  window end up compared to how it started" signal.
- Signal-quality indices (bvp, acc magnitude): skewness and kurtosis are
  widely used PPG signal-quality/motion indicators (a clean pulse has a
  mildly-skewed, near-Gaussian shape; motion corruption flattens/skews it).
- Zero-crossing rate (bvp): average spacing between sign changes of the
  centered waveform gives an FFT-free, correlation-free frequency estimate
  that is independent of the peak-morphology heart-rate estimate above.
- Motion features (accelerometer): signal magnitude area (mean of
  |ax|+|ay|+|az|) and jerk magnitude (norm of the per-axis first
  difference) are standard time-domain activity-recognition features; axis
  interaction means (acc_x*acc_y etc.) are explicitly suggested in the
  assignment statement.
- Pulse-to-motion energy ratio: a *ratio*, not a correlation, between BVP
  variance and accelerometer-magnitude variance -- a high accelerometer
  share flags a likely motion-corrupted pulse window.
- EDA tonic/phasic split: trailing moving average approximates the slow
  tonic skin-conductance level; the phasic residual's spread and the
  tonic slope are kept (same construction as part_3_v6.py, no correlation
  involved).
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

BVP_FS = 64.0
ACC_FS = 32.0
EDA_FS = 4.0

BVP_EWMA_HALF_LIVES_S = (0.5, 2.0)
ACC_EWMA_HALF_LIVES_S = (1.0,)
EDA_EWMA_HALF_LIVES_S = (3.0,)

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


def skew_kurtosis_features(arr, prefix):
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + 1e-8
    z = (arr - mean) / std
    skew = np.mean(z ** 3, axis=1)
    kurtosis = np.mean(z ** 4, axis=1) - 3.0
    feats = np.stack([skew, kurtosis], axis=1)
    names = [f"{prefix}_skew", f"{prefix}_kurtosis"]
    return feats, names


def zero_crossing_features(arr, fs, prefix):
    n, T = arr.shape
    centered = arr - arr.mean(axis=1, keepdims=True)
    signs = np.sign(centered)
    signs[signs == 0] = 1.0
    crossings = np.sum(signs[:, 1:] != signs[:, :-1], axis=1).astype(np.float64)
    duration_s = (T - 1) / fs
    zcr_bpm = 60.0 * (crossings / 2.0) / duration_s
    feats = np.stack([crossings, zcr_bpm], axis=1)
    names = [f"{prefix}_zcr_count", f"{prefix}_zcr_bpm"]
    return feats, names


def second_derivative_features(arr, prefix):
    d2 = np.diff(arr, n=2, axis=1)
    feats = np.stack([d2.mean(axis=1), d2.std(axis=1), np.sum(d2 ** 2, axis=1)], axis=1)
    names = [f"{prefix}_d2_mean", f"{prefix}_d2_std", f"{prefix}_d2_energy"]
    return feats, names


def peak_morphology_features(arr, fs, prefix):
    """Local-maxima/minima pulse detection (strictly-greater-than-both-
    neighbours), per row. For every peak we pair it with its nearest
    preceding trough (rise leg) and nearest following trough (fall leg) via
    a sorted-array searchsorted lookup, avoiding a per-peak Python loop --
    only the per-row loop (needed since peak/trough counts vary per row)
    remains.
    """
    n, T = arr.shape
    feats = np.zeros((n, 13), dtype=np.float64)
    for i in range(n):
        row = arr[i]
        is_peak = (row[1:-1] > row[:-2]) & (row[1:-1] > row[2:])
        is_trough = (row[1:-1] < row[:-2]) & (row[1:-1] < row[2:])
        peak_idx = np.flatnonzero(is_peak) + 1
        trough_idx = np.flatnonzero(is_trough) + 1

        feats[i, 0] = peak_idx.size
        feats[i, 1] = trough_idx.size

        if peak_idx.size and trough_idx.size:
            pos = np.searchsorted(trough_idx, peak_idx)
            has_prev = pos > 0
            has_next = pos < trough_idx.size
            prev_t = trough_idx[np.clip(pos - 1, 0, trough_idx.size - 1)]
            next_t = trough_idx[np.clip(pos, 0, trough_idx.size - 1)]

            rise = np.where(has_prev, (peak_idx - prev_t) / fs, np.nan)
            fall = np.where(has_next, (next_t - peak_idx) / fs, np.nan)
            amp_prev = row[peak_idx] - row[prev_t]
            amp_next = row[peak_idx] - row[next_t]
            amp = np.where(has_prev, amp_prev, np.where(has_next, amp_next, np.nan))

            def nanmean(x):
                x = x[~np.isnan(x)]
                return x.mean() if x.size else 0.0

            def nanstd(x):
                x = x[~np.isnan(x)]
                return x.std() if x.size else 0.0

            feats[i, 2] = nanmean(amp)
            feats[i, 3] = nanstd(amp)
            feats[i, 4] = nanmean(rise)
            feats[i, 5] = nanstd(rise)
            feats[i, 6] = nanmean(fall)
            feats[i, 7] = nanstd(fall)
            if feats[i, 6] > 0:
                feats[i, 8] = feats[i, 4] / (feats[i, 6] + 1e-8)

        if peak_idx.size >= 2:
            iv = np.diff(peak_idx) / fs
            iv_mean = iv.mean()
            iv_std = iv.std()
            feats[i, 9] = iv_mean
            feats[i, 10] = iv_std
            feats[i, 11] = iv_std / (iv_mean + 1e-8)
            feats[i, 12] = 60.0 / (iv_mean + 1e-8)

    names = [
        f"{prefix}_peak_count", f"{prefix}_trough_count",
        f"{prefix}_amp_mean", f"{prefix}_amp_std",
        f"{prefix}_rise_mean", f"{prefix}_rise_std",
        f"{prefix}_fall_mean", f"{prefix}_fall_std",
        f"{prefix}_rise_fall_ratio",
        f"{prefix}_iv_mean", f"{prefix}_iv_std", f"{prefix}_iv_cv", f"{prefix}_peak_bpm",
    ]
    return feats, names


def ewma_features(arr, half_lives_s, fs, prefix):
    """For each half-life: an exponential moving average (level) and
    exponential moving std (spread), both evaluated at the last sample
    (closest to the prediction time t), plus a momentum term (level minus
    the whole-window mean). Implemented as a plain recursive filter over
    the (small, <=640-sample) time axis, vectorized across rows.
    """
    n, T = arr.shape
    whole_mean = arr.mean(axis=1)
    feats_cols, names = [], []
    for hl in half_lives_s:
        hl_samples = max(hl * fs, 1e-6)
        alpha = 1.0 - 0.5 ** (1.0 / hl_samples)

        ewma = np.empty_like(arr)
        ewma[:, 0] = arr[:, 0]
        for t in range(1, T):
            ewma[:, t] = alpha * arr[:, t] + (1.0 - alpha) * ewma[:, t - 1]

        dev2 = (arr - ewma) ** 2
        ewvar = np.empty_like(arr)
        ewvar[:, 0] = dev2[:, 0]
        for t in range(1, T):
            ewvar[:, t] = alpha * dev2[:, t] + (1.0 - alpha) * ewvar[:, t - 1]

        level = ewma[:, -1]
        spread = np.sqrt(ewvar[:, -1] + 1e-12)
        momentum = level - whole_mean

        feats_cols.extend([level, spread, momentum])
        tag = str(hl).replace(".", "p")
        names.extend([f"{prefix}_ewma_hl{tag}_level", f"{prefix}_ewma_hl{tag}_std",
                      f"{prefix}_ewma_hl{tag}_momentum"])

    feats = np.stack(feats_cols, axis=1)
    return feats, names


def block_summary(arr, n_blocks, block_len):
    n = arr.shape[0]
    blocks = arr.reshape(n, n_blocks, block_len)
    return blocks.mean(axis=2), blocks.std(axis=2), np.sum(blocks ** 2, axis=2)


def _slope_over_blocks(values):
    n, n_blocks = values.shape
    t = np.arange(n_blocks, dtype=np.float64)
    t_mean = t.mean()
    denom = np.sum((t - t_mean) ** 2)
    centered = values - values.mean(axis=1, keepdims=True)
    return np.sum(centered * (t - t_mean)[None, :], axis=1) / denom


def block_trend_features(block_mean, block_std, block_energy, prefix):
    feats = np.stack(
        [_slope_over_blocks(block_mean), _slope_over_blocks(block_std),
         _slope_over_blocks(block_energy)],
        axis=1,
    )
    names = [f"{prefix}_block_mean_slope", f"{prefix}_block_std_slope", f"{prefix}_block_energy_slope"]
    return feats, names


def exp_trend_features(block_energy, prefix):
    """log(block energy) fit linearly against the block index == fitting
    an exponential a*exp(b*t) to the (always positive) block energies. The
    slope b is the exponential growth/decay rate; a is the log-scale level.
    """
    n, n_blocks = block_energy.shape
    log_e = np.log(block_energy + 1e-6)
    t = np.arange(n_blocks, dtype=np.float64)
    t_mean = t.mean()
    denom = np.sum((t - t_mean) ** 2)
    centered = log_e - log_e.mean(axis=1, keepdims=True)
    b = np.sum(centered * (t - t_mean)[None, :], axis=1) / denom
    a = log_e.mean(axis=1) - b * t_mean
    feats = np.stack([b, a], axis=1)
    names = [f"{prefix}_exp_trend_rate", f"{prefix}_exp_trend_logscale"]
    return feats, names


def recency_contrast_features(block_mean, block_std, prefix):
    """Literal last-block-minus-first-block delta: how the window ended up
    compared to how it started, in raw (not slope-fitted) units.
    """
    delta_mean = block_mean[:, -1] - block_mean[:, 0]
    delta_std = block_std[:, -1] - block_std[:, 0]
    feats = np.stack([delta_mean, delta_std], axis=1)
    names = [f"{prefix}_recency_mean_delta", f"{prefix}_recency_std_delta"]
    return feats, names


def acc_axis_interaction_features(acc_x, acc_y, acc_z, prefix="acc"):
    xy = np.mean(acc_x * acc_y, axis=1)
    xz = np.mean(acc_x * acc_z, axis=1)
    yz = np.mean(acc_y * acc_z, axis=1)
    feats = np.stack([xy, xz, yz], axis=1)
    names = [f"{prefix}_xy_interact", f"{prefix}_xz_interact", f"{prefix}_yz_interact"]
    return feats, names


def sma_feature(acc_x, acc_y, acc_z, prefix="acc"):
    sma = np.mean(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z), axis=1)
    return sma[:, None], [f"{prefix}_sma"]


def jerk_features(acc_x, acc_y, acc_z, fs, prefix="acc"):
    jx = np.diff(acc_x, axis=1) * fs
    jy = np.diff(acc_y, axis=1) * fs
    jz = np.diff(acc_z, axis=1) * fs
    jerk_mag = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2)
    feats = np.stack(
        [jerk_mag.mean(axis=1), jerk_mag.std(axis=1), np.sum(jerk_mag ** 2, axis=1)], axis=1
    )
    names = [f"{prefix}_jerk_mean", f"{prefix}_jerk_std", f"{prefix}_jerk_energy"]
    return feats, names


def pulse_motion_ratio_features(bvp, acc_mag, prefix="motion"):
    bvp_energy = np.sum((bvp - bvp.mean(axis=1, keepdims=True)) ** 2, axis=1)
    acc_energy = np.sum((acc_mag - acc_mag.mean(axis=1, keepdims=True)) ** 2, axis=1)
    ratio = bvp_energy / (acc_energy + 1e-6)
    log_ratio = np.log1p(ratio)
    feats = np.stack([ratio, log_ratio], axis=1)
    names = [f"{prefix}_pulse_to_motion_ratio", f"{prefix}_pulse_to_motion_log_ratio"]
    return feats, names


def eda_tonic_phasic_features(eda, window, prefix="eda"):
    n, T = eda.shape
    csum = np.cumsum(eda, axis=1)
    csum = np.hstack([np.zeros((n, 1)), csum])
    tonic = np.empty_like(eda)
    for i in range(T):
        lo = max(0, i - window + 1)
        tonic[:, i] = (csum[:, i + 1] - csum[:, lo]) / (i + 1 - lo)
    phasic = eda - tonic

    t = np.arange(T, dtype=np.float64)
    t_mean = t.mean()
    denom = np.sum((t - t_mean) ** 2)
    tonic_c = tonic - tonic.mean(axis=1, keepdims=True)
    tonic_slope = np.sum(tonic_c * (t - t_mean)[None, :], axis=1) / denom

    phasic_std = phasic.std(axis=1)
    phasic_range = phasic.max(axis=1) - phasic.min(axis=1)

    feats = np.stack([tonic_slope, phasic_std, phasic_range], axis=1)
    names = [f"{prefix}_tonic_slope", f"{prefix}_phasic_std", f"{prefix}_phasic_range"]
    return feats, names


def build_features(X):
    signals = split_signals(X)
    blocks, names = [], []

    for name, arr in signals.items():
        f, n = signal_stats(arr, name)
        blocks.append(f); names.extend(n)

    bvp = signals["bvp"]
    acc_mag = signals["acc_mag"]
    acc_x, acc_y, acc_z = signals["acc_x"], signals["acc_y"], signals["acc_z"]
    eda = signals["eda"]

    # --- BVP ---
    f, n = skew_kurtosis_features(bvp, "bvp"); blocks.append(f); names.extend(n)
    f, n = zero_crossing_features(bvp, BVP_FS, "bvp"); blocks.append(f); names.extend(n)
    f, n = second_derivative_features(bvp, "bvp"); blocks.append(f); names.extend(n)
    f, n = peak_morphology_features(bvp, BVP_FS, "bvp"); blocks.append(f); names.extend(n)
    f, n = ewma_features(bvp, BVP_EWMA_HALF_LIVES_S, BVP_FS, "bvp"); blocks.append(f); names.extend(n)

    bvp_bmean, bvp_bstd, bvp_benergy = block_summary(bvp, N_BLOCKS, BVP_LEN)
    f, n = exp_trend_features(bvp_benergy, "bvp"); blocks.append(f); names.extend(n)
    f, n = recency_contrast_features(bvp_bmean, bvp_bstd, "bvp"); blocks.append(f); names.extend(n)

    # --- Accelerometer magnitude ---
    f, n = skew_kurtosis_features(acc_mag, "acc_mag"); blocks.append(f); names.extend(n)
    acc_bmean, acc_bstd, acc_benergy = block_summary(acc_mag, N_BLOCKS, ACC_LEN)
    f, n = block_trend_features(acc_bmean, acc_bstd, acc_benergy, "acc_mag"); blocks.append(f); names.extend(n)
    f, n = ewma_features(acc_mag, ACC_EWMA_HALF_LIVES_S, ACC_FS, "acc_mag"); blocks.append(f); names.extend(n)
    f, n = exp_trend_features(acc_benergy, "acc_mag"); blocks.append(f); names.extend(n)
    f, n = recency_contrast_features(acc_bmean, acc_bstd, "acc_mag"); blocks.append(f); names.extend(n)

    # --- Accelerometer axes jointly ---
    f, n = acc_axis_interaction_features(acc_x, acc_y, acc_z); blocks.append(f); names.extend(n)
    f, n = sma_feature(acc_x, acc_y, acc_z); blocks.append(f); names.extend(n)
    f, n = jerk_features(acc_x, acc_y, acc_z, ACC_FS); blocks.append(f); names.extend(n)

    # --- Cross-modal ---
    f, n = pulse_motion_ratio_features(bvp, acc_mag); blocks.append(f); names.extend(n)

    # --- EDA ---
    f, n = eda_tonic_phasic_features(eda, window=4); blocks.append(f); names.extend(n)
    f, n = ewma_features(eda, EDA_EWMA_HALF_LIVES_S, EDA_FS, "eda"); blocks.append(f); names.extend(n)

    return np.hstack(blocks), names


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 part_3_v7.py train.csv test.csv predictions.txt")
        sys.exit(1)

    train_path, test_path, predictions_path = sys.argv[1:4]

    X_train_raw, y_train = load_train(train_path)
    X_test_raw = load_test(test_path)

    Z_train, _ = build_features(X_train_raw)
    Z_test, _ = build_features(X_test_raw)

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_test_s = scaler.transform(Z_test)

    model = RidgeCV(alphas=np.logspace(-3, 4, 40), cv=5)
    model.fit(Z_train_s, y_train)

    preds = model.predict(Z_test_s)
    np.savetxt(predictions_path, preds, fmt="%.10f")


if __name__ == "__main__":
    main()
