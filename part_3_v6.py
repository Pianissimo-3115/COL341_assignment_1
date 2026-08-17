#!/usr/bin/env python3
"""Part (c) variant v6 (research pass): a broader, literature-backed feature
set on top of the same per-modality statistics used by part_c_better.py,
fit with RidgeCV instead of ElasticNetCV.

Usage:
    python3 part_3_v6.py train.csv test.csv predictions.txt

Same library rules as part_c.py (stdlib/NumPy-no-fft/pandas for feature
creation; sklearn only for scaling, CV/hyperparameter selection, and
fitting a permitted linear model).

New feature families (beyond signal_stats + best-lag autocorrelation +
sliding cross-correlation, all kept from part_c_better.py)
-----------------------------------------------------------------------
- Harmonic-sum autocorrelation (bvp): instead of only the single lag with
  the highest autocorrelation, also score each candidate lag L by
  corr(L) + corr(2L) + corr(3L). A true pulse period reinforces at its
  harmonics; a spurious lag usually does not. This is the time-domain
  analogue of harmonic-sum pitch detection used for PPG heart-rate
  estimation under motion (see "Harmonic Sum-based Method for Heart Rate
  Estimation using PPG Signals Affected with Motion Artifacts",
  arXiv:1610.05112). We also keep the plain best-lag peak, a second
  (guarded) local-maximum peak, and the ratio between them as a periodicity
  "confidence" signal.
- Zero-crossing rate (bvp): the average spacing between sign changes of the
  mean-centered signal gives an independent, FFT-free frequency estimate
  (two crossings per cycle), complementing the lag-search autocorrelation.
- Skewness / kurtosis (bvp, acc magnitude): standard time-domain PPG
  signal-quality indices; motion-corrupted or noisy windows skew away from
  the smooth, mildly-skewed shape of a clean pulse waveform.
- Second-derivative (APG-like) statistics (bvp): the second difference of
  the waveform highlights curvature/sharpness of the pulse upstroke,
  distinct information from the first-difference features already present.
- Per-1-second-block trend (bvp, acc magnitude): splitting the 10s window
  back into its 10 one-second blocks and taking the linear slope (across
  blocks) of each block's std/mean/energy captures within-window drift
  (e.g. accelerating heart rate or ramping motion) that whole-window
  statistics cannot see.
- Cross-modal motion/pulse correlation: correlating the per-block BVP std
  against the per-block accelerometer-magnitude std (10 points each) gives
  a direct motion-artifact indicator -- when movement energy tracks pulse-
  signal energy across the window, the BVP signal is likely motion-
  corrupted.
- Accelerometer axis interactions (acc_x*acc_y, acc_x*acc_z, acc_y*acc_z
  means): explicitly suggested in the assignment statement; captures
  coordinated (rather than single-axis) movement.
- Polynomial trend coefficients (bvp): degree-3 polynomial fit to the whole
  10s waveform (constant term dropped as redundant with the mean feature),
  fit for every row at once via a single fixed pseudo-inverse-free least-
  squares matrix (numpy.linalg.inv only, no numpy.linalg.pinv).
- EDA tonic/phasic split: a short trailing moving average approximates the
  slow tonic skin-conductance level; the residual (phasic) component's
  spread and the tonic component's slope are kept as features.
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
HR_MIN_BPM = 40.0
HR_MAX_BPM = 200.0
N_HARMONICS = 3

XCORR_REF_LEN = 64   # most recent 1-second BVP window, used as the fixed reference
XCORR_STEP = 8        # slide the reference in 8-sample (~0.125s) steps

POLY_DEGREE = 3
EDA_TONIC_WINDOW = 4  # 1 second at 4 Hz

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


def sliding_xcorr_features(arr, ref_len, step, prefix):
    """Fixed most-recent `ref_len`-sample window correlated against
    same-length windows slid across the rest of the signal every `step`
    samples; every offset's normalized correlation is kept as its own
    feature. See part_c_better.py.
    """
    n, T = arr.shape
    ref = arr[:, T - ref_len:]
    ref_c = ref - ref.mean(axis=1, keepdims=True)
    ref_energy = np.sum(ref_c ** 2, axis=1)

    offsets = list(range(0, T - ref_len, step))
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


def harmonic_autocorr_bank(arr, fs, hr_min, hr_max, prefix, n_harmonics=N_HARMONICS):
    """Single-pass lag-correlation matrix reused for: the plain best-lag
    peak, a guarded second peak (periodicity-confidence ratio), and a
    harmonic-sum lag search that reinforces lags whose small integer
    multiples also correlate well (robust to octave errors under motion).
    """
    n, T = arr.shape
    lag_min = max(1, int(np.floor(60.0 * fs / hr_max)))
    lag_max = min(T - 1, int(np.ceil(60.0 * fs / hr_min)))
    max_lag = min(T - 1, n_harmonics * lag_max)

    centered = arr - arr.mean(axis=1, keepdims=True)
    zero_lag_energy = np.sum(centered ** 2, axis=1) + 1e-8

    corr = np.zeros((n, max_lag + 1), dtype=np.float64)  # corr[:, 0] unused
    for lag in range(1, max_lag + 1):
        corr[:, lag] = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / zero_lag_energy

    lags = np.arange(lag_min, lag_max + 1)
    band = corr[:, lag_min:lag_max + 1]

    # Plain best-matching lag.
    best_idx = np.argmax(band, axis=1)
    best_lag = lags[best_idx]
    best_corr = band[np.arange(n), best_idx]
    bpm1 = 60.0 * fs / best_lag

    # Guarded second peak: mask lags within `guard` samples of the best lag
    # so the second peak is a genuinely distinct candidate, not a shoulder
    # of the same peak.
    guard = 3
    idx_range = np.arange(band.shape[1])[None, :]
    mask = np.abs(idx_range - best_idx[:, None]) <= guard
    band_masked = np.where(mask, -np.inf, band)
    second_idx = np.argmax(band_masked, axis=1)
    second_corr = band_masked[np.arange(n), second_idx]
    second_corr = np.where(np.isfinite(second_corr), second_corr, 0.0)
    second_lag = lags[second_idx]
    bpm2 = 60.0 * fs / np.maximum(second_lag, 1)
    peak_ratio = second_corr / (np.abs(best_corr) + 1e-8)

    # Harmonic-sum score: for each candidate fundamental lag L, average the
    # correlation at L, 2L and 3L (dropping multiples that fall outside the
    # lags we actually computed).
    l2 = np.minimum(2 * lags, max_lag)
    l3 = np.minimum(3 * lags, max_lag)
    valid2 = (2 * lags <= max_lag).astype(np.float64)[None, :]
    valid3 = (3 * lags <= max_lag).astype(np.float64)[None, :]
    score = band + valid2 * corr[:, l2] + valid3 * corr[:, l3]
    score = score / (1.0 + valid2 + valid3)
    h_idx = np.argmax(score, axis=1)
    h_lag = lags[h_idx]
    h_score = score[np.arange(n), h_idx]
    h_bpm = 60.0 * fs / h_lag

    feats = np.stack([bpm1, best_corr, bpm2, second_corr, peak_ratio, h_bpm, h_score], axis=1)
    names = [
        f"{prefix}_autocorr_bpm", f"{prefix}_autocorr_peak",
        f"{prefix}_autocorr_bpm2", f"{prefix}_autocorr_peak2",
        f"{prefix}_autocorr_peak_ratio",
        f"{prefix}_harmonic_bpm", f"{prefix}_harmonic_score",
    ]
    return feats, names


def zero_crossing_features(arr, fs, prefix):
    """Average spacing between sign changes of the centered signal gives
    an FFT-free frequency estimate independent of the lag-search above
    (two zero crossings per cycle).
    """
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


def skew_kurtosis_features(arr, prefix):
    """Time-domain PPG signal-quality indices: clean pulse waveforms have a
    characteristic mild skew, while motion/noise flattens or distorts it.
    """
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + 1e-8
    z = (arr - mean) / std
    skew = np.mean(z ** 3, axis=1)
    kurtosis = np.mean(z ** 4, axis=1) - 3.0
    feats = np.stack([skew, kurtosis], axis=1)
    names = [f"{prefix}_skew", f"{prefix}_kurtosis"]
    return feats, names


def second_derivative_features(arr, prefix):
    d2 = np.diff(arr, n=2, axis=1)
    feats = np.stack([d2.mean(axis=1), d2.std(axis=1), np.sum(d2 ** 2, axis=1)], axis=1)
    names = [f"{prefix}_d2_mean", f"{prefix}_d2_std", f"{prefix}_d2_energy"]
    return feats, names


def block_summary(arr, n_blocks, block_len):
    """Per-1-second-block mean/std/energy, shape (n, n_blocks) each."""
    n = arr.shape[0]
    blocks = arr.reshape(n, n_blocks, block_len)
    return blocks.mean(axis=2), blocks.std(axis=2), np.sum(blocks ** 2, axis=2)


def _slope_over_blocks(values):
    """values: (n, n_blocks) -> per-row linear slope across the block index."""
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


def motion_bvp_block_corr(bvp_block_std, accmag_block_std, prefix):
    """Correlation, across the 10 one-second blocks, between BVP energy and
    accelerometer-magnitude energy: high correlation flags a window where
    the pulse waveform's amplitude is likely driven by movement rather than
    the cardiac cycle.
    """
    a = bvp_block_std - bvp_block_std.mean(axis=1, keepdims=True)
    b = accmag_block_std - accmag_block_std.mean(axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a ** 2, axis=1) * np.sum(b ** 2, axis=1)) + 1e-8
    corr = num / den
    return corr[:, None], [f"{prefix}_bvp_acc_block_corr"]


def acc_axis_interaction_features(acc_x, acc_y, acc_z, prefix="acc"):
    xy = np.mean(acc_x * acc_y, axis=1)
    xz = np.mean(acc_x * acc_z, axis=1)
    yz = np.mean(acc_y * acc_z, axis=1)
    feats = np.stack([xy, xz, yz], axis=1)
    names = [f"{prefix}_xy_interact", f"{prefix}_xz_interact", f"{prefix}_yz_interact"]
    return feats, names


def poly_trend_features(arr, degree, prefix):
    """Degree-`degree` polynomial fit to the whole window, one shared
    least-squares solve matrix applied to every row at once (numpy.linalg
    .inv only -- no pinv, no per-row Python loop). The constant term is
    dropped since it duplicates the existing mean feature.
    """
    n, T = arr.shape
    t = np.linspace(-1.0, 1.0, T)
    V = np.vander(t, N=degree + 1, increasing=True)  # (T, degree+1)
    coef_map = np.linalg.inv(V.T @ V) @ V.T           # (degree+1, T)
    coeffs = arr @ coef_map.T                          # (n, degree+1)
    feats = coeffs[:, 1:]
    names = [f"{prefix}_poly_c{k}" for k in range(1, degree + 1)]
    return feats, names


def eda_tonic_phasic_features(eda, window, prefix="eda"):
    """Trailing moving average approximates the slow tonic skin-conductance
    level; the phasic residual's spread and the tonic slope are kept.
    """
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
        blocks.append(f)
        names.extend(n)

    f, n = harmonic_autocorr_bank(signals["bvp"], BVP_FS, HR_MIN_BPM, HR_MAX_BPM, "bvp")
    blocks.append(f); names.extend(n)

    f, n = sliding_xcorr_features(signals["bvp"], XCORR_REF_LEN, XCORR_STEP, "bvp")
    blocks.append(f); names.extend(n)

    f, n = zero_crossing_features(signals["bvp"], BVP_FS, "bvp")
    blocks.append(f); names.extend(n)

    f, n = skew_kurtosis_features(signals["bvp"], "bvp")
    blocks.append(f); names.extend(n)

    f, n = skew_kurtosis_features(signals["acc_mag"], "acc_mag")
    blocks.append(f); names.extend(n)

    f, n = second_derivative_features(signals["bvp"], "bvp")
    blocks.append(f); names.extend(n)

    bvp_bmean, bvp_bstd, bvp_benergy = block_summary(signals["bvp"], N_BLOCKS, BVP_LEN)
    f, n = block_trend_features(bvp_bmean, bvp_bstd, bvp_benergy, "bvp")
    blocks.append(f); names.extend(n)

    acc_bmean, acc_bstd, acc_benergy = block_summary(signals["acc_mag"], N_BLOCKS, ACC_LEN)
    f, n = block_trend_features(acc_bmean, acc_bstd, acc_benergy, "acc_mag")
    blocks.append(f); names.extend(n)

    f, n = motion_bvp_block_corr(bvp_bstd, acc_bstd, "motion")
    blocks.append(f); names.extend(n)

    f, n = acc_axis_interaction_features(signals["acc_x"], signals["acc_y"], signals["acc_z"])
    blocks.append(f); names.extend(n)

    f, n = poly_trend_features(signals["bvp"], POLY_DEGREE, "bvp")
    blocks.append(f); names.extend(n)

    f, n = eda_tonic_phasic_features(signals["eda"], EDA_TONIC_WINDOW, "eda")
    blocks.append(f); names.extend(n)

    return np.hstack(blocks), names


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 part_3_v6.py train.csv test.csv predictions.txt")
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
