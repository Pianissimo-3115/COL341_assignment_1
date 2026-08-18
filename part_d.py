#!/usr/bin/env python3
"""
COL774 Assignment 1 - Part (d): CGM glucose estimation from wearable signals.

    python3 part_d.py train              d1 train_d1/ model_d1.pkl
    python3 part_d.py feature_engineering d1 test_d1/  model_d1.pkl features_d1.npy

The evaluator computes  Yhat = intercept + Z_test @ coef  with no clipping and
no inverse transform, so everything this script learns has to live inside one
intercept and one coefficient vector.

Feature extraction
-------------------
One 5-minute window carries eight raw signals; each is turned into a fixed,
named, finite feature vector.

* Everything is computed per window from that window's own samples, so no
  transformation is ever fitted on test data.
* Arrays are processed one modality at a time and freed immediately, because a
  single participant file can hold ~2 GB in one array (zephyr_acc).
* NaN is the documented missing-value marker.  Every extractor tolerates
  all-NaN input and reports a companion `*_miss` indicator - missingness is
  itself informative (the E4 comes off the wrist during activity).
* R-peak / pulse detection is implemented here (Pan-Tompkins style) rather than
  via NeuroKit2: fewer version surprises and much faster on ~10^3 windows.

Physiological rationale for the feature groups (glucose excursions perturb
autonomic tone, so the cardiac-timing and repolarisation features carry most of
the documented signal):
  ECG   - HRV time/frequency/Poincare + an ensemble-averaged beat template.
          QT prolongation and T-wave flattening are established hypoglycaemia
          markers, so the template is sampled and summarised explicitly.
  BVP   - pulse-rate variability (a PPG mirror of HRV) plus pulse morphology,
          which tracks vascular tone.
  EDA   - tonic level and phasic storm count: sympathetic arousal / sweating.
  TEMP  - skin temperature level and drift, a documented glycaemia biomarker.
  ACC   - activity context (exercise lowers glucose) and artefact gating.
  RESP  - respiration rate and variability.
  cross - pulse arrival time (ECG R-peak -> PPG foot), a vascular-tone proxy
          that needs both chest and wrist devices.

Protocol-aware modelling
-------------------------
The three protocols differ in what generalisation is being asked for, so the
validation scheme (and therefore the selected model) differs:

  d1  random within-subject   -> KFold(shuffle).  The same participant appears
                                 in train and test, so absolute per-person
                                 levels (skin temperature, EDA baseline, resting
                                 HR) are legitimately predictive and are kept.
  d2  temporal within-subject -> a per-subject time-ordered split: validate on
                                 each subject's latest windows, exactly mirroring
                                 the real protocol.
  d3  cross-subject           -> leave-one-subject-out.  Absolute per-person
                                 levels now actively mislead, so d3 additionally
                                 evaluates a WITHIN-SUBJECT-CENTRED variant:
                                 features are centred per training subject, which
                                 makes the fit learn within-person relationships.
                                 Prediction stays exactly ybar + w.(z - zbar),
                                 i.e. still one intercept and one coefficient
                                 vector, so it remains a linear model.

subject_id and timestamp are used only to build these validation splits, never
as predictive features - they are dropped before the design matrix is formed.

The graded metric is NMAE, so after the L2 fit the model is refit by IRLS
towards an L1 objective (an L1-fitted linear model targets the conditional
median, which is what an absolute-error metric rewards).

Permitted libraries only: numpy, pandas, scipy, scikit-learn (all explicitly
allowed for part (d)).  No external data or pretrained models are used.
"""

import gc
import glob
import os
import pickle
import sys
import time

import numpy as np

try:
    from scipy import signal as sps
    HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    HAVE_SCIPY = False

try:
    import resource
    HAVE_RESOURE = True
except Exception:                                    # pragma: no cover (non-Unix)
    HAVE_RESOURE = False

T0 = time.time()
FORMAT_VERSION = 1
MAX_FEATURES = 480          # spec allows < 500; leave headroom
RNG = np.random.RandomState(0)


def _peak_rss_mb():
    """peak resident set size of this process so far, in MiB (Linux: ru_maxrss
    is in KiB).  Lets the 24 GB budget be checked against real numbers instead
    of guessed from array shapes -- watch this in the Kaggle log and tune
    WINDOW_CHUNK / PARTD_WINDOW_CHUNK if it climbs too close to the ceiling."""
    if not HAVE_RESOURE:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def log(msg):
    sys.stderr.write("[%7.1fs | peak %6.0fMB] %s\n" % (time.time() - T0, _peak_rss_mb(), msg))
    sys.stderr.flush()


# ===========================================================================
# Feature extraction (one 5-minute window -> a fixed, named, finite vector)
# ===========================================================================
FS = {"e4_bvp": 64.0, "e4_hr": 1.0, "e4_eda": 4.0, "e4_temp": 4.0, "e4_acc": 32.0,
      "zephyr_ecg": 250.0, "zephyr_acc": 100.0, "zephyr_breathing": 25.0}
WIN_S = 300.0
BIG = 1e12


# ---------------------------------------------------------------- utilities
def _finite(a):
    return np.isfinite(a)


def nan_clean(x, fill=0.0):
    """replace non-finite entries row-wise with the row median, else `fill`."""
    x = np.asarray(x, np.float64)
    bad = ~_finite(x)
    if not bad.any():
        return x
    x = x.copy()
    if x.ndim == 1:
        good = x[~bad]
        x[bad] = np.median(good) if good.size else fill
        return x
    med = np.where(_finite(x), x, np.nan)
    with np.errstate(all="ignore"):
        m = np.nanmedian(med, axis=1)
    m = np.where(_finite(m), m, fill)
    idx = np.where(bad)
    x[idx] = m[idx[0]]
    return x


def safe(v, fill=0.0):
    v = np.asarray(v, np.float64)
    return np.where(_finite(v), np.clip(v, -BIG, BIG), fill)


def stats_block(x, prefix, qs=(5, 25, 50, 75, 95)):
    """generic distribution summary of a (n, L) signal"""
    n = len(x)
    val = _finite(x)
    frac = val.mean(1)
    xs = np.where(val, x, np.nan)
    with np.errstate(all="ignore"):
        mean = np.nanmean(xs, 1)
        std = np.nanstd(xs, 1)
        quant = np.nanpercentile(xs, qs, axis=1)
        mn = np.nanmin(xs, 1)
        mx = np.nanmax(xs, 1)
        dif = np.nanmean(np.abs(np.diff(xs, axis=1)), 1)
    t = np.arange(x.shape[1], dtype=np.float64)
    t = (t - t.mean()) / (t.std() + 1e-9)
    with np.errstate(all="ignore"):
        slope = np.nanmean(np.where(val, xs * t, np.nan), 1)
    cols, names = [], []
    for nm, v in (("mean", mean), ("std", std), ("min", mn), ("max", mx),
                  ("rng", mx - mn), ("mad", dif), ("slope", slope),
                  ("cv", std / (np.abs(mean) + 1e-6)), ("valid", frac)):
        cols.append(safe(v)); names.append("%s_%s" % (prefix, nm))
    for i, q in enumerate(qs):
        cols.append(safe(quant[i])); names.append("%s_q%d" % (prefix, q))
    cols.append(safe(quant[-1] - quant[0])); names.append("%s_iqr9" % prefix)
    cols.append((frac < 0.5).astype(np.float64)); names.append("%s_miss" % prefix)
    return np.stack(cols, 1), names


def butter_filt(x, fs, lo=None, hi=None, order=3):
    """zero-phase band-pass along axis 1; falls back to moving averages."""
    if HAVE_SCIPY:
        nyq = 0.5 * fs
        try:
            if lo and hi:
                b, a = sps.butter(order, [lo / nyq, min(hi / nyq, 0.99)], btype="band")
            elif hi:
                b, a = sps.butter(order, min(hi / nyq, 0.99), btype="low")
            else:
                b, a = sps.butter(order, lo / nyq, btype="high")
            return sps.filtfilt(b, a, x, axis=1)
        except Exception:
            pass
    y = x
    if hi:
        w = max(int(fs / hi), 1)
        y = _movavg(y, w)
    if lo:
        w = max(int(fs / lo), 1)
        y = y - _movavg(y, w)
    return y


def _movavg(X, w):
    if w <= 1:
        return X
    pad = w // 2
    Xp = np.pad(X, ((0, 0), (pad, w - 1 - pad)), mode="edge")
    cs = np.cumsum(Xp, axis=1)
    cs = np.concatenate([np.zeros((len(X), 1)), cs], 1)
    return (cs[:, w:] - cs[:, :-w]) / w


def welch_bands(x, fs, bands, prefix, nperseg=None):
    """band powers of a (n, L) signal, plus spectral centroid / entropy."""
    n, L = x.shape
    nperseg = nperseg or min(L, int(fs * 60) if fs >= 1 else L)
    if HAVE_SCIPY:
        f, P = sps.welch(x, fs=fs, nperseg=min(nperseg, L), axis=1)
    else:
        f = np.linspace(0, fs / 2, L // 2 + 1)
        P = _dft_psd(x, fs, f)
    tot = P.sum(1) + 1e-20
    cols, names = [], []
    for lo, hi, nm in bands:
        m = (f >= lo) & (f < hi)
        p = P[:, m].sum(1) if m.any() else np.zeros(n)
        cols.append(safe(np.log(p + 1e-12))); names.append("%s_lp_%s" % (prefix, nm))
        cols.append(safe(p / tot)); names.append("%s_rp_%s" % (prefix, nm))
    Pn = P / tot[:, None]
    cols.append(safe(Pn @ f)); names.append("%s_centroid" % prefix)
    cols.append(safe(-(Pn * np.log(Pn + 1e-12)).sum(1))); names.append("%s_sent" % prefix)
    cols.append(safe(np.log(tot))); names.append("%s_ltot" % prefix)
    cols.append(safe(f[np.argmax(P, 1)])); names.append("%s_fpeak" % prefix)
    return np.stack(cols, 1), names


def _dft_psd(x, fs, f):
    t = np.arange(x.shape[1]) / fs
    ang = 2 * np.pi * np.outer(t, f)
    xc = x - x.mean(1, keepdims=True)
    return (xc @ np.cos(ang)) ** 2 + (xc @ np.sin(ang)) ** 2


# ------------------------------------------------------------ beat detection
def detect_peaks_rows(xf, fs, min_bpm=35.0, max_bpm=200.0, thr=0.35):
    """Pan-Tompkins style detection on an already band-passed (n, L) array.
    Returns a list of index arrays (one per row)."""
    d = np.diff(xf, axis=1, prepend=xf[:, :1])
    e = d * d
    w = max(int(0.10 * fs), 1)
    ig = _movavg(e, w)
    dist = int(fs * 60.0 / max_bpm)
    out = []
    for i in range(len(ig)):
        row = ig[i]
        if not np.isfinite(row).any():
            out.append(np.empty(0, np.int64)); continue
        row = np.nan_to_num(row)
        ref = np.percentile(row, 98)
        if ref <= 0:
            out.append(np.empty(0, np.int64)); continue
        if HAVE_SCIPY:
            pk, _ = sps.find_peaks(row, height=thr * ref, distance=max(dist, 1))
        else:
            pk = _simple_peaks(row, thr * ref, max(dist, 1))
        out.append(pk)
    return out


def _simple_peaks(row, height, dist):
    cand = np.where((row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:]) & (row[1:-1] > height))[0] + 1
    keep = []
    last = -10 ** 9
    for c in cand:
        if c - last >= dist:
            keep.append(c); last = c
        elif keep and row[c] > row[keep[-1]]:
            keep[-1] = c; last = c
    return np.asarray(keep, np.int64)


def rr_features(peaks, fs, prefix, n):
    """HRV / PRV from a list of peak-index arrays."""
    keys = ["nbeat", "hr", "hrstd", "meanrr", "sdnn", "rmssd", "pnn50", "pnn20",
            "cvnn", "medrr", "iqrrr", "sd1", "sd2", "sd12", "lf", "hf", "lfhf",
            "vlf", "tot", "acc_frac", "rrmin", "rrmax", "rrslope", "rrskew"]
    out = np.zeros((n, len(keys)))
    for i, pk in enumerate(peaks):
        if len(pk) < 6:
            continue
        rr = np.diff(pk) / fs
        ok = (rr > 0.28) & (rr < 2.2)
        acc_frac = ok.mean()
        rr = rr[ok]
        if len(rr) < 5:
            out[i, keys.index("acc_frac")] = acc_frac
            continue
        # drop ectopic-looking jumps
        med = np.median(rr)
        rr = rr[np.abs(rr - med) < 0.5 * med + 1e-9]
        if len(rr) < 5:
            out[i, keys.index("acc_frac")] = acc_frac
            continue
        d = np.diff(rr)
        v = {"nbeat": len(rr), "hr": 60.0 / np.mean(rr), "hrstd": np.std(60.0 / rr),
             "meanrr": np.mean(rr), "sdnn": np.std(rr),
             "rmssd": np.sqrt(np.mean(d ** 2)),
             "pnn50": np.mean(np.abs(d) > 0.05), "pnn20": np.mean(np.abs(d) > 0.02),
             "cvnn": np.std(rr) / (np.mean(rr) + 1e-9), "medrr": np.median(rr),
             "iqrrr": np.percentile(rr, 75) - np.percentile(rr, 25),
             "acc_frac": acc_frac, "rrmin": rr.min(), "rrmax": rr.max()}
        sd1 = np.sqrt(0.5) * np.std(d)
        sd2 = np.sqrt(max(2 * np.var(rr) - 0.5 * np.var(d), 0.0))
        v["sd1"], v["sd2"] = sd1, sd2
        v["sd12"] = sd1 / (sd2 + 1e-9)
        tt = np.arange(len(rr), dtype=np.float64)
        tt = (tt - tt.mean()) / (tt.std() + 1e-9)
        v["rrslope"] = float(np.mean(rr * tt))
        s = np.std(rr) + 1e-12
        v["rrskew"] = float(np.mean(((rr - np.mean(rr)) / s) ** 3))
        # tachogram spectrum on a uniform 4 Hz grid
        tcum = np.cumsum(rr)
        grid = np.arange(0, tcum[-1], 0.25)
        if len(grid) > 16:
            xi = np.interp(grid, tcum, rr)
            xi = xi - xi.mean()
            # explicit DFT keeps this independent of numpy.fft availability
            fgrid = np.linspace(0.003, 0.5, 120)
            ang = 2 * np.pi * np.outer(np.arange(len(xi)) * 0.25, fgrid)
            P = (xi @ np.cos(ang)) ** 2 + (xi @ np.sin(ang)) ** 2
            def bp(a, b):
                m = (fgrid >= a) & (fgrid < b)
                return float(P[m].sum()) if m.any() else 0.0
            vlf, lf, hf = bp(0.003, 0.04), bp(0.04, 0.15), bp(0.15, 0.4)
            v["vlf"], v["lf"], v["hf"] = np.log1p(vlf), np.log1p(lf), np.log1p(hf)
            v["lfhf"] = lf / (hf + 1e-9)
            v["tot"] = np.log1p(vlf + lf + hf)
        for k, val in v.items():
            out[i, keys.index(k)] = val
    return safe(out), ["%s_%s" % (prefix, k) for k in keys]


def beat_template(x, peaks, fs, pre_s, post_s, ntpl, prefix):
    """ensemble-averaged beat, amplitude-normalised, resampled to `ntpl` points.
    Captures QRS/T morphology (repolarisation changes track glycaemia)."""
    n, L = x.shape
    pre, post = int(pre_s * fs), int(post_s * fs)
    tpl = np.zeros((n, ntpl))
    extra = np.zeros((n, 6))
    src = np.linspace(0, pre + post - 1, ntpl)
    for i in range(n):
        pk = peaks[i]
        pk = pk[(pk >= pre) & (pk + post < L)]
        if len(pk) < 3:
            continue
        if len(pk) > 60:
            pk = pk[np.linspace(0, len(pk) - 1, 60).astype(int)]
        seg = np.stack([x[i, p - pre:p + post] for p in pk])
        seg = np.where(_finite(seg), seg, np.nan)
        with np.errstate(all="ignore"):
            m = np.nanmean(seg, 0)
        if not _finite(m).all():
            m = nan_clean(m[None, :])[0]
        amp = np.percentile(m, 99) - np.percentile(m, 1)
        mn = (m - np.median(m)) / (amp + 1e-9)
        tpl[i] = np.interp(src, np.arange(pre + post), mn)
        # repolarisation window: 0.15-0.45 s after the fiducial point
        a, b = pre + int(0.15 * fs), pre + int(min(0.45, post_s) * fs)
        if b > a + 2:
            seg_t = mn[a:b]
            j = int(np.argmax(np.abs(seg_t)))
            extra[i] = [amp, seg_t[j], (j + a - pre) / fs,
                        float(np.trapz(seg_t) / fs), float(np.std(seg_t)),
                        float(np.mean(np.abs(np.diff(mn))))]
        else:
            extra[i, 0] = amp
    names = ["%s_tpl%02d" % (prefix, k) for k in range(ntpl)]
    names += ["%s_amp" % prefix, "%s_twave" % prefix, "%s_ttime" % prefix,
              "%s_tarea" % prefix, "%s_tstd" % prefix, "%s_rough" % prefix]
    return safe(np.hstack([tpl, extra])), names


# ------------------------------------------------------------- per-modality
def feat_ecg(ecg):
    fs = FS["zephyr_ecg"]
    x = nan_clean(ecg)
    xf = butter_filt(x, fs, lo=5.0, hi=25.0)
    peaks = detect_peaks_rows(xf, fs)
    F1, N1 = rr_features(peaks, fs, "ecg", len(x))
    xb = butter_filt(x, fs, lo=0.5, hi=40.0)
    F2, N2 = beat_template(xb, peaks, fs, 0.25, 0.45, 32, "ecg")
    F3, N3 = stats_block(np.asarray(ecg, np.float64)[:, ::10], "ecgraw")
    return np.hstack([F1, F2, F3]), N1 + N2 + N3, peaks


def feat_bvp(bvp):
    fs = FS["e4_bvp"]
    x = nan_clean(bvp)
    xf = butter_filt(x, fs, lo=0.7, hi=3.5)
    peaks = detect_peaks_rows(xf, fs, thr=0.30)
    F1, N1 = rr_features(peaks, fs, "ppg", len(x))
    F2, N2 = beat_template(xf, peaks, fs, 0.20, 0.45, 24, "ppg")
    F3, N3 = stats_block(np.asarray(bvp, np.float64)[:, ::4], "bvpraw")
    F4, N4 = welch_bands(xf, fs, [(0.7, 1.2, "a"), (1.2, 2.0, "b"), (2.0, 3.5, "c"),
                                  (3.5, 8.0, "d")], "bvp")
    return np.hstack([F1, F2, F3, F4]), N1 + N2 + N3 + N4, peaks


def feat_eda(eda):
    fs = FS["e4_eda"]
    x = np.asarray(eda, np.float64)
    F1, N1 = stats_block(x, "eda")
    xc = nan_clean(x)
    tonic = _movavg(xc, int(fs * 20))
    phasic = xc - tonic
    ph = np.abs(phasic)
    thr = (np.percentile(ph, 95, axis=1) + 1e-6)[:, None]
    cols = [safe(tonic.mean(1)), safe(tonic.std(1)),
            safe(phasic.std(1)), safe(ph.mean(1)),
            safe((ph > 0.02).mean(1)), safe((ph > thr * 0.8).sum(1)),
            safe(np.log(phasic.std(1) + 1e-4)),
            safe(tonic[:, -1] - tonic[:, 0])]
    names = ["eda_tonic_mean", "eda_tonic_std", "eda_phasic_std", "eda_phasic_amp",
             "eda_scr_frac", "eda_scr_n", "eda_lphasic", "eda_drift"]
    return np.hstack([F1, np.stack(cols, 1)]), N1 + names


def feat_temp(temp):
    F, N = stats_block(np.asarray(temp, np.float64), "temp")
    x = nan_clean(temp)
    d = x[:, -int(FS["e4_temp"] * 60):].mean(1) - x[:, :int(FS["e4_temp"] * 60)].mean(1)
    return np.hstack([F, safe(d)[:, None]]), N + ["temp_delta"]


def feat_hr1(hr):
    F, N = stats_block(np.asarray(hr, np.float64), "e4hr")
    x = nan_clean(hr)
    d = np.diff(x, axis=1)
    extra = np.stack([safe(np.abs(d).mean(1)), safe(d.std(1)),
                      safe(x[:, -60:].mean(1) - x[:, :60].mean(1))], 1)
    return np.hstack([F, extra]), N + ["e4hr_absdiff", "e4hr_dstd", "e4hr_delta"]


def feat_acc(acc, tag):
    """acc: (n, L, 3).  Activity context + posture + artefact level."""
    a = np.asarray(acc, np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    fs = FS["e4_acc" if tag == "e4acc" else "zephyr_acc"]
    mag = np.sqrt(np.nansum(a ** 2, axis=2))
    F1, N1 = stats_block(mag, tag + "mag")
    m = nan_clean(mag)
    enmo = np.maximum(m - np.median(m, axis=1, keepdims=True), 0).mean(1)
    hp = m - _movavg(m, int(fs * 2))
    cols = [safe(enmo), safe(np.log(hp.std(1) + 1e-4)),
            safe((np.abs(hp) < 0.02 * (np.std(hp, axis=1) + 1e-6)[:, None]).mean(1)),
            safe(np.abs(np.diff(m, axis=1)).mean(1))]
    names = [tag + "_enmo", tag + "_lhp", tag + "_still", tag + "_jerk"]
    for k in range(a.shape[2]):
        ax = np.where(_finite(a[:, :, k]), a[:, :, k], np.nan)
        with np.errstate(all="ignore"):
            cols += [safe(np.nanmean(ax, 1)), safe(np.nanstd(ax, 1))]
        names += ["%s_m%d" % (tag, k), "%s_s%d" % (tag, k)]
    F2, N2 = welch_bands(hp, fs, [(0.1, 0.5, "vl"), (0.5, 1.5, "l"), (1.5, 3.0, "m"),
                                  (3.0, 8.0, "h")], tag)
    return np.hstack([F1, np.stack(cols, 1), F2]), N1 + names + N2


def feat_resp(br):
    fs = FS["zephyr_breathing"]
    x = nan_clean(br)
    F1, N1 = stats_block(np.asarray(br, np.float64), "resp")
    xf = butter_filt(x, fs, lo=0.08, hi=0.8)
    F2, N2 = welch_bands(xf, fs, [(0.08, 0.16, "slow"), (0.16, 0.30, "norm"),
                                  (0.30, 0.60, "fast")], "resp")
    pk = detect_peaks_rows(xf, fs, min_bpm=5.0, max_bpm=40.0, thr=0.25)
    F3, N3 = rr_features(pk, fs, "resprr", len(x))
    keep = [i for i, nm in enumerate(N3) if nm.split("_")[-1] in
            ("nbeat", "hr", "sdnn", "rmssd", "cvnn", "medrr")]
    return np.hstack([F1, F2, F3[:, keep]]), N1 + N2 + [N3[i] for i in keep]


def feat_cross(ecg_peaks, ppg_peaks, n):
    """pulse arrival time: ECG R-peak -> next BVP pulse.  A vascular-tone proxy
    that only a chest+wrist pair can give, and one of the few features whose
    physiology (arterial stiffness / viscosity) links plausibly to glucose."""
    fe, fp = FS["zephyr_ecg"], FS["e4_bvp"]
    out = np.zeros((n, 6))
    for i in range(n):
        pe, pp = ecg_peaks[i], ppg_peaks[i]
        if len(pe) < 4 or len(pp) < 4:
            continue
        te, tp = pe / fe, pp / fp
        j = np.searchsorted(tp, te)
        j = np.clip(j, 0, len(tp) - 1)
        pat = tp[j] - te
        ok = (pat > 0.08) & (pat < 0.60)
        hre = 60.0 / (np.median(np.diff(te)) + 1e-9) if len(te) > 2 else 0.0
        hrp = 60.0 / (np.median(np.diff(tp)) + 1e-9) if len(tp) > 2 else 0.0
        out[i, 4] = hre - hrp
        out[i, 5] = ok.mean()
        if ok.sum() >= 3:
            p = pat[ok]
            out[i, :4] = [np.median(p), np.std(p),
                          np.percentile(p, 75) - np.percentile(p, 25), np.mean(p)]
    names = ["pat_med", "pat_std", "pat_iqr", "pat_mean", "hr_ecg_ppg_diff", "pat_frac"]
    return safe(out), names


def feat_context(blocks, names):
    """a handful of derived, unit-free combinations: sleep-likeness (low motion,
    low HR, stable temperature) and normalised ratios that transfer across
    people better than absolute levels do."""
    idx = {nm: k for k, nm in enumerate(names)}

    def g(nm):
        return blocks[:, idx[nm]] if nm in idx else np.zeros(len(blocks))

    hr = g("ecg_hr")
    hr = np.where(hr > 0, hr, g("ppg_hr"))
    still = g("e4accmag_std")
    cols = [safe(g("ecg_rmssd") * hr / 60.0),
            safe(g("ecg_sdnn") / (g("ecg_meanrr") + 1e-6)),
            safe(g("ppg_hr") / (hr + 1e-6)),
            safe(np.log(still + 1e-4) - np.log(g("zaccmag_std") + 1e-4)),
            safe(g("temp_mean") - g("temp_q50")),
            safe(g("eda_phasic_std") / (g("eda_mean") + 1e-3)),
            safe(g("resp_fpeak") * 60.0),
            safe(g("resp_fpeak") * 60.0 / (hr + 1e-6)),
            safe(-np.log(still + 1e-4) - hr / 20.0)]
    nm = ["ctx_rmssd_hr", "ctx_sdnn_rr", "ctx_hr_ratio", "ctx_motion_ratio",
          "ctx_temp_skew", "ctx_eda_ratio", "ctx_rr_bpm", "ctx_rr_hr", "ctx_sleepiness"]
    return np.stack(cols, 1), nm


# ---------------------------------------------------------------- top level
MODALITIES = ("e4_bvp", "e4_hr", "e4_eda", "e4_temp", "e4_acc",
              "zephyr_ecg", "zephyr_acc", "zephyr_breathing")

# Row-batch size for feature computation.  numpy.load on an .npz member
# decompresses the WHOLE array on first access regardless of any slicing
# (see the assignment PDF's "Working with Part (d) .npz files" note), so
# reading a modality in smaller pieces would only decompress it repeatedly
# for no memory benefit.  What chunking DOES bound is the transient working
# set every feat_* function allocates per call (float64 casts, band-passed
# copies, diff arrays, beat-template stacks, ...), which is O(rows) and can
# be several times the size of the raw array itself.  A participant file
# with tens of thousands of windows would otherwise materialise all of that
# at once; capping it at WINDOW_CHUNK rows keeps peak RSS bounded no matter
# how many windows one file holds.  Fixed constant, not read from the
# environment: the evaluator only runs the exact command lines the PDF
# specifies, with no env vars set, so this must work as shipped.  Measured
# on the real dataset: chunk=1000 -> ~5GB peak RSS, well under the 24GB
# budget, so this is scaled up to target ~15GB (comfortable margin) rather
# than the smallest safe value.
WINDOW_CHUNK = 3000


def _chunked2(fn, arr, chunk):
    """Apply a (F, names) feat_* function over row-batches of `arr` and
    vstack the results.  Every feat_* function here computes each row's
    features independently (no cross-row/batch statistics), so this is
    exactly equivalent to a single call over the whole array."""
    n = len(arr)
    if n <= chunk:
        return fn(arr)
    Fs, names = [], None
    for i in range(0, n, chunk):
        F, names = fn(arr[i:i + chunk])
        Fs.append(F)
        del F
        gc.collect()
    return np.vstack(Fs), names


def _chunked3(fn, arr, chunk):
    """Like _chunked2 but for the (F, names, peaks) signature of feat_ecg /
    feat_bvp; peaks is a plain list of per-row index arrays and is simply
    concatenated in order."""
    n = len(arr)
    if n <= chunk:
        return fn(arr)
    Fs, names, peaks = [], None, []
    for i in range(0, n, chunk):
        F, names, pk = fn(arr[i:i + chunk])
        Fs.append(F)
        peaks.extend(pk)
        del F
        gc.collect()
    return np.vstack(Fs), names, peaks


def features_for_file(path, verbose=False):
    """Extract features for one participant .npz, one modality at a time,
    each modality processed in row-batches of WINDOW_CHUNK windows."""
    blocks, names = [], []
    ecg_peaks = ppg_peaks = None
    with np.load(path, allow_pickle=False) as z:
        avail = set(z.files)
        n = None
        for key in ("glucose", "subject_id", "timestamp", "e4_hr", "e4_eda"):
            if key in avail:
                n = len(z[key]); break
        if n is None:
            raise ValueError("cannot determine window count in %s" % path)

        def take(key):
            return np.asarray(z[key]) if key in avail else None

        a = take("zephyr_ecg")
        if a is not None:
            F, N, ecg_peaks = _chunked3(feat_ecg, a, WINDOW_CHUNK); blocks.append(F); names += N
            del a; gc.collect()
        a = take("e4_bvp")
        if a is not None:
            F, N, ppg_peaks = _chunked3(feat_bvp, a, WINDOW_CHUNK); blocks.append(F); names += N
            del a; gc.collect()
        for key, fn in (("e4_hr", feat_hr1), ("e4_eda", feat_eda),
                        ("e4_temp", feat_temp), ("zephyr_breathing", feat_resp)):
            a = take(key)
            if a is not None:
                F, N = _chunked2(fn, a, WINDOW_CHUNK); blocks.append(F); names += N
                del a; gc.collect()
        for key, tag in (("e4_acc", "e4acc"), ("zephyr_acc", "zacc")):
            a = take(key)
            if a is not None:
                F, N = _chunked2(lambda x, tag=tag: feat_acc(x, tag), a, WINDOW_CHUNK)
                blocks.append(F); names += N
                del a; gc.collect()
        if ecg_peaks is not None and ppg_peaks is not None:
            F, N = feat_cross(ecg_peaks, ppg_peaks, n); blocks.append(F); names += N
        y = np.asarray(z["glucose"], np.float64) if "glucose" in avail else None
        sid = np.asarray(z["subject_id"]).astype(str) if "subject_id" in avail else \
            np.full(n, path, dtype=object)
        ts = np.asarray(z["timestamp"], np.float64) if "timestamp" in avail else \
            np.arange(n, dtype=np.float64)

    X = np.hstack(blocks) if blocks else np.zeros((n, 0))
    C, CN = feat_context(X, names)
    X = np.hstack([X, C]); names = names + CN
    X = np.where(_finite(X), np.clip(X, -BIG, BIG), np.nan)
    if verbose:
        print("  %s -> %d windows, %d raw features" % (path, len(X), X.shape[1]))
    return X, y, np.asarray(sid, dtype=object), ts, names


# ===========================================================================
# Data loading
# ===========================================================================
def list_subject_files(directory):
    """Must match eval_d.py exactly or labels will not line up with rows."""
    paths = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not paths:
        raise SystemExit("no .npz files found in %s" % directory)
    return paths


def load_directory(directory):
    Xs, ys, sids, tss, names = [], [], [], [], None
    for path in list_subject_files(directory):
        X, y, sid, ts, nm = features_for_file(path)
        if names is None:
            names = nm
        elif nm != names:
            common = [c for c in names if c in set(nm)]
            idx_new = {c: i for i, c in enumerate(nm)}
            X = X[:, [idx_new[c] for c in common]]
            for k in range(len(Xs)):
                idx_old = {c: i for i, c in enumerate(names)}
                Xs[k] = Xs[k][:, [idx_old[c] for c in common]]
            names = common
        Xs.append(X)
        ys.append(y if y is not None else np.full(len(X), np.nan))
        sids.append(sid)
        tss.append(ts)
        log("  %s: %d windows" % (os.path.basename(path), len(X)))
        gc.collect()
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    sid = np.concatenate(sids)
    ts = np.concatenate(tss)
    return X, y, sid, ts, names


# ===========================================================================
# Preprocessing (fitted on training data only, stored for reuse at test time)
# ===========================================================================
def fit_preprocess(X, names):
    """Impute -> drop degenerate -> winsorise -> standardise.  All statistics
    come from the training matrix only and are stored for reuse at test time."""
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med)

    lo = np.percentile(Xi, 0.5, axis=0)
    hi = np.percentile(Xi, 99.5, axis=0)
    span = hi - lo
    hi = np.where(span > 0, hi, lo + 1.0)
    Xw = np.clip(Xi, lo, hi)

    sd = Xw.std(0)
    keep = np.where(sd > 1e-9)[0]
    if len(keep) == 0:
        raise SystemExit("all features are constant")
    Xw = Xw[:, keep]
    mu = Xw.mean(0)
    sd = Xw.std(0) + 1e-9
    state = {"median": med, "lo": lo, "hi": hi, "keep": keep, "mu": mu, "sd": sd,
             "names_in": list(names)}
    return (Xw - mu) / sd, state, [names[i] for i in keep]


# ===========================================================================
# Validation splits (subject_id / timestamp are permitted for this only)
# ===========================================================================
def make_splits(protocol, sid, ts, n, nfold=5):
    subs = np.unique(sid)
    splits = []
    if protocol == "d3":
        for s in subs:                                   # leave-one-subject-out
            te = np.where(sid == s)[0]
            tr = np.where(sid != s)[0]
            if len(te) and len(tr):
                splits.append((tr, te))
        if len(splits) > 8:
            splits = splits[:8]
    elif protocol == "d2":
        tr, te = [], []
        for s in subs:                                   # latest 25% per subject
            idx = np.where(sid == s)[0]
            idx = idx[np.argsort(ts[idx], kind="stable")]
            cut = max(int(len(idx) * 0.75), 1)
            tr.append(idx[:cut]); te.append(idx[cut:])
        tr, te = np.concatenate(tr), np.concatenate(te)
        if len(te):
            splits.append((tr, te))
        tr2, te2 = [], []                                # a second, earlier cut
        for s in subs:
            idx = np.where(sid == s)[0]
            idx = idx[np.argsort(ts[idx], kind="stable")]
            a, b = max(int(len(idx) * 0.5), 1), max(int(len(idx) * 0.75), 2)
            tr2.append(idx[:a]); te2.append(idx[a:b])
        tr2, te2 = np.concatenate(tr2), np.concatenate(te2)
        if len(te2):
            splits.append((tr2, te2))
    else:                                                # d1: random KFold
        order = RNG.permutation(n)
        folds = np.array_split(order, nfold)
        for k in range(nfold):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(nfold) if j != k])
            splits.append((tr, te))
    if not splits:
        cut = int(n * 0.75)
        splits = [(np.arange(cut), np.arange(cut, n))]
    return splits


# ===========================================================================
# Model: ridge on the CV-selected alpha, IRLS-refit towards an L1 objective
# ===========================================================================
ALPHAS = 10.0 ** np.arange(-1.0, 5.01, 0.5)


def nmae(y, p):
    return float(np.abs(y - p).sum() / (np.abs(y - y.mean()).sum() + 1e-12))


def nmse(y, p):
    return float(((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))


def ridge_path(Xtr, ytr, Xva, alphas):
    """one eigendecomposition gives every alpha."""
    ym = ytr.mean()
    G = Xtr.T @ Xtr
    b = Xtr.T @ (ytr - ym)
    d, V = np.linalg.eigh(G)
    bt = V.T @ b
    Pv = Xva @ V
    for a in alphas:
        yield a, Pv @ (bt / (d + a)) + ym, V, bt, d, ym


def fit_ridge(X, y, alpha):
    ym = y.mean()
    G = X.T @ X + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(G, X.T @ (y - ym))
    return w, ym


def irls_refit(X, y, alpha, w, ym, iters=4):
    """push the L2 solution toward an L1 objective (metric is NMAE)."""
    best_w, best_ym = w, ym
    best = nmae(y, X @ w + ym)
    for _ in range(iters):
        r = y - (X @ w + ym)
        floor = 1.345 * np.median(np.abs(r - np.median(r))) + 1e-6
        sw = 1.0 / np.sqrt(np.maximum(np.abs(r), floor))
        Wm = sw ** 2
        ym_w = float((Wm * y).sum() / Wm.sum())
        Xw = X * sw[:, None]
        G = Xw.T @ Xw + alpha * float((Wm).mean()) * np.eye(X.shape[1])
        w_new = np.linalg.solve(G, Xw.T @ ((y - ym_w) * sw))
        s = nmae(y, X @ w_new + ym_w)
        if s < best:
            best, best_w, best_ym = s, w_new, ym_w
        w, ym = w_new, ym_w
    return best_w, best_ym


def select_features(X, y, names, k):
    """keep the k features with the strongest |correlation| with the target.
    Fitted on training data only; the chosen indices are stored in the model."""
    if X.shape[1] <= k:
        return np.arange(X.shape[1])
    yc = y - y.mean()
    denom = (X.std(0) + 1e-12) * (yc.std() + 1e-12) * len(y)
    corr = np.abs((X * yc[:, None]).sum(0) / denom)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    return np.sort(np.argsort(-corr)[:k])


def subject_center(X, sid):
    """centre each feature within each subject; returns centred X and the grand
    mean that keeps the prediction affine."""
    Xc = X.copy()
    for s in np.unique(sid):
        m = sid == s
        Xc[m] -= Xc[m].mean(0)
    return Xc


def evaluate_config(X, y, sid, ts, protocol, splits, centred, alphas):
    scores = {a: [] for a in alphas}
    for tr, te in splits:
        Xtr, ytr = X[tr], y[tr]
        if centred:
            Xtr = subject_center(Xtr, sid[tr])
        for a, pred, _, _, _, _ in ridge_path(Xtr, ytr, X[te], alphas):
            scores[a].append(nmae(y[te], pred))
    return {a: float(np.mean(v)) for a, v in scores.items()}


# ===========================================================================
# train
# ===========================================================================
def do_train(protocol, train_dir, model_path):
    log("loading %s" % train_dir)
    X, y, sid, ts, names = load_directory(train_dir)
    log("raw features %s" % (X.shape,))
    if not np.isfinite(y).all():
        m = np.isfinite(y)
        log("dropping %d training rows with non-finite glucose" % (~m).sum())
        X, y, sid, ts = X[m], y[m], sid[m], ts[m]
    if len(y) < 20:
        raise SystemExit("not enough labelled training windows (%d)" % len(y))

    Xs, prep, names_kept = fit_preprocess(X, names)
    del X
    gc.collect()
    log("after preprocessing %s" % (Xs.shape,))

    sel = select_features(Xs, y, names_kept, MAX_FEATURES)
    Xs = Xs[:, sel]
    names_kept = [names_kept[i] for i in sel]
    prep["select"] = sel
    log("selected %d features" % Xs.shape[1])

    splits = make_splits(protocol, sid, ts, len(y))
    log("%d validation split(s) for %s" % (len(splits), protocol))

    variants = [False, True] if protocol == "d3" else [False]
    best = None
    for centred in variants:
        sc = evaluate_config(Xs, y, sid, ts, protocol, splits, centred, ALPHAS)
        a_best = min(sc, key=sc.get)
        log("  centred=%s  best CV NMAE %.4f at alpha=%.3g" % (centred, sc[a_best], a_best))
        if best is None or sc[a_best] < best[0]:
            best = (sc[a_best], a_best, centred)
    cv_nmae, alpha, centred = best
    log("chosen: alpha=%.3g centred=%s (CV NMAE %.4f)" % (alpha, centred, cv_nmae))

    Xfit = subject_center(Xs, sid) if centred else Xs
    w, ym = fit_ridge(Xfit, y, alpha)
    w, ym = irls_refit(Xfit, y, alpha, w, ym)

    # Fold the centring offset into the intercept so the stored model stays
    # exactly  yhat = intercept + z @ coef  on the untouched feature scale.
    offset = Xs.mean(0) - Xfit.mean(0) if centred else np.zeros(Xs.shape[1])
    intercept_std = float(ym - offset @ w)

    # Undo standardisation so coef applies to the raw selected features.
    mu = prep["mu"][sel]
    sd = prep["sd"][sel]
    coef = w / sd
    intercept = float(intercept_std - float(mu @ coef))

    train_nmae = nmae(y, Xs @ w + intercept_std)
    log("train NMAE %.4f  NMSE %.4f" % (train_nmae, nmse(y, Xs @ w + intercept_std)))

    state = {
        "format_version": FORMAT_VERSION,
        "protocol": protocol,
        "intercept": intercept,
        "coef": np.asarray(coef, np.float64),
        "feature_names": list(names_kept),
        "preprocessing_state": {
            "median": prep["median"], "lo": prep["lo"], "hi": prep["hi"],
            "keep": prep["keep"], "select": sel,
            "names_in": prep["names_in"], "names_out": list(names_kept),
            "cv_nmae": cv_nmae, "alpha": float(alpha), "centred": bool(centred),
        },
    }
    assert len(state["coef"]) == len(state["feature_names"]) < 500
    with open(model_path, "wb") as f:
        pickle.dump(state, f, protocol=4)
    log("wrote %s (%d features)" % (model_path, len(coef)))
    return 0


# ===========================================================================
# feature_engineering
# ===========================================================================
def do_features(protocol, test_dir, model_path, out_path):
    with open(model_path, "rb") as f:
        state = pickle.load(f)
    if state.get("protocol") != protocol:
        log("WARNING: model protocol %r != requested %r" % (state.get("protocol"), protocol))
    ps = state["preprocessing_state"]

    log("loading %s" % test_dir)
    X, _, _, _, names = load_directory(test_dir)
    log("raw test features %s" % (X.shape,))

    # align raw columns to the training layout by name where possible
    if names != ps["names_in"]:
        idx = {c: i for i, c in enumerate(names)}
        cols = []
        for c in ps["names_in"]:
            cols.append(idx[c] if c in idx else -1)
        Xa = np.full((len(X), len(cols)), np.nan)
        for j, c in enumerate(cols):
            if c >= 0:
                Xa[:, j] = X[:, c]
        X = Xa
        log("realigned test columns to the training feature layout")

    med, lo, hi = ps["median"], ps["lo"], ps["hi"]
    Xi = np.where(np.isfinite(X), X, med)
    Z = np.clip(Xi, lo, hi)[:, ps["keep"]][:, ps["select"]]
    Z = np.where(np.isfinite(Z), Z, 0.0)

    if Z.shape[1] != len(state["coef"]):
        raise SystemExit("built %d features but the model has %d coefficients"
                         % (Z.shape[1], len(state["coef"])))
    if not np.isfinite(Z).all():
        raise SystemExit("feature matrix contains non-finite values")

    np.save(out_path, np.ascontiguousarray(Z, dtype=np.float64))
    log("wrote %s with shape %s" % (out_path, Z.shape))
    pred = state["intercept"] + Z @ state["coef"]
    log("prediction summary: mean %.1f  sd %.1f  min %.1f  max %.1f mg/dL"
        % (pred.mean(), pred.std(), pred.min(), pred.max()))
    return 0


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    mode = sys.argv[1]
    if mode == "train":
        if len(sys.argv) != 5:
            sys.stderr.write("usage: part_d.py train <protocol> <train_dir> <model.pkl>\n")
            return 2
        return do_train(sys.argv[2], sys.argv[3], sys.argv[4])
    if mode == "feature_engineering":
        if len(sys.argv) != 6:
            sys.stderr.write("usage: part_d.py feature_engineering <protocol> "
                             "<test_dir> <model.pkl> <features.npy>\n")
            return 2
        return do_features(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    sys.stderr.write("unknown mode %r (expected 'train' or 'feature_engineering')\n" % mode)
    return 2


if __name__ == "__main__":
    sys.exit(main())
