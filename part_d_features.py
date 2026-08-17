#!/usr/bin/env python3
"""
COL774 A1 Part (d) - feature extraction for CGM glucose estimation.

One 5-minute window carries eight raw signals; this module turns each window
into a fixed, named, finite feature vector.  It is imported by part_d.py.

Design notes
------------
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
"""

import numpy as np

try:
    from scipy import signal as sps
    HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    HAVE_SCIPY = False

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
        f = np.fft.rfftfreq(L, 1.0 / fs) if False else np.linspace(0, fs / 2, L // 2 + 1)
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
            fr = np.fft.rfftfreq(len(xi), 0.25) if hasattr(np.fft, "rfftfreq") else None
            # explicit DFT keeps this independent of numpy.fft availability rules
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


def features_for_file(path, verbose=False):
    """Extract features for one participant .npz, one modality at a time."""
    import gc
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
            F, N, ecg_peaks = feat_ecg(a); blocks.append(F); names += N
            del a; gc.collect()
        a = take("e4_bvp")
        if a is not None:
            F, N, ppg_peaks = feat_bvp(a); blocks.append(F); names += N
            del a; gc.collect()
        for key, fn in (("e4_hr", feat_hr1), ("e4_eda", feat_eda),
                        ("e4_temp", feat_temp), ("zephyr_breathing", feat_resp)):
            a = take(key)
            if a is not None:
                F, N = fn(a); blocks.append(F); names += N
                del a; gc.collect()
        for key, tag in (("e4_acc", "e4acc"), ("zephyr_acc", "zacc")):
            a = take(key)
            if a is not None:
                F, N = feat_acc(a, tag); blocks.append(F); names += N
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
