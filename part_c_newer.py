#!/usr/bin/env python3
"""
COL774 Assignment 1 - Part (c): Feature Engineering for Heart-Rate Prediction.

    python3 part_c_newer.py train.csv test.csv predictions.txt

Approach
--------
The target is the Empatica E4 heart-rate value, which the device derives from the
BVP signal over a ~10 s span.  So the task is fundamentally *pulse-rate estimation*
from the 640-sample (64 Hz) BVP window, helped by motion/arousal context from the
accelerometer and EDA.

Empirically on this data:
  * the global peak of the BVP periodogram lands within 5 bpm of the true HR only
    ~40% of the time, but the true HR *is* one of the top-5 local maxima ~85% of
    the time (oracle NMAE 0.32).  The hard part is peak SELECTION, not detection;
  * windows that fail are the ones with LARGE BVP amplitude (the E4's BVP output
    is inflated by motion), not small ones.

So the pipeline is:
  1. Band-pass the BVP with cumulative-sum moving averages (0.5-4 Hz-ish).
  2. Build a library of log-domain "evidence surfaces" over a bpm grid:
     periodogram, Burg/Yule-Walker AR spectra, band-limited autocorrelation,
     multi-scale geometric-mean sub-window consensus (an AND operator that
     demands the rate be present throughout the window), harmonic look-ups,
     spectral whitening, the accelerometer spectrum and a population HR prior.
  3. Combine them with fixed, CV-tuned weights into one selection score S, and
     read off the rate by parabolic interpolation.  This single estimator alone
     roughly halves the error of the naive spectral peak.
  4. Turn everything into a representation a LINEAR model can exploit:
     normalised spectra / softmax(S) as distributions over bpm (so that
     w . p behaves like a learned soft-argmax), per-scalar spline (bump) bases
     (a GAM), and tensor-product bases of {rate estimate} x {confidence gate}
     so the model can learn "trust the pulse estimate when the evidence is
     sharp, otherwise fall back on the activity-conditioned prior".
  5. Fit a group-regularised ridge (separate penalty per feature block, chosen
     on an internal split via one eigendecomposition per configuration), then
     refit with IRLS towards an L1 loss because the metric is NMAE.

The final model is linear in the created features: yhat = w0 + w . z.

Libraries: numpy (no numpy.fft - all transforms are explicit cos/sin projections
written here), pandas for CSV reading, python stdlib.  No scipy, no sklearn,
no feature-extraction packages.
"""

import gc
import os
import sys
import time

import numpy as np
import pandas as pd

# Speed knob.  The full configuration is the accurate one; set PARTC_FAST=1 in
# the environment if the run has to fit a tighter wall-clock budget.  FAST
# trims the sub-window consensus hops, the spline resolution and the number of
# regularisation presets - it costs a little accuracy, nothing structural.
FAST = os.environ.get("PARTC_FAST", "0") == "1"

T0 = time.time()


def log(msg):
    sys.stderr.write("[%7.1fs] %s\n" % (time.time() - T0, msg))
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Layout of one example: 10 one-second blocks, each
#   acc_x[32] acc_y[32] acc_z[32] bvp[64] eda[4]   -> 164 values, 1640 total
# --------------------------------------------------------------------------
NBLK, BLK = 10, 164
IDX_BVP = np.concatenate([np.arange(BLK * s + 96, BLK * s + 160) for s in range(NBLK)])
IDX_ACC = [np.concatenate([np.arange(BLK * s + 32 * a, BLK * s + 32 * (a + 1))
                           for s in range(NBLK)]) for a in range(3)]
IDX_EDA = np.concatenate([np.arange(BLK * s + 160, BLK * s + 164) for s in range(NBLK)])

FS_B, FS_A = 64.0, 32.0
NB, NA, NE = 640, 320, 40

# candidate heart-rate grid (bpm).  0.5 bpm steps -> the DFT is evaluated at
# arbitrary frequencies, which is a zero-padding-free way of oversampling.
BG = np.arange(36.0, 216.0, 0.5)
NBG = len(BG)
BG32 = BG.astype(np.float32)
FG = BG / 60.0                       # Hz
COARSE = 4
NC = NBG // COARSE                   # 90 coarse bins for distributed features


def LG(x):
    return np.log(np.maximum(x, 1e-12)).astype(np.float32)


def _basis(nsamp, fs, grid_hz):
    """Hann-windowed cos/sin projection matrices -> a direct DFT at arbitrary
    frequencies (numpy.fft is not permitted, and we want a non-uniform grid)."""
    t = np.arange(nsamp) / fs
    w = np.hanning(nsamp)
    ang = 2.0 * np.pi * np.outer(t, grid_hz)
    return ((np.cos(ang) * w[:, None]).astype(np.float32),
            (np.sin(ang) * w[:, None]).astype(np.float32))


SUBLEN = (640, 320, 256, 160, 128, 80)
BAS = {L: _basis(L, FS_B, FG) for L in SUBLEN}
CA, SA = _basis(NA, FS_A, FG)
# cos(2 pi f * lag) with lag = one period of each candidate rate:
# multiplying a power spectrum by this gives the band-limited autocorrelation
# evaluated exactly at each candidate period (Wiener-Khinchin).
ACK = np.cos(2.0 * np.pi * FG[:, None] * (60.0 / BG)[None, :]).astype(np.float32)


# --------------------------------------------------------------------------
# small signal-processing helpers
# --------------------------------------------------------------------------
def movavg(X, w):
    """O(n) moving average along axis 1 via cumulative sums."""
    if w <= 1:
        return X
    pad = w // 2
    Xp = np.pad(X, ((0, 0), (pad, w - 1 - pad)), mode="edge")
    cs = np.cumsum(Xp, axis=1, dtype=np.float32)
    cs = np.concatenate([np.zeros((len(X), 1), np.float32), cs], 1)
    return (cs[:, w:] - cs[:, :-w]) * np.float32(1.0 / w)


def bandpass(X, w_hi, w_lo):
    """difference of two moving averages ~ band-pass.
    (5, 61) at 64 Hz keeps roughly 0.5-4 Hz, i.e. 30-240 bpm."""
    return movavg(X, w_hi) - movavg(X, w_lo)


def freq_smooth(M, w):
    pad = w // 2
    Mp = np.pad(M, ((0, 0), (pad, pad)), mode="edge")
    cs = np.cumsum(Mp, 1, dtype=np.float32)
    cs = np.concatenate([np.zeros((len(M), 1), np.float32), cs], 1)
    return (cs[:, w:] - cs[:, :-w]) * np.float32(1.0 / w)


def pgram(X, C, S):
    x = X - X.mean(1, keepdims=True)
    return (x @ C) ** 2 + (x @ S) ** 2


def nrm(P):
    return P / (P.sum(1, keepdims=True) + np.float32(1e-20))


def coarse(P):
    return P.reshape(len(P), NC, COARSE).sum(2)


def softmax_rows(S, tau):
    z = (S - S.max(1, keepdims=True)) * np.float32(1.0 / tau)
    e = np.exp(z, dtype=np.float32)
    return e / (e.sum(1, keepdims=True) + np.float32(1e-20))


def ar_spectrum(X, order, ridge=1e-3):
    """Yule-Walker autoregressive spectral estimate.  Gives much sharper peaks
    than the periodogram on a short (10 s) window."""
    x = X - X.mean(1, keepdims=True)
    x = x / (x.std(1, keepdims=True) + np.float32(1e-6))
    n, L = x.shape
    rr = np.empty((n, order + 1), np.float32)
    for k in range(order + 1):
        rr[:, k] = np.einsum("ij,ij->i", x[:, :L - k], x[:, k:]) / L
    ii = np.abs(np.arange(order)[:, None] - np.arange(order)[None, :])
    Rm = rr[:, ii].copy()                                   # Toeplitz, batched
    Rm[:, np.arange(order), np.arange(order)] *= np.float32(1.0 + ridge)
    a = np.linalg.solve(Rm, rr[:, 1:order + 1][..., None])[..., 0]
    sig = np.maximum(rr[:, 0] - (a * rr[:, 1:order + 1]).sum(1), 1e-6)
    k = np.arange(1, order + 1)
    ang = 2.0 * np.pi * np.outer(FG / FS_B, k)
    Cc = np.ascontiguousarray(np.cos(ang).T.astype(np.float32))
    Ss = np.ascontiguousarray(np.sin(ang).T.astype(np.float32))
    return (sig[:, None] / ((1 - a @ Cc) ** 2 + (a @ Ss) ** 2 + 1e-9)).astype(np.float32)


def parab_peak(M):
    """argmax with parabolic (sub-bin) refinement."""
    i = np.argmax(M, 1)
    r = np.arange(len(M))
    ic = np.clip(i, 1, M.shape[1] - 2)
    y0, y1, y2 = M[r, ic - 1], M[r, ic], M[r, ic + 1]
    d = y0 - 2 * y1 + y2
    ok = np.abs(d) > 1e-20
    off = np.clip(np.where(ok, 0.5 * (y0 - y2) / np.where(ok, d, 1.0), 0.0), -1, 1)
    return (BG32[ic] + off * np.float32(BG[1] - BG[0])).astype(np.float32), M[r, i].astype(np.float32)


def topk_peaks(M, k=3, guard=10.0):
    """k strongest local maxima, each `guard` bpm away from the previous ones."""
    Mw = M.copy()
    vals, scores = [], []
    for _ in range(k):
        pk, sv = parab_peak(Mw)
        vals.append(pk)
        scores.append(sv)
        Mw = np.where(np.abs(BG32[None, :] - pk[:, None]) < np.float32(guard),
                      np.float32(-1e30), Mw)
    return np.stack(vals, 1), np.stack(scores, 1)


def interp_at(M, targets):
    """columns of M looked up at a fixed set of bpm targets (shared by all rows)."""
    idx = np.clip((np.asarray(targets) - BG[0]) / (BG[1] - BG[0]), 0, NBG - 1.001)
    lo = idx.astype(np.int32)
    w = (idx - lo).astype(np.float32)
    return M[:, lo] * (1 - w) + M[:, lo + 1] * w


def interp_row(M, targets):
    """per-row lookup: value of M[i] at bpm targets[i]."""
    idx = np.clip((np.asarray(targets) - BG[0]) / (BG[1] - BG[0]), 0, NBG - 1.001)
    lo = idx.astype(np.int32)
    w = (idx - lo).astype(np.float32)
    r = np.arange(len(M))
    return (M[r, lo] * (1 - w) + M[r, lo + 1] * w).astype(np.float32)


def bumps(v, centers, width):
    """normalised Gaussian bump basis: sum_j w_j b_j(v) is a smooth lookup
    table over v, so a linear model can learn an arbitrary 1-D response."""
    z = (np.asarray(v, np.float32)[:, None] -
         np.asarray(centers, np.float32)[None, :]) * np.float32(1.0 / max(width, 1e-6))
    b = np.exp(-0.5 * z * z)
    return (b / (b.sum(1, keepdims=True) + np.float32(1e-9))).astype(np.float32)


# --------------------------------------------------------------------------
# the evidence-surface library
# --------------------------------------------------------------------------
# sub-window consensus configurations: (window length, hop, name).
# The geometric mean of normalised sub-window spectra (== mean of logs) is a
# soft AND: a rate only scores well if it is present in *every* part of the
# window.  This turned out to be by far the strongest single cue.
CONS_CFG = ((320, 64, "g5"), (256, 48, "g4"), (160, 32, "g25"), (128, 32, "g2"), (80, 16, "g125"))
if FAST:
    CONS_CFG = ((320, 64, "g5"), (256, 64, "g4"), (160, 48, "g25"), (128, 48, "g2"), (80, 32, "g125"))

# weights found by coordinate search on training data only (holdout-verified:
# tuning-set MAE 10.90 vs holdout MAE 10.90, i.e. these 19 numbers do not overfit).
SCORE_W = {
    "lP": 0.25, "lPwhite": 0.5, "lAR": -0.25, "lAR2": 4.0, "lR": 0.5, "lRaw": -0.5,
    "g5": 0.5, "g4": -0.5, "g25": 1.0, "g2": 4.0, "g125": 4.0, "g25w": -2.0,
    "prior": 4.0, "lP2": 0.5, "lP3": 0.25, "lPh": 0.25, "lPh3": 0.5,
    "g5_2": -0.5, "g5_h": 1.0,
}
# a deliberately different, consensus-only score and a classical spectrum-only
# score; giving the linear model several *independent* selection rules helps it
# hedge, since argmax itself cannot be learned by a linear model.
SCORE_W2 = {"g2": 1.0, "g25": 1.0, "g5": 1.0, "lP": 0.5, "prior": 0.5}
SCORE_W3 = {"lP": 1.0, "lAR": 1.0, "lR": 0.5}


def build_terms(bvp, acc, lprior):
    B = np.ascontiguousarray(bvp, np.float32)
    Bbp = bandpass(B, 5, 61)
    T = {}

    P = nrm(pgram(Bbp, *BAS[640]))
    T["lP"] = LG(P)
    T["lPwhite"] = T["lP"] - freq_smooth(T["lP"], 61)      # remove the 1/f trend
    T["lAR"] = LG(nrm(ar_spectrum(Bbp, 32)))
    T["lAR2"] = LG(nrm(ar_spectrum(Bbp, 20)))
    R = P @ ACK                                            # autocorr at each period
    T["lR"] = LG(nrm(np.maximum(R - R.min(1, keepdims=True), 1e-9)))
    T["lRaw"] = LG(nrm(pgram(B, *BAS[640])))               # unfiltered, for contrast

    A = np.ascontiguousarray(acc, np.float32)
    PA = nrm(pgram(A[:, 0], CA, SA) + pgram(A[:, 1], CA, SA) + pgram(A[:, 2], CA, SA))
    T["lPA"] = LG(PA)
    T["prior"] = np.repeat(lprior, len(B), 0)

    for L, hop, nm in CONS_CFG:                            # accumulate, never stack
        C, S = BAS[L]
        acc_log = None
        k = 0
        for st in range(0, Bbp.shape[1] - L + 1, hop):
            l = LG(nrm(pgram(Bbp[:, st:st + L], C, S)))
            acc_log = l if acc_log is None else acc_log + l
            k += 1
        T[nm] = acc_log * np.float32(1.0 / k)
    T["g25w"] = T["g25"] - freq_smooth(T["g25"], 61)

    # harmonic look-ups: is there energy at 2f / 3f (supports f as fundamental)
    # or at f/2 / f/3 (suggests f is itself a harmonic)?
    T["lP2"] = interp_at(T["lP"], BG * 2)
    T["lP3"] = interp_at(T["lP"], BG * 3)
    T["lPh"] = interp_at(T["lP"], BG / 2)
    T["lPh3"] = interp_at(T["lP"], BG / 3)
    T["g5_2"] = interp_at(T["g5"], BG * 2)
    T["g5_h"] = interp_at(T["g5"], BG / 2)
    return T, Bbp, P, PA


def combine(T, w):
    S = None
    for k, v in w.items():
        if v and k in T:
            S = np.float32(v) * T[k] if S is None else S + np.float32(v) * T[k]
    S = S - S.mean(1, keepdims=True)
    return S / (S.std(1, keepdims=True) + np.float32(1e-6))


# --------------------------------------------------------------------------
# feature assembly
# --------------------------------------------------------------------------
TAUS = (0.12, 0.25, 0.5, 1.0)


def surf_desc(M, tag, out, names):
    """peak / shape descriptors of one evidence surface"""
    pk, _ = parab_peak(M)
    Pn = softmax_rows(M, 1.0)
    cen = Pn @ BG32
    cs = np.cumsum(Pn, 1)
    med = BG32[np.argmax(cs >= 0.5, 1)]
    q25 = BG32[np.argmax(cs >= 0.25, 1)]
    q75 = BG32[np.argmax(cs >= 0.75, 1)]
    ent = -(Pn * np.log(Pn + 1e-12)).sum(1)
    sd = np.sqrt(np.maximum(Pn @ (BG32 ** 2) - cen ** 2, 0))
    tv, ts = topk_peaks(M, 3, 10.0)
    for nm, v in (("pk", pk), ("p2", tv[:, 1]), ("p3", tv[:, 2]),
                  ("m12", ts[:, 0] - ts[:, 1]), ("m13", ts[:, 0] - ts[:, 2]),
                  ("d12", tv[:, 1] - pk), ("d13", tv[:, 2] - pk),
                  ("cen", cen), ("med", med), ("q25", q25), ("q75", q75),
                  ("iqr", q75 - q25), ("ent", ent), ("sd", sd)):
        out.append(np.asarray(v, np.float32)[:, None])
        names.append(tag + "_" + nm)
    return pk, ts, ent


def featurize(bvp, acc, eda, lprior):
    out, names, reps = [], [], {}
    n = len(bvp)
    T, Bbp, P, PA = build_terms(bvp, acc, lprior)

    # ---- the three selection scores ------------------------------------
    S = combine(T, SCORE_W)
    sel, ts, sent = surf_desc(S, "S", out, names)
    for i, tau in enumerate(TAUS):
        reps["sm%d" % i] = coarse(softmax_rows(S, tau))
    S2 = combine(T, SCORE_W2)
    sel2, _, _ = surf_desc(S2, "S2", out, names)
    reps["sm_c"] = coarse(softmax_rows(S2, 0.25))
    S3 = combine(T, SCORE_W3)
    sel3, _, _ = surf_desc(S3, "S3", out, names)
    del S, S2, S3

    # ---- individual surfaces -------------------------------------------
    pk_P, _, _ = surf_desc(T["lP"], "P", out, names)
    pk_AR, _, _ = surf_desc(T["lAR"], "AR", out, names)
    pk_g2, _, _ = surf_desc(T["g2"], "G2", out, names)
    pk_g25, _, _ = surf_desc(T["g25"], "G25", out, names)
    pk_g5, _, _ = surf_desc(T["g5"], "G5", out, names)
    pk_R, _, _ = surf_desc(T["lR"], "R", out, names)
    surf_desc(T["lPA"], "AC", out, names)
    reps["specP"] = coarse(P)
    reps["specS"] = coarse(nrm(np.sqrt(P)))
    reps["ar"] = coarse(softmax_rows(T["lAR"], 1.0))
    reps["g25"] = coarse(softmax_rows(T["g25"], 1.0))
    reps["accs"] = coarse(PA)

    # ---- agreement between the independent estimators -------------------
    ests = np.stack([sel, sel2, sel3, pk_P, pk_AR, pk_g2, pk_g25, pk_g5, pk_R], 1)
    out.append(ests.std(1, keepdims=True)); names.append("x_std")
    out.append(np.median(ests, 1, keepdims=True).astype(np.float32)); names.append("x_med")
    out.append(ests.mean(1, keepdims=True)); names.append("x_mean")
    for j, nm in enumerate(("P", "AR", "g2", "g25", "g5", "R")):
        out.append(np.abs(ests[:, 3 + j] - sel)[:, None]); names.append("x_dev_" + nm)
    xagree = (np.abs(ests - sel[:, None]) < 4).mean(1).astype(np.float32)
    out.append(xagree[:, None]); names.append("x_agree")
    for mlt, nm in ((1.0, "f"), (2.0, "2f"), (0.5, "hf"), (3.0, "3f")):
        out.append(interp_row(T["lP"], sel * mlt)[:, None]); names.append("sp_" + nm)
        out.append(interp_row(T["g2"], sel * mlt)[:, None]); names.append("sg_" + nm)
    lprom = interp_row(T["lP"], sel).astype(np.float32)
    del T

    # ---- time-domain BVP ------------------------------------------------
    B = np.ascontiguousarray(bvp, np.float32)
    Bwd = bandpass(B, 9, 81)
    sdb = Bbp.std(1) + np.float32(1e-6)
    z = Bbp / sdb[:, None]
    d1 = np.diff(Bbp, axis=1)
    seg = Bbp.reshape(n, 10, 64).std(2)          # per-second amplitude profile
    for nm, v in (("zc", (np.diff(np.signbit(Bbp), axis=1).sum(1) / 2.0) * 6.0),
                  ("zc2", (np.diff(np.signbit(Bwd), axis=1).sum(1) / 2.0) * 6.0),
                  ("lamp", np.log(sdb)), ("lraw", np.log(B.std(1) + 1e-6)),
                  ("lmad", np.log(np.abs(Bbp).mean(1) + 1e-6)),
                  ("kurt", (z ** 4).mean(1)), ("skew", (z ** 3).mean(1)),
                  ("ldiff", np.log(np.abs(d1).mean(1) + 1e-6)),
                  ("crest", np.abs(Bbp).max(1) / sdb),
                  ("lrange", np.log(B.max(1) - B.min(1) + 1e-6)),
                  ("nfrac", (np.abs(z) > 3).mean(1)),
                  ("envcv", seg.std(1) / sdb),
                  ("envmax", np.log(seg.max(1) + 1e-6)),
                  ("envmin", np.log(seg.min(1) + 1e-6))):
        out.append(np.asarray(v, np.float32)[:, None]); names.append("t_" + nm)
    lamp = np.log(sdb).astype(np.float32)

    # ---- accelerometer (activity level is a strong HR prior) ------------
    A = np.ascontiguousarray(acc, np.float32)
    mag = np.sqrt((A ** 2).sum(1))
    magbp = bandpass(mag, 3, 41)
    asd = A.std(2)
    amean = A.mean(2)
    absum = np.abs(amean).sum(1) + np.float32(1e-3)
    for nm, v in (("magm", mag.mean(1)), ("magsd", mag.std(1)),
                  ("lmagsd", np.log(mag.std(1) + 1e-3)),
                  ("lbpsd", np.log(magbp.std(1) + 1e-3)),
                  ("sdx", asd[:, 0]), ("sdy", asd[:, 1]), ("sdz", asd[:, 2]),
                  ("lsd", np.log(asd.sum(1) + 1e-3)),
                  ("mx", amean[:, 0]), ("my", amean[:, 1]), ("mz", amean[:, 2]),
                  ("nx", amean[:, 0] / absum), ("ny", amean[:, 1] / absum),
                  ("nz", amean[:, 2] / absum),
                  ("magrange", mag.max(1) - mag.min(1)),
                  ("jerk", np.log(np.abs(np.diff(mag, axis=1)).mean(1) + 1e-3)),
                  ("q90", np.percentile(np.abs(magbp), 90, axis=1)),
                  ("still", (np.abs(magbp) < 2).mean(1))):
        out.append(np.asarray(v, np.float32)[:, None]); names.append("a_" + nm)
    lacc = np.log(asd.sum(1) + 1e-3).astype(np.float32)
    env = movavg(np.abs(Bbp), 33)[:, ::2][:, :NA]
    ec = env - env.mean(1, keepdims=True)
    mc = magbp - magbp.mean(1, keepdims=True)
    out.append(((ec * mc).sum(1) /
                (np.sqrt((ec ** 2).sum(1) * (mc ** 2).sum(1)) + 1e-6)).astype(np.float32)[:, None])
    names.append("x_envacc")

    # ---- EDA -------------------------------------------------------------
    E = np.ascontiguousarray(eda, np.float32)
    tt = np.arange(E.shape[1], dtype=np.float32)
    tt = (tt - tt.mean()) / tt.std()
    for nm, v in (("lmean", np.log(E.mean(1) + 1e-3)), ("lstd", np.log(E.std(1) + 1e-4)),
                  ("slope", (E * tt).mean(1)), ("zero", (E.max(1) <= 1e-6).astype(np.float32)),
                  ("lrng", np.log(E.max(1) - E.min(1) + 1e-4))):
        out.append(np.asarray(v, np.float32)[:, None]); names.append("e_" + nm)

    scal = np.hstack(out).astype(np.float32)
    keys = {"sel": sel, "sel2": sel2, "sel3": sel3, "pk_g2": pk_g2, "pk_P": pk_P,
            "pk_AR": pk_AR, "pk_g25": pk_g25, "pk_g5": pk_g5,
            "margin": (ts[:, 0] - ts[:, 1]).astype(np.float32),
            "sent": sent.astype(np.float32), "lamp": lamp, "lacc": lacc,
            "xstd": ests.std(1).astype(np.float32), "xagree": xagree, "lprom": lprom}
    return scal, names, reps, keys


def featurize_chunked(bvp, acc, eda, lprior, chunk=8192, tag=""):
    S, R, K, nm = [], None, None, None
    n = len(bvp)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        s, names, r, k = featurize(bvp[i:j], acc[i:j], eda[i:j], lprior)
        S.append(s)
        if R is None:
            nm = names
            R = {a: [b] for a, b in r.items()}
            K = {a: [b] for a, b in k.items()}
        else:
            for a, b in r.items():
                R[a].append(b)
            for a, b in k.items():
                K[a].append(b)
        del s, r, k
        gc.collect()
        log("  %s features %d/%d" % (tag, j, n))
    return (np.concatenate(S), nm,
            {a: np.concatenate(b) for a, b in R.items()},
            {a: np.concatenate(b) for a, b in K.items()})


# --------------------------------------------------------------------------
# non-linear expansions that keep the MODEL linear
# --------------------------------------------------------------------------
NKN = 7 if FAST else 9
CT = np.arange(40.0, 200.1, 8.0 if FAST else 6.0)
GATES = ("lamp", "xagree", "margin") if FAST else ("lamp", "xagree", "margin", "lacc", "sent")
PAIRS = (("lacc", "lamp"), ("lacc", "sent"), ("lamp", "xagree"))


def fit_gam_knots(Str):
    kn = [np.percentile(Str[:, j], np.linspace(2, 98, NKN)) for j in range(Str.shape[1])]
    wd = [max((k[-1] - k[0]) / (NKN - 1), 1e-6) for k in kn]
    return kn, wd


def gam(X, kn, wd):
    """one spline basis per scalar -> ridge becomes a generalised additive model."""
    return np.hstack([bumps(X[:, j], kn[j], wd[j]) for j in range(X.shape[1])]).astype(np.float32)


def fit_gate_knots(K):
    g = {q: np.percentile(K[q], np.linspace(5, 95, 5)) for q in GATES}
    p = {a: np.percentile(K[a], np.linspace(4, 96, 7)) for pr in PAIRS for a in pr}
    return g, p


def tensor(K, gk, pk_):
    """{rate estimate} x {confidence gate} tensor bases.  These let one linear
    model behave like a gated mixture: trust the pulse estimate where the
    evidence is sharp, otherwise lean on the activity-conditioned prior."""
    cols = []
    for e in ("sel", "sel2", "pk_g2"):
        be = bumps(K[e], CT, 6.0)
        for q in GATES:
            kq = gk[q]
            bq = bumps(K[q], kq, max((kq[-1] - kq[0]) / 4.0, 1e-6))
            cols.append((be[:, :, None] * bq[:, None, :]).reshape(len(be), -1))
    for a, b in PAIRS:
        ka, kb = pk_[a], pk_[b]
        ba = bumps(K[a], ka, max((ka[-1] - ka[0]) / 6.0, 1e-6))
        bb = bumps(K[b], kb, max((kb[-1] - kb[0]) / 6.0, 1e-6))
        cols.append((ba[:, :, None] * bb[:, None, :]).reshape(len(ba), -1))
    return np.hstack(cols).astype(np.float32)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def read_csv(path, chunksize=20000):
    """stream the CSV and keep only the per-modality arrays (same footprint as
    the raw matrix, but laid out for the transforms)."""
    bv, ac, ed, ys = [], [], [], []
    tot = 0
    for ch in pd.read_csv(path, chunksize=chunksize, dtype=np.float32):
        y = ch["hr"].to_numpy(np.float32) if "hr" in ch.columns else None
        if y is not None:
            ch = ch.drop(columns=["hr"])
        X = ch.to_numpy(np.float32)
        del ch
        if X.shape[1] != NBLK * BLK:
            raise ValueError("expected %d feature columns, got %d" % (NBLK * BLK, X.shape[1]))
        bv.append(X[:, IDX_BVP])
        ac.append(np.stack([X[:, IDX_ACC[0]], X[:, IDX_ACC[1]], X[:, IDX_ACC[2]]], 1))
        ed.append(X[:, IDX_EDA])
        if y is not None:
            ys.append(y)
        tot += len(X)
        del X
        gc.collect()
    log("read %s: %d rows" % (path, tot))
    return (np.concatenate(bv), np.concatenate(ac), np.concatenate(ed),
            np.concatenate(ys).astype(np.float64) if ys else None)


# --------------------------------------------------------------------------
# group-regularised ridge, fitted towards an L1 loss (the metric is NMAE)
# --------------------------------------------------------------------------
ALPHAS = 10.0 ** np.arange(-0.5, 6.01, 0.25)
# Relative penalty multipliers per block: (scalars, gam, tensor, spectra).
# Each costs one p x p eigendecomposition, so keep the list short.  The last
# entry all but switches the expansions off, so the preset search can always
# fall back to the plain-scalar model that is known to score ~0.626.
PRESETS = ((0.1, 1.0, 10.0, 10.0),
           (1.0, 1.0, 1.0, 1.0),
           (0.03, 0.3, 3.0, 30.0),
           (0.01, 30.0, 100.0, 100.0))
if FAST:
    PRESETS = PRESETS[:2]


def nmae(y, p):
    return np.abs(y - p).sum() / np.abs(y - y.mean()).sum()


def solve_path(G, b, scale, Xva, yva, ym):
    """one eigendecomposition gives the whole alpha path for free."""
    d, V = np.linalg.eigh(G * scale[None, :] * scale[:, None])
    bt = V.T @ (b * scale)
    Pv = (Xva * scale[None, :].astype(np.float32)) @ V.astype(np.float32)
    best = None
    for a in ALPHAS:
        w = bt / (d + a)
        pred = Pv @ w.astype(np.float32) + ym
        s = nmae(yva, pred)
        if best is None or s < best[0]:
            best = (s, a, (V @ w) * scale)
    return best


def fit_model(Xtr, ytr, gid, nblocks):
    n = len(Xtr)
    cut = int(n * 0.75)
    ym = ytr[:cut].mean()
    log("  gram (internal split)")
    A = Xtr[:cut]
    G = (A.T @ A).astype(np.float64)
    b = (A.T @ (ytr[:cut] - ym).astype(np.float32)).astype(np.float64)
    del A
    gc.collect()
    Xva, yva = Xtr[cut:], ytr[cut:]
    best = None
    for pi, pre in enumerate(PRESETS):
        mult = np.asarray(pre, np.float64)[:nblocks]
        scale = (1.0 / np.sqrt(mult))[gid]              # penalty m <-> scale 1/sqrt(m)
        s, a, w = solve_path(G, b, scale, Xva, yva, ym)
        log("    preset %d  internal NMAE %.4f (alpha=%.3g)" % (pi, s, a))
        if best is None or s < best[0]:
            best = (s, a, pre)
    del G, b, Xva
    gc.collect()
    log("  best preset %s alpha %.3g internal NMAE %.4f" % (best[2], best[1], best[0]))

    mult = np.asarray(best[2], np.float64)[:nblocks]
    scale = (1.0 / np.sqrt(mult))[gid]
    alpha = best[1]
    ymf = ytr.mean()
    yc = (ytr - ymf).astype(np.float32)
    Xs = (Xtr * scale[None, :].astype(np.float32))
    G = (Xs.T @ Xs).astype(np.float64)
    bb = (Xs.T @ yc).astype(np.float64)
    p = G.shape[0]
    w = np.linalg.solve(G + alpha * np.eye(p), bb)
    log("  L2 fit done")

    # IRLS towards L1: reweight by 1/|residual| (Huber-floored) a few times.
    wbest, sbest = w, nmae(ytr[cut:], Xs[cut:] @ w.astype(np.float32) + ymf)
    for it in range(3):
        r = Xs @ w.astype(np.float32) - yc
        floor = np.float32(1.345 * np.median(np.abs(r - np.median(r))) + 1e-3)
        sw = np.sqrt(1.0 / np.maximum(np.abs(r), floor)).astype(np.float32)
        Xw = Xs * sw[:, None]
        Gw = (Xw.T @ Xw).astype(np.float64)
        bw = (Xw.T @ (yc * sw)).astype(np.float64)
        del Xw
        gc.collect()
        w = np.linalg.solve(Gw + alpha * float((sw ** 2).mean()) * np.eye(p), bw)
        s = nmae(ytr[cut:], Xs[cut:] @ w.astype(np.float32) + ymf)
        log("    irls %d internal NMAE %.4f" % (it, s))
        if s < sbest:
            sbest, wbest = s, w
        del Gw, bw
        gc.collect()
    del Xs
    gc.collect()
    return wbest * scale, ymf


# --------------------------------------------------------------------------
def main():
    if len(sys.argv) != 4:
        sys.stderr.write("usage: part_c_newer.py train.csv test.csv predictions.txt\n")
        return 2
    train_path, test_path, out_path = sys.argv[1:4]

    bvp, acc, eda, ytr = read_csv(train_path)
    if ytr is None:
        raise ValueError("train.csv must contain an 'hr' column")

    # population HR prior, smoothed, in log domain (a term in the selection score)
    hist, _ = np.histogram(ytr, bins=np.append(BG - 0.25, BG[-1] + 0.25), density=True)
    k = np.hanning(21)
    pri = np.convolve(hist, k / k.sum(), mode="same") + 1e-6
    lprior = LG(pri / pri.sum())[None, :].astype(np.float32)

    Str, names, Rtr, Ktr = featurize_chunked(bvp, acc, eda, lprior, tag="train")
    del bvp, acc, eda
    gc.collect()
    log("train scalars %s" % (Str.shape,))

    bvp, acc, eda, _ = read_csv(test_path)
    ntest = len(bvp)
    Ste, _, Rte, Kte = featurize_chunked(bvp, acc, eda, lprior, tag="test")
    del bvp, acc, eda
    gc.collect()

    kn, wd = fit_gam_knots(Str)
    gk, pk_ = fit_gate_knots(Ktr)
    blocks = list(Rtr.keys())

    def design(S, R, K):
        return [S, gam(S, kn, wd), tensor(K, gk, pk_),
                np.hstack([R[b] for b in blocks]).astype(np.float32)]

    parts_tr = design(Str, Rtr, Ktr)
    del Str, Rtr, Ktr
    gc.collect()
    sizes = [p.shape[1] for p in parts_tr]
    gid = np.concatenate([[i] * s for i, s in enumerate(sizes)])
    Xtr = np.hstack(parts_tr)
    del parts_tr
    gc.collect()
    log("design matrix %s  block sizes %s" % (Xtr.shape, sizes))

    mu = Xtr.mean(0)
    sd = Xtr.std(0) + np.float32(1e-8)
    Xtr = ((Xtr - mu) / sd).astype(np.float32)

    w, ym = fit_model(Xtr, ytr, gid, len(sizes))
    del Xtr
    gc.collect()

    parts_te = design(Ste, Rte, Kte)
    del Ste, Rte, Kte
    gc.collect()
    Xte = np.hstack(parts_te)
    del parts_te
    gc.collect()
    Xte = ((Xte - mu) / sd).astype(np.float32)
    pred = (Xte @ w.astype(np.float32) + ym).astype(np.float64)
    del Xte
    gc.collect()

    # keep predictions physiologically sane and always finite
    lo, hi = ytr.min() - 15.0, ytr.max() + 15.0
    pred = np.clip(np.nan_to_num(pred, nan=ym, posinf=hi, neginf=lo), max(lo, 25.0), min(hi, 230.0))
    assert len(pred) == ntest, "prediction count mismatch"
    with open(out_path, "w") as f:
        f.write("\n".join("%.6f" % v for v in pred))
        f.write("\n")
    log("wrote %d predictions to %s" % (ntest, out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
