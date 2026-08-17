#!/usr/bin/env python3
"""
Synthetic PhysioCGM-shaped data for Part (d).

The real dataset is not available locally, so this generator fabricates .npz
files with EXACTLY the documented schema, sampling rates and array shapes, and
with a deliberately learnable (but noisy) glucose relationship.  It exists so
part_d_newer.py can be exercised end to end - shapes, NaN handling, the pickle
contract, train/test column alignment and eval_d.py compatibility - before the
real data is ever touched.

It is a TEST FIXTURE, not part of the submission, and the numbers it produces
say nothing about real accuracy.

    python3 make_synthetic_d.py outdir [--windows 40] [--subjects 6]

Creates outdir/{d1,d2,d3}/{train,test}/*.npz
"""

import argparse
import os

import numpy as np

FS = {"e4_bvp": 64, "e4_hr": 1, "e4_eda": 4, "e4_temp": 4, "e4_acc": 32,
      "zephyr_ecg": 250, "zephyr_acc": 100, "zephyr_breathing": 25}
WIN = 300


def ecg_beat(fs):
    """one crude PQRST complex, ~0.8 s long."""
    t = np.arange(int(0.8 * fs)) / fs

    def g(mu, sd, amp):
        return amp * np.exp(-0.5 * ((t - mu) / sd) ** 2)
    return (g(0.16, 0.025, 0.12) - g(0.23, 0.010, 0.16) + g(0.25, 0.010, 1.0)
            - g(0.28, 0.012, 0.25) + g(0.42, 0.045, 0.30))


def ppg_beat(fs):
    t = np.arange(int(0.8 * fs)) / fs
    return (np.exp(-0.5 * ((t - 0.20) / 0.055) ** 2)
            + 0.42 * np.exp(-0.5 * ((t - 0.36) / 0.075) ** 2))


def beat_train(n_samples, fs, hr_bpm, rr_sd, tmpl, rng, jitter_amp=0.12):
    x = np.zeros(n_samples + len(tmpl) + 8)
    t = 0.0
    period = 60.0 / hr_bpm
    peaks = []
    while True:
        rr = period * (1.0 + rr_sd * rng.randn())
        rr = float(np.clip(rr, 0.33, 2.0))
        t += rr
        i = int(t * fs)
        if i + len(tmpl) >= len(x):
            break
        x[i:i + len(tmpl)] += tmpl * (1.0 + jitter_amp * rng.randn())
        peaks.append(i)
    return x[:n_samples], np.asarray(peaks)


def make_subject(sid, n_win, rng, base_glucose):
    """One participant.  Glucose is driven by HR, HRV, skin temperature and
    activity, plus a strong per-subject offset - which is exactly why d3
    (cross-subject) is hard and d1/d2 are not."""
    out = {}
    hr = np.clip(62 + 14 * rng.randn(n_win), 45, 130)
    rr_sd = np.clip(0.055 + 0.022 * rng.randn(n_win), 0.008, 0.16)
    temp = np.clip(32.5 + 1.6 * rng.randn(n_win), 27, 37)
    act = np.abs(0.35 * rng.randn(n_win))
    eda_lv = np.abs(1.2 + 0.9 * rng.randn(n_win))

    glucose = (base_glucose
               + 1.35 * (hr - 62)
               - 320.0 * (rr_sd - 0.055)
               + 6.5 * (temp - 32.5)
               - 44.0 * act
               + 5.0 * eda_lv
               + 17.0 * rng.randn(n_win))
    glucose = np.clip(glucose, 40, 400)

    n = n_win
    ecg_t, ppg_t = ecg_beat(FS["zephyr_ecg"]), ppg_beat(FS["e4_bvp"])
    L_ecg, L_bvp = FS["zephyr_ecg"] * WIN, FS["e4_bvp"] * WIN

    ecg = np.zeros((n, L_ecg), np.float32)
    bvp = np.zeros((n, L_bvp), np.float32)
    for i in range(n):
        e, _ = beat_train(L_ecg, FS["zephyr_ecg"], hr[i], rr_sd[i], ecg_t, rng)
        e += 0.05 * rng.randn(L_ecg) + 0.25 * np.sin(2 * np.pi * 0.25 *
                                                     np.arange(L_ecg) / FS["zephyr_ecg"])
        ecg[i] = e
        p, _ = beat_train(L_bvp, FS["e4_bvp"], hr[i], rr_sd[i], ppg_t, rng)
        p = np.roll(p, int(0.22 * FS["e4_bvp"]))          # pulse arrival delay
        bvp[i] = 90 * (p - p.mean()) + 8 * rng.randn(L_bvp)
    out["zephyr_ecg"] = ecg
    out["e4_bvp"] = bvp

    out["e4_hr"] = (hr[:, None] + 3 * rng.randn(n, FS["e4_hr"] * WIN)).astype(np.float32)
    tt = np.arange(FS["e4_eda"] * WIN) / (FS["e4_eda"] * WIN)
    out["e4_eda"] = (eda_lv[:, None] * (1 + 0.25 * np.sin(6 * np.pi * tt))
                     + 0.05 * rng.randn(n, FS["e4_eda"] * WIN)).astype(np.float32)
    out["e4_temp"] = (temp[:, None] + 0.12 * rng.randn(n, FS["e4_temp"] * WIN)
                      + 0.3 * tt).astype(np.float32)
    out["zephyr_breathing"] = (np.sin(2 * np.pi * (0.22 + 0.05 * rng.rand(n))[:, None]
                                      * np.arange(FS["zephyr_breathing"] * WIN)
                                      / FS["zephyr_breathing"])
                               + 0.15 * rng.randn(n, FS["zephyr_breathing"] * WIN)).astype(np.float32)
    for key, fs in (("e4_acc", FS["e4_acc"]), ("zephyr_acc", FS["zephyr_acc"])):
        a = act[:, None, None] * rng.randn(n, fs * WIN, 3)
        a[:, :, 2] += 1.0                                  # gravity on z
        out[key] = a.astype(np.float32)

    # realistic missingness: the E4 comes off the wrist
    for key in ("e4_bvp", "e4_eda", "e4_temp"):
        arr = out[key]
        for i in range(n):
            if rng.rand() < 0.10:
                a = rng.randint(0, arr.shape[1] // 2)
                arr[i, a:a + arr.shape[1] // 3] = np.nan

    out["subject_id"] = np.array([sid] * n)
    out["glucose"] = glucose.astype(np.float64)
    return out, glucose


def write(path, d, ts, with_glucose=True):
    payload = {k: v for k, v in d.items() if k != "glucose"}
    payload["timestamp"] = ts
    if with_glucose:
        payload["glucose"] = d["glucose"]
    np.savez(path, **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--subjects", type=int, default=6)
    args = ap.parse_args()
    rng = np.random.RandomState(7)

    subjects = []
    for s in range(args.subjects):
        base = 110 + 35 * rng.rand()
        d, g = make_subject("S%02d" % s, args.windows, rng, base)
        ts = np.arange(args.windows, dtype=np.float64) * 300.0 + s * 1e6
        subjects.append(("S%02d" % s, d, ts))
        print("  subject S%02d  glucose mean %.0f sd %.0f" % (s, g.mean(), g.std()))

    def sub(d, idx):
        return {k: (v[idx] if hasattr(v, "__len__") and len(v) == args.windows else v)
                for k, v in d.items()}

    for proto in ("d1", "d2", "d3"):
        for part in ("train", "test"):
            os.makedirs(os.path.join(args.outdir, proto, part), exist_ok=True)
    nw = args.windows

    for sid, d, ts in subjects:
        # d1: random within-subject split
        perm = np.random.RandomState(1).permutation(nw)
        tr, te = perm[: int(nw * 0.75)], perm[int(nw * 0.75):]
        write(os.path.join(args.outdir, "d1", "train", sid + ".npz"), sub(d, tr), ts[tr])
        write(os.path.join(args.outdir, "d1", "test", sid + ".npz"), sub(d, te), ts[te])
        # d2: temporal within-subject split
        cut = int(nw * 0.75)
        a, b = np.arange(cut), np.arange(cut, nw)
        write(os.path.join(args.outdir, "d2", "train", sid + ".npz"), sub(d, a), ts[a])
        write(os.path.join(args.outdir, "d2", "test", sid + ".npz"), sub(d, b), ts[b])

    # d3: disjoint participants
    ncut = max(int(len(subjects) * 0.7), 1)
    for k, (sid, d, ts) in enumerate(subjects):
        part = "train" if k < ncut else "test"
        write(os.path.join(args.outdir, "d3", part, sid + ".npz"), d, ts)

    print("wrote synthetic protocols under", args.outdir)


if __name__ == "__main__":
    main()
