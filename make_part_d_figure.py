#!/usr/bin/env python3
"""Representative Part (d) five-minute window (report item 2ii).

    python3 make_part_d_figure.py <a_labelled_dir> fig_d_window.pdf

Reads the smallest .npz in the directory, picks a window whose signals are all
finite, and plots BVP, normalised squared acceleration, EDA, ECG and heart rate
against the corresponding CGM glucose target.
"""
import glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

srcDir, outPath = sys.argv[1], sys.argv[2]
path = min(glob.glob(os.path.join(srcDir, "*.npz")), key=os.path.getsize)

with np.load(path, allow_pickle=False) as z:
    glucose = np.asarray(z["glucose"], float)
    hrAll = np.asarray(z["e4_hr"], np.float32)
    ok = np.isfinite(hrAll).all(1) & np.isfinite(glucose)
    i = np.where(ok)[0][np.argmin(np.abs(glucose[ok] - np.median(glucose[ok])))]
    bvp = np.asarray(z["e4_bvp"][i], np.float32)
    acc = np.asarray(z["e4_acc"][i], np.float32)
    eda = np.asarray(z["e4_eda"][i], np.float32)
    ecg = np.asarray(z["zephyr_ecg"][i], np.float32)
    hr = hrAll[i]
    g = glucose[i]
print("file %s  window %d  glucose %.1f mg/dL" % (os.path.basename(path), i, g))

asq = (acc ** 2).sum(1)
asqNorm = asq / np.nanmean(asq)
INK, ACC, SEC = "#16201f", "#a8323c", "#1f6f6b"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": "#5c6b68", "ytick.color": "#5c6b68",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
fig, ax = plt.subplots(5, 1, figsize=(6.3, 6.0), sharex=True, gridspec_kw={"hspace": 0.3})
for a, (sig, hz, lab, col) in zip(ax, [
        (bvp, 64.0, "BVP\n(device units)", ACC),
        (asqNorm, 32.0, r"$\tilde{a}_{sq}(t)$" "\n(dimensionless)", SEC),
        (eda, 4.0, "EDA\n(µS)", INK),
        (ecg, 250.0, "Zephyr ECG\n(mV)", ACC),
        (hr, 1.0, "E4 heart rate\n(bpm)", SEC)]):
    a.plot(np.arange(len(sig)) / hz, sig, color=col, lw=0.5)
    a.set_ylabel(lab)
    a.grid(axis="y", color="#dbe3e0", lw=0.5)
ax[1].axhline(1.0, color="#5c6b68", lw=0.6, ls=":")
ax[0].set_title("Representative Part (d) window — CGM glucose target = %.1f mg/dL" % g,
                fontsize=8.5, pad=6)
ax[-1].set_xlabel("Time within the five-minute window (s)")
ax[-1].set_xlim(0, 300)
fig.savefig(outPath, bbox_inches="tight")
print("wrote", outPath)
