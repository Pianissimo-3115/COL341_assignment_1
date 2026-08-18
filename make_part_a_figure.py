import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NBLK, BLK = 10, 164
df = pd.read_csv("part_a_rows.csv")
hr = df["hr"].to_numpy(float)
X = df.drop(columns=["hr"]).to_numpy(float)

bvpIdx = np.concatenate([np.arange(BLK*b+96, BLK*b+160) for b in range(NBLK)])
accIdx = [np.concatenate([np.arange(BLK*b+32*a, BLK*b+32*(a+1)) for b in range(NBLK)]) for a in range(3)]
edaIdx = np.concatenate([np.arange(BLK*b+160, BLK*b+164) for b in range(NBLK)])

bvp = X[:, bvpIdx]; eda = X[:, edaIdx]
acc = np.stack([X[:, accIdx[0]], X[:, accIdx[1]], X[:, accIdx[2]]], 1)
# representative: heart rate closest to the median, and a clean (non-flat, non-spiking) BVP
z = np.abs((bvp - bvp.mean(1, keepdims=True)) / (bvp.std(1, keepdims=True) + 1e-9))
ok = ((bvp.std(1) > np.percentile(bvp.std(1), 25)) & (z.max(1) < 5)
      & (eda.min(1) > 1e-6))
cand = np.where(ok)[0]
i = cand[np.argmin(np.abs(hr[cand] - np.median(hr)))]
print("row %d  hr = %.1f bpm  (median of sample = %.1f)" % (i, hr[i], np.median(hr)))

tB = np.arange(640)/64.0; tA = np.arange(320)/32.0; tE = np.arange(40)/4.0
asq = (acc[i]**2).sum(0)
asqNorm = asq/asq.mean()

INK="#16201f"; ACC="#a8323c"; SEC="#1f6f6b"; MUT="#5c6b68"
plt.rcParams.update({"font.size":8,"axes.edgecolor":INK,"axes.labelcolor":INK,
                     "text.color":INK,"xtick.color":MUT,"ytick.color":MUT,
                     "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
fig, ax = plt.subplots(3, 1, figsize=(6.3, 4.4), sharex=True,
                       gridspec_kw={"height_ratios":[1.25,1,1],"hspace":0.28})

ax[0].plot(tB, bvp[i], color=ACC, lw=0.8)
ax[0].set_ylabel("BVP\n(device units)")
ax[0].set_title("Representative Part (a) window — heart-rate target = %.1f bpm" % hr[i],
                fontsize=9, color=INK, pad=6)

ax[1].plot(tA, asqNorm, color=SEC, lw=0.8)
ax[1].axhline(1.0, color=MUT, lw=0.6, ls=":")
ax[1].set_ylabel(r"$\tilde{a}_{sq}(t)$" "\n(dimensionless)")

ax[2].plot(tE, eda[i], color=INK, lw=1.0, marker="o", ms=2.5)
ax[2].set_ylabel("EDA\n(µS)")
ax[2].set_xlabel("Time within the 10 s window (s)")
ax[2].set_xlim(0, 10)

for a in ax: a.grid(axis="y", color="#dbe3e0", lw=0.5)
fig.savefig("part_a_window.pdf", bbox_inches="tight")
fig.savefig("part_a_window.png", bbox_inches="tight", dpi=200)
print("wrote part_a_window.pdf / .png")
