#!/usr/bin/env python3
"""
Produces the two numbers the report needs for part (c), against the model that
is actually submitted.  NOT a submission file.  Needs part_c_pruned.py alongside.

    python3 part_c_report.py train.csv report_c.txt

  * top features by |standardized regression coefficient| (report item 3)
  * model vs median-of-training-target baseline, NMAE and NMSE, on a held-out
    25% that the fit never sees (report item 4)
"""
import sys
import time

import numpy as np

import part_c_pruned as pipeline

T0 = time.time()

MODALITY = {"t_": "BVP (time domain)", "a_": "Accelerometer", "e_": "EDA",
            "sp_": "BVP spectrum at the selected rate", "sg_": "BVP consensus at the selected rate",
            "x_": "Cross-estimator agreement", "AC_": "Accelerometer spectrum"}


def say(msg):
    sys.stderr.write("[%7.1fs] %s\n" % (time.time() - T0, msg))
    sys.stderr.flush()


def modalityOf(name):
    base = name[4:] if name.startswith("gam:") else name
    base = base.split("#")[0]
    for prefix, label in MODALITY.items():
        if base.startswith(prefix):
            return label
    return "BVP evidence surface (%s)" % base.split("_")[0]


def nmse(y, pred):
    return float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: part_c_report.py train.csv report_c.txt\n")
        return 2
    trainPath, outPath = sys.argv[1:3]

    bvp, acc, eda, y = pipeline.readSignals(trainPath)
    if y is None:
        raise ValueError("train.csv must contain an 'hr' column")
    edges = np.append(pipeline.candidateBpm - 0.25, pipeline.candidateBpm[-1] + 0.25)
    hist, _ = np.histogram(y, bins=edges, density=True)
    window = np.hanning(21)
    prior = np.convolve(hist, window / window.sum(), mode="same") + 1e-6
    logPrior = pipeline.logSafe(prior / prior.sum())[None, :].astype(np.float32)

    features, names = pipeline.buildFeaturesChunked(bvp, acc, eda, logPrior, tag="train")
    del bvp, acc, eda
    centres, widths = pipeline.fitBumpKnots(features)
    parts = [features, pipeline.expandToBumps(features, centres, widths)]
    blockSizes = [p.shape[1] for p in parts]
    blockOf = np.concatenate([[i] * n for i, n in enumerate(blockSizes)])
    allNames = list(names) + ["gam:%s#%d" % (nm, i)
                              for nm in names for i in range(pipeline.bumpsPerScalar)]
    X = np.hstack(parts)
    del parts
    if len(allNames) != X.shape[1]:
        raise SystemExit("%d names for %d columns" % (len(allNames), X.shape[1]))
    say("design %s blocks %s" % (X.shape, blockSizes))

    mean, std = X.mean(0), X.std(0) + np.float32(1e-8)
    X = ((X - mean) / std).astype(np.float32)
    cut = int(len(y) * 0.75)

    say("holdout: fitting on the first 75% of rows")
    holdW, holdMean = pipeline.fitBlockRidge(X[:cut], y[:cut], blockOf, len(blockSizes))
    heldY = y[cut:]
    modelPred = X[cut:] @ np.asarray(holdW, np.float32) + holdMean
    medianPred = np.full_like(heldY, np.median(y[:cut]))

    say("full fit for the coefficient table")
    w, _ = pipeline.fitBlockRidge(X, y, blockOf, len(blockSizes))
    w = np.asarray(w, np.float64)
    order = np.argsort(-np.abs(w))

    with open(outPath, "w") as f:
        f.write("PART (C) REPORT NUMBERS\n")
        f.write("design: %d columns (%d scalar + %d spline), %d training rows\n\n"
                % (X.shape[1], blockSizes[0], blockSizes[1], len(y)))
        f.write("--- item 4: model vs median baseline ---\n")
        f.write("fit on rows [0,%d), evaluated on rows [%d,%d)\n" % (cut, cut, len(y)))
        f.write("%-22s %8s %8s\n" % ("", "NMAE", "NMSE"))
        f.write("%-22s %8.4f %8.4f\n" % ("this model",
                pipeline.nmae(heldY, modelPred), nmse(heldY, modelPred)))
        f.write("%-22s %8.4f %8.4f\n" % ("median baseline",
                pipeline.nmae(heldY, medianPred), nmse(heldY, medianPred)))
        f.write("median of training hr = %.2f bpm\n\n" % np.median(y[:cut]))

        f.write("--- item 3: top features by |standardized coefficient| ---\n")
        f.write("%-24s %10s   %s\n" % ("feature", "|std coef|", "modality"))
        for i in order[:15]:
            f.write("%-24s %10.4f   %s\n" % (allNames[i], abs(w[i]), modalityOf(allNames[i])))
    say("wrote %s" % outPath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
