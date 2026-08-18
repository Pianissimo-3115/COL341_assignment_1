#!/usr/bin/env python3
"""
Diagnostic: does a top-K subset of the 1960-column design, refitted, match the
whole thing?  NOT a submission file.  Needs part_c_pruned.py alongside it.

    python3 part_c_topk.py train.csv out.txt

Ranking weights are fitted on the first 75% of rows only; every K is scored on
the untouched last 25%, so the selection never sees the evaluation rows.
One Gram matrix is built and then sliced per K, so the sweep is nearly free.
"""
import sys
import time

import numpy as np

import part_c_pruned as pipeline

T0 = time.time()
SIZES = (50, 100, 200, 300, 400, 600, 800, 1200, 1600, 1960)


def say(msg):
    sys.stderr.write("[%7.1fs] %s\n" % (time.time() - T0, msg))
    sys.stderr.flush()


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: part_c_topk.py train.csv out.txt\n")
        return 2
    trainPath, outPath = sys.argv[1:3]

    bvp, acc, eda, target = pipeline.readSignalsCsv(trainPath)
    if target is None:
        raise ValueError("train.csv must contain an 'hr' column")
    histogram, _ = np.histogram(
        target, bins=np.append(pipeline.candidateBpm - 0.25, pipeline.candidateBpm[-1] + 0.25),
        density=True)
    win = np.hanning(21)
    prior = np.convolve(histogram, win / win.sum(), mode="same") + 1e-6
    logRatePrior = pipeline.logSafe(prior / prior.sum())[None, :].astype(np.float32)

    scalars, scalarNames = pipeline.buildScalarFeaturesChunked(
        bvp, acc, eda, logRatePrior, tag="train")
    del bvp, acc, eda
    knotCentres, knotWidths = pipeline.fitScalarBumpKnots(scalars)
    parts = [scalars, pipeline.expandScalarsToBumps(scalars, knotCentres, knotWidths)]
    blockSizes = [p.shape[1] for p in parts]
    blockIdOfColumn = np.concatenate([[i] * s for i, s in enumerate(blockSizes)])
    names = list(scalarNames) + ["gam:%s#%d" % (nm, i)
                                 for nm in scalarNames for i in range(pipeline.bumpsPerScalar)]
    design = np.hstack(parts)
    del parts
    say("design %s blocks %s" % (design.shape, blockSizes))

    design = ((design - design.mean(0)) / (design.std(0) + np.float32(1e-8))).astype(np.float32)
    splitIndex = int(len(target) * 0.75)
    splitMean = target[:splitIndex].mean()
    columnScale = (1.0 / np.sqrt(np.asarray(pipeline.blockPenaltyPresets[0], np.float64)))[blockIdOfColumn]

    say("building the full Gram once")
    gram, rhs = pipeline.normalEquations(
        design[:splitIndex], (target[:splitIndex] - splitMean).astype(np.float32))

    fullScore, fullPenalty, fullWeights = pipeline.bestPenaltyOnRidgePath(
        gram, rhs, columnScale, design[splitIndex:], target[splitIndex:], splitMean)
    say("all 1960 columns: NMAE %.4f (penalty %.3g)" % (fullScore, fullPenalty))
    ranking = np.argsort(-np.abs(np.asarray(fullWeights, np.float64)))

    lines = ["# ranking weights fitted on rows [0,%d), scored on rows [%d,%d)"
             % (splitIndex, splitIndex, len(target)),
             "# all 1960 columns: NMAE %.4f" % fullScore,
             "K\tscalars\tgam\tNMAE\tpenalty\tvsFull"]
    for k in SIZES:
        sel = np.sort(ranking[:k])
        score, penalty, _ = pipeline.bestPenaltyOnRidgePath(
            gram[np.ix_(sel, sel)], rhs[sel], columnScale[sel],
            np.ascontiguousarray(design[splitIndex:][:, sel]), target[splitIndex:], splitMean)
        nScalar = int((blockIdOfColumn[sel] == 0).sum())
        lines.append("%d\t%d\t%d\t%.4f\t%.3g\t%+.4f"
                     % (k, nScalar, k - nScalar, score, penalty, score - fullScore))
        say("  K=%-5d scalars=%-4d gam=%-5d NMAE %.4f (%+.4f)"
            % (k, nScalar, k - nScalar, score, score - fullScore))
        with open(outPath, "w") as f:
            f.write("\n".join(lines) + "\n")

    best = np.sort(ranking[:200])
    with open(outPath, "a") as f:
        f.write("\n# the top-200 columns, in design order\n")
        for i in best:
            f.write("%d\t%s\n" % (i, names[i]))
    say("wrote %s" % outPath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
