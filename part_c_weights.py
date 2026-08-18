#!/usr/bin/env python3
"""
Diagnostic run for part (c).  NOT a submission file.

    python3 part_c_weights.py train.csv outdir/

Requires part_c_clean.py in the same directory: the pipeline is imported, not
copied, so the weights below are exactly the weights the submission produces.

Writes into outdir/:
    weights.txt          every column: index, block, name, standardized weight
    summary.txt          per-block and per-group aggregates + top columns
    baseline.txt         model vs median-baseline on an honest 75/25 holdout
    ablation.txt         internal NMAE with each expansion block switched off

Phases 2 and 3 are skippable:  PARTC_HOLDOUT=0  PARTC_ABLATE=0
"""
import os
import sys
import time

import numpy as np

import part_c_clean as pipeline

T0 = time.time()


def say(msg):
    sys.stderr.write("[%7.1fs] %s\n" % (time.time() - T0, msg))
    sys.stderr.flush()


def columnNames(scalarNames, distributionOrder):
    """names for all four blocks, in exactly the order design() concatenates."""
    names = list(scalarNames)
    blocks = [0] * len(scalarNames)

    for nm in scalarNames:
        for i in range(pipeline.bumpsPerScalar):
            names.append("gam:%s#%d" % (nm, i))
            blocks.append(1)

    nRate = len(pipeline.rateBumpCentres)
    for rateName in ("sel", "sel2", "pk_g2"):
        for gateName in pipeline.confidenceGateNames:
            for ri in range(nRate):
                for gi in range(5):
                    names.append("tns:%sx%s[r%d,g%d]" % (rateName, gateName, ri, gi))
                    blocks.append(2)
    for firstName, secondName in pipeline.gatePairNames:
        for i in range(7):
            for j in range(7):
                names.append("pair:%sx%s[%d,%d]" % (firstName, secondName, i, j))
                blocks.append(2)

    nBins = pipeline.candidateCount // pipeline.coarseFactor
    for blockName in distributionOrder:
        for b in range(nBins):
            names.append("dist:%s@%gbpm" % (blockName, pipeline.candidateBpm[b * pipeline.coarseFactor]))
            blocks.append(3)
    return names, np.asarray(blocks)


def nmse(y, p):
    return float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: part_c_weights.py train.csv outdir/\n")
        return 2
    trainPath, outDir = sys.argv[1:3]
    if not os.path.isdir(outDir):
        os.makedirs(outDir)

    bvp, acc, eda, target = pipeline.readSignalsCsv(trainPath)
    if target is None:
        raise ValueError("train.csv must contain an 'hr' column")

    histogram, _ = np.histogram(
        target, bins=np.append(pipeline.candidateBpm - 0.25, pipeline.candidateBpm[-1] + 0.25),
        density=True)
    smoothingWindow = np.hanning(21)
    smoothedPrior = np.convolve(histogram, smoothingWindow / smoothingWindow.sum(),
                                mode="same") + 1e-6
    logRatePrior = pipeline.logSafe(smoothedPrior / smoothedPrior.sum())[None, :].astype(np.float32)

    scalars, scalarNames, distributions, gates = pipeline.buildScalarFeaturesChunked(
        bvp, acc, eda, logRatePrior, tag="train")
    del bvp, acc, eda
    say("scalars %s" % (scalars.shape,))

    knotCentres, knotWidths = pipeline.fitScalarBumpKnots(scalars)
    gateKnots, pairKnots = pipeline.fitGateBumpKnots(gates)
    distributionOrder = list(distributions.keys())
    parts = [scalars,
             pipeline.expandScalarsToBumps(scalars, knotCentres, knotWidths),
             pipeline.expandRateByConfidence(gates, gateKnots, pairKnots),
             np.hstack([distributions[nm] for nm in distributionOrder]).astype(np.float32)]
    blockSizes = [p.shape[1] for p in parts]
    blockIdOfColumn = np.concatenate([[i] * s for i, s in enumerate(blockSizes)])
    design = np.hstack(parts)
    del parts, distributions
    say("design %s blocks %s" % (design.shape, blockSizes))

    names, blockOfName = columnNames(scalarNames, distributionOrder)
    if len(names) != design.shape[1]:
        raise SystemExit("name generator produced %d names for %d columns"
                         % (len(names), design.shape[1]))
    if not np.array_equal(blockOfName, blockIdOfColumn):
        raise SystemExit("name generator block layout disagrees with design layout")
    say("column names verified against design layout")

    designMean = design.mean(0)
    designStd = design.std(0) + np.float32(1e-8)
    design = ((design - designMean) / designStd).astype(np.float32)

    # ---- phase 1: the real fit, on all training rows -----------------------
    weights, targetMean = pipeline.fitBlockRidgeThenIrls(
        design, target, blockIdOfColumn, len(blockSizes))
    weights = np.asarray(weights, np.float64)

    blockName = {0: "scalar", 1: "gam", 2: "tensor", 3: "dist"}
    with open(os.path.join(outDir, "weights.txt"), "w") as f:
        f.write("# intercept\t%.10g\n" % targetMean)
        f.write("# rows\t%d\n# columns\t%d\n" % (len(target), design.shape[1]))
        f.write("# blockSizes\t%s\n" % ",".join(str(s) for s in blockSizes))
        f.write("# fastMode\t%s\n" % pipeline.fastMode)
        f.write("index\tblock\tname\tweight\n")
        for i, (b, nm, w) in enumerate(zip(blockIdOfColumn, names, weights)):
            f.write("%d\t%s\t%s\t%.10g\n" % (i, blockName[int(b)], nm, w))
    say("wrote weights.txt")

    absWeight = np.abs(weights)
    with open(os.path.join(outDir, "summary.txt"), "w") as f:
        f.write("block\tcolumns\tsum|w|\tmean|w|\tmax|w|\tshare\n")
        total = absWeight.sum()
        for b, size in enumerate(blockSizes):
            m = blockIdOfColumn == b
            f.write("%s\t%d\t%.6g\t%.6g\t%.6g\t%.3f%%\n" % (
                blockName[b], size, absWeight[m].sum(), absWeight[m].mean(),
                absWeight[m].max(), 100.0 * absWeight[m].sum() / total))

        f.write("\n# gam groups (all bumps of one scalar), by sum|w|\n")
        gamStart = blockSizes[0]
        rows = []
        for j, nm in enumerate(scalarNames):
            s = slice(gamStart + j * pipeline.bumpsPerScalar,
                      gamStart + (j + 1) * pipeline.bumpsPerScalar)
            rows.append((absWeight[s].sum(), nm))
        for v, nm in sorted(rows, reverse=True):
            f.write("%.6g\t%s\n" % (v, nm))

        f.write("\n# tensor groups, by sum|w|\n")
        groups = {}
        for i in np.where(blockIdOfColumn == 2)[0]:
            key = names[i].split("[")[0]
            groups[key] = groups.get(key, 0.0) + absWeight[i]
        for key, v in sorted(groups.items(), key=lambda kv: -kv[1]):
            f.write("%.6g\t%s\n" % (v, key))

        f.write("\n# distribution blocks, by sum|w|\n")
        groups = {}
        for i in np.where(blockIdOfColumn == 3)[0]:
            key = names[i].split("@")[0]
            groups[key] = groups.get(key, 0.0) + absWeight[i]
        for key, v in sorted(groups.items(), key=lambda kv: -kv[1]):
            f.write("%.6g\t%s\n" % (v, key))

        f.write("\n# top 60 individual columns by |w|\n")
        for i in np.argsort(-absWeight)[:60]:
            f.write("%.6g\t%s\t%s\n" % (weights[i], blockName[int(blockIdOfColumn[i])], names[i]))

        f.write("\n# scalar columns by |w| (all %d)\n" % blockSizes[0])
        for i in np.argsort(-absWeight[:blockSizes[0]]):
            f.write("%.6g\t%s\n" % (weights[i], names[i]))
    say("wrote summary.txt")

    splitIndex = int(len(target) * 0.75)

    # ---- phase 2: honest holdout, model vs median baseline -----------------
    if os.environ.get("PARTC_HOLDOUT", "1") == "1":
        say("holdout: refitting on the first 75%% of rows")
        holdWeights, holdMean = pipeline.fitBlockRidgeThenIrls(
            design[:splitIndex], target[:splitIndex],
            blockIdOfColumn, len(blockSizes))
        heldY = target[splitIndex:]
        modelPred = design[splitIndex:] @ np.asarray(holdWeights, np.float32) + holdMean
        medianPred = np.full_like(heldY, np.median(target[:splitIndex]))
        with open(os.path.join(outDir, "baseline.txt"), "w") as f:
            f.write("# fit on rows [0,%d), evaluated on rows [%d,%d)\n"
                    % (splitIndex, splitIndex, len(target)))
            f.write("model\tNMAE\t%.6f\n" % pipeline.normalisedMeanAbsError(heldY, modelPred))
            f.write("model\tNMSE\t%.6f\n" % nmse(heldY, modelPred))
            f.write("median\tNMAE\t%.6f\n" % pipeline.normalisedMeanAbsError(heldY, medianPred))
            f.write("median\tNMSE\t%.6f\n" % nmse(heldY, medianPred))
            f.write("medianValue\t%.4f\n" % np.median(target[:splitIndex]))
        say("wrote baseline.txt")

    # ---- phase 3: what is each expansion block worth? ----------------------
    if os.environ.get("PARTC_ABLATE", "1") == "1":
        OFF = 1e9
        configs = (("scalars only", (1.0, OFF, OFF, OFF)),
                   ("scalars+gam", (1.0, 1.0, OFF, OFF)),
                   ("scalars+tensor", (1.0, OFF, 1.0, OFF)),
                   ("scalars+dist", (1.0, OFF, OFF, 1.0)),
                   ("all, equal penalty", (1.0, 1.0, 1.0, 1.0)),
                   ("all, best preset", pipeline.blockPenaltyPresets[0]))
        splitMean = target[:splitIndex].mean()
        gram, rhs = pipeline.normalEquations(
            design[:splitIndex], (target[:splitIndex] - splitMean).astype(np.float32))
        with open(os.path.join(outDir, "ablation.txt"), "w") as f:
            f.write("# ridge only (no IRLS), selected on rows [%d,%d)\n"
                    % (splitIndex, len(target)))
            f.write("config\tinternalNMAE\tpenalty\n")
            for label, multipliers in configs:
                columnScale = (1.0 / np.sqrt(
                    np.asarray(multipliers, np.float64)[:len(blockSizes)]))[blockIdOfColumn]
                score, penalty, _ = pipeline.bestPenaltyOnRidgePath(
                    gram, rhs, columnScale, design[splitIndex:], target[splitIndex:], splitMean)
                f.write("%s\t%.4f\t%.3g\n" % (label, score, penalty))
                f.flush()
                say("  %-20s NMAE %.4f" % (label, score))
        say("wrote ablation.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
