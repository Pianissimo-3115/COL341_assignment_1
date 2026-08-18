import gc
import os
import sys
import time

import numpy as np
import pandas as pd

fastMode = os.environ.get("PARTC_FAST", "0") == "1"
cvFolds = int(os.environ.get("PARTC_CV_FOLDS", "5"))
cvEnabled = os.environ.get("PARTC_CV", "1") == "1" and cvFolds >= 2

startTime = time.time()


def log(message):
    sys.stderr.write("[%7.1fs] %s\n" % (time.time() - startTime, message))
    sys.stderr.flush()


blocksPerWindow, valuesPerBlock = 10, 164
bvpColumns = np.concatenate([np.arange(valuesPerBlock * block + 96, valuesPerBlock * block + 160)
                             for block in range(blocksPerWindow)])
accColumns = [np.concatenate([np.arange(valuesPerBlock * block + 32 * axis,
                                               valuesPerBlock * block + 32 * (axis + 1))
                                     for block in range(blocksPerWindow)]) for axis in range(3)]
edaColumns = np.concatenate([np.arange(valuesPerBlock * block + 160, valuesPerBlock * block + 164)
                             for block in range(blocksPerWindow)])

bvpHz, accHz = 64.0, 32.0
accSamples = 320

candidateBpm = np.arange(36.0, 216.0, 0.5)
candidateCount = len(candidateBpm)
candidateBpm32 = candidateBpm.astype(np.float32)
candidateHz = candidateBpm / 60.0
def logSafe(value):
    return np.log(np.maximum(value, 1e-12)).astype(np.float32)


def cosSinProjection(n, rateHz, targetHz):
    t = np.arange(n) / rateHz
    window = np.hanning(n)
    angle = 2.0 * np.pi * np.outer(t, targetHz)
    return ((np.cos(angle) * window[:, None]).astype(np.float32),
            (np.sin(angle) * window[:, None]).astype(np.float32))


subWindowLengths = (640, 320, 256, 160, 128, 80)
bvpProjection = {length: cosSinProjection(length, bvpHz, candidateHz)
                         for length in subWindowLengths}
accCos, accSin = cosSinProjection(accSamples, accHz, candidateHz)
autocorrKernel = np.cos(2.0 * np.pi * candidateHz[:, None]
                        * (60.0 / candidateBpm)[None, :]).astype(np.float32)


def movingAverage(rows, width):
    if width <= 1:
        return rows
    pad = width // 2
    padded = np.pad(rows, ((0, 0), (pad, width - 1 - pad)), mode="edge")
    cumsum = np.cumsum(padded, axis=1, dtype=np.float32)
    cumsum = np.concatenate([np.zeros((len(rows), 1), np.float32), cumsum], 1)
    return (cumsum[:, width:] - cumsum[:, :-width]) * np.float32(1.0 / width)


def bandPass(rows, short, long):
    return movingAverage(rows, short) - movingAverage(rows, long)


def powerSpectrum(rows, cosB, sinB):
    centred = rows - rows.mean(1, keepdims=True)
    return (centred @ cosB) ** 2 + (centred @ sinB) ** 2


def normaliseRows(rows):
    return rows / (rows.sum(1, keepdims=True) + np.float32(1e-20))


def softmaxRows(scores, temp):
    scaled = (scores - scores.max(1, keepdims=True)) * np.float32(1.0 / temp)
    weights = np.exp(scaled, dtype=np.float32)
    return weights / (weights.sum(1, keepdims=True) + np.float32(1e-20))


def arSpectrum(rows, order, ridge=1e-3):
    z = rows - rows.mean(1, keepdims=True)
    z = z / (z.std(1, keepdims=True) + np.float32(1e-6))
    n, span = z.shape
    autocov = np.empty((n, order + 1), np.float32)
    for lag in range(order + 1):
        autocov[:, lag] = np.einsum("ij,ij->i", z[:, :span - lag], z[:, lag:]) / span
    toeplitzIdx = np.abs(np.arange(order)[:, None] - np.arange(order)[None, :])
    toeplitz = autocov[:, toeplitzIdx].copy()
    toeplitz[:, np.arange(order), np.arange(order)] *= np.float32(1.0 + ridge)
    arCoef = np.linalg.solve(toeplitz, autocov[:, 1:order + 1][..., None])[..., 0]
    innovation = np.maximum(autocov[:, 0] - (arCoef * autocov[:, 1:order + 1]).sum(1), 1e-6)
    lag = np.arange(1, order + 1)
    angle = 2.0 * np.pi * np.outer(candidateHz / bvpHz, lag)
    cosB = np.ascontiguousarray(np.cos(angle).T.astype(np.float32))
    sinB = np.ascontiguousarray(np.sin(angle).T.astype(np.float32))
    return (innovation[:, None]
            / ((1 - arCoef @ cosB) ** 2 + (arCoef @ sinB) ** 2 + 1e-9)).astype(np.float32)


def peakBpm(surface):
    peak = np.argmax(surface, 1)
    row = np.arange(len(surface))
    centre = np.clip(peak, 1, surface.shape[1] - 2)
    left, mid, right = surface[row, centre - 1], surface[row, centre], surface[row, centre + 1]
    curve = left - 2 * mid + right
    valid = np.abs(curve) > 1e-20
    offset = np.clip(np.where(valid, 0.5 * (left - right) / np.where(valid, curve, 1.0), 0.0),
                     -1, 1)
    return ((candidateBpm32[centre]
             + offset * np.float32(candidateBpm[1] - candidateBpm[0])).astype(np.float32),
            surface[row, peak].astype(np.float32))


def strongestPeaks(surface, count=3, guard=10.0):
    masked = surface.copy()
    bpms, scores = [], []
    for _ in range(count):
        bpm, score = peakBpm(masked)
        bpms.append(bpm)
        scores.append(score)
        masked = np.where(np.abs(candidateBpm32[None, :] - bpm[:, None]) < np.float32(guard),
                      np.float32(-1e30), masked)
    return np.stack(bpms, 1), np.stack(scores, 1)


def lookupBpm(surface, bpm):
    pos = np.clip((np.asarray(bpm) - candidateBpm[0]) / (candidateBpm[1] - candidateBpm[0]),
                  0, candidateCount - 1.001)
    lo = pos.astype(np.int32)
    frac = (pos - lo).astype(np.float32)
    return surface[:, lo] * (1 - frac) + surface[:, lo + 1] * frac


def lookupBpmPerRow(surface, bpm):
    pos = np.clip((np.asarray(bpm) - candidateBpm[0]) / (candidateBpm[1] - candidateBpm[0]),
                  0, candidateCount - 1.001)
    lo = pos.astype(np.int32)
    frac = (pos - lo).astype(np.float32)
    row = np.arange(len(surface))
    return (surface[row, lo] * (1 - frac) + surface[row, lo + 1] * frac).astype(np.float32)


def addColumns(columns, names, prefix, pairs):
    for suffix, values in pairs:
        columns.append(np.asarray(values, np.float32)[:, None])
        names.append(prefix + suffix)


def bumpBasis(values, centres, width):
    z = (np.asarray(values, np.float32)[:, None] -
         np.asarray(centres, np.float32)[None, :]) * np.float32(1.0 / max(width, 1e-6))
    bump = np.exp(-0.5 * z * z)
    return (bump / (bump.sum(1, keepdims=True) + np.float32(1e-9))).astype(np.float32)


consensusConfigs = ((320, 64, "g5"), (256, 48, "g4"), (160, 32, "g25"),
                    (128, 32, "g2"), (80, 16, "g125"))
if fastMode:
    consensusConfigs = ((320, 64, "g5"), (256, 64, "g4"), (160, 48, "g25"),
                        (128, 48, "g2"), (80, 32, "g125"))

mainScoreWeights = {
    "lP": 0.25, "lPwhite": 0.5, "lAR": -0.25, "lAR2": 4.0, "lR": 0.5, "lRaw": -0.5,
    "g5": 0.5, "g4": -0.5, "g25": 1.0, "g2": 4.0, "g125": 4.0, "g25w": -2.0,
    "prior": 4.0, "lP2": 0.5, "lP3": 0.25, "lPh": 0.25, "lPh3": 0.5,
    "g5_2": -0.5, "g5_h": 1.0,
}
consensusScoreWeights = {"g2": 1.0, "g25": 1.0, "g5": 1.0, "lP": 0.5, "prior": 0.5}
spectralScoreWeights = {"lP": 1.0, "lAR": 1.0, "lR": 0.5}


def buildSurfaces(bvp, acc, prior):
    raw = np.ascontiguousarray(bvp, np.float32)
    band = bandPass(raw, 5, 61)
    surfaces = {}

    spectrum = normaliseRows(powerSpectrum(band, *bvpProjection[640]))
    surfaces["lP"] = logSafe(spectrum)
    surfaces["lPwhite"] = surfaces["lP"] - movingAverage(surfaces["lP"], 61)
    surfaces["lAR"] = logSafe(normaliseRows(arSpectrum(band, 32)))
    surfaces["lAR2"] = logSafe(normaliseRows(arSpectrum(band, 20)))
    autocorr = spectrum @ autocorrKernel
    surfaces["lR"] = logSafe(normaliseRows(
        np.maximum(autocorr - autocorr.min(1, keepdims=True), 1e-9)))
    surfaces["lRaw"] = logSafe(normaliseRows(powerSpectrum(raw, *bvpProjection[640])))

    accRaw = np.ascontiguousarray(acc, np.float32)
    accSpectrum = normaliseRows(powerSpectrum(accRaw[:, 0], accCos, accSin)
                                + powerSpectrum(accRaw[:, 1], accCos, accSin)
                                + powerSpectrum(accRaw[:, 2], accCos, accSin))
    surfaces["lPA"] = logSafe(accSpectrum)
    surfaces["prior"] = np.repeat(prior, len(raw), 0)

    for span, hop, name in consensusConfigs:
        cosB, sinB = bvpProjection[span]
        logSum = None
        count = 0
        for start in range(0, band.shape[1] - span + 1, hop):
            logSpec = logSafe(normaliseRows(powerSpectrum(band[:, start:start + span], cosB, sinB)))
            logSum = logSpec if logSum is None else logSum + logSpec
            count += 1
        surfaces[name] = logSum * np.float32(1.0 / count)
    surfaces["g25w"] = surfaces["g25"] - movingAverage(surfaces["g25"], 61)

    surfaces["lP2"] = lookupBpm(surfaces["lP"], candidateBpm * 2)
    surfaces["lP3"] = lookupBpm(surfaces["lP"], candidateBpm * 3)
    surfaces["lPh"] = lookupBpm(surfaces["lP"], candidateBpm / 2)
    surfaces["lPh3"] = lookupBpm(surfaces["lP"], candidateBpm / 3)
    surfaces["g5_2"] = lookupBpm(surfaces["g5"], candidateBpm * 2)
    surfaces["g5_h"] = lookupBpm(surfaces["g5"], candidateBpm / 2)
    return surfaces, band, spectrum, accSpectrum


def combineSurfaces(surfaces, weights):
    total = None
    for name, weight in weights.items():
        if weight and name in surfaces:
            total = (np.float32(weight) * surfaces[name] if total is None
                     else total + np.float32(weight) * surfaces[name])
    total = total - total.mean(1, keepdims=True)
    return total / (total.std(1, keepdims=True) + np.float32(1e-6))


def describeSurface(surface, tag, columns, names):
    peakRate, _ = peakBpm(surface)
    dist = softmaxRows(surface, 1.0)
    centroid = dist @ candidateBpm32
    cumsum = np.cumsum(dist, 1)
    median = candidateBpm32[np.argmax(cumsum >= 0.5, 1)]
    q25 = candidateBpm32[np.argmax(cumsum >= 0.25, 1)]
    q75 = candidateBpm32[np.argmax(cumsum >= 0.75, 1)]
    entropy = -(dist * np.log(dist + 1e-12)).sum(1)
    spread = np.sqrt(np.maximum(dist @ (candidateBpm32 ** 2) - centroid ** 2, 0))
    bpms, scores = strongestPeaks(surface, 3, 10.0)
    addColumns(columns, names, tag + "_",
         (("pk", peakRate), ("p2", bpms[:, 1]), ("p3", bpms[:, 2]),
          ("m12", scores[:, 0] - scores[:, 1]), ("m13", scores[:, 0] - scores[:, 2]),
          ("d12", bpms[:, 1] - peakRate), ("d13", bpms[:, 2] - peakRate),
          ("cen", centroid), ("med", median), ("q25", q25), ("q75", q75),
          ("iqr", q75 - q25), ("ent", entropy), ("sd", spread)))
    return peakRate, scores, entropy


def addBvpFeatures(bvp, band, columns, names):
    raw = np.ascontiguousarray(bvp, np.float32)
    wide = bandPass(raw, 9, 81)
    amplitude = band.std(1) + np.float32(1e-6)
    z = band / amplitude[:, None]
    diff = np.diff(band, axis=1)
    perSecond = band.reshape(len(band), 10, 64).std(2)
    addColumns(columns, names, "t_", (
        ("zc", (np.diff(np.signbit(band), axis=1).sum(1) / 2.0) * 6.0),
        ("zc2", (np.diff(np.signbit(wide), axis=1).sum(1) / 2.0) * 6.0),
        ("lamp", np.log(amplitude)), ("lraw", np.log(raw.std(1) + 1e-6)),
        ("lmad", np.log(np.abs(band).mean(1) + 1e-6)),
        ("kurt", (z ** 4).mean(1)), ("skew", (z ** 3).mean(1)),
        ("ldiff", np.log(np.abs(diff).mean(1) + 1e-6)),
        ("crest", np.abs(band).max(1) / amplitude),
        ("lrange", np.log(raw.max(1) - raw.min(1) + 1e-6)),
        ("nfrac", (np.abs(z) > 3).mean(1)),
        ("envcv", perSecond.std(1) / amplitude),
        ("envmax", np.log(perSecond.max(1) + 1e-6)),
        ("envmin", np.log(perSecond.min(1) + 1e-6))))
    return np.log(amplitude).astype(np.float32)


def addMotionFeatures(acc, band, columns, names):
    raw = np.ascontiguousarray(acc, np.float32)
    magnitude = np.sqrt((raw ** 2).sum(1))
    motion = bandPass(magnitude, 3, 41)
    axisStd = raw.std(2)
    axisMean = raw.mean(2)
    axisTotal = np.abs(axisMean).sum(1) + np.float32(1e-3)
    addColumns(columns, names, "a_", (
        ("magm", magnitude.mean(1)), ("magsd", magnitude.std(1)),
        ("lmagsd", np.log(magnitude.std(1) + 1e-3)),
        ("lbpsd", np.log(motion.std(1) + 1e-3)),
        ("sdx", axisStd[:, 0]), ("sdy", axisStd[:, 1]), ("sdz", axisStd[:, 2]),
        ("lsd", np.log(axisStd.sum(1) + 1e-3)),
        ("mx", axisMean[:, 0]), ("my", axisMean[:, 1]), ("mz", axisMean[:, 2]),
        ("nx", axisMean[:, 0] / axisTotal), ("ny", axisMean[:, 1] / axisTotal),
        ("nz", axisMean[:, 2] / axisTotal),
        ("magrange", magnitude.max(1) - magnitude.min(1)),
        ("jerk", np.log(np.abs(np.diff(magnitude, axis=1)).mean(1) + 1e-3)),
        ("q90", np.percentile(np.abs(motion), 90, axis=1)),
        ("still", (np.abs(motion) < 2).mean(1))))
    envelope = movingAverage(np.abs(band), 33)[:, ::2][:, :accSamples]
    envC = envelope - envelope.mean(1, keepdims=True)
    motC = motion - motion.mean(1, keepdims=True)
    addColumns(columns, names, "x_", (("envacc", (envC * motC).sum(1) /
                             (np.sqrt((envC ** 2).sum(1) * (motC ** 2).sum(1)) + 1e-6)),))
    return np.log(axisStd.sum(1) + 1e-3).astype(np.float32)


def addEdaFeatures(eda, columns, names):
    raw = np.ascontiguousarray(eda, np.float32)
    t = np.arange(raw.shape[1], dtype=np.float32)
    t = (t - t.mean()) / t.std()
    addColumns(columns, names, "e_", (
        ("lmean", np.log(raw.mean(1) + 1e-3)), ("lstd", np.log(raw.std(1) + 1e-4)),
        ("slope", (raw * t).mean(1)), ("zero", (raw.max(1) <= 1e-6).astype(np.float32)),
        ("lrng", np.log(raw.max(1) - raw.min(1) + 1e-4))))


def buildFeatures(bvp, acc, eda, prior):
    columns, names = [], []
    surfaces, band, _, _ = buildSurfaces(bvp, acc, prior)

    mainScore = combineSurfaces(surfaces, mainScoreWeights)
    mainRate, _, _ = describeSurface(mainScore, "S", columns, names)
    consensusScore = combineSurfaces(surfaces, consensusScoreWeights)
    consensusRate, _, _ = describeSurface(consensusScore, "S2", columns, names)
    spectralScore = combineSurfaces(surfaces, spectralScoreWeights)
    spectralRate, _, _ = describeSurface(spectralScore, "S3", columns, names)
    del mainScore, consensusScore, spectralScore

    peaks = {}
    for key, tag in (("lP", "P"), ("lAR", "AR"), ("g2", "G2"), ("g25", "G25"),
                     ("g5", "G5"), ("lR", "R"), ("lPA", "AC")):
        peaks[tag], _, _ = describeSurface(surfaces[key], tag, columns, names)
    periodogramRate, arRate, rate2s = peaks["P"], peaks["AR"], peaks["G2"]
    rate25s, rate5s, autocorrRate = peaks["G25"], peaks["G5"], peaks["R"]

    estimates = np.stack([mainRate, consensusRate, spectralRate, periodogramRate,
                          arRate, rate2s, rate25s, rate5s, autocorrRate], 1)
    columns.append(estimates.std(1, keepdims=True)); names.append("x_std")
    columns.append(np.median(estimates, 1, keepdims=True).astype(np.float32)); names.append("x_med")
    columns.append(estimates.mean(1, keepdims=True)); names.append("x_mean")
    for i, name in enumerate(("P", "AR", "g2", "g25", "g5", "R")):
        columns.append(np.abs(estimates[:, 3 + i] - mainRate)[:, None])
        names.append("x_dev_" + name)
    agreement = (np.abs(estimates - mainRate[:, None]) < 4).mean(1).astype(np.float32)
    columns.append(agreement[:, None]); names.append("x_agree")
    for multiple, name in ((1.0, "f"), (2.0, "2f"), (0.5, "hf"), (3.0, "3f")):
        columns.append(lookupBpmPerRow(surfaces["lP"], mainRate * multiple)[:, None])
        names.append("sp_" + name)
        columns.append(lookupBpmPerRow(surfaces["g2"], mainRate * multiple)[:, None])
        names.append("sg_" + name)
    del surfaces

    addBvpFeatures(bvp, band, columns, names)
    addMotionFeatures(acc, band, columns, names)
    addEdaFeatures(eda, columns, names)

    return np.hstack(columns).astype(np.float32), names


def buildFeaturesChunked(bvp, acc, eda, prior, chunkRows=8192, tag=""):
    chunks, names = [], None
    n = len(bvp)
    for start in range(0, n, chunkRows):
        stop = min(start + chunkRows, n)
        block, blockNames = buildFeatures(bvp[start:stop], acc[start:stop], eda[start:stop], prior)
        chunks.append(block)
        if names is None:
            names = blockNames
        del block
        gc.collect()
        log("  %s features %d/%d" % (tag, stop, n))
    return np.concatenate(chunks), names


bumpsPerScalar = 7 if fastMode else 9
def fitBumpKnots(features):
    centres = [np.percentile(features[:, j], np.linspace(2, 98, bumpsPerScalar))
               for j in range(features.shape[1])]
    widths = [max((knots[-1] - knots[0]) / (bumpsPerScalar - 1), 1e-6) for knots in centres]
    return centres, widths


def expandToBumps(features, centres, widths):
    return np.hstack([bumpBasis(features[:, j], centres[j], widths[j])
                      for j in range(features.shape[1])]).astype(np.float32)


def readSignals(path, chunkRows=20000):
    bvpChunks, accChunks, edaChunks, targetChunks = [], [], [], []
    total = 0
    for frame in pd.read_csv(path, chunksize=chunkRows, dtype=np.float32):
        target = frame["hr"].to_numpy(np.float32) if "hr" in frame.columns else None
        if target is not None:
            frame = frame.drop(columns=["hr"])
        values = frame.to_numpy(np.float32)
        del frame
        if values.shape[1] != blocksPerWindow * valuesPerBlock:
            raise ValueError("expected %d feature columns, got %d"
                             % (blocksPerWindow * valuesPerBlock, values.shape[1]))
        bvpChunks.append(values[:, bvpColumns])
        accChunks.append(np.stack([values[:, accColumns[0]], values[:, accColumns[1]],
                                   values[:, accColumns[2]]], 1))
        edaChunks.append(values[:, edaColumns])
        if target is not None:
            targetChunks.append(target)
        total += len(values)
        del values
        gc.collect()
    log("read %s: %d rows" % (path, total))
    return (np.concatenate(bvpChunks), np.concatenate(accChunks), np.concatenate(edaChunks),
            np.concatenate(targetChunks).astype(np.float64) if targetChunks else None)


ridgePenalties = 10.0 ** np.arange(-0.5, 6.01, 0.25)
penaltyPresets = ((0.1, 1.0),
           (1.0, 1.0),
           (0.03, 0.3),
           (0.01, 30.0))
if fastMode:
    penaltyPresets = penaltyPresets[:2]


def nmae(y, pred):
    return np.abs(y - pred).sum() / np.abs(y - y.mean()).sum()


def nmse(y, pred):
    return ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def bestOnRidgePath(gram, rhs, scale, validX, validY, mean):
    eigval, eigvec = np.linalg.eigh(gram * scale[None, :] * scale[:, None])
    projRhs = eigvec.T @ (rhs * scale)
    projX = (validX * scale[None, :].astype(np.float32)) @ eigvec.astype(np.float32)
    best = None
    for penalty in ridgePenalties:
        w = projRhs / (eigval + penalty)
        pred = projX @ w.astype(np.float32) + mean
        score = nmae(validY, pred)
        if best is None or score < best[0]:
            best = (score, penalty, (eigvec @ w) * scale)
    return best


def normalEquations(X, y):
    return (X.T @ X).astype(np.float64), (X.T @ y).astype(np.float64)


def fitBlockRidge(X, y, blockOf, blocks):
    n = len(X)
    cut = int(n * 0.75)
    splitMean = y[:cut].mean()
    log("  gram (internal split)")
    head = X[:cut]
    gram, rhs = normalEquations(head, (y[:cut] - splitMean).astype(np.float32))
    del head
    gc.collect()
    validX, validY = X[cut:], y[cut:]
    best = None
    for i, preset in enumerate(penaltyPresets):
        mult = np.asarray(preset, np.float64)[:blocks]
        scale = (1.0 / np.sqrt(mult))[blockOf]
        score, penalty, w = bestOnRidgePath(gram, rhs, scale, validX, validY, splitMean)
        log("    preset %d  internal NMAE %.4f (alpha=%.3g)" % (i, score, penalty))
        if best is None or score < best[0]:
            best = (score, penalty, preset)
    del gram, rhs, validX
    gc.collect()
    log("  best preset %s alpha %.3g internal NMAE %.4f" % (best[2], best[1], best[0]))

    mult = np.asarray(best[2], np.float64)[:blocks]
    scale = (1.0 / np.sqrt(mult))[blockOf]
    penalty = best[1]
    mean = y.mean()
    centred = (y - mean).astype(np.float32)
    Xs = (X * scale[None, :].astype(np.float32))
    gram, rhs = normalEquations(Xs, centred)
    p = gram.shape[0]
    w = np.linalg.solve(gram + penalty * np.eye(p), rhs)
    log("  L2 fit done")

    bestW, bestScore = w, nmae(y[cut:], Xs[cut:] @ w.astype(np.float32) + mean)
    for it in range(3):
        resid = Xs @ w.astype(np.float32) - centred
        floor = np.float32(1.345 * np.median(np.abs(resid - np.median(resid))) + 1e-3)
        sqrtW = np.sqrt(1.0 / np.maximum(np.abs(resid), floor)).astype(np.float32)
        Xw = Xs * sqrtW[:, None]
        gramW, rhsW = normalEquations(Xw, centred * sqrtW)
        del Xw
        gc.collect()
        w = np.linalg.solve(gramW + penalty * float((sqrtW ** 2).mean()) * np.eye(p), rhsW)
        score = nmae(y[cut:], Xs[cut:] @ w.astype(np.float32) + mean)
        log("    irls %d internal NMAE %.4f" % (it, score))
        if score < bestScore:
            bestScore, bestW = score, w
        del gramW, rhsW
        gc.collect()
    del Xs
    gc.collect()
    return bestW * scale, mean


def clipPredictions(pred, y, intercept):
    lo, hi = y.min() - 15.0, y.max() + 15.0
    return np.clip(np.nan_to_num(pred, nan=intercept, posinf=hi, neginf=lo),
                   max(lo, 25.0), min(hi, 230.0))


def crossValidate(X, y, blockOf, blocks, folds):
    """K-fold CV over contiguous row blocks, using the exact fitting pipeline
    (preset search + ridge path + IRLS) that produces the final model.

    Alongside the model, each fold also scores the baseline that predicts the
    median of that fold's training targets for every validation example, so both
    predictors are measured on identical validation rows."""
    n = len(X)
    bounds = np.linspace(0, n, folds + 1).astype(int)
    oofPred = np.full(n, np.nan)
    oofMedian = np.full(n, np.nan)
    scores = []
    for k in range(folds):
        lo, hi = bounds[k], bounds[k + 1]
        validIdx = np.arange(lo, hi)
        trainIdx = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        log("cv fold %d/%d: fit on %d rows, hold out rows [%d, %d)"
            % (k + 1, folds, len(trainIdx), lo, hi))
        foldX = np.ascontiguousarray(X[trainIdx])
        foldY = y[trainIdx]
        w, intercept = fitBlockRidge(foldX, foldY, blockOf, blocks)
        del foldX
        gc.collect()
        pred = (X[validIdx] @ w.astype(np.float32) + intercept).astype(np.float64)
        pred = clipPredictions(pred, foldY, intercept)
        oofPred[validIdx] = pred

        foldMedian = np.median(foldY)
        medianPred = np.full(len(validIdx), foldMedian)
        oofMedian[validIdx] = medianPred

        validY = y[validIdx]
        scores.append((nmae(validY, pred), nmse(validY, pred),
                       nmae(validY, medianPred), nmse(validY, medianPred), foldMedian))
        log("cv fold %d/%d: model NMAE %.4f NMSE %.4f | median baseline (%.2f bpm) "
            "NMAE %.4f NMSE %.4f"
            % (k + 1, folds, scores[-1][0], scores[-1][1], foldMedian,
               scores[-1][2], scores[-1][3]))
        del w, pred, medianPred
        gc.collect()

    scores = np.asarray(scores)
    print("")
    print("%d-fold cross-validation (contiguous folds)" % folds)
    print("  %-6s %20s %30s" % ("", "Part (c) model", "Median of training targets"))
    print("  %-6s %9s %10s %14s %10s %13s" % ("fold", "NMAE", "NMSE", "NMAE", "NMSE", "median"))
    for k in range(folds):
        print("  %-6d %9.4f %10.4f %14.4f %10.4f %13.2f"
              % (k + 1, scores[k, 0], scores[k, 1], scores[k, 2], scores[k, 3], scores[k, 4]))
    means = scores.mean(0)
    stds = scores.std(0)
    print("  %-6s %9.4f %10.4f %14.4f %10.4f %13.2f"
          % ("mean", means[0], means[1], means[2], means[3], means[4]))
    print("  %-6s %9.4f %10.4f %14.4f %10.4f" % ("std", stds[0], stds[1], stds[2], stds[3]))
    print("")
    print("  Pooled over all out-of-fold predictions")
    print("  %-34s %9s %10s" % ("Predictor", "NMAE", "NMSE"))
    print("  %-34s %9.4f %10.4f"
          % ("Part (c) model, %d-fold CV" % folds, nmae(y, oofPred), nmse(y, oofPred)))
    print("  %-34s %9.4f %10.4f"
          % ("Median of training targets", nmae(y, oofMedian), nmse(y, oofMedian)))
    print("  (median of all training targets: %.2f bpm)" % np.median(y))
    print("")
    sys.stdout.flush()
    return scores, oofPred, oofMedian


def main():
    if len(sys.argv) != 4:
        sys.stderr.write("usage: part_c_pruned_cv.py train.csv test.csv predictions.txt\n")
        return 2
    trainPath, testPath, outPath = sys.argv[1:4]

    bvp, acc, eda, trainY = readSignals(trainPath)
    if trainY is None:
        raise ValueError("train.csv must contain an 'hr' column")

    hist, _ = np.histogram(trainY, bins=np.append(candidateBpm - 0.25, candidateBpm[-1] + 0.25),
                           density=True)
    window = np.hanning(21)
    prior = np.convolve(hist, window / window.sum(), mode="same") + 1e-6
    logPrior = logSafe(prior / prior.sum())[None, :].astype(np.float32)

    trainFeatures, names = buildFeaturesChunked(bvp, acc, eda, logPrior, tag="train")
    del bvp, acc, eda
    gc.collect()
    log("train scalars %s" % (trainFeatures.shape,))

    bvp, acc, eda, _ = readSignals(testPath)
    testCount = len(bvp)
    testFeatures, _ = buildFeaturesChunked(bvp, acc, eda, logPrior, tag="test")
    del bvp, acc, eda
    gc.collect()

    centres, widths = fitBumpKnots(trainFeatures)

    def buildDesign(features):
        return [features, expandToBumps(features, centres, widths)]

    trainParts = buildDesign(trainFeatures)
    del trainFeatures
    gc.collect()
    blockSizes = [part.shape[1] for part in trainParts]
    blockOf = np.concatenate([[i] * s for i, s in enumerate(blockSizes)])
    trainX = np.hstack(trainParts)
    del trainParts
    gc.collect()
    log("design matrix %s  block sizes %s" % (trainX.shape, blockSizes))

    mean = trainX.mean(0)
    std = trainX.std(0) + np.float32(1e-8)
    trainX = ((trainX - mean) / std).astype(np.float32)

    if cvEnabled:
        crossValidate(trainX, trainY, blockOf, len(blockSizes), cvFolds)
        gc.collect()

    log("final fit on all training rows")
    w, intercept = fitBlockRidge(trainX, trainY, blockOf, len(blockSizes))
    del trainX
    gc.collect()

    testParts = buildDesign(testFeatures)
    del testFeatures
    gc.collect()
    testX = np.hstack(testParts)
    del testParts
    gc.collect()
    testX = ((testX - mean) / std).astype(np.float32)
    pred = (testX @ w.astype(np.float32) + intercept).astype(np.float64)
    del testX
    gc.collect()

    pred = clipPredictions(pred, trainY, intercept)
    assert len(pred) == testCount, "prediction count mismatch"
    with open(outPath, "w") as f:
        f.write("\n".join("%.6f" % v for v in pred))
        f.write("\n")
    log("wrote %d predictions to %s" % (testCount, outPath))
    return 0


if __name__ == "__main__":
    sys.exit(main())
