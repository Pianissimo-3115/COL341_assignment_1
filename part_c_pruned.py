import gc
import os
import sys
import time

import numpy as np
import pandas as pd

fastMode = os.environ.get("PARTC_FAST", "0") == "1"

startTime = time.time()


def logProgress(message):
    sys.stderr.write("[%7.1fs] %s\n" % (time.time() - startTime, message))
    sys.stderr.flush()


blocksPerWindow, valuesPerBlock = 10, 164
bvpColumns = np.concatenate([np.arange(valuesPerBlock * block + 96, valuesPerBlock * block + 160)
                             for block in range(blocksPerWindow)])
accColumnsPerAxis = [np.concatenate([np.arange(valuesPerBlock * block + 32 * axis,
                                               valuesPerBlock * block + 32 * (axis + 1))
                                     for block in range(blocksPerWindow)]) for axis in range(3)]
edaColumns = np.concatenate([np.arange(valuesPerBlock * block + 160, valuesPerBlock * block + 164)
                             for block in range(blocksPerWindow)])

bvpSampleRateHz, accSampleRateHz = 64.0, 32.0
accSamplesPerWindow = 320

candidateBpm = np.arange(36.0, 216.0, 0.5)
candidateCount = len(candidateBpm)
candidateBpm32 = candidateBpm.astype(np.float32)
candidateHz = candidateBpm / 60.0                       
def logSafe(value):
    return np.log(np.maximum(value, 1e-12)).astype(np.float32)


def cosSinProjection(sampleCount, sampleRateHz, targetHz):
    timeSeconds = np.arange(sampleCount) / sampleRateHz
    hannWindow = np.hanning(sampleCount)
    angle = 2.0 * np.pi * np.outer(timeSeconds, targetHz)
    return ((np.cos(angle) * hannWindow[:, None]).astype(np.float32),
            (np.sin(angle) * hannWindow[:, None]).astype(np.float32))


subWindowLengths = (640, 320, 256, 160, 128, 80)
bvpProjectionByLength = {length: cosSinProjection(length, bvpSampleRateHz, candidateHz)
                         for length in subWindowLengths}
accCosProjection, accSinProjection = cosSinProjection(accSamplesPerWindow, accSampleRateHz, candidateHz)
autocorrelationKernel = np.cos(2.0 * np.pi * candidateHz[:, None] * (60.0 / candidateBpm)[None, :]).astype(np.float32)


def movingAverage(rows, windowLength):
    if windowLength <= 1:
        return rows
    leftPad = windowLength // 2
    padded = np.pad(rows, ((0, 0), (leftPad, windowLength - 1 - leftPad)), mode="edge")
    cumulative = np.cumsum(padded, axis=1, dtype=np.float32)
    cumulative = np.concatenate([np.zeros((len(rows), 1), np.float32), cumulative], 1)
    return (cumulative[:, windowLength:] - cumulative[:, :-windowLength]) * np.float32(1.0 / windowLength)


def bandPass(rows, shortWindow, longWindow):
    return movingAverage(rows, shortWindow) - movingAverage(rows, longWindow)


def powerSpectrum(rows, cosBasis, sinBasis):
    centred = rows - rows.mean(1, keepdims=True)
    return (centred @ cosBasis) ** 2 + (centred @ sinBasis) ** 2


def normaliseRows(rows):
    return rows / (rows.sum(1, keepdims=True) + np.float32(1e-20))


def softmaxRows(scores, temperature):
    scaled = (scores - scores.max(1, keepdims=True)) * np.float32(1.0 / temperature)
    exponentials = np.exp(scaled, dtype=np.float32)
    return exponentials / (exponentials.sum(1, keepdims=True) + np.float32(1e-20))


def autoregressiveSpectrum(rows, order, ridgeFactor=1e-3):
    normalised = rows - rows.mean(1, keepdims=True)
    normalised = normalised / (normalised.std(1, keepdims=True) + np.float32(1e-6))
    rowCount, length = normalised.shape
    autocovariance = np.empty((rowCount, order + 1), np.float32)
    for lagIndex in range(order + 1):
        autocovariance[:, lagIndex] = np.einsum("ij,ij->i", normalised[:, :length - lagIndex], normalised[:, lagIndex:]) / length
    toeplitzIndex = np.abs(np.arange(order)[:, None] - np.arange(order)[None, :])
    toeplitzMatrix = autocovariance[:, toeplitzIndex].copy()                                   
    toeplitzMatrix[:, np.arange(order), np.arange(order)] *= np.float32(1.0 + ridgeFactor)
    arCoefficients = np.linalg.solve(toeplitzMatrix, autocovariance[:, 1:order + 1][..., None])[..., 0]
    innovationVariance = np.maximum(autocovariance[:, 0] - (arCoefficients * autocovariance[:, 1:order + 1]).sum(1), 1e-6)
    lagIndex = np.arange(1, order + 1)
    angle = 2.0 * np.pi * np.outer(candidateHz / bvpSampleRateHz, lagIndex)
    cosTerms = np.ascontiguousarray(np.cos(angle).T.astype(np.float32))
    sinTerms = np.ascontiguousarray(np.sin(angle).T.astype(np.float32))
    return (innovationVariance[:, None] / ((1 - arCoefficients @ cosTerms) ** 2 + (arCoefficients @ sinTerms) ** 2 + 1e-9)).astype(np.float32)


def peakBpmSubBin(surface):
    peakIndex = np.argmax(surface, 1)
    rowIndex = np.arange(len(surface))
    clampedIndex = np.clip(peakIndex, 1, surface.shape[1] - 2)
    leftScore, centreScore, rightScore = surface[rowIndex, clampedIndex - 1], surface[rowIndex, clampedIndex], surface[rowIndex, clampedIndex + 1]
    curvature = leftScore - 2 * centreScore + rightScore
    wellFormed = np.abs(curvature) > 1e-20
    subBinOffset = np.clip(np.where(wellFormed, 0.5 * (leftScore - rightScore) / np.where(wellFormed, curvature, 1.0), 0.0), -1, 1)
    return (candidateBpm32[clampedIndex] + subBinOffset * np.float32(candidateBpm[1] - candidateBpm[0])).astype(np.float32), surface[rowIndex, peakIndex].astype(np.float32)


def strongestPeaks(surface, peakCount=3, guardBpm=10.0):
    masked = surface.copy()
    peakBpms, peakScores = [], []
    for _ in range(peakCount):
        peakBpm, peakScore = peakBpmSubBin(masked)
        peakBpms.append(peakBpm)
        peakScores.append(peakScore)
        masked = np.where(np.abs(candidateBpm32[None, :] - peakBpm[:, None]) < np.float32(guardBpm),
                      np.float32(-1e30), masked)
    return np.stack(peakBpms, 1), np.stack(peakScores, 1)


def lookupSharedBpm(surface, targetBpm):
    position = np.clip((np.asarray(targetBpm) - candidateBpm[0]) / (candidateBpm[1] - candidateBpm[0]), 0, candidateCount - 1.001)
    lowerIndex = position.astype(np.int32)
    fraction = (position - lowerIndex).astype(np.float32)
    return surface[:, lowerIndex] * (1 - fraction) + surface[:, lowerIndex + 1] * fraction


def lookupPerRowBpm(surface, targetBpm):
    position = np.clip((np.asarray(targetBpm) - candidateBpm[0]) / (candidateBpm[1] - candidateBpm[0]), 0, candidateCount - 1.001)
    lowerIndex = position.astype(np.int32)
    fraction = (position - lowerIndex).astype(np.float32)
    rowIndex = np.arange(len(surface))
    return (surface[rowIndex, lowerIndex] * (1 - fraction) + surface[rowIndex, lowerIndex + 1] * fraction).astype(np.float32)


def appendNamedColumns(columns, columnNames, prefix, namedValues):
    for suffix, values in namedValues:
        columns.append(np.asarray(values, np.float32)[:, None])
        columnNames.append(prefix + suffix)


def gaussianBumpBasis(values, centres, width):
    distance = (np.asarray(values, np.float32)[:, None] -
         np.asarray(centres, np.float32)[None, :]) * np.float32(1.0 / max(width, 1e-6))
    bump = np.exp(-0.5 * distance * distance)
    return (bump / (bump.sum(1, keepdims=True) + np.float32(1e-9))).astype(np.float32)


subWindowConsensusConfigs = ((320, 64, "g5"), (256, 48, "g4"), (160, 32, "g25"), (128, 32, "g2"), (80, 16, "g125"))
if fastMode:
    subWindowConsensusConfigs = ((320, 64, "g5"), (256, 64, "g4"), (160, 48, "g25"), (128, 48, "g2"), (80, 32, "g125"))

mainScoreWeights = {
    "lP": 0.25, "lPwhite": 0.5, "lAR": -0.25, "lAR2": 4.0, "lR": 0.5, "lRaw": -0.5,
    "g5": 0.5, "g4": -0.5, "g25": 1.0, "g2": 4.0, "g125": 4.0, "g25w": -2.0,
    "prior": 4.0, "lP2": 0.5, "lP3": 0.25, "lPh": 0.25, "lPh3": 0.5,
    "g5_2": -0.5, "g5_h": 1.0,
}
consensusScoreWeights = {"g2": 1.0, "g25": 1.0, "g5": 1.0, "lP": 0.5, "prior": 0.5}
spectralScoreWeights = {"lP": 1.0, "lAR": 1.0, "lR": 0.5}


def buildEvidenceSurfaces(bvp, acc, logRatePrior):
    bvpRaw = np.ascontiguousarray(bvp, np.float32)
    bvpBandPassed = bandPass(bvpRaw, 5, 61)
    surfaces = {}

    spectrum = normaliseRows(powerSpectrum(bvpBandPassed, *bvpProjectionByLength[640]))
    surfaces["lP"] = logSafe(spectrum)
    surfaces["lPwhite"] = surfaces["lP"] - movingAverage(surfaces["lP"], 61)      
    surfaces["lAR"] = logSafe(normaliseRows(autoregressiveSpectrum(bvpBandPassed, 32)))
    surfaces["lAR2"] = logSafe(normaliseRows(autoregressiveSpectrum(bvpBandPassed, 20)))
    autocorrelation = spectrum @ autocorrelationKernel                                            
    surfaces["lR"] = logSafe(normaliseRows(np.maximum(autocorrelation - autocorrelation.min(1, keepdims=True), 1e-9)))
    surfaces["lRaw"] = logSafe(normaliseRows(powerSpectrum(bvpRaw, *bvpProjectionByLength[640])))               

    accRaw = np.ascontiguousarray(acc, np.float32)
    accSpectrum = normaliseRows(powerSpectrum(accRaw[:, 0], accCosProjection, accSinProjection) + powerSpectrum(accRaw[:, 1], accCosProjection, accSinProjection) + powerSpectrum(accRaw[:, 2], accCosProjection, accSinProjection))
    surfaces["lPA"] = logSafe(accSpectrum)
    surfaces["prior"] = np.repeat(logRatePrior, len(bvpRaw), 0)

    for length, hop, name in subWindowConsensusConfigs:                            
        cosBasis, sinBasis = bvpProjectionByLength[length]
        logSpectrumSum = None
        subWindowCount = 0
        for start in range(0, bvpBandPassed.shape[1] - length + 1, hop):
            logSpectrum = logSafe(normaliseRows(powerSpectrum(bvpBandPassed[:, start:start + length], cosBasis, sinBasis)))
            logSpectrumSum = logSpectrum if logSpectrumSum is None else logSpectrumSum + logSpectrum
            subWindowCount += 1
        surfaces[name] = logSpectrumSum * np.float32(1.0 / subWindowCount)
    surfaces["g25w"] = surfaces["g25"] - movingAverage(surfaces["g25"], 61)

    surfaces["lP2"] = lookupSharedBpm(surfaces["lP"], candidateBpm * 2)
    surfaces["lP3"] = lookupSharedBpm(surfaces["lP"], candidateBpm * 3)
    surfaces["lPh"] = lookupSharedBpm(surfaces["lP"], candidateBpm / 2)
    surfaces["lPh3"] = lookupSharedBpm(surfaces["lP"], candidateBpm / 3)
    surfaces["g5_2"] = lookupSharedBpm(surfaces["g5"], candidateBpm * 2)
    surfaces["g5_h"] = lookupSharedBpm(surfaces["g5"], candidateBpm / 2)
    return surfaces, bvpBandPassed, spectrum, accSpectrum


def combineSurfaces(surfaces, weights):
    total = None
    for name, weight in weights.items():
        if weight and name in surfaces:
            total = np.float32(weight) * surfaces[name] if total is None else total + np.float32(weight) * surfaces[name]
    total = total - total.mean(1, keepdims=True)
    return total / (total.std(1, keepdims=True) + np.float32(1e-6))


def describeSurface(surface, tag, columns, columnNames):
    peakBpm, _ = peakBpmSubBin(surface)
    distribution = softmaxRows(surface, 1.0)
    centroidBpm = distribution @ candidateBpm32
    cumulative = np.cumsum(distribution, 1)
    medianBpm = candidateBpm32[np.argmax(cumulative >= 0.5, 1)]
    lowerQuartileBpm = candidateBpm32[np.argmax(cumulative >= 0.25, 1)]
    upperQuartileBpm = candidateBpm32[np.argmax(cumulative >= 0.75, 1)]
    entropy = -(distribution * np.log(distribution + 1e-12)).sum(1)
    spreadBpm = np.sqrt(np.maximum(distribution @ (candidateBpm32 ** 2) - centroidBpm ** 2, 0))
    peakBpms, peakScores = strongestPeaks(surface, 3, 10.0)
    appendNamedColumns(columns, columnNames, tag + "_",
         (("pk", peakBpm), ("p2", peakBpms[:, 1]), ("p3", peakBpms[:, 2]),
          ("m12", peakScores[:, 0] - peakScores[:, 1]), ("m13", peakScores[:, 0] - peakScores[:, 2]),
          ("d12", peakBpms[:, 1] - peakBpm), ("d13", peakBpms[:, 2] - peakBpm),
          ("cen", centroidBpm), ("med", medianBpm), ("q25", lowerQuartileBpm), ("q75", upperQuartileBpm),
          ("iqr", upperQuartileBpm - lowerQuartileBpm), ("ent", entropy), ("sd", spreadBpm)))
    return peakBpm, peakScores, entropy


def addBvpWaveformFeatures(bvp, bvpBandPassed, columns, columnNames):
    bvpRaw = np.ascontiguousarray(bvp, np.float32)
    bvpWideBand = bandPass(bvpRaw, 9, 81)
    pulseAmplitude = bvpBandPassed.std(1) + np.float32(1e-6)
    standardised = bvpBandPassed / pulseAmplitude[:, None]
    firstDifference = np.diff(bvpBandPassed, axis=1)
    perSecondAmplitude = bvpBandPassed.reshape(len(bvpBandPassed), 10, 64).std(2)   
    appendNamedColumns(columns, columnNames, "t_", (
        ("zc", (np.diff(np.signbit(bvpBandPassed), axis=1).sum(1) / 2.0) * 6.0),
        ("zc2", (np.diff(np.signbit(bvpWideBand), axis=1).sum(1) / 2.0) * 6.0),
        ("lamp", np.log(pulseAmplitude)), ("lraw", np.log(bvpRaw.std(1) + 1e-6)),
        ("lmad", np.log(np.abs(bvpBandPassed).mean(1) + 1e-6)),
        ("kurt", (standardised ** 4).mean(1)), ("skew", (standardised ** 3).mean(1)),
        ("ldiff", np.log(np.abs(firstDifference).mean(1) + 1e-6)),
        ("crest", np.abs(bvpBandPassed).max(1) / pulseAmplitude),
        ("lrange", np.log(bvpRaw.max(1) - bvpRaw.min(1) + 1e-6)),
        ("nfrac", (np.abs(standardised) > 3).mean(1)),
        ("envcv", perSecondAmplitude.std(1) / pulseAmplitude),
        ("envmax", np.log(perSecondAmplitude.max(1) + 1e-6)),
        ("envmin", np.log(perSecondAmplitude.min(1) + 1e-6))))
    return np.log(pulseAmplitude).astype(np.float32)


def addMotionFeatures(acc, bvpBandPassed, columns, columnNames):
    accRaw = np.ascontiguousarray(acc, np.float32)
    magnitude = np.sqrt((accRaw ** 2).sum(1))
    magnitudeBandPassed = bandPass(magnitude, 3, 41)
    axisStd = accRaw.std(2)
    axisMean = accRaw.mean(2)
    axisMeanTotal = np.abs(axisMean).sum(1) + np.float32(1e-3)
    appendNamedColumns(columns, columnNames, "a_", (
        ("magm", magnitude.mean(1)), ("magsd", magnitude.std(1)),
        ("lmagsd", np.log(magnitude.std(1) + 1e-3)),
        ("lbpsd", np.log(magnitudeBandPassed.std(1) + 1e-3)),
        ("sdx", axisStd[:, 0]), ("sdy", axisStd[:, 1]), ("sdz", axisStd[:, 2]),
        ("lsd", np.log(axisStd.sum(1) + 1e-3)),
        ("mx", axisMean[:, 0]), ("my", axisMean[:, 1]), ("mz", axisMean[:, 2]),
        ("nx", axisMean[:, 0] / axisMeanTotal), ("ny", axisMean[:, 1] / axisMeanTotal),
        ("nz", axisMean[:, 2] / axisMeanTotal),
        ("magrange", magnitude.max(1) - magnitude.min(1)),
        ("jerk", np.log(np.abs(np.diff(magnitude, axis=1)).mean(1) + 1e-3)),
        ("q90", np.percentile(np.abs(magnitudeBandPassed), 90, axis=1)),
        ("still", (np.abs(magnitudeBandPassed) < 2).mean(1))))
    bvpEnvelope = movingAverage(np.abs(bvpBandPassed), 33)[:, ::2][:, :accSamplesPerWindow]
    envelopeCentred = bvpEnvelope - bvpEnvelope.mean(1, keepdims=True)
    motionCentred = magnitudeBandPassed - magnitudeBandPassed.mean(1, keepdims=True)
    appendNamedColumns(columns, columnNames, "x_", (("envacc", (envelopeCentred * motionCentred).sum(1) /
                             (np.sqrt((envelopeCentred ** 2).sum(1) * (motionCentred ** 2).sum(1)) + 1e-6)),))
    return np.log(axisStd.sum(1) + 1e-3).astype(np.float32)


def addEdaFeatures(eda, columns, columnNames):
    edaRaw = np.ascontiguousarray(eda, np.float32)
    normalisedTime = np.arange(edaRaw.shape[1], dtype=np.float32)
    normalisedTime = (normalisedTime - normalisedTime.mean()) / normalisedTime.std()
    appendNamedColumns(columns, columnNames, "e_", (
        ("lmean", np.log(edaRaw.mean(1) + 1e-3)), ("lstd", np.log(edaRaw.std(1) + 1e-4)),
        ("slope", (edaRaw * normalisedTime).mean(1)), ("zero", (edaRaw.max(1) <= 1e-6).astype(np.float32)),
        ("lrng", np.log(edaRaw.max(1) - edaRaw.min(1) + 1e-4))))


def buildScalarFeatures(bvp, acc, eda, logRatePrior):
    columns, columnNames = [], []
    surfaces, bvpBandPassed, _, _ = buildEvidenceSurfaces(bvp, acc, logRatePrior)

    mainScore = combineSurfaces(surfaces, mainScoreWeights)
    mainRateBpm, _, _ = describeSurface(mainScore, "S", columns, columnNames)
    consensusScore = combineSurfaces(surfaces, consensusScoreWeights)
    consensusRateBpm, _, _ = describeSurface(consensusScore, "S2", columns, columnNames)
    spectralScore = combineSurfaces(surfaces, spectralScoreWeights)
    spectralRateBpm, _, _ = describeSurface(spectralScore, "S3", columns, columnNames)
    del mainScore, consensusScore, spectralScore

    peakByTag = {}
    for surfaceKey, tag in (("lP", "P"), ("lAR", "AR"), ("g2", "G2"), ("g25", "G25"),
                     ("g5", "G5"), ("lR", "R"), ("lPA", "AC")):
        peakByTag[tag], _, _ = describeSurface(surfaces[surfaceKey], tag, columns, columnNames)
    periodogramRateBpm, autoregressiveRateBpm, consensus2sRateBpm = peakByTag["P"], peakByTag["AR"], peakByTag["G2"]
    consensus25sRateBpm, consensus5sRateBpm, autocorrelationRateBpm = peakByTag["G25"], peakByTag["G5"], peakByTag["R"]

    rateEstimates = np.stack([mainRateBpm, consensusRateBpm, spectralRateBpm, periodogramRateBpm, autoregressiveRateBpm, consensus2sRateBpm, consensus25sRateBpm, consensus5sRateBpm, autocorrelationRateBpm], 1)
    columns.append(rateEstimates.std(1, keepdims=True)); columnNames.append("x_std")
    columns.append(np.median(rateEstimates, 1, keepdims=True).astype(np.float32)); columnNames.append("x_med")
    columns.append(rateEstimates.mean(1, keepdims=True)); columnNames.append("x_mean")
    for estimatorIndex, name in enumerate(("P", "AR", "g2", "g25", "g5", "R")):
        columns.append(np.abs(rateEstimates[:, 3 + estimatorIndex] - mainRateBpm)[:, None]); columnNames.append("x_dev_" + name)
    estimatorAgreement = (np.abs(rateEstimates - mainRateBpm[:, None]) < 4).mean(1).astype(np.float32)
    columns.append(estimatorAgreement[:, None]); columnNames.append("x_agree")
    for harmonicMultiple, name in ((1.0, "f"), (2.0, "2f"), (0.5, "hf"), (3.0, "3f")):
        columns.append(lookupPerRowBpm(surfaces["lP"], mainRateBpm * harmonicMultiple)[:, None]); columnNames.append("sp_" + name)
        columns.append(lookupPerRowBpm(surfaces["g2"], mainRateBpm * harmonicMultiple)[:, None]); columnNames.append("sg_" + name)
    del surfaces

    addBvpWaveformFeatures(bvp, bvpBandPassed, columns, columnNames)
    addMotionFeatures(acc, bvpBandPassed, columns, columnNames)
    addEdaFeatures(eda, columns, columnNames)

    return np.hstack(columns).astype(np.float32), columnNames


def buildScalarFeaturesChunked(bvp, acc, eda, logRatePrior, chunkRows=8192, tag=""):
    scalarChunks, columnNames = [], None
    rowCount = len(bvp)
    for start in range(0, rowCount, chunkRows):
        stop = min(start + chunkRows, rowCount)
        scalars, chunkColumnNames = buildScalarFeatures(bvp[start:stop], acc[start:stop], eda[start:stop], logRatePrior)
        scalarChunks.append(scalars)
        if columnNames is None:
            columnNames = chunkColumnNames
        del scalars
        gc.collect()
        logProgress("  %s features %d/%d" % (tag, stop, rowCount))
    return np.concatenate(scalarChunks), columnNames


bumpsPerScalar = 7 if fastMode else 9
def fitScalarBumpKnots(scalars):
    knotCentres = [np.percentile(scalars[:, column], np.linspace(2, 98, bumpsPerScalar)) for column in range(scalars.shape[1])]
    knotWidths = [max((knots[-1] - knots[0]) / (bumpsPerScalar - 1), 1e-6) for knots in knotCentres]
    return knotCentres, knotWidths


def expandScalarsToBumps(scalars, knotCentres, knotWidths):
    return np.hstack([gaussianBumpBasis(scalars[:, column], knotCentres[column], knotWidths[column]) for column in range(scalars.shape[1])]).astype(np.float32)


def readSignalsCsv(path, chunkRows=20000):
    bvpChunks, accChunks, edaChunks, targetChunks = [], [], [], []
    rowCount = 0
    for frame in pd.read_csv(path, chunksize=chunkRows, dtype=np.float32):
        target = frame["hr"].to_numpy(np.float32) if "hr" in frame.columns else None
        if target is not None:
            frame = frame.drop(columns=["hr"])
        values = frame.to_numpy(np.float32)
        del frame
        if values.shape[1] != blocksPerWindow * valuesPerBlock:
            raise ValueError("expected %d feature columns, got %d" % (blocksPerWindow * valuesPerBlock, values.shape[1]))
        bvpChunks.append(values[:, bvpColumns])
        accChunks.append(np.stack([values[:, accColumnsPerAxis[0]], values[:, accColumnsPerAxis[1]], values[:, accColumnsPerAxis[2]]], 1))
        edaChunks.append(values[:, edaColumns])
        if target is not None:
            targetChunks.append(target)
        rowCount += len(values)
        del values
        gc.collect()
    logProgress("read %s: %d rows" % (path, rowCount))
    return (np.concatenate(bvpChunks), np.concatenate(accChunks), np.concatenate(edaChunks),
            np.concatenate(targetChunks).astype(np.float64) if targetChunks else None)


ridgePenalties = 10.0 ** np.arange(-0.5, 6.01, 0.25)
blockPenaltyPresets = ((0.1, 1.0),
           (1.0, 1.0),
           (0.03, 0.3),
           (0.01, 30.0))
if fastMode:
    blockPenaltyPresets = blockPenaltyPresets[:2]


def normalisedMeanAbsError(target, prediction):
    return np.abs(target - prediction).sum() / np.abs(target - target.mean()).sum()


def bestPenaltyOnRidgePath(gram, rhs, columnScale, validDesign, validTarget, targetMean):
    eigenvalues, eigenvectors = np.linalg.eigh(gram * columnScale[None, :] * columnScale[:, None])
    projectedRhs = eigenvectors.T @ (rhs * columnScale)
    projectedValid = (validDesign * columnScale[None, :].astype(np.float32)) @ eigenvectors.astype(np.float32)
    best = None
    for penalty in ridgePenalties:
        scaledWeights = projectedRhs / (eigenvalues + penalty)
        prediction = projectedValid @ scaledWeights.astype(np.float32) + targetMean
        score = normalisedMeanAbsError(validTarget, prediction)
        if best is None or score < best[0]:
            best = (score, penalty, (eigenvectors @ scaledWeights) * columnScale)
    return best


def normalEquations(design, target):
    return (design.T @ design).astype(np.float64), (design.T @ target).astype(np.float64)


def fitBlockRidgeThenIrls(design, target, blockIdOfColumn, blockCount):
    rowCount = len(design)
    splitIndex = int(rowCount * 0.75)
    splitTargetMean = target[:splitIndex].mean()
    logProgress("  gram (internal split)")
    trainPart = design[:splitIndex]
    gram, rhs = normalEquations(trainPart, (target[:splitIndex] - splitTargetMean).astype(np.float32))
    del trainPart
    gc.collect()
    validDesign, validTarget = design[splitIndex:], target[splitIndex:]
    best = None
    for presetIndex, preset in enumerate(blockPenaltyPresets):
        multipliers = np.asarray(preset, np.float64)[:blockCount]
        columnScale = (1.0 / np.sqrt(multipliers))[blockIdOfColumn]              
        score, penalty, weights = bestPenaltyOnRidgePath(gram, rhs, columnScale, validDesign, validTarget, splitTargetMean)
        logProgress("    preset %d  internal NMAE %.4f (alpha=%.3g)" % (presetIndex, score, penalty))
        if best is None or score < best[0]:
            best = (score, penalty, preset)
    del gram, rhs, validDesign
    gc.collect()
    logProgress("  best preset %s alpha %.3g internal NMAE %.4f" % (best[2], best[1], best[0]))

    multipliers = np.asarray(best[2], np.float64)[:blockCount]
    columnScale = (1.0 / np.sqrt(multipliers))[blockIdOfColumn]
    penalty = best[1]
    targetMean = target.mean()
    centredTarget = (target - targetMean).astype(np.float32)
    scaledDesign = (design * columnScale[None, :].astype(np.float32))
    gram, rhs = normalEquations(scaledDesign, centredTarget)
    columnCount = gram.shape[0]
    weights = np.linalg.solve(gram + penalty * np.eye(columnCount), rhs)
    logProgress("  L2 fit done")

    bestWeights, bestScore = weights, normalisedMeanAbsError(target[splitIndex:], scaledDesign[splitIndex:] @ weights.astype(np.float32) + targetMean)
    for iteration in range(3):
        residual = scaledDesign @ weights.astype(np.float32) - centredTarget
        huberFloor = np.float32(1.345 * np.median(np.abs(residual - np.median(residual))) + 1e-3)
        irlsWeights = np.sqrt(1.0 / np.maximum(np.abs(residual), huberFloor)).astype(np.float32)
        reweightedDesign = scaledDesign * irlsWeights[:, None]
        reweightedGram, reweightedRhs = normalEquations(reweightedDesign, centredTarget * irlsWeights)
        del reweightedDesign
        gc.collect()
        weights = np.linalg.solve(reweightedGram + penalty * float((irlsWeights ** 2).mean()) * np.eye(columnCount), reweightedRhs)
        score = normalisedMeanAbsError(target[splitIndex:], scaledDesign[splitIndex:] @ weights.astype(np.float32) + targetMean)
        logProgress("    irls %d internal NMAE %.4f" % (iteration, score))
        if score < bestScore:
            bestScore, bestWeights = score, weights
        del reweightedGram, reweightedRhs
        gc.collect()
    del scaledDesign
    gc.collect()
    return bestWeights * columnScale, targetMean


def main():
    if len(sys.argv) != 4:
        sys.stderr.write("usage: part_c_pruned.py train.csv test.csv predictions.txt\n")
        return 2
    trainPath, testPath, predictionsPath = sys.argv[1:4]

    bvp, acc, eda, trainTarget = readSignalsCsv(trainPath)
    if trainTarget is None:
        raise ValueError("train.csv must contain an 'hr' column")

    histogram, _ = np.histogram(trainTarget, bins=np.append(candidateBpm - 0.25, candidateBpm[-1] + 0.25), density=True)
    smoothingWindow = np.hanning(21)
    smoothedPrior = np.convolve(histogram, smoothingWindow / smoothingWindow.sum(), mode="same") + 1e-6
    logRatePrior = logSafe(smoothedPrior / smoothedPrior.sum())[None, :].astype(np.float32)

    trainScalars, columnNames = buildScalarFeaturesChunked(bvp, acc, eda, logRatePrior, tag="train")
    del bvp, acc, eda
    gc.collect()
    logProgress("train scalars %s" % (trainScalars.shape,))

    bvp, acc, eda, _ = readSignalsCsv(testPath)
    testRows = len(bvp)
    testScalars, _ = buildScalarFeaturesChunked(bvp, acc, eda, logRatePrior, tag="test")
    del bvp, acc, eda
    gc.collect()

    knotCentres, knotWidths = fitScalarBumpKnots(trainScalars)

    def buildDesign(scalars):
        return [scalars, expandScalarsToBumps(scalars, knotCentres, knotWidths)]

    trainParts = buildDesign(trainScalars)
    del trainScalars
    gc.collect()
    blockSizes = [part.shape[1] for part in trainParts]
    blockIdOfColumn = np.concatenate([[i] * s for i, s in enumerate(blockSizes)])
    trainDesign = np.hstack(trainParts)
    del trainParts
    gc.collect()
    logProgress("design matrix %s  block sizes %s" % (trainDesign.shape, blockSizes))

    designMean = trainDesign.mean(0)
    designStd = trainDesign.std(0) + np.float32(1e-8)
    trainDesign = ((trainDesign - designMean) / designStd).astype(np.float32)

    weights, targetMean = fitBlockRidgeThenIrls(trainDesign, trainTarget, blockIdOfColumn, len(blockSizes))
    del trainDesign
    gc.collect()

    testParts = buildDesign(testScalars)
    del testScalars
    gc.collect()
    testDesign = np.hstack(testParts)
    del testParts
    gc.collect()
    testDesign = ((testDesign - designMean) / designStd).astype(np.float32)
    prediction = (testDesign @ weights.astype(np.float32) + targetMean).astype(np.float64)
    del testDesign
    gc.collect()

    lowerBound, upperBound = trainTarget.min() - 15.0, trainTarget.max() + 15.0
    prediction = np.clip(np.nan_to_num(prediction, nan=targetMean, posinf=upperBound, neginf=lowerBound), max(lowerBound, 25.0), min(upperBound, 230.0))
    assert len(prediction) == testRows, "prediction count mismatch"
    with open(predictionsPath, "w") as handle:
        handle.write("\n".join("%.6f" % value for value in prediction))
        handle.write("\n")
    logProgress("wrote %d predictions to %s" % (testRows, predictionsPath))
    return 0


if __name__ == "__main__":
    sys.exit(main())
