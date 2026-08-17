#!/usr/bin/env python3
"""
COL774 Assignment 1 - Part (d): CGM glucose estimation from wearable signals.

    python3 part_d.py train              d1 train_d1/ model_d1.pkl
    python3 part_d.py feature_engineering d1 test_d1/  model_d1.pkl features_d1.npy

The evaluator computes  Yhat = intercept + Z_test @ coef  with no clipping and
no inverse transform, so everything this script learns has to live inside one
intercept and one coefficient vector.

Protocol-aware modelling
------------------------
The three protocols differ in what generalisation is being asked for, so the
validation scheme (and therefore the selected model) differs:

  d1  random within-subject   -> KFold(shuffle).  The same participant appears
                                 in train and test, so absolute per-person
                                 levels (skin temperature, EDA baseline, resting
                                 HR) are legitimately predictive and are kept.
  d2  temporal within-subject -> a per-subject time-ordered split: validate on
                                 each subject's latest windows, exactly mirroring
                                 the real protocol.
  d3  cross-subject           -> leave-one-subject-out.  Absolute per-person
                                 levels now actively mislead, so d3 additionally
                                 evaluates a WITHIN-SUBJECT-CENTRED variant:
                                 features are centred per training subject, which
                                 makes the fit learn within-person relationships.
                                 Prediction stays exactly ybar + w.(z - zbar),
                                 i.e. still one intercept and one coefficient
                                 vector, so it remains a linear model.

subject_id and timestamp are used only to build these validation splits, never
as predictive features - they are dropped before the design matrix is formed.

The graded metric is NMAE, so after the L2 fit the model is refit by IRLS
towards an L1 objective (an L1-fitted linear model targets the conditional
median, which is what an absolute-error metric rewards).

Permitted libraries only: numpy, pandas, scipy, scikit-learn (all explicitly
allowed for part (d)).  No external data or pretrained models are used.
"""

import gc
import glob
import os
import pickle
import sys
import time

import numpy as np

import part_d_features as PDF

T0 = time.time()
FORMAT_VERSION = 1
MAX_FEATURES = 480          # spec allows < 500; leave headroom
RNG = np.random.RandomState(0)


def log(msg):
    sys.stderr.write("[%7.1fs] %s\n" % (time.time() - T0, msg))
    sys.stderr.flush()


# ------------------------------------------------------------------ loading
def list_subject_files(directory):
    """Must match eval_d.py exactly or labels will not line up with rows."""
    paths = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not paths:
        raise SystemExit("no .npz files found in %s" % directory)
    return paths


def load_directory(directory):
    Xs, ys, sids, tss, names = [], [], [], [], None
    for path in list_subject_files(directory):
        X, y, sid, ts, nm = PDF.features_for_file(path)
        if names is None:
            names = nm
        elif nm != names:
            common = [c for c in names if c in set(nm)]
            idx_new = {c: i for i, c in enumerate(nm)}
            X = X[:, [idx_new[c] for c in common]]
            for k in range(len(Xs)):
                idx_old = {c: i for i, c in enumerate(names)}
                Xs[k] = Xs[k][:, [idx_old[c] for c in common]]
            names = common
        Xs.append(X)
        ys.append(y if y is not None else np.full(len(X), np.nan))
        sids.append(sid)
        tss.append(ts)
        log("  %s: %d windows" % (os.path.basename(path), len(X)))
        gc.collect()
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    sid = np.concatenate(sids)
    ts = np.concatenate(tss)
    return X, y, sid, ts, names


# ----------------------------------------------------------- preprocessing
def fit_preprocess(X, names):
    """Impute -> drop degenerate -> winsorise -> standardise.  All statistics
    come from the training matrix only and are stored for reuse at test time."""
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med)

    lo = np.percentile(Xi, 0.5, axis=0)
    hi = np.percentile(Xi, 99.5, axis=0)
    span = hi - lo
    hi = np.where(span > 0, hi, lo + 1.0)
    Xw = np.clip(Xi, lo, hi)

    sd = Xw.std(0)
    keep = np.where(sd > 1e-9)[0]
    if len(keep) == 0:
        raise SystemExit("all features are constant")
    Xw = Xw[:, keep]
    mu = Xw.mean(0)
    sd = Xw.std(0) + 1e-9
    state = {"median": med, "lo": lo, "hi": hi, "keep": keep, "mu": mu, "sd": sd,
             "names_in": list(names)}
    return (Xw - mu) / sd, state, [names[i] for i in keep]


def apply_preprocess(X, state):
    med, lo, hi = state["median"], state["lo"], state["hi"]
    if X.shape[1] != len(med):
        raise SystemExit("test matrix has %d raw features, model expects %d"
                         % (X.shape[1], len(med)))
    Xi = np.where(np.isfinite(X), X, med)
    Xw = np.clip(Xi, lo, hi)[:, state["keep"]]
    return (Xw - state["mu"]) / state["sd"]


# --------------------------------------------------------------- CV splits
def make_splits(protocol, sid, ts, n, nfold=5):
    """subject_id / timestamp are permitted for validation splitting only."""
    subs = np.unique(sid)
    splits = []
    if protocol == "d3":
        for s in subs:                                   # leave-one-subject-out
            te = np.where(sid == s)[0]
            tr = np.where(sid != s)[0]
            if len(te) and len(tr):
                splits.append((tr, te))
        if len(splits) > 8:
            splits = splits[:8]
    elif protocol == "d2":
        tr, te = [], []
        for s in subs:                                   # latest 25% per subject
            idx = np.where(sid == s)[0]
            idx = idx[np.argsort(ts[idx], kind="stable")]
            cut = max(int(len(idx) * 0.75), 1)
            tr.append(idx[:cut]); te.append(idx[cut:])
        tr, te = np.concatenate(tr), np.concatenate(te)
        if len(te):
            splits.append((tr, te))
        tr2, te2 = [], []                                # a second, earlier cut
        for s in subs:
            idx = np.where(sid == s)[0]
            idx = idx[np.argsort(ts[idx], kind="stable")]
            a, b = max(int(len(idx) * 0.5), 1), max(int(len(idx) * 0.75), 2)
            tr2.append(idx[:a]); te2.append(idx[a:b])
        tr2, te2 = np.concatenate(tr2), np.concatenate(te2)
        if len(te2):
            splits.append((tr2, te2))
    else:                                                # d1: random KFold
        order = RNG.permutation(n)
        folds = np.array_split(order, nfold)
        for k in range(nfold):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(nfold) if j != k])
            splits.append((tr, te))
    if not splits:
        cut = int(n * 0.75)
        splits = [(np.arange(cut), np.arange(cut, n))]
    return splits


# ------------------------------------------------------------------- model
ALPHAS = 10.0 ** np.arange(-1.0, 5.01, 0.5)


def nmae(y, p):
    return float(np.abs(y - p).sum() / (np.abs(y - y.mean()).sum() + 1e-12))


def nmse(y, p):
    return float(((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))


def ridge_path(Xtr, ytr, Xva, alphas):
    """one eigendecomposition gives every alpha."""
    ym = ytr.mean()
    G = Xtr.T @ Xtr
    b = Xtr.T @ (ytr - ym)
    d, V = np.linalg.eigh(G)
    bt = V.T @ b
    Pv = Xva @ V
    for a in alphas:
        yield a, Pv @ (bt / (d + a)) + ym, V, bt, d, ym


def fit_ridge(X, y, alpha):
    ym = y.mean()
    G = X.T @ X + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(G, X.T @ (y - ym))
    return w, ym


def irls_refit(X, y, alpha, w, ym, iters=4):
    """push the L2 solution toward an L1 objective (metric is NMAE)."""
    best_w, best_ym = w, ym
    best = nmae(y, X @ w + ym)
    for _ in range(iters):
        r = y - (X @ w + ym)
        floor = 1.345 * np.median(np.abs(r - np.median(r))) + 1e-6
        sw = 1.0 / np.sqrt(np.maximum(np.abs(r), floor))
        Wm = sw ** 2
        ym_w = float((Wm * y).sum() / Wm.sum())
        Xw = X * sw[:, None]
        G = Xw.T @ Xw + alpha * float((Wm).mean()) * np.eye(X.shape[1])
        w_new = np.linalg.solve(G, Xw.T @ ((y - ym_w) * sw))
        s = nmae(y, X @ w_new + ym_w)
        if s < best:
            best, best_w, best_ym = s, w_new, ym_w
        w, ym = w_new, ym_w
    return best_w, best_ym


def select_features(X, y, names, k):
    """keep the k features with the strongest |correlation| with the target.
    Fitted on training data only; the chosen indices are stored in the model."""
    if X.shape[1] <= k:
        return np.arange(X.shape[1])
    yc = y - y.mean()
    denom = (X.std(0) + 1e-12) * (yc.std() + 1e-12) * len(y)
    corr = np.abs((X * yc[:, None]).sum(0) / denom)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    return np.sort(np.argsort(-corr)[:k])


def subject_center(X, sid):
    """centre each feature within each subject; returns centred X and the grand
    mean that keeps the prediction affine."""
    Xc = X.copy()
    for s in np.unique(sid):
        m = sid == s
        Xc[m] -= Xc[m].mean(0)
    return Xc


def evaluate_config(X, y, sid, ts, protocol, splits, centred, alphas):
    scores = {a: [] for a in alphas}
    for tr, te in splits:
        Xtr, ytr = X[tr], y[tr]
        if centred:
            Xtr = subject_center(Xtr, sid[tr])
        for a, pred, _, _, _, _ in ridge_path(Xtr, ytr, X[te], alphas):
            scores[a].append(nmae(y[te], pred))
    return {a: float(np.mean(v)) for a, v in scores.items()}


# ------------------------------------------------------------------- train
def do_train(protocol, train_dir, model_path):
    log("loading %s" % train_dir)
    X, y, sid, ts, names = load_directory(train_dir)
    log("raw features %s" % (X.shape,))
    if not np.isfinite(y).all():
        m = np.isfinite(y)
        log("dropping %d training rows with non-finite glucose" % (~m).sum())
        X, y, sid, ts = X[m], y[m], sid[m], ts[m]
    if len(y) < 20:
        raise SystemExit("not enough labelled training windows (%d)" % len(y))

    Xs, prep, names_kept = fit_preprocess(X, names)
    del X
    gc.collect()
    log("after preprocessing %s" % (Xs.shape,))

    sel = select_features(Xs, y, names_kept, MAX_FEATURES)
    Xs = Xs[:, sel]
    names_kept = [names_kept[i] for i in sel]
    prep["select"] = sel
    log("selected %d features" % Xs.shape[1])

    splits = make_splits(protocol, sid, ts, len(y))
    log("%d validation split(s) for %s" % (len(splits), protocol))

    variants = [False, True] if protocol == "d3" else [False]
    best = None
    for centred in variants:
        sc = evaluate_config(Xs, y, sid, ts, protocol, splits, centred, ALPHAS)
        a_best = min(sc, key=sc.get)
        log("  centred=%s  best CV NMAE %.4f at alpha=%.3g" % (centred, sc[a_best], a_best))
        if best is None or sc[a_best] < best[0]:
            best = (sc[a_best], a_best, centred)
    cv_nmae, alpha, centred = best
    log("chosen: alpha=%.3g centred=%s (CV NMAE %.4f)" % (alpha, centred, cv_nmae))

    Xfit = subject_center(Xs, sid) if centred else Xs
    w, ym = fit_ridge(Xfit, y, alpha)
    w, ym = irls_refit(Xfit, y, alpha, w, ym)

    # Fold the centring offset into the intercept so the stored model stays
    # exactly  yhat = intercept + z @ coef  on the untouched feature scale.
    offset = Xs.mean(0) - Xfit.mean(0) if centred else np.zeros(Xs.shape[1])
    intercept_std = float(ym - offset @ w)

    # Undo standardisation so coef applies to the raw selected features.
    mu = prep["mu"][sel]
    sd = prep["sd"][sel]
    coef = w / sd
    intercept = float(intercept_std - float(mu @ coef))

    train_nmae = nmae(y, Xs @ w + intercept_std)
    log("train NMAE %.4f  NMSE %.4f" % (train_nmae, nmse(y, Xs @ w + intercept_std)))

    state = {
        "format_version": FORMAT_VERSION,
        "protocol": protocol,
        "intercept": intercept,
        "coef": np.asarray(coef, np.float64),
        "feature_names": list(names_kept),
        "preprocessing_state": {
            "median": prep["median"], "lo": prep["lo"], "hi": prep["hi"],
            "keep": prep["keep"], "select": sel,
            "names_in": prep["names_in"], "names_out": list(names_kept),
            "cv_nmae": cv_nmae, "alpha": float(alpha), "centred": bool(centred),
        },
    }
    assert len(state["coef"]) == len(state["feature_names"]) < 500
    with open(model_path, "wb") as f:
        pickle.dump(state, f, protocol=4)
    log("wrote %s (%d features)" % (model_path, len(coef)))
    return 0


# ------------------------------------------------------ feature engineering
def do_features(protocol, test_dir, model_path, out_path):
    with open(model_path, "rb") as f:
        state = pickle.load(f)
    if state.get("protocol") != protocol:
        log("WARNING: model protocol %r != requested %r" % (state.get("protocol"), protocol))
    ps = state["preprocessing_state"]

    log("loading %s" % test_dir)
    X, _, _, _, names = load_directory(test_dir)
    log("raw test features %s" % (X.shape,))

    # align raw columns to the training layout by name where possible
    if names != ps["names_in"]:
        idx = {c: i for i, c in enumerate(names)}
        cols = []
        for c in ps["names_in"]:
            cols.append(idx[c] if c in idx else -1)
        Xa = np.full((len(X), len(cols)), np.nan)
        for j, c in enumerate(cols):
            if c >= 0:
                Xa[:, j] = X[:, c]
        X = Xa
        log("realigned test columns to the training feature layout")

    med, lo, hi = ps["median"], ps["lo"], ps["hi"]
    Xi = np.where(np.isfinite(X), X, med)
    Z = np.clip(Xi, lo, hi)[:, ps["keep"]][:, ps["select"]]
    Z = np.where(np.isfinite(Z), Z, 0.0)

    if Z.shape[1] != len(state["coef"]):
        raise SystemExit("built %d features but the model has %d coefficients"
                         % (Z.shape[1], len(state["coef"])))
    if not np.isfinite(Z).all():
        raise SystemExit("feature matrix contains non-finite values")

    np.save(out_path, np.ascontiguousarray(Z, dtype=np.float64))
    log("wrote %s with shape %s" % (out_path, Z.shape))
    pred = state["intercept"] + Z @ state["coef"]
    log("prediction summary: mean %.1f  sd %.1f  min %.1f  max %.1f mg/dL"
        % (pred.mean(), pred.std(), pred.min(), pred.max()))
    return 0


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    mode = sys.argv[1]
    if mode == "train":
        if len(sys.argv) != 5:
            sys.stderr.write("usage: part_d.py train <protocol> <train_dir> <model.pkl>\n")
            return 2
        return do_train(sys.argv[2], sys.argv[3], sys.argv[4])
    if mode == "feature_engineering":
        if len(sys.argv) != 6:
            sys.stderr.write("usage: part_d.py feature_engineering <protocol> "
                             "<test_dir> <model.pkl> <features.npy>\n")
            return 2
        return do_features(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    sys.stderr.write("unknown mode %r (expected 'train' or 'feature_engineering')\n" % mode)
    return 2


if __name__ == "__main__":
    sys.exit(main())
