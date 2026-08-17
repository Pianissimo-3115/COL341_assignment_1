""" python3 part_d.py train d1 part_d_datasets/d1/train model_d1.pkl
    python3 part_d.py feature_engineering d1 part_d_datasets/d1/public_test model_d1.pkl features_d1.npy
    python3 eval_d.py --model model_d1.pkl --features features_d1.npy --labels part_d_datasets/d1/public_test
"""
import argparse
import glob
import os
import pickle

import numpy as np

LABEL = "glucose"


def list_subject_files(directory):
    """Same sorted-filename convention part_d.py/evaluate_d.py use to
    concatenate a directory of per-subject files must match exactly, or
    labels won't line up with the rows in features.npy."""
    paths = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not paths:
        raise SystemExit("no .npz files found in %s" % directory)
    return paths


def load_labels(directory):
    parts = []
    for path in list_subject_files(directory):
        with np.load(path, allow_pickle=False) as data:
            if LABEL not in data.files:
                raise SystemExit("%s has no '%s' -- point --labels at a labelled directory "
                                  "(public_test/ or instructor_test/, not the unlabelled copy "
                                  "the evaluator hands the student's feature_engineering step)"
                                  % (path, LABEL))
            parts.append(np.asarray(data[LABEL], dtype=np.float64))
    labels = np.concatenate(parts, axis=0)
    if not np.isfinite(labels).all():
        raise SystemExit("%s: glucose contains non-finite values" % directory)
    return labels


def load_model(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    intercept = float(state["intercept"])
    coef = np.asarray(state["coef"], dtype=np.float64)
    return intercept, coef, state.get("feature_names")


def load_features(path, count, width):
    features = np.load(path, allow_pickle=False)
    if features.ndim != 2 or features.shape != (count, width):
        raise SystemExit("features must have shape (%d, %d); found %s" % (count, width, features.shape))
    if not np.isfinite(features).all():
        raise SystemExit("%s contains non-finite values" % path)
    return features


def nmse(y, p):
    yb = y.mean()
    return float(np.sum((y - p) ** 2) / np.sum((y - yb) ** 2))


def nmae(y, p):
    yb = y.mean()
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y - yb)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model.pkl from part_d.py train")
    ap.add_argument("--features", required=True, help="features.npy from part_d.py feature_engineering")
    ap.add_argument("--labels", required=True,
                     help="labelled directory of per-subject .npz files matching --features' row order "
                          "(e.g. the same public_test/ or instructor_test/ dir feature_engineering was run on)")
    ap.add_argument("--top-features", type=int, default=5,
                     help="print this many features by |standardized coefficient| (0 to skip)")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    intercept, coef, feature_names = load_model(args.model)
    features = load_features(args.features, len(labels), len(coef))

    predictions = intercept + features @ coef
    print("n = %d" % len(labels))
    print("NMSE = %.6f" % nmse(labels, predictions))
    print("NMAE = %.6f" % nmae(labels, predictions))

    if args.top_features and feature_names:
        order = np.argsort(-np.abs(coef))[:args.top_features]
        print("\ntop %d features by |coef|:" % args.top_features)
        for i in order:
            print("  %-28s coef=%+.4f" % (feature_names[i], coef[i]))


if __name__ == "__main__":
    main()
