#evaluator for Part (a)
# use: python3 eval_a.py part_a.py train.csv test.csv
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

FEATURES, LABEL = 1640, "hr"


class CheckError(Exception):
    pass

# validation for CSV
def read_csv(path, columns=None):
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise CheckError("cannot read %s: %s" % (path, exc))
    
    if LABEL not in frame:
        raise CheckError("%s must contain the heart rate column" % (path))
    
    names = [c for c in frame.columns if c != LABEL]
    
    if len(names) != FEATURES:
        raise CheckError("%s has %d features; expected %d" % (path, len(names), FEATURES))
    
    if columns is not None and names != columns:
        raise CheckError("training and public-test feature names/order differ")
    
    try:
        x, y = frame[names].to_numpy(np.float64), frame[LABEL].to_numpy(np.float64)
    except (TypeError, ValueError) as exc:
        raise CheckError("non-numeric CSV data: %s" % exc)
    
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise CheckError("CSV contains NaN or infinite values")
    
    return frame, names, x, y

# validation for vector files
def vector(path, count, name):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CheckError("%s was not created: %s" % (name, exc))
    if len(lines) != count:
        raise CheckError("%s must have %d lines; found %d" % (name, count, len(lines)))
    result = []
    for i, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 1:
            raise CheckError("%s line %d must contain one value" % (name, i))
        try:
            result.append(float(fields[0]))
        except ValueError:
            raise CheckError("%s line %d is not numeric" % (name, i))
    result = np.asarray(result)
    if not np.isfinite(result).all():
        raise CheckError("%s contains NaN or infinite values" % name)
    return result


def metric(y, prediction, absolute):
    centered = y - y.mean()
    denominator = np.abs(centered).sum() if absolute else (centered ** 2).sum()
    if denominator <= 0:
        raise CheckError("metric is undefined for constant labels")
    numerator = np.abs(y - prediction).sum() if absolute else ((y - prediction) ** 2).sum()
    return float(numerator / denominator)


def run(command, timeout):
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise CheckError("student program exceeded %d seconds" % timeout)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise CheckError("student program exited with %d%s" %
                         (completed.returncode, "\n" + detail[-2000:] if detail else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program")
    parser.add_argument("train")
    parser.add_argument("public_test")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    try:
        _, names, _, _ = read_csv(args.train)
        public, _, x, y = read_csv(args.public_test, names)
        with tempfile.TemporaryDirectory(prefix="col774_a_") as tmp:
            tmp = Path(tmp)
            test, predictions_file, weights_file = tmp / "test.csv", tmp / "predictions.txt", tmp / "weights.txt"
            public.drop(columns=[LABEL]).to_csv(test, index=False)
            run([sys.executable, str(Path(args.program).resolve()), str(Path(args.train).resolve()),
                 str(test), str(predictions_file), str(weights_file)], args.timeout)
            prediction = vector(predictions_file, len(y), "predictions.txt")
            weights = vector(weights_file, FEATURES + 1, "weights.txt")
        reconstructed = np.column_stack((np.ones(len(x)), x)) @ weights
        consistent = np.allclose(prediction, reconstructed, rtol=args.rtol, atol=args.atol)
        print("COL774 A1 Part (a) public evaluation")
        print("Output format: PASS")
        print("Predictions match submitted weights: %s" % ("PASS" if consistent else "FAIL"))
        print("Public NMAE: %.8f" % metric(y, prediction, True))
        print("Public NMSE: %.8f" % metric(y, prediction, False))
        return 0 if consistent else 1
    except CheckError as exc:
        print("PUBLIC CHECK: FAIL — %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
