#!/usr/bin/env python3
"""Evaluator for Part (c).

Usage: python3 evaluate_c.py part_c.py train.csv public_test.csv
"""
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


def csv_data(path, expected=None):
    try:
        frame = pd.read_csv(path)
        if LABEL not in frame:
            raise CheckError("missing 'hr' column")
        names = [c for c in frame.columns if c != LABEL]
        if len(names) != FEATURES or (expected is not None and names != expected):
            raise CheckError("CSV must have the matching ordered set of 1640 features")
        x, y = frame[names].to_numpy(float), frame[LABEL].to_numpy(float)
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
        raise CheckError("cannot read %s: %s" % (path, exc))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise CheckError("CSV contains non-finite values")
    return frame, names, y


def predictions(path, count):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CheckError("predictions.txt was not created: %s" % exc)
    if len(lines) != count:
        raise CheckError("predictions.txt must have %d lines" % count)
    try:
        answer = np.asarray([float(x) if len(x.split()) == 1 else np.nan for x in lines])
    except ValueError:
        raise CheckError("predictions.txt contains a non-numeric value")
    if not np.isfinite(answer).all():
        raise CheckError("predictions.txt must have one finite value per line")
    return answer


def score(y, p, absolute):
    c = y - y.mean()
    denominator = np.abs(c).sum() if absolute else (c ** 2).sum()
    if denominator <= 0:
        raise CheckError("metric is undefined for constant labels")
    return float((np.abs(y-p).sum() if absolute else ((y-p)**2).sum()) / denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program")
    parser.add_argument("train")
    parser.add_argument("public_test")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    try:
        _, names, _ = csv_data(args.train)
        public, _, y = csv_data(args.public_test, names)
        with tempfile.TemporaryDirectory(prefix="col774_c_") as tmp:
            tmp = Path(tmp); test, output = tmp / "test.csv", tmp / "predictions.txt"
            public.drop(columns=[LABEL]).to_csv(test, index=False)
            try:
                done = subprocess.run([sys.executable, str(Path(args.program).resolve()),
                    str(Path(args.train).resolve()), str(test), str(output)], text=True,
                    capture_output=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                raise CheckError("student program exceeded %d seconds" % args.timeout)
            if done.returncode:
                raise CheckError("student program exited with %d: %s" %
                    (done.returncode, (done.stderr or done.stdout)[-2000:]))
            p = predictions(output, len(y))
        print("COL774 A1 Part (c) public evaluation")
        print("Output format: PASS")
        print("Public NMAE: %.8f" % score(y, p, True))
        print("Public NMSE: %.8f" % score(y, p, False))
        return 0
    except CheckError as exc:
        print("PUBLIC CHECK: FAIL — %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
