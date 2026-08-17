#!/usr/bin/env python3
#evaluator for part b
import argparse
import csv
from pathlib import Path
import re
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
        names = [x for x in frame.columns if x != LABEL]
        if len(names) != FEATURES or (expected is not None and names != expected):
            raise CheckError("CSV must have the matching ordered set of 1640 features")
        x, y = frame[names].to_numpy(float), frame[LABEL].to_numpy(float)
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
        raise CheckError("cannot read %s: %s" % (path, exc))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise CheckError("CSV contains non-finite values")
    return frame, names, x, y

# parse folds.txt
def parse_folds_file(folds_path, count):
    
    try:
        content = Path(folds_path).read_text(encoding="utf-8")
        match = re.search(r"\[(.*?)\]", content)
        if not match:
            raise CheckError("Could not parse list from folds.txt")
        fold_ends = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        
        folds = np.zeros(count, dtype=int)
        start = 0
        for fold_idx, end in enumerate(fold_ends):
            folds[start:end] = fold_idx
            start = end
            
        if len(folds) != count:
            raise CheckError("folds count (%d) does not match train rows (%d)" % (len(folds), count))
        return folds
    except Exception as exc:
        raise CheckError("cannot read folds.txt: %s" % exc)


def one_column(path, count, name):
    try: lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc: raise CheckError("%s was not created: %s" % (name, exc))
    if len(lines) != count: raise CheckError("%s must have %d lines" % (name, count))
    result = []
    for i, line in enumerate(lines, 1):
        if len(line.split()) != 1: raise CheckError("%s line %d must have one value" % (name, i))
        try: result.append(float(line))
        except ValueError: raise CheckError("%s line %d is not numeric" % (name, i))
    result = np.asarray(result)
    if not np.isfinite(result).all(): raise CheckError("%s contains non-finite values" % name)
    return result


def cv_file(path, candidates):
    try:
        with open(path, encoding="utf-8", newline="") as handle: rows = list(csv.reader(handle))
    except OSError as exc: raise CheckError("crossvalidation_errors.txt was not created: %s" % exc)
    if len(rows) != len(candidates): raise CheckError("crossvalidation_errors.txt must have %d rows" % len(candidates))
    values = []
    for i, row in enumerate(rows, 1):
        if len(row) != 2: raise CheckError("CV row %d must contain lambda,error" % i)
        try: values.append((float(row[0]), float(row[1])))
        except ValueError: raise CheckError("CV row %d is not numeric" % i)
    values = np.asarray(values)
    if not np.isfinite(values).all(): raise CheckError("CV file contains non-finite values")
    return values[:, 0], values[:, 1]


def score(y, p, absolute):
    c = y-y.mean(); d = np.abs(c).sum() if absolute else (c*c).sum()
    if d <= 0: raise CheckError("metric is undefined for constant labels")
    return float((np.abs(y-p).sum() if absolute else ((y-p)**2).sum()) / d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program"); parser.add_argument("train"); parser.add_argument("public_test")
    parser.add_argument("folds"); parser.add_argument("regularization")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    try:
        train, names, _, _ = csv_data(args.train)
        public, _, x, y = csv_data(args.public_test, names)
        candidate_count = len(Path(args.regularization).read_text(encoding="utf-8").splitlines())
        candidates = one_column(args.regularization, candidate_count, "regularization.txt")
        
        # Updated fold parsing:
        folds = parse_folds_file(args.folds, len(train))
        
        if not len(candidates) or (candidates < 0).any(): raise CheckError("candidate lambdas must be non-negative and non-empty")
        if not np.array_equal(folds, folds.astype(int)) or not np.isin(folds, range(5)).all(): raise CheckError("folds.txt must contain integer values 0--4")
        with tempfile.TemporaryDirectory(prefix="col774_b_") as tmp:
            tmp = Path(tmp); test = tmp/"test.csv"; pfile=tmp/"predictions.txt"; wfile=tmp/"weights.txt"; lfile=tmp/"bestlambda.txt"; cfile=tmp/"crossvalidation_errors.txt"
            public.drop(columns=[LABEL]).to_csv(test, index=False)
            command = [sys.executable, str(Path(args.program).resolve()), str(Path(args.train).resolve()), str(test), str(Path(args.folds).resolve()), str(Path(args.regularization).resolve()), str(pfile), str(wfile), str(lfile), str(cfile)]
            try: done = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
            except subprocess.TimeoutExpired: raise CheckError("student program exceeded %d seconds" % args.timeout)
            if done.returncode: raise CheckError("student program exited with %d: %s" % (done.returncode, (done.stderr or done.stdout)[-2000:]))
            p, w, best = one_column(pfile, len(y), "predictions.txt"), one_column(wfile, FEATURES+1, "weights.txt"), one_column(lfile, 1, "bestlambda.txt")[0]
            reported_lambdas, reported_errors = cv_file(cfile, candidates)
        ordered = np.allclose(reported_lambdas, candidates, rtol=args.rtol, atol=args.atol)
        listed = np.any(np.isclose(best, candidates, rtol=args.rtol, atol=args.atol))
        cv_best = reported_lambdas[int(np.argmin(reported_errors))]
        best_ok = ordered and np.isclose(best, cv_best, rtol=args.rtol, atol=args.atol)
        weights_ok = np.allclose(p, np.column_stack((np.ones(len(x)), x)) @ w, rtol=args.rtol, atol=args.atol)
        print("COL774 A1 Part (b) public evaluation")
        print("Output format: PASS")
        print("Candidate-lambda order: %s" % ("PASS" if ordered else "FAIL"))
        print("bestlambda is a candidate: %s" % ("PASS" if listed else "FAIL"))
        print("bestlambda matches minimum reported CV error: %s" % ("PASS" if best_ok else "FAIL"))
        print("Predictions match submitted weights: %s" % ("PASS" if weights_ok else "FAIL"))
        print("Public NMAE: %.8f" % score(y,p,True)); print("Public NMSE: %.8f" % score(y,p,False))
        return 0 if all((ordered, listed, best_ok, weights_ok)) else 1
    except CheckError as exc:
        print("PUBLIC CHECK: FAIL — %s" % exc, file=sys.stderr); return 2


if __name__ == "__main__": 
    sys.exit(main())
