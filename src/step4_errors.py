"""Step 4: where does the model fail, and why?

Pooled over N runs, every test-set prediction is collected with its process
conditions, then residuals are stratified. One explanation is tested rather than
many plots.

Claim built up here:
  - Misses are not spread out. They concentrate at precursor temperature 55 C
    (and, weaker, at low deposition temperature).
  - Restricting to prec >= 60 C, all three properties are essentially solved.
  - Explanation tested: is prec=55 hard because it is under-sampled (1/5 of the
    grid) or because the response surface is intrinsically steep/non-monotonic
    there?  Test A: within-wafer spread of each target by prec level.
                Test B: does a 3-NN interpolator miss the SAME points?  If yes,
                the difficulty is in the data, not the DNN.
  - Side check: recompute WER accuracy with the 5 known-bad cells removed.

    python src/step4_errors.py --runs 10
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from ald import (FEATURES, TARGETS, TOLERANCES, MinMax, bad_wer_mask, load_data,
                 run_once, split, tolerance_accuracy)

TOL = {t: TOLERANCES[t] for t in TARGETS}


def pooled_test_frame(df, runs, **kw):
    """Run the model `runs` times; return every test row with abs errors and a
    'hit_<target>' flag, plus the matching 3-NN abs errors."""
    parts = []
    for seed in range(runs):
        acc, info = run_once(df, seed, **kw)
        te = info["test_df"].copy().reset_index(drop=True)
        err = np.abs(info["pred"] - info["true"])

        tr = info["train_df"]
        xs = MinMax(kw.get("standard", False)).fit(tr[FEATURES].values)
        knn = KNeighborsRegressor(n_neighbors=3).fit(
            xs.transform(tr[FEATURES].values), tr[TARGETS].values)
        kerr = np.abs(knn.predict(xs.transform(te[FEATURES].values))
                      - te[TARGETS].values)

        for i, t in enumerate(TARGETS):
            te[f"ae_{t}"] = err[:, i]
            te[f"hit_{t}"] = err[:, i] <= TOL[t]
            te[f"knn_ae_{t}"] = kerr[:, i]
            te[f"knn_hit_{t}"] = kerr[:, i] <= TOL[t]
        te["seed"] = seed
        parts.append(te)
    return pd.concat(parts, ignore_index=True)


def by(frame, col):
    """Hit-rate (%) per target, grouped by `col`."""
    g = frame.groupby(col)
    out = pd.DataFrame({t: 100 * g[f"hit_{t}"].mean() for t in TARGETS})
    out["n"] = g.size()
    return out.round(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--standard", action="store_true")
    args = ap.parse_args()
    kw = dict(epochs=args.epochs, batch_size=args.batch_size, standard=args.standard)

    df = load_data()
    f = pooled_test_frame(df, args.runs, **kw)
    f["radius_cm"] = np.hypot(f.x_cm, f.y_cm - 7.5)   # wafer centre approx (0,7.5)

    print(f"pooled test predictions: {len(f)} rows over {args.runs} runs\n")

    print("=== hit-rate by precursor temperature ===")
    print(by(f, "prec_temp_C"))
    print("\n=== hit-rate by deposition temperature ===")
    print(by(f, "dep_temp_C"))
    print("\n=== hit-rate by wafer radius band ===")
    f["rband"] = pd.cut(f.radius_cm, [-.1, 1.6, 3.1, 9])
    print(by(f, "rband"))

    lo = f[f.prec_temp_C == 55]
    hi = f[f.prec_temp_C >= 60]
    print("\n=== prec = 55 C  vs  prec >= 60 C ===")
    for t in TARGETS:
        print(f"  {t:18s}  hit@55={100*lo[f'hit_{t}'].mean():5.1f}%   "
              f"hit>=60={100*hi[f'hit_{t}'].mean():5.1f}%   "
              f"MAE@55={lo[f'ae_{t}'].mean():.3f}  MAE>=60={hi[f'ae_{t}'].mean():.3f}")

    print("\n=== Test A: intrinsic steepness -- within-wafer spread of each target ===")
    g = df.groupby(["dep_temp_C", "prec_temp_C"])[TARGETS].std()
    print(g.groupby("prec_temp_C").mean().round(3))

    print("\n=== Test B: does 3-NN miss the same points? (prec = 55 C rows) ===")
    for t in TARGETS:
        both_miss = ((~lo[f"hit_{t}"]) & (~lo[f"knn_hit_{t}"])).sum()
        dnn_miss = (~lo[f"hit_{t}"]).sum()
        knn_miss = (~lo[f"knn_hit_{t}"]).sum()
        print(f"  {t:18s}  DNN misses={dnn_miss:4d}  kNN misses={knn_miss:4d}  "
              f"both={both_miss:4d}  "
              f"(kNN hit-rate@55 = {100*lo[f'knn_hit_{t}'].mean():.1f}%)")

    print("\n=== Side check: WER accuracy with the 5 known-bad cells removed ===")
    bad = bad_wer_mask(f)
    full = 100 * f["hit_wer_A_min"].mean()
    clean = 100 * f.loc[~bad, "hit_wer_A_min"].mean()
    print(f"  bad rows in pooled test set: {bad.sum()} of {len(f)}")
    print(f"  WER hit-rate  all rows: {full:.1f}%   without bad cells: {clean:.1f}%")


if __name__ == "__main__":
    main()
