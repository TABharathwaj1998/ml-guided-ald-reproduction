"""Step 3: interrogate the "accuracy above 90% within +/-1.0 nm" metric.

Three questions from the brief:

1. What exactly does the sentence mean?
   acc_j = 100 * mean_over_test( |y_pred - y_true| <= tol_j ),  physical units,
   per property, one value per run, reported as mean over runs. It is a pass-rate,
   not an error size.

2. What is the number under a different but equally reasonable definition?
   We recompute for: strict '<'; half / double tolerance; a relative 10% window;
   a spread-relative window (|err| <= 0.5 * sigma_target); and we also report the
   plain error sizes (MAE, RMSE, R2, median|err|) that the pass-rate hides.

3. How much of the score is the model, and how much is the tolerance being wide
   relative to the data spread?
   We compare the DNN pass-rate to three reference predictors evaluated with the
   SAME tolerance, chosen BEFORE seeing their scores:
     - mean     : always predict the training mean. The "no-model" floor - its
                  pass-rate is purely "how much of the test data already sits
                  within tol of a constant", i.e. the tolerance-width effect.
     - linear   : ordinary least squares on the 4 raw inputs. Answers "how much
                  of the response is just linear?" - if the DNN barely beats it,
                  the nonlinearity argument is weak.
     - knn(k=3) : average of the 3 nearest training points in scaled-input space.
                  The data is a dense regular 7x5x9 grid, so local interpolation
                  is the natural non-parametric competitor. If the DNN does not
                  beat 3-NN, it is not doing more than "look up nearby runs".
   Skill score  s = (acc_model - acc_ref) / (100 - acc_ref)  is the fraction of
   the remaining gap to 100% that the model closes beyond the reference.

    python src/step3_metric.py --runs 10
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

from ald import (FEATURES, TARGETS, TOLERANCES, MinMax, load_data, predict,
                 run_once, split, tolerance_accuracy, train_model)

TOL = np.array([TOLERANCES[t] for t in TARGETS])


# --- reference predictors (fixed before looking at results) ------------------
def ref_predictions(tr, te, standard=False):
    """Return {name: pred_phys (n_test, 3)} for the three reference models."""
    Xtr_raw, Xte_raw = tr[FEATURES].values, te[FEATURES].values
    ytr = tr[TARGETS].values
    xs = MinMax(standard).fit(Xtr_raw)
    Xtr, Xte = xs.transform(Xtr_raw), xs.transform(Xte_raw)

    out = {}
    out["mean"] = np.repeat(ytr.mean(0, keepdims=True), len(te), axis=0)
    out["linear"] = LinearRegression().fit(Xtr, ytr).predict(Xte)
    out["knn(k=3)"] = KNeighborsRegressor(n_neighbors=3).fit(Xtr, ytr).predict(Xte)
    return out


def alt_metrics(true, pred):
    """Dict of alternative summaries for one (true, pred) pair, per target."""
    err = np.abs(pred - true)
    d = {}
    for i, t in enumerate(TARGETS):
        e, y = err[:, i], true[:, i]
        d[t] = {
            "acc<=tol": 100 * np.mean(e <= TOL[i]),
            "acc<tol": 100 * np.mean(e < TOL[i]),
            "acc<=tol/2": 100 * np.mean(e <= TOL[i] / 2),
            "acc<=2tol": 100 * np.mean(e <= 2 * TOL[i]),
            "acc<=10%rel": 100 * np.mean(e <= 0.10 * np.abs(y)),
            "acc<=0.5std": 100 * np.mean(e <= 0.5 * y.std()),
            "MAE": e.mean(),
            "RMSE": np.sqrt((e ** 2).mean()),
            "medAE": np.median(e),
            "R2": r2_score(y, pred[:, i]),
            "tol/std": TOL[i] / y.std(),
        }
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--standard", action="store_true")
    args = ap.parse_args()

    df = load_data()

    alt_rows = {t: [] for t in TARGETS}
    skill = {name: {t: [] for t in TARGETS} for name in ["mean", "linear", "knn(k=3)"]}
    ref_acc = {name: {t: [] for t in TARGETS} for name in ["mean", "linear", "knn(k=3)"]}
    dnn_acc = {t: [] for t in TARGETS}

    for seed in range(args.runs):
        acc, info = run_once(df, seed, standard=args.standard,
                             epochs=args.epochs, batch_size=args.batch_size)
        true, pred = info["true"], info["pred"]
        for t in TARGETS:
            dnn_acc[t].append(acc[t])
        am = alt_metrics(true, pred)
        for t in TARGETS:
            alt_rows[t].append(am[t])

        refs = ref_predictions(info["train_df"], info["test_df"], args.standard)
        for name, rp in refs.items():
            ra = tolerance_accuracy(true, rp, TOL)
            for t in TARGETS:
                ref_acc[name][t].append(ra[t])
                s = (acc[t] - ra[t]) / (100 - ra[t]) if ra[t] < 100 else np.nan
                skill[name][t].append(s)

    # ---- report -----------------------------------------------------------
    print(f"\n=== Q1/Q2: metric under alternative definitions "
          f"(mean over {args.runs} runs) ===")
    for t in TARGETS:
        A = pd.DataFrame(alt_rows[t]).mean()
        print(f"\n{t}  (paper tol +/-{TOLERANCES[t]}):")
        for k in ["acc<=tol", "acc<tol", "acc<=tol/2", "acc<=2tol",
                  "acc<=10%rel", "acc<=0.5std"]:
            print(f"    {k:14s} {A[k]:6.1f}%")
        for k in ["MAE", "RMSE", "medAE", "R2", "tol/std"]:
            print(f"    {k:14s} {A[k]:7.3f}")

    print(f"\n=== Q3: model skill vs tolerance width "
          f"(mean over {args.runs} runs) ===")
    print(f"{'target':18s} {'DNN':>7} | "
          + " | ".join(f"{n:>9} acc / skill" for n in ["mean", "linear", "knn(k=3)"]))
    for t in TARGETS:
        line = f"{t:18s} {np.mean(dnn_acc[t]):6.1f}% |"
        for name in ["mean", "linear", "knn(k=3)"]:
            line += (f" {np.mean(ref_acc[name][t]):6.1f}% /"
                     f" {np.nanmean(skill[name][t]):5.2f} |")
        print(line)
    print("\nskill = (acc_DNN - acc_ref) / (100 - acc_ref); "
          "1.0 = DNN closes the whole remaining gap, 0 = no better than ref, "
          "<0 = worse.")


if __name__ == "__main__":
    main()
