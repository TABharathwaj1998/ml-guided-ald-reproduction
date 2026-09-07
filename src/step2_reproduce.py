"""Step 2: reproduce the headline accuracy claim.

Run the paper's model N times with fresh random 215/100 splits; report the mean
and the run-to-run spread of the tolerance-accuracy for each property, beside the
paper's reported ranges.

    python src/step2_reproduce.py --runs 10
    python src/step2_reproduce.py --runs 10 --standard      # z-score instead of min-max
    python src/step2_reproduce.py --runs 10 --elu-output    # ELU on the output layer
"""
import argparse

import numpy as np
import pandas as pd

from ald import TARGETS, TOLERANCES, load_data, run_once

# paper, "test accuracy" ranges (main text, ALD model prediction accuracy)
PAPER = {"thickness_nm": (92, 95), "refractive_index": (95, 97), "wer_A_min": (90, 95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--standard", action="store_true", help="z-score not min-max")
    ap.add_argument("--elu-output", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="minibatch size; default = full batch")
    args = ap.parse_args()

    df = load_data()
    rows, maes = [], []
    for seed in range(args.runs):
        acc, info = run_once(df, seed, standard=args.standard,
                             elu_output=args.elu_output, epochs=args.epochs,
                             batch_size=args.batch_size)
        rows.append(acc)
        maes.append(info["mae"])
        print(f"seed {seed}: " +
              "  ".join(f"{t.split('_')[0]:>9}={acc[t]:5.1f}%" for t in TARGETS))

    A = pd.DataFrame(rows)
    M = np.array(maes)
    print(f"\n--- summary over {args.runs} runs "
          f"(scaler={'zscore' if args.standard else 'minmax'}, "
          f"elu_output={args.elu_output}, epochs={args.epochs}, "
          f"batch={args.batch_size or 'full'}) ---")
    print(f"{'target':18s} {'tol':>6}  {'acc mean':>9} {'acc min':>8} "
          f"{'acc max':>8}   {'MAE mean':>9}   paper")
    for i, t in enumerate(TARGETS):
        print(f"{t:18s} {TOLERANCES[t]:>6}  {A[t].mean():8.1f}% {A[t].min():7.1f}% "
              f"{A[t].max():7.1f}%   {M[:, i].mean():9.3f}   {PAPER[t][0]}-{PAPER[t][1]}%")


if __name__ == "__main__":
    main()
