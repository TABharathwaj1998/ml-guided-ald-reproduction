"""Step 1: structural checks on the extracted dataset. Prints a report; exits
non-zero if any hard check fails.

    python src/checks.py
"""
import sys

import numpy as np

from ald import FEATURES, TARGETS, bad_wer_mask, load_data

DEP = [100, 125, 150, 175, 200, 225, 250]
PREC = [55, 60, 65, 70, 75]
POS = [(0, 10.5), (-1.5, 9), (1.5, 9), (-3, 7.5), (0, 7.5),
       (3, 7.5), (-1.5, 6), (1.5, 6), (0, 4.5)]

def main():
    df = load_data()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    check("row count == 315", len(df) == 315, f"got {len(df)}")
    check("columns == features + targets", list(df.columns) == FEATURES + TARGETS)
    check("no duplicate (dep,prec,x,y) keys",
          df.duplicated(FEATURES).sum() == 0,
          f"{df.duplicated(FEATURES).sum()} dups")
    check("dep levels", sorted(df.dep_temp_C.unique()) == DEP)
    check("prec levels", sorted(df.prec_temp_C.unique()) == PREC)
    check("9 wafer positions",
          sorted(map(tuple, df[["x_cm", "y_cm"]].drop_duplicates().values.tolist()))
          == sorted(map(tuple, POS)))
    check("full factorial 7x5x9",
          df.groupby(["dep_temp_C", "prec_temp_C"]).size().eq(9).all())
    check("no NaN", df.notna().all().all())

    # physical-plausibility ranges (loose; from the paper's figure colour bars)
    check("Tox in 2..18 nm", df.thickness_nm.between(2, 18).all())
    check("RI in 2.0..2.8", df.refractive_index.between(2.0, 2.8).all())
    check("WER in 0..21 A/min", df.wer_A_min.between(0, 21).all())

    # trend sanity: at fixed position, higher deposition temp -> lower Tox,
    # higher RI (paper Fig. 3a). Check the sign of the correlation on medians.
    g = df.groupby("dep_temp_C")[["thickness_nm", "refractive_index"]].median()
    check("Tox decreases with dep temp (median)",
          np.corrcoef(g.index, g.thickness_nm)[0, 1] < 0,
          f"r={np.corrcoef(g.index, g.thickness_nm)[0, 1]:.2f}")
    check("RI increases with dep temp (median)",
          np.corrcoef(g.index, g.refractive_index)[0, 1] > 0,
          f"r={np.corrcoef(g.index, g.refractive_index)[0, 1]:.2f}")

    # soft flag: known-bad WER cells are present and still low (not a failure)
    bad = bad_wer_mask(df)
    print(f"[FLAG] {bad.sum()} known-bad WER cells present, "
          f"values {sorted(df.loc[bad, 'wer_A_min'].tolist())} "
          f"(kept verbatim to match source; see EXTRACTION_NOTES.md)")

    print("\nAll hard checks passed." if ok else "\nSOME CHECKS FAILED.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
