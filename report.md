# Reproducing "ML-guided optimization of ALD" (Lee et al., Commun. Mater. 2026)

## What I found (up front)

1. **The dataset extraction is exact.** All 315 x 3 printed values in
   Supplementary Fig. 1 were transcribed and then read back a second time against
   the source panels: 945/945 match. I found **5 corrupted WER values** in the SI
   (printed as 0.24-0.30 A/min, sitting between ~2.7 A/min neighbours; they look
   like a lost leading digit / decimal shift). Kept verbatim in the committed
   CSV, flagged, and shown in step 4 to cost ~1 accuracy point.

2. **Thickness reproduces; refractive index and wet-etch-rate do not (as tested).**
   With uniformly random 215/100 splits, 10 runs, my numbers are
   **Tox 93 +/- 2%**, **RI 91 +/- 2%**, **WER 84 +/- 4%**, against the paper's
   92-95 / 95-97 / 90-95%. RI is ~4 pts low, WER ~6-10 pts low. This is robust
   across four under-specified choices (batch size, scaler type, output
   activation - see Assumptions); it is not an artefact of one bad guess.

3. **The whole gap is one regime.** At precursor temperature >= 60 C the model
   hits **99.5% (Tox), 99.8% (RI), 91.0% (WER)** - it meets or beats the paper.
   At precursor temperature = 55 C (20% of the data) it collapses to
   **67 / 57 / 58%**. Forcing the 45 prec=55 C points into the training set
   raises the random-split numbers to **99 / 98 / 89%**. The paper's Fig. 4a
   splits look hand-constructed rather than uniform-random; a split that keeps
   the low-precursor regime mostly in training reproduces their headline. My best
   explanation for the difference is **split composition, not model quality.**

4. **The headline metric is generous and fragile.** It is a pass-rate,
   `100 * mean(|y_pred - y_true| <= tol)`, on the test set, per property. Halving
   the tolerance drops Tox to 79% and WER to 60%. An equally reasonable "within
   10% relative" definition puts RI at 99% but WER at 49%. Predicting a constant
   (the training mean) already scores **43 / 67 / 39%** within the paper's
   windows, so a large part of the RI score in particular is the window being
   wide relative to the spread (tol ~= 0.3-0.4 sigma). The DNN clearly beats a
   linear model (skill 0.7-0.9), but **3-nearest-neighbour interpolation on the
   grid scores 84 / 86 / 68%**; the DNN's skill *over interpolation* is only
   0.58 / 0.36 / 0.52. On a dense 7x5x9 factorial grid, most of the accuracy is
   interpolation, not learned physics.

## 1. Data (`src/build_dataset.py`, `src/checks.py`, `data/EXTRACTION_NOTES.md`)

The SI `.docx` has no data tables; the 315 points are text labels on the wafer
maps in Supplementary Fig. 1. Extracted the three panels with PyMuPDF, transcribed
by eye. Structural checks (all pass): 315 rows, full 7x5x9 factorial, no
duplicates, ranges physically plausible, correct sign of the Tox/RI-vs-temperature
trends. Source spot-check: the prec=55 C column reproduced independently in main
Fig. 3b - identical. Residual uncertainty: possible +/-0.1 last-digit misreads in
a handful of cells where a label overlaps a contour; well inside the model's
run-to-run spread, so no conclusion depends on it.

## 2. Reproduction (`src/ald.py`, `src/step2_reproduce.py`)

Model exactly as specified: 4-128-64-32-16-8-3 dense, ELU, bias every layer,
inputs and targets normalised, 1000 epochs, metric as above. I train 10 times
with a fresh random split each time and report mean and min/max. Run-to-run
spread is large (+/-4-6 pts for WER) because the test set is only 100 points.

| scaler | batch | Tox | RI | WER |
|---|---|---|---|---|
| min-max | full (1000 updates) | 91.4 | 92.1 | 82.4 |
| min-max | 32 (Keras default)  | 93.2 | 91.3 | 84.6 |
| z-score | 32 | 93.4 | 90.8 | 79.9 |
| min-max + ELU output | 32 | 93.6 | 92.4 | 82.8 |
| min-max | 32, prec=55 C in train | 98.7 | 98.5 | 89.3 |

## 3. Interrogating the metric (`src/step3_metric.py`)

Exact meaning: for each property, the percentage of held-out points whose
absolute error is within the stated window; reported as the mean over runs. It
says nothing about error *size*. Under the model I built:
Tox R2 0.90 / RMSE 0.74 nm, RI R2 0.76 / RMSE 0.044, WER R2 0.53 / RMSE 1.76
A/min (median WER error 0.38 - a good centre with a heavy tail).

Separating model from tolerance width - three references chosen before looking,
same tolerance, skill = `(acc_DNN - acc_ref)/(100 - acc_ref)`:

| reference | Tox acc / skill | RI acc / skill | WER acc / skill |
|---|---|---|---|
| predict training mean | 43% / 0.88 | 67% / 0.74 | 39% / 0.74 |
| linear regression | 51% / 0.86 | 58% / 0.79 | 51% / 0.69 |
| 3-NN interpolation | 84% / 0.58 | 86% / 0.36 | 68% / 0.52 |

Reading: the problem is real (mean and linear are poor) but on this grid a local
interpolator captures most of it; the DNN's genuine advantage over "look up the
nearest measured wafers" is modest, and largest for WER.

## 4. Where the model fails (`src/step4_errors.py`)

Pooled 1000 test predictions, stratified. Misses are not spread out:

- **By precursor temp:** prec>=60 C -> 99-100% (Tox, RI), 89-93% (WER).
  prec=55 C -> 67 / 57 / 58%. MAE at prec=55 is 4x (Tox), 7x (RI), 5x (WER) the
  MAE at prec>=60.
- **By deposition temp:** weaker same-direction gradient (100 C worst, 250 C
  best) - the same low-temperature regime.
- **By wafer radius:** no effect (92.9 vs 93.2%) - not an edge-of-wafer problem.

Explanation tested - is prec=55 C hard because it is under-sampled or
intrinsically steep?
- **Test A:** within-wafer std at prec=55 C is 2.9 nm / 0.13 / 2.3 vs
  ~0.5 / 0.01 / 0.4 at prec>=60 C (5-12x). The RI spread across one prec=55 C
  wafer (~0.13) is **3x its tolerance window (0.04)**; WER (~2.3) is **2.6x its
  window (0.9)**. The nine points of such a wafer cannot all sit inside the
  window for any smooth function of the 4 coordinates.
- **Test B:** 3-NN interpolation fails at prec=55 C too, and *worse*
  (56 / 51 / 41%), on largely the same points (49/64, 74/85, 73/82 shared
  misses). The difficulty is in the data, not the DNN.

Physically this is the sub-vaporisation precursor regime the paper itself flags
as non-uniform (TDMA-Hf vapour pressure too low below ~60 C -> supply-limited,
position-sensitive growth). Side check: removing the 5 corrupted WER cells moves
pooled WER accuracy 84.6 -> 85.9%.

## Assumptions (explicit)

Stated in the paper and used as-is: architecture, ELU, per-layer bias,
input+target normalisation, 215/100 random split, 1000 epochs, the accuracy
metric and its three tolerance windows.

Inferred by me (paper silent), each flagged `# INFERRED` in `src/ald.py`:
- optimiser Adam, learning rate 1e-3, loss MSE (equal weight per normalised
  target).
- batch size: not stated. Full-batch and minibatch-32 both reported;
  minibatch-32 is the headline.
- scaler type: min-max to [0,1] (the paper's NN-output axes run ~0-1 "arb.");
  z-score reported as a variant.
- output layer: linear. ELU-output reported as a variant.
- weight init: PyTorch default (Kaiming uniform).
- "10 independent runs": I redraw the split each run, so the spread covers split
  + init variance together.
- wafer (x, y): nominal positions from Fig. 1a, not measured coordinates.
- the 5 anomalous WER values are a figure error and are kept, not repaired.
- deterministic CPU training; `OMP_NUM_THREADS` left free.

Ambiguities resolved by me: (a) "printed as numbers" - taken to mean the labels
on the figure, since there is no table; (b) which split - uniform random as the
neutral choice, with the prec=55-in-train variant reported as the likely
explanation of the paper's numbers; (c) metric on physical units, test set only.

## What I got wrong / did not finish

- **First instinct on extraction was wrong:** I looked for a data table in the
  `.docx`. There is none - the numbers are on the figures. Confirmed directly
  before spending time on conversion tools.
- **I did not reproduce the paper's exact split**, so I cannot prove the
  split-composition explanation - only show it is sufficient (prec=55-in-train
  gives 99/98/89%). The paper does not publish the point membership of Fig. 4a's
  sets.
- **Three-fold CV (Supp. Fig. 3) and the 100-training-point result not
  reproduced** - out of scope for the time; the prec=55 C finding already
  explains the headline gap and would only sharpen with less data.
- **No hyperparameter search.** By choice (brief: do not tune to match). It is
  possible a specific lr/epoch/batch combination lifts RI/WER a few points on
  random splits; I did not look, because the stratified analysis shows the
  ceiling is set by the data in one regime, not by optimisation.
- **WER tail not modelled.** median error 0.38 vs RMSE 1.76 says a few
  predictions are badly off; I identified *where* (prec=55 C) but did not try a
  heteroscedastic or regime-split model.
