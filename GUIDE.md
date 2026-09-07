# Guide: understanding and testing this repo

A companion to the code for the live walkthrough. Read top to bottom once, then
use Part 5 as a testing checklist and Part 6/7 as walkthrough prep.

---

## Part 0 - The one-paragraph mental model

The paper trains one small neural network to predict **3 numbers**
(film thickness, refractive index, wet etch rate) from **4 numbers**
(deposition temperature, precursor temperature, x, y position on the wafer). It
has **315 measured rows**. It splits them 215 train / 100 test, trains the net,
and reports - per property - **the percentage of test rows the net got within a
fixed tolerance** (+/-1.0 nm, +/-0.04, +/-0.9 A/min). Claim: all three above 90%.
This repo re-extracts the data, rebuilds that net, checks the claim, takes the
metric apart, and finds where the model fails.

### Data flow

```
Supplementary Fig. 1  (numbers printed on wafer-map images)
        |  hand-transcribed, verified twice against the panels
        v
data/ald_hfox_dataset.csv        315 rows x 7 columns
        |  load_data()
        v
split(seed) --> 215 train rows  +  100 test rows
        |
        |  MinMax().fit(train)      <- scaler learns min/max from TRAIN ONLY
        v
   normalise X (4 cols) and y (3 cols) to [0,1]
        |
        v
   train_model()  ->  DNN 4-128-64-32-16-8-3, ELU, Adam, MSE, 1000 epochs
        |
        v
   predict(test X)  ->  ys.inverse(...)   <- back to nm / index / (A/min)
        |
        v
   tolerance_accuracy(true, pred, tol)  ->  {thickness: 93.2%, ri: 91.3%, wer: 84.6%}
```

Everything in `src/` is either this pipeline (`ald.py`) or a script that runs it
many times and slices the output a particular way (`step2/3/4`).

---

## Part 1 - Setup

```bash
cd "ML assignment ATONARP"
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt         # Linux / macOS
```

`requirements.txt` pins the versions actually used. `torch` is the CPU build - no
GPU needed. Total install ~1 GB (torch is most of it).

The scripts `import ald`, so **run them from `src/`**:

```bash
cd src
python checks.py
```

`ald.load_data()` finds the CSV via `REPO_ROOT` (computed from `__file__`), so it
works regardless of where you launch from - only the `import ald` needs `src/` on
the path.

---

## Part 2 - The files, one by one

### `data/ald_hfox_dataset.csv`

315 rows. Columns, in order:

| column | meaning | values |
|---|---|---|
| `dep_temp_C` | deposition (chamber) temperature | 100,125,150,175,200,225,250 |
| `prec_temp_C` | TDMA-Hf precursor temperature | 55,60,65,70,75 |
| `x_cm`, `y_cm` | wafer position, cm from an origin at the inlet | 9 fixed (x,y) pairs |
| `thickness_nm` | film thickness Tox | ~2.7 - 16.7 |
| `refractive_index` | RI | ~2.06 - 2.69 |
| `wer_A_min` | wet etch rate | ~0.24 - 19.7 |

It is a **full factorial grid**: 7 dep x 5 prec x 9 positions = 315, every
combination present exactly once.

### `src/build_dataset.py` - transcription of record

Not part of the analysis. It holds the 315x3 numbers as Python dicts (exactly as
read off the figure) and writes the CSV. Run once from the repo root
(`python src/build_dataset.py`) to regenerate the CSV; it asserts 315 rows.
Its only job is to make the manual extraction auditable and reproducible.

### `src/ald.py` - the core (the file the walkthrough will focus on)

Read this one closely. Section by section:

**Constants (lines ~19-39)**
- `FEATURES`, `TARGETS` - column names, also fix the column order everywhere.
- `TOLERANCES` - the paper's three windows, in physical units. This is the whole
  metric definition in one dict.
- `KNOWN_BAD_WER` - the 5 corrupted WER cells (see `data/EXTRACTION_NOTES.md`),
  keyed by `(dep, prec, x, y)`. Only used by step 4 to measure their effect.

**`load_data(path)`** - `pd.read_csv` + two asserts (columns exact, length 315).
The asserts are cheap tripwires: if the CSV is edited wrongly, everything else
fails loudly here instead of silently later.

**`bad_wer_mask(df)`** - returns a boolean Series, `True` on the 5 flagged rows.
Used as `df[~mask]` to drop them.

**`split(df, seed)`** -
```python
rng = np.random.default_rng(seed)
idx = rng.permutation(len(df))
return df.iloc[idx[N_TEST:]].copy(), df.iloc[idx[:N_TEST]].copy()
#            \_ train (215) _/         \_ test (100) _/
```
`default_rng(seed)` = a fresh independent random generator; same `seed` -> same
permutation -> **fully reproducible split**. Returns `(train, test)` in that
order. `.copy()` so later column assignments don't warn about views.

**`class MinMax`** - the scaler. `fit` learns two vectors from an array `A`
(shape `(n, k)`): a centre `c` and a scale `s`, per column.
- default (min-max): `c = column min`, `s = max - min` -> `transform` maps each
  column to `[0, 1]`.
- `standard=True` (z-score): `c = mean`, `s = std` -> mean 0, std 1.
- `np.where(s == 0, 1.0, s)` - if a column is constant, `s` would be 0 and we'd
  divide by zero; replace with 1 (the column then maps to all-zeros, harmless).
- `inverse` undoes it: `A * s + c`. **Used to turn normalised predictions back
  into nm / index / (A/min) before scoring.**
- Fit on **train only** - the test set must not influence the scaling, or the
  reported accuracy is optimistic (leakage).

**`class DNN`** - the network.
```python
widths = [4, 128, 64, 32, 16, 8, 3]
for i in range(len(widths) - 1):
    layers.append(nn.Linear(widths[i], widths[i + 1]))   # 4->128, 128->64, ...
    is_last = i == len(widths) - 2                        # the 8->3 layer
    if not is_last or elu_output:
        layers.append(nn.ELU())
```
So: `Linear(4,128) -> ELU -> Linear(128,64) -> ELU -> ... -> Linear(8,3)`. ELU
after every layer **except the last** (linear output head). `nn.Linear` includes
a bias by default - that is the paper's `b1..b6`. `elu_output=True` adds an ELU
after the last layer too (the alternative reading of the paper's wording).
`nn.Sequential` just chains them; `forward` runs the chain.

**`train_model(Xn, yn, seed, epochs, lr, batch_size, elu_output)`**
```python
torch.manual_seed(seed)          # reproducible weight init + batch shuffling
model = DNN(elu_output)
opt = torch.optim.Adam(model.parameters(), lr=lr)   # INFERRED optimiser + lr
loss_fn = nn.MSELoss()                              # INFERRED loss
X = torch.tensor(Xn, float32); y = torch.tensor(yn, float32)
bs = n if batch_size is None else batch_size        # None => full batch
g = torch.Generator().manual_seed(seed)
for _ in range(epochs):                             # 1000
    order = torch.randperm(n, generator=g)          # reshuffle each epoch
    for s in range(0, n, bs):                       # walk minibatches
        b = order[s:s+bs]
        opt.zero_grad()                             # clear old gradients
        loss_fn(model(X[b]), y[b]).backward()       # forward + backprop
        opt.step()                                  # update weights
model.eval()
```
`zero_grad -> forward -> loss -> backward -> step` is the standard PyTorch
training loop. One **epoch** = one pass over all 215 rows. With `batch_size=None`
that is 1 weight update per epoch (1000 total); with `batch_size=32` it is
`ceil(215/32)=7` updates per epoch (~7000 total). That difference is why "1000
epochs" is ambiguous and why step 2 reports both.

**`predict(model, Xn)`** - `@torch.no_grad()` (don't build the gradient graph,
faster) + `.numpy()` so the rest of the code stays in numpy/pandas.

**`tolerance_accuracy(y_true, y_pred, tol_vec)`** - **the metric**, verbatim:
```python
err = np.abs(y_pred - y_true)            # shape (n, 3), physical units
{TARGETS[i]: 100 * np.mean(err[:, i] <= tol_vec[i]) for i in range(3)}
```
For each of the 3 columns: fraction of rows whose absolute error is within that
column's tolerance, times 100. A pass-rate. Nothing about error size.

**`run_once(df, seed, ...)`** - glues it together: split -> fit two scalers on
train -> train -> predict test -> inverse-scale -> score. Returns
`(acc_dict, info)` where `info` carries `pred`, `true`, `test_df`, `train_df`,
`mae` so the scripts can do their own slicing.

### `src/checks.py` - step 1

Runs ~13 structural checks and prints `[PASS]/[FAIL]`, exits non-zero if any hard
check fails. Groups:
- shape / uniqueness: 315 rows, columns correct, no duplicate `(dep,prec,x,y)`.
- grid completeness: dep has all 7 levels, prec all 5, exactly 9 positions,
  every `(dep,prec)` cell has 9 rows.
- plausibility: value ranges inside loose physical bounds.
- trend sanity: median thickness **decreases** with dep temp, median RI
  **increases** (the direction the paper describes) - checked via the sign of a
  correlation coefficient.
- a soft `[FLAG]` line listing the 5 known-bad WER cells (not a failure).

### `src/step2_reproduce.py` - step 2

Loop `seed in range(runs)`: `run_once`, collect the 3 accuracies and the MAE
vector. Print per-seed lines, then a summary table with `mean / min / max` per
property next to the paper's ranges (`PAPER` dict). Flags: `--runs`, `--epochs`,
`--standard`, `--elu-output`, `--batch-size`.

### `src/step3_metric.py` - step 3

Two blocks.
- **`alt_metrics(true, pred)`** - for each property, recompute the score under
  other reasonable definitions: strict `<`, half tolerance, double tolerance,
  "within 10% relative" (`err <= 0.10*|y|`), "within 0.5 std", and the plain
  error sizes MAE / RMSE / median|err| / R2, plus `tol/std` (how wide the window
  is vs the data spread).
- **`ref_predictions(tr, te)`** - three baselines, **fixed before looking**:
  `mean` (predict the training mean for every row), `linear`
  (`LinearRegression` on the scaled inputs), `knn(k=3)` (`KNeighborsRegressor`,
  average of the 3 nearest training rows). Each scored with the *same*
  `tolerance_accuracy`. Then `skill = (acc_DNN - acc_ref) / (100 - acc_ref)` =
  the fraction of the "gap left to 100%" the DNN closes on top of that baseline.

### `src/step4_errors.py` - step 4

`pooled_test_frame` runs `run_once` for `runs` seeds and stacks every test row
(1000 total) with columns `ae_<t>` (abs error), `hit_<t>` (within tol?), and the
same for a 3-NN fit on that run's training set. Then:
- `by(frame, col)` - hit-rate per property grouped by `col`. Printed for
  `prec_temp_C`, `dep_temp_C`, and a wafer-radius band.
- explicit `prec == 55` vs `prec >= 60` comparison (hit-rate + MAE).
- **Test A**: within-wafer std of each target, averaged by prec level (is the
  surface intrinsically steep at prec=55?).
- **Test B**: for prec=55 rows, how many points does 3-NN miss, and how many are
  missed by *both* (is the hard region model-specific or data-intrinsic?).
- side check: WER hit-rate with the 5 `KNOWN_BAD_WER` rows removed.

---

## Part 3 - Poke at it in a REPL (understand by touching)

```bash
cd src
python
```

```python
from ald import *
df = load_data()
df.shape                      # (315, 7)
df.head()
df.describe()                 # note wer_A_min min = 0.24  <- the anomaly

# the split is deterministic in the seed
tr, te = split(df, seed=0)
len(tr), len(te)              # (215, 100)
tr2, te2 = split(df, seed=0)
(te.index == te2.index).all() # True  -> same seed, same split
tr3, te3 = split(df, seed=1)
(te.index == te3.index).all() # False -> different seed, different split

# the scaler - fit on train, then look at where TEST lands
ys = MinMax().fit(tr[TARGETS].values)
ys.transform(tr[TARGETS].values).min(0)   # ~[0,0,0]
ys.transform(tr[TARGETS].values).max(0)   # ~[1,1,1]
ys.transform(te[TARGETS].values).min(0)   # e.g. [-0.007, 0.018, 0.003]
ys.transform(te[TARGETS].values).max(0)   # e.g. [ 0.971, 1.125, 0.959]
# test values fall slightly OUTSIDE [0,1] - proof the scaler never saw them.
# (the 4 inputs are a grid: every level appears in both splits, so their
#  transformed test range is ~[0,1] too - the leakage point shows on targets.)

# the model object
m = DNN()
print(m)                                   # see the 6 Linear + 5 ELU layers
sum(p.numel() for p in m.parameters())     # ~11k parameters, on 215 rows

# one full run
acc, info = run_once(df, seed=0, batch_size=32)
acc                                        # {'thickness_nm': ~93, ...}
info['mae']                                # array of 3 MAEs, physical units
info['pred'].shape                         # (100, 3)

# the metric by hand, to prove you know what it computes
import numpy as np
err = np.abs(info['pred'] - info['true'])
100 * np.mean(err[:,0] <= 1.0)             # == acc['thickness_nm']
```

If you can reproduce those outputs and explain each line, you understand `ald.py`.

---

## Part 4 - The testing checklist (run these in order)

All from `src/`. Times are rough, CPU.

### Test 1 - data integrity (instant)
```bash
python checks.py
```
**Expect:** every line `[PASS]`, one `[FLAG]` about 5 WER cells, final
`All hard checks passed.` **Proves:** the CSV is the right shape, the grid is
complete, no duplicates, trends have the right sign. If any `[FAIL]`, the CSV was
corrupted - stop and fix it.

### Test 2 - headline reproduction (~40 s full-batch, ~4 min minibatch)
```bash
python step2_reproduce.py --runs 10                    # full-batch
python step2_reproduce.py --runs 10 --batch-size 32    # minibatch (headline)
```
**Expect (minibatch):** roughly `Tox 93 (91-97) / RI 91 (87-95) / WER 85 (81-88)`.
Numbers move a bit run to run - that is the point of `min`/`max`. **Read it as:**
Tox lands in the paper's 92-95 band; RI and WER sit a few points below.

### Test 3 - are the inferred choices to blame? (~4 min each)
```bash
python step2_reproduce.py --runs 10 --batch-size 32 --standard      # z-score
python step2_reproduce.py --runs 10 --batch-size 32 --elu-output    # ELU output
```
**Expect:** all within ~2 points of Test 2. **Proves:** the RI/WER gap is not an
artefact of one guessed hyperparameter.

### Test 4 - metric interrogation (~4 min)
```bash
python step3_metric.py --runs 10 --batch-size 32
```
**Expect, Q1/Q2 block:** for each property the pass-rate at tol, tol/2, 2*tol,
10% relative, 0.5 std, plus MAE/RMSE/R2. Look at how far `acc<=tol/2` falls
below `acc<=tol` (Tox ~93 -> ~79; WER ~85 -> ~60). **Expect, Q3 block:** a table
`DNN | mean acc/skill | linear acc/skill | knn acc/skill`. Key rows: `mean` acc
~43/67/39 (that is the tolerance-width floor); `knn(k=3)` acc ~84/86/68 with
skill ~0.58/0.36/0.52 (the DNN's edge over interpolation is small).

### Test 5 - error investigation (~4 min)
```bash
python step4_errors.py --runs 10 --batch-size 32
```
**Expect:** "hit-rate by precursor temperature" shows ~99-100% at 60-75 C and a
cliff to ~57-68% at 55 C. "prec = 55 vs prec >= 60" shows MAE 4-7x worse at 55.
Test A: within-wafer std ~5-12x larger at prec 55. Test B: 3-NN also fails at
prec 55, on the same points. Side check: removing bad WER cells nudges WER
84.6 -> ~85.9%.

### Test 6 - the split hypothesis (~1 min)
```bash
python - <<'PY'
import numpy as np, pandas as pd
from ald import *
tol = [TOLERANCES[t] for t in TARGETS]
def run(seed, keep55):
    rng = np.random.default_rng(seed); idx = np.arange(len(load_data()))
    df = load_data()
    pool = idx[df.prec_temp_C.values != 55] if keep55 else idx
    test_idx = rng.permutation(pool)[:N_TEST]
    tr = df.drop(index=test_idx); te = df.iloc[test_idx]
    xs = MinMax().fit(tr[FEATURES].values); ys = MinMax().fit(tr[TARGETS].values)
    m = train_model(xs.transform(tr[FEATURES].values), ys.transform(tr[TARGETS].values),
                    seed=seed, epochs=1000, batch_size=32)
    p = ys.inverse(predict(m, xs.transform(te[FEATURES].values)))
    return tolerance_accuracy(te[TARGETS].values, p, tol)
for lab, f in [("random", False), ("prec55 in train", True)]:
    A = pd.DataFrame([run(s, f) for s in range(6)])
    print(lab, {k: round(A[k].mean(),1) for k in TARGETS})
PY
```
**Expect:** `random` ~93/91/85, `prec55 in train` ~99/98/89. **Proves:** the gap
to the paper is explained by which rows land in the test set.

### Test 7 - determinism
Run any script twice with the same flags. The per-seed lines must be **identical**
(all randomness is seeded). If they are not, something is reading unseeded
randomness - a bug.

---

## Part 5 - Reading the numbers

**Reproduction table (step 2).** "93 (91-97)" = mean 93%, min 91%, max 97% over
10 random splits. Compare the *mean* to the paper's range; use *min/max* to argue
the estimate is noisy (only 100 test points -> one point = 1%).

**Alt-metric block (step 3).** The pass-rate is not an error size. Tox R2 0.90,
RMSE 0.74 nm - the model is genuinely good. WER R2 0.53, median error 0.38 but
RMSE 1.76 - a good centre with a heavy tail (a few big misses). `tol/std ~ 0.3-0.4`
means the window is a third of a standard deviation wide: generous.

**Skill table (step 3).** `skill = (acc_DNN - acc_ref)/(100 - acc_ref)`.
- vs `mean` (~0.8): the DNN is doing real work - a constant is poor.
- vs `linear` (~0.75): the problem is meaningfully nonlinear.
- vs `knn(k=3)` (0.36-0.58): on a dense grid, "predict like the nearest measured
  wafers" already gets you most of the way; the DNN's extra is modest.

**Stratified hit-rates (step 4).** Not an average - a *breakdown*. The model is
~99% solved everywhere except precursor temperature 55 C, which is ~20% of the
data and carries essentially all the RI/WER misses. That reframes "91% RI" as
"~99% in the stable ALD regime, ~57% in the low-precursor regime".

---

## Part 6 - Walkthrough Q&A bank

**Q: What does `tolerance_accuracy` compute, exactly?**
For each target column: the percentage of rows where `|prediction - truth|` (in
physical units) is `<= that column's tolerance`. It is a classification-style
pass-rate derived from a regression.

**Q: Why fit the scaler on train only?**
If the test set influences the min/max (or mean/std), information about the test
distribution leaks into training and the reported accuracy is optimistic. Fitting
on train and applying to test is the honest setup.

**Q: Why min-max and not z-score?**
The paper only says "normalised to a comparable scale". Its figure axes for the
NN output run ~0-1 ["arb."], which is what min-max gives. It is a guess - flagged
`# INFERRED` - and `--standard` shows the choice changes the result by <2 points.

**Q: Why is the output layer linear?**
It is a regression; the targets, once normalised, live in `[0,1]` and a linear
head can produce them directly. The paper's "activated by ELU" reads as the
generic per-layer description. `--elu-output` tests the other reading; ~same
result.

**Q: Full-batch or minibatch? Why does it matter?**
The paper says "1000 epochs" but not the batch size. Full-batch = 1000 weight
updates; minibatch-32 = ~7000. More updates -> better fit in the same epoch
budget, which is why minibatch gets Tox into the paper's band. Both are reported.

**Q: Why re-draw the split every run instead of fixing one?**
The paper reports a *range* over "10 independent training/testing runs". Re-drawing
the split captures both sources of variation (which rows are in test + random
init) and is the honest way to show the estimate's uncertainty.

**Q: Where does the model fail and why?**
Precursor temperature 55 C. Below ~60 C the TDMA-Hf vapour pressure is too low, so
growth is supply-limited and varies sharply across the wafer - the 9 points on
one wafer span more than the tolerance window itself. No smooth function of the 4
inputs can fit that, and a 3-NN interpolator fails there too (worse), on the same
points - so it is the data, not the network.

**Q: How much of the 90% is the model vs the tolerance being wide?**
Predicting a constant (the training mean) already scores 43% (Tox), 67% (RI),
39% (WER) within the paper's windows - that part is pure window width. The DNN
closes ~74-88% of the remaining gap, so it is adding real signal, but a simple
grid interpolator captures most of that.

**Q: Your numbers are below the paper. Failure?**
No - a finding. Tox reproduces. RI/WER are 4-10 points low with uniform random
splits; keeping the hard prec=55 C rows in training (as the paper's hand-drawn
splits appear to) recovers 99/98/89%. The difference is split composition.

**Q: What's the anomaly in the data?**
5 WER cells printed as 0.24-0.30 A/min between ~2.7 neighbours - a lost digit /
decimal shift in the SI figure. Kept verbatim in the CSV (it must match source),
flagged in `KNOWN_BAD_WER`, and shown to cost ~1 accuracy point.

---

## Part 7 - "Change a requirement on the spot" drills

Practise making these edits and predicting the effect.

**Add a 5th input (e.g. wafer radius `sqrt(x^2 + (y-7.5)^2)`).**
- `FEATURES` append the column; `build_dataset.py` or a derived column in
  `load_data`; change `DNN` `widths[0]` from 4 to 5. Retrain.
- Predict: little change - radius is a function of x,y which the net already has.

**Change the thickness tolerance to +/-0.5 nm.**
- `TOLERANCES["thickness_nm"] = 0.5`. Nothing else. Rerun step 2.
- Predict: Tox pass-rate drops to ~79% (you saw this in step 3's `acc<=tol/2`).

**Use 3 separate single-output networks instead of one 3-output net.**
- `DNN` `widths[-1] = 1`; loop over `TARGETS`, train one model each on that one
  normalised column; stack predictions.
- Predict: similar accuracy; you lose the shared representation but the targets
  are only loosely coupled here. More params, slower.

**Add early stopping.**
- In `train_model`, hold out ~15% of train as validation, track val MSE, keep the
  best-so-far `state_dict`, stop after N epochs without improvement.
- Predict: less overfitting, train accuracy drops from ~100%, test roughly flat
  or slightly up; removes the "1000 epochs" guess.

**Report RMSE instead of pass-rate.**
- Replace `tolerance_accuracy` with
  `{t: sqrt(mean((pred[:,i]-true[:,i])**2)) for ...}`. Lower is better; no
  tolerance needed. This is the "different but equally reasonable definition"
  from step 3 - Tox ~0.74 nm, WER ~1.76 A/min.

**Switch optimiser to plain SGD.**
- `torch.optim.SGD(model.parameters(), lr=..., momentum=0.9)`. Will need a bigger
  lr and probably more epochs; Adam is more forgiving on an unscaled-ish problem.

**Make the split stratified by precursor temperature.**
- In `split`, `groupby('prec_temp_C')` and sample the same fraction from each
  group into test. Guarantees prec=55 is represented proportionally every run;
  narrows the run-to-run spread; mean roughly unchanged.
