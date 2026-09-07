# Learning path: understanding this project from scratch

For someone new to data science who wants to genuinely understand every part of
this project and defend it in a live, no-AI walkthrough.

Work through it in order. Each concept is explained plainly, then tied to exactly
where it appears in this repo, then to a question you might get asked.

**Companion docs:** `report.md` (the findings), `GUIDE.md` (how to run/test the
code), `data/EXTRACTION_NOTES.md` (how the data was made).

---

## Contents

- Stage 0 - What this project is, in plain English
- Stage 1 - The data-science vocabulary you actually need (22 concepts)
- Stage 2 - The domain: what ALD is and why the numbers behave as they do
- Stage 3 - The assignment, requirement by requirement
- Stage 4 - The pipeline end to end, tied to the code
- Stage 5 - The results, and what each one means
- Stage 6 - A concrete study plan
- Stage 7 - Interview / walkthrough preparation

---

## Stage 0 - What this project is, in plain English

A research group ran a chemistry process (growing a thin film of hafnium oxide on
a silicon wafer) **315 times**, changing a few settings each time, and each time
measured **3 properties** of the resulting film. They then trained a **neural
network** to predict those 3 properties from the settings, so future films could
be predicted instead of run in the lab (each real run costs >$1000).

They published a headline claim: the network predicts each property correctly
**more than 90% of the time**, within a stated error margin.

Your job is not to build a model - a model is easy. Your job is to **check
whether that claim holds up**, understand exactly what it means, find where the
model fails, and be able to explain every line of your own code.

This repo does that in 5 steps mirroring the assignment:

1. **Get the data** out of the paper and check it is trustworthy.
2. **Rebuild the model** and see if you get the same accuracy.
3. **Interrogate the metric** - what does "90% accurate" actually mean, and is it
   impressive or just a generous ruler?
4. **Investigate the errors** - where and why does the model fail?
5. **Write it up** short.

---

## Stage 1 - The data-science vocabulary you actually need

22 concepts. For each: what it is, where it lives in this repo, a likely question.

### 1. Machine learning

Instead of writing rules by hand ("if temperature > 200 then thickness is
about..."), you show a program many examples of inputs paired with correct
outputs and let it **find the pattern itself**. The output is a "model": a
function you can feed new inputs to.

- *Here:* the model is a neural network in `src/ald.py`, class `DNN`.
- *Q: "Why ML and not a formula?"* Because the 3 properties depend on 4 settings
  in a tangled, non-linear way (settings interact); no simple formula fits all
  315 points well. The paper's Fig. 3 shows this.

### 2. Supervised learning

The flavour of ML where every training example has a **known correct answer**
(a "label"). The model learns to reproduce the answers, then is judged on new
examples it has not seen.

- *Here:* inputs = the 4 process settings; labels = the 3 measured properties.
- *Q: "Supervised or unsupervised?"* Supervised - we have measured targets.

### 3. Regression vs classification

**Classification** predicts a category (cat / dog, spam / not-spam).
**Regression** predicts a number (thickness = 12.4 nm).

- *Here:* pure regression - 3 continuous numbers per row.
- Note: the paper's *metric* turns the regression into a yes/no ("within
  tolerance?") and reports a percentage, which *looks* like classification
  accuracy. That sleight of hand is the subject of Stage 3, step 3.

### 4. Features and targets (X and y)

**Features** (a.k.a. inputs, predictors, `X`): what you know going in.
**Targets** (a.k.a. labels, outputs, `y`): what you want to predict.

- *Here, `src/ald.py`:*
  ```python
  FEATURES = ["dep_temp_C", "prec_temp_C", "x_cm", "y_cm"]      # 4
  TARGETS  = ["thickness_nm", "refractive_index", "wer_A_min"]   # 3
  ```
- *Q: "Why is wafer position a feature?"* Because the film grows unevenly across
  the wafer, so *where* you measure changes the answer - position carries real
  predictive information.

### 5. The dataset as a table

A "tidy" dataset has **one row per example** and **one column per variable**.
315 rows here, 7 columns (4 features + 3 targets).

- *Here:* `data/ald_hfox_dataset.csv`. Load with `load_data()`.
- It is a **full factorial grid**: every combination of 7 deposition temperatures
  x 5 precursor temperatures x 9 wafer positions = 315. No combination missing,
  none repeated. `src/checks.py` verifies this.
- *Q: "Why does the grid structure matter?"* Because a model only has to
  *interpolate* between points that are close together - which is why a simple
  nearest-neighbour method (concept 18) does almost as well as the neural net.

### 6. Train / test split

You cannot judge a model on the same data it learned from - it could just
memorise. So you **randomly hold back** part of the data ("test set"), train on
the rest ("training set"), and measure accuracy only on the held-back part.

- *Here:* paper uses 215 train / 100 test. `src/ald.py`, function `split(df, seed)`.
- The **cardinal rule**: nothing about the test set may influence training -
  including the scaling numbers (concept 8).
- *Q: "Why 215/100 and not, say, 250/65?"* We match the paper. More test points =
  less noisy accuracy estimate but less training data. 100 test points means each
  point is worth 1% of the score - hence the run-to-run wobble.

### 7. Generalisation and overfitting

**Generalisation** = doing well on new, unseen data. **Overfitting** = doing
great on training data by memorising its quirks, then failing on new data.

- *Here:* with 1000 epochs and ~11,700 parameters trained on 215 rows, the model
  fits the training set almost perfectly (~100%, the paper says so too) but
  scores lower on test. That gap is overfitting. It is expected and not
  disqualifying; it is why the *test* number is the one that counts.
- *Q: "Is this model overfit?"* Yes, mildly - train ~100% vs test ~91%. The
  paper's 3-fold cross-validation (concept 20) argues it is not *severe*.

### 8. Normalisation / feature scaling

The 4 features live on wildly different scales: temperature ~100-250, position
~-3 to 10. Many algorithms (neural nets, k-NN) work badly when one column's
numbers dwarf another's. **Scaling** rewrites every column onto a common range.

Two common ways:
- **Min-max**: `(value - min) / (max - min)` -> everything in `[0, 1]`.
- **Standardisation (z-score)**: `(value - mean) / std` -> mean 0, std 1.

- *Here, `src/ald.py`, class `MinMax`:* default min-max; `standard=True` switches
  to z-score. Both features **and** targets are scaled (the paper scales both).
  Predictions are scaled back to real units with `.inverse()` before scoring.
- **Fit on training data only.** The scaler's min/max come from the training set;
  the same numbers are then applied to the test set. If you computed them over
  all the data, information about the test set would leak into training. In the
  REPL you can see test values land slightly outside `[0,1]` - proof the scaler
  never saw them.
- *Q: "Why min-max and not z-score here?"* The paper only says "normalised to a
  comparable scale". Its figures show network outputs on a 0-1 axis, which is
  what min-max gives. It is an assumption - flagged `# INFERRED` - and swapping
  it changes the result by under 2 points.

### 9. Neural network (dense / fully-connected)

A stack of **layers**. Each layer takes a vector of numbers, multiplies it by a
matrix of **weights**, adds a **bias** vector, and passes the result through a
non-linear function (concept 10). Stacking several layers lets the network
represent complicated input-output relationships.

"Dense" / "fully-connected" = every number in one layer feeds every number in the
next. The layer sizes here: `4 -> 128 -> 64 -> 32 -> 16 -> 8 -> 3`. The `4` is
the inputs, the `3` is the outputs, the middle numbers ("hidden layers") are a
design choice from the paper.

- *Here, `src/ald.py`, class `DNN`:* built with `nn.Linear` (the weights+bias
  step) and `nn.ELU` (the non-linearity), chained with `nn.Sequential`.
- Parameter count ~11,700 (all the weights + biases). That is large relative to
  215 rows - hence overfitting.
- *Q: "Why do the layers shrink 128 -> 8?"* The paper says the taper avoids
  "distortion of net output values". Practically, a funnel forces the network to
  compress information toward the 3 outputs.

### 10. Activation function

Without a non-linear function between layers, stacking layers is pointless -
matrix times matrix is just another matrix, so the whole network would collapse
to a single linear model. The activation bends the signal so the network can
model curves and interactions.

- **ELU** ("exponential linear unit"): acts like the identity for positive
  inputs, and a smooth curve saturating at -1 for negative inputs. The paper
  chose it for its "smooth... stable gradient behaviour".
- **Output layer**: usually **no** activation for regression, so the network can
  output any real number. `src/ald.py` uses a linear output head; `elu_output=True`
  tests the alternative reading of the paper.
- *Q: "What does ELU do to a negative number?"* Squashes it smoothly toward -1
  (vs ReLU, which sets all negatives to exactly 0).

### 11. Loss function

A single number saying how wrong the model currently is, which training tries to
minimise. For regression the standard choice is **MSE** (mean squared error):
average of `(prediction - truth)^2` over the batch. Squaring punishes big misses
more than small ones.

- *Here, `src/ald.py`, `train_model`:* `loss_fn = nn.MSELoss()`, computed on the
  **scaled** targets (so all 3 properties contribute comparably).
- *Q: "Why squared, not absolute, error?"* Squared error has a smooth derivative
  everywhere and penalises outliers harder; it is the conventional default. Mean
  absolute error is a valid alternative and would down-weight the WER tail.

### 12. Gradient descent and backpropagation

**Gradient descent**: to minimise the loss, compute which direction to nudge
every weight to make the loss go down a little, take a small step, repeat
thousands of times. **Backpropagation** is the efficient algorithm that computes
those directions ("gradients") for every weight in one backward pass.

- *Here, the training loop in `train_model`:*
  ```python
  opt.zero_grad()                      # forget last step's gradients
  loss_fn(model(X[b]), y[b]).backward()  # forward pass + backprop -> gradients
  opt.step()                           # nudge every weight downhill
  ```
- *Q: "What does `zero_grad` do and why?"* PyTorch **accumulates** gradients by
  default; without clearing them each step you would add this step's gradient to
  all previous ones and the update would be wrong.

### 13. Optimiser and learning rate

The **optimiser** decides *how* to turn gradients into weight updates. **SGD** is
the plain version. **Adam** adapts the step size per weight using a running
average of recent gradients - more forgiving, the common default. The
**learning rate** is the overall step-size knob (too big = diverge, too small =
crawl). `1e-3` (0.001) is Adam's usual default.

- *Here:* `torch.optim.Adam(model.parameters(), lr=1e-3)`. Both are `# INFERRED`
  - the paper does not state them.
- *Q: "How would you switch to SGD?"* `torch.optim.SGD(..., lr=..., momentum=0.9)`
  and probably raise the learning rate and epoch count.

### 14. Epoch, batch, iteration

- **Epoch**: one full pass over all the training rows.
- **Batch** (minibatch): the training rows are chopped into chunks; the model
  updates its weights once per chunk.
- **Iteration / step**: one weight update = one batch.

With 215 training rows: **full-batch** (one chunk of 215) = 1 update per epoch =
1000 updates over 1000 epochs. **Minibatch of 32** = ceil(215/32) = 7 updates per
epoch = ~7000 updates. **Same epoch count, 7x the learning.** This is why "1000
epochs" alone is ambiguous, and why Tox only reaches the paper's band with
minibatch.

- *Here:* `train_model(..., batch_size=None)` = full batch; `batch_size=32` =
  minibatch. `step2_reproduce.py --batch-size 32` is the headline setting.
- *Q: "What is the difference between an epoch and an iteration?"* An epoch is one
  sweep of the data; an iteration is one weight update on one batch. They are
  equal only when batch size = dataset size.

### 15. Random seeds and reproducibility

Training involves randomness: initial weights are random, the order batches are
drawn is random, the train/test split is random. A **seed** fixes that randomness
so the exact run can be repeated.

- *Here:* `np.random.default_rng(seed)` for the split, `torch.manual_seed(seed)`
  for weights, a seeded `torch.Generator` for batch order. Run any script twice
  with the same flags -> identical numbers (`GUIDE.md` Test 7).
- *Q: "Why do your 10 runs give different accuracies if everything is seeded?"*
  Because each run uses a **different** seed (0,1,2,...), so a different split and
  a different weight init. The spread across seeds is the point - it measures how
  much the answer depends on luck.

### 16. Evaluation metrics

Different ways to score predictions against truth:
- **The paper's "tolerance accuracy"**: percentage of test rows where
  `|prediction - truth|` is within a fixed window (`+/-1.0 nm`, `+/-0.04`,
  `+/-0.9 A/min`). A pass/fail rate.
- **MAE** (mean absolute error): average size of the miss, in real units.
- **RMSE** (root mean squared error): like MAE but inflated by big misses; if
  RMSE >> MAE there is a heavy tail of bad predictions.
- **Median absolute error**: the typical miss, ignoring outliers.
- **R^2** ("R squared"): fraction of the target's variance the model explains;
  1.0 = perfect, 0 = no better than predicting the mean, negative = worse.

- *Here:* `tolerance_accuracy` in `src/ald.py` is the paper's metric.
  `src/step3_metric.py` also computes MAE / RMSE / medAE / R^2 to show what the
  pass-rate hides. Example: WER median error 0.38 but RMSE 1.76 - most
  predictions great, a few terrible.
- *Q: "Which metric is best?"* None absolutely. The pass-rate answers "how often
  is it good enough for this application"; RMSE/R^2 answer "how good is it
  overall". A serious report shows more than one, which is the point of step 3.

### 17. Baselines

A number is only impressive **compared to something**. A **baseline** is a
deliberately simple method you score the same way, chosen *before* you look at its
result, so you can say "the fancy model beats this by X".

- *Here, `src/step3_metric.py`, `ref_predictions`:*
  - **predict the mean**: always output the average of the training targets.
    Its pass-rate tells you how many test points are within the tolerance of a
    *constant* - i.e. how much of the score is just a wide window.
  - **linear regression** (concept 19).
  - **k-NN** (concept 18).
- *Q: "Why pick the baseline before seeing its score?"* So you cannot
  unconsciously choose the baseline that flatters your model. It is a
  pre-registration discipline.

### 18. k-nearest neighbours (k-NN)

To predict for a new point, find the `k` training points closest to it (in scaled
feature space) and average their target values. No training, no parameters - just
a lookup. On a **dense regular grid** it is essentially local interpolation and
is therefore a *strong* baseline here.

- *Here:* `KNeighborsRegressor(n_neighbors=3)` in `step3_metric.py` and
  `step4_errors.py`.
- **Key finding:** 3-NN scores 84 / 86 / 68% within the paper's tolerances,
  close to the neural net's 93 / 91 / 85. The net's advantage *over simply
  looking up nearby experiments* is modest.
- *Q: "Why is k-NN such a good baseline in this project?"* Because the data is a
  complete 7x5x9 grid; any new point is surrounded by measured neighbours, so
  averaging them is already accurate. It answers: "is the network learning
  physics, or just interpolating a grid?"

### 19. Linear regression

The simplest predictive model: output = weighted sum of inputs + constant. Fits a
straight line / flat plane through the data. If a complex model barely beats
linear regression, the problem was mostly linear and the complexity is not
earning its keep.

- *Here:* `LinearRegression()` baseline in `step3_metric.py`. It scores ~50-58% -
  well below the network, confirming the problem is genuinely non-linear (so the
  network *is* justified over a linear model, even if not over k-NN).

### 20. Cross-validation

Instead of one train/test split, cut the data into `k` folds; train on `k-1`,
test on the held-out fold; rotate so every fold is the test set once; average.
Gives a more stable accuracy estimate on small datasets and reduces the chance
one lucky split flatters you.

- *Here:* not re-implemented (time-boxed), but the paper's Supplementary Fig. 3
  uses 3-fold CV. This repo instead runs 10 random splits and reports the
  spread, which serves the same purpose - showing the estimate's uncertainty.
- *Q: "Why not just do one split?"* One split's score is noisy, especially with
  100 test points; repeating shows how much of "91%" is real and how much is
  luck.

### 21. Uncertainty of an estimate

Any accuracy you measure is itself uncertain. With only 100 test points, one
misclassified point moves the score by 1%. Reporting a single "91%" hides this;
reporting **mean and range over many runs** ("91%, range 87-95") is honest.

- *Here:* every script takes `--runs N` and reports `mean / min / max`. The WER
  spread (76-88 in one sweep) is wide *because* the hard prec=55 C points land in
  the test set in random amounts each run.

### 22. Data quality and verification

"Garbage in, garbage out." Before modelling, you check the data is real:
structural checks (right shape, no duplicates, complete grid), range checks
(values physically plausible), and **source spot-checks** (does the extracted
number match the original document?).

- *Here:* `src/checks.py` + the second full read-through documented in
  `data/EXTRACTION_NOTES.md`. This caught **5 corrupted WER values** in the
  paper's own supplementary figure (printed as 0.24-0.30 where neighbours are
  ~2.7 - a lost digit). Finding plausible-looking errors is explicitly what the
  assignment rewards.

---

## Stage 2 - The domain: what ALD is

You do **not** need chemistry to do this project, but a mental picture stops the
numbers feeling arbitrary.

### Atomic layer deposition (ALD)

A way to grow an extremely thin, even coating on a surface, **one atomic layer at
a time**. You pulse in gas A (here: a hafnium-carrying molecule, "TDMA-Hf"), it
sticks to the surface and *stops* once the surface is covered (self-limiting).
You purge, then pulse gas B (water), which reacts to leave a layer of hafnium
oxide. Repeat - here, 100 cycles per run. Because each pulse self-limits, you get
sub-nanometre thickness control. This is how insulating layers in modern chips
are made.

### The film: hafnium oxide (HfOx)

A "high-k dielectric" - an electrical insulator used in transistor gates. Chip
makers need it **thin, uniform, and dense**. The three measured properties are
proxies for that:

| property | symbol | what it measures | good value |
|---|---|---|---|
| thickness | Tox | how thick the film is (nm) | as targeted, uniform across wafer |
| refractive index | RI | how much it slows light ~ how dense/pure the film is | higher = denser (~2.1 for good HfOx) |
| wet etch rate | WER | how fast dilute acid dissolves it (A/min) | **lower** = denser, higher quality |

RI and WER both report **film density** from different angles: a dense film bends
light more (high RI) and resists acid (low WER). They move oppositely.

### The four knobs

- **Deposition temperature** (100-250 C): the wafer/chamber temperature. Too low
  -> reactions do not complete, film is loose and impure. Too high -> the
  precursor decomposes. Somewhere in between is the "**ALD window**" where growth
  is clean and self-limiting.
- **Precursor temperature** (55-75 C): how hot the hafnium-source bottle is,
  which sets how much hafnium vapour is delivered each pulse. Below ~60 C there
  is **not enough vapour** - the surface is starved, growth depends on where on
  the wafer you are (near the inlet vs far), and the film is very non-uniform.
  This is the regime where your model fails.
- **x, y position** on the 4-inch wafer (9 points): lets the model account for
  the film being different near the gas inlet vs the far edge.

### Why the low-precursor-temperature rows are the whole story

At precursor temperature 55 C the film properties vary enormously **across a
single wafer** (thickness from 3 nm near one edge to 15 nm at the centre). The
nine points of one wafer span a wider range than the model's entire tolerance
window. No smooth function of just `(dep_temp, prec_temp, x, y)` can hit all nine
- there is fine spatial structure the 4 coordinates do not capture. Above 60 C
the process saturates, films are uniform, and the model is near-perfect. Stage 5
shows this in numbers.

---

## Stage 3 - The assignment, requirement by requirement

Read `ASSIGNMENT.1.pdf` alongside this. What each part asks, and what it is
really testing.

### "1. Prepare and check the data"

*Asks:* extract the 315 points into a machine-readable file; run structural
checks and source spot-checks; record what you checked and what uncertainty
remains.

*Really testing:* do you **verify against the source**, or accept data that
merely looks fine? The 5 corrupted WER values are the trap - they are subtle
(right order of magnitude if you are not paying attention).

*This repo:* `src/build_dataset.py` (the transcription), `src/checks.py`
(structural), `data/EXTRACTION_NOTES.md` (what was checked, the anomaly, residual
uncertainty). Every one of the 945 numbers was read twice against the figure.

### "2. Reproduce the headline result"

*Asks:* rebuild the model as described, reproduce the accuracy claim, report what
you get. If it differs, say by how much and why. **Do not tune until it
matches.** Separate stated choices from inferred ones. Report variability across
runs.

*Really testing:* can you follow an under-specified recipe honestly, and treat a
mismatch as a *finding* rather than something to hide by fiddling knobs?

*This repo:* `src/step2_reproduce.py`. Result: Tox reproduces (~93%), RI ~91%,
WER ~84% (paper 92-95 / 95-97 / 90-95). Every inferred choice is `# INFERRED` in
`src/ald.py` and swept in `report.md`. Variability reported as min/max over 10
runs.

### "3. Interrogate the metric"

*Asks:* work out exactly what "accuracy above 90% within +/-1.0 nm" means. Then:
(a) what would the number be under a different but equally reasonable definition?
(b) how much of the score is the model vs the tolerance being wide relative to
the data spread - design something to separate them and run it. (c) compare
against at least one simple baseline chosen before you see its result.

*Really testing:* do you understand that a metric is a *choice*, and can you
design a comparison that isolates one effect?

*This repo:* `src/step3_metric.py`.
- (a) recomputed under strict `<`, half/double tolerance, "within 10% relative",
  "within 0.5 std", plus MAE/RMSE/R^2.
- (b) the "predict the mean" baseline already scores 43 / 67 / 39% within the
  paper's windows - that part is pure window width. A **skill score**
  `(model - baseline) / (100 - baseline)` isolates the model's contribution.
- (c) baselines fixed in advance: mean, linear regression, 3-NN.

### "4. Investigate the errors"

*Asks:* find where the model does well and where it fails; test at least one
explanation; prefer a small analysis that changes your conclusion over many
plots.

*Really testing:* can you form a hypothesis about a failure and actually test it,
rather than just describing plots?

*This repo:* `src/step4_errors.py`. Pools 1000 test predictions, stratifies by
precursor temp / deposition temp / wafer radius. Finding: misses concentrate
almost entirely at precursor temp 55 C. Two tests: (A) within-wafer spread by
precursor temp - is the surface intrinsically steep there? (B) does 3-NN miss the
same points - is it the model or the data? Both point to "the data in that
regime", not the network.

### "5. Write it up" + deliverables

*Asks:* max 2 pages. Must state what you found up front, every assumption
(including obvious ones), and what you got wrong or could not finish. Plus: a git
repo with a README that reproduces from a clean checkout; a daily log; a 30-min
live walkthrough with no AI.

*Really testing:* can you communicate concisely, be explicit about assumptions,
and own your gaps?

*This repo:* `report.md`, `README.md`, `daily_log.md`, and this whole repo.

### What the brief says it is grading

From `ASSIGNMENT.1.pdf`, "What we are looking for": (1) verification against
something real, (2) assumptions made explicit, (3) what failed, (4) knowing what
your code does - you must explain and modify any line live, (5) scoping / ruthless
prioritisation. **Not** graded: amount of code, speed, extra features, polish on
easy parts.

---

## Stage 4 - The pipeline end to end, tied to the code

Follow one prediction from raw data to a score. Open `src/ald.py` beside this.

1. **Load** - `load_data()` reads `data/ald_hfox_dataset.csv` into a table of 315
   rows x 7 columns. Two asserts guard the shape.

2. **Split** - `split(df, seed)` shuffles the 315 row-numbers with a seeded random
   generator, takes the first 100 as test, the rest (215) as train. Same seed ->
   same split, always.

3. **Fit scalers on training data** - `run_once` creates two `MinMax` scalers,
   one for the 4 feature columns, one for the 3 target columns, and fits them
   **on the training rows only** (learns each column's min and max).

4. **Transform** - both train and test features are mapped to `[0,1]` with the
   *training* min/max; likewise the targets. Now every column is comparable in
   size, which is what gradient-based training needs.

5. **Build the network** - `DNN()` creates the layer stack
   `Linear(4,128) -> ELU -> ... -> Linear(8,3)`. ~11,700 random initial weights.

6. **Train** - `train_model` runs the loop: for 1000 epochs, shuffle the training
   rows, walk them in batches of 32, and for each batch do
   `zero_grad -> forward -> MSE loss -> backward -> optimiser step`. Weights
   crawl downhill on the loss. ~7000 updates total.

7. **Predict** - `predict` runs the trained network on the **scaled test
   features**, giving scaled predictions in `[0,1]`-ish.

8. **Un-scale** - `ys.inverse(...)` maps predictions back to real units (nm,
   index, A/min) using the training target min/max.

9. **Score** - `tolerance_accuracy` compares predictions to the true test
   targets: for each property, the percentage of the 100 test rows within that
   property's tolerance window. Returns e.g. `{thickness: 93.0, ri: 91.0,
   wer: 85.0}`.

10. **Repeat** - `step2_reproduce.py` does steps 2-9 for seeds 0..9 and reports
    the mean and range. `step3` and `step4` reuse `run_once` and slice the
    predictions differently.

---

## Stage 5 - The results, and what each one means

### Reproduction (step 2)

| property | this repo (10 random splits) | paper |
|---|---|---|
| thickness | 93%, range 91-97 | 92-95% |
| refractive index | 91%, range 87-95 | 95-97% |
| wet etch rate | 84%, range 76-88 | 90-95% |

*Meaning:* thickness reproduces. RI is ~4 points low, WER ~6-10 low. This holds
across all four inferred choices (batch size, scaler type, output activation), so
it is not a guessing error. The range is wide because the test set is small and
the hard points land in it in random amounts.

### Metric under other definitions (step 3, part a)

- Halve the tolerance: thickness 93 -> 79%, WER 85 -> 60%. The headline is
  sensitive to where you draw the line.
- "Within 10% relative" instead of a fixed window: RI -> 99% (its values are ~2.1
  so 10% is a huge 0.21 window), WER -> 49% (its values are ~2-3 so 10% is a tiny
  0.2-0.3 window). An *equally defensible* definition flips WER from "pass" to
  "fail".
- Underlying error sizes: thickness R^2 0.90 / RMSE 0.74 nm (genuinely good);
  WER R^2 0.53 / median error 0.38 but RMSE 1.76 (good centre, heavy tail).

### Model vs tolerance width (step 3, part b/c)

| baseline | thickness acc / skill | RI acc / skill | WER acc / skill |
|---|---|---|---|
| predict the mean | 43% / 0.88 | 67% / 0.74 | 39% / 0.74 |
| linear regression | 51% / 0.86 | 58% / 0.79 | 51% / 0.69 |
| 3-NN interpolation | 84% / 0.58 | 86% / 0.36 | 68% / 0.52 |

*Meaning:* the model clearly beats "predict a constant" and "fit a plane" (skill
0.7-0.9), so it is learning real structure. But a naive **3-NN interpolator on
the grid** already gets 84 / 86 / 68%; the model's extra skill over "look up the
nearest measured wafers" is only 0.36-0.58. On a dense factorial grid, much of
the accuracy is interpolation plus a generous window (the tolerance is ~0.3-0.4
of a standard deviation).

### Where it fails (step 4)

| precursor temp | thickness hit-rate | RI | WER |
|---|---|---|---|
| >= 60 C | 99-100% | 99-100% | 89-93% |
| 55 C | 67% | 57% | 58% |

*Meaning:* restricted to the stable ALD regime, the model **meets or beats the
paper**. All the shortfall is the precursor-temp = 55 C rows (20% of the data).

- **Test A:** within-wafer spread at 55 C is 5-12x larger than at >=60 C, and for
  RI/WER it exceeds the tolerance window itself. The nine points of one such
  wafer cannot all fit inside the window for any smooth 4-input function.
- **Test B:** 3-NN also fails at 55 C, worse, on the same points. So it is the
  data (a steep, under-sampled surface), not the network.
- **Split test:** if you keep the 55 C rows in the training set (as the paper's
  hand-drawn splits appear to), random-split accuracy jumps to 99 / 98 / 89%.
  That is the most likely explanation for the paper's higher RI/WER: split
  composition, not a better model.

### The data anomaly

5 WER values printed as 0.24-0.30 A/min sitting between ~2.7 neighbours - a lost
digit in the paper's figure. Kept verbatim in the CSV (it must match the source),
flagged in `KNOWN_BAD_WER`, and shown to cost ~1 accuracy point.

---

## Stage 6 - A concrete study plan

Roughly 8-12 focused hours, spread over a few days.

**Session 1 (2 h) - vocabulary.** Read Stage 1 slowly. For each concept, open
`src/ald.py`, find where it appears, read that code. Do not move on until you can
say the concept in one sentence without notes.

**Session 2 (1.5 h) - run everything.** Follow `GUIDE.md` Part 4, tests 1-7. Watch
the numbers. Re-read Stage 5 as each script finishes so the output has meaning.

**Session 3 (2 h) - the REPL.** `GUIDE.md` Part 3. Type every line yourself. Then
go further: change one thing (a tolerance, the batch size, a layer width),
predict what will happen, run it, see if you were right. Being wrong here is how
you learn.

**Session 4 (2 h) - the code, line by line.** `GUIDE.md` Part 2 (the `ald.py`
walkthrough). For every function write, in your own words, a one-line comment of
what it does and why. Pay special attention to: the scaler fit-on-train,
`zero_grad`, the epoch/batch loop, and `tolerance_accuracy`.

**Session 5 (1.5 h) - the argument.** Read `report.md` end to end. For each
finding, trace it back to the script and the numbers that support it. Practise
saying the four-line summary out loud: data verified + 1 anomaly; Tox
reproduces, RI/WER low; the gap is the 55 C regime; the metric leans on a wide
window and grid interpolation.

**Session 6 (1.5 h) - the drills.** Stage 7 question bank + `GUIDE.md` Part 7
modification drills. Actually make the edits.

**Rule:** if you cannot explain a line, do not leave it in. Delete it or replace
it with something you understand. The walkthrough is graded on *your*
understanding, not the code's cleverness.

---

## Stage 7 - Interview / walkthrough preparation

### The format (from the brief)

30 minutes, live, **no AI**. They will (1) pick parts of your code and ask why
they are the way they are, and (2) change a requirement and ask you to modify the
code on the spot. Also expect questions on your assumptions and what failed.

### How to talk about it

- **Lead with the finding, then the evidence.** "Thickness reproduces; RI and WER
  are a few points low, and that whole gap is the low-precursor-temperature
  regime - here's the stratified table."
- **Name assumptions before they ask.** "The paper doesn't state the batch size,
  so I tried both; here's what changes."
- **A mismatch is a result.** Never say "I couldn't get it to match". Say "I get
  X, the paper gets Y, the difference is explained by Z, and I chose not to tune
  because the brief says so."
- **Say what you don't know.** "I didn't reproduce their exact split because the
  point membership isn't published; I showed a split that reproduces their
  numbers is plausible."

### Core question bank (answers in your own words)

1. What problem is the model solving? (4 settings -> 3 film properties, regression)
2. What is the metric, exactly? (percent of test points within a fixed error
   window, per property)
3. Why split into train and test? What is the cardinal rule? (judge on unseen
   data; the test set must never influence training, including the scaler)
4. Why scale the inputs? Why fit the scaler on train only? (comparable
   magnitudes for gradient descent; fitting on all data leaks test info)
5. Why min-max and not z-score? (assumption; paper says only "comparable scale";
   swap changes result <2 pts)
6. What does ELU do, and why no activation on the output? (smooth nonlinearity;
   linear head so the network can output any real number for regression)
7. Walk me through your training loop. (zero_grad -> forward -> loss -> backward
   -> step, 1000 epochs, batches of 32)
8. Why does `zero_grad` matter? (PyTorch accumulates gradients)
9. Full-batch vs minibatch - why does it change the result? (same epochs, 7x more
   weight updates; minibatch gets Tox into the paper's band)
10. Your 10 runs disagree - why, if it's all seeded? (different seed per run ->
    different split and init; the spread is the uncertainty)
11. How much of "90%" is the model and how much is a generous tolerance? (predict
    the mean already scores 43/67/39%; skill score isolates the model)
12. Why is 3-NN such a strong baseline? (dense grid -> every new point is
    surrounded by measured neighbours)
13. Where does the model fail and why? (precursor temp 55 C; within-wafer spread
    exceeds the tolerance; 3-NN fails there too -> it's the data)
14. What's wrong with the dataset? (5 WER values printed ~10x too low; a figure
    error; kept verbatim, flagged, costs ~1 pt)
15. What would you do with two more days? (reproduce their exact split;
    3-fold CV; a regime-split or heteroscedastic model for the WER tail)

### "Modify it on the spot" - rehearse these

From `GUIDE.md` Part 7. Be able to do each in under 2 minutes and say the
expected effect:

- change a tolerance value (one line in `TOLERANCES`; pass-rate moves predictably)
- switch full-batch <-> minibatch (the `batch_size` argument)
- add a 5th input feature (edit `FEATURES`, `widths[0]`)
- swap the metric to RMSE (rewrite `tolerance_accuracy`)
- use 3 separate 1-output networks instead of one 3-output net
- add early stopping (hold out a validation slice, track its loss, keep the best)
- make the split stratified by precursor temperature

### The one thing they are really checking

That you can be handed your own submission cold and **reason about it** - explain
a choice, predict the effect of a change, make the change. Everything above is in
service of that.
