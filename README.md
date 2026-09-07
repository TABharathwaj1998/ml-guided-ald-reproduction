# HfOx ALD - reproduction of Lee et al. (Commun. Mater. 2026)

Reproduces the headline result of "Machine learning-guided optimization of atomic
layer deposition process" (DOI 10.1038/s43246-026-01206-w): a 4-input,
3-output dense network predicting HfOx film thickness, refractive index, and wet
etch rate from ALD process parameters.

See **`report.md`** for findings and **`daily_log.md`** for the day-by-day log.
**`LEARNING.md`** is a from-scratch guide to the data-science background and the
project; **`GUIDE.md`** is how to run and test the code.

## Layout

```
data/
  ald_hfox_dataset.csv     315 rows, extracted from Supplementary Fig. 1
  EXTRACTION_NOTES.md       how it was extracted, checks, the 5 flagged WER cells
  source_panels/            the SI figure panels the numbers were read from
src/
  ald.py                    core: data, scaler, the DNN, training, the metric
  build_dataset.py          transcription of record -> writes data/ald_hfox_dataset.csv
  checks.py                 step 1: structural checks on the CSV
  step2_reproduce.py        step 2: headline accuracy, N runs, random splits
  step3_metric.py           step 3: metric under alternative defs + baselines
  step4_errors.py           step 4: where/why the model fails
report.md, daily_log.md, requirements.txt
```

## Reproduce from a clean checkout

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/mac: .venv/bin/pip
# torch is the CPU wheel; the whole thing runs on CPU in a few minutes.

cd src        # scripts import ald.py by name, so run them from here
python checks.py                          # step 1: all hard checks pass
python step2_reproduce.py --runs 10 --batch-size 32     # step 2
python step3_metric.py   --runs 10 --batch-size 32      # step 3
python step4_errors.py   --runs 10 --batch-size 32      # step 4
```

`build_dataset.py` is run once from the repo root (`python src/build_dataset.py`)
and only regenerates `data/ald_hfox_dataset.csv` from the hand-transcribed tables
inside it. You do not need to run it to reproduce the analysis.

## Key options (all scripts)

| flag | meaning | default |
|---|---|---|
| `--runs N` | independent runs, fresh random split each | 10 |
| `--batch-size B` | minibatch size; omit for full-batch | full (step2) / 32 (step3,4) |
| `--standard` | z-score scaling instead of min-max | off (min-max) |
| `--elu-output` | ELU on the output layer instead of linear | off (linear) |
| `--epochs E` | training epochs | 1000 |

## What is stated vs inferred

The paper fixes the architecture, ELU, biases, normalisation, 215/100 split, 1000
epochs, and the accuracy metric. Optimiser, learning rate, loss, batch size,
scaler type, output activation, and weight init are not stated; every such choice
is marked `# INFERRED` in `src/ald.py` and the main ones are swept in `report.md`.

## Headline result (10 random splits, minibatch-32)

| property | tol | this repo | paper |
|---|---|---|---|
| thickness | +/-1.0 nm | 93 +/- 2% | 92-95% |
| refractive index | +/-0.04 | 91 +/- 2% | 95-97% |
| wet etch rate | +/-0.9 A/min | 84 +/- 4% | 90-95% |

The refractive-index and WER gaps are entirely the precursor-temperature = 55 C
regime; restricting to precursor temperature >= 60 C gives 99.5 / 99.8 / 91.0%.
See `report.md`.
