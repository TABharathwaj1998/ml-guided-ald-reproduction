# Daily log

One or two lines per day. AI errors noted where they happened.

## Day 1 - data extraction & checks
- Confirmed the SI `.docx` has no data tables; the 315 points are printed on the
  wafer-map images in Supplementary Fig. 1. Extracted the three panels with
  PyMuPDF and transcribed all 315 x 3 values by eye.
- Wrote `src/build_dataset.py` (transcription of record) + `src/checks.py`
  (structural checks). All hard checks pass.
- Caught 5 WER cells printed ~10x too low (0.24-0.30 between ~2.7 neighbours) -
  treated as an SI figure-export error, kept verbatim, flagged.
- AI note: first instinct was to look for a text table in the `.docx`; there is
  none. Verifying that directly saved time chasing a conversion path.

## Day 2 - reproduce headline (step 2)
- Built `src/ald.py` (data / min-max scaler / 128-64-32-16-8 ELU net / metric)
  and `src/step2_reproduce.py`.
- Full-batch, 1000 epochs, min-max, 10 random splits:
  Tox 91.4%, RI 92.1%, WER 82.4% (paper 92-95 / 95-97 / 90-95).
- Minibatch 32 (Keras default; ~7x more updates per "epoch"):
  Tox 93.2%, RI 91.3%, WER 84.6%. Tox now in band; RI and WER still short.
- z-score: 93.4 / 90.8 / 79.9. ELU-output: 93.6 / 92.4 / 82.8. None of the
  inferred-choice variants close the RI/WER gap - it's robust.
- Conclusion: reproduce Tox; RI ~4 pts low, WER ~6-10 pts low. Treated as a
  finding, not tuned.

## Day 3 - interrogate the metric (step 3)
- `src/step3_metric.py`: pass-rate under alternative definitions (strict `<`,
  tol/2, 2*tol, 10% relative, 0.5*sigma) + MAE / RMSE / R2 / median the pass-rate
  hides. Tox R2 0.90, RI 0.76, WER 0.53. Halving tol: Tox 79%, WER 60%.
- Baselines fixed before looking: predict-mean (43/67/39% within paper tol),
  linear (51/58/51%), 3-NN interpolation (84/86/68%).
- Finding: DNN clearly beats mean/linear (skill 0.7-0.9) but its skill *over
  3-NN interpolation* is only 0.58/0.36/0.52. On a dense factorial grid most of
  the score is interpolation + a generous window (tol ~= 0.3-0.4 sigma).

## Day 4 - error investigation (step 4)
- `src/step4_errors.py`: pooled 1000 test residuals, stratified.
- Misses concentrate at precursor temp = 55 C: hit-rate 67/57/58% there vs
  99.5/99.8/91.0% at prec>=60 C. Deposition-temp gradient is the same regime.
  No wafer-radius effect.
- Test A: within-wafer target std at prec=55 C is 5-12x larger and exceeds the
  RI/WER tolerance windows themselves. Test B: 3-NN fails there too, worse, on
  the same points -> difficulty is in the data, not the DNN.
- Extra: forcing prec=55 C into training lifts random-split accuracy to
  99/98/89%. Best explanation for the paper gap = split composition (their
  Fig. 4a splits look hand-made). 5 corrupted WER cells cost ~1 pt.

## Day 5 - write-up
- Report, README, repo cleanup.
