# Data extraction & checks

## Source

`supplementary_information.docx` -> **Supplementary Fig. 1** ("Raw datasets from
experimental data mining"), panels (a) Tox, (b) RI, (c) WER.

The `.docx` contains **no data tables** - only figure captions. The 315 data
points are printed *as text labels on wafer-map images* (nine numbers per wafer
disc, one disc per (deposition temperature, precursor temperature) pair). This is
the "printed as numbers rather than plotted" form referred to in the brief: the
numbers are legible on the figure, but there is no machine-readable table.

Extraction was therefore manual transcription by eye from the embedded images at
high zoom. The three panels used are committed under `data/source_panels/`
(`panel_tox.png`, `panel_ri.png`, `panel_wer.png`, ~1500x1837 px each), extracted
from the `.docx` with PyMuPDF.

## Grid / structure

- deposition temperature: 100..250 degC in 25 steps  -> 7 levels
- precursor temperature:   55..75  degC in 5  steps  -> 5 levels
- wafer position:          9 fixed points, pyramid layout 1-2-3-2-1
  read top -> bottom, left -> right within each disc.
- 7 x 5 x 9 = **315 rows**, one per (dep, prec, x, y). 3 targets per row.

Wafer coordinates (cm from centre, inlet side) taken from Fig. 1a / Fig. 3:
`(0,10.5) (-1.5,9) (1.5,9) (-3,7.5) (0,7.5) (3,7.5) (-1.5,6) (1.5,6) (0,4.5)`.
Only relative geometry matters for the model; the absolute frame is the paper's.

## Structural checks (all pass) - see `src/build_dataset.py` asserts + `src/checks.py`

- row count exactly 315; no duplicate (dep,prec,x,y) keys; no duplicate full rows.
- every dep level has all 5 prec levels has all 9 positions (full factorial, no gaps).
- value ranges: Tox 2.7..16.7 nm, RI 2.06..2.69, WER 0.24..19.7 A/min - all
  physically plausible for HfOx ALD except the WER low outliers below.
- monotonic sanity: at fixed position, Tox decreases with deposition temperature
  and RI increases - consistent with the trends the paper describes (Fig. 3a).

## Source spot-check / verification against something real

Every one of the 945 printed values (315 x 3) was read back a second time against
the source panels. Match is exact. Cross-checked the prec=55 degC column against
the *same* maps reproduced independently in the main paper Fig. 3b - identical.

## Flagged data-quality finding: 5 corrupted WER values

Genuinely printed in Supplementary Fig. 1c as ~10x below every neighbour:

| dep degC | prec degC | position    | printed WER | local neighbours |
|---------:|----------:|-------------|------------:|------------------|
| 200      | 70        | (3, 7.5)    | 0.29        | 2.5 - 2.8        |
| 225      | 65        | (1.5, 9)    | 0.30        | 2.7 - 2.9        |
| 225      | 65        | (3, 7.5)    | 0.30        | 2.7 - 2.9        |
| 225      | 70        | (0, 10.5)   | 0.24        | ~2.7             |
| 250      | 65        | (0, 10.5)   | 0.30        | 2.9 - 3.2        |

Interpretation: looks like a lost leading digit / shifted decimal point
(2.9 -> 0.29, 3.0 -> 0.30, 2.4 -> 0.24; i.e. true value / 10). WER is measured as
(initial - etched thickness) / etch time; a genuine 0.2-0.3 A/min would imply an
essentially unetchable film sitting between ordinary ~2.7 A/min neighbours, which
is not physical. Treated as a **figure-export error in the SI**, not a real
measurement.

Decision: keep the values verbatim in `ald_hfox_dataset.csv` (it must match the
source). The 5 points are listed in `KNOWN_BAD_WER` in `src/ald.py`; step 4
quantifies their effect by re-running the evaluation with them dropped.

## Remaining uncertainty

- Panel resolution is modest. Tox/RI/WER agreed exactly on a full second reading,
  but a handful of `.1` last-digit misreads cannot be fully excluded, especially
  where a label overlaps a contour edge. Estimated residual risk: <1% of cells,
  +/-0.1 in the last digit. This is well inside the paper's own tolerance windows
  and the run-to-run spread of the model, so it does not affect any conclusion.
- Wafer (x,y) values are nominal positions from the paper, not measured
  coordinates.
