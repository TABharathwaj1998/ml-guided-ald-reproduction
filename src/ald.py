"""Core library for the HfOx ALD reproduction: data, model, training, metric.

Everything the assignment scripts need lives here. Kept deliberately small so
every line can be explained in the walkthrough. Choices the paper does NOT state
are marked  # INFERRED.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "data" / "ald_hfox_dataset.csv"

# --- problem definition (all stated in the paper) ----------------------------
FEATURES = ["dep_temp_C", "prec_temp_C", "x_cm", "y_cm"]
TARGETS = ["thickness_nm", "refractive_index", "wer_A_min"]

# Headline tolerance windows (paper abstract + Fig. 4): a prediction counts as
# "correct" for a target if |y_pred - y_true| <= tol, in PHYSICAL units.
TOLERANCES = {"thickness_nm": 1.0, "refractive_index": 0.04, "wer_A_min": 0.9}

N_TEST = 100          # paper: 215 train / 100 test
N_TRAIN = 215

# 5 WER cells printed ~10x too low in Supplementary Fig. 1c (value looks like the
# true one / 10). Keys are (dep_temp_C, prec_temp_C, x_cm, y_cm).
# See data/EXTRACTION_NOTES.md. Used by step 4 to measure their effect.
KNOWN_BAD_WER = [
    (200, 70, 3.0, 7.5),
    (225, 65, 1.5, 9.0),
    (225, 65, 3.0, 7.5),
    (225, 70, 0.0, 10.5),
    (250, 65, 0.0, 10.5),
]


def load_data(path=DEFAULT_CSV):
    """Read the extracted dataset. One row per (dep, prec, x, y)."""
    df = pd.read_csv(path)
    assert list(df.columns) == FEATURES + TARGETS, df.columns
    assert len(df) == 315, len(df)
    return df


def bad_wer_mask(df):
    """Boolean Series: True for the 5 flagged WER rows."""
    keys = set(KNOWN_BAD_WER)
    return df[FEATURES].apply(lambda r: (r.dep_temp_C, r.prec_temp_C,
                                         r.x_cm, r.y_cm) in keys, axis=1)


def split(df, seed):
    """Random 215/100 train/test split (paper: 'randomly partitioned').

    INFERRED: the paper gives neither seed nor split, so we draw a fresh one per
    run and report the spread across seeds (assignment step 2 asks for this).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    return df.iloc[idx[N_TEST:]].copy(), df.iloc[idx[:N_TEST]].copy()


class MinMax:
    """Affine scaler fit on training data only (no test leakage).

    Default: min-max to [0, 1]. INFERRED - the paper says only "normalized to a
    comparable scale"; its NN-output axes run ~0..1 ["arb."], which is what
    min-max produces. standard=True switches to z-score for the step-2 comparison.
    """

    def __init__(self, standard=False):
        self.standard = standard

    def fit(self, A):
        A = np.asarray(A, float)
        if self.standard:
            self.c, self.s = A.mean(0), A.std(0)
        else:
            self.c, self.s = A.min(0), A.max(0) - A.min(0)
        self.s = np.where(self.s == 0, 1.0, self.s)   # guard constant columns
        return self

    def transform(self, A):
        return (np.asarray(A, float) - self.c) / self.s

    def inverse(self, A):
        return np.asarray(A, float) * self.s + self.c


class DNN(nn.Module):
    """4 -> 128 -> 64 -> 32 -> 16 -> 8 -> 3 dense net, ELU on hidden layers.

    Widths, ELU, and per-layer bias are stated in the paper (Fig. 2b).
    INFERRED: linear output head. The paper's "y ... then activated by ELU" reads
    as the generic per-layer rule; a linear head is standard for regression.
    elu_output=True tests the alternative reading.
    """

    def __init__(self, elu_output=False):
        super().__init__()
        widths = [4, 128, 64, 32, 16, 8, 3]
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            is_last = i == len(widths) - 2
            if not is_last or elu_output:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(Xn, yn, seed=0, epochs=1000, lr=1e-3, batch_size=None,
                elu_output=False):
    """Adam training on normalized data. Returns the trained model.

    INFERRED: optimizer=Adam, lr=1e-3, loss=MSE (the paper states only the
    1000-epoch count). batch_size=None means full-batch (1000 updates total);
    batch_size=32 is the Keras default and gives ~7x more updates for the same
    epoch count. Step 2 reports both readings.
    """
    torch.manual_seed(seed)
    model = DNN(elu_output=elu_output)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.tensor(Xn, dtype=torch.float32)
    y = torch.tensor(yn, dtype=torch.float32)
    n = len(X)
    bs = n if batch_size is None else batch_size
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            b = order[s:s + bs]
            opt.zero_grad()
            loss_fn(model(X[b]), y[b]).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, Xn):
    return model(torch.tensor(Xn, dtype=torch.float32)).numpy()


def tolerance_accuracy(y_true, y_pred, tol_vec):
    """Paper's metric: per target, percent of rows with |err| <= tol.

    y_true, y_pred in PHYSICAL units, shape (n, 3). Returns {target: percent}.
    """
    err = np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))
    return {t: 100.0 * np.mean(err[:, i] <= tol_vec[i])
            for i, t in enumerate(TARGETS)}


def run_once(df, seed, standard=False, elu_output=False, epochs=1000,
             batch_size=None):
    """One full train/eval cycle. Returns (test_acc dict, info dict)."""
    tr, te = split(df, seed)
    xs = MinMax(standard).fit(tr[FEATURES].values)
    ys = MinMax(standard).fit(tr[TARGETS].values)

    model = train_model(xs.transform(tr[FEATURES].values),
                        ys.transform(tr[TARGETS].values),
                        seed=seed, epochs=epochs, batch_size=batch_size,
                        elu_output=elu_output)

    pred_phys = ys.inverse(predict(model, xs.transform(te[FEATURES].values)))
    true_phys = te[TARGETS].values
    tol = [TOLERANCES[t] for t in TARGETS]
    acc = tolerance_accuracy(true_phys, pred_phys, tol)
    info = {"pred": pred_phys, "true": true_phys, "test_df": te, "train_df": tr,
            "mae": np.abs(pred_phys - true_phys).mean(0)}
    return acc, info
