"""Walk-forward weekly return forecasts. The least-squares problem is solved by MiniZinc."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import mzn_utils
from config import Config
from data_prep import ASSETS, build_features


class Forecaster:
    def __init__(self, weekly: pd.DataFrame, cfg: Config):
        self.cfg = cfg
        self.feat = build_features(weekly)
        self.model = cfg.models_dir / "regression_ols.mzn"

    def predict(self, t: int, dzn_dir: Path | None = None) -> dict[str, float]:
        """Forecast the return (fraction) of week t using only weeks before t:
        train on weeks t-104 .. t-1 (their returns are known), predict week t."""
        out = {}
        for a in ASSETS:
            X, y = self.feat[a]
            lo = max(0, t - self.cfg.train_weeks)
            Xtr, ytr = X.iloc[lo:t], y.iloc[lo:t]
            ok = Xtr.notna().all(axis=1) & ytr.notna()
            Xtr, ytr = Xtr[ok].to_numpy(float), ytr[ok].to_numpy(float)
            x_now = X.iloc[t].to_numpy(float)
            if len(ytr) < 30 or np.isnan(x_now).any():
                out[a] = 0.0
                continue
            Xd = np.hstack([np.ones((len(Xtr), 1)), Xtr])          # intercept column
            dzn = dzn_dir / f"regression_{a}_week{t:03d}.dzn" if dzn_dir else None
            res = mzn_utils.solve(self.model, dict(n=Xd.shape[0], k=Xd.shape[1], X=Xd, y=ytr),
                                  self.cfg.solver, 60, dzn)
            if res.solution is None:
                out[a] = 0.0
                continue
            w = np.array(res["w"], float)
            out[a] = float(w[0] + x_now @ w[1:]) / 100.0
        return out
