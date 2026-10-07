"""Load the three price files, build weekly series and regression features."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ASSETS = ["gold", "btc", "eth"]


def load_daily(data_dir: Path) -> pd.DataFrame:
    """Read gold.csv, btc.csv, eth.csv (Yahoo format, only the Close column is used).
    Gold does not trade at weekends, so its last close is carried forward on a calendar-day grid."""
    cols = {}
    for name in ASSETS:
        raw = pd.read_csv(Path(data_dir) / f"{name}.csv", index_col=0, parse_dates=True)
        raw.columns = [c.strip().lower() for c in raw.columns]
        s = raw["close"].astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        cols[name] = s[~s.index.duplicated(keep="last")]
    df = pd.DataFrame(cols).sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D")).ffill().dropna()
    df.index.name = "date"
    return df


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly close = Sunday close (weeks run Monday..Sunday). An incomplete last week is dropped."""
    wk = daily.resample("W-SUN").last()
    if daily.index.max().dayofweek != 6:
        wk = wk.iloc[:-1]
    return wk


def build_features(weekly: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """For each asset: X (row t uses only information up to the end of week t-1) and y (return of week t, in %)."""
    r = weekly.pct_change() * 100.0
    feats = {
        "gold": pd.DataFrame({"own_l1": r["gold"], "own_l2": r["gold"].shift(1), "btc_l1": r["btc"]}),
        "btc":  pd.DataFrame({"own_l1": r["btc"],  "own_l2": r["btc"].shift(1),  "eth_l1": r["eth"]}),
        "eth":  pd.DataFrame({"own_l1": r["eth"],  "own_l2": r["eth"].shift(1),  "btc_l1": r["btc"]}),
    }
    # shift(1): the value stored at week t is what was known at the end of week t-1
    return {a: (feats[a].shift(1), r[a]) for a in ASSETS}


def weekly_sigma(weekly: pd.DataFrame, t: int, window: int) -> dict[str, float]:
    """Std of weekly returns (fraction) using weeks up to t-1 only."""
    r = weekly[ASSETS].pct_change().iloc[max(1, t - window): t]
    return {a: float(max(r[a].std(ddof=1), 0.005)) for a in ASSETS}
