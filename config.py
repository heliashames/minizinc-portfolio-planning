"""All tunable settings in one place."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: Path = ROOT / "data"
    out_dir: Path = ROOT / "results"
    models_dir: Path = ROOT / "models"
    solver: str = "highs"

    # simulation (given in the project statement)
    start_date: str = "2024-04-01"
    initial_capital: float = 50_000.0

    # forecasting
    train_weeks: int = 104          # rolling training window for the regression
    vol_weeks: int = 26             # window for the weekly volatility estimate

    # gold: one lot = 1 % of the gold price (whole-number percentages only)
    gold_lot_pct: int = 1

    # bonds: 0.55 % per month, locked for one month = 4 weeks
    bond_rate_month: float = 0.0055
    bond_lock_weeks: int = 4
    max_bond_frac: float = 0.5      # at most 50 % of wealth locked in bonds

    # margin trading
    max_lev: int = 3                # leverage 1..3
    margin_max_weeks: int = 4       # must be closed within one month
    tp_k: float = 1.5               # take profit  = 1.5 * weekly volatility
    sl_k: float = 1.0               # stop loss    = 1.0 * weekly volatility
    sl_cap: float = 0.25            # stop loss never farther than 25 %

    # risk budget used by the optimisation model
    z: float = 1.65                 # stress multiplier on weekly volatility
    rho: float = 0.08               # max stressed loss per week = 8 % of wealth
