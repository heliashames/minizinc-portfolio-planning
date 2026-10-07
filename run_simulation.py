"""Run:  python run_simulation.py
Week by week from 2024-04-01 with $50,000:  forecast (MiniZinc) -> optimise (MiniZinc) -> trade -> replay the week."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import data_prep as dp
import plots
from config import Config


def summary_row(w: pd.Series) -> dict:
    return dict(final_wealth=round(w.iloc[-1], 2), roi_pct=round((w.iloc[-1] / w.iloc[0] - 1) * 100, 2),
                max_drawdown_pct=round((w / w.cummax() - 1).min() * 100, 2))


def main() -> None:
    cfg = Config()
    cfg.out_dir.mkdir(exist_ok=True)
    dzn_dir = cfg.out_dir / "dzn"
    dzn_dir.mkdir(exist_ok=True)

    daily = dp.load_daily(cfg.data_dir)
    weekly = dp.to_weekly(daily)
    first = int(np.argmax(weekly.index >= pd.Timestamp(cfg.start_date) + pd.Timedelta(days=6)))
    last = len(weekly) - 1
    print(f"data {weekly.index[0].date()} .. {weekly.index[-1].date()} | simulating "
          f"{weekly.index[first].date()} .. {weekly.index[last].date()} ({last - first + 1} weeks)")

    from simulator import Simulator
    sim = Simulator(weekly, daily, cfg)
    t0 = time.perf_counter()
    for t in range(first, last + 1):
        sim.run_week(t, dzn_dir if t == first else None)       # keep the .dzn files of the first week as examples
        if (t - first) % 10 == 0 or t == last:
            r = sim.log[-1]
            print(f"  {r['date'].date()}  wealth ${r['wealth']:>10,.2f}  [{r['solver_status']}]")
    print(f"runtime {time.perf_counter() - t0:.1f}s")

    log = pd.DataFrame(sim.log)
    log.to_csv(cfg.out_dir / "weekly_log.csv", index=False)
    pd.DataFrame(sim.trades).to_csv(cfg.out_dir / "trades.csv", index=False)

    wealth = pd.concat([pd.Series([cfg.initial_capital], index=[weekly.index[first - 1]]), log.set_index("date")["wealth"]])

    # benchmarks over the same period
    idx = weekly.index[first - 1: last + 1]
    bench = pd.DataFrame({f"buy&hold {a.upper()}": cfg.initial_capital * weekly[a].loc[idx] / weekly[a].loc[idx[0]]
                          for a in dp.ASSETS})
    bench["bonds only"] = cfg.initial_capital * (1 + cfg.bond_rate_month / cfg.bond_lock_weeks) ** np.arange(len(idx))

    table = pd.DataFrame({"Strategy (MiniZinc)": summary_row(wealth), **{c: summary_row(bench[c]) for c in bench}}).T
    table.to_csv(cfg.out_dir / "performance_summary.csv")
    print("\n", table.to_string())

    # forecast quality
    fq = pd.DataFrame([dict(asset=a,
                            MAE_pct=(log[f"fc_{a}"] - log[f"act_{a}"]).abs().mean() * 100,
                            MAE_zero_forecast_pct=log[f"act_{a}"].abs().mean() * 100,
                            direction_accuracy_pct=(np.sign(log[f"fc_{a}"]) == np.sign(log[f"act_{a}"])).mean() * 100)
                       for a in dp.ASSETS]).round(2)
    fq.to_csv(cfg.out_dir / "forecast_quality.csv", index=False)
    print("\n", fq.to_string(index=False))
    print(f"\nnon-optimal solver weeks: {(log['solver_status'] != 'OPTIMAL_SOLUTION').sum()}")

    plots.make_all(log, bench, wealth, cfg.out_dir)
    print(f"results written to {cfg.out_dir}")


if __name__ == "__main__":
    main()
