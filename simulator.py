"""Weekly simulator: prepare data for portfolio.mzn, solve it, execute the decisions,
then replay the week day by day to see whether any take-profit / stop-loss is hit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import mzn_utils
from config import Config
from data_prep import ASSETS, weekly_sigma
from forecasting import Forecaster

COINS = ["btc", "eth"]


@dataclass
class MarginPos:
    coin: int        # 0 = BTC, 1 = ETH
    sign: int        # +1 long, -1 short
    lev: int
    q: float         # units of the coin
    K: float         # collateral ($)
    entry: float     # entry price
    age: int = 0     # weeks held

    def equity(self, P: float) -> float:
        return self.K + self.sign * self.q * (P - self.entry)


def pos_return(price_ret: float, sign: int, tp: float, sl: float) -> float:
    """Expected weekly return per $ notional: the position return is capped by TP and floored by SL."""
    return float(np.clip(sign * price_ret, -sl, tp))


class Simulator:
    def __init__(self, weekly: pd.DataFrame, daily: pd.DataFrame, cfg: Config):
        self.weekly, self.daily, self.cfg = weekly, daily, cfg
        self.fc = Forecaster(weekly, cfg)
        self.model = cfg.models_dir / "portfolio.mzn"
        self.cash = cfg.initial_capital
        self.gold_lots = 0
        self.spot = [0.0, 0.0]                      # units of BTC, ETH
        self.margins: list[MarginPos] = []
        self.bonds: list[tuple[float, int]] = []    # (principal, week of purchase)
        self.log: list[dict] = []
        self.trades: list[dict] = []

    # ---------------------------------------------------------------- helpers
    def _prices(self, t: int):
        row = self.weekly.iloc[t]
        return float(row["gold"]), [float(row["btc"]), float(row["eth"])]

    def _bond_value(self, t: int, offset: int) -> float:
        """Bonds accrue linearly over the lock-up (offset = 0 at week start, 1 at week end)."""
        c = self.cfg
        return sum(p * (1 + c.bond_rate_month * min(1.0, (t - s + offset) / c.bond_lock_weeks)) for p, s in self.bonds)

    def _wealth(self, Pg, Pc, t, offset) -> float:
        lot = Pg * self.cfg.gold_lot_pct / 100
        return (self.cash + self.gold_lots * lot + sum(self.spot[c] * Pc[c] for c in range(2))
                + sum(m.equity(Pc[m.coin]) for m in self.margins) + self._bond_value(t, offset))

    # ---------------------------------------------------------------- one week
    def run_week(self, t: int, dzn_dir: Path | None) -> None:
        cfg = self.cfg
        date_end = self.weekly.index[t]
        P0g, P0c = self._prices(t - 1)                       # prices at the START of week t

        # bonds whose month is over pay principal + 0.55 %
        due = [(p, s) for p, s in self.bonds if t - s >= cfg.bond_lock_weeks]
        self.cash += sum(p * (1 + cfg.bond_rate_month) for p, _ in due)
        self.bonds = [b for b in self.bonds if b not in due]

        wealth0 = self._wealth(P0g, P0c, t, 0)
        fc = self.fc.predict(t, dzn_dir)                     # regression (MiniZinc)
        sig = weekly_sigma(self.weekly, t, cfg.vol_weeks)
        tp = {a: min(cfg.tp_k * sig[a], 0.5) for a in ASSETS}
        sl = {a: min(cfg.sl_k * sig[a], cfg.sl_cap) for a in ASSETS}

        # data for the optimisation model
        ms = self.margins
        data = dict(
            NC=2, NP=len(ms), max_lev=cfg.max_lev,
            cash=self.cash, wealth=wealth0, z=cfg.z, rho=cfg.rho, max_bond_frac=cfg.max_bond_frac,
            gold_price=P0g, gold_lot_pct=cfg.gold_lot_pct, gold_held=int(self.gold_lots),
            gold_ret=fc["gold"], gold_sigma=sig["gold"],
            spot_value=np.array([max(self.spot[c] * P0c[c], 0.0) for c in range(2)]),
            c_ret=np.array([fc[a] for a in COINS]), c_sigma=np.array([sig[a] for a in COINS]),
            c_pret=np.array([[pos_return(fc[a], s, tp[a], sl[a]) for s in (1, -1)] for a in COINS]),
            c_sl=np.array([sl[a] for a in COINS]),
            p_coin=np.array([m.coin + 1 for m in ms], dtype=int),
            p_forced=np.array([int(m.age >= cfg.margin_max_weeks) for m in ms], dtype=int),
            p_notional=np.array([m.q * P0c[m.coin] for m in ms]),
            p_equity=np.array([max(m.equity(P0c[m.coin]), 0.0) for m in ms]),
            p_pret=np.array([pos_return(fc[COINS[m.coin]], m.sign, tp[COINS[m.coin]], sl[COINS[m.coin]]) for m in ms]),
            bond_locked=sum(p for p, _ in self.bonds), bond_rate_week=cfg.bond_rate_month / cfg.bond_lock_weeks,
        )
        dzn = dzn_dir / f"portfolio_week{t:03d}.dzn" if dzn_dir else None
        res = mzn_utils.solve(self.model, data, cfg.solver, 60, dzn)
        status = str(res.status).split(".")[-1]

        notes: list[str] = []
        if res.solution is not None:
            self._apply(res, t, P0g, P0c, tp, sl, notes, date_end)
        else:
            notes.append(f"no solution ({status}) -> hold")

        # replay the week day by day for margin positions
        days = self.daily.loc[self.weekly.index[t - 1] + pd.Timedelta(days=1): date_end]
        self._replay_margins(days, P0c, tp, sl, notes)

        Pg1, Pc1 = self._prices(t)
        lot1 = Pg1 * cfg.gold_lot_pct / 100
        self.log.append(dict(
            week=t, date=date_end, wealth_start=wealth0, wealth=self._wealth(Pg1, Pc1, t, 1),
            cash=self.cash, gold_value=self.gold_lots * lot1, gold_lots=self.gold_lots,
            spot_value=sum(self.spot[c] * Pc1[c] for c in range(2)),
            margin_equity=sum(m.equity(Pc1[m.coin]) for m in self.margins), n_margin=len(self.margins),
            bonds=self._bond_value(t, 1),
            fc_gold=fc["gold"], fc_btc=fc["btc"], fc_eth=fc["eth"],
            act_gold=Pg1 / P0g - 1, act_btc=Pc1[0] / P0c[0] - 1, act_eth=Pc1[1] / P0c[1] - 1,
            solver_status=status, decisions="; ".join(notes),
        ))

    # ---------------------------------------------------------------- execute decisions
    def _apply(self, res, t, P0g, P0c, tp, sl, notes, date_end):
        cfg = self.cfg
        lot = P0g * cfg.gold_lot_pct / 100

        # close the margin positions that are not kept
        keep = [round(k) for k in res["keep"]] if self.margins else []
        still = []
        for i, m in enumerate(self.margins):
            if keep[i] == 0:
                self.cash += max(m.equity(P0c[m.coin]), 0.0)
                notes.append(f"close {'LONG' if m.sign == 1 else 'SHORT'} {COINS[m.coin].upper()} x{m.lev} (age {m.age})")
                self.trades.append(dict(week=t, date=date_end, action="close", coin=COINS[m.coin], lev=m.lev))
            else:
                still.append(m)
        self.margins = still

        # gold: whole lots only
        gb, gs = round(res["gold_buy"]), min(round(res["gold_sell"]), self.gold_lots)
        if gb or gs:
            self.cash += (gs - gb) * lot
            self.gold_lots += gb - gs
            notes.append(f"gold {'+' if gb >= gs else '-'}{abs(gb - gs)} lots")
            self.trades.append(dict(week=t, date=date_end, action="gold", lots=gb - gs))

        # spot crypto
        for c, a in enumerate(COINS):
            buy, sell = float(res["spot_buy"][c]), min(float(res["spot_sell"][c]), self.spot[c] * P0c[c])
            if buy > 1e-6 or sell > 1e-6:
                self.cash += sell - buy
                self.spot[c] += (buy - sell) / P0c[c]
                if self.spot[c] * P0c[c] < 1e-6:
                    self.spot[c] = 0.0                       # remove floating-point dust
                notes.append(f"spot {a.upper()} {'+' if buy >= sell else '-'}${abs(buy - sell):,.0f}")
                self.trades.append(dict(week=t, date=date_end, action="spot", coin=a, usd=buy - sell))

        # new margin positions
        for c, a in enumerate(COINS):
            for d, sign in enumerate((1, -1)):
                for l in range(cfg.max_lev):
                    amt = float(res["m"][c][d][l])
                    if round(res["y"][c][d][l]) == 1 and amt > 1e-6:
                        lev = l + 1
                        self.cash -= amt
                        self.margins.append(MarginPos(c, sign, lev, lev * amt / P0c[c], amt, P0c[c]))
                        notes.append(f"open {'LONG' if sign == 1 else 'SHORT'} {a.upper()} x{lev} margin ${amt:,.0f}")
                        self.trades.append(dict(week=t, date=date_end, action="open", coin=a, sign=sign,
                                                lev=lev, margin=amt, tp=tp[a], sl=sl[a]))

        # bonds
        bb = float(res["bond_buy"])
        if bb > 1e-6:
            self.cash -= bb
            self.bonds.append((bb, t))
            notes.append(f"bonds +${bb:,.0f}")
            self.trades.append(dict(week=t, date=date_end, action="bond", usd=bb))

        assert self.cash > -1e-3, f"negative cash in week {t}"
        self.cash = max(self.cash, 0.0)

    # ---------------------------------------------------------------- TP / SL during the week
    def _replay_margins(self, days, P0c, tp, sl, notes):
        alive = []
        for m in self.margins:
            a = COINS[m.coin]
            P0 = P0c[m.coin]
            closed = False
            for px in days[a].to_numpy(float):
                ret = m.sign * (px / P0 - 1)                 # position return since the start of the week
                if ret <= -sl[a]:                            # stop loss: market order at the day's close
                    exit_px, tag = px, "SL"
                elif ret >= tp[a]:                           # take profit: limit order at the TP level
                    exit_px, tag = P0 * (1 + m.sign * tp[a]), "TP"
                else:
                    continue
                self.cash += max(m.equity(exit_px), 0.0)
                notes.append(f"{tag} {a.upper()} {'LONG' if m.sign == 1 else 'SHORT'} x{m.lev}")
                closed = True
                break
            if not closed:
                m.age += 1
                alive.append(m)
        self.margins = alive
