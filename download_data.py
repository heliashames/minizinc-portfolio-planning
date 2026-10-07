"""Optional: download fresh daily prices (needs internet and `pip install yfinance`). Writes only the Close column."""
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = {"gold": "GC=F", "btc": "BTC-USD", "eth": "ETH-USD"}
out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
for name, tic in TICKERS.items():
    df = yf.download(tic, start="2020-01-01", auto_adjust=False, progress=False)
    close = df["Close"]
    close = (close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close).dropna().rename("Close")
    close.index.name = "Date"
    close.to_csv(out / f"{name}.csv")
    print(name, len(close), close.index[0].date(), "->", close.index[-1].date())
