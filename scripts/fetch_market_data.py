"""Download the S&P 500 price caches that this repository does not redistribute.

    python scripts/fetch_market_data.py

Writes
  source/market/yahoo_gspc_daily.csv  (date, sp500)       Yahoo Finance ^GSPC daily close, 2000-01-03 to 2024-12-31
  source/market/SP500.csv             (observation_date, SP500)  FRED SP500 series

Afterwards `python reproduce.py` also rebuilds data/alternative_return_targets.csv
from these caches. Providers revise history and round differently, so a fresh
download need not match the archived file byte for byte. The script prints the
archived SHA-256 for comparison. Check the rebuilt targets with `git diff data/`.
Use of the downloaded data is subject to Yahoo's and S&P Dow Jones Indices' terms.
"""
from pathlib import Path
import datetime as dt
import hashlib
import io
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "source" / "market"
ARCHIVED = {
    "yahoo_gspc_daily.csv": "d77806f49ee68e6f6cc91743fc0ddf95b44f0125ef0b654e7092daee6766a2d1",
    "SP500.csv": "46cd98adebb7d23b815cab9b29d7f5735bd1c21027608d12f080813a5ce72a68",
}
YAHOO = ("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
         "?period1=946684800&period2=1735689600&interval=1d&events=history")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
HEADERS = {"User-Agent": "Mozilla/5.0 (research replication; fomc-reward-benchmark)"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        return r.read()


def yahoo() -> pd.DataFrame:
    import json
    res = json.loads(get(YAHOO))["chart"]["result"][0]
    ts = [dt.datetime.fromtimestamp(t, dt.timezone.utc).date() for t in res["timestamp"]]
    close = res["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"date": pd.to_datetime(ts), "sp500": close}).dropna()
    df = df[(df.date >= "2000-01-03") & (df.date <= "2024-12-31")].drop_duplicates("date")
    df["date"] = df.date.dt.strftime("%Y-%m-%d")
    return df


def main() -> None:
    MARKET.mkdir(parents=True, exist_ok=True)
    out = MARKET / "yahoo_gspc_daily.csv"
    yahoo().to_csv(out, index=False)
    try:
        fred = pd.read_csv(io.BytesIO(get(FRED)))
        (MARKET / "SP500.csv").write_bytes(fred.to_csv(index=False).encode())
    except Exception as exc:  # FRED is only used for a Yahoo/FRED overlap check
        print(f"FRED download failed ({exc}). Save {FRED} as source/market/SP500.csv by hand.")
    for name, archived in ARCHIVED.items():
        if not (MARKET / name).exists():
            continue
        digest = hashlib.sha256((MARKET / name).read_bytes()).hexdigest()
        status = "matches archive" if digest == archived else "differs from archive (expected after provider revisions)"
        print(f"{name}: {digest[:16]}... {status}")
    print("Now run: python reproduce.py")


if __name__ == "__main__":
    main()
