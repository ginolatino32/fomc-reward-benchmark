"""Download official FRED federal funds target/range series.

Series:
- DFEDTAR: Federal funds target rate through 2008-12-15.
- DFEDTARU: Federal funds target range upper limit from 2008-12-16.
- DFEDTARL: Federal funds target range lower limit from 2008-12-16.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import OUT_DIR

RAW_DIR = OUT_DIR / "data_raw" / "fred_ffr_target_range"
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTAR,DFEDTARU,DFEDTARL"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(CSV_URL)
    raw = raw.rename(columns={"observation_date": "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    for col in ["DFEDTAR", "DFEDTARU", "DFEDTARL"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw["ffr_target_lower"] = np.where(raw["DFEDTAR"].notna(), raw["DFEDTAR"], raw["DFEDTARL"])
    raw["ffr_target_upper"] = np.where(raw["DFEDTAR"].notna(), raw["DFEDTAR"], raw["DFEDTARU"])
    raw["ffr_target_midpoint"] = (raw["ffr_target_lower"] + raw["ffr_target_upper"]) / 2.0
    raw["ffr_target_source"] = np.where(
        raw["DFEDTAR"].notna(),
        "FRED_DFEDTAR_single_target",
        "FRED_DFEDTARL_DFEDTARU_target_range",
    )
    raw.loc[raw["ffr_target_midpoint"].isna(), "ffr_target_source"] = "missing"
    raw["ffr_daily_change_pct"] = raw["ffr_target_midpoint"].diff()
    raw["ffr_daily_change_bp"] = raw["ffr_daily_change_pct"] * 100.0

    out = RAW_DIR / "fred_federal_funds_target_range.csv"
    raw.to_csv(out, index=False)
    metadata = {
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "source_url": CSV_URL,
        "series": {
            "DFEDTAR": "Federal funds target rate; single target series through 2008-12-15.",
            "DFEDTARU": "Federal funds target range upper limit from 2008-12-16.",
            "DFEDTARL": "Federal funds target range lower limit from 2008-12-16.",
        },
        "output": str(out),
        "rows": int(len(raw)),
        "date_min": str(raw["date"].min()),
        "date_max": str(raw["date"].max()),
    }
    (RAW_DIR / "fred_federal_funds_target_range_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"WROTE {out} rows={len(raw)} date_range={raw['date'].min()}..{raw['date'].max()}")


if __name__ == "__main__":
    main()
