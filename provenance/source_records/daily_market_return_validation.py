"""Full-sample daily S&P 500 intermeeting return validation.

This script complements the local 2018-2024 futures validation with public FRED
daily data. It links minutes-only positive/negative stock-market mentions to
intermeeting S&P 500 excess returns from the day after the prior FOMC meeting
through two calendar days before the current meeting.
"""

from __future__ import annotations

import math
import json
import urllib.error
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_table6_official_ffr_panel import (
    dataframe_to_markdown,
    pvalue_normal,
    regression_covariance,
    standardize,
)
from project_paths import DATA_DIR, DOC_DIR, FIGURE_DIR, RAW_DIR, TABLE_DIR, ensure_output_dirs


RAW_DAILY_DIR = RAW_DIR / "fred_daily_market_validation"
MINUTES_PANEL = DATA_DIR / "table6_official_ffr_panel_minutes_only.csv"
SERIES = {
    "DTB3": "3-month Treasury bill secondary-market discount rate",
}


def fred_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def download_series(series_id: str):
    RAW_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DAILY_DIR / f"{series_id}.csv"
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        urllib.request.urlretrieve(fred_url(series_id), path)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Could not download FRED series {series_id}: {exc}") from exc
    return path


def load_series(series_id: str) -> pd.DataFrame:
    path = download_series(series_id)
    df = pd.read_csv(path)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    value_col = series_id if series_id in df.columns else df.columns[-1]
    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: series_id.lower()})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[series_id.lower()] = pd.to_numeric(out[series_id.lower()].replace(".", np.nan), errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_sp500_daily() -> pd.DataFrame:
    """Load daily S&P 500 levels.

    FRED's `SP500` CSV currently starts in 2016. For the 2000-2024 validation,
    use yfinance/Yahoo when available and keep the downloaded CSV in data_raw.
    FRED remains a fallback for environments where yfinance cannot download.
    """
    RAW_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    yahoo_path = RAW_DAILY_DIR / "yahoo_gspc_daily.csv"
    if yahoo_path.exists() and yahoo_path.stat().st_size > 0:
        cached = pd.read_csv(yahoo_path)
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
        cached["sp500"] = pd.to_numeric(cached["sp500"], errors="coerce")
        return cached.dropna(subset=["date", "sp500"]).sort_values("date").reset_index(drop=True)

    try:
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
            "?period1=946684800&period2=1735689600&interval=1d&events=history"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        payload = json.loads(urllib.request.urlopen(request, timeout=30).read().decode("utf-8"))
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
                "sp500": closes,
            }
        )
        out["sp500"] = pd.to_numeric(out["sp500"], errors="coerce")
        out = out.dropna(subset=["date", "sp500"]).sort_values("date").reset_index(drop=True)
        if out.empty or out["date"].min() > pd.Timestamp("2001-01-01"):
            raise RuntimeError("Yahoo chart API history did not cover the 2000 sample start")
        out.to_csv(yahoo_path, index=False)
        return out
    except Exception:
        pass

    try:
        import yfinance as yf

        raw = yf.download(
            "^GSPC",
            start="2000-01-01",
            end="2025-01-01",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned an empty ^GSPC dataframe")
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].iloc[:, 0] if isinstance(raw["Close"], pd.DataFrame) else raw["Close"]
        else:
            close = raw["Close"]
        out = close.reset_index()
        out.columns = ["date", "sp500"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["sp500"] = pd.to_numeric(out["sp500"], errors="coerce")
        out = out.dropna(subset=["date", "sp500"]).sort_values("date").reset_index(drop=True)
        if out.empty or out["date"].min() > pd.Timestamp("2001-01-01"):
            raise RuntimeError("yfinance ^GSPC history did not cover the 2000 sample start")
        out.to_csv(yahoo_path, index=False)
        return out
    except Exception:
        fred = load_series("SP500").rename(columns={"sp500": "sp500"})
        fred["date"] = pd.to_datetime(fred["date"], errors="coerce")
        fred["sp500"] = pd.to_numeric(fred["sp500"], errors="coerce")
        return fred.dropna(subset=["date", "sp500"]).sort_values("date").reset_index(drop=True)


def build_daily_data() -> pd.DataFrame:
    sp = load_sp500_daily()
    tbill = load_series("DTB3")
    daily = pd.merge_ordered(sp, tbill, on="date", how="outer").sort_values("date")
    daily[["sp500", "dtb3"]] = daily[["sp500", "dtb3"]].ffill()
    daily = daily.dropna(subset=["sp500"]).copy()
    daily.to_csv(RAW_DAILY_DIR / "fred_sp500_dtb3_daily.csv", index=False)
    return daily


def close_on_or_after(daily: pd.DataFrame, date_value: pd.Timestamp):
    eligible = daily[(daily["date"] >= date_value) & daily["sp500"].notna()]
    if eligible.empty:
        return None, None
    row = eligible.iloc[0]
    return pd.Timestamp(row["date"]), float(row["sp500"])


def close_on_or_before(daily: pd.DataFrame, date_value: pd.Timestamp):
    eligible = daily[(daily["date"] <= date_value) & daily["sp500"].notna()]
    if eligible.empty:
        return None, None
    row = eligible.iloc[-1]
    return pd.Timestamp(row["date"]), float(row["sp500"])


def build_panel() -> pd.DataFrame:
    daily = build_daily_data()
    minutes = pd.read_csv(MINUTES_PANEL)
    minutes["date"] = pd.to_datetime(minutes["date"])
    minutes = minutes.sort_values("date").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for idx, row in minutes.iterrows():
        if idx == 0:
            continue
        prev_date = pd.Timestamp(minutes.loc[idx - 1, "date"])
        current_date = pd.Timestamp(row["date"])
        start_target = prev_date + pd.Timedelta(days=1)
        end_target = current_date - pd.Timedelta(days=2)
        if end_target <= start_target:
            continue
        start_date, start_close = close_on_or_after(daily, start_target)
        end_date, end_close = close_on_or_before(daily, end_target)
        if start_close is None or end_close is None or start_close <= 0 or end_close <= 0:
            continue
        if end_date <= start_date:
            continue
        window = daily[(daily["date"] >= start_date) & (daily["date"] <= end_date)].copy()
        calendar_days = max((end_date - start_date).days, 1)
        mean_tbill_yield = pd.to_numeric(window["dtb3"], errors="coerce").mean()
        rf_return_pct = (mean_tbill_yield / 100.0) * (calendar_days / 365.0) * 100.0
        sp500_log_return_pct = 100.0 * math.log(end_close / start_close)
        excess_return_pct = sp500_log_return_pct - rf_return_pct
        rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "prev_fomc_date": prev_date.strftime("%Y-%m-%d"),
                "return_start_date": start_date.strftime("%Y-%m-%d"),
                "return_end_date": end_date.strftime("%Y-%m-%d"),
                "calendar_days": calendar_days,
                "sp500_start": start_close,
                "sp500_end": end_close,
                "sp500_log_return_pct": sp500_log_return_pct,
                "mean_dtb3_annual_pct": mean_tbill_yield,
                "tbill_return_approx_pct": rf_return_pct,
                "sp500_excess_log_return_pct": excess_return_pct,
                "sp500_excess_rx_neg_pct": min(0.0, excess_return_pct),
                "sp500_excess_rx_pos_pct": max(0.0, excess_return_pct),
                "negative_stock_mentions": row["negative_stock_mentions"],
                "positive_stock_mentions": row["positive_stock_mentions"],
                "stock_mentions_total": row["stock_mentions_total"],
                "document_sentence_count": row["document_sentence_count"],
                "ffr_change_bp": row["ffr_change_bp"],
            }
        )
    panel = pd.DataFrame(rows)
    for col in [
        "sp500_excess_rx_neg_pct",
        "sp500_excess_rx_pos_pct",
        "document_sentence_count",
        "negative_stock_mentions",
        "positive_stock_mentions",
    ]:
        panel[f"z_{col}"] = standardize(panel[col])
    return panel


def run_regressions(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    specs = {
        "negative_mentions_on_daily_excess_returns": "negative_stock_mentions",
        "positive_mentions_on_daily_excess_returns": "positive_stock_mentions",
        "z_negative_mentions_on_daily_excess_returns": "z_negative_stock_mentions",
        "z_positive_mentions_on_daily_excess_returns": "z_positive_stock_mentions",
    }
    x_cols = [
        "z_sp500_excess_rx_neg_pct",
        "z_sp500_excess_rx_pos_pct",
        "z_document_sentence_count",
    ]
    for spec, y_col in specs.items():
        data = panel[[y_col, *x_cols]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) <= len(x_cols) + 3:
            rows.append(pd.DataFrame([{"spec": spec, "term": "SKIPPED", "n_obs": len(data)}]))
            continue
        y = data[y_col].astype(float).to_numpy()
        x = data[x_cols].astype(float).to_numpy()
        x = np.column_stack([np.ones(len(x)), x])
        terms = ["const", *x_cols]
        beta = np.linalg.pinv(x.T @ x) @ x.T @ y
        resid = y - x @ beta
        cov = regression_covariance(x, resid, cov_type="HAC", hac_lags=4)
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se != 0)
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
        rows.append(
            pd.DataFrame(
                [
                    {
                        "spec": spec,
                        "term": term,
                        "coefficient": coef,
                        "robust_se": robust_se,
                        "t_stat": t_stat,
                        "p_value": pvalue_normal(t_stat) if np.isfinite(t_stat) else np.nan,
                        "n_obs": len(data),
                        "r_squared": r2,
                        "estimator": "numpy_OLS_HAC_lags4_normal_pvalue",
                    }
                    for term, coef, robust_se, t_stat in zip(terms, beta, se, t_stats)
                ]
            )
        )
    return pd.concat(rows, ignore_index=True)


def write_figure(panel: pd.DataFrame, results: pd.DataFrame) -> None:
    if panel.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, y_col, title, spec in [
        (
            axes[0],
            "negative_stock_mentions",
            "Negative mentions",
            "negative_mentions_on_daily_excess_returns",
        ),
        (
            axes[1],
            "positive_stock_mentions",
            "Positive mentions",
            "positive_mentions_on_daily_excess_returns",
        ),
    ]:
        x = panel["sp500_excess_log_return_pct"].astype(float)
        y = panel[y_col].astype(float)
        ax.scatter(x, y, alpha=0.7)
        ok = x.notna() & y.notna()
        if ok.sum() > 2 and x[ok].std() > 0:
            slope, intercept = np.polyfit(x[ok], y[ok], 1)
            xs = np.linspace(x[ok].min(), x[ok].max(), 100)
            ax.plot(xs, intercept + slope * xs, color="black", linewidth=1.0)
        key = results[
            (results["spec"] == spec)
            & (results["term"].isin(["z_sp500_excess_rx_neg_pct", "z_sp500_excess_rx_pos_pct"]))
        ]
        note = "; ".join(
            f"{row['term'].replace('z_sp500_excess_', '').replace('_pct', '')}: p={row['p_value']:.3f}"
            for _, row in key.iterrows()
        )
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_title(f"{title} (N={ok.sum()})")
        ax.set_xlabel("Intermeeting S&P 500 excess log return, %")
        ax.set_ylabel("Minutes mentions")
        ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=7)
    fig.suptitle("Daily S&P 500 intermeeting return validation, 2000-2024")
    fig.savefig(FIGURE_DIR / "daily_market_return_validation_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    panel = build_panel()
    results = run_regressions(panel)
    panel.to_csv(DATA_DIR / "daily_market_return_validation_panel.csv", index=False)
    results.to_csv(TABLE_DIR / "daily_market_return_validation_regressions.csv", index=False)
    results.to_html(TABLE_DIR / "daily_market_return_validation_regressions.html", index=False)
    (TABLE_DIR / "daily_market_return_validation_regressions.md").write_text(
        dataframe_to_markdown(results), encoding="utf-8"
    )
    write_figure(panel, results)
    key = results[results["term"].isin(["z_sp500_excess_rx_neg_pct", "z_sp500_excess_rx_pos_pct"])][
        ["spec", "term", "coefficient", "robust_se", "p_value", "n_obs"]
    ]
    lines = [
        "# Daily Market-return Validation",
        "",
        "This check links minutes-only stock-market mentions to public daily S&P 500 intermeeting excess returns from FRED.",
        "The S&P 500 level is downloaded from Yahoo Finance through `yfinance` when available, with FRED `SP500` as a fallback; the T-bill rate comes from FRED `DTB3`.",
        "The return window runs from the first trading close on or after one calendar day after meeting m-1 through the last trading close on or before two calendar days before meeting m.",
        "The risk-free component is approximated from FRED `DTB3` over the same window.",
        "",
        f"Validation rows: {len(panel)}.",
        "",
        "## Return Coefficients",
        "",
        dataframe_to_markdown(key),
        "",
        "## Files",
        "",
        "- `data_raw/fred_daily_market_validation/yahoo_gspc_daily.csv`",
        "- `data_raw/fred_daily_market_validation/fred_sp500_dtb3_daily.csv`",
        "- `data_intermediate/daily_market_return_validation_panel.csv`",
        "- `tables/daily_market_return_validation_regressions.csv`",
        "- `figures/daily_market_return_validation_scatter.png`",
    ]
    (DOC_DIR / "DAILY_MARKET_RETURN_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {DATA_DIR / 'daily_market_return_validation_panel.csv'} rows={len(panel)}")
    print(f"WROTE {TABLE_DIR / 'daily_market_return_validation_regressions.csv'}")


if __name__ == "__main__":
    main()
