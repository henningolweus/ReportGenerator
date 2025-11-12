import argparse
import os
from pathlib import Path
import sys
import json
import datetime as dt
import math

import yaml
import pandas as pd

try:  # optional dependency
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from . import calc, data, charts, ppt, commentary


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def build(config_path: str) -> str:
    cfg = load_config(config_path)

    out_dir = cfg["output"]["ppt_out_dir"]
    charts_dir = cfg["output"]["charts_dir"]
    ensure_dirs([out_dir, charts_dir])

    as_of = dt.date.fromisoformat(str(cfg["as_of"]))
    start_date_cfg = cfg.get("start_date")
    start_date = pd.Timestamp(start_date_cfg) if start_date_cfg else None
    target_end = pd.Timestamp(as_of)
    if start_date and start_date > target_end:
        raise ValueError("start_date in config cannot be after as_of date")

    # Minimal demo: generate a placeholder chart and write two text fields
    perf_line_path = os.path.join(charts_dir, "perf_line.png")
    charts.render_min_demo_line(perf_line_path)
    monthly_bars_path = os.path.join(charts_dir, "monthly_bars.png")
    charts.render_monthly_bars(monthly_bars_path)
    pie_sector_path = os.path.join(charts_dir, "pie_sector.png")
    charts.render_pie(pie_sector_path, ["Tech","Financials","Materials","Others"], [36, 21, 13, 30])
    pie_region_path = os.path.join(charts_dir, "pie_region.png")
    charts.render_pie(pie_region_path, ["N. America","Europe","Asia","Africa"], [75, 12, 10, 3])

    prs_path = cfg["output"]["ppt_template"]
    base_name = f"Kukula_Factsheet_{as_of.isoformat()}"
    version = 1
    while True:
        candidate = os.path.join(out_dir, f"{base_name}_v{version}.pptx")
        if not os.path.exists(candidate):
            out_path = candidate
            break
        version += 1

    def format_pct(value: float | None, decimals: int = 2) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{value * 100:.{decimals}f}%"

    def format_ratio(value: float | None, decimals: int = 2) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{value:.{decimals}f}"

    def build_monthly_table(port_monthly: pd.Series | None,
                            bench_monthly: pd.Series | None,
                            start_ts: pd.Timestamp,
                            end_ts: pd.Timestamp,
                            port_ytd: float | None,
                            bench_ytd: float | None) -> pd.DataFrame | None:
        if port_monthly is None or bench_monthly is None:
            return None
        if port_monthly.empty or bench_monthly.empty:
            return None
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        month_end_as_of = end_ts + pd.offsets.MonthEnd(0)
        start_year = start_ts.year
        years = list(range(start_year, end_ts.year + 1))
        rows = []
        for year in years:
            port_row = {"Year": str(year)}
            bench_row = {"Year": "Benchmark"}
            for idx, month_name in enumerate(month_names, start=1):
                month_end = pd.Timestamp(year, idx, 1) + pd.offsets.MonthEnd(0)
                future = month_end > month_end_as_of
                port_val = port_monthly.get(month_end, None)
                bench_val = bench_monthly.get(month_end, None)
                port_row[month_name] = "NA" if future else (format_pct(port_val) if port_val is not None and not pd.isna(port_val) else "")
                bench_row[month_name] = "NA" if future else (format_pct(bench_val) if bench_val is not None and not pd.isna(bench_val) else "")
            port_year_slice = port_monthly.loc[(port_monthly.index.year == year) & (port_monthly.index <= month_end_as_of)]
            bench_year_slice = bench_monthly.loc[(bench_monthly.index.year == year) & (bench_monthly.index <= month_end_as_of)]
            if year == end_ts.year:
                fy_port = port_ytd
                fy_bench = bench_ytd
            else:
                fy_port = (1 + port_year_slice).prod() - 1 if not port_year_slice.empty else None
                fy_bench = (1 + bench_year_slice).prod() - 1 if not bench_year_slice.empty else None
            port_row["FY/YTD"] = format_pct(fy_port) if fy_port is not None and not pd.isna(fy_port) else ""
            bench_row["FY/YTD"] = format_pct(fy_bench) if fy_bench is not None and not pd.isna(fy_bench) else ""
            rows.append(port_row)
            rows.append(bench_row)
        return pd.DataFrame(rows, columns=["Year"] + month_names + ["FY/YTD"])

    def prepare_pie_data(weights: dict | None,
                         ordered_labels: list[str]) -> tuple[list[str], list[float]]:
        if not weights:
            return ordered_labels, [0.0] * len(ordered_labels)
        ordered = list(ordered_labels)
        values = [weights.get(lbl, 0.0) for lbl in ordered]
        # capture any additional labels not in the ordered list
        extras = [lbl for lbl in weights if lbl not in ordered]
        ordered.extend(extras)
        values.extend([weights[lbl] for lbl in extras])
        total = sum(values)
        if total > 0:
            values = [v / total for v in values]
        return ordered, values

    # Replace a couple of text placeholders if present
    prs = ppt.open_presentation(prs_path)
    # Use the as_of date from config, not today's date
    issued_text = f"Fact Sheet|Issued {as_of.strftime('%d %B %Y')}"
    date_primary_ok = ppt.set_text_by_name(prs.slides, "DATE_ISSUED", issued_text)
    date_secondary_ok = ppt.set_text_by_name(prs.slides, "DATE_ISSUED_2", issued_text)
    date_ok = date_primary_ok or date_secondary_ok
    bench_ok = ppt.set_text_by_name(prs.slides, "BENCH_NAME", cfg["benchmarks"][0]["name"]) if cfg.get("benchmarks") else False

    # Drop demo chart into placeholder if present
    # Try native chart updates first; fallback to PNG
    # Conform dummy series to chart structure (same number of points & series)
    perf_struct = ppt.get_chart_structure(prs.slides, "CHART_PERF_LINE")
    # Prefer real daily data from the Performance sheet; otherwise fall back to synthetic paths
    inferred_start: pd.Timestamp | None = None
    perf_categories = None
    port_usd = None
    data_sources = cfg.get("data_sources", {})
    perf_file_cfg = data_sources.get("aggregated_amounts_file") or cfg.get("aggregated_amounts_file")
    perf_dir_cfg = data_sources.get("aggregated_amounts_dir")
    perf_pattern = data_sources.get("aggregated_amounts_pattern", "*.xlsx")

    perf_file: Path | None = None
    if perf_file_cfg:
        perf_file = Path(perf_file_cfg)
    elif perf_dir_cfg:
        dir_path = Path(perf_dir_cfg)
        dir_path.mkdir(parents=True, exist_ok=True)
        if dir_path.is_dir():
            candidates = sorted(
                dir_path.glob(perf_pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                perf_file = candidates[0]
    if perf_file is None:
        perf_file = Path("AggregatedAmounts_15336611_2021-06-01_2025-10-15 (1).xlsx")
    if not perf_file.exists():
        raise FileNotFoundError(f"Aggregated amounts file not found: {perf_file}")

    # Load daily returns once for metrics
    daily_returns_full = None
    daily_decimals = None
    ytd_return = None
    ret_since_incept = None
    avg_yearly_return = None
    ytd_start = pd.Timestamp(target_end.year, 1, 1)
    month_end_as_of = target_end + pd.offsets.MonthEnd(0)
    top_holdings_df = None
    holdings_raw = None
    try:
        daily_returns_full = data.load_daily_portfolio_returns_from_performance(str(perf_file))
        daily_returns_full = daily_returns_full.sort_index()
        # Filter to use only data within the config date range
        if start_date is not None:
            daily_returns_full = daily_returns_full.loc[(daily_returns_full.index >= start_date) & (daily_returns_full.index <= target_end)]
        else:
            daily_returns_full = daily_returns_full.loc[daily_returns_full.index <= target_end]
        if not daily_returns_full.empty:
            daily_decimals = daily_returns_full
            ret_since_incept = (1.0 + daily_decimals).prod() - 1.0
            first_ret_date = daily_decimals.index.min()
            years = (target_end - first_ret_date).days / 365.25
            if years > 0:
                avg_yearly_return = (1.0 + ret_since_incept) ** (1.0 / years) - 1.0
            ytd_start = pd.Timestamp(target_end.year, 1, 1)
            ytd_series = daily_decimals.loc[(daily_decimals.index >= ytd_start)]
            if not ytd_series.empty:
                ytd_return = (1.0 + ytd_series).prod() - 1.0
    except Exception:
        daily_returns_full = None

    def gen_path(mu=0.0004, sigma=0.01, n=None, seed: int | None = None):
        """Generate a synthetic cumulative return path with daily cadence."""
        import numpy as np

        n = n or (len(perf_categories) if perf_categories is not None else 0)
        if n <= 0:
            return []
        rng = np.random.default_rng(seed)
        rets = rng.normal(loc=mu, scale=sigma, size=n)
        cum = (1 + rets).cumprod()
        if cum.size:
            if cum[0] == 0:
                cum[0] = 1.0
            cum = cum / cum[0] - 1.0
            cum[0] = 0.0
        return cum.tolist()

    # Preferred source: accumulated TWR column
    try:
        cum = data.load_accumulated_twr_from_performance(str(perf_file))
        cum = cum.sort_index()
        if not cum.empty:
            # Use config start_date if specified, otherwise use first available data
            if start_date is not None:
                start = start_date
                inferred_start = start_date
            else:
                inferred_start = cum.index.min()
                start = inferred_start
            # Always use config as_of date as the end
            end = target_end
            # Filter data to only include the date range from config
            cum = cum.loc[(cum.index >= start) & (cum.index <= end)]
            if not cum.empty and end >= start:
                full_index = pd.date_range(start=start, end=end, freq="D")
                cum = cum.reindex(full_index).ffill()
                perf_categories = full_index
                port_usd = cum.tolist()
    except Exception:
        pass

    if perf_categories is None or port_usd is None:
        try:
            daily = data.load_daily_portfolio_returns_from_performance(str(perf_file))
            daily = daily.sort_index()
            if not daily.empty:
                # Use config start_date if specified, otherwise use first available data
                if start_date is not None:
                    start = start_date
                    if inferred_start is None:
                        inferred_start = start_date
                else:
                    if inferred_start is None:
                        inferred_start = daily.index.min()
                    start = inferred_start
                # Always use config as_of date as the end
                end = target_end
                # Filter data to only include the date range from config
                daily = daily.loc[(daily.index >= start) & (daily.index <= end)]
                if not daily.empty:
                    full_index = pd.date_range(start=start, end=end, freq="D")
                    daily = daily.reindex(full_index).fillna(0.0)
                    cum = (1 + daily).cumprod()
                    if not cum.empty:
                        if cum.iloc[0] == 0:
                            cum.iloc[0] = 1.0
                        cum = cum / cum.iloc[0] - 1.0
                        cum.iloc[0] = 0.0
                        perf_categories = full_index
                        port_usd = cum.tolist()
        except Exception:
            # Ignore failures and fall back to synthetic series below
            pass

    if perf_categories is None or port_usd is None:
        synthetic_start = start_date or (inferred_start if inferred_start is not None else target_end - pd.DateOffset(years=4))
        perf_categories = pd.date_range(start=synthetic_start, end=target_end, freq="D")
        port_usd = gen_path(mu=0.00045, sigma=0.0095, n=len(perf_categories), seed=42)

    total_points = len(perf_categories)

    fx_series = None
    try:
        fx_series = data.fx_from_yfinance("USD", "ZMW", perf_categories[0].date(), target_end.date())
        fx_series = fx_series.sort_index()
        fx_series = fx_series.loc[(fx_series.index >= perf_categories[0]) & (fx_series.index <= perf_categories[-1])]
        fx_series = fx_series.reindex(perf_categories).ffill().bfill()
        if fx_series.empty:
            fx_series = None
    except Exception:
        fx_series = None

    bench_usd = None
    bench_zmw = None
    bench_daily = None
    bench_monthly = None
    bench_ytd = None
    bench_cagr = None
    try:
        bench_prices = data.load_benchmark_prices("ACWI", perf_categories[0], perf_categories[-1])
        bench_prices = bench_prices.reindex(perf_categories).ffill()
        bench_daily = bench_prices.pct_change().fillna(0.0)
        bench_series = (1 + bench_daily).cumprod() - 1
        bench_series.iloc[0] = 0.0
        bench_usd = bench_series.tolist()
        bench_total_return = bench_series.iloc[-1]
        bench_years = (bench_daily.index[-1] - bench_daily.index[0]).days / 365.25
        if bench_years > 0:
            bench_cagr = (1.0 + bench_total_return) ** (1.0 / bench_years) - 1.0
        ytd_mask = bench_daily.index >= ytd_start
        if ytd_mask.any():
            bench_ytd = (1 + bench_daily.loc[ytd_mask]).prod() - 1.0
        bench_monthly = (1 + bench_daily).resample("ME").prod() - 1.0
        if fx_series is not None and not fx_series.empty:
            fx_rel = fx_series / fx_series.iloc[0]
            bench_zmw_series = (1 + bench_series) * fx_rel - 1.0
            bench_zmw = bench_zmw_series.tolist()
    except Exception:
        bench_usd = None
        bench_zmw = None
        bench_daily = None
        bench_monthly = None
        bench_ytd = None
        bench_cagr = None

    if bench_usd is None:
        bench_usd = gen_path(mu=0.00035, sigma=0.0085, n=total_points, seed=43)
    if bench_zmw is None:
        bench_zmw = gen_path(mu=0.0004, sigma=0.01, n=total_points, seed=45)

    port_daily_series = daily_decimals
    if port_daily_series is None and perf_categories is not None and port_usd is not None:
        port_series = pd.Series(port_usd, index=perf_categories)
        port_daily_series = (1 + port_series).pct_change().fillna(0.0)

    bench_daily_series = bench_daily
    if bench_daily_series is None and perf_categories is not None and bench_usd is not None:
        bench_series_tmp = pd.Series(bench_usd, index=perf_categories)
        bench_daily_series = (1 + bench_series_tmp).pct_change().fillna(0.0)
    if bench_ytd is None and bench_daily_series is not None and not bench_daily_series.empty:
        mask = bench_daily_series.index >= ytd_start
        if mask.any():
            bench_ytd = (1 + bench_daily_series.loc[mask]).prod() - 1.0
    if bench_cagr is None and bench_daily_series is not None and not bench_daily_series.empty:
        bench_total_alt = (1 + bench_daily_series).prod() - 1.0
        bench_years_alt = (bench_daily_series.index[-1] - bench_daily_series.index[0]).days / 365.25
        if bench_years_alt > 0:
            bench_cagr = (1 + bench_total_alt) ** (1.0 / bench_years_alt) - 1.0

    port_monthly = None
    if port_daily_series is not None and not port_daily_series.empty:
        port_monthly = (1 + port_daily_series).resample("ME").prod() - 1.0
    if bench_monthly is None and bench_daily_series is not None and not bench_daily_series.empty:
        bench_monthly = (1 + bench_daily_series).resample("ME").prod() - 1.0

    monthly_table_df = build_monthly_table(
        port_monthly,
        bench_monthly,
        perf_categories[0],
        perf_categories[-1],
        ytd_return,
        bench_ytd,
    ) if perf_categories is not None else None

    sector_display_order = [
        "Transportation",
        "Industrials",
        "Consumer Discretionary",
        "Technology",
        "Consumer Staples",
        "Materials",
        "Financial services",
        "Healthcare",
        "Utilities",
        "Cash",
    ]
    sector_alias_map = {
        "Industrials": "Industrials",
        "Transportation": "Transportation",
        "Airlines": "Transportation",
        "Marine Shipping": "Transportation",
        "Railroads": "Transportation",
        "Consumer Cyclical": "Consumer Discretionary",
        "Consumer Discretionary": "Consumer Discretionary",
        "Communication Services": "Consumer Discretionary",
        "Consumer Discretionary ": "Consumer Discretionary",
        "Technology": "Technology",
        "Information Technology": "Technology",
        "Tech": "Technology",
        "Consumer Defensive": "Consumer Staples",
        "Consumer Staples": "Consumer Staples",
        "Basic Materials": "Materials",
        "Materials": "Materials",
        "Financial Services": "Financial services",
        "Financial": "Financial services",
        "Financials": "Financial services",
        "Health Care": "Healthcare",
        "Healthcare": "Healthcare",
        "Utilities": "Utilities",
        "Cash": "Cash",
        "Energy": "Transportation",
    }
    region_display_order = ["North America", "Europe", "Asia", "Africa", "Cash"]
    region_alias_map = {
        "united states": "North America",
        "canada": "North America",
        "mexico": "North America",
        "denmark": "Europe",
        "ireland": "Europe",
        "united kingdom": "Europe",
        "france": "Europe",
        "germany": "Europe",
        "switzerland": "Europe",
        "netherlands": "Europe",
        "italy": "Europe",
        "norway": "Europe",
        "sweden": "Europe",
        "taiwan": "Asia",
        "japan": "Asia",
        "china": "Asia",
        "hong kong": "Asia",
        "south korea": "Asia",
        "singapore": "Asia",
        "india": "Asia",
        "vietnam": "Asia",
        "south africa": "Africa",
        "zambia": "Africa",
        "cash": "Cash",
    }

    sector_weights = None
    region_weights = None
    classified_records = []
    classification_cache_path = cfg.get("tables", {}).get("classification_cache")
    classification_cache = data.load_classification_cache(classification_cache_path)
    overrides = data.load_symbol_overrides(cfg.get("tables", {}).get("instrument_overrides_file"))
    region_map = data.load_country_region_map(cfg.get("tables", {}).get("region_map_file"))
    sector_lookup = data.load_sector_lookup(cfg.get("tables", {}).get("sector_aliases_file"))

    if holdings_raw is None:
        try:
            holdings_raw = data.load_latest_holdings_from_aggregated(str(perf_file), top_n=None)
        except Exception:
            holdings_raw = None

    if holdings_raw is not None:
        if top_holdings_df is None:
            top_limit = cfg.get("tables", {}).get("top_holdings_limit", 10)
            table_slice = holdings_raw.head(top_limit)
            top_holdings_df = pd.DataFrame(
                {
                    "Name": table_slice["name"],
                    "% of NAV": table_slice["weight"].apply(lambda x: format_pct(x)),
                }
            )
        sector_weights = {}
        region_weights = {}
        cache_dirty = False
        for row in holdings_raw.itertuples():
            raw_sector, raw_industry, raw_country, override_sector, override_region, override_country, yf_symbol_override, changed = data.resolve_symbol_classification(
                getattr(row, "symbol", None),
                getattr(row, "name", None),
                overrides,
                classification_cache,
                sector_lookup,
            )
            if changed:
                cache_dirty = True

            symbol_key = data.canonical_symbol(getattr(row, "symbol", None))
            name_key = str(getattr(row, "name", "")).strip().upper()

            def map_sector_value() -> tuple[str, bool]:
                if override_sector:
                    return override_sector.strip(), True
                for key_name, value in (("symbol", symbol_key), ("name", name_key), ("yf_sector", raw_sector), ("industry", raw_industry)):
                    if not value:
                        continue
                    mapped = sector_lookup.get(key_name.lower(), {}).get(str(value).strip().lower())
                    if mapped:
                        return mapped, False
                if raw_sector:
                    mapped = sector_alias_map.get(str(raw_sector).strip(), None)
                    if mapped:
                        return mapped, False
                if raw_industry:
                    mapped = sector_alias_map.get(str(raw_industry).strip(), None)
                    if mapped:
                        return mapped, False
                return "Unclassified", False

            sector_normalized, sector_from_override = map_sector_value()
            if sector_normalized not in sector_display_order and not sector_from_override:
                alias = sector_alias_map.get(sector_normalized, None) if not sector_from_override else None
                if alias:
                    sector_normalized = alias
            if sector_normalized not in sector_display_order:
                sector_normalized = "Unclassified"

            def map_region_value() -> tuple[str, bool]:
                if override_region:
                    return override_region.strip(), True
                country_candidate = override_country or raw_country
                if country_candidate:
                    mapped = region_map.get(str(country_candidate).strip().lower())
                    if mapped:
                        return mapped, False
                if raw_country:
                    mapped = region_alias_map.get(str(raw_country).strip().lower())
                    if mapped:
                        return mapped, False
                return "Other", False

            region_value, region_from_override = map_region_value()
            if region_value not in region_display_order and not region_from_override:
                alias = region_alias_map.get(str(region_value).strip().lower(), None)
                if alias:
                    region_value = alias
            if region_value not in region_display_order:
                region_value = "Cash" if str(region_value).strip().lower() == "cash" else "Other"

            weight = float(getattr(row, "weight", 0.0))
            sector_weights[sector_normalized] = sector_weights.get(sector_normalized, 0.0) + weight
            region_weights[region_value] = region_weights.get(region_value, 0.0) + weight
            classified_records.append(
                {
                    "date": str(getattr(row, "Date", "")),
                    "symbol": getattr(row, "symbol", ""),
                    "name": getattr(row, "name", ""),
                    "weight": weight,
                    "weight_pct": weight * 100.0,
                    "yf_symbol": yf_symbol_override or symbol_key,
                    "raw_sector": raw_sector,
                    "raw_industry": raw_industry,
                    "override_sector": override_sector,
                    "sector": sector_normalized,
                    "raw_country": raw_country,
                    "override_region": override_region or override_country,
                    "region": region_value,
                }
            )
        if cache_dirty:
            data.save_classification_cache(classification_cache, classification_cache_path)
        total_sector = sum(sector_weights.values())
        if total_sector > 0:
            for key in list(sector_weights.keys()):
                sector_weights[key] /= total_sector
        total_region = sum(region_weights.values())
        if total_region > 0:
            for key in list(region_weights.keys()):
                region_weights[key] /= total_region

    vol_table_df = None
    if port_monthly is not None and bench_monthly is not None:
        port_monthly_clean = port_monthly.dropna()
        bench_monthly_clean = bench_monthly.dropna()
        if not port_monthly_clean.empty and not bench_monthly_clean.empty:
            port_mean = port_monthly_clean.mean()
            bench_mean = bench_monthly_clean.mean()
            port_std = port_monthly_clean.std(ddof=1)
            bench_std = bench_monthly_clean.std(ddof=1)
            port_ann_vol = port_std * math.sqrt(12) if pd.notna(port_std) else None
            bench_ann_vol = bench_std * math.sqrt(12) if pd.notna(bench_std) else None
            port_sharpe = None
            if avg_yearly_return is not None and port_ann_vol and port_ann_vol != 0:
                port_sharpe = avg_yearly_return / port_ann_vol
            bench_sharpe = None
            if bench_cagr is not None and bench_ann_vol and bench_ann_vol != 0:
                bench_sharpe = bench_cagr / bench_ann_vol
            vol_table_df = pd.DataFrame(
                {
                    "Metric": ["Mean monthly return", "Std Dev (Monthly)", "Annualized Volatility", "Sharpe ratio"],
                    "Portfolio": [
                        format_pct(port_mean),
                        format_pct(port_std),
                        format_pct(port_ann_vol),
                        format_ratio(port_sharpe),
                    ],
                    "Benchmark": [
                        format_pct(bench_mean),
                        format_pct(bench_std),
                        format_pct(bench_ann_vol),
                        format_ratio(bench_sharpe),
                    ],
                }
            )

    port_zmw = None
    if fx_series is not None and not fx_series.empty:
        try:
            fx_rel = fx_series / fx_series.iloc[0]
            fx_rel.iloc[0] = 1.0
            port_series = pd.Series(port_usd, index=perf_categories)
            port_zmw_series = (1 + port_series) * fx_rel - 1.0
            port_zmw_series.iloc[0] = 0.0
            port_zmw = port_zmw_series.tolist()
        except Exception:
            port_zmw = None
    if port_zmw is None:
        port_zmw = gen_path(mu=0.0006, sigma=0.012, n=total_points, seed=44)

    struct_series_names = perf_struct["series_names"] if perf_struct else None
    series_names = struct_series_names or [
        "Portfolio return (USD)",
        "Benchmark return (USD)",
        "Portfolio return (ZMW)",
        "Benchmark return (ZMW)",
    ]
    # Map by name intent; default to portfolio/benchmark usd/zmw
    perf_series = {}
    for n in series_names:
        ln = n.lower()
        if "zmw" in ln and "bench" in ln:
            perf_series[n] = bench_zmw
        elif "zmw" in ln:
            perf_series[n] = port_zmw
        elif "bench" in ln:
            perf_series[n] = bench_usd
        else:
            perf_series[n] = port_usd
    perf_native_ok = ppt.replace_native_chart_data_by_name(
        prs.slides,
        "CHART_PERF_LINE",
        perf_categories.to_pydatetime().tolist(),
        perf_series,
        number_format="0%",
        percent_y_axis=True,
        date_categories=True,
        date_format="dd.mm.yyyy",
    )
    chart_ok = True
    if not perf_native_ok:
        chart_ok = ppt.replace_picture_by_name(prs.slides, "CHART_PERF_LINE", perf_line_path)

    if ret_since_incept is None and port_usd:
        ret_since_incept = port_usd[-1]
    if avg_yearly_return is None and ret_since_incept is not None and perf_categories is not None and len(perf_categories) > 1:
        total_years = (perf_categories[-1] - perf_categories[0]).days / 365.25
        if total_years > 0:
            avg_yearly_return = (1.0 + ret_since_incept) ** (1.0 / total_years) - 1.0

    bars_native_ok = False
    if port_monthly is not None and bench_monthly is not None:
        recent_index = port_monthly.index.union(bench_monthly.index)
        recent_index = recent_index[recent_index <= month_end_as_of]
        recent_index = recent_index.sort_values()
        recent_index = recent_index[-24:]
        if len(recent_index) > 0:
            categories = [ts.strftime("%b %y") for ts in recent_index]
            port_vals = port_monthly.reindex(recent_index).fillna(0.0).tolist()
            bench_vals = bench_monthly.reindex(recent_index).fillna(0.0).tolist()
            bars_native_ok = ppt.replace_native_chart_data_by_name(
                prs.slides,
                "CHART_MONTHLY_BARS",
                categories,
                {"Portfolio": port_vals, "Benchmark": bench_vals},
                number_format="0.00%",
                percent_y_axis=True,
                force_text_categories=True,
            )
    if not bars_native_ok:
        import numpy as np
        bars_categories = list(range(1, 25))
        port = np.random.normal(loc=0.01, scale=0.05, size=24)
        bench = port - np.random.normal(loc=0.003, scale=0.02, size=24)
        bars_native_ok = ppt.replace_native_chart_data_by_name(
            prs.slides,
            "CHART_MONTHLY_BARS",
            bars_categories,
            {"Portfolio": port.tolist(), "Benchmark": bench.tolist()},
            number_format="0.00%",
            percent_y_axis=True,
            force_text_categories=True,
        )
    if not bars_native_ok:
        ppt.replace_picture_by_name(prs.slides, "CHART_MONTHLY_BARS", monthly_bars_path)
    # Try native pies; fallback to PNGs
    sector_struct = ppt.get_chart_structure(prs.slides, "PIE_SECTOR")
    default_sector_labels = sector_display_order
    sector_labels_existing = sector_struct["categories"] if sector_struct else None
    sector_labels, sector_values = prepare_pie_data(sector_weights, default_sector_labels)
    sector_ok = ppt.replace_native_chart_data_by_name(
        prs.slides,
        "PIE_SECTOR",
        sector_labels,
        {"Weight": sector_values},
        number_format="0%",
    )
    if not sector_ok:
        ppt.replace_picture_by_name(prs.slides, "PIE_SECTOR", pie_sector_path)

    region_struct = ppt.get_chart_structure(prs.slides, "PIE_REGION")
    default_region_labels = region_display_order
    region_labels, region_values = prepare_pie_data(region_weights, default_region_labels)
    region_ok = ppt.replace_native_chart_data_by_name(
        prs.slides,
        "PIE_REGION",
        region_labels,
        {"Weight": region_values},
        number_format="0%",
    )
    if not region_ok:
        ppt.replace_picture_by_name(prs.slides, "PIE_REGION", pie_region_path)

    # Fallbacks: if placeholders missing, append slides so output is visible
    if not chart_ok:
        ppt.add_picture_slide(prs, perf_line_path, title_text="Performance (demo)")
    if not date_ok:
        ppt.add_textbox_slide(prs, title="Date Issued", body=issued_text)

    commentary_cfg = cfg.get("commentary", {})
    if commentary_cfg.get("enabled", False):
        sector_for_prompt: list[tuple[str, float]] = []
        if sector_weights:
            for label in sector_display_order:
                value = sector_weights.get(label, 0.0)
                if value > 0.001:
                    sector_for_prompt.append((label, value))
        region_for_prompt: list[tuple[str, float]] = []
        if region_weights:
            for label in region_display_order:
                value = region_weights.get(label, 0.0)
                if value > 0.001:
                    region_for_prompt.append((label, value))
        holdings_for_prompt = sorted(classified_records, key=lambda item: item.get("weight", 0.0), reverse=True)[:10]

        # Get monthly returns for all instruments and add to holdings
        monthly_returns_by_symbol = {}
        try:
            monthly_returns_by_symbol = data.get_monthly_returns_by_symbol(str(perf_file), as_of)
            # Add monthly return to each holding
            for holding in holdings_for_prompt:
                symbol = holding.get('yf_symbol') or holding.get('symbol')
                if symbol and symbol in monthly_returns_by_symbol:
                    holding['monthly_return'] = monthly_returns_by_symbol[symbol]
                    holding['monthly_return_pct'] = monthly_returns_by_symbol[symbol] * 100
        except Exception as e:
            print(f"Warning: Failed to get monthly returns: {e}")

        # Get current month performance
        port_mtd = None
        bench_mtd = None
        if port_monthly is not None and not port_monthly.empty:
            port_mtd = port_monthly.iloc[-1]
        if bench_monthly is not None and not bench_monthly.empty:
            bench_mtd = bench_monthly.iloc[-1]

        # Calculate top winners and losers for the month (by value add/loss)
        top_winners = []
        top_losers = []
        print(f"\nCalculating monthly winners/losers by value impact from {perf_file}...")
        try:
            winners, losers = data.calculate_monthly_winners_losers(
                str(perf_file),
                as_of,
                holdings=classified_records,
                top_n=4
            )
            top_winners = winners
            top_losers = losers
            print(f"  ✓ Found {len(winners)} winners and {len(losers)} losers")
            if winners:
                w = winners[0]
                print(f"  Top winner: {w['name']} (return: {w['return']:.2%}, value add: {w['value_add']:.2%})")
            if losers:
                l = losers[0]
                print(f"  Top loser: {l['name']} (return: {l['return']:.2%}, value add: {l['value_add']:.2%})")
        except Exception as e:
            print(f"  ✗ Warning: Failed to calculate monthly winners/losers: {e}")
            import traceback
            traceback.print_exc()

        commentary_context = {
            "as_of": as_of.isoformat(),
            "portfolio": {
                "ytd": format_pct(ytd_return),
                "since_inception": format_pct(ret_since_incept),
                "cagr": format_pct(avg_yearly_return),
                "mtd": format_pct(port_mtd) if port_mtd is not None and pd.notna(port_mtd) else "n/a",
            },
            "benchmark": {
                "ytd": format_pct(bench_ytd) if bench_ytd is not None else "n/a",
                "cagr": format_pct(bench_cagr) if bench_cagr is not None else "n/a",
                "mtd": format_pct(bench_mtd) if bench_mtd is not None and pd.notna(bench_mtd) else "n/a",
            },
            "sectors": sector_for_prompt,
            "regions": region_for_prompt,
            "holdings": holdings_for_prompt,
            "top_winners": top_winners,
            "top_losers": top_losers,
        }

        commentary_result = commentary.generate_commentary(commentary_context, commentary_cfg)
        commentary_text = commentary_result.get("text", "")
        if commentary_text:
            ppt.set_text_by_name(prs.slides, "COMMENTARY_TEXT", commentary_text)
        history_path = commentary_cfg.get("history_file") or commentary_cfg.get("save_debug_file")
        if history_path:
            history_path_obj = Path(history_path)
            history_path_obj.parent.mkdir(parents=True, exist_ok=True)
            debug_payload = {
                "context": commentary_context,
                "result": commentary_result,
                "run_at": dt.datetime.now().isoformat(),
            }
            commentary.save_debug_payload(debug_payload, str(history_path_obj))
    ppt.set_text_by_name(prs.slides, "RET_SINCE_INCEPT", format_pct(ret_since_incept))
    ppt.set_text_by_name(prs.slides, "YTD_RETURN_USD", format_pct(ytd_return))
    ppt.set_text_by_name(prs.slides, "AVG_Y_COMP_RET_SINCE_INCEPT", format_pct(avg_yearly_return))

    review_path = cfg.get("tables", {}).get("classification_review_file")
    if classified_records and review_path:
        try:
            Path(review_path).parent.mkdir(parents=True, exist_ok=True)
            with open(review_path, "w", encoding="utf-8") as f:
                json.dump(classified_records, f, indent=2, sort_keys=False)
        except OSError:
            pass

    if top_holdings_df is None:
        top_holdings_df = pd.DataFrame(
            {
                "Name": ["NVIDIA", "TSMC", "First Quantum", "JPMorgan", "Meta"],
                "% of NAV": ["8.44%", "7.74%", "7.50%", "7.36%", "6.43%"],
            }
        )
    ppt.write_table_by_name(prs.slides, "TOP_HOLDINGS_TABLE", top_holdings_df)

    # Volatility table
    if vol_table_df is None:
        vol_table_df = pd.DataFrame(
            {
                "Metric": ["Mean monthly return", "Std Dev (Monthly)", "Annualized Volatility", "Sharpe ratio"],
                "Portfolio": ["1.05%", "5.36%", "18.58%", "0.48"],
                "Benchmark": ["0.81%", "4.51%", "15.64%", "0.38"],
            }
        )
    ppt.write_table_by_name(prs.slides, "VOLATILITY_TABLE", vol_table_df)

    # Monthly returns table
    if monthly_table_df is None:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","FY/YTD"]
        def row_vals(seed: int):
            base = [((i * seed) % 9 - 4) / 10 for i in range(1, 13)]
            fy = sum(base) / 12 * 12
            return [f"{v:+.2f}%" for v in base] + [f"{fy:+.2f}%"]

        rows = []
        for yr, s_port, s_bm in [(2021, 3, 2), (2022, 5, 4), (2023, 7, 6), (2024, 9, 8), (2025, 11, 10)]:
            rows.append([str(yr)] + row_vals(s_port))
            rows.append(["Benchmark"] + row_vals(s_bm))

        monthly_table_df = pd.DataFrame(rows, columns=["Year"] + months)
    ppt.write_table_by_name(prs.slides, "MONTHLY_RETURNS_TABLE", monthly_table_df)

    # Fee details table demo (two columns: Item / Value)
    fees_df = pd.DataFrame(
        {
            "Item": [
                "Annual Management Fee (Incl. VAT)",
                "Performance Fee",
                "Transaction Costs & Commissions",
                "Platform and custodian charges",
            ],
            "Value": [
                "1 %",
                "None",
                "See schedule",
                "7.5 bps p.a. on NAV",
            ],
        }
    )
    ppt.write_table_by_name(prs.slides, "FEES_TABLE", fees_df)

    # Save
    prs.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build factsheet deck")
    parser.add_argument("command", choices=["build", "audit"], help="Command to run")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    if args.command == "build":
        out = build(args.config)
        print(out)
    elif args.command == "audit":
        cfg = load_config(args.config)
        prs = ppt.open_presentation(cfg["output"]["ppt_template"])
        report = ppt.list_shape_names(prs)
        # Print concise summary grouped by slide
        by_slide = {}
        for row in report:
            by_slide.setdefault(row["slide"], []).append(row)
        for slide_num in sorted(by_slide):
            print(f"Slide {slide_num}:")
            for row in by_slide[slide_num]:
                name = row["name"]
                has_text = "text" if row["has_text"] else ""
                print(f"  - {name} {has_text}")
        # Also print quick placeholder presence summary
        wanted = ["DATE_ISSUED","BENCH_NAME","CHART_PERF_LINE","CHART_MONTHLY_BARS","TOP_HOLDINGS_TABLE","VOLATILITY_TABLE","MONTHLY_RETURNS_TABLE","PIE_SECTOR","PIE_REGION","FEES_TABLE"]
        present = {w: False for w in wanted}
        for row in report:
            if row["name"] in present:
                present[row["name"]] = True
        print("Placeholders:")
        for w, ok in present.items():
            print(f"  {w}: {'OK' if ok else 'MISSING'}")


if __name__ == "__main__":
    main()



