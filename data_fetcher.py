"""
data_fetcher.py — Fixed version
================================
Key fixes:
- Cache saved as CSV (no pyarrow needed) with parquet as secondary try
- Better API pagination and error handling  
- Real FX volatility from year-range data
- Proper governance score from WGI indicators
"""

from __future__ import annotations
import os, time
import requests
import pandas as pd
import numpy as np

WB_BASE = "https://api.worldbank.org/v2"
CSV_CACHE   = "data_cache.csv"       # Always works, no extra deps
PARQUET_CACHE = "data_cache.parquet" # Used if pyarrow available

AGGREGATE_ISO3 = {
    "WLD","EAP","ECA","LAC","MNA","NAC","SAS","SSA","HIC","LIC",
    "LMC","UMC","MIC","EUU","EMU","ARB","FCS","LMY","OED","PST",
    "IBD","IDB","IDX","INX","TEC","TEA","TLA","TMN","TSA","TSS",
}

INDICATORS_SNAPSHOT = {
    "trade_openness": "NE.TRD.GNFS.ZS",
    "tariff_rate":    "TM.TAX.MRCH.WM.AR.ZS",
    "lpi_score":      "LP.LPI.OVRL.XQ",
    "gdp_growth":     "NY.GDP.MKTP.KD.ZG",
    "inflation":      "FP.CPI.TOTL.ZG",
}
FX_INDICATOR = "PA.NUS.FCRF"
WGI_CODES    = ["GE.EST","PV.EST","RL.EST","CC.EST","RQ.EST","VA.EST"]


# ── Cache helpers ──────────────────────────────────────────────────────────
def _write_cache(df: pd.DataFrame) -> None:
    """Save to CSV (always) and parquet (if pyarrow available)."""
    try:
        df.to_csv(CSV_CACHE, index=False)
        print(f"  💾  Cache written → {CSV_CACHE}")
    except Exception as e:
        print(f"  ⚠  CSV cache write failed: {e}")
    try:
        df.to_parquet(PARQUET_CACHE, index=False)
        print(f"  💾  Cache written → {PARQUET_CACHE}")
    except Exception:
        pass  # pyarrow not installed, that's fine


def _read_cache() -> pd.DataFrame | None:
    """Try CSV first, then parquet."""
    # CSV
    if os.path.exists(CSV_CACHE):
        try:
            df = pd.read_csv(CSV_CACHE)
            if len(df) > 10:
                print(f"  📦  Read cache from {CSV_CACHE} ({len(df)} rows)")
                return df
        except Exception:
            pass
    # Parquet
    if os.path.exists(PARQUET_CACHE):
        try:
            df = pd.read_parquet(PARQUET_CACHE)
            if len(df) > 10:
                print(f"  📦  Read cache from {PARQUET_CACHE} ({len(df)} rows)")
                return df
        except Exception:
            pass
    return None


# ── WB API helpers ─────────────────────────────────────────────────────────
def _wb_get(url: str, params: dict, timeout: int = 45) -> list:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _api_ok() -> bool:
    try:
        js = _wb_get(
            f"{WB_BASE}/country/USA/indicator/NY.GDP.MKTP.KD.ZG",
            {"format":"json","mrv":1,"per_page":5,"page":1}, timeout=10)
        return isinstance(js, list) and len(js) >= 2
    except Exception:
        return False


def _fetch_indicator(indicator: str, *, date: str | None = None,
                     mrv: int | None = None, source: int | None = None) -> pd.DataFrame:
    """Fetch all countries for one indicator (paginated)."""
    params = {"format":"json","per_page":1000,"page":1}
    if date:    params["date"]   = date
    if mrv:     params["mrv"]    = mrv
    if source:  params["source"] = str(source)

    url = f"{WB_BASE}/country/all/indicator/{indicator}"
    rows, page = [], 1

    while True:
        params["page"] = page
        try:
            data = _wb_get(url, params)
        except Exception as e:
            print(f"    ⚠  {indicator} page {page}: {e}")
            break

        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            break

        for item in data[1]:
            val = item.get("value")
            iso = item.get("countryiso3code","")
            name = (item.get("country") or {}).get("value","")
            yr   = item.get("date","0")
            if val is not None and iso and name and len(iso) == 3:
                rows.append({"iso3": iso, "country": name,
                             "year": int(yr), "value": float(val)})

        pages = int((data[0] or {}).get("pages", 1))
        if page >= pages:
            break
        page += 1
        time.sleep(0.05)

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["iso3","country","year","value"])


def _latest(raw: pd.DataFrame, col: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["iso3", col])
    latest = raw.sort_values("year").groupby("iso3", as_index=False).tail(1)
    return latest[["iso3","country","value"]].rename(columns={"value": col})


def _fx_volatility(fx_hist: pd.DataFrame) -> pd.DataFrame:
    if fx_hist.empty:
        return pd.DataFrame(columns=["iso3","fx_volatility"])
    d = fx_hist.sort_values(["iso3","year"]).copy()
    d["pct"] = d.groupby("iso3")["value"].pct_change() * 100.0
    vol = (d.groupby("iso3")["pct"]
            .std()
            .reset_index()
            .rename(columns={"pct":"fx_volatility"}))
    return vol


def _governance(wgi_start=2019, wgi_end=2022) -> pd.DataFrame:
    blocks = []
    for code in WGI_CODES:
        raw = _fetch_indicator(code, date=f"{wgi_start}:{wgi_end}", source=3)
        if raw.empty:
            continue
        agg = (raw.groupby("iso3")["value"]
                  .mean().reset_index()
                  .rename(columns={"value": code}))
        blocks.append(agg)
        time.sleep(0.1)

    if not blocks:
        return pd.DataFrame(columns=["iso3","governance_score"])

    gov = blocks[0]
    for b in blocks[1:]:
        gov = gov.merge(b, on="iso3", how="outer")

    wgi_cols = [c for c in gov.columns if c != "iso3"]
    gov["governance_score_raw"] = gov[wgi_cols].mean(axis=1)
    mn, mx = gov["governance_score_raw"].min(), gov["governance_score_raw"].max()
    if pd.notna(mn) and pd.notna(mx) and mx > mn:
        gov["governance_score"] = (gov["governance_score_raw"] - mn) / (mx - mn) * 10.0
    else:
        gov["governance_score"] = 5.0
    return gov[["iso3","governance_score"]]


def _filter_countries(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["iso3"].notna() & (df["iso3"].str.len() == 3)].copy()
    df = df[~df["iso3"].isin(AGGREGATE_ISO3)]
    df = df[~df["country"].str.contains("Aggregates|income|dividend|IBRD|IDA|HIPC",
                                         case=False, na=False, regex=True)]
    return df.reset_index(drop=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["trade_openness","tariff_rate","lpi_score","fx_volatility",
                "governance_score","gdp_growth","inflation"]
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        med = df[c].median()
        df[c] = df[c].fillna(med if pd.notna(med) else 0.0)

    # Cap extreme outliers at 99th percentile
    for c in ["fx_volatility","inflation","tariff_rate"]:
        cap = df[c].quantile(0.99)
        if pd.notna(cap):
            df[c] = df[c].clip(upper=cap)

    return df[["iso3","country","trade_openness","tariff_rate","lpi_score",
               "fx_volatility","governance_score","gdp_growth","inflation"]].copy()


# ── Fallback sample data ───────────────────────────────────────────────────
def _get_fallback_data() -> pd.DataFrame:
    data = [
        ("USA","United States",27.5,1.6,4.23,1.2,2.1,4.7,8.2),
        ("CHN","China",37.4,3.4,3.68,0.5,5.2,2.1,4.8),
        ("DEU","Germany",88.1,1.5,4.20,0.8,1.8,7.9,8.5),
        ("GBR","United Kingdom",64.2,2.0,4.07,5.2,4.0,9.1,8.3),
        ("FRA","France",63.8,1.5,3.84,0.8,2.6,5.9,8.0),
        ("JPN","Japan",39.8,2.5,4.03,7.2,1.0,2.5,8.1),
        ("IND","India",46.0,7.1,3.18,4.1,6.7,6.7,4.9),
        ("BRA","Brazil",35.2,8.0,2.99,12.5,2.9,9.3,4.5),
        ("CAN","Canada",66.0,1.0,3.99,3.8,3.4,6.8,8.4),
        ("AUS","Australia",47.8,1.3,3.79,5.1,3.7,6.6,8.6),
        ("KOR","Korea, Rep.",84.7,8.7,3.59,3.2,2.6,5.1,7.5),
        ("MEX","Mexico",79.8,4.4,3.13,6.8,3.0,7.9,4.0),
        ("IDN","Indonesia",40.3,3.7,3.15,5.9,5.3,4.2,4.8),
        ("TUR","Turkey",62.6,2.3,3.38,28.4,5.6,72.3,3.8),
        ("SAU","Saudi Arabia",68.6,4.6,3.16,0.3,8.7,3.1,4.7),
        ("ARG","Argentina",34.6,8.7,2.80,45.2,8.3,72.4,3.2),
        ("ZAF","South Africa",60.2,5.8,3.38,9.4,1.9,7.0,5.2),
        ("NGA","Nigeria",26.4,10.5,2.53,18.7,3.4,18.8,2.5),
        ("EGY","Egypt",40.6,9.3,2.82,22.1,6.6,14.6,3.1),
        ("NLD","Netherlands",155.3,1.5,4.23,0.8,4.9,11.6,8.8),
        ("CHE","Switzerland",120.6,4.0,4.10,3.1,2.0,2.8,9.0),
        ("SWE","Sweden",91.6,1.5,4.05,5.8,2.8,10.9,9.1),
        ("NOR","Norway",83.5,2.2,4.05,5.2,3.3,5.8,9.2),
        ("DNK","Denmark",101.8,1.5,4.18,0.8,3.8,8.5,9.3),
        ("SGP","Singapore",320.5,0.2,4.30,0.9,3.6,6.1,9.1),
        ("VNM","Vietnam",209.4,3.7,2.96,2.1,8.0,3.2,4.5),
        ("PAK","Pakistan",29.0,8.5,2.43,12.8,5.7,19.9,2.9),
        ("IRN","Iran",42.6,22.4,2.35,85.2,1.7,43.4,2.1),
        ("RUS","Russia",53.8,5.5,2.76,18.2,4.7,13.7,2.8),
        ("UKR","Ukraine",87.2,2.7,2.57,22.4,3.4,20.2,3.5),
        ("KEN","Kenya",33.8,12.6,2.81,7.8,4.9,7.9,3.8),
        ("ETH","Ethiopia",28.6,14.6,2.30,18.5,6.3,33.5,2.8),
        ("GHA","Ghana",70.2,12.5,2.62,26.2,3.2,31.5,5.2),
        ("MAR","Morocco",76.4,11.0,2.97,4.2,7.9,6.6,4.8),
        ("DZA","Algeria",49.8,16.5,2.43,4.8,3.4,9.3,3.2),
        ("TUN","Tunisia",102.0,13.4,2.75,8.6,3.1,8.3,4.9),
        ("ARE","United Arab Emirates",176.6,4.8,4.00,0.3,7.4,4.8,7.2),
        ("NZL","New Zealand",57.4,1.4,3.88,5.8,2.4,7.2,9.0),
        ("ESP","Spain",70.8,1.5,3.73,0.8,5.5,8.4,7.5),
        ("ITA","Italy",63.6,1.5,3.76,0.8,3.7,8.7,6.8),
        ("POL","Poland",110.5,1.5,3.43,5.2,5.7,14.4,6.8),
        ("COL","Colombia",42.6,5.8,2.94,7.2,10.6,10.2,4.5),
        ("CHL","Chile",59.8,1.0,3.32,9.4,11.7,12.8,7.2),
        ("PER","Peru",52.6,2.0,2.85,8.6,2.7,7.9,4.8),
        ("VEN","Venezuela",44.4,14.2,2.03,156.4,-3.5,284.4,1.2),
        ("PHL","Philippines",67.5,4.3,2.90,4.6,7.6,5.8,4.7),
        ("MYS","Malaysia",137.8,5.0,3.22,3.8,8.7,3.4,6.0),
        ("THA","Thailand",122.4,5.4,3.26,3.2,2.6,6.1,5.5),
        ("BGD","Bangladesh",38.5,13.2,2.59,3.4,7.1,5.5,3.5),
        ("PRY","Paraguay",82.6,4.6,2.72,8.1,4.2,9.8,3.8),
        ("BOL","Bolivia",45.8,8.6,2.46,5.2,6.0,1.7,3.5),
        ("ECU","Ecuador",51.4,5.2,2.72,5.8,4.8,3.5,4.2),
        ("GTM","Guatemala",58.2,4.8,2.63,3.8,8.0,6.4,3.8),
        ("HND","Honduras",90.2,5.2,2.52,6.2,12.5,8.6,3.4),
        ("SEN","Senegal",50.2,12.8,2.68,5.6,5.2,5.9,4.6),
        ("CIV","Cote d'Ivoire",82.4,11.2,2.71,3.8,6.8,5.4,4.2),
        ("CMR","Cameroon",45.2,16.8,2.36,4.2,3.8,6.8,3.2),
        ("TZA","Tanzania",39.2,10.8,2.51,5.4,4.9,4.0,3.5),
        ("UGA","Uganda",41.6,12.4,2.42,6.8,6.0,3.8,3.2),
        ("ZMB","Zambia",72.6,14.6,2.38,22.4,4.2,24.6,3.1),
    ]
    cols = ["iso3","country","trade_openness","tariff_rate","lpi_score",
            "fx_volatility","gdp_growth","inflation","governance_score"]
    df = pd.DataFrame(data, columns=cols)
    mn, mx = df["governance_score"].min(), df["governance_score"].max()
    if mx > mn:
        df["governance_score"] = (df["governance_score"] - mn) / (mx - mn) * 10.0
    return df


# ── Public entry point ─────────────────────────────────────────────────────
def fetch_all_indicators(
    *,
    use_live_api: bool = True,
    cache_path: str = PARQUET_CACHE,
    allow_cache_read: bool = True,
    allow_cache_write: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Returns clean DataFrame with 9 columns:
    iso3, country, trade_openness, tariff_rate, lpi_score,
    fx_volatility, governance_score, gdp_growth, inflation
    """

    if not use_live_api:
        cached = _read_cache()
        if cached is not None:
            return cached
        return _get_fallback_data()

    print("  🔌  Checking World Bank API…")
    if not _api_ok():
        print("  ⚠  API unreachable.")
        cached = _read_cache()
        return cached if cached is not None else _get_fallback_data()

    print("  ✅  API OK — fetching live data…")

    # ── Snapshot indicators ────────────────────────────────────────────────
    merged = None
    for col, code in INDICATORS_SNAPSHOT.items():
        print(f"  📡  {col}…")
        raw = _fetch_indicator(code, mrv=1)
        snap = _latest(raw, col)
        if merged is None:
            merged = snap.copy()
        else:
            merged = merged.merge(snap[["iso3", col]], on="iso3", how="left")
        time.sleep(0.1)

    if merged is None or len(merged) < 10:
        print("  ⚠  Snapshot fetch returned empty — trying cache.")
        cached = _read_cache()
        return cached if cached is not None else _get_fallback_data()

    # ── FX volatility ──────────────────────────────────────────────────────
    print("  📡  FX rates (2010-2024)…")
    fx_hist = _fetch_indicator(FX_INDICATOR, date="2010:2024")
    fx_vol  = _fx_volatility(fx_hist)
    merged  = merged.merge(fx_vol, on="iso3", how="left")

    # ── Governance ─────────────────────────────────────────────────────────
    print("  📡  Governance indicators (WGI)…")
    gov    = _governance(2019, 2022)
    merged = merged.merge(gov, on="iso3", how="left")

    # ── Clean ─────────────────────────────────────────────────────────────
    merged = _filter_countries(merged)
    merged = _clean(merged)

    n = len(merged)
    print(f"  ✅  Live data ready: {n} countries")
    print(f"      Trade open: mean={merged['trade_openness'].mean():.1f}%  "
          f"Tariff: mean={merged['tariff_rate'].mean():.1f}%  "
          f"LPI: mean={merged['lpi_score'].mean():.2f}  "
          f"FX vol: mean={merged['fx_volatility'].mean():.2f}")

    if allow_cache_write:
        _write_cache(merged)

    return merged