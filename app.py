"""
Equity A/B Testing — Streamlit

Statistically compare two baskets of stocks (Group A vs Group B) on a chosen
performance metric. Instead of an eyeball "side-by-side comparison", this runs
a proper A/B test: permutation test, Welch's t-test and Mann-Whitney U, plus
effect sizes (Hedges' g, Cliff's delta, common-language effect size) and
bootstrap / parametric confidence intervals for the difference.

Ships with AI-related tickers as defaults, but you can test any tickers you add.

All Streamlit calls live inside main()/render helpers, so the module can be
imported for unit testing without a running Streamlit context.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe backend (set before pyplot)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy import stats as sstats

import streamlit as st
import yfinance as yf


# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

CACHE_TTL = 15 * 60          # re-use downloaded prices for 15 minutes
HISTORY_PERIOD = "2y"        # pulled per ticker (>=200 bars so MA200 is valid)
MIN_BARS = 5                 # below this a ticker is treated as "no data"
TRADING_DAYS = {"1mo": 22, "3mo": 66, "6mo": 132, "1y": 252}

# Two-colour categorical palette (A cool / B warm), legible on white.
COL_A = "#4f46e5"   # indigo
COL_B = "#d97706"   # amber
COL_NULL = "#94a3b8"
COL_INK = "#0f172a"
COL_MUTED = "#64748b"

DEFAULT_A = ["NVDA", "AMD", "AVGO", "TSM", "ASML", "MU"]
DEFAULT_B = ["MSFT", "GOOGL", "META", "PLTR", "SNOW", "CRM"]

HYPOTHESES = {
    "Two-sided (A ≠ B)": "two-sided",
    "A greater than B": "greater",
    "B greater than A": "less",
}


# --------------------------------------------------------------------------- #
# Ticker universe (kept verbatim from the ecosystem network-map app)
# --------------------------------------------------------------------------- #

ECOSYSTEMS = {

    "Semiconductors": {
        "SECTOR_COMPANIES": {
            "Chip Designers": ["NVDA", "AMD", "QCOM", "AVGO", "ARM", "AAPL"],
            "Foundries": ["TSM", "0981.HK"],
            "Equipment": ["ASML", "AMAT", "LRCX", "KLAC", "ASM.AS", "BESI.AS", "ICHR", "UCTT"],
            "IDM": ["INTC"],
            "Analog/Power": ["ADI", "TXN", "ON", "STM", "WOLF"],
            "Automated Test Equipment": ["TER", "6857.T", "AEHR"],
            "Test & Assembly (OSAT)": ["AMKR", "ASX", "2449.TW", "6239.TW", "IMOS"],
            "Wafers & Materials": ["4063.T", "3436.T", "6488.TWO", "ENTG"],
        },
        "COMPANY_COUNTRIES": {
            "NVDA": "USA", "AMD": "USA", "QCOM": "USA", "AVGO": "USA", "ARM": "UK", "AAPL": "USA",
            "TSM": "Taiwan", "0981.HK": "China",
            "ASML": "Netherlands", "ASM.AS": "Netherlands", "BESI.AS": "Netherlands",
            "AMAT": "USA", "LRCX": "USA", "KLAC": "USA", "INTC": "USA",
            "ADI": "USA", "TXN": "USA", "ON": "USA", "STM": "Switzerland", "WOLF": "USA",
            "TER": "USA", "6857.T": "Japan", "AEHR": "USA",
            "AMKR": "USA", "ASX": "Taiwan", "2449.TW": "Taiwan", "6239.TW": "Taiwan", "IMOS": "Taiwan",
            "4063.T": "Japan", "3436.T": "Japan", "6488.TWO": "Taiwan", "ENTG": "USA",
            "ICHR": "USA", "UCTT": "USA",
        },
        "SHORT_NAMES": {
            "NVDA": "NVIDIA", "AMD": "AMD", "QCOM": "Qualcomm", "AVGO": "Broadcom", "ARM": "Arm",
            "AAPL": "Apple",
            "TSM": "TSMC", "0981.HK": "SMIC",
            "ASML": "ASML", "AMAT": "Applied Materials", "LRCX": "Lam Research",
            "KLAC": "KLA", "ASM.AS": "ASM Int.", "BESI.AS": "BE Semiconductor",
            "INTC": "Intel", "ADI": "Analog Devices", "TXN": "Texas Instr.", "ON": "Onsemi",
            "STM": "STMicroelectronics", "WOLF": "Wolfspeed",
            "TER": "Teradyne", "6857.T": "Advantest", "AEHR": "Aehr Test Systems",
            "AMKR": "Amkor", "ASX": "ASE Technology", "2449.TW": "King Yuan",
            "6239.TW": "Powertech", "IMOS": "ChipMOS",
            "4063.T": "Shin-Etsu Chemical", "3436.T": "SUMCO", "6488.TWO": "GlobalWafers",
            "ENTG": "Entegris",
            "ICHR": "Ichor Holdings", "UCTT": "Ultra Clean Holdings",
        },
    },

    "Optical & Photonics": {
        "SECTOR_COMPANIES": {
            "Optical Components & Transceivers": ["LITE", "AAOI", "POET", "LWLG"],
            "Optical Semiconductors & Connectivity": ["MTSI", "AXTI"],
            "Photonics & Laser Systems": ["COHR", "LASR"],
            "Optical Manufacturing": ["FN"],
            "Optical Networking": ["NOK"],
            "Fiber Infrastructure": ["GLW"],
        },
        "COMPANY_COUNTRIES": {
            "COHR": "USA", "LITE": "USA", "MTSI": "USA",
            "AAOI": "USA", "POET": "Canada", "FN": "Thailand",
            "LWLG": "USA", "NOK": "Finland", "AXTI": "USA",
            "LASR": "USA", "GLW": "USA",
        },
        "SHORT_NAMES": {
            "COHR": "Coherent", "LITE": "Lumentum", "MTSI": "MACOM",
            "AAOI": "AOI", "FN": "Fabrinet", "POET": "POET",
            "LWLG": "Lightwave", "AXTI": "AXT", "LASR": "nLIGHT",
            "NOK": "Nokia", "GLW": "Corning",
        },
    },

    "Quantum Computing": {
        "SECTOR_COMPANIES": {
            "Quantum Hardware (Trapped Ion)": ["IONQ", "INFQ"],
            "Quantum Hardware (Superconducting)": ["RGTI", "QBTS"],
            "Quantum Hardware (Photonic)": ["XNDU"],
            "Quantum Software & Applications": ["HQ", "QUBT"],
            "Quantum Networking & Security": ["ARQQ"],
            "Quantum Ecosystem Partners": ["IBM", "HON", "GFS"],
        },
        "COMPANY_COUNTRIES": {
            "IONQ": "USA", "INFQ": "USA", "RGTI": "USA", "QBTS": "Canada",
            "XNDU": "Canada", "HQ": "Singapore", "QUBT": "USA", "ARQQ": "UK",
            "IBM": "USA", "HON": "USA", "GFS": "USA",
        },
        "SHORT_NAMES": {
            "IONQ": "IonQ", "INFQ": "Infleqtion", "RGTI": "Rigetti", "QBTS": "D-Wave",
            "XNDU": "Xanadu", "HQ": "Horizon", "QUBT": "QCI", "ARQQ": "Arqit",
            "IBM": "IBM", "HON": "Honeywell", "GFS": "GlobalFoundries",
        },
    },

    "Memory & Storage": {
        "SECTOR_COMPANIES": {
            "DRAM Manufacturers": ["005930.KS", "000660.KS", "MU", "2408.TW", "2344.TW"],
            "NAND Flash Manufacturers": ["285A.T"],
            "Storage & SSD": ["SNDK"],
            "Specialty Memory": ["2344.TW", "2408.TW"],
            "Data Storage Infrastructure": ["STX", "WDC"],
        },
        "COMPANY_COUNTRIES": {
            "005930.KS": "South Korea", "000660.KS": "South Korea", "MU": "USA",
            "STX": "USA", "SNDK": "USA", "285A.T": "Japan",
            "WDC": "USA", "2408.TW": "Taiwan", "2344.TW": "Taiwan",
        },
        "SHORT_NAMES": {
            "005930.KS": "Samsung", "000660.KS": "SK Hynix", "MU": "Micron",
            "STX": "Seagate", "SNDK": "SanDisk", "285A.T": "Kioxia",
            "WDC": "WDC", "2408.TW": "Nanya", "2344.TW": "Winbond",
        },
    },

    "AI Data Center Infrastructure": {
        "SECTOR_COMPANIES": {
            "Data Center Power & Cooling": ["VRT", "ETN"],
            "AI Servers & Infrastructure": ["DELL", "HPE", "SMCI"],
            "AI Data Centers": ["APLD", "EQIX", "DLR"],
            "Server ODMs & Assembly": ["2317.TW", "2382.TW", "6669.TW"],
            "Liquid Cooling": ["3017.TW"],
        },
        "COMPANY_COUNTRIES": {
            "VRT": "USA", "ETN": "Ireland", "DELL": "USA", "HPE": "USA",
            "SMCI": "USA", "APLD": "USA", "EQIX": "USA", "DLR": "USA",
            "2317.TW": "Taiwan", "2382.TW": "Taiwan", "6669.TW": "Taiwan",
            "3017.TW": "Taiwan",
        },
        "SHORT_NAMES": {
            "VRT": "Vertiv", "ETN": "Eaton", "DELL": "Dell", "HPE": "HPE",
            "SMCI": "Supermicro", "APLD": "Applied Digital",
            "EQIX": "Equinix", "DLR": "Digital Realty",
            "2317.TW": "Foxconn (Hon Hai)", "2382.TW": "Quanta Computer", "6669.TW": "Wiwynn",
            "3017.TW": "Asia Vital Components",
        },
    },

    "AI Networking & Connectivity": {
        "SECTOR_COMPANIES": {
            "Data Center Networking": ["ANET", "CSCO"],
            "High-Speed Interconnects": ["APH", "ALAB"],
            "AI Connectivity & Custom Silicon": ["MRVL", "CRDO"],
            "RF Semiconductors": ["QRVO"],
        },
        "COMPANY_COUNTRIES": {
            "ANET": "USA", "CSCO": "USA", "APH": "USA", "ALAB": "USA",
            "MRVL": "USA", "CRDO": "USA", "QRVO": "USA",
        },
        "SHORT_NAMES": {
            "ANET": "Arista", "CSCO": "Cisco", "APH": "Amphenol",
            "ALAB": "Astera Labs", "MRVL": "Marvell", "CRDO": "Credo",
            "QRVO": "Qorvo",
        },
    },

    "Semiconductor Design & Specialty Chips": {
        "SECTOR_COMPANIES": {
            "Electronic Design Automation": ["SNPS", "CDNS"],
            "Edge AI Semiconductors": ["AMBA"],
            "Automotive Semiconductors": ["INDI"],
            "Power Management Semiconductors": ["MPWR"],
            "Microcontrollers & Embedded": ["MCHP"],
        },
        "COMPANY_COUNTRIES": {
            "SNPS": "USA", "CDNS": "USA", "AMBA": "USA", "INDI": "USA", "MPWR": "USA",
            "MCHP": "USA",
        },
        "SHORT_NAMES": {
            "SNPS": "Synopsys", "CDNS": "Cadence", "AMBA": "Ambarella",
            "INDI": "indie Semiconductor", "MPWR": "Monolithic Power",
            "MCHP": "Microchip Technology",
        },
    },

    "Cloud Platforms & AI Compute": {
        "SECTOR_COMPANIES": {
            "Hyperscale Cloud": ["MSFT", "AMZN", "ORCL", "BABA", "GOOGL"],
            "AI Cloud": ["CRWV", "NBIS", "DOCN"],
        },
        "COMPANY_COUNTRIES": {
            "MSFT": "USA", "AMZN": "USA", "ORCL": "USA", "BABA": "China",
            "GOOGL": "USA", "CRWV": "USA", "NBIS": "Netherlands",
            "DOCN": "USA",
        },
        "SHORT_NAMES": {
            "MSFT": "Microsoft", "AMZN": "Amazon", "ORCL": "Oracle",
            "BABA": "Alibaba", "GOOGL": "Alphabet", "CRWV": "CoreWeave",
            "NBIS": "Nebius", "DOCN": "DigitalOcean",
        },
    },

    "Enterprise AI & Software": {
        "SECTOR_COMPANIES": {
            "Enterprise AI": ["PLTR", "AI", "NOW", "IOT"],
            "CRM & Business Applications": ["CRM", "ADBE", "WDAY", "INTU", "VEEV"],
            "Data Platforms": ["SNOW", "MDB"],
            "Developer & Observability Platforms": ["DDOG", "ESTC", "TEAM", "GTLB", "DT"],
            "Automation": ["PATH", "APPN"],
            "Consumer & Social AI": ["META"],
            "Voice & Conversational AI": ["SOUN", "TWLO"],
            "Design & Engineering Software": ["ADSK"],
            "Marketing & Sales Platforms": ["HUBS"],
            "Cloud Storage & Collaboration": ["BOX"],
            "3D & Interactive Development Platforms": ["U"],
        },
        "COMPANY_COUNTRIES": {
            "PLTR": "USA", "AI": "USA", "NOW": "USA", "IOT": "USA", "CRM": "USA", "ADBE": "USA",
            "SNOW": "USA", "MDB": "USA", "DDOG": "USA", "ESTC": "USA",
            "PATH": "USA", "META": "USA", "SOUN": "USA",
            "WDAY": "USA", "ADSK": "USA", "INTU": "USA", "VEEV": "USA",
            "HUBS": "USA", "TEAM": "Australia", "TWLO": "USA", "APPN": "USA",
            "BOX": "USA", "U": "USA", "DT": "USA", "GTLB": "USA",
        },
        "SHORT_NAMES": {
            "PLTR": "Palantir", "AI": "C3.ai", "NOW": "ServiceNow",
            "IOT": "Samsara",
            "CRM": "Salesforce", "ADBE": "Adobe", "SNOW": "Snowflake",
            "MDB": "MongoDB", "DDOG": "Datadog", "ESTC": "Elastic",
            "PATH": "UiPath", "META": "Meta", "SOUN": "SoundHound AI",
            "WDAY": "Workday", "ADSK": "Autodesk", "INTU": "Intuit",
            "VEEV": "Veeva Systems", "HUBS": "HubSpot", "TEAM": "Atlassian",
            "TWLO": "Twilio", "APPN": "Appian", "BOX": "Box",
            "U": "Unity", "DT": "Dynatrace", "GTLB": "GitLab",
        },
    },

    "Cybersecurity": {
        "SECTOR_COMPANIES": {
            "Endpoint & Cloud Security": ["CRWD"],
            "Network Security": ["PANW", "FTNT"],
            "Cloud & Edge Security": ["NET"],
        },
        "COMPANY_COUNTRIES": {
            "CRWD": "USA", "PANW": "USA", "FTNT": "USA", "NET": "USA",
        },
        "SHORT_NAMES": {
            "CRWD": "CrowdStrike", "PANW": "Palo Alto Networks",
            "FTNT": "Fortinet", "NET": "Cloudflare",
        },
    },

    "Energy & Power Infrastructure": {
        "SECTOR_COMPANIES": {
            "Nuclear & Power Generation": ["CEG"],
            "Power Generation & Grid": ["GEV", "NEE"],
        },
        "COMPANY_COUNTRIES": {
            "CEG": "USA", "GEV": "USA", "NEE": "USA",
        },
        "SHORT_NAMES": {
            "CEG": "Constellation Energy",
            "GEV": "GE Vernova",
            "NEE": "NextEra Energy",
        },
    },
}


def build_catalog(ecosystems: dict) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]], List[str]]:
    """Flatten the ecosystem config into lookup tables used by the UI.

    Returns (name_map, country_map, packs, all_tickers) where `packs` maps each
    ecosystem name to the sorted set of tickers it contains.
    """
    name_map: Dict[str, str] = {}
    country_map: Dict[str, str] = {}
    packs: Dict[str, List[str]] = {}

    for eco, cfg in ecosystems.items():
        tickers: set = set()
        for sector_tickers in cfg["SECTOR_COMPANIES"].values():
            for t in sector_tickers:
                tickers.add(t)
                name_map.setdefault(t, cfg["SHORT_NAMES"].get(t, t))
                country_map.setdefault(t, cfg["COMPANY_COUNTRIES"].get(t, "—"))
        packs[eco] = sorted(tickers)

    return name_map, country_map, packs, sorted(name_map)


def validate_ecosystems(ecosystems: dict) -> List[str]:
    """Catch structural mistakes: tickers missing a country or short name."""
    issues: List[str] = []
    for eco, cfg in ecosystems.items():
        seen: set = set()
        for tickers in cfg["SECTOR_COMPANIES"].values():
            seen.update(tickers)
        for t in seen:
            if t not in cfg["COMPANY_COUNTRIES"]:
                issues.append(f"[{eco}] '{t}' missing from COMPANY_COUNTRIES")
            if t not in cfg["SHORT_NAMES"]:
                issues.append(f"[{eco}] '{t}' missing from SHORT_NAMES")
    return issues


# --------------------------------------------------------------------------- #
# Per-ticker metrics
# --------------------------------------------------------------------------- #

def _window(close: pd.Series, timeframe: str) -> pd.Series:
    return close.tail(TRADING_DAYS.get(timeframe, len(close)))


def m_total_return(close: pd.Series, timeframe: str) -> float:
    w = _window(close, timeframe)
    if len(w) < 2 or w.iloc[0] == 0:
        return np.nan
    return (w.iloc[-1] / w.iloc[0] - 1.0) * 100.0


def m_volatility(close: pd.Series, timeframe: str) -> float:
    r = _window(close, timeframe).pct_change().dropna()
    if len(r) < 2:
        return np.nan
    return r.std(ddof=1) * math.sqrt(252) * 100.0


def m_sharpe(close: pd.Series, timeframe: str) -> float:
    r = _window(close, timeframe).pct_change().dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0:
        return np.nan
    return (r.mean() / sd) * math.sqrt(252)


def m_vs_ma200(close: pd.Series, timeframe: str) -> float:
    # Always uses full history — a 200-day average needs 200 bars.
    if len(close) < 200:
        return np.nan
    ma = close.rolling(200).mean().iloc[-1]
    last = close.iloc[-1]
    if not np.isfinite(ma) or ma == 0:
        return np.nan
    return (last / ma - 1.0) * 100.0


def m_max_drawdown(close: pd.Series, timeframe: str) -> float:
    w = _window(close, timeframe)
    if len(w) < 2:
        return np.nan
    return (w / w.cummax() - 1.0).min() * 100.0


METRICS = {
    "Total Return": {
        "fn": m_total_return, "unit": "%", "better": "high", "uses_window": True,
        "help": "Price change from the start to the end of the window.",
    },
    "Annualized Volatility": {
        "fn": m_volatility, "unit": "%", "better": "low", "uses_window": True,
        "help": "Standard deviation of daily returns in the window, annualized.",
    },
    "Sharpe (Return/Risk)": {
        "fn": m_sharpe, "unit": "", "better": "high", "uses_window": True,
        "help": "Mean ÷ std. dev. of daily returns, annualized (risk-free rate = 0).",
    },
    "vs 200-day MA": {
        "fn": m_vs_ma200, "unit": "%", "better": "high", "uses_window": False,
        "help": "Latest price relative to its 200-day moving average. Uses full history, ignores the window.",
    },
    "Max Drawdown": {
        "fn": m_max_drawdown, "unit": "%", "better": "high", "uses_window": True,
        "help": "Largest peak-to-trough drop within the window (a value closer to 0 is better).",
    },
}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected standardized mean difference (A − B)."""
    na, nb = len(a), len(b)
    if na + nb <= 2:
        return np.nan
    pooled = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    sd = math.sqrt(pooled) if pooled > 0 else 0.0
    if sd == 0:
        return np.nan
    d = (a.mean() - b.mean()) / sd
    correction = 1 - 3 / (4 * (na + nb) - 9)
    return d * correction


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]; positive means A tends to exceed B."""
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.mean())


def common_language_es(a: np.ndarray, b: np.ndarray) -> float:
    """P(a random A > a random B), counting ties as half."""
    greater = (a[:, None] > b[None, :]).mean()
    ties = (a[:, None] == b[None, :]).mean()
    return float(greater + 0.5 * ties)


def permutation_test(a: np.ndarray, b: np.ndarray, alt: str,
                     iters: int, seed: int) -> Tuple[float, float, np.ndarray]:
    """Permutation test on the difference in means (A − B)."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    na, n = len(a), len(pooled)
    obs = a.mean() - b.mean()

    order = np.argsort(rng.random((iters, n)), axis=1)
    perm = pooled[order]
    diffs = perm[:, :na].mean(axis=1) - perm[:, na:].mean(axis=1)

    eps = 1e-12
    if alt == "greater":
        p = (np.sum(diffs >= obs - eps) + 1) / (iters + 1)
    elif alt == "less":
        p = (np.sum(diffs <= obs + eps) + 1) / (iters + 1)
    else:
        p = (np.sum(np.abs(diffs) >= abs(obs) - eps) + 1) / (iters + 1)
    return obs, float(p), diffs


def bootstrap_diff(a: np.ndarray, b: np.ndarray, iters: int, seed: int,
                   alpha: float) -> Tuple[np.ndarray, float, float]:
    """Percentile bootstrap CI for the difference in means (A − B)."""
    rng = np.random.default_rng(seed + 7)
    na, nb = len(a), len(b)
    ba = a[rng.integers(0, na, size=(iters, na))].mean(axis=1)
    bb = b[rng.integers(0, nb, size=(iters, nb))].mean(axis=1)
    diffs = ba - bb
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diffs, float(lo), float(hi)


def _shapiro_p(x: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    try:
        return float(sstats.shapiro(x).pvalue)
    except Exception:
        return np.nan


def run_test(a_tickers: Sequence[str], b_tickers: Sequence[str],
             prices: Dict[str, pd.DataFrame],
             name_map: Dict[str, str], country_map: Dict[str, str],
             metric: str, window: str, alpha: float, hypothesis: str,
             iters: int, seed: int) -> dict:
    """Fetch-free A/B test. `prices` is {ticker: history}. Returns a results dict."""
    spec = METRICS[metric]
    fn, alt = spec["fn"], HYPOTHESES[hypothesis]

    def collect(tickers):
        recs, no_price = [], []
        for t in tickers:
            hist = prices.get(t)
            if hist is None or hist.empty or len(hist) < MIN_BARS:
                no_price.append(t)
                val = np.nan
            else:
                val = fn(hist["Close"], window)
            recs.append({
                "Ticker": t,
                "Company": name_map.get(t, t),
                "Country": country_map.get(t, "—"),
                "Value": val,
            })
        df = pd.DataFrame(recs)
        valid = df.dropna(subset=["Value"]).reset_index(drop=True)
        nan_metric = [r.Ticker for r in df.itertuples()
                      if pd.isna(r.Value) and r.Ticker not in no_price]
        return df, valid, no_price, nan_metric

    df_a, valid_a, noprice_a, nan_a = collect(a_tickers)
    df_b, valid_b, noprice_b, nan_b = collect(b_tickers)

    a = valid_a["Value"].to_numpy(dtype=float)
    b = valid_b["Value"].to_numpy(dtype=float)
    na, nb = len(a), len(b)

    res: dict = {
        "metric": metric, "unit": spec["unit"], "better": spec["better"],
        "uses_window": spec["uses_window"], "window": window, "alpha": alpha,
        "hypothesis": hypothesis, "iters": iters,
        "name_a": None, "name_b": None,  # filled by caller
        "valid_a": valid_a, "valid_b": valid_b,
        "noprice_a": noprice_a, "noprice_b": noprice_b,
        "nan_a": nan_a, "nan_b": nan_b,
        "na": na, "nb": nb, "a": a, "b": b,
        "ok": na >= 2 and nb >= 2,
    }
    if not res["ok"]:
        return res

    delta = float(a.mean() - b.mean())
    res.update({
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "sd_a": float(a.std(ddof=1)), "sd_b": float(b.std(ddof=1)),
        "delta": delta,
    })

    # Welch's t-test + parametric CI for the difference.
    try:
        t_stat, welch_p = sstats.ttest_ind(a, b, equal_var=False, alternative=alt)
        t_stat, welch_p = float(t_stat), float(welch_p)
    except Exception:
        t_stat, welch_p = np.nan, np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se > 0 and na > 1 and nb > 1:
        df_w = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        tcrit = float(sstats.t.ppf(1 - alpha / 2, df_w))
        ci_w = (delta - tcrit * se, delta + tcrit * se)
    else:
        df_w, ci_w = np.nan, (delta, delta)

    # Mann-Whitney U (rank-based).
    try:
        u_stat, mw_p = sstats.mannwhitneyu(a, b, alternative=alt)
        u_stat, mw_p = float(u_stat), float(mw_p)
    except Exception:
        u_stat, mw_p = np.nan, np.nan

    # Permutation test + bootstrap CI.
    _, perm_p, perm_diffs = permutation_test(a, b, alt, iters, seed)
    boot_diffs, ci_b_lo, ci_b_hi = bootstrap_diff(a, b, iters, seed, alpha)

    sh_a, sh_b = _shapiro_p(a), _shapiro_p(b)
    small = min(na, nb) < 8
    nonnormal = (np.isfinite(sh_a) and sh_a < 0.05) or (np.isfinite(sh_b) and sh_b < 0.05)
    primary = "Permutation test" if (small or nonnormal) else "Welch's t-test"
    primary_p = perm_p if primary == "Permutation test" else welch_p

    res.update({
        "t_stat": t_stat, "welch_p": welch_p, "df_w": df_w, "ci_w": ci_w,
        "u_stat": u_stat, "mw_p": mw_p,
        "perm_p": perm_p, "perm_diffs": perm_diffs,
        "boot_diffs": boot_diffs, "ci_b": (ci_b_lo, ci_b_hi),
        "hedges_g": hedges_g(a, b), "cliffs_delta": cliffs_delta(a, b),
        "cles": common_language_es(a, b),
        "shapiro_a": sh_a, "shapiro_b": sh_b,
        "primary": primary, "primary_p": primary_p,
        "significant": np.isfinite(primary_p) and primary_p < alpha,
    })
    return res


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def fmt(x: float, dec: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{dec}f}"


def fmt_signed(x: float, unit: str = "", dec: int = 2) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:+.{dec}f}{unit}"


def fmt_level(x: float, unit: str = "", dec: int = 2) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{dec}f}{unit}"


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def interp_g(g: float) -> str:
    if not np.isfinite(g):
        return "n/a"
    a = abs(g)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def interp_cliff(d: float) -> str:
    if not np.isfinite(d):
        return "n/a"
    a = abs(d)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


# --------------------------------------------------------------------------- #
# Data download (cached) — the only place that touches the network
# --------------------------------------------------------------------------- #

def _extract_history(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        hist = raw[ticker].copy()
    else:
        hist = raw.copy()
    hist.columns = [str(c).capitalize() for c in hist.columns]
    if "Close" not in hist.columns:
        return pd.DataFrame()
    return hist.dropna(subset=["Close"])


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def download_prices(tickers: tuple, period: str) -> Tuple[Dict[str, pd.DataFrame], str]:
    """Batch-download price history. Returns ({ticker: history}, error_message)."""
    if not tickers:
        return {}, ""
    try:
        raw = yf.download(
            list(tickers), period=period, auto_adjust=True,
            progress=False, group_by="ticker", threads=True,
        )
    except Exception as exc:  # network / API failure
        return {}, str(exc)

    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        hist = _extract_history(raw, t)
        if not hist.empty:
            out[t] = hist
    return out, ""


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def _base_ax(figsize=(7.2, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.title.set_color(COL_INK)
    ax.xaxis.label.set_color("#475569")
    ax.yaxis.label.set_color("#475569")
    return fig, ax


def chart_per_ticker(res: dict):
    unit = res["unit"]
    va = res["valid_a"].assign(Group="A")
    vb = res["valid_b"].assign(Group="B")
    combined = pd.concat([va, vb]).sort_values("Value").reset_index(drop=True)

    fig, ax = _base_ax(figsize=(7.4, max(3.0, 0.34 * len(combined) + 1.0)))
    colors = [COL_A if g == "A" else COL_B for g in combined["Group"]]
    y = np.arange(len(combined))
    ax.barh(y, combined["Value"], color=colors, alpha=0.9, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(combined["Ticker"], fontsize=8)
    ax.axvline(0, color="#e2e8f0", lw=1)
    ax.axvline(res["mean_a"], color=COL_A, ls="--", lw=1.4, alpha=0.9)
    ax.axvline(res["mean_b"], color=COL_B, ls="--", lw=1.4, alpha=0.9)
    ax.set_xlabel(f"{res['metric']} ({unit})" if unit else res["metric"])
    ax.set_title("Per-ticker values (dashed lines = group means)", fontsize=11, pad=10)
    ax.grid(axis="x", color="#f1f5f9", lw=1)
    ax.set_axisbelow(True)
    legend = [Patch(color=COL_A, label=res["name_a"]), Patch(color=COL_B, label=res["name_b"])]
    ax.legend(handles=legend, frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def chart_box_strip(res: dict):
    a, b = res["a"], res["b"]
    unit = res["unit"]
    fig, ax = _base_ax(figsize=(6.4, 4.3))

    bp = ax.boxplot([a, b], positions=[1, 2], widths=0.52, showfliers=False,
                    patch_artist=True, medianprops=dict(color=COL_INK, lw=1.6),
                    whiskerprops=dict(color="#94a3b8"), capprops=dict(color="#94a3b8"))
    for patch, col in zip(bp["boxes"], (COL_A, COL_B)):
        patch.set_facecolor(col)
        patch.set_alpha(0.16)
        patch.set_edgecolor(col)

    rng = np.random.default_rng(1)
    for vals, x, col in ((a, 1, COL_A), (b, 2, COL_B)):
        jitter = rng.normal(0, 0.05, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, color=col, alpha=0.8,
                   s=34, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter([1, 2], [a.mean(), b.mean()], marker="D", color=COL_INK,
               s=46, zorder=4, label="mean")

    ax.set_xticks([1, 2])
    ax.set_xticklabels([res["name_a"], res["name_b"]], fontsize=10)
    ax.set_ylabel(f"{res['metric']} ({unit})" if unit else res["metric"])
    ax.set_title("Distribution by group", fontsize=11, pad=10)
    ax.grid(axis="y", color="#f1f5f9", lw=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def chart_bootstrap(res: dict):
    diffs = res["boot_diffs"]
    lo, hi = res["ci_b"]
    unit = res["unit"]
    fig, ax = _base_ax(figsize=(7.2, 3.9))
    ax.hist(diffs, bins=42, color="#6366f1", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvspan(lo, hi, color="#6366f1", alpha=0.12)
    ax.axvline(0, color="#334155", ls="--", lw=1.4)
    ax.axvline(res["delta"], color=COL_INK, lw=1.8)
    conf = int(round((1 - res["alpha"]) * 100))
    ax.set_title(f"Bootstrap difference in means (A − B), {conf}% CI shaded", fontsize=11, pad=10)
    ax.set_xlabel(f"difference ({unit})" if unit else "difference")
    ax.set_ylabel("resamples")
    ax.grid(axis="y", color="#f1f5f9", lw=1)
    ax.set_axisbelow(True)
    ax.annotate(f"{conf}% CI  [{lo:+.2f}, {hi:+.2f}]{unit}", xy=(0.02, 0.92),
                xycoords="axes fraction", fontsize=9, color=COL_MUTED)
    fig.tight_layout()
    return fig


def chart_permutation(res: dict):
    diffs = res["perm_diffs"]
    obs = res["delta"]
    unit = res["unit"]
    alt = HYPOTHESES[res["hypothesis"]]
    fig, ax = _base_ax(figsize=(7.2, 3.9))
    ax.hist(diffs, bins=42, color="#cbd5e1", alpha=0.85, edgecolor="white", linewidth=0.4)

    lo_x, hi_x = ax.get_xlim()
    if alt == "two-sided":
        ax.axvspan(lo_x, -abs(obs), color="#fca5a5", alpha=0.18)
        ax.axvspan(abs(obs), hi_x, color="#fca5a5", alpha=0.18)
    elif alt == "greater":
        ax.axvspan(obs, hi_x, color="#fca5a5", alpha=0.18)
    else:
        ax.axvspan(lo_x, obs, color="#fca5a5", alpha=0.18)
    ax.set_xlim(lo_x, hi_x)

    ax.axvline(obs, color=COL_A, lw=2.0)
    ax.set_title("Permutation null vs observed difference (shaded = p-value area)", fontsize=11, pad=10)
    ax.set_xlabel(f"difference under H₀ ({unit})" if unit else "difference under H₀")
    ax.set_ylabel("permutations")
    ax.grid(axis="y", color="#f1f5f9", lw=1)
    ax.set_axisbelow(True)
    ax.annotate(f"observed {obs:+.2f}{unit}  ·  p = {fmt_p(res['perm_p'])}",
                xy=(0.02, 0.92), xycoords="axes fraction", fontsize=9, color=COL_MUTED)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
  --bg:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e5e7eb;
  --panel:#f8fafc; --A:#4f46e5; --B:#d97706;
}
html, body, .stApp,
[data-testid="stAppViewContainer"], [data-testid="stHeader"]{
  background: var(--bg) !important;
}
[data-testid="stHeader"]{ border-bottom:1px solid var(--line); }
[data-testid="stSidebar"]{ background:#fbfcfe !important; border-right:1px solid var(--line); }
.block-container{ padding-top:2.0rem; padding-bottom:3rem; max-width:1180px; }

h1,h2,h3,h4{ font-family:'Space Grotesk',sans-serif; color:var(--ink); letter-spacing:-.01em; }
body, p, div, span, label, li, .stMarkdown, [data-testid="stWidgetLabel"]{
  font-family:'Inter',sans-serif; color:var(--ink);
}

.hero-title{ font-family:'Space Grotesk'; font-size:2.05rem; font-weight:700; margin:0 0 .2rem; }
.hero-sub{ color:var(--muted); font-size:1.02rem; margin:0; max-width:74ch; line-height:1.5; }

.ghead{ display:flex; align-items:center; gap:.5rem; font-family:'Space Grotesk';
  font-weight:600; font-size:1.08rem; margin:.2rem 0 .3rem; }
.dot{ width:.72rem; height:.72rem; border-radius:50%; display:inline-block; }
.dotA{ background:var(--A); } .dotB{ background:var(--B); }

[data-testid="stMetricValue"]{ font-family:'Space Grotesk'; font-variant-numeric:tabular-nums; }
[data-testid="stMetricLabel"]{ color:var(--muted); }

.verdict{ border:1px solid var(--line); border-radius:16px; padding:20px 22px; margin:.2rem 0 1.1rem; }
.verdict.sig{ border-color:#bbf7d0; background:#f0fdf4; }
.verdict.null{ border-color:#e2e8f0; background:#f8fafc; }
.verdict h3{ margin:.15rem 0 .4rem; font-size:1.3rem; }
.verdict .sub{ color:#334155; font-size:.96rem; line-height:1.5; }
.verdict .note{ color:var(--muted); font-size:.9rem; margin-top:.5rem; }

.tag{ display:inline-block; padding:.14rem .55rem; border-radius:999px;
  font-size:.72rem; font-weight:600; font-family:'Inter'; }
.tagA{ background:#eef2ff; color:#4338ca; } .tagB{ background:#fffbeb; color:#b45309; }
.tag-sig{ background:#dcfce7; color:#166534; } .tag-null{ background:#e2e8f0; color:#475569; }

.cfg{ color:var(--muted); font-size:.86rem; line-height:1.7; }
.cfg b{ color:var(--ink); font-weight:600; }

hr{ border:none; border-top:1px solid var(--line); margin:1.1rem 0; }
"""


def hero():
    st.markdown(
        '<div class="hero-title">Equity A/B Testing</div>'
        '<p class="hero-sub">Define two baskets of stocks, pick a performance metric, '
        'and test whether the difference between them is real or just noise — with '
        'permutation, Welch and Mann-Whitney tests, effect sizes, and bootstrap '
        'confidence intervals.</p>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Rendering the results
# --------------------------------------------------------------------------- #

def _verdict_block(res: dict):
    unit, delta = res["unit"], res["delta"]
    higher = res["name_a"] if delta > 0 else res["name_b"]
    lower = res["name_b"] if delta > 0 else res["name_a"]
    sig = res["significant"]
    cls = "sig" if sig else "null"
    tag = ('<span class="tag tag-sig">Significant at α = %.2f</span>' % res["alpha"]) if sig \
        else '<span class="tag tag-null">Not significant</span>'

    if abs(delta) < 1e-9:
        head = "The two baskets are effectively identical"
    elif sig:
        head = f"{higher} has a higher {res['metric'].lower()}"
    else:
        head = "No statistically significant difference"

    sub = (f"Difference (A − B): <b>{fmt_signed(delta, unit)}</b> &nbsp;·&nbsp; "
           f"{res['primary']} p = <b>{fmt_p(res['primary_p'])}</b> &nbsp;·&nbsp; "
           f"Hedges' g <b>{fmt(res['hedges_g'])}</b> ({interp_g(res['hedges_g'])} effect)")

    note = ""
    if sig and res["better"] == "low":
        note = (f'<div class="note">Lower is generally preferable on this metric, '
                f'so {lower} looks stronger here.</div>')

    st.markdown(f'<div class="verdict {cls}">{tag}<h3>{head}</h3>'
                f'<div class="sub">{sub}</div>{note}</div>', unsafe_allow_html=True)


def _config_strip(res: dict):
    win = res["window"] if res["uses_window"] else "full history"
    st.markdown(
        f'<div class="cfg">Metric <b>{res["metric"]}</b> &nbsp;·&nbsp; '
        f'Window <b>{win}</b> &nbsp;·&nbsp; α <b>{res["alpha"]:.2f}</b> &nbsp;·&nbsp; '
        f'Hypothesis <b>{res["hypothesis"]}</b> &nbsp;·&nbsp; '
        f'n <b>{res["na"]}</b> vs <b>{res["nb"]}</b> &nbsp;·&nbsp; '
        f'resamples <b>{res["iters"]:,}</b></div>',
        unsafe_allow_html=True,
    )


def _tab_statistics(res: dict):
    unit, alpha = res["unit"], res["alpha"]
    conf = int(round((1 - alpha) * 100))

    def reject(p):
        return "n/a" if not np.isfinite(p) else ("Yes" if p < alpha else "No")

    tests = pd.DataFrame([
        {"Test": f"Permutation ({res['iters']:,} resamples)",
         "Statistic": f"mean diff = {fmt_signed(res['delta'], unit)}",
         "p-value": fmt_p(res["perm_p"]), f"Reject H₀ (α={alpha:g})": reject(res["perm_p"])},
        {"Test": "Welch's t-test",
         "Statistic": f"t = {fmt(res['t_stat'])}, df ≈ {fmt(res['df_w'], 1)}",
         "p-value": fmt_p(res["welch_p"]), f"Reject H₀ (α={alpha:g})": reject(res["welch_p"])},
        {"Test": "Mann–Whitney U",
         "Statistic": f"U = {fmt(res['u_stat'], 1)}",
         "p-value": fmt_p(res["mw_p"]), f"Reject H₀ (α={alpha:g})": reject(res["mw_p"])},
    ])
    st.markdown("**Hypothesis tests**")
    st.dataframe(tests, hide_index=True, use_container_width=True)
    st.caption(f"Primary test for these samples: **{res['primary']}** "
               f"(chosen from sample size and the normality check below).")

    ci_w_lo, ci_w_hi = res["ci_w"]
    ci_b_lo, ci_b_hi = res["ci_b"]
    effects = pd.DataFrame([
        {"Measure": "Hedges' g", "Value": fmt(res["hedges_g"]),
         "Reading": f"{interp_g(res['hedges_g'])} standardized effect"},
        {"Measure": "Cliff's delta", "Value": fmt(res["cliffs_delta"]),
         "Reading": f"{interp_cliff(res['cliffs_delta'])} (rank-based)"},
        {"Measure": "P(random A > random B)", "Value": f"{res['cles'] * 100:.1f}%",
         "Reading": "common-language effect size"},
        {"Measure": f"{conf}% CI (Welch)", "Value": f"[{ci_w_lo:+.2f}, {ci_w_hi:+.2f}]{unit}",
         "Reading": "parametric interval for A − B"},
        {"Measure": f"{conf}% CI (bootstrap)", "Value": f"[{ci_b_lo:+.2f}, {ci_b_hi:+.2f}]{unit}",
         "Reading": "resampling interval for A − B"},
    ])
    st.markdown("**Effect size & confidence intervals**")
    st.dataframe(effects, hide_index=True, use_container_width=True)

    def normal_label(p):
        if not np.isfinite(p):
            return "too few points (n < 3)"
        return "consistent with normal" if p >= 0.05 else "departs from normal"

    norm = pd.DataFrame([
        {"Group": res["name_a"], "Shapiro–Wilk p": fmt_p(res["shapiro_a"]),
         "Reading": normal_label(res["shapiro_a"])},
        {"Group": res["name_b"], "Shapiro–Wilk p": fmt_p(res["shapiro_b"]),
         "Reading": normal_label(res["shapiro_b"])},
    ])
    st.markdown("**Normality check**")
    st.dataframe(norm, hide_index=True, use_container_width=True)


def _tab_charts(res: dict):
    for maker in (chart_per_ticker, chart_box_strip, chart_bootstrap, chart_permutation):
        fig = maker(res)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def _tab_data(res: dict):
    unit = res["unit"]

    def show(name, valid, noprice, nan_metric, color_tag):
        st.markdown(f'<div class="ghead"><span class="dot dot{color_tag}"></span>{name}</div>',
                    unsafe_allow_html=True)
        if not valid.empty:
            disp = valid.copy()
            disp["Value"] = disp["Value"].round(2)
            disp = disp.rename(columns={"Value": f"{res['metric']} ({unit})" if unit else res["metric"]})
            st.dataframe(disp, hide_index=True, use_container_width=True)
        skipped = []
        if noprice:
            skipped.append("no price history: " + ", ".join(sorted(noprice)))
        if nan_metric:
            skipped.append("metric not computable: " + ", ".join(sorted(nan_metric)))
        if skipped:
            st.caption("Skipped — " + " · ".join(skipped))

    c1, c2 = st.columns(2)
    with c1:
        show(res["name_a"], res["valid_a"], res["noprice_a"], res["nan_a"], "A")
    with c2:
        show(res["name_b"], res["valid_b"], res["noprice_b"], res["nan_b"], "B")

    export = pd.concat([
        res["valid_a"].assign(Group=res["name_a"]),
        res["valid_b"].assign(Group=res["name_b"]),
    ])[["Group", "Ticker", "Company", "Country", "Value"]]
    export["Value"] = export["Value"].round(4)
    st.download_button(
        "Download results (CSV)",
        export.to_csv(index=False).encode("utf-8"),
        file_name="ab_test_results.csv", mime="text/csv",
    )


def _tab_method(res: dict):
    st.markdown(
        f"""
**What is being tested.** Each stock is one observation. Its value is the chosen
metric — here **{res['metric']}** — measured over the selected window. The test asks
whether basket A and basket B differ on that metric by more than sampling noise would
explain.

**The three tests**
- **Permutation test** — shuffles the group labels many times to build the distribution
  of the mean difference under "no real difference", then sees how extreme the observed
  gap is. Makes almost no assumptions and is the most reliable choice for small baskets.
- **Welch's t-test** — compares means without assuming equal variances. Trustworthy when
  each basket is reasonably large and roughly normal.
- **Mann–Whitney U** — compares ranks rather than means, so a single outlier can't
  dominate. A useful cross-check.

**Effect size** answers "how big", separately from "is it significant". Hedges' g is the
standardized mean gap (bias-corrected for small samples); Cliff's delta and
P(random A > random B) are rank-based and outlier-robust.

**Confidence intervals** for the difference (A − B) are shown two ways: a parametric
Welch interval and a percentile bootstrap. If the interval excludes 0, the difference is
significant at that level.

**Small samples.** With only a handful of tickers per side, lean on the permutation and
Mann–Whitney results and read the t-test cautiously. Averaging across very different
basket sizes is fragile — the app flags large imbalances.

*Data via Yahoo Finance (`yfinance`), cached for {CACHE_TTL // 60} minutes. Not investment advice.*
"""
    )


PLAIN_METRIC = {
    "Total Return": "how much money you'd have made or lost by holding the stock over this period",
    "Annualized Volatility": "how much the stock's price jumps around day to day — a rougher ride, "
                             "whether it's making or losing money",
    "Sharpe (Return/Risk)": "how much reward the stock gave you for each unit of bumpiness it put "
                            "you through — a way to ask \"was the ride worth it\"",
    "vs 200-day MA": "whether the stock is trading above or below its own typical price over "
                     "roughly the last year",
    "Max Drawdown": "the worst drop the stock took from a peak to its lowest point — \"at its "
                    "worst, how much would you have been down\"",
}


def _plain_chance(p: float) -> str:
    """Turn a p-value into a '~X times out of 100' phrase a non-statistician can picture."""
    if not np.isfinite(p):
        return "couldn't be worked out from this data"
    pct = p * 100
    if pct < 1:
        return "less than 1 time out of 100"
    return f"about {pct:.0f} times out of 100"


def _tab_plain(res: dict):
    unit = res["unit"]
    name_a, name_b = res["name_a"], res["name_b"]
    metric = res["metric"]
    sig = res["significant"]
    delta = res["delta"]
    higher = name_a if delta > 0 else name_b
    lower = name_b if delta > 0 else name_a
    verb = "more" if delta > 0 else "less"
    plain_desc = PLAIN_METRIC.get(metric, metric.lower())
    chance_phrase = _plain_chance(res["primary_p"])
    cles_pct = res["cles"] * 100

    st.markdown(
        "A/B testing just means: split things into two groups, measure something about each "
        "group, then ask *\"is the difference we see real, or could it just be luck?\"* Here, "
        "the two groups are baskets of stocks instead of, say, two versions of a website button."
    )

    cls = "sig" if sig else "null"
    if abs(delta) < 1e-9:
        headline = f"{name_a} and {name_b} came out basically tied"
        sub = f"On average the two groups scored almost the same on {metric.lower()}."
    elif sig:
        headline = f"Yes — {higher} really does look different here"
        sub = (f"On average, {name_a} scored {fmt_signed(delta, unit)} {verb} than {name_b} on "
               f"{metric.lower()}. That gap is big enough that it probably isn't just luck.")
    else:
        headline = "We can't be confident there's a real difference"
        sub = (f"On average, {name_a} scored {fmt_signed(delta, unit)} {verb} than {name_b} on "
               f"{metric.lower()} — but a gap that size could easily happen just by chance.")
    st.markdown(f'<div class="verdict {cls}"><h3>{headline}</h3><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)

    st.markdown(f"""
**What did we actually compare?**

{metric} measures {plain_desc}. We scored every stock in **{name_a}** ({res['na']} stocks) and
every stock in **{name_b}** ({res['nb']} stocks) on that, then compared the two groups.

**Could this just be luck?**

Imagine flipping two coins a bunch of times and comparing how often each lands heads. Even if the
coins are identical, you'd still see small differences between them now and then, purely by chance.
A statistical test asks: *if there were really no difference between the groups, how often would a
gap this big turn up anyway, just from randomness?*

Here, the answer is **{chance_phrase}**. {"That's rare enough that we call it a real, meaningful difference — not a fluke." if sig else "That's common enough that we can't rule out plain chance, so we call this result *not statistically significant*."}

**How often would one side actually come out ahead?**

If you picked one random stock from {name_a} and one random stock from {name_b}, {name_a} would
come out ahead on {metric.lower()} about **{cles_pct:.0f}% of the time** (a tie counts as a coin flip).
""")

    if res["better"] == "low":
        st.info(f"For {metric}, a *lower* number is usually seen as better (less risk of loss), "
                "so keep that in mind when reading \"ahead\" above.")

    if min(res["na"], res["nb"]) < 5:
        st.warning("Heads up: this test only had a handful of stocks per side. With so few "
                   "examples, the result is fragile — swapping in one or two different stocks "
                   "could change the picture a lot. Treat this as a first impression, not a "
                   "final verdict.")

    with st.expander("A few honest caveats"):
        st.markdown("""
- **Past performance isn't a promise.** Everything here is based on history. It doesn't tell you
  what either group of stocks will do next.
- **"Significant" doesn't mean "big."** With enough data, even a tiny, real difference can be
  statistically significant. Always look at the actual size of the gap (shown above), not just
  whether the test called it significant.
- **This isn't investment advice.** It's a way to compare how groups of stocks have behaved —
  not a recommendation to buy, sell, or hold anything.
""")

    with st.expander("What do the technical words in the other tabs mean?"):
        st.markdown("""
- **p-value** — the chance of seeing a gap this big purely by luck, if there were really no
  difference. Smaller means more surprising, which means more likely to be a real difference.
- **Statistically significant** — the gap was surprising enough (below the threshold you set)
  that it's probably not just luck.
- **Confidence interval** — a range that most likely contains the *true* difference, rather than
  just the one number we happened to measure from this sample of stocks.
- **Effect size (Hedges' g, Cliff's delta)** — how *big* the difference is, on a scale that
  doesn't depend on how many stocks were tested.
- **Volatility / standard deviation** — how spread out or jumpy the numbers are.
""")


def render_results(res: dict):
    if not res["ok"]:
        st.error("Each basket needs at least 2 tickers with usable price history. "
                 "Add more tickers or check the symbols, then run the test again.")
        # Still surface which tickers failed, to help fix the input.
        for name, noprice in ((res["name_a"], res["noprice_a"]), (res["name_b"], res["noprice_b"])):
            if noprice:
                st.caption(f"{name}: no price history for {', '.join(sorted(noprice))}")
        return

    _config_strip(res)
    _verdict_block(res)

    unit = res["unit"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(f"{res['name_a']} mean", fmt_level(res["mean_a"], unit))
        st.caption(f"median {fmt_level(res['median_a'], unit)} · sd {fmt(res['sd_a'])}")
    with c2:
        st.metric(f"{res['name_b']} mean", fmt_level(res["mean_b"], unit))
        st.caption(f"median {fmt_level(res['median_b'], unit)} · sd {fmt(res['sd_b'])}")
    with c3:
        st.metric("Difference (A − B)", fmt_signed(res["delta"], unit))
        st.caption("positive favours A")
    with c4:
        st.metric(f"{res['primary']} p-value", fmt_p(res["primary_p"]))
        st.caption(f"vs α = {res['alpha']:.2f}")

    if min(res["na"], res["nb"]) < 5:
        st.warning(f"Small samples ({res['na']} vs {res['nb']}). Prefer the permutation and "
                   "Mann–Whitney results; treat the t-test with caution.")
    lo, hi = sorted((res["na"], res["nb"]))
    if lo and hi / lo >= 2:
        st.info(f"Unbalanced baskets ({res['na']} vs {res['nb']}). You're comparing averages "
                "over different-sized groups — read them with care.")

    tabs = st.tabs(["Statistics", "Charts", "Per-ticker data", "Method", "Plain English"])
    with tabs[0]:
        _tab_statistics(res)
    with tabs[1]:
        _tab_charts(res)
    with tabs[2]:
        _tab_data(res)
    with tabs[3]:
        _tab_method(res)
    with tabs[4]:
        _tab_plain(res)


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #

def parse_custom(text: str) -> List[str]:
    if not text:
        return []
    out = []
    for token in re.split(r"[\s,;]+", text.strip()):
        token = token.strip().upper()
        if token:
            out.append(token)
    return out


def resolve_group(individual, pack_names, custom_text, packs) -> List[str]:
    ordered = list(individual)
    for pack in pack_names:
        ordered += packs.get(pack, [])
    ordered += parse_custom(custom_text)
    seen, result = set(), []
    for t in ordered:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def group_controls(letter: str, catalog, defaults, default_name):
    name_map, _, packs, all_tickers = catalog
    color = "A" if letter == "A" else "B"
    st.markdown(f'<div class="ghead"><span class="dot dot{color}"></span>Group {letter}</div>',
                unsafe_allow_html=True)
    name = st.text_input("Name", value=default_name, key=f"name_{letter}")
    tickers = st.multiselect(
        "Tickers", options=all_tickers, default=defaults, key=f"tk_{letter}",
        format_func=lambda t: f"{t} — {name_map.get(t, t)}",
    )
    with st.expander("Add more"):
        pack_names = st.multiselect(
            "Whole ecosystems", options=list(packs), default=[], key=f"pk_{letter}",
            help="Adds every ticker in the selected ecosystem(s).",
        )
        custom = st.text_input(
            "Custom tickers", value="", key=f"cx_{letter}",
            placeholder="e.g. TSLA, BABA, 7203.T",
            help="Any Yahoo Finance symbol. Separate with commas or spaces.",
        )
    resolved = resolve_group(tickers, pack_names, custom, packs)
    st.caption(f"{len(resolved)} tickers: {', '.join(resolved) if resolved else '—'}")
    return name.strip() or f"Group {letter}", resolved


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    st.set_page_config(page_title="Equity A/B Testing", page_icon="🧪", layout="wide")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

    catalog = build_catalog(ECOSYSTEMS)

    hero()

    with st.sidebar:
        st.markdown("### Test settings")
        metric = st.selectbox("Metric", list(METRICS), index=0,
                              help="What each stock is scored on.")
        st.caption(METRICS[metric]["help"])
        window = st.select_slider("Window", options=list(TRADING_DAYS), value="1y",
                                  disabled=not METRICS[metric]["uses_window"])
        if not METRICS[metric]["uses_window"]:
            st.caption("This metric uses full history, so the window is ignored.")
        hypothesis = st.radio("Hypothesis", list(HYPOTHESES), index=0)
        alpha = st.select_slider("Significance level α", options=[0.10, 0.05, 0.01], value=0.05)

        with st.expander("Advanced"):
            iters = st.slider("Permutation / bootstrap resamples", 2000, 50000, 10000, step=2000)
            seed = st.number_input("Random seed", value=42, step=1)

        st.markdown("---")
        if st.button("Clear cached prices", use_container_width=True):
            download_prices.clear()
            st.success("Cache cleared — the next run re-downloads prices.")

    col_a, col_b = st.columns(2)
    with col_a:
        name_a, tickers_a = group_controls("A", catalog, DEFAULT_A, "AI Hardware")
    with col_b:
        name_b, tickers_b = group_controls("B", catalog, DEFAULT_B, "AI Software")

    overlap = sorted(set(tickers_a) & set(tickers_b))
    if overlap:
        st.warning(f"These tickers are in both baskets: {', '.join(overlap)}. "
                   "A/B groups should be independent for the test to be meaningful.")

    run = st.button("Run A/B test", type="primary", use_container_width=True)

    signature = (tuple(tickers_a), tuple(tickers_b), name_a, name_b,
                 metric, window, float(alpha), hypothesis, int(iters), int(seed))

    if run:
        if len(tickers_a) < 2 or len(tickers_b) < 2:
            st.error("Each basket needs at least 2 tickers. Add more, then run again.")
        else:
            union = tuple(sorted(set(tickers_a) | set(tickers_b)))
            with st.spinner("Downloading price history…"):
                prices, err = download_prices(union, HISTORY_PERIOD)
            if err:
                st.error(f"Could not download prices: {err}")
            else:
                res = run_test(tickers_a, tickers_b, prices,
                               catalog[0], catalog[1],
                               metric, window, float(alpha), hypothesis,
                               int(iters), int(seed))
                res["name_a"], res["name_b"] = name_a, name_b
                st.session_state["results"] = res
                st.session_state["signature"] = signature

    st.markdown("---")

    if "results" in st.session_state:
        if st.session_state.get("signature") != signature:
            st.info("Inputs changed since the last run. Press **Run A/B test** to refresh.")
        render_results(st.session_state["results"])
    else:
        st.markdown(
            '<div class="verdict null"><h3>No test run yet</h3>'
            '<div class="sub">Set up two baskets above, choose a metric in the sidebar, '
            'then run the test. Defaults compare an AI-hardware basket against AI-software.</div>'
            '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
