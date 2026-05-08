"""
IV Surface — US Index Options  (OOP Architecture)
==================================================
Architecture:
  DataSource (ABC)
    ├── CBOELiveSource        — ข้อมูลปัจจุบัน  (ฟรี ไม่ต้อง Key)
    └── NasdaqHistoricalSource — ข้อมูลย้อนหลัง (ต้องมี API Key ฟรี)

  IVSurfaceEngine             — คำนวณ IV + build surface (shared model)
  IVSurfacePlotter            — สร้าง Plotly chart (shared visualization)

  LiveMode                    — UI mode ปัจจุบัน
  HistoricalMode              — UI mode ย้อนหลัง

รัน:    streamlit run iv_surface_app.py
ติดตั้ง: pip install streamlit numpy pandas scipy plotly requests nasdaq-data-link
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

# ── stdlib ────────────────────────────────────
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional
import re

# ── 3rd party ─────────────────────────────────
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata, interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st


# ════════════════════════════════════════════════════════════════
# DOMAIN MODELS
# ════════════════════════════════════════════════════════════════

@dataclass
class OptionQuote:
    """single option contract"""
    expiry:     str
    strike:     float
    opt_type:   str       # "call" | "put"
    bid:        float
    ask:        float
    volume:     int
    iv:         float     # 0–5 scale (0.20 = 20%)
    T:          float     # years to expiry
    days:       int
    moneyness:  float     # log(K/S)


@dataclass
class SurfaceData:
    """output of IVSurfaceEngine"""
    k_grid:     np.ndarray
    t_grid:     np.ndarray
    iv_surface: np.ndarray
    df_clean:   pd.DataFrame
    spot:       float
    symbol:     str
    as_of:      str        # date string


# ════════════════════════════════════════════════════════════════
# DATA SOURCES  (Strategy Pattern)
# ════════════════════════════════════════════════════════════════

class DataSource(ABC):
    """Abstract base: ทุก data source ต้อง implement fetch_quotes()"""

    @abstractmethod
    def fetch_quotes(self, symbol: str, **kwargs) -> tuple[pd.DataFrame, float]:
        """คืน (DataFrame of raw quotes, spot price)"""
        ...

    @abstractmethod
    def get_available_dates(self, symbol: str, **kwargs) -> list[str]:
        """คืน list ของ expiry / trading dates ที่ดึงได้"""
        ...


# ── CBOE Live Source ──────────────────────────
class CBOELiveSource(DataSource):
    """
    ดึง options chain ปัจจุบันจาก CBOE CDN (delayed 15 min)
    ฟรี ไม่ต้อง Key ใดๆ
    Endpoint: cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json
    """

    BASE = "https://cdn.cboe.com/api/global/delayed_quotes/options"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer":    "https://www.cboe.com/",
        "Accept":     "application/json",
    }

    # ETF ใช้ชื่อตรง, Index ใช้ underscore นำหน้า
    SYMBOL_MAP = {
        "SPY": "SPY",  "QQQ": "QQQ",  "IWM": "IWM",  "DIA": "DIA",
        "SPX": "_SPX", "NDX": "_NDX", "RUT": "_RUT",
    }

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_quotes(_self, symbol: str, **kwargs) -> tuple[pd.DataFrame, float]:
        sym = _self.SYMBOL_MAP.get(symbol, symbol)
        url = f"{_self.BASE}/{sym}.json"
        r   = requests.get(url, headers=_self.HEADERS, timeout=15)
        r.raise_for_status()
        data  = r.json().get("data", {})
        spot  = float(data.get("current_price", 0))
        opts  = data.get("options", [])
        if not opts:
            return pd.DataFrame(), spot
        return pd.DataFrame(opts), spot

    def get_available_dates(self, symbol: str, **kwargs) -> list[str]:
        """CBOE ไม่มี historical — คืนแค่วันนี้"""
        return [datetime.now().strftime("%Y-%m-%d")]


# ── Nasdaq Data Link Historical Source ────────
class NasdaqHistoricalSource(DataSource):
    """
    ดึงข้อมูล options chain ย้อนหลังจาก Nasdaq Data Link
    Dataset: QDL/OPT — US Equity & Index Options (end-of-day)
    ย้อนหลังได้ถึง: 2005 (ขึ้นกับ subscription tier)
    API Key: ฟรีที่ data.nasdaq.com (free tier = 50 calls/day)
              ข้อมูล options ต้องมี subscription ($29/เดือน)

    Fields ที่ได้:
      date, symbol, expiration, strike, call_put,
      bid, ask, volume, open_interest, implied_volatility,
      delta, gamma, theta, vega
    """

    DATASET = "QDL/OPT"

    def __init__(self, api_key: str):
        self.api_key = api_key
        import nasdaqdatalink
        nasdaqdatalink.ApiConfig.api_key = api_key
        self._ndl = nasdaqdatalink

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_quotes(_self, symbol: str, as_of_date: str = "", **kwargs) -> tuple[pd.DataFrame, float]:
        """
        ดึง options chain ของวันที่กำหนด
        as_of_date: "YYYY-MM-DD"  ถ้าไม่ระบุ = ล่าสุด
        """
        target_date = as_of_date or datetime.now().strftime("%Y-%m-%d")

        try:
            df = _self._ndl.get_table(
                _self.DATASET,
                ticker  = symbol,
                date    = target_date,
                paginate= True,
            )
        except Exception as e:
            raise RuntimeError(f"Nasdaq Data Link error: {e}")

        if df.empty:
            return pd.DataFrame(), 0.0

        # คำนวณ spot จาก ATM options (ใช้ call+put parity หรือ mid strike ที่ IV ต่ำสุด)
        atm_row = df.loc[df["implied_volatility"].astype(float).idxmin()]
        spot    = float(atm_row.get("underlying_price", atm_row["strike"]))

        return df, spot

    def get_available_dates(self, symbol: str, months: int = 6, **kwargs) -> list[str]:
        """คืน trading dates ย้อนหลัง N เดือน (เฉพาะ weekdays)"""
        end   = date.today()
        start = end - timedelta(days=months * 30)
        dates = pd.bdate_range(start=start, end=end)
        return [d.strftime("%Y-%m-%d") for d in reversed(dates)]


# ════════════════════════════════════════════════════════════════
# IV SURFACE ENGINE  (shared model — ไม่ขึ้นกับ DataSource)
# ════════════════════════════════════════════════════════════════

class IVSurfaceEngine:
    """
    รับ raw DataFrame จาก DataSource
    → parse → filter OTM → build interpolated surface
    ไม่มี Streamlit code ใดๆ — testable แยกได้
    """

    GRID_NK       = 60
    GRID_NT       = 50
    RISK_FREE     = 0.053
    MONEYNESS_MIN = -0.50
    MONEYNESS_MAX = 0.40
    MAX_IV        = 5.0

    # ── Black-Scholes ─────────────────────────
    @staticmethod
    def _bs_price(S: float, K: float, T: float, r: float,
                  sigma: float, opt_type: str) -> float:
        if T <= 1e-6 or sigma <= 1e-6:
            return max(0.0, (S - K) if opt_type == "call" else (K - S))
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _calc_iv(self, price: float, S: float, K: float,
                 T: float, opt_type: str) -> float:
        r = self.RISK_FREE
        if price <= 0 or T <= 1e-6:
            return np.nan
        fwd       = S * np.exp(r * T)
        intrinsic = max(0.0, (fwd - K) if opt_type == "call" else (K - fwd))
        if price <= intrinsic * np.exp(-r * T) * 0.999:
            return np.nan
        def obj(sig): return self._bs_price(S, K, T, r, sig, opt_type) - price
        try:
            if obj(1e-4) * obj(self.MAX_IV) > 0:
                return np.nan
            iv = brentq(obj, 1e-4, self.MAX_IV, xtol=1e-7, maxiter=200)
            return iv if 0.005 <= iv <= self.MAX_IV else np.nan
        except Exception:
            return np.nan

    # ── Parse CBOE raw ────────────────────────
    def parse_cboe(self, df_raw: pd.DataFrame, spot: float) -> pd.DataFrame:
        df    = df_raw.copy()
        today = datetime.now()

        # store raw columns for debug
        st.session_state["cboe_columns"] = list(df.columns)

        # expiry
        exp_col = next((c for c in ["expiration","expiry","exp_date","ExpirationDate"]
                        if c in df.columns), None)
        if exp_col is None:
            sym_col = next((c for c in ["option","symbol","Symbol"] if c in df.columns), None)
            if sym_col:
                df["expiry"] = df[sym_col].apply(self._occ_expiry)
            else:
                return pd.DataFrame()
        else:
            df["expiry"] = df[exp_col].astype(str).str[:10]

        # option type
        type_col = next((c for c in ["option_type","type","Type","call_put"] if c in df.columns), None)
        if type_col is None:
            sym_col = next((c for c in ["option","symbol","Symbol"] if c in df.columns), None)
            df["type"] = df[sym_col].apply(self._occ_type) if sym_col else "call"
        else:
            df["type"] = df[type_col].map({
                "C":"call","P":"put","c":"call","p":"put",
                "call":"call","put":"put","CALL":"call","PUT":"put",
            })

        # strike
        k_col = next((c for c in ["strike","Strike","strike_price"] if c in df.columns), None)
        if k_col is None:
            sym_col = next((c for c in ["option","symbol","Symbol"] if c in df.columns), None)
            df["strike"] = df[sym_col].apply(self._occ_strike) if sym_col else np.nan
        else:
            df["strike"] = pd.to_numeric(df[k_col], errors="coerce")

        # bid / ask / volume
        df["bid"]    = pd.to_numeric(df.get("bid",    pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        df["ask"]    = pd.to_numeric(df.get("ask",    pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        df["volume"] = pd.to_numeric(df.get("volume", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)

        # IV (CBOE ให้มาแล้ว)
        iv_col = next((c for c in ["iv","IV","implied_volatility","ImpliedVolatility"] if c in df.columns), None)
        if iv_col:
            df["iv"] = pd.to_numeric(df[iv_col], errors="coerce")
            if df["iv"].dropna().max() > 5:
                df["iv"] = df["iv"] / 100.0
        else:
            # คำนวณจาก mid price
            df["mid"] = (df["bid"] + df["ask"]) / 2.0
            df["iv"]  = df.apply(
                lambda r: self._calc_iv(r["mid"], spot, r["strike"],
                    max((datetime.strptime(r["expiry"][:10], "%Y-%m-%d") - today).days, 0) / 365.0,
                    r["type"]) if r["mid"] > 0 else np.nan, axis=1)

        return self._finalize(df, spot, today)

    # ── Parse Nasdaq Data Link raw ─────────────
    def parse_nasdaq(self, df_raw: pd.DataFrame, spot: float) -> pd.DataFrame:
        df    = df_raw.copy()
        today = datetime.now()

        col_map = {
            "expiration":         "expiry",
            "call_put":           "type",
            "implied_volatility": "iv",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        df["type"]   = df["type"].map({"C":"call","P":"put","call":"call","put":"put"})
        df["strike"] = pd.to_numeric(df.get("strike", np.nan), errors="coerce")
        df["bid"]    = pd.to_numeric(df.get("bid",    0),      errors="coerce").fillna(0)
        df["ask"]    = pd.to_numeric(df.get("ask",    0),      errors="coerce").fillna(0)
        df["volume"] = pd.to_numeric(df.get("volume", 0),      errors="coerce").fillna(0).astype(int)
        df["expiry"] = df["expiry"].astype(str).str[:10]
        df["iv"]     = pd.to_numeric(df["iv"], errors="coerce")

        # Nasdaq IV อาจเป็น % หรือ decimal
        if df["iv"].dropna().max() > 5:
            df["iv"] = df["iv"] / 100.0

        return self._finalize(df, spot, today)

    # ── OCC symbol parsers ────────────────────
    @staticmethod
    def _occ_expiry(s: str) -> Optional[str]:
        m = re.search(r'(\d{6})[CP]', str(s))
        if m:
            d = m.group(1)
            return f"20{d[:2]}-{d[2:4]}-{d[4:6]}"
        return None

    @staticmethod
    def _occ_type(s: str) -> str:
        m = re.search(r'\d{6}([CP])', str(s))
        return "call" if m and m.group(1) == "C" else "put"

    @staticmethod
    def _occ_strike(s: str) -> float:
        m = re.search(r'[CP](\d{8})$', str(s))
        return float(m.group(1)) / 1000.0 if m else np.nan

    # ── Finalize: compute T, moneyness, OTM filter ──
    def _finalize(self, df: pd.DataFrame, spot: float, today: datetime) -> pd.DataFrame:
        df["T"]    = df["expiry"].apply(
            lambda x: max((datetime.strptime(x[:10], "%Y-%m-%d") - today).days, 0) / 365.0
            if len(str(x)) >= 10 else 0)
        df["days"]      = (df["T"] * 365).round().astype(int)
        df["moneyness"] = np.log(df["strike"] / spot)

        # OTM only filter (มาตรฐาน IV Surface)
        otm = (
            ((df["type"] == "put")  & (df["moneyness"] <= 0.01)) |
            ((df["type"] == "call") & (df["moneyness"] >= -0.01))
        )
        df = df[
            otm &
            (df["iv"]   > 0.001) & (df["iv"] < self.MAX_IV) &
            (df["days"] > 0) &
            (df["moneyness"] >= self.MONEYNESS_MIN) &
            (df["moneyness"] <= self.MONEYNESS_MAX) &
            (df["bid"]  > 0)
        ].dropna(subset=["strike","iv","expiry","type"])

        return df.reset_index(drop=True)

    # ── Build IV Surface grid ─────────────────
    def build_surface(self, df: pd.DataFrame, spot: float,
                      symbol: str, as_of: str) -> Optional[SurfaceData]:
        # IQR filter per expiry
        clean = []
        for _, grp in df.groupby("expiry"):
            if len(grp) < 3:
                continue
            q1, q3   = grp["iv"].quantile([0.05, 0.95])
            iqr      = q3 - q1
            filtered = grp[(grp["iv"] >= q1 - 1.5*iqr) & (grp["iv"] <= q3 + 1.5*iqr)]
            if len(filtered) >= 3:
                clean.append(filtered)

        if not clean:
            return None

        df_c   = pd.concat(clean)
        k_grid = np.linspace(df_c["moneyness"].quantile(0.02),
                             df_c["moneyness"].quantile(0.98), self.GRID_NK)

        # Single expiry → 1D spline
        if df_c["T"].nunique() == 1:
            T_val  = float(df_c["T"].iloc[0])
            t_grid = np.array([T_val])
            grp2   = df_c.groupby("moneyness", as_index=False)["iv"].mean().sort_values("moneyness")
            if len(grp2) < 2:
                return None
            kind = "cubic" if len(grp2) >= 4 else "linear"
            f    = interp1d(grp2["moneyness"].values, grp2["iv"].values, kind=kind,
                            bounds_error=False,
                            fill_value=(grp2["iv"].values[0], grp2["iv"].values[-1]))
            surf = np.clip(f(k_grid), 0.005, self.MAX_IV).reshape(1, -1)

        # Multi expiry → 2D griddata
        else:
            t_log  = np.linspace(np.log(df_c["T"].min()), np.log(df_c["T"].max()), self.GRID_NT)
            t_grid = np.exp(t_log)
            KK, TT = np.meshgrid(k_grid, t_grid)
            pts    = df_c[["moneyness","T"]].values
            vals   = df_c["iv"].values
            surf   = griddata(pts, vals, (KK, TT), method="cubic")
            near   = griddata(pts, vals, (KK, TT), method="nearest")
            surf[np.isnan(surf)] = near[np.isnan(surf)]
            surf   = np.clip(surf, 0.005, self.MAX_IV)

        return SurfaceData(k_grid=k_grid, t_grid=t_grid, iv_surface=surf,
                           df_clean=df_c, spot=spot, symbol=symbol, as_of=as_of)


# ════════════════════════════════════════════════════════════════
# IV SURFACE PLOTTER  (shared visualization)
# ════════════════════════════════════════════════════════════════

class IVSurfacePlotter:
    """
    รับ SurfaceData → คืน Plotly Figure
    x_mode: "moneyness" | "strike"
    """

    SURF_COLORS = [
        [0.00, "#143dd9"], [0.22, "#00b3cc"], [0.42, "#00cc4d"],
        [0.62, "#d9d100"], [0.80, "#f25800"], [1.00, "#cc0018"],
    ]
    LINE_COLORS = [
        "#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db",
        "#9b59b6","#1abc9c","#e91e63","#00bcd4","#ff5722","#8bc34a","#ff9800",
    ]

    @staticmethod
    def _fmt_T(t: float) -> str:
        d = int(round(t * 365))
        return f"{d}d" if d < 30 else (f"{d//30}M" if d < 365 else f"{d/365:.1f}Y")

    def plot(self, sd: SurfaceData, x_mode: str = "moneyness") -> go.Figure:
        k_grid     = sd.k_grid
        t_grid     = sd.t_grid
        iv_surface = sd.iv_surface
        df_raw     = sd.df_clean
        S          = sd.spot
        sym        = sd.symbol

        # X-axis values
        if x_mode == "strike":
            x_grid    = np.exp(k_grid) * S
            x_scatter = df_raw["strike"]
            x_atm     = S
            x_title   = f"Strike Price  (Spot={S:,.0f})"
            x_hover   = "Strike: %{x:,.0f}"
        else:
            x_grid    = np.exp(k_grid) * 100
            x_scatter = np.exp(df_raw["moneyness"]) * 100
            x_atm     = 100.0
            x_title   = "Moneyness  K/S (%)"
            x_hover   = "Moneyness: %{x:.1f}%"

        t_labels = [self._fmt_T(t) for t in t_grid]
        KK, TT   = np.meshgrid(x_grid, np.arange(len(t_grid)))
        single   = df_raw["T"].nunique() == 1

        label    = f"{sym}  [{sd.as_of}]" if sd.as_of else sym

        return (self._plot_smile(x_grid, t_grid, iv_surface, df_raw,
                                 x_scatter, x_atm, x_title, label, S)
                if single else
                self._plot_surface(KK, TT, x_grid, t_grid, iv_surface, df_raw,
                                   x_scatter, x_atm, x_title, x_hover,
                                   t_labels, sym, label, S))

    def _plot_smile(self, x_grid, t_grid, iv_surface, df_raw,
                    x_scatter, x_atm, x_title, label, S):
        fig = go.Figure()
        for otype, col in [("put","#e74c3c"), ("call","#3498db")]:
            sub = df_raw[df_raw["type"] == otype]
            fig.add_trace(go.Scatter(
                x=x_scatter[sub.index], y=sub["iv"] * 100,
                mode="markers", marker=dict(size=6, color=col), name=otype,
                text=sub.apply(
                    lambda r: f"K={r['strike']:.0f}  IV={r['iv']*100:.1f}%  Vol={r['volume']}", axis=1),
                hoverinfo="text",
            ))
        t_mid = len(t_grid) // 2
        smile = iv_surface[t_mid, :] * 100
        valid = ~np.isnan(smile)
        fig.add_trace(go.Scatter(
            x=x_grid[valid], y=smile[valid],
            mode="lines", line=dict(color="#00e0ff", width=2.5), name="Smile fit",
        ))
        fig.add_shape(type="line", x0=x_atm, x1=x_atm, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="rgba(255,255,255,0.4)", dash="dot", width=1))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
            title=f"IV Smile — {label}  Expiry: {df_raw['expiry'].iloc[0]}  Spot: {S:,.2f}",
            xaxis_title=x_title, yaxis_title="Implied Volatility (%)",
            font=dict(family="monospace", size=12, color="#c8d8f0"), height=520,
        )
        return fig

    def _plot_surface(self, KK, TT, x_grid, t_grid, iv_surface, df_raw,
                      x_scatter, x_atm, x_title, x_hover, t_labels, sym, label, S):
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type":"surface"}, {"type":"scatter"}]],
            column_widths=[0.67, 0.33],
            subplot_titles=[f"{sym} Implied Volatility Surface", "Smile by Expiry"],
        )
        fig.add_trace(go.Surface(
            x=KK, y=TT, z=iv_surface * 100,
            colorscale=self.SURF_COLORS,
            colorbar=dict(title="IV (%)", x=0.63, len=0.85, thickness=12),
            opacity=0.93,
            lighting=dict(ambient=0.7, diffuse=0.85, specular=0.3),
            hovertemplate=f"{x_hover}<br>IV: %{{z:.1f}}%<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=x_scatter,
            y=[int(np.argmin(np.abs(t_grid - t))) for t in df_raw["T"]],
            z=df_raw["iv"] * 100,
            mode="markers",
            marker=dict(size=2, color="rgba(255,255,255,0.25)"),
            hovertemplate="IV:%{z:.1f}%<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

        for j, t_val in enumerate(sorted(df_raw["T"].unique())):
            t_idx = int(np.argmin(np.abs(t_grid - t_val)))
            smile = iv_surface[t_idx, :] * 100
            valid = ~np.isnan(smile)
            fig.add_trace(go.Scatter(
                x=x_grid[valid], y=smile[valid],
                mode="lines", name=self._fmt_T(t_val),
                line=dict(color=self.LINE_COLORS[j % len(self.LINE_COLORS)], width=2),
                hovertemplate="IV:%{y:.1f}%<extra></extra>",
            ), row=1, col=2)

        y_min = float(np.nanmin(iv_surface) * 100)
        y_max = float(np.nanmax(iv_surface) * 100)
        fig.add_trace(go.Scatter(
            x=[x_atm, x_atm], y=[y_min, y_max],
            mode="lines", line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=2)

        tick_idx = np.round(np.linspace(0, len(t_grid)-1, min(7, len(t_grid)))).astype(int)
        fig.update_layout(
            scene=dict(
                xaxis=dict(title=x_title, gridcolor="#1a2540", color="#8aadee"),
                yaxis=dict(title="Expiry",
                           tickvals=list(tick_idx),
                           ticktext=[t_labels[i] for i in tick_idx],
                           gridcolor="#1a2540", color="#8aadee"),
                zaxis=dict(title="IV (%)", gridcolor="#1a2540", color="#8aadee"),
                bgcolor="#080d1c",
                camera=dict(eye=dict(x=1.6, y=-1.8, z=1.0)),
                aspectratio=dict(x=1.2, y=1.0, z=0.65),
            ),
        )
        fig.update_xaxes(title_text=x_title,   gridcolor="#1a2540", color="#8aadee", row=1, col=2)
        fig.update_yaxes(title_text="IV (%)",   gridcolor="#1a2540", color="#8aadee", row=1, col=2)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
            title=f"IV Surface — {label}  Spot: {S:,.2f}",
            font=dict(family="monospace", size=11, color="#c8d8f0"),
            margin=dict(l=0, r=0, t=50, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="rgba(100,150,255,0.3)",
                        borderwidth=1, font=dict(size=10)),
            height=650,
        )
        return fig


# ════════════════════════════════════════════════════════════════
# UI MODES
# ════════════════════════════════════════════════════════════════

class BaseMode(ABC):
    """Base class สำหรับ UI mode"""

    engine  = IVSurfaceEngine()
    plotter = IVSurfacePlotter()

    TICKERS = {
        "SPY — S&P 500 ETF":       "SPY",
        "QQQ — Nasdaq 100 ETF":    "QQQ",
        "IWM — Russell 2000 ETF":  "IWM",
        "SPX — S&P 500 Index":     "SPX",
        "NDX — Nasdaq 100 Index":  "NDX",
        "RUT — Russell 2000 Index":"RUT",
    }

    @abstractmethod
    def render_sidebar(self) -> None: ...

    @abstractmethod
    def render_main(self) -> None: ...

    def _render_result(self, sd: SurfaceData, x_mode: str, df_sel: pd.DataFrame) -> None:
        """แสดง metric + chart + raw data (ใช้ร่วมกันทั้ง 2 mode)"""
        # Metrics
        st.subheader("📊 ATM Implied Volatility")
        k_grid     = sd.k_grid
        t_grid     = sd.t_grid
        iv_surface = sd.iv_surface
        df_clean   = sd.df_clean
        atm_idx    = int(np.argmin(np.abs(k_grid)))
        sorted_Ts  = sorted(df_clean["T"].unique())
        cols       = st.columns(min(len(sorted_Ts), 6))

        for j, t_val in enumerate(sorted_Ts[:6]):
            t_idx  = int(np.argmin(np.abs(t_grid - t_val)))
            atm_iv = iv_surface[t_idx, atm_idx]
            p_idx  = int(np.argmin(np.abs(k_grid - (-0.10))))
            put_iv = iv_surface[t_idx, p_idx]
            skew   = put_iv - atm_iv if not np.isnan(put_iv) else np.nan
            with cols[j]:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{IVSurfacePlotter._fmt_T(t_val)} ATM</div>
                  <div class="metric-value">{atm_iv*100:.1f}%</div>
                  <div class="metric-sub">Skew {f"{skew*100:+.1f}%" if not np.isnan(skew) else "N/A"}</div>
                </div>
                """, unsafe_allow_html=True)

        # Chart
        st.subheader("📈 IV Surface")
        fig = self.plotter.plot(sd, x_mode=x_mode)
        st.plotly_chart(fig, use_container_width=True)

        # Raw data
        with st.expander("📋 Raw Data"):
            show = df_sel[["expiry","type","strike","moneyness","iv","bid","ask","volume"]].copy()
            show["iv_%"]      = (show["iv"] * 100).round(2)
            show["moneyness"] = show["moneyness"].round(4)
            st.dataframe(show.drop(columns=["iv"]), use_container_width=True, height=300)

        # Download
        html = fig.to_html(include_plotlyjs="cdn")
        st.download_button("⬇️ Download HTML", data=html,
            file_name=f"iv_{sd.symbol}_{sd.as_of}.html", mime="text/html",
            use_container_width=True)


# ── Live Mode (CBOE) ──────────────────────────
class LiveMode(BaseMode):
    """
    Mode ปัจจุบัน — ดึงจาก CBOE CDN
    ฟรี ไม่ต้อง Key
    """

    KEY = "live"   # session_state prefix

    def __init__(self):
        self.source = CBOELiveSource()

    def render_sidebar(self) -> None:
        st.subheader("1️⃣ เลือก Underlying")
        label = st.selectbox("Index / ETF", list(self.TICKERS.keys()), key="live_ticker")
        sym   = self.TICKERS[label]
        st.session_state["live_sym"] = sym

        st.divider()
        st.subheader("2️⃣ โหลด Expiry List")
        st.caption("CBOE Delayed · ฟรี ไม่ต้อง Key")

        if st.button("🔄 โหลดข้อมูล", use_container_width=True, key="live_load"):
            with st.spinner(f"กำลังดึงข้อมูล {sym} จาก CBOE ..."):
                try:
                    df_raw, spot = self.source.fetch_quotes(sym)
                except Exception as e:
                    st.error(f"CBOE Error: {e}"); return

            df_parsed = self.engine.parse_cboe(df_raw, spot)
            exp_info  = (df_parsed.groupby("expiry")
                         .agg(days=("days","first"), T=("T","first"), n=("iv","count"))
                         .reset_index().sort_values("days"))

            st.session_state[f"{self.KEY}_parsed"]   = df_parsed
            st.session_state[f"{self.KEY}_exp_info"] = exp_info
            st.session_state[f"{self.KEY}_spot"]     = spot
            st.success(f"✅ Spot: {spot:,.2f}  |  {len(exp_info)} expiry  |  {len(df_parsed):,} options")

        st.divider()

        if f"{self.KEY}_exp_info" not in st.session_state:
            return

        exp_info = st.session_state[f"{self.KEY}_exp_info"]
        self._render_expiry_selector(exp_info)

        st.divider()
        st.subheader("4️⃣ แสดงผล")
        st.radio("แกน X", ["moneyness","strike"], horizontal=True,
                 key="live_xmode",
                 help="moneyness=K/S%  |  strike=ราคาจริง")
        st.button("🚀 สร้าง IV Surface", type="primary",
                  use_container_width=True, key="live_run",
                  disabled=len(st.session_state.get("live_selected_dates", [])) == 0)

    def _render_expiry_selector(self, exp_info: pd.DataFrame) -> None:
        st.subheader("3️⃣ เลือก Expiry")
        mode = st.radio("โหมด", [
            "🗓️ 1 วัน", "📅 ช่วงวัน", "⭐ Preset", "✏️ หลายวัน",
        ], key="live_expmode", index=1)

        def fmt(r): return f"{r['expiry']} ({int(r['days'])}d)"

        if mode == "🗓️ 1 วัน":
            opts   = exp_info.to_dict("records")
            chosen = st.selectbox("Expiry", opts, format_func=fmt,
                                  index=min(2, len(opts)-1), key="live_single")
            dates  = [chosen["expiry"]]

        elif mode == "📅 ช่วงวัน":
            d_min = st.number_input("วันเริ่มต้น", 1,  60, 7,   key="live_dmin")
            d_max = st.number_input("วันสิ้นสุด",  30, 730, 365, key="live_dmax")
            n_max = st.slider("สูงสุด", 2, 20, 10, key="live_nmax")
            filt  = exp_info[(exp_info["days"] >= d_min) & (exp_info["days"] <= d_max)]
            if len(filt) > n_max:
                idx  = np.round(np.linspace(0, len(filt)-1, n_max)).astype(int)
                filt = filt.iloc[idx]
            dates = filt["expiry"].tolist()
            st.success(f"เลือก {len(dates)} expiry")

        elif mode == "⭐ Preset":
            PRESETS = {
                "Short-term 1W→1Y":  [7,14,30,60,90,180,365],
                "Near-term  1W→3M":  [7,14,21,30,45,60,90],
                "Long-term  3M→2Y":  [90,180,270,365,540,730],
                "Sparse 1W/1M/3M/1Y":[7,30,90,365],
            }
            preset = st.selectbox("เลือก", list(PRESETS.keys()), key="live_preset")
            rows   = []
            for t in PRESETS[preset]:
                sub = exp_info.iloc[((exp_info["days"]-t).abs()).argsort()[:1]]
                if not sub.empty: rows.append(sub.iloc[0]["expiry"])
            dates = list(dict.fromkeys(rows))
            st.success(f"เลือก {len(dates)} expiry")

        else:  # หลายวัน
            all_d  = exp_info["expiry"].tolist()
            chosen = st.multiselect("เลือก", all_d, default=all_d[:min(5,len(all_d))], key="live_multi")
            dates  = chosen

        st.session_state["live_selected_dates"] = dates

    def render_main(self) -> None:
        if f"{self.KEY}_spot" not in st.session_state:
            st.info("👈 กด **โหลดข้อมูล** ใน Sidebar"); return

        spot = st.session_state[f"{self.KEY}_spot"]
        sym  = st.session_state.get("live_sym","SPY")
        c1,c2,c3 = st.columns(3)
        c1.metric("Underlying", sym)
        c2.metric("Spot", f"{spot:,.2f}")
        c3.metric("Source", "CBOE Delayed")

        if st.session_state.get("live_run"):
            dates     = st.session_state.get("live_selected_dates", [])
            df_parsed = st.session_state[f"{self.KEY}_parsed"]
            df_sel    = df_parsed[df_parsed["expiry"].isin(dates)].copy()

            with st.spinner("Building surface ..."):
                sd = self.engine.build_surface(df_sel, spot, sym,
                                               datetime.now().strftime("%Y-%m-%d"))
            if sd is None:
                st.error("ข้อมูลไม่พอ"); return

            st.session_state[f"{self.KEY}_sd"]  = sd
            st.session_state[f"{self.KEY}_dsel"] = df_sel
            st.success(f"✅ {len(df_sel):,} points | {df_sel['T'].nunique()} expiry")

        if f"{self.KEY}_sd" in st.session_state:
            sd     = st.session_state[f"{self.KEY}_sd"]
            df_sel = st.session_state[f"{self.KEY}_dsel"]
            x_mode = st.session_state.get("live_xmode", "moneyness")
            self._render_result(sd, x_mode, df_sel)
        elif f"{self.KEY}_parsed" in st.session_state:
            st.info("👈 เลือก Expiry แล้วกด **🚀 สร้าง IV Surface**")


# ── Historical Mode (Nasdaq Data Link) ────────
class HistoricalMode(BaseMode):
    """
    Mode ย้อนหลัง — ดึงจาก Nasdaq Data Link (QDL/OPT)
    ต้องมี API Key (ฟรีที่ data.nasdaq.com)
    ข้อมูล options ย้อนหลังถึง 2005 (ต้องมี subscription $29/เดือน)
    """

    KEY = "hist"

    DATASET_INFO = {
        "QDL/OPT — US Equity & Index Options (End-of-Day)": {
            "code":    "QDL/OPT",
            "since":   "2005-01-03",
            "note":    "ต้องมี Options subscription ($29/เดือน)",
            "fields":  ["date","ticker","expiration","strike","call_put",
                        "bid","ask","volume","open_interest","implied_volatility",
                        "delta","gamma","theta","vega"],
        },
        "CBOE/VIX — VIX Index History (Free)": {
            "code":    "CBOE/VIX",
            "since":   "1990-01-02",
            "note":    "ฟรี แต่เป็นแค่ VIX index ไม่ใช่ options chain",
            "fields":  ["Date","VIX Open","VIX High","VIX Low","VIX Close"],
        },
    }

    def __init__(self):
        self.source = None   # สร้างตอน user ใส่ API key

    def render_sidebar(self) -> None:
        st.subheader("🔑 Nasdaq Data Link API Key")
        st.caption("สมัครฟรีที่ [data.nasdaq.com](https://data.nasdaq.com/)")

        default_key = ""
        try:
            default_key = st.secrets.get("NASDAQ_API_KEY", "")
        except Exception:
            pass

        api_key = st.text_input("API Key", value=default_key,
                                type="password", key="hist_apikey",
                                placeholder="ใส่ Nasdaq Data Link API Key")

        if not api_key:
            st.warning("ต้องใส่ API Key ก่อน")
            with st.expander("📖 วิธีรับ API Key ฟรี"):
                st.markdown("""
1. ไปที่ **https://data.nasdaq.com/**
2. กด **Sign Up Free**
3. เข้า Account Settings → API Key
4. คัดลอกมาวางที่นี่

**Dataset ที่ต้องการ:**
- `QDL/OPT` = Historical US Options (ต้องซื้อ $29/เดือน)
- ทดสอบ free tier ได้ด้วย `CBOE/VIX` ก่อน
                """)
            return

        st.divider()

        # เลือก Dataset
        st.subheader("1️⃣ เลือก Dataset")
        ds_label = st.selectbox("Dataset", list(self.DATASET_INFO.keys()), key="hist_dataset")
        ds_info  = self.DATASET_INFO[ds_label]
        st.caption(f"ย้อนหลังถึง: **{ds_info['since']}**  |  {ds_info['note']}")

        st.divider()

        # เลือก Underlying
        st.subheader("2️⃣ เลือก Underlying")
        label = st.selectbox("Index / ETF", list(self.TICKERS.keys()), key="hist_ticker")
        sym   = self.TICKERS[label]
        st.session_state["_hist_sym"] = sym

        st.divider()

        # เลือกวันที่
        st.subheader("3️⃣ เลือกวันที่ / ช่วงเวลา")
        date_mode = st.radio("โหมด", ["📅 วันเดียว", "📆 ช่วงเวลา"],
                             key="hist_datemode", horizontal=True)

        if date_mode == "📅 วันเดียว":
            as_of = st.date_input("วันที่", value=date.today() - timedelta(days=1),
                                  key="hist_date_single",
                                  min_value=date(2005,1,3), max_value=date.today())
            computed_dates = [as_of.strftime("%Y-%m-%d")]

        else:  # ช่วงเวลา
            col1, col2 = st.columns(2)
            d_from = col1.date_input("จาก", value=date.today() - timedelta(days=30),
                                     key="hist_from", min_value=date(2005,1,3))
            d_to   = col2.date_input("ถึง", value=date.today() - timedelta(days=1),
                                     key="hist_to", max_value=date.today())
            freq   = st.selectbox("ความถี่", ["ทุกวัน","ทุกสัปดาห์","ทุกเดือน"],
                                  key="hist_freq")
            freq_map  = {"ทุกวัน": "B", "ทุกสัปดาห์": "W-FRI", "ทุกเดือน": "BMS"}
            dates_rng = pd.date_range(d_from, d_to, freq=freq_map[freq])
            computed_dates = [d.strftime("%Y-%m-%d") for d in dates_rng]
            st.info(f"จะโหลด **{len(computed_dates)} วัน**")

        # เก็บใน non-widget keys (ไม่ conflict กับ widget key)
        st.session_state["_hist_dates"]   = computed_dates
        st.session_state["_hist_apikey"]  = api_key
        st.session_state["_hist_ds_code"] = ds_info["code"]
        st.session_state["_hist_sym"]     = sym

        st.divider()
        st.subheader("4️⃣ แสดงผล")
        st.radio("แกน X", ["moneyness","strike"], horizontal=True,
                 key="hist_xmode",
                 help="moneyness=K/S%  |  strike=ราคาจริง")
        st.button("🚀 โหลดข้อมูลย้อนหลัง", type="primary",
                  use_container_width=True, key="hist_run")

    def render_main(self) -> None:
        sym = st.session_state.get("_hist_sym", "SPY")
        st.subheader(f"📅 ข้อมูลย้อนหลัง — {sym}")

        if not st.session_state.get("_hist_apikey"):
            st.info("👈 ใส่ API Key และตั้งค่าใน Sidebar"); return

        if st.session_state.get("hist_run"):
            api_key  = st.session_state["_hist_apikey"]
            ds_code  = st.session_state["_hist_ds_code"]
            dates    = st.session_state.get("_hist_dates", [])
            x_mode   = st.session_state.get("hist_xmode", "moneyness")

            if not dates:
                st.warning("ยังไม่ได้เลือกวันที่"); return

            try:
                import nasdaqdatalink as ndl
                ndl.ApiConfig.api_key = api_key
            except Exception as e:
                st.error(f"Import error: {e}"); return

            prog   = st.progress(0)
            status = st.empty()
            all_dfs = []

            for i, as_of in enumerate(dates):
                status.text(f"⏳ กำลังโหลด {as_of} [{i+1}/{len(dates)}]")
                prog.progress((i+1) / len(dates))
                try:
                    df = ndl.get_table(ds_code, ticker=sym, date=as_of, paginate=True)
                    if not df.empty:
                        df["_as_of"] = as_of
                        all_dfs.append(df)
                except Exception as e:
                    st.warning(f"⚠️ {as_of}: {e}")
                    continue

            prog.empty(); status.empty()

            if not all_dfs:
                st.error("ไม่ได้ข้อมูล — ตรวจสอบ API Key หรือ subscription"); return

            df_all = pd.concat(all_dfs, ignore_index=True)
            st.success(f"✅ โหลดสำเร็จ {len(df_all):,} rows จาก {len(all_dfs)} วัน")

            # Parse
            # ใช้วันสุดท้ายที่โหลดมาเป็น surface หลัก
            latest_date = max(all_dfs, key=lambda d: d["_as_of"].iloc[0])["_as_of"].iloc[0]
            df_latest   = df_all[df_all["_as_of"] == latest_date].copy()

            # ประมาณ spot จาก ATM
            if "underlying_price" in df_latest.columns:
                spot = float(df_latest["underlying_price"].dropna().median())
            else:
                # ใช้ strike ที่ IV ต่ำสุด ≈ ATM
                iv_col = next((c for c in ["implied_volatility","iv","IV"] if c in df_latest.columns), None)
                if iv_col:
                    spot = float(df_latest.loc[pd.to_numeric(df_latest[iv_col], errors="coerce").idxmin(), "strike"])
                else:
                    spot = float(df_latest["strike"].median())

            df_parsed = self.engine.parse_nasdaq(df_latest, spot)

            if df_parsed.empty:
                st.error("Parse ไม่สำเร็จ — columns ไม่ตรง")
                with st.expander("Debug columns"):
                    st.code(str(list(df_all.columns)))
                return

            with st.spinner("Building surface ..."):
                sd = self.engine.build_surface(df_parsed, spot, sym, latest_date)

            if sd is None:
                st.error("ข้อมูลไม่พอสร้าง surface"); return

            st.session_state["hist_sd"]     = sd
            st.session_state["hist_df_sel"] = df_parsed
            st.session_state["hist_df_all"] = df_all

        # Render results
        if "hist_sd" in st.session_state:
            sd     = st.session_state["hist_sd"]
            df_sel = st.session_state["hist_df_sel"]
            x_mode = st.session_state.get("hist_xmode", "moneyness")

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Symbol",     sd.symbol)
            c2.metric("As of Date", sd.as_of)
            c3.metric("Spot",       f"{sd.spot:,.2f}")
            c4.metric("IV Points",  f"{len(df_sel):,}")

            self._render_result(sd, x_mode, df_sel)

            # IV History chart ถ้ามีหลายวัน
            df_all = st.session_state.get("hist_df_all")
            if df_all is not None and df_all["_as_of"].nunique() > 1:
                self._render_iv_history(df_all)

        else:
            st.info("👈 ตั้งค่าใน Sidebar แล้วกด **🚀 โหลดข้อมูลย้อนหลัง**")
            self._render_dataset_info()

    def _render_iv_history(self, df_all: pd.DataFrame) -> None:
        st.subheader("📉 ATM IV ย้อนหลัง")
        iv_col = next((c for c in ["implied_volatility","iv","IV"] if c in df_all.columns), None)
        if not iv_col: return

        history = []
        for as_of, grp in df_all.groupby("_as_of"):
            grp_f = grp[pd.to_numeric(grp[iv_col], errors="coerce") > 0]
            if grp_f.empty: continue
            atm_iv = pd.to_numeric(grp_f[iv_col], errors="coerce").median()
            history.append({"date": as_of, "atm_iv": atm_iv})

        if not history: return
        df_hist = pd.DataFrame(history).sort_values("date")

        fig = go.Figure(go.Scatter(
            x=df_hist["date"], y=df_hist["atm_iv"] * 100,
            mode="lines+markers",
            line=dict(color="#3498db", width=2),
            marker=dict(size=5),
            name="Median IV (%)",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
            title="ATM IV History",
            xaxis_title="Date", yaxis_title="IV (%)",
            font=dict(family="monospace", size=11, color="#c8d8f0"),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_dataset_info(self) -> None:
        st.markdown("### 📚 ข้อมูลเกี่ยวกับ Dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**QDL/OPT — US Options (Paid)**
- ย้อนหลังถึง **2005**
- End-of-day ทุก trading day
- รองรับ SPY, QQQ, IWM, SPX, NDX, RUT
- Fields: bid, ask, IV, Delta, Gamma, Theta, Vega
- ราคา: **$29/เดือน** หรือ $249/ปี
            """)
        with col2:
            st.markdown("""
**CBOE/VIX — VIX Index (Free)**
- ย้อนหลังถึง **1990**
- เป็นแค่ VIX index value
- ไม่ใช่ options chain ไม่สร้าง surface ได้
- ใช้ดู volatility regime ย้อนหลัง

**ทดสอบ API Key ฟรีได้เลย:**
- ไปที่ data.nasdaq.com → Sign Up → API Key
- ลอง CBOE/VIX ก่อนเพื่อยืนยัน key ใช้ได้
            """)


# ════════════════════════════════════════════════════════════════
# STREAMLIT APP  (entry point)
# ════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="IV Surface — US Index",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    [data-testid="stSidebar"] { background: #0d1425; }
    .metric-card {
        background: #0d1a30; border: 1px solid #1a2e50;
        border-radius: 8px; padding: 12px 16px;
        text-align: center; margin-bottom: 8px;
    }
    .metric-label { font-size: 11px; color: #6080a0; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 600; color: #e0f0ff; font-family: monospace; }
    .metric-sub   { font-size: 11px; color: #8090b0; margin-top: 2px; }
    .mode-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 11px; font-weight: 600; margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Mode selector ─────────────────────────
    with st.sidebar:
        st.markdown("## 📈 IV Surface")
        mode_label = st.radio(
            "เลือก Mode",
            ["🟢 Live (CBOE)", "🕐 ย้อนหลัง (Nasdaq)"],
            key="app_mode",
        )
        st.divider()

    # ── Instantiate & render ──────────────────
    if mode_label == "🟢 Live (CBOE)":
        mode = LiveMode()
        st.title("📈 IV Surface — Live (CBOE Delayed)")
        st.caption("ข้อมูล: CBOE Public API · Delayed 15 min · ฟรี ไม่ต้อง Key")
    else:
        mode = HistoricalMode()
        st.title("🕐 IV Surface — ย้อนหลัง (Nasdaq Data Link)")
        st.caption("ข้อมูล: Nasdaq Data Link QDL/OPT · ย้อนหลังถึง 2005 · ต้องมี API Key")

    with st.sidebar:
        mode.render_sidebar()

    mode.render_main()


# ── Guard ─────────────────────────────────────
if __name__ == "__main__":
    main()
else:
    # Streamlit runs by importing — call main directly
    main()
