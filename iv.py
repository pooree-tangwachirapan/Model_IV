"""
IV Surface จากข้อมูลจริง — US Index Options
=============================================
แหล่งข้อมูล : yfinance (Yahoo Finance — ฟรี, delayed ~15 min)
วิธีคำนวณ  : Black-Scholes + Brent's Method (invert price → IV)
Visualization: Plotly 3D Surface (interactive HTML)

ติดตั้ง dependencies:
  pip install yfinance numpy pandas scipy plotly

รัน:
  python iv_surface_real.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TICKERS = {
   " "SPY": "S&P 500 ETF (SPY)",
    "QQQ": "Nasdaq 100 ETF (QQQ)",
 "   "IWM": "Russell 2000 ETF (IWM)",
"    "SPX": "S&P 500 Index (^SPX)",
}

RISK_FREE_RATE = 0.053       # US 3M T-Bill ~5.3% (ปรับตามจริง)
MIN_EXPIRY_DAYS = 7          # ตัด expiry ที่สั้นเกินไป
MAX_EXPIRY_YEARS = 2.5       # ตัด expiry ที่ยาวเกินไป
MIN_BID = 0.05               # ตัด option ที่ bid ต่ำมาก (illiquid)
MIN_VOLUME = 1               # ตัด option ที่ไม่มี volume
MONEYNESS_MIN = -0.45        # log(K/S) min  ≈ 63% OTM put
MONEYNESS_MAX = 0.35         # log(K/S) max  ≈ 142% OTM call
MAX_IV = 3.0                 # cap IV ที่ 300% (filter outliers)
GRID_NK = 60                 # จำนวน grid points ด้าน strike
GRID_NT = 50                 # จำนวน grid points ด้าน expiry


# ─────────────────────────────────────────────
# BLACK-SCHOLES
# ─────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float,
             sigma: float, opt_type: str = "call") -> float:
    """Black-Scholes European option price"""
    if T <= 1e-6 or sigma <= 1e-6:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calc_iv(market_price: float, S: float, K: float, T: float,
            r: float, opt_type: str = "call") -> float:
    """
    คำนวณ Implied Volatility โดย invert Black-Scholes
    ใช้ Brent's Method (guaranteed convergence)
    คืน NaN ถ้าคำนวณไม่ได้
    """
    if market_price <= 0 or T <= 1e-6:
        return np.nan

    # Intrinsic value check — ราคา market ต้องสูงกว่า intrinsic
    fwd = S * np.exp(r * T)
    intrinsic = max(0.0, (fwd - K) if opt_type == "call" else (K - fwd))
    intrinsic_disc = intrinsic * np.exp(-r * T)
    if market_price <= intrinsic_disc * 0.999:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, opt_type) - market_price

    try:
        # ตรวจสอบ sign ที่ boundary
        f_lo = objective(1e-4)
        f_hi = objective(MAX_IV)
        if f_lo * f_hi > 0:
            return np.nan
        iv = brentq(objective, 1e-4, MAX_IV, xtol=1e-7, maxiter=200)
        return iv if 0.005 <= iv <= MAX_IV else np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────

def fetch_options(ticker_symbol: str, r: float = RISK_FREE_RATE) -> tuple:
    """
    ดึง options chain จาก Yahoo Finance
    คำนวณ IV แต่ละ contract
    คืน (DataFrame ของ IV points, spot price)
    """
    print(f"\n{'='*55}")
    print(f"  ดึงข้อมูล Options: {ticker_symbol}")
    print(f"{'='*55}")

    # Map SPX → ^SPX สำหรับ yfinance
    yf_symbol = "^SPX" if ticker_symbol == "SPX" else ticker_symbol
    stock = yf.Ticker(yf_symbol)

    # Spot price
    hist = stock.history(period="2d")
    if hist.empty:
        raise ValueError(f"ไม่พบข้อมูลราคา {ticker_symbol}")
    S = float(hist["Close"].iloc[-1])
    print(f"  Spot Price : {S:,.2f}")

    # รายการ expiry ที่ available
    expiries = stock.options
    print(f"  Expiries   : {len(expiries)} dates")

    records = []
    today = datetime.now()

    for exp in expiries:
        exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        T_days = (exp_dt - today).days
        T = T_days / 365.0

        if T_days < MIN_EXPIRY_DAYS or T > MAX_EXPIRY_YEARS:
            continue

        print(f"  Processing {exp} (T={T:.3f}y) ...", end=" ")

        try:
            chain = stock.option_chain(exp)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        n_added = 0
        for df_raw, otype in [(chain.calls, "call"), (chain.puts, "put")]:
            df = df_raw.copy()

            # Mid price
            df["mid"] = (df["bid"] + df["ask"]) / 2.0
            df["price"] = np.where(df["mid"] > 0, df["mid"], df["lastPrice"])

            # Filter: liquidity
            mask = (
                (df["bid"] >= MIN_BID) &
                (df["volume"].fillna(0) >= MIN_VOLUME) &
                (df["price"] > 0)
            )
            df = df[mask]

            for _, row in df.iterrows():
                K = float(row["strike"])
                price = float(row["price"])
                moneyness = np.log(K / S)

                if not (MONEYNESS_MIN <= moneyness <= MONEYNESS_MAX):
                    continue

                iv_val = calc_iv(price, S, K, T, r, otype)
                if np.isnan(iv_val):
                    continue

                records.append({
                    "strike":     K,
                    "expiry":     exp,
                    "T":          T,
                    "T_days":     T_days,
                    "moneyness":  moneyness,
                    "iv":         iv_val,
                    "type":       otype,
                    "bid":        float(row["bid"]),
                    "ask":        float(row["ask"]),
                    "volume":     int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                    "oi":         int(row["openInterest"]) if not pd.isna(row.get("openInterest", np.nan)) else 0,
                })
                n_added += 1

        print(f"{n_added} points")

    df_all = pd.DataFrame(records)
    print(f"\n  รวม IV points ที่คำนวณได้ : {len(df_all):,}")
    return df_all, S


# ─────────────────────────────────────────────
# SMOOTHING & INTERPOLATION
# ─────────────────────────────────────────────

def build_iv_surface(df: pd.DataFrame):
    """
    Interpolate scattered IV points → regular grid
    ใช้ cubic interpolation จาก scipy
    """
    # Remove outliers: IQR method per expiry
    clean_rows = []
    for exp, grp in df.groupby("expiry"):
        q1, q3 = grp["iv"].quantile([0.05, 0.95])
        iqr = q3 - q1
        cleaned = grp[(grp["iv"] >= q1 - 1.5 * iqr) & (grp["iv"] <= q3 + 1.5 * iqr)]
        clean_rows.append(cleaned)
    df_clean = pd.concat(clean_rows)

    # Grid axes
    k_grid = np.linspace(MONEYNESS_MIN * 0.9, MONEYNESS_MAX * 0.9, GRID_NK)
    t_vals = np.sort(df_clean["T"].unique())
    t_log_grid = np.linspace(
        np.log(df_clean["T"].min()),
        np.log(df_clean["T"].max()),
        GRID_NT
    )
    t_grid = np.exp(t_log_grid)

    KK, TT = np.meshgrid(k_grid, t_grid)

    # Scipy griddata interpolation
    points = df_clean[["moneyness", "T"]].values
    values = df_clean["iv"].values

    iv_surface = griddata(
        points, values,
        (KK, TT),
        method="cubic"
    )

    # Fill NaN boundary with nearest
    iv_near = griddata(points, values, (KK, TT), method="nearest")
    mask_nan = np.isnan(iv_surface)
    iv_surface[mask_nan] = iv_near[mask_nan]

    return k_grid, t_grid, iv_surface, df_clean


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_iv_surface(k_grid, t_grid, iv_surface, df_raw, S, ticker):
    """
    สร้าง Plotly 3D IV Surface แบบ interactive
    """
    # Convert log-moneyness → moneyness %
    mono_pct = np.exp(k_grid) * 100  # e.g. 90, 100, 110...
    
    # Expiry labels
    t_labels = []
    for t in t_grid:
        d = int(round(t * 365))
        if d < 30:
            t_labels.append(f"{d}d")
        elif d < 365:
            t_labels.append(f"{d//30}M")
        else:
            t_labels.append(f"{d/365:.1f}Y")

    KK, TT = np.meshgrid(mono_pct, np.arange(len(t_grid)))

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "scatter"}]],
        column_widths=[0.72, 0.28],
        subplot_titles=[
            f"{ticker} Implied Volatility Surface",
            "Volatility Smile by Expiry"
        ]
    )

    # ── 3D Surface ──
    surf = go.Surface(
        x=KK,
        y=TT,
        z=iv_surface * 100,  # แปลงเป็น %
        colorscale=[
            [0.00, "#143dd9"],
            [0.22, "#00b3cc"],
            [0.42, "#00cc4d"],
            [0.62, "#d9d100"],
            [0.80, "#f25800"],
            [1.00, "#cc0018"],
        ],
        colorbar=dict(
            title="IV (%)",
            titleside="right",
            x=0.68,
            len=0.9,
            thickness=12,
        ),
        opacity=0.92,
        lighting=dict(ambient=0.7, diffuse=0.8, specular=0.3),
        hovertemplate=(
            "Moneyness: %{x:.1f}%<br>"
            "IV: %{z:.1f}%<extra></extra>"
        ),
    )
    fig.add_trace(surf, row=1, col=1)

    # Raw scatter points
    fig.add_trace(go.Scatter3d(
        x=np.exp(df_raw["moneyness"]) * 100,
        y=[list(np.sort(df_raw["T"].unique())).index(t) if t in np.sort(df_raw["T"].unique()) else 0
           for t in df_raw["T"]],
        z=df_raw["iv"] * 100,
        mode="markers",
        marker=dict(size=2, color="rgba(255,255,255,0.4)", symbol="circle"),
        hovertemplate=(
            "Strike: %{x:.1f}%<br>"
            "IV: %{z:.1f}%<br>"
            "Type: %{customdata}<extra></extra>"
        ),
        customdata=df_raw["type"],
        name="Market quotes",
        showlegend=True,
    ), row=1, col=1)

    # ── Smile curves (right panel) ──
    SMILE_EXPIRIES = [7, 30, 90, 180, 365]
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6"]

    for target_days, color in zip(SMILE_EXPIRIES, colors):
        target_t = target_days / 365.0
        t_idx = int(np.argmin(np.abs(t_grid - target_t)))
        label_days = int(round(t_grid[t_idx] * 365))

        smile_iv = iv_surface[t_idx, :] * 100
        valid = ~np.isnan(smile_iv)

        if valid.sum() < 3:
            continue

        lbl = f"{label_days}d" if label_days < 30 else (
            f"{label_days//30}M" if label_days < 365 else f"{label_days//365}Y"
        )
        fig.add_trace(go.Scatter(
            x=mono_pct[valid],
            y=smile_iv[valid],
            mode="lines",
            name=lbl,
            line=dict(color=color, width=2),
            hovertemplate="Moneyness: %{x:.1f}%<br>IV: %{y:.1f}%<extra></extra>",
        ), row=1, col=2)

    # ATM line
    fig.add_vline(x=100, line=dict(color="white", width=1, dash="dot"), row=1, col=2)

    # ── Layout ──
    tick_indices = np.linspace(0, len(t_grid) - 1, min(8, len(t_grid)), dtype=int)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080d1c",
        plot_bgcolor="#0d1425",
        font=dict(family="monospace", size=11, color="#c8d8f0"),
        title=dict(
            text=f"<b>IV Surface — {TICKERS.get(ticker, ticker)}</b>"
                 f"   Spot: {S:,.2f}   |   {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            font=dict(size=14, color="#e0eeff"),
            x=0.01,
        ),
        margin=dict(l=0, r=0, t=50, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(100,150,255,0.3)",
            borderwidth=1,
            font=dict(size=10),
        ),
        scene=dict(
            xaxis=dict(title="Moneyness (%)", gridcolor="#1a2540", color="#8aadee"),
            yaxis=dict(
                title="Expiry",
                tickvals=list(tick_indices),
                ticktext=[t_labels[i] for i in tick_indices],
                gridcolor="#1a2540",
                color="#8aadee",
            ),
            zaxis=dict(title="IV (%)", gridcolor="#1a2540", color="#8aadee"),
            bgcolor="#080d1c",
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.0)),
            aspectratio=dict(x=1.2, y=1.0, z=0.7),
        ),
        height=660,
        width=1200,
    )

    fig.update_xaxes(
        title_text="Moneyness (%)",
        gridcolor="#1a2540",
        color="#8aadee",
        row=1, col=2,
    )
    fig.update_yaxes(
        title_text="IV (%)",
        gridcolor="#1a2540",
        color="#8aadee",
        row=1, col=2,
    )

    return fig


# ─────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────

def print_summary(df: pd.DataFrame, k_grid, t_grid, iv_surface, S, ticker):
    print(f"\n{'='*55}")
    print(f"  IV Surface Summary — {ticker}  (Spot: {S:,.2f})")
    print(f"{'='*55}")

    atm_idx = int(np.argmin(np.abs(k_grid - 0)))

    expiry_map = {7: "1W", 30: "1M", 90: "3M", 180: "6M", 365: "1Y", 730: "2Y"}
    print(f"\n  {'Expiry':>8}  {'ATM IV':>8}  {'25Δ Skew':>10}  {'Skew/ATM':>10}")
    print(f"  {'-'*42}")

    for days, lbl in expiry_map.items():
        t_target = days / 365.0
        if t_target < t_grid.min() or t_target > t_grid.max():
            continue
        t_idx = int(np.argmin(np.abs(t_grid - t_target)))
        atm_iv = iv_surface[t_idx, atm_idx]
        if np.isnan(atm_iv):
            continue

        # 25Δ put: k ≈ -0.15 * sqrt(T) roughly
        put_k = -0.10
        put_idx = int(np.argmin(np.abs(k_grid - put_k)))
        put_iv = iv_surface[t_idx, put_idx]
        skew = (put_iv - atm_iv) if not np.isnan(put_iv) else np.nan

        print(
            f"  {lbl:>8}  {atm_iv*100:>7.1f}%"
            f"  {skew*100:>+9.1f}%"
            f"  {(skew/atm_iv)*100:>+9.1f}%" if not np.isnan(skew) else
            f"  {lbl:>8}  {atm_iv*100:>7.1f}%  {'N/A':>10}  {'N/A':>10}"
        )

    print(f"\n  Total liquid option quotes : {len(df):,}")
    print(f"  Expiries covered          : {df['T'].nunique()}")
    print(f"  Strikes covered           : {df['strike'].nunique()}")
    print(f"  IV range                  : {df['iv'].min()*100:.1f}% – {df['iv'].max()*100:.1f}%")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  US Index Options — IV Surface Builder")
    print("  ข้อมูล: Yahoo Finance (yfinance)")
    print("="*55)
    print("\nเลือก Index / ETF:")
    for i, (sym, name) in enumerate(TICKERS.items(), 1):
        print(f"  [{i}] {sym:5s} — {name}")

    choice = input("\nเลือก (1-4, default=1): ").strip() or "1"
    ticker = list(TICKERS.keys())[int(choice) - 1] if choice.isdigit() else "SPY"

    # Fetch & calculate
    df_iv, spot = fetch_options(ticker)

    if df_iv.empty:
        print("\nERROR: ไม่ได้ข้อมูล option เลย — ตรวจสอบ connection")
        sys.exit(1)

    # Build surface
    print("\n  Interpolating IV surface ...", end=" ")
    k_grid, t_grid, iv_surface, df_clean = build_iv_surface(df_iv)
    print("done")

    # Summary
    print_summary(df_clean, k_grid, t_grid, iv_surface, spot, ticker)

    # Plot
    print("\n  Generating interactive chart ...", end=" ")
    fig = plot_iv_surface(k_grid, t_grid, iv_surface, df_clean, spot, ticker)
    print("done")

    out_file = f"iv_surface_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    fig.write_html(out_file, include_plotlyjs="cdn")
    print(f"\n  บันทึกไฟล์ : {out_file}")
    print(f"  เปิดไฟล์นี้ใน browser เพื่อดู 3D surface\n")

    fig.show()


if __name__ == "__main__":
    main()
