"""
IV Surface จากข้อมูลจริง — US Index Options
=============================================
แหล่งข้อมูล : yfinance (Yahoo Finance — delayed ~15 min)
วิธีคำนวณ  : Black-Scholes + Brent's Method (invert price → IV)
Visualization: Plotly 3D Surface (HTML interactive)

ติดตั้ง:
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
from datetime import datetime
import sys

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKERS = {
    "1": ("SPY",  "S&P 500 ETF (SPY)"),
    "2": ("QQQ",  "Nasdaq 100 ETF (QQQ)"),
    "3": ("IWM",  "Russell 2000 ETF (IWM)"),
    "4": ("^SPX", "S&P 500 Index (SPX)"),
}

RISK_FREE_RATE = 0.053
MIN_BID        = 0.05
MIN_VOLUME     = 1
MONEYNESS_MIN  = -0.45
MONEYNESS_MAX  = 0.35
MAX_IV         = 3.0
GRID_NK        = 60
GRID_NT        = 50


# ─────────────────────────────────────────────
# BLACK-SCHOLES
# ─────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, opt_type="call"):
    if T <= 1e-6 or sigma <= 1e-6:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calc_iv(market_price, S, K, T, r, opt_type="call"):
    if market_price <= 0 or T <= 1e-6:
        return np.nan
    fwd = S * np.exp(r * T)
    intrinsic = max(0.0, (fwd - K) if opt_type == "call" else (K - fwd))
    if market_price <= intrinsic * np.exp(-r * T) * 0.999:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, opt_type) - market_price

    try:
        if objective(1e-4) * objective(MAX_IV) > 0:
            return np.nan
        iv = brentq(objective, 1e-4, MAX_IV, xtol=1e-7, maxiter=200)
        return iv if 0.005 <= iv <= MAX_IV else np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────
# STEP 1 — เลือก Index
# ─────────────────────────────────────────────
def select_ticker():
    print("\n" + "="*52)
    print("  US Index IV Surface  —  เลือก Underlying")
    print("="*52)
    for k, (sym, name) in TICKERS.items():
        print(f"  [{k}] {sym:6s}  {name}")
    c = input("\nเลือก (1-4, default=1): ").strip() or "1"
    sym, name = TICKERS.get(c, TICKERS["1"])
    print(f"  -> เลือก: {name}")
    return sym, name


# ─────────────────────────────────────────────
# STEP 2 — โหลดแค่รายการ Expiry (เร็ว)
# ─────────────────────────────────────────────
def fetch_expiry_list(sym):
    print(f"\n  กำลังโหลดรายการ Expiry dates ...")
    stock = yf.Ticker(sym)

    hist = stock.history(period="2d")
    if hist.empty:
        raise ValueError(f"ไม่พบข้อมูลราคา {sym}")
    S = float(hist["Close"].iloc[-1])

    expiries = stock.options   # แค่ดึง list string — ไม่โหลด chain
    today    = datetime.now()

    rows = []
    for exp in expiries:
        exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        days   = (exp_dt - today).days
        if days < 1:
            continue
        rows.append({"date": exp, "days": days, "T": days / 365.0})

    df_exp = pd.DataFrame(rows)
    print(f"  Spot Price : {S:,.2f}")
    print(f"  พบ Expiry  : {len(df_exp)} dates (จาก {df_exp['date'].iloc[0]} ถึง {df_exp['date'].iloc[-1]})")
    return df_exp, S, stock


# ─────────────────────────────────────────────
# STEP 3 — User เลือก Expiry Mode
# ─────────────────────────────────────────────
def select_expiries(df_exp):
    print("\n" + "="*52)
    print("  เลือกโหมดการโหลด Expiry")
    print("="*52)
    print("  [1]  เลือก 1 วัน          (ดู Smile เดี่ยว)")
    print("  [2]  เลือกช่วงวัน          (สร้าง Surface หลาย expiry)")
    print("  [3]  เลือก Preset          (1W 1M 3M 6M 1Y ...)")
    print("  [4]  เลือกเองทีละวัน       (เลือก multiple จาก list)")

    mode = input("\nเลือก mode (1-4, default=2): ").strip() or "2"

    if   mode == "1": return _select_single(df_exp)
    elif mode == "3": return _select_preset(df_exp)
    elif mode == "4": return _select_manual(df_exp)
    else:             return _select_range(df_exp)


def _fmt_lbl(days):
    if days < 30:   return f"{days}d"
    elif days < 365: return f"{days//30}M"
    else:            return f"{days/365:.1f}Y"


def _show_table(df_exp):
    print(f"\n  {'No':>4}  {'Date':>12}  {'Days':>6}  {'Label':>6}")
    print(f"  {'─'*33}")
    for i, row in df_exp.iterrows():
        print(f"  {i+1:>4}  {row['date']:>12}  {int(row['days']):>6}  {_fmt_lbl(int(row['days'])):>6}")


def _select_single(df_exp):
    _show_table(df_exp)
    idx = int(input(f"\nเลือกหมายเลข (1-{len(df_exp)}): ").strip()) - 1
    sel = df_exp.iloc[[idx]].reset_index(drop=True)
    print(f"  -> เลือก: {sel.iloc[0]['date']}  ({int(sel.iloc[0]['days'])} วัน)")
    return sel


def _select_range(df_exp):
    print(f"\n  Expiry available: {int(df_exp['days'].min())} – {int(df_exp['days'].max())} วัน\n")
    d_min   = int(input("  ช่วงเริ่มต้น (วัน, default=7):    ").strip() or "7")
    d_max   = int(input("  ช่วงสิ้นสุด  (วัน, default=365):  ").strip() or "365")
    n_max   = int(input("  โหลดสูงสุด  (expiry, default=10): ").strip() or "10")

    filtered = df_exp[(df_exp["days"] >= d_min) & (df_exp["days"] <= d_max)]
    if filtered.empty:
        print("  ไม่พบ expiry ในช่วงนี้ — ใช้ทั้งหมด")
        filtered = df_exp

    # Log-space sampling ถ้ามีเยอะ
    if len(filtered) > n_max:
        log_idx = np.round(np.linspace(0, len(filtered)-1, n_max)).astype(int)
        filtered = filtered.iloc[log_idx]

    sel = filtered.reset_index(drop=True)
    print(f"\n  -> เลือก {len(sel)} expiry:")
    for _, r in sel.iterrows():
        print(f"     {r['date']}  ({int(r['days'])} วัน  /  {_fmt_lbl(int(r['days']))})")
    return sel


def _select_preset(df_exp):
    PRESETS = {
        "1": ([7, 14, 30, 60, 90, 180, 365],    "Weekly+Monthly  1W→1Y"),
        "2": ([30, 60, 90, 180, 365, 730],       "Monthly-Long    1M→2Y"),
        "3": ([7, 14, 21, 30, 45, 60, 90],       "Near-term       1W→3M"),
        "4": ([90, 180, 270, 365, 540, 730],      "Long-term       3M→2Y"),
        "5": ([7, 30, 90, 365],                   "Sparse 4 pts    1W/1M/3M/1Y"),
    }
    print()
    for k, (_, lbl) in PRESETS.items():
        print(f"  [{k}]  {lbl}")
    pc = input("\nเลือก preset (1-5, default=1): ").strip() or "1"
    targets, label = PRESETS.get(pc, PRESETS["1"])

    sel_rows = []
    for t in targets:
        if t < df_exp["days"].min() or t > df_exp["days"].max():
            continue
        idx = (df_exp["days"] - t).abs().idxmin()
        sel_rows.append(df_exp.loc[idx])

    sel = pd.DataFrame(sel_rows).drop_duplicates("date").reset_index(drop=True)
    print(f"\n  -> Preset: {label}  ({len(sel)} expiry):")
    for _, r in sel.iterrows():
        print(f"     {r['date']}  ({int(r['days'])} วัน  /  {_fmt_lbl(int(r['days']))})")
    return sel


def _select_manual(df_exp):
    _show_table(df_exp)
    raw = input(f"\nพิมพ์หมายเลขที่ต้องการ คั่นด้วยเว้นวรรค (เช่น: 1 3 5 8): ").strip()
    idxs = [int(x)-1 for x in raw.split() if x.isdigit()]
    idxs = [i for i in idxs if 0 <= i < len(df_exp)]
    sel  = df_exp.iloc[idxs].reset_index(drop=True)
    print(f"\n  -> เลือก {len(sel)} expiry:")
    for _, r in sel.iterrows():
        print(f"     {r['date']}  ({int(r['days'])} วัน  /  {_fmt_lbl(int(r['days']))})")
    return sel


# ─────────────────────────────────────────────
# STEP 4 — โหลด Options Chain เฉพาะที่เลือก
# ─────────────────────────────────────────────
def fetch_selected_options(stock, df_sel, S, r=RISK_FREE_RATE):
    print(f"\n  กำลังโหลด Options Chain {len(df_sel)} expiry ...\n")
    records = []
    total   = len(df_sel)

    for i, (_, row) in enumerate(df_sel.iterrows(), 1):
        exp  = row["date"]
        T    = float(row["T"])
        done = i / total
        bar  = ("█" * i) + ("░" * (total - i))
        print(f"  [{bar}] {done*100:5.0f}%  {exp}  ({int(row['days'])}d)", end="\r", flush=True)

        try:
            chain = stock.option_chain(exp)
        except Exception as e:
            print(f"\n  ! {exp}: {e}")
            continue

        for df_raw, otype in [(chain.calls, "call"), (chain.puts, "put")]:
            df = df_raw.copy()
            df["mid"]   = (df["bid"] + df["ask"]) / 2.0
            df["price"] = np.where(df["mid"] > 0, df["mid"], df["lastPrice"])

            mask = (
                (df["bid"] >= MIN_BID) &
                (df["volume"].fillna(0) >= MIN_VOLUME) &
                (df["price"] > 0)
            )
            df = df[mask]

            for _, ropt in df.iterrows():
                K    = float(ropt["strike"])
                price = float(ropt["price"])
                mono = np.log(K / S)
                if not (MONEYNESS_MIN <= mono <= MONEYNESS_MAX):
                    continue
                iv_val = calc_iv(price, S, K, T, r, otype)
                if np.isnan(iv_val):
                    continue
                records.append({
                    "strike": K, "expiry": exp, "T": T,
                    "days":   int(row["days"]),
                    "moneyness": mono, "iv": iv_val, "type": otype,
                    "bid":    float(ropt["bid"]),
                    "ask":    float(ropt["ask"]),
                    "volume": int(ropt["volume"]) if not pd.isna(ropt["volume"]) else 0,
                })

    print(f"\n\n  IV points ที่คำนวณได้: {len(records):,}")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# BUILD SURFACE
# ─────────────────────────────────────────────
def build_iv_surface(df):
    clean = []
    for _, grp in df.groupby("expiry"):
        q1, q3 = grp["iv"].quantile([0.05, 0.95])
        iqr = q3 - q1
        clean.append(grp[(grp["iv"] >= q1 - 1.5*iqr) & (grp["iv"] <= q3 + 1.5*iqr)])
    df_c = pd.concat(clean)

    k_grid = np.linspace(MONEYNESS_MIN*0.9, MONEYNESS_MAX*0.9, GRID_NK)
    t_log  = np.linspace(np.log(df_c["T"].min()), np.log(df_c["T"].max()), GRID_NT)
    t_grid = np.exp(t_log)

    KK, TT = np.meshgrid(k_grid, t_grid)
    pts    = df_c[["moneyness","T"]].values
    vals   = df_c["iv"].values
    surf   = griddata(pts, vals, (KK, TT), method="cubic")
    near   = griddata(pts, vals, (KK, TT), method="nearest")
    surf[np.isnan(surf)] = near[np.isnan(surf)]
    return k_grid, t_grid, surf, df_c


# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
def plot_surface(k_grid, t_grid, iv_surface, df_raw, S, sym, name):
    mono_pct = np.exp(k_grid) * 100

    def fmt_T(t):
        d = int(round(t * 365))
        return f"{d}d" if d < 30 else (f"{d//30}M" if d < 365 else f"{d/365:.1f}Y")

    t_labels = [fmt_T(t) for t in t_grid]
    KK, TT   = np.meshgrid(mono_pct, np.arange(len(t_grid)))
    single   = df_raw["T"].nunique() == 1

    if single:
        fig = go.Figure()
        colors_type = np.where(df_raw["type"] == "put", "#e74c3c", "#3498db")
        fig.add_trace(go.Scatter(
            x=np.exp(df_raw["moneyness"]) * 100,
            y=df_raw["iv"] * 100,
            mode="markers",
            marker=dict(size=6, color=list(colors_type)),
            text=df_raw.apply(lambda r: f"K={r['strike']:.0f}  IV={r['iv']*100:.1f}%  {r['type']}", axis=1),
            hoverinfo="text", name="Market quotes",
        ))
        t_mid  = int(len(t_grid) // 2)
        smile  = iv_surface[t_mid, :] * 100
        valid  = ~np.isnan(smile)
        fig.add_trace(go.Scatter(
            x=mono_pct[valid], y=smile[valid],
            mode="lines", line=dict(color="#00e0ff", width=2),
            name="Smile (interpolated)",
        ))
        fig.add_vline(x=100, line=dict(color="white", dash="dot", width=1))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
            title=f"<b>IV Smile — {name}  Expiry: {df_raw['expiry'].iloc[0]}  Spot: {S:,.2f}</b>",
            xaxis_title="Moneyness (%)", yaxis_title="IV (%)",
            font=dict(family="monospace", size=11, color="#c8d8f0"),
            height=520, width=900,
        )

    else:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type":"surface"},{"type":"scatter"}]],
            column_widths=[0.70, 0.30],
            subplot_titles=[f"{sym} Implied Volatility Surface", "Smile by Expiry"],
        )
        fig.add_trace(go.Surface(
            x=KK, y=TT, z=iv_surface * 100,
            colorscale=[[0,"#143dd9"],[0.22,"#00b3cc"],[0.42,"#00cc4d"],
                        [0.62,"#d9d100"],[0.80,"#f25800"],[1.00,"#cc0018"]],
            colorbar=dict(title="IV (%)", x=0.67, len=0.9, thickness=12),
            opacity=0.92,
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.3),
            hovertemplate="Moneyness: %{x:.1f}%<br>IV: %{z:.1f}%<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=np.exp(df_raw["moneyness"]) * 100,
            y=[int(np.argmin(np.abs(t_grid - t))) for t in df_raw["T"]],
            z=df_raw["iv"] * 100,
            mode="markers",
            marker=dict(size=2, color="rgba(255,255,255,0.3)"),
            hovertemplate="M:%{x:.1f}%<br>IV:%{z:.1f}%<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

        clrs = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db","#9b59b6","#1abc9c","#e91e63","#00bcd4","#ff5722"]
        for j, t_val in enumerate(sorted(df_raw["T"].unique())):
            t_idx = int(np.argmin(np.abs(t_grid - t_val)))
            smile = iv_surface[t_idx, :] * 100
            valid = ~np.isnan(smile)
            fig.add_trace(go.Scatter(
                x=mono_pct[valid], y=smile[valid],
                mode="lines", name=fmt_T(t_val),
                line=dict(color=clrs[j % len(clrs)], width=2),
                hovertemplate="M:%{x:.1f}%<br>IV:%{y:.1f}%<extra></extra>",
            ), row=1, col=2)

        fig.add_vline(x=100, line=dict(color="white", width=1, dash="dot"), row=1, col=2)

        tick_idx = np.round(np.linspace(0, len(t_grid)-1, min(8, len(t_grid)))).astype(int)
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Moneyness (%)", gridcolor="#1a2540", color="#8aadee"),
                yaxis=dict(title="Expiry",
                           tickvals=list(tick_idx),
                           ticktext=[t_labels[i] for i in tick_idx],
                           gridcolor="#1a2540", color="#8aadee"),
                zaxis=dict(title="IV (%)", gridcolor="#1a2540", color="#8aadee"),
                bgcolor="#080d1c",
                camera=dict(eye=dict(x=1.6, y=-1.8, z=1.0)),
                aspectratio=dict(x=1.2, y=1.0, z=0.7),
            ),
        )
        fig.update_xaxes(title_text="Moneyness (%)", gridcolor="#1a2540", color="#8aadee", row=1, col=2)
        fig.update_yaxes(title_text="IV (%)", gridcolor="#1a2540", color="#8aadee", row=1, col=2)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
            title=f"<b>IV Surface — {name}  Spot: {S:,.2f}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}</b>",
            font=dict(family="monospace", size=11, color="#c8d8f0"),
            margin=dict(l=0, r=0, t=50, b=20),
            legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="rgba(100,150,255,0.3)",
                        borderwidth=1, font=dict(size=10)),
            height=660, width=1200,
        )

    return fig


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def print_summary(df, k_grid, t_grid, iv_surface, S):
    atm_idx = int(np.argmin(np.abs(k_grid)))
    print(f"\n  {'Expiry':>8}  {'Days':>5}  {'ATM IV':>8}  {'25d Skew':>10}")
    print(f"  {'─'*38}")
    for t_val in sorted(df["T"].unique()):
        t_idx  = int(np.argmin(np.abs(t_grid - t_val)))
        atm_iv = iv_surface[t_idx, atm_idx]
        if np.isnan(atm_iv):
            continue
        p_idx  = int(np.argmin(np.abs(k_grid - (-0.10))))
        put_iv = iv_surface[t_idx, p_idx]
        skew   = (put_iv - atm_iv) if not np.isnan(put_iv) else np.nan
        days   = int(round(t_val * 365))
        lbl    = _fmt_lbl(days)
        sk_str = f"{skew*100:+.1f}%" if not np.isnan(skew) else "N/A"
        print(f"  {lbl:>8}  {days:>5}  {atm_iv*100:>7.1f}%  {sk_str:>10}")
    print(f"\n  IV points  : {len(df):,}")
    print(f"  IV range   : {df['iv'].min()*100:.1f}% – {df['iv'].max()*100:.1f}%")


def _fmt_lbl(days):
    if days < 30:    return f"{days}d"
    elif days < 365: return f"{days//30}M"
    else:            return f"{days/365:.1f}Y"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # 1. เลือก Underlying
    sym, name = select_ticker()

    # 2. โหลดรายการ Expiry (รวดเร็ว ไม่โหลด chain)
    df_exp, S, stock = fetch_expiry_list(sym)

    # 3. User เลือก Expiry
    df_sel = select_expiries(df_exp)
    if df_sel.empty:
        print("  ไม่ได้เลือก expiry"); return

    confirm = input(f"\n  ยืนยันโหลด {len(df_sel)} expiry? (y/n, default=y): ").strip().lower()
    if confirm == "n":
        print("  ยกเลิก"); return

    # 4. โหลด Options Chain เฉพาะที่เลือก
    df_iv = fetch_selected_options(stock, df_sel, S)
    if df_iv.empty:
        print("\n  ERROR: ไม่ได้ข้อมูล — ตรวจสอบ internet หรือเลือก expiry ใหม่"); return

    # 5. Build Surface
    print("  Interpolating surface ...", end=" ", flush=True)
    k_grid, t_grid, iv_surface, df_clean = build_iv_surface(df_iv)
    print("done")

    # 6. Summary
    print_summary(df_clean, k_grid, t_grid, iv_surface, S)

    # 7. Plot & save
    print("\n  Generating chart ...", end=" ", flush=True)
    fig = plot_surface(k_grid, t_grid, iv_surface, df_clean, S, sym, name)
    print("done")

    out = f"iv_surface_{sym.replace('^','')}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\n  Saved: {out}")
    print(f"  เปิดไฟล์ใน Browser เพื่อดู 3D Surface\n")
    fig.show()


if __name__ == "__main__":
    main()
