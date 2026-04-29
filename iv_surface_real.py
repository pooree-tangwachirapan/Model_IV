"""
IV Surface — US Index Options  (Streamlit App)
==============================================
รัน:  streamlit run iv_surface_app.py
ติดตั้ง: pip install streamlit yfinance numpy pandas scipy plotly
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
import streamlit as st

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="IV Surface — US Index",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d1425; }
body, .stApp { background: #080d1c; color: #c8d8f0; }
h1,h2,h3 { color: #e0eeff; }
.metric-card {
    background: #0d1a30;
    border: 1px solid #1a2e50;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.metric-label { font-size: 11px; color: #6080a0; margin-bottom: 4px; }
.metric-value { font-size: 24px; font-weight: 600; color: #e0f0ff; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────
RISK_FREE_RATE = 0.053
MIN_BID        = 0.05
MIN_VOLUME     = 1
MONEYNESS_MIN  = -0.45
MONEYNESS_MAX  = 0.35
MAX_IV         = 3.0
GRID_NK        = 60
GRID_NT        = 50

TICKERS = {
    "SPY  — S&P 500 ETF":      "SPY",
    "QQQ  — Nasdaq 100 ETF":   "QQQ",
    "IWM  — Russell 2000 ETF": "IWM",
    "^SPX — S&P 500 Index":    "^SPX",
}

# ── Black-Scholes ────────────────────────────
def bs_price(S, K, T, r, sigma, opt_type="call"):
    if T <= 1e-6 or sigma <= 1e-6:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calc_iv(market_price, S, K, T, r, opt_type="call"):
    if market_price <= 0 or T <= 1e-6:
        return np.nan
    fwd = S * np.exp(r * T)
    intrinsic = max(0.0, (fwd - K) if opt_type == "call" else (K - fwd))
    if market_price <= intrinsic * np.exp(-r * T) * 0.999:
        return np.nan
    def obj(sigma): return bs_price(S, K, T, r, sigma, opt_type) - market_price
    try:
        if obj(1e-4) * obj(MAX_IV) > 0: return np.nan
        iv = brentq(obj, 1e-4, MAX_IV, xtol=1e-7, maxiter=200)
        return iv if 0.005 <= iv <= MAX_IV else np.nan
    except Exception:
        return np.nan


# ── Fetch expiry list (cached 5 min) ─────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_expiry_list(sym):
    stock = yf.Ticker(sym)
    hist  = stock.history(period="2d")
    if hist.empty:
        return None, None, None
    S       = float(hist["Close"].iloc[-1])
    today   = datetime.now()
    rows    = []
    for exp in stock.options:
        exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        days   = (exp_dt - today).days
        if days < 1: continue
        rows.append({"date": exp, "days": days, "T": days / 365.0})
    return pd.DataFrame(rows), S, stock


# ── Fetch options chain ───────────────────────
def fetch_options(stock, selected_dates, S, progress_bar, status_text):
    records = []
    r       = RISK_FREE_RATE
    total   = len(selected_dates)

    for i, row in enumerate(selected_dates):
        exp = row["date"]
        T   = row["T"]
        status_text.text(f"⏳ กำลังโหลด {exp}  ({int(row['days'])} วัน)  [{i+1}/{total}]")
        progress_bar.progress((i + 1) / total)

        try:
            chain = stock.option_chain(exp)
        except Exception:
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
                K     = float(ropt["strike"])
                price = float(ropt["price"])
                mono  = np.log(K / S)
                if not (MONEYNESS_MIN <= mono <= MONEYNESS_MAX): continue
                iv_val = calc_iv(price, S, K, T, r, otype)
                if np.isnan(iv_val): continue
                records.append({
                    "strike": K, "expiry": exp, "T": T,
                    "days": int(row["days"]),
                    "moneyness": mono, "iv": iv_val, "type": otype,
                    "bid": float(ropt["bid"]), "ask": float(ropt["ask"]),
                    "volume": int(ropt["volume"]) if not pd.isna(ropt["volume"]) else 0,
                })

    return pd.DataFrame(records)


# ── Build surface ─────────────────────────────
def build_surface(df):
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

    pts  = df_c[["moneyness","T"]].values
    vals = df_c["iv"].values
    surf = griddata(pts, vals, (KK, TT), method="cubic")
    near = griddata(pts, vals, (KK, TT), method="nearest")
    surf[np.isnan(surf)] = near[np.isnan(surf)]
    return k_grid, t_grid, surf, df_c


# ── Plot ──────────────────────────────────────
def fmt_T(t):
    d = int(round(t * 365))
    return f"{d}d" if d < 30 else (f"{d//30}M" if d < 365 else f"{d/365:.1f}Y")


def plot_surface(k_grid, t_grid, iv_surface, df_raw, S, sym, name):
    mono_pct = np.exp(k_grid) * 100
    t_labels = [fmt_T(t) for t in t_grid]
    KK, TT   = np.meshgrid(mono_pct, np.arange(len(t_grid)))
    single   = df_raw["T"].nunique() == 1

    COLORS = [[0,"#143dd9"],[0.22,"#00b3cc"],[0.42,"#00cc4d"],
              [0.62,"#d9d100"],[0.80,"#f25800"],[1.00,"#cc0018"]]

    if single:
        fig = go.Figure()
        col_map = {"put": "#e74c3c", "call": "#3498db"}
        for otype in ["put", "call"]:
            sub = df_raw[df_raw["type"] == otype]
            fig.add_trace(go.Scatter(
                x=np.exp(sub["moneyness"]) * 100, y=sub["iv"] * 100,
                mode="markers",
                marker=dict(size=6, color=col_map[otype]),
                name=otype,
                text=sub.apply(lambda r: f"K={r['strike']:.0f}  IV={r['iv']*100:.1f}%", axis=1),
                hoverinfo="text",
            ))
        t_mid = len(t_grid) // 2
        smile = iv_surface[t_mid, :] * 100
        valid = ~np.isnan(smile)
        fig.add_trace(go.Scatter(
            x=mono_pct[valid], y=smile[valid],
            mode="lines", line=dict(color="#00e0ff", width=2.5),
            name="Smile (fit)",
        ))
        fig.add_vline(x=100, line=dict(color="rgba(255,255,255,0.4)", dash="dot", width=1))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
            title=f"IV Smile — {name}   Expiry: {df_raw['expiry'].iloc[0]}   Spot: {S:,.2f}",
            xaxis_title="Moneyness (%)", yaxis_title="IV (%)",
            font=dict(family="monospace", size=12, color="#c8d8f0"),
            height=500,
        )
    else:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type":"surface"},{"type":"scatter"}]],
            column_widths=[0.68, 0.32],
            subplot_titles=[f"{sym} Implied Volatility Surface", "Smile by Expiry"],
        )
        fig.add_trace(go.Surface(
            x=KK, y=TT, z=iv_surface * 100,
            colorscale=COLORS,
            colorbar=dict(title="IV (%)", x=0.65, len=0.85, thickness=12),
            opacity=0.93,
            lighting=dict(ambient=0.7, diffuse=0.85, specular=0.3),
            hovertemplate="Moneyness: %{x:.1f}%<br>IV: %{z:.1f}%<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=np.exp(df_raw["moneyness"]) * 100,
            y=[int(np.argmin(np.abs(t_grid - t))) for t in df_raw["T"]],
            z=df_raw["iv"] * 100,
            mode="markers",
            marker=dict(size=2, color="rgba(255,255,255,0.28)"),
            hovertemplate="M:%{x:.1f}%<br>IV:%{z:.1f}%<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

        clrs = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db",
                "#9b59b6","#1abc9c","#e91e63","#00bcd4","#ff5722","#8bc34a","#ff9800"]
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

        fig.add_vline(x=100, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"), row=1, col=2)

        tick_idx = np.round(np.linspace(0, len(t_grid)-1, min(7, len(t_grid)))).astype(int)
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
                aspectratio=dict(x=1.2, y=1.0, z=0.65),
            ),
        )
        fig.update_xaxes(title_text="Moneyness (%)", gridcolor="#1a2540", color="#8aadee", row=1, col=2)
        fig.update_yaxes(title_text="IV (%)", gridcolor="#1a2540", color="#8aadee", row=1, col=2)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
            title=f"IV Surface — {name}   Spot: {S:,.2f}   {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            font=dict(family="monospace", size=11, color="#c8d8f0"),
            margin=dict(l=0, r=0, t=50, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="rgba(100,150,255,0.3)",
                        borderwidth=1, font=dict(size=10)),
            height=640,
        )

    return fig


# ═══════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════
st.title("📈 IV Surface — US Index Options")
st.caption("ข้อมูลจาก Yahoo Finance · Black-Scholes Inversion · Plotly 3D")

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ การตั้งค่า")

    # 1. เลือก Underlying
    st.subheader("1️⃣ Underlying")
    ticker_label = st.selectbox("เลือก Index / ETF", list(TICKERS.keys()), index=0)
    sym  = TICKERS[ticker_label]
    name = ticker_label.split("—")[-1].strip()

    st.divider()

    # 2. โหลด expiry list
    st.subheader("2️⃣ โหลด Expiry List")
    load_list_btn = st.button("🔄 โหลดรายการ Expiry", use_container_width=True)

    if load_list_btn or "df_exp" not in st.session_state or st.session_state.get("sym") != sym:
        with st.spinner("กำลังโหลด expiry list ..."):
            df_exp, S, stock = fetch_expiry_list(sym)
        if df_exp is None or df_exp.empty:
            st.error("โหลดข้อมูลไม่ได้ — ตรวจสอบ internet")
            st.stop()
        st.session_state["df_exp"] = df_exp
        st.session_state["S"]      = S
        st.session_state["stock"]  = stock
        st.session_state["sym"]    = sym
        st.success(f"พบ {len(df_exp)} expiry dates")

    st.divider()

    # 3. เลือก Expiry
    st.subheader("3️⃣ เลือก Expiry")

    if "df_exp" in st.session_state:
        df_exp = st.session_state["df_exp"]

        mode = st.radio("โหมด", [
            "🗓️ เลือก 1 วัน",
            "📅 ช่วงวัน (range)",
            "⭐ Preset",
            "✏️ เลือกเองหลายวัน",
        ], index=1)

        selected_rows = []

        # ── Mode A: Single ─────────────────────
        if mode == "🗓️ เลือก 1 วัน":
            date_options = df_exp["date"].tolist()
            chosen = st.selectbox("เลือก Expiry Date", date_options, index=min(3, len(date_options)-1))
            row = df_exp[df_exp["date"] == chosen].iloc[0]
            selected_rows = [row.to_dict()]
            st.info(f"Expiry: **{chosen}**  ({int(row['days'])} วัน)")

        # ── Mode B: Range ──────────────────────
        elif mode == "📅 ช่วงวัน (range)":
            d_min = st.slider("วันเริ่มต้น (days)", 1, 30, 7)
            d_max = st.slider("วันสิ้นสุด (days)", 30, 730, 365)
            n_max = st.slider("จำนวน Expiry สูงสุด", 2, 20, 8)
            filtered = df_exp[(df_exp["days"] >= d_min) & (df_exp["days"] <= d_max)]
            if filtered.empty:
                st.warning("ไม่มี expiry ในช่วงนี้")
            else:
                if len(filtered) > n_max:
                    log_idx = np.round(np.linspace(0, len(filtered)-1, n_max)).astype(int)
                    filtered = filtered.iloc[log_idx]
                selected_rows = filtered.to_dict("records")
                st.success(f"เลือก {len(selected_rows)} expiry")
                for r in selected_rows:
                    st.caption(f"• {r['date']}  ({int(r['days'])} วัน)")

        # ── Mode C: Preset ─────────────────────
        elif mode == "⭐ Preset":
            PRESETS = {
                "Short-term  1W→1Y":  [7, 14, 30, 60, 90, 180, 365],
                "Near-term   1W→3M":  [7, 14, 21, 30, 45, 60, 90],
                "Medium-term 1M→2Y":  [30, 60, 90, 180, 365, 730],
                "Long-term   3M→2Y":  [90, 180, 270, 365, 540, 730],
                "Sparse 4pts 1W/1M/3M/1Y": [7, 30, 90, 365],
            }
            preset_name = st.selectbox("เลือก Preset", list(PRESETS.keys()))
            targets     = PRESETS[preset_name]
            rows = []
            for t in targets:
                sub = df_exp[(df_exp["days"] - t).abs() == (df_exp["days"] - t).abs().min()]
                if not sub.empty:
                    rows.append(sub.iloc[0].to_dict())
            seen = set()
            selected_rows = [r for r in rows if not (r["date"] in seen or seen.add(r["date"]))]
            st.success(f"เลือก {len(selected_rows)} expiry")
            for r in selected_rows:
                st.caption(f"• {r['date']}  ({int(r['days'])} วัน)")

        # ── Mode D: Manual ─────────────────────
        elif mode == "✏️ เลือกเองหลายวัน":
            date_options = df_exp["date"].tolist()
            chosen_many  = st.multiselect(
                "เลือก Expiry Dates",
                date_options,
                default=date_options[:min(6, len(date_options))],
            )
            selected_rows = df_exp[df_exp["date"].isin(chosen_many)].to_dict("records")
            st.success(f"เลือก {len(selected_rows)} expiry")

        st.divider()

        # 4. โหลดข้อมูล + คำนวณ
        st.subheader("4️⃣ โหลดข้อมูลจริง")
        st.write(f"จะโหลด **{len(selected_rows)} expiry** — ใช้เวลาประมาณ {len(selected_rows)*3}–{len(selected_rows)*6} วินาที")
        run_btn = st.button("🚀 โหลดและคำนวณ IV Surface", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────
if "df_exp" not in st.session_state:
    st.info("👈 กด **โหลดรายการ Expiry** ใน Sidebar ก่อน")
    st.stop()

S = st.session_state["S"]

# Spot price display
col1, col2, col3 = st.columns(3)
col1.metric("Underlying", sym)
col2.metric("Spot Price", f"{S:,.2f}")
col3.metric("Risk-Free Rate", f"{RISK_FREE_RATE*100:.1f}%")

# Run calculation
if "run_btn" in dir() and run_btn and selected_rows:
    stock = st.session_state["stock"]

    st.subheader("⏳ กำลังโหลดข้อมูล ...")
    prog_bar   = st.progress(0)
    status_txt = st.empty()

    df_iv = fetch_options(stock, selected_rows, S, prog_bar, status_txt)

    prog_bar.empty()
    status_txt.empty()

    if df_iv.empty:
        st.error("❌ ไม่ได้ข้อมูล — ลองเปลี่ยน expiry หรือตรวจ internet")
        st.stop()

    with st.spinner("Interpolating surface ..."):
        k_grid, t_grid, iv_surface, df_clean = build_surface(df_iv)

    st.success(f"✅ คำนวณเสร็จ — IV points: {len(df_iv):,}  |  Expiry: {df_iv['T'].nunique()}")

    # ── Summary metrics ──
    st.subheader("📊 Summary")
    atm_idx   = int(np.argmin(np.abs(k_grid)))
    sorted_Ts = sorted(df_clean["T"].unique())

    metric_cols = st.columns(min(len(sorted_Ts), 6))
    for j, t_val in enumerate(sorted_Ts[:6]):
        t_idx  = int(np.argmin(np.abs(t_grid - t_val)))
        atm_iv = iv_surface[t_idx, atm_idx]
        p_idx  = int(np.argmin(np.abs(k_grid - (-0.10))))
        put_iv = iv_surface[t_idx, p_idx]
        skew   = put_iv - atm_iv if not np.isnan(put_iv) else np.nan
        lbl    = fmt_T(t_val)
        with metric_cols[j]:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{lbl} ATM</div>
              <div class="metric-value">{atm_iv*100:.1f}%</div>
              <div class="metric-label">Skew: {f"{skew*100:+.1f}%" if not np.isnan(skew) else "N/A"}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Chart ──
    st.subheader("📈 IV Surface / Smile")
    fig = plot_surface(k_grid, t_grid, iv_surface, df_clean, S, sym, name)
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw data table ──
    with st.expander("📋 ดู Raw Data"):
        st.dataframe(
            df_iv[["expiry","type","strike","moneyness","iv","bid","ask","volume"]]
            .assign(
                moneyness=lambda x: x["moneyness"].round(4),
                iv=lambda x: (x["iv"]*100).round(2),
            )
            .rename(columns={"iv": "IV (%)", "moneyness": "log(K/S)"}),
            use_container_width=True,
        )

    # ── Download ──
    fig_html = fig.to_html(include_plotlyjs="cdn")
    st.download_button(
        label="⬇️ Download IV Surface (HTML)",
        data=fig_html,
        file_name=f"iv_surface_{sym.replace('^','')}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
        use_container_width=True,
    )

elif "df_exp" in st.session_state:
    st.info("👈 เลือก Expiry แล้วกด **โหลดและคำนวณ IV Surface**")
