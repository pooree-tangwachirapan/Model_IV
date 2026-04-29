"""
IV Surface — US Index Options  (Streamlit App)
แหล่งข้อมูล: Tradier API (ฟรี, ไม่ถูก block บน cloud)
==============================================
1. สมัครฟรีที่ https://developer.tradier.com/
2. รับ Sandbox Token (ฟรี, delayed data)
3. ใส่ Token ใน sidebar หรือ .streamlit/secrets.toml

รัน:  streamlit run iv_surface_app.py
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
import requests
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
.metric-card {
    background: #0d1a30;
    border: 1px solid #1a2e50;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label { font-size: 11px; color: #6080a0; margin-bottom: 4px; }
.metric-value { font-size: 22px; font-weight: 600; color: #e0f0ff; font-family: monospace; }
.metric-sub   { font-size: 11px; color: #8090b0; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────
RISK_FREE_RATE = 0.053
MIN_BID        = 0.05
MIN_OPEN_INT   = 0
MONEYNESS_MIN  = -0.45
MONEYNESS_MAX  = 0.35
MAX_IV         = 3.0
GRID_NK        = 60
GRID_NT        = 50

TRADIER_SANDBOX = "https://sandbox.tradier.com/v1"
TRADIER_LIVE    = "https://api.tradier.com/v1"

TICKERS = {
    "SPY  — S&P 500 ETF":      "SPY",
    "QQQ  — Nasdaq 100 ETF":   "QQQ",
    "IWM  — Russell 2000 ETF": "IWM",
    "DIA  — Dow Jones ETF":    "DIA",
}


# ── Tradier API helpers ───────────────────────
def tradier_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


@st.cache_data(ttl=120, show_spinner=False)
def tradier_get_quote(sym: str, token: str, sandbox: bool) -> float:
    base = TRADIER_SANDBOX if sandbox else TRADIER_LIVE
    url  = f"{base}/markets/quotes"
    r    = requests.get(url, headers=tradier_headers(token),
                        params={"symbols": sym, "greeks": "false"}, timeout=10)
    r.raise_for_status()
    data = r.json()
    return float(data["quotes"]["quote"]["last"])


@st.cache_data(ttl=120, show_spinner=False)
def tradier_get_expirations(sym: str, token: str, sandbox: bool) -> list[str]:
    base = TRADIER_SANDBOX if sandbox else TRADIER_LIVE
    url  = f"{base}/markets/options/expirations"
    r    = requests.get(url, headers=tradier_headers(token),
                        params={"symbol": sym, "includeAllRoots": "true"}, timeout=10)
    r.raise_for_status()
    data = r.json()
    exps = data.get("expirations", {}).get("date", [])
    if isinstance(exps, str):
        exps = [exps]
    return sorted(exps)


def tradier_get_chain(sym: str, exp: str, token: str, sandbox: bool) -> pd.DataFrame:
    base = TRADIER_SANDBOX if sandbox else TRADIER_LIVE
    url  = f"{base}/markets/options/chains"
    r    = requests.get(url, headers=tradier_headers(token),
                        params={"symbol": sym, "expiration": exp, "greeks": "false"}, timeout=15)
    r.raise_for_status()
    data    = r.json()
    options = data.get("options", {}).get("option", [])
    if not options:
        return pd.DataFrame()
    df = pd.DataFrame(options)
    return df


# ── Black-Scholes ────────────────────────────
def bs_price(S, K, T, r, sigma, opt_type="call"):
    if T <= 1e-6 or sigma <= 1e-6:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calc_iv(price, S, K, T, r, opt_type="call"):
    if price <= 0 or T <= 1e-6:
        return np.nan
    fwd       = S * np.exp(r * T)
    intrinsic = max(0.0, (fwd - K) if opt_type == "call" else (K - fwd))
    if price <= intrinsic * np.exp(-r * T) * 0.999:
        return np.nan
    def obj(sig): return bs_price(S, K, T, r, sig, opt_type) - price
    try:
        if obj(1e-4) * obj(MAX_IV) > 0: return np.nan
        iv = brentq(obj, 1e-4, MAX_IV, xtol=1e-7, maxiter=200)
        return iv if 0.005 <= iv <= MAX_IV else np.nan
    except Exception:
        return np.nan


# ── Fetch options for selected expiries ───────
def fetch_options(sym, selected_rows, S, token, sandbox, prog, status):
    records = []
    r       = RISK_FREE_RATE
    total   = len(selected_rows)

    for i, row in enumerate(selected_rows):
        exp  = row["date"]
        T    = row["T"]
        days = int(row["days"])
        status.text(f"⏳ {exp}  ({days} วัน)  [{i+1}/{total}]")
        prog.progress((i + 1) / total)

        try:
            df_chain = tradier_get_chain(sym, exp, token, sandbox)
        except Exception as e:
            st.warning(f"⚠️ {exp}: {e}")
            continue

        if df_chain.empty:
            continue

        for otype in ["call", "put"]:
            sub = df_chain[df_chain["option_type"] == otype].copy()
            if sub.empty:
                continue

            # mid price
            sub["mid"] = (pd.to_numeric(sub["bid"], errors="coerce") +
                          pd.to_numeric(sub["ask"], errors="coerce")) / 2.0
            sub["price"] = np.where(
                sub["mid"].fillna(0) > 0, sub["mid"],
                pd.to_numeric(sub.get("last", 0), errors="coerce").fillna(0)
            )

            mask = (
                pd.to_numeric(sub["bid"], errors="coerce").fillna(0) >= MIN_BID
            ) & (sub["price"] > 0)
            sub = sub[mask]

            for _, ropt in sub.iterrows():
                try:
                    K     = float(ropt["strike"])
                    price = float(ropt["price"])
                    mono  = np.log(K / S)
                    if not (MONEYNESS_MIN <= mono <= MONEYNESS_MAX):
                        continue
                    iv_val = calc_iv(price, S, K, T, r, otype)
                    if np.isnan(iv_val):
                        continue
                    records.append({
                        "strike":    K,
                        "expiry":    exp,
                        "T":         T,
                        "days":      days,
                        "moneyness": mono,
                        "iv":        iv_val,
                        "type":      otype,
                        "bid":       float(ropt.get("bid", 0) or 0),
                        "ask":       float(ropt.get("ask", 0) or 0),
                        "volume":    int(float(ropt.get("volume", 0) or 0)),
                        "open_interest": int(float(ropt.get("open_interest", 0) or 0)),
                    })
                except Exception:
                    continue

    return pd.DataFrame(records)


# ── Build surface ─────────────────────────────
def build_surface(df):
    clean = []
    for _, grp in df.groupby("expiry"):
        q1, q3 = grp["iv"].quantile([0.05, 0.95])
        iqr = q3 - q1
        filtered = grp[(grp["iv"] >= q1 - 1.5*iqr) & (grp["iv"] <= q3 + 1.5*iqr)]
        if len(filtered) >= 3:
            clean.append(filtered)
    if not clean:
        return None, None, None, df
    df_c = pd.concat(clean)

    k_grid = np.linspace(MONEYNESS_MIN*0.9, MONEYNESS_MAX*0.9, GRID_NK)
    t_log  = np.linspace(np.log(df_c["T"].min()), np.log(df_c["T"].max()), GRID_NT)
    t_grid = np.exp(t_log)
    KK, TT = np.meshgrid(k_grid, t_grid)

    pts  = df_c[["moneyness", "T"]].values
    vals = df_c["iv"].values
    surf = griddata(pts, vals, (KK, TT), method="cubic")
    near = griddata(pts, vals, (KK, TT), method="nearest")
    surf[np.isnan(surf)] = near[np.isnan(surf)]
    return k_grid, t_grid, surf, df_c


# ── Plot ──────────────────────────────────────
def fmt_T(t):
    d = int(round(t * 365))
    return f"{d}d" if d < 30 else (f"{d//30}M" if d < 365 else f"{d/365:.1f}Y")


SURF_COLORS = [
    [0.00, "#143dd9"], [0.22, "#00b3cc"], [0.42, "#00cc4d"],
    [0.62, "#d9d100"], [0.80, "#f25800"], [1.00, "#cc0018"],
]
LINE_COLORS = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db",
               "#9b59b6","#1abc9c","#e91e63","#00bcd4","#ff5722",
               "#8bc34a","#ff9800"]


def plot_surface(k_grid, t_grid, iv_surface, df_raw, S, sym, name):
    mono_pct = np.exp(k_grid) * 100
    t_labels = [fmt_T(t) for t in t_grid]
    KK, TT   = np.meshgrid(mono_pct, np.arange(len(t_grid)))
    single   = df_raw["T"].nunique() == 1

    if single:
        fig = go.Figure()
        for otype, col in [("put", "#e74c3c"), ("call", "#3498db")]:
            sub = df_raw[df_raw["type"] == otype]
            fig.add_trace(go.Scatter(
                x=np.exp(sub["moneyness"]) * 100, y=sub["iv"] * 100,
                mode="markers",
                marker=dict(size=6, color=col),
                name=otype,
                text=sub.apply(lambda r: f"K={r['strike']:.0f}  IV={r['iv']*100:.1f}%  Vol={r['volume']}", axis=1),
                hoverinfo="text",
            ))
        t_mid  = len(t_grid) // 2
        smile  = iv_surface[t_mid, :] * 100
        valid  = ~np.isnan(smile)
        fig.add_trace(go.Scatter(
            x=mono_pct[valid], y=smile[valid],
            mode="lines", line=dict(color="#00e0ff", width=2.5), name="Smile fit",
        ))
        fig.add_vline(x=100, line=dict(color="rgba(255,255,255,0.4)", dash="dot", width=1))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
            title=f"IV Smile — {name}  Expiry: {df_raw['expiry'].iloc[0]}  Spot: {S:,.2f}",
            xaxis_title="Moneyness (%)", yaxis_title="IV (%)",
            font=dict(family="monospace", size=12, color="#c8d8f0"),
            height=500,
        )
        return fig

    # Multi-expiry surface
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "scatter"}]],
        column_widths=[0.68, 0.32],
        subplot_titles=[f"{sym} Implied Volatility Surface", "Smile by Expiry"],
    )

    fig.add_trace(go.Surface(
        x=KK, y=TT, z=iv_surface * 100,
        colorscale=SURF_COLORS,
        colorbar=dict(title="IV (%)", x=0.64, len=0.85, thickness=12),
        opacity=0.93,
        lighting=dict(ambient=0.7, diffuse=0.85, specular=0.3),
        hovertemplate="Moneyness: %{x:.1f}%<br>IV: %{z:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=np.exp(df_raw["moneyness"]) * 100,
        y=[int(np.argmin(np.abs(t_grid - t))) for t in df_raw["T"]],
        z=df_raw["iv"] * 100,
        mode="markers",
        marker=dict(size=2, color="rgba(255,255,255,0.25)"),
        hovertemplate="M:%{x:.1f}%<br>IV:%{z:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    for j, t_val in enumerate(sorted(df_raw["T"].unique())):
        t_idx = int(np.argmin(np.abs(t_grid - t_val)))
        smile = iv_surface[t_idx, :] * 100
        valid = ~np.isnan(smile)
        fig.add_trace(go.Scatter(
            x=mono_pct[valid], y=smile[valid],
            mode="lines", name=fmt_T(t_val),
            line=dict(color=LINE_COLORS[j % len(LINE_COLORS)], width=2),
            hovertemplate="M:%{x:.1f}%<br>IV:%{y:.1f}%<extra></extra>",
        ), row=1, col=2)

    fig.add_vline(x=100, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"), row=1, col=2)

    tick_idx = np.round(np.linspace(0, len(t_grid)-1, min(7, len(t_grid)))).astype(int)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Moneyness (%)", gridcolor="#1a2540", color="#8aadee"),
            yaxis=dict(
                title="Expiry",
                tickvals=list(tick_idx),
                ticktext=[t_labels[i] for i in tick_idx],
                gridcolor="#1a2540", color="#8aadee",
            ),
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
        title=f"IV Surface — {name}  Spot: {S:,.2f}  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        font=dict(family="monospace", size=11, color="#c8d8f0"),
        margin=dict(l=0, r=0, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="rgba(100,150,255,0.3)",
                    borderwidth=1, font=dict(size=10)),
        height=650,
    )
    return fig


# ════════════════════════════════════════════════
# STREAMLIT UI
# ════════════════════════════════════════════════
st.title("📈 IV Surface — US Index Options")
st.caption("ข้อมูลจาก Tradier API · Black-Scholes Inversion · Plotly 3D Interactive")

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ ตั้งค่า")

    # ── API Token ──────────────────────────────
    st.subheader("🔑 Tradier API Token")

    # อ่านจาก secrets ถ้ามี
    default_token = ""
    try:
        default_token = st.secrets.get("TRADIER_TOKEN", "")
    except Exception:
        pass

    use_sandbox = st.toggle("ใช้ Sandbox (delayed, ฟรี)", value=True)

    token_input = st.text_input(
        "Token",
        value=default_token,
        type="password",
        placeholder="ใส่ Tradier API Token",
        help="สมัครฟรีที่ developer.tradier.com → รับ Sandbox Token",
    )

    if not token_input:
        st.warning("⚠️ ต้องใส่ API Token ก่อน")
        st.markdown("""
        **วิธีรับ Token ฟรี:**
        1. ไปที่ [developer.tradier.com](https://developer.tradier.com/)
        2. กด **Get an API Key**
        3. สมัครฟรี (ไม่ต้องบัตรเครดิต)
        4. คัดลอก **Sandbox Token**
        5. วางใน field ด้านบน
        """)
        st.stop()

    token   = token_input
    sandbox = use_sandbox
    st.divider()

    # ── เลือก Underlying ───────────────────────
    st.subheader("1️⃣ เลือก Underlying")
    ticker_label = st.selectbox("Index / ETF", list(TICKERS.keys()))
    sym  = TICKERS[ticker_label]
    name = ticker_label.split("—")[-1].strip()

    st.divider()

    # ── โหลด Expiry list ───────────────────────
    st.subheader("2️⃣ โหลด Expiry List")
    load_btn = st.button("🔄 โหลดรายการ Expiry", use_container_width=True)

    cache_key = f"{sym}_{sandbox}"
    if load_btn or st.session_state.get("cache_key") != cache_key:
        with st.spinner("โหลด expiry dates ..."):
            try:
                S    = tradier_get_quote(sym, token, sandbox)
                exps = tradier_get_expirations(sym, token, sandbox)
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

        today = datetime.now()
        rows  = []
        for exp in exps:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            days   = (exp_dt - today).days
            if days < 1: continue
            rows.append({"date": exp, "days": days, "T": days / 365.0})

        st.session_state["df_exp"]    = pd.DataFrame(rows)
        st.session_state["S"]         = S
        st.session_state["cache_key"] = cache_key
        st.success(f"✅ พบ {len(rows)} expiry")

    st.divider()

    # ── เลือก Expiry ───────────────────────────
    selected_rows = []

    if "df_exp" in st.session_state:
        df_exp = st.session_state["df_exp"]
        S      = st.session_state["S"]

        st.subheader("3️⃣ เลือก Expiry")
        mode = st.radio("โหมด", [
            "🗓️ เลือก 1 วัน",
            "📅 ช่วงวัน",
            "⭐ Preset",
            "✏️ เลือกหลายวัน",
        ], index=1)

        if mode == "🗓️ เลือก 1 วัน":
            opts   = df_exp["date"].tolist()
            chosen = st.selectbox("Expiry Date", opts, index=min(2, len(opts)-1))
            row    = df_exp[df_exp["date"] == chosen].iloc[0]
            selected_rows = [row.to_dict()]
            st.info(f"{chosen}  ({int(row['days'])} วัน)")

        elif mode == "📅 ช่วงวัน":
            c1, c2 = st.columns(2)
            d_min  = c1.number_input("วันเริ่มต้น", 1, 60, 7)
            d_max  = c2.number_input("วันสิ้นสุด", 30, 730, 365)
            n_max  = st.slider("จำนวน Expiry สูงสุด", 2, 15, 8)
            filt   = df_exp[(df_exp["days"] >= d_min) & (df_exp["days"] <= d_max)]
            if filt.empty:
                st.warning("ไม่มี expiry ในช่วงนี้")
            else:
                if len(filt) > n_max:
                    idx  = np.round(np.linspace(0, len(filt)-1, n_max)).astype(int)
                    filt = filt.iloc[idx]
                selected_rows = filt.to_dict("records")
                st.success(f"เลือก {len(selected_rows)} expiry")
                with st.expander("ดูรายละเอียด"):
                    for r in selected_rows:
                        st.caption(f"• {r['date']}  ({int(r['days'])} วัน / {fmt_T(r['T'])})")

        elif mode == "⭐ Preset":
            PRESETS = {
                "Short-term 1W→1Y":      [7, 14, 30, 60, 90, 180, 365],
                "Near-term  1W→3M":      [7, 14, 21, 30, 45, 60, 90],
                "Medium-term 1M→2Y":     [30, 60, 90, 180, 365, 730],
                "Long-term  3M→2Y":      [90, 180, 270, 365, 540, 730],
                "Sparse 4pts 1W/1M/3M/1Y": [7, 30, 90, 365],
            }
            preset = st.selectbox("เลือก Preset", list(PRESETS.keys()))
            rows_p = []
            for t in PRESETS[preset]:
                sub = df_exp.iloc[((df_exp["days"] - t).abs()).argsort()[:1]]
                if not sub.empty:
                    rows_p.append(sub.iloc[0].to_dict())
            seen = set()
            selected_rows = [r for r in rows_p if not (r["date"] in seen or seen.add(r["date"]))]
            st.success(f"เลือก {len(selected_rows)} expiry")
            with st.expander("ดูรายละเอียด"):
                for r in selected_rows:
                    st.caption(f"• {r['date']}  ({int(r['days'])} วัน / {fmt_T(r['T'])})")

        elif mode == "✏️ เลือกหลายวัน":
            all_dates = df_exp["date"].tolist()
            chosen_m  = st.multiselect(
                "เลือก Expiry Dates",
                all_dates,
                default=all_dates[:min(5, len(all_dates))],
            )
            selected_rows = df_exp[df_exp["date"].isin(chosen_m)].to_dict("records")
            st.success(f"เลือก {len(selected_rows)} expiry")

        st.divider()
        st.subheader("4️⃣ คำนวณ")
        est = len(selected_rows) * 4
        if selected_rows:
            st.caption(f"ประมาณ {est}–{est*2} วินาที")
        run_btn = st.button(
            "🚀 โหลดและคำนวณ IV Surface",
            type="primary",
            use_container_width=True,
            disabled=len(selected_rows) == 0,
        )

# ── Main panel ────────────────────────────────
if "S" not in st.session_state:
    st.info("👈 ใส่ **API Token** และกด **โหลดรายการ Expiry** ใน sidebar")
    with st.expander("📖 วิธีรับ Tradier Token ฟรี"):
        st.markdown("""
        Yahoo Finance ถูก block บน Streamlit Cloud
        จึงเปลี่ยนมาใช้ **Tradier API** ซึ่งฟรีและใช้ได้บน cloud ครับ

        **ขั้นตอน (ใช้เวลา ~2 นาที):**
        1. ไปที่ https://developer.tradier.com/
        2. กด **Get an API Key** → Register ฟรี
        3. เข้า Dashboard → คัดลอก **Sandbox Token**
        4. วาง Token ใน Sidebar

        **Sandbox vs Live:**
        - **Sandbox** = ฟรี, ข้อมูล delayed 15 min, เหมาะ dev/test
        - **Live** = ต้องมี Tradier brokerage account
        """)
    st.stop()

S = st.session_state["S"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Underlying", sym)
col2.metric("Spot Price", f"{S:,.2f}")
col3.metric("Mode", "Sandbox" if sandbox else "Live")
col4.metric("Risk-Free Rate", f"{RISK_FREE_RATE*100:.1f}%")

# ── Run ───────────────────────────────────────
if "run_btn" in dir() and run_btn and selected_rows:
    st.subheader("⏳ กำลังโหลดข้อมูล Options ...")
    prog   = st.progress(0)
    status = st.empty()

    df_iv = fetch_options(sym, selected_rows, S, token, sandbox, prog, status)

    prog.empty()
    status.empty()

    if df_iv.empty:
        st.error("❌ ไม่ได้ข้อมูลเลย — ตรวจสอบ Token หรือเลือก Expiry อื่น")
        st.stop()

    with st.spinner("Interpolating IV surface ..."):
        k_grid, t_grid, iv_surface, df_clean = build_surface(df_iv)

    if k_grid is None:
        st.error("ข้อมูลไม่พอสร้าง surface — ลองเพิ่ม expiry")
        st.stop()

    st.success(f"✅ เสร็จ — IV points: {len(df_iv):,}  |  Expiry: {df_iv['T'].nunique()}  |  Strikes: {df_iv['strike'].nunique()}")

    # Summary metrics
    st.subheader("📊 ATM IV Summary")
    atm_idx   = int(np.argmin(np.abs(k_grid)))
    sorted_Ts = sorted(df_clean["T"].unique())
    n_cols    = min(len(sorted_Ts), 6)
    metric_cols = st.columns(n_cols)

    for j, t_val in enumerate(sorted_Ts[:n_cols]):
        t_idx  = int(np.argmin(np.abs(t_grid - t_val)))
        atm_iv = iv_surface[t_idx, atm_idx]
        p_idx  = int(np.argmin(np.abs(k_grid - (-0.10))))
        put_iv = iv_surface[t_idx, p_idx]
        skew   = put_iv - atm_iv if not np.isnan(put_iv) else np.nan
        lbl    = fmt_T(t_val)
        with metric_cols[j]:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{lbl} ATM IV</div>
              <div class="metric-value">{atm_iv*100:.1f}%</div>
              <div class="metric-sub">Skew: {f"{skew*100:+.1f}%" if not np.isnan(skew) else "N/A"}</div>
            </div>
            """, unsafe_allow_html=True)

    # Chart
    st.subheader("📈 IV Surface")
    fig = plot_surface(k_grid, t_grid, iv_surface, df_clean, S, sym, name)
    st.plotly_chart(fig, use_container_width=True)

    # Raw data
    with st.expander("📋 Raw Data Table"):
        show_df = df_iv[["expiry","type","strike","moneyness","iv","bid","ask","volume","open_interest"]].copy()
        show_df["moneyness"] = show_df["moneyness"].round(4)
        show_df["iv_pct"]    = (show_df["iv"] * 100).round(2)
        show_df = show_df.drop(columns=["iv"]).rename(columns={"iv_pct": "IV (%)", "moneyness": "log(K/S)"})
        st.dataframe(show_df, use_container_width=True, height=300)

    # Download
    fig_html = fig.to_html(include_plotlyjs="cdn")
    st.download_button(
        "⬇️ Download IV Surface (HTML)",
        data=fig_html,
        file_name=f"iv_surface_{sym}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
        use_container_width=True,
    )

elif "df_exp" in st.session_state:
    st.info("👈 เลือก Expiry แล้วกด **🚀 โหลดและคำนวณ IV Surface**")
