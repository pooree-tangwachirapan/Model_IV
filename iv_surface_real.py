"""
IV Surface — US Index Options
แหล่งข้อมูล: CBOE Public Delayed API (ฟรี ไม่ต้องสมัคร ไม่ต้อง Key)
Endpoint: https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json

รัน: streamlit run iv_surface_app.py
ติดตั้ง: pip install streamlit numpy pandas scipy plotly requests
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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
CBOE_BASE = "https://cdn.cboe.com/api/global/delayed_quotes/options"

CBOE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":    "https://www.cboe.com/",
    "Accept":     "application/json",
}

# CBOE symbol format: ETF = SPY, Index = _SPX _NDX _RUT
TICKERS = {
    "SPY  — S&P 500 ETF":       "SPY",
    "QQQ  — Nasdaq 100 ETF":    "QQQ",
    "IWM  — Russell 2000 ETF":  "IWM",
    "SPX  — S&P 500 Index":     "_SPX",
    "NDX  — Nasdaq 100 Index":  "_NDX",
    "RUT  — Russell 2000 Index":"_RUT",
}

MONEYNESS_MIN = -0.50
MONEYNESS_MAX = 0.40
GRID_NK = 60
GRID_NT = 50


# ── CBOE Fetch ────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def cboe_fetch_all(sym: str) -> tuple[pd.DataFrame, float]:
    """
    ดึง options chain ทั้งหมดจาก CBOE CDN
    คืน (DataFrame ทุก option, spot price)
    CBOE return IV มาให้แล้ว ไม่ต้องคำนวณเอง
    """
    url = f"{CBOE_BASE}/{sym}.json"
    r   = requests.get(url, headers=CBOE_HEADERS, timeout=15)
    r.raise_for_status()
    raw = r.json()

    data  = raw.get("data", {})
    S     = float(data.get("current_price", 0))
    opts  = data.get("options", [])

    if not opts:
        return pd.DataFrame(), S

    df = pd.DataFrame(opts)
    return df, S


def parse_options(df_raw: pd.DataFrame, S: float) -> pd.DataFrame:
    """
    แปลง raw CBOE data → DataFrame พร้อมใช้
    CBOE columns: option, bid, ask, iv, expiration, strike, option_type
    """
    df = df_raw.copy()

    # rename columns ให้สม่ำเสมอ
    rename = {
        "expiration":  "expiry",
        "option_type": "type",
        "iv":          "iv_raw",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # แปลง type: C → call, P → put
    if "type" in df.columns:
        df["type"] = df["type"].map({"C": "call", "P": "put", "call": "call", "put": "put"})

    df["strike"] = pd.to_numeric(df.get("strike", np.nan), errors="coerce")
    df["bid"]    = pd.to_numeric(df.get("bid",    0),     errors="coerce").fillna(0)
    df["ask"]    = pd.to_numeric(df.get("ask",    0),     errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df.get("volume", 0),     errors="coerce").fillna(0).astype(int)
    df["iv"]     = pd.to_numeric(df.get("iv_raw", np.nan), errors="coerce")

    # CBOE IV อาจเป็น 0-1 หรือ 0-100 → normalize เป็น 0-1
    if df["iv"].max() > 5:
        df["iv"] = df["iv"] / 100.0

    # expiry → T (years)
    today = datetime.now()
    df["expiry"] = df["expiry"].astype(str)
    df["T"]    = df["expiry"].apply(
        lambda x: max((datetime.strptime(x[:10], "%Y-%m-%d") - today).days, 0) / 365.0
    )
    df["days"] = (df["T"] * 365).round().astype(int)

    # moneyness = log(K/S)
    df["moneyness"] = np.log(df["strike"] / S)

    # filter
    df = df[
        (df["iv"] > 0.001) & (df["iv"] < 5.0) &
        (df["days"] > 0) &
        (df["moneyness"] >= MONEYNESS_MIN) &
        (df["moneyness"] <= MONEYNESS_MAX) &
        (df["bid"] > 0)
    ].dropna(subset=["strike", "iv", "expiry", "type"])

    return df.reset_index(drop=True)


# ── Build surface ─────────────────────────────
def build_surface(df: pd.DataFrame):
    clean = []
    for _, grp in df.groupby("expiry"):
        if len(grp) < 4:
            continue
        q1, q3 = grp["iv"].quantile([0.05, 0.95])
        iqr    = q3 - q1
        filtered = grp[(grp["iv"] >= q1 - 1.5*iqr) & (grp["iv"] <= q3 + 1.5*iqr)]
        if len(filtered) >= 3:
            clean.append(filtered)

    if not clean:
        return None, None, None, df

    df_c   = pd.concat(clean)
    k_grid = np.linspace(df_c["moneyness"].quantile(0.02),
                         df_c["moneyness"].quantile(0.98), GRID_NK)
    t_log  = np.linspace(np.log(df_c["T"].min()), np.log(df_c["T"].max()), GRID_NT)
    t_grid = np.exp(t_log)

    KK, TT = np.meshgrid(k_grid, t_grid)
    pts    = df_c[["moneyness", "T"]].values
    vals   = df_c["iv"].values

    surf = griddata(pts, vals, (KK, TT), method="cubic")
    near = griddata(pts, vals, (KK, TT), method="nearest")
    surf[np.isnan(surf)] = near[np.isnan(surf)]
    return k_grid, t_grid, surf, df_c


# ── Helpers ───────────────────────────────────
def fmt_T(t: float) -> str:
    d = int(round(t * 365))
    return f"{d}d" if d < 30 else (f"{d//30}M" if d < 365 else f"{d/365:.1f}Y")


SURF_COLORS = [
    [0.00, "#143dd9"], [0.22, "#00b3cc"], [0.42, "#00cc4d"],
    [0.62, "#d9d100"], [0.80, "#f25800"], [1.00, "#cc0018"],
]
LINE_COLORS = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db",
               "#9b59b6","#1abc9c","#e91e63","#00bcd4","#ff5722","#8bc34a","#ff9800"]


# ── Plot ──────────────────────────────────────
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
                x=np.exp(sub["moneyness"]) * 100,
                y=sub["iv"] * 100,
                mode="markers",
                marker=dict(size=6, color=col),
                name=otype,
                text=sub.apply(
                    lambda r: f"K={r['strike']:.0f}  IV={r['iv']*100:.1f}%  Vol={r['volume']}", axis=1),
                hoverinfo="text",
            ))
        t_mid = len(t_grid) // 2
        smile = iv_surface[t_mid, :] * 100
        valid = ~np.isnan(smile)
        fig.add_trace(go.Scatter(
            x=mono_pct[valid], y=smile[valid],
            mode="lines", line=dict(color="#00e0ff", width=2.5), name="Smile fit",
        ))
        fig.add_vline(x=100, line=dict(color="rgba(255,255,255,0.4)", dash="dot", width=1))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
            title=f"IV Smile — {name}  Expiry: {df_raw['expiry'].iloc[0]}  Spot: {S:,.2f}",
            xaxis_title="Moneyness (%)", yaxis_title="Implied Volatility (%)",
            font=dict(family="monospace", size=12, color="#c8d8f0"),
            height=520,
        )
        return fig

    # ── 3D Surface ──
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "scatter"}]],
        column_widths=[0.67, 0.33],
        subplot_titles=[f"{sym} Implied Volatility Surface", "Smile by Expiry"],
    )

    fig.add_trace(go.Surface(
        x=KK, y=TT, z=iv_surface * 100,
        colorscale=SURF_COLORS,
        colorbar=dict(title="IV (%)", x=0.63, len=0.85, thickness=12),
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
    fig.update_yaxes(title_text="IV (%)",         gridcolor="#1a2540", color="#8aadee", row=1, col=2)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
        title=f"IV Surface — {name}  Spot: {S:,.2f}  {datetime.now().strftime('%Y-%m-%d %H:%M')}  [CBOE Delayed]",
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
st.caption("ข้อมูลจาก **CBOE Public API** (Delayed 15 min) · ไม่ต้องสมัคร ไม่ต้อง Key")

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ การตั้งค่า")

    # 1. เลือก Underlying
    st.subheader("1️⃣ เลือก Underlying")
    ticker_label = st.selectbox("Index / ETF", list(TICKERS.keys()))
    sym  = TICKERS[ticker_label]
    name = ticker_label.split("—")[-1].strip()

    st.divider()

    # 2. โหลด Expiry List
    st.subheader("2️⃣ โหลด Expiry List")
    st.caption("ดึงข้อมูลจาก CBOE โดยตรง · delayed 15 min · ฟรี")
    load_btn = st.button("🔄 โหลดข้อมูล", use_container_width=True)

    if load_btn or st.session_state.get("loaded_sym") != sym:
        with st.spinner(f"กำลังดึงข้อมูล {sym} จาก CBOE ..."):
            try:
                df_raw_all, S = cboe_fetch_all(sym)
            except requests.exceptions.HTTPError as e:
                st.error(f"CBOE API Error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.stop()

        if df_raw_all.empty or S == 0:
            st.error("ไม่ได้ข้อมูล — ลองกด Load ใหม่")
            st.stop()

        df_parsed = parse_options(df_raw_all, S)

        # สร้าง expiry list
        exp_info = (
            df_parsed.groupby("expiry")
            .agg(days=("days", "first"), T=("T", "first"), n=("iv", "count"))
            .reset_index()
            .sort_values("days")
        )

        st.session_state["df_parsed"]  = df_parsed
        st.session_state["exp_info"]   = exp_info
        st.session_state["S"]          = S
        st.session_state["loaded_sym"] = sym
        st.success(f"✅ {sym}  Spot: {S:,.2f}  |  {len(exp_info)} expiry  |  {len(df_parsed):,} options")

    st.divider()

    # 3. เลือก Expiry
    selected_dates = []

    if "exp_info" in st.session_state:
        exp_info = st.session_state["exp_info"]

        st.subheader("3️⃣ เลือก Expiry")
        mode = st.radio("โหมด", [
            "🗓️ เลือก 1 วัน",
            "📅 ช่วงวัน",
            "⭐ Preset",
            "✏️ เลือกหลายวัน",
        ], index=1)

        # ── Mode A: Single ──────────────────────
        if mode == "🗓️ เลือก 1 วัน":
            opts   = exp_info["expiry"].tolist()
            labels = [f"{r['expiry']}  ({int(r['days'])}d / {fmt_T(r['T'])})" for _, r in exp_info.iterrows()]
            idx    = st.selectbox("Expiry", range(len(opts)), format_func=lambda i: labels[i],
                                  index=min(2, len(opts)-1))
            selected_dates = [exp_info.iloc[idx]["expiry"]]
            r = exp_info.iloc[idx]
            st.info(f"{r['expiry']}  |  {int(r['days'])} วัน  |  {int(r['n'])} options")

        # ── Mode B: Range ───────────────────────
        elif mode == "📅 ช่วงวัน":
            d_min = st.number_input("วันเริ่มต้น", 1, 60, 7)
            d_max = st.number_input("วันสิ้นสุด", 30, 730, 365)
            n_max = st.slider("จำนวน Expiry สูงสุด", 2, 20, 10)

            filt = exp_info[(exp_info["days"] >= d_min) & (exp_info["days"] <= d_max)]
            if filt.empty:
                st.warning("ไม่มี expiry ในช่วงนี้")
            else:
                if len(filt) > n_max:
                    idx  = np.round(np.linspace(0, len(filt)-1, n_max)).astype(int)
                    filt = filt.iloc[idx]
                selected_dates = filt["expiry"].tolist()
                st.success(f"เลือก {len(selected_dates)} expiry")
                with st.expander("ดูรายการ"):
                    for _, r in filt.iterrows():
                        st.caption(f"• {r['expiry']}  ({int(r['days'])}d / {fmt_T(r['T'])} / {int(r['n'])} options)")

        # ── Mode C: Preset ──────────────────────
        elif mode == "⭐ Preset":
            PRESETS = {
                "Short-term  1W→1Y":      [7, 14, 30, 60, 90, 180, 365],
                "Near-term   1W→3M":      [7, 14, 21, 30, 45, 60, 90],
                "Medium-term 1M→2Y":      [30, 60, 90, 180, 365, 730],
                "Long-term   3M→2Y":      [90, 180, 270, 365, 540, 730],
                "Sparse 1W/1M/3M/1Y":     [7, 30, 90, 365],
            }
            preset = st.selectbox("เลือก Preset", list(PRESETS.keys()))
            rows_p = []
            for t in PRESETS[preset]:
                sub = exp_info.iloc[((exp_info["days"] - t).abs()).argsort()[:1]]
                if not sub.empty:
                    rows_p.append(sub.iloc[0]["expiry"])
            selected_dates = list(dict.fromkeys(rows_p))  # deduplicate
            st.success(f"เลือก {len(selected_dates)} expiry")
            with st.expander("ดูรายการ"):
                for d in selected_dates:
                    r = exp_info[exp_info["expiry"] == d].iloc[0]
                    st.caption(f"• {d}  ({int(r['days'])}d / {fmt_T(r['T'])})")

        # ── Mode D: Manual ──────────────────────
        elif mode == "✏️ เลือกหลายวัน":
            all_labels = {
                f"{r['expiry']}  ({int(r['days'])}d)": r["expiry"]
                for _, r in exp_info.iterrows()
            }
            defaults = list(all_labels.keys())[:min(6, len(all_labels))]
            chosen = st.multiselect("เลือก Expiry", list(all_labels.keys()), default=defaults)
            selected_dates = [all_labels[c] for c in chosen]
            st.success(f"เลือก {len(selected_dates)} expiry")

        st.divider()
        st.subheader("4️⃣ แสดงผล")
        run_btn = st.button(
            "🚀 สร้าง IV Surface",
            type="primary",
            use_container_width=True,
            disabled=len(selected_dates) == 0,
        )

# ── Main panel ────────────────────────────────
if "S" not in st.session_state:
    st.info("👈 กด **โหลดข้อมูล** ใน Sidebar เพื่อเริ่ม")
    st.stop()

S = st.session_state["S"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Underlying", sym)
c2.metric("Spot Price", f"{S:,.2f}")
c3.metric("Data Source", "CBOE Delayed")
c4.metric("Expiry Available", len(st.session_state.get("exp_info", [])))

# ── Build & Plot ──────────────────────────────
if "run_btn" in dir() and run_btn and selected_dates:
    df_parsed = st.session_state["df_parsed"]

    # filter เฉพาะ expiry ที่เลือก
    df_sel = df_parsed[df_parsed["expiry"].isin(selected_dates)].copy()

    if df_sel.empty:
        st.error("ไม่มีข้อมูลสำหรับ expiry ที่เลือก")
        st.stop()

    with st.spinner("กำลัง Interpolate IV Surface ..."):
        k_grid, t_grid, iv_surface, df_clean = build_surface(df_sel)

    if k_grid is None:
        st.error("ข้อมูลไม่เพียงพอ — ลองเพิ่ม expiry")
        st.stop()

    st.success(f"✅ IV points: {len(df_sel):,}  |  Expiry: {df_sel['T'].nunique()}  |  Strike range: {df_sel['strike'].min():.0f}–{df_sel['strike'].max():.0f}")

    # ── Metric cards ──
    st.subheader("📊 ATM Implied Volatility")
    atm_idx   = int(np.argmin(np.abs(k_grid)))
    sorted_Ts = sorted(df_clean["T"].unique())
    n_cols    = min(len(sorted_Ts), 6)
    cols      = st.columns(n_cols)

    for j, t_val in enumerate(sorted_Ts[:n_cols]):
        t_idx  = int(np.argmin(np.abs(t_grid - t_val)))
        atm_iv = iv_surface[t_idx, atm_idx]
        p_idx  = int(np.argmin(np.abs(k_grid - (-0.10))))
        put_iv = iv_surface[t_idx, p_idx]
        skew   = put_iv - atm_iv if not np.isnan(put_iv) else np.nan
        with cols[j]:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{fmt_T(t_val)} ATM</div>
              <div class="metric-value">{atm_iv*100:.1f}%</div>
              <div class="metric-sub">Skew {f"{skew*100:+.1f}%" if not np.isnan(skew) else "N/A"}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Chart ──
    st.subheader("📈 IV Surface")
    fig = plot_surface(k_grid, t_grid, iv_surface, df_clean, S, sym, name)
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw data ──
    with st.expander("📋 Raw Data"):
        show = df_sel[["expiry","type","strike","moneyness","iv","bid","ask","volume"]].copy()
        show["iv_%"]      = (show["iv"] * 100).round(2)
        show["moneyness"] = show["moneyness"].round(4)
        st.dataframe(show.drop(columns=["iv"]), use_container_width=True, height=300)

    # ── Download ──
    html = fig.to_html(include_plotlyjs="cdn")
    st.download_button(
        "⬇️ Download IV Surface (HTML)",
        data=html,
        file_name=f"iv_surface_{sym}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
        use_container_width=True,
    )

elif "df_parsed" in st.session_state:
    st.info("👈 เลือก Expiry แล้วกด **🚀 สร้าง IV Surface**")
