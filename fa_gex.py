"""
fa_gex.py — GEX Analysis + FlashAlpha API client + Data Comparison
ใช้คู่กับ iv_surface_real.py (Model_IV)

GEX formula (dealer convention — ตาม Institutional Options Analysis methodology):
    GEX_strike = Gamma × OI × 100 × Spot² × 0.01     [USD notional ต่อการขยับ 1%]
    Call GEX = บวก / Put GEX = ลบ  (สมมติฐาน: dealer long call gamma, short put gamma)
    Net GEX     = Σ call_gex + Σ put_gex
    Gamma Flip  = spot ที่ net GEX (reprice gamma ทุกระดับ) ข้ามศูนย์ — ดู gamma_profile()
    Call Wall   = strike ที่ call_gex สูงสุด
    Put Wall    = strike ที่ |put_gex| สูงสุด

FlashAlpha API (https://lab.flashalpha.com):
    auth: header X-Api-Key
    GET /v1/exposure/gex/{symbol}?expiration=YYYY-MM-DD
    GET /v1/exposure/levels/{symbol}
    Free tier: 5 req/วัน + หุ้นรายตัวเท่านั้น + ห้าม 0DTE → cache หนัก + โหลด manual + preflight guard
"""

import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from datetime import datetime
from scipy.stats import norm

FA_BASE = "https://lab.flashalpha.com"
GEX_MONEYNESS_WIN = 0.25          # เก็บ strike ±25% รอบ spot พอสำหรับหา walls
RISK_FREE = 0.05                  # ใช้เฉพาะ BS gamma fallback

# ── ข้อจำกัด FlashAlpha ตาม tier (ยืนยันจากการยิงจริง 2026-07-30) ──
#   Free   : หุ้นรายตัว + expiration ที่ไม่ใช่วันนี้        5 req/วัน
#   Basic  : + ETF/Index (SPY/QQQ/SPX)                    100 req/วัน
#   Growth : + 0DTE (same-day expiration)               2,500 req/วัน
FA_TIER_NOTE = ("Free = หุ้นรายตัว & ไม่ใช่ 0DTE · ETF/Index ต้อง Basic · "
                "0DTE ต้อง Growth")

# ETF/Index ที่รู้ว่า Free tier ตอบ 403 — กันฝั่ง client ไม่ให้เสีย quota ฟรี ๆ
FA_BLOCKED_FREE = {
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "EEM", "EFA", "XLF", "XLE", "XLK",
    "SMH", "SOXL", "TQQQ", "SQQQ", "GLD", "SLV", "TLT", "HYG", "ARKK", "UVXY", "VXX",
    "SPX", "NDX", "RUT", "VIX", "XSP", "SPXW",
}


# ════════════════════════════════════════════════
# API key resolution: st.secrets → env → sidebar input
# ════════════════════════════════════════════════
def get_fa_key():
    try:
        if "FLASHALPHA_API_KEY" in st.secrets:
            k = str(st.secrets["FLASHALPHA_API_KEY"]).strip()
            if k:
                return k, "st.secrets"
    except Exception:
        pass
    k = os.environ.get("FLASHALPHA_API_KEY", "").strip()
    if k:
        return k, "env"
    k = str(st.session_state.get("fa_key_input", "") or "").strip()
    if k:
        return k, "manual"
    return None, None


# ════════════════════════════════════════════════
# CBOE chain → GEX (parser เฉพาะทาง — ไม่แตะ parse_options เดิม)
# ════════════════════════════════════════════════
def _safe_num(df, candidates, default=np.nan):
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def parse_cboe_for_gex(df_raw: pd.DataFrame, S: float) -> pd.DataFrame:
    """
    แปลง raw CBOE chain → DataFrame สำหรับ GEX:
    [expiry, days, type, strike, gamma, open_interest, iv]
    กรองเบากว่า parse_options: รวม 0DTE (days=0) และไม่ตัด bid=0 (OTM ไกลยังมี OI)
    """
    if df_raw is None or df_raw.empty or not S:
        return pd.DataFrame()
    df = df_raw.copy()

    sym_col = next((c for c in ["option", "symbol", "Symbol"] if c in df.columns), None)

    # expiry
    exp_col = next((c for c in ["expiration", "expiry", "exp_date", "expiration_date"]
                    if c in df.columns), None)
    if exp_col:
        df["expiry"] = df[exp_col].astype(str).str[:10]
    elif sym_col:
        def _exp(s):
            m = re.search(r"(\d{6})[CP]", str(s))
            return f"20{m.group(1)[:2]}-{m.group(1)[2:4]}-{m.group(1)[4:6]}" if m else None
        df["expiry"] = df[sym_col].apply(_exp)
    else:
        return pd.DataFrame()

    # type
    type_col = next((c for c in ["option_type", "type", "call_put", "cp_flag"] if c in df.columns), None)
    if type_col:
        df["type"] = df[type_col].astype(str).str.lower().map(
            {"c": "call", "p": "put", "call": "call", "put": "put"})
    elif sym_col:
        def _typ(s):
            m = re.search(r"\d{6}([CP])", str(s))
            return {"C": "call", "P": "put"}.get(m.group(1)) if m else None
        df["type"] = df[sym_col].apply(_typ)
    else:
        return pd.DataFrame()

    # strike
    strike_col = next((c for c in ["strike", "strike_price"] if c in df.columns), None)
    if strike_col:
        df["strike"] = pd.to_numeric(df[strike_col], errors="coerce")
    elif sym_col:
        def _stk(s):
            m = re.search(r"[CP](\d{8})$", str(s))
            return float(m.group(1)) / 1000.0 if m else np.nan
        df["strike"] = df[sym_col].apply(_stk)
    else:
        return pd.DataFrame()

    df["open_interest"] = _safe_num(df, ["open_interest", "openInterest", "oi"], 0).fillna(0)
    df["gamma"] = _safe_num(df, ["gamma", "Gamma"])
    df["iv"] = _safe_num(df, ["iv", "IV", "implied_volatility"])
    # ใช้ median ไม่ใช่ max — deep-OTM ระยะสั้น IV เกิน 5.0 (500%) ได้จริง
    # ถ้าใช้ max เป็นเกณฑ์จะหาร 100 ทั้งกระดานผิด ๆ ทำให้ gamma/flip เพี้ยนตามไปด้วย
    iv_med = df["iv"].dropna().median()
    if pd.notna(iv_med) and iv_med > 5:
        df["iv"] = df["iv"] / 100.0

    # เทียบเป็น "วันที่" ไม่ใช่ datetime — ไม่งั้น expiry วันนี้ติดลบตอนบ่าย
    # และห้าม clamp เป็น 0 ไม่งั้น expiry ที่หมดอายุแล้วจะถูกนับเป็น 0DTE ปลอม
    today = datetime.now().date()
    df["days"] = df["expiry"].apply(
        lambda x: (datetime.strptime(str(x)[:10], "%Y-%m-%d").date() - today).days
        if x and len(str(x)) >= 10 else -999)

    df = df[
        df["strike"].notna() & df["type"].notna() & (df["days"] >= 0) &
        (df["open_interest"] > 0) &
        (np.abs(np.log(df["strike"] / S)) <= GEX_MONEYNESS_WIN)
    ].copy()

    # gamma fallback: BS gamma จาก iv ถ้า CBOE ไม่ให้มา
    miss = df["gamma"].isna() | (df["gamma"] <= 0)
    if miss.any():
        sub = df[miss & df["iv"].notna() & (df["iv"] > 0)]
        if not sub.empty:
            T = np.maximum(sub["days"] / 365.0, 0.5 / 365.0)   # กัน T=0 ของ 0DTE
            d1 = (np.log(S / sub["strike"]) + (RISK_FREE + 0.5 * sub["iv"] ** 2) * T) \
                 / (sub["iv"] * np.sqrt(T))
            df.loc[sub.index, "gamma"] = norm.pdf(d1) / (S * sub["iv"] * np.sqrt(T))
    df = df[df["gamma"].notna() & (df["gamma"] > 0)]

    return df.reset_index(drop=True)


def compute_gex(df: pd.DataFrame, S: float) -> pd.DataFrame:
    """per-strike GEX: [strike, call_gex, put_gex, net_gex] (USD ต่อ 1% move)"""
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["gex"] = d["gamma"] * d["open_interest"] * 100 * S * S * 0.01
    d.loc[d["type"] == "put", "gex"] *= -1
    ps = d.pivot_table(index="strike", columns="type", values="gex", aggfunc="sum")
    for c in ("call", "put"):
        if c not in ps.columns:
            ps[c] = 0.0
    ps = ps.rename(columns={"call": "call_gex", "put": "put_gex"}).fillna(0)
    ps["net_gex"] = ps["call_gex"] + ps["put_gex"]
    return ps.reset_index().sort_values("strike").reset_index(drop=True)


def gamma_profile(df: pd.DataFrame, S: float, span: float = 0.12, n: int = 121):
    """
    Net GEX profile ถ้า spot ย้ายไปที่ราคาต่าง ๆ — reprice gamma ใหม่ทุกจุด (BS)
    คืน (spot_ladder, net_gex_at_each_spot)

    ทำไมต้องทำแบบนี้: gamma ไม่ใช่ค่าคงที่ มันขึ้นกับระยะห่าง spot↔strike
    วิธี cumulative-sum ข้าม strike เป็นแค่ approximation และพังทันทีถ้า chain
    เป็นลบทั้งเส้น (ไม่มี zero-crossing เลย → Flip = N/A ตลอด)
    """
    d = df[df["iv"].notna() & (df["iv"] > 0)]
    if d.empty:
        return None, None

    ladder = np.linspace(S * (1 - span), S * (1 + span), n)
    K   = d["strike"].to_numpy()[:, None]
    iv  = d["iv"].to_numpy()[:, None]
    T   = np.maximum(d["days"].to_numpy()[:, None], 0.5) / 365.0   # 0DTE → ครึ่งวัน
    OI  = d["open_interest"].to_numpy()[:, None]
    sgn = np.where(d["type"].to_numpy()[:, None] == "put", -1.0, 1.0)

    Ss  = ladder[None, :]
    vol = iv * np.sqrt(T)
    d1  = (np.log(Ss / K) + (RISK_FREE + 0.5 * iv ** 2) * T) / vol
    gam = norm.pdf(d1) / (Ss * vol)
    return ladder, (sgn * gam * OI * 100 * Ss ** 2 * 0.01).sum(axis=0)


def _zero_cross(x: np.ndarray, y: np.ndarray, near: float):
    """หาจุดที่ y ตัดศูนย์ (interpolate) แล้วเลือกจุดที่ใกล้ near สุด"""
    hits = []
    for i in range(1, len(y)):
        if y[i - 1] == 0:
            hits.append(float(x[i - 1]))
        elif (y[i - 1] < 0) != (y[i] < 0):
            hits.append(float(x[i - 1] + (0 - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])))
    return min(hits, key=lambda v: abs(v - near)) if hits else None


def find_levels(ps: pd.DataFrame, S: float, df_contracts: pd.DataFrame | None = None) -> dict:
    """
    Flip / Call Wall / Put Wall / Net
    ส่ง df_contracts (ผลจาก parse_cboe_for_gex) มาด้วย → Flip คำนวณแบบ spot-ladder
    ไม่ส่งมา → fallback เป็น cumulative-sum ข้าม strike (อ่อนกว่า มักได้ N/A)
    """
    out = {"net": 0.0, "flip": None, "call_wall": None, "put_wall": None, "flip_method": None}
    if ps.empty:
        return out
    out["net"] = float(ps["net_gex"].sum())
    if (ps["call_gex"] > 0).any():
        out["call_wall"] = float(ps.loc[ps["call_gex"].idxmax(), "strike"])
    if (ps["put_gex"] < 0).any():
        out["put_wall"] = float(ps.loc[ps["put_gex"].idxmin(), "strike"])

    # ── วิธีหลัก: spot ladder (reprice gamma) ──
    if df_contracts is not None and not df_contracts.empty:
        ladder, prof = gamma_profile(df_contracts, S)
        if ladder is not None:
            flip = _zero_cross(ladder, prof, S)
            if flip is not None:
                out["flip"], out["flip_method"] = flip, "spot-ladder"
                out["profile"] = (ladder, prof)
            else:
                out["profile"] = (ladder, prof)
                out["flip_method"] = "ไม่มี zero-crossing ใน ±12% รอบ spot"

    # ── fallback: cumulative sum ข้าม strike ──
    if out["flip"] is None and out["flip_method"] != "spot-ladder":
        cum = ps["net_gex"].cumsum().to_numpy()
        flip = _zero_cross(ps["strike"].to_numpy(), cum, S)
        if flip is not None:
            out["flip"], out["flip_method"] = flip, "cumulative-sum"
    return out


def plot_gamma_profile(ladder, prof, S: float, flip) -> go.Figure:
    """Net GEX ถ้า spot ย้าย — จุดตัดศูนย์คือ Gamma Flip"""
    scale, unit = (1e9, "$Bn") if np.abs(prof).max() >= 1e9 else (1e6, "$M")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ladder, y=prof / scale, mode="lines", name="Net GEX",
        line=dict(color="#00e0ff", width=2.5),
        hovertemplate="Spot %{x:,.1f}<br>Net %{y:.2f} " + unit + "<extra></extra>"))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.35)", line_width=1)
    fig.add_vline(x=S, line_color="#ffffff", line_dash="dash",
                  annotation_text=f"Spot {S:,.2f}")
    if flip:
        fig.add_vline(x=flip, line_color="#f39c12", line_dash="dot",
                      annotation_text=f"Flip {flip:,.1f}")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
        title="Gamma Profile — Net GEX ถ้า spot ย้ายไปแต่ละระดับ (จุดตัด 0 = Gamma Flip)",
        xaxis_title="Hypothetical Spot", yaxis_title=f"Net GEX ({unit})",
        font=dict(family="monospace", size=11, color="#c8d8f0"),
        height=380, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return fig


def fmt_usd(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    a = abs(x)
    if a >= 1e9:
        return f"{x/1e9:+.2f}B"
    if a >= 1e6:
        return f"{x/1e6:+.1f}M"
    if a >= 1e3:
        return f"{x/1e3:+.0f}K"
    return f"{x:+.0f}"


def plot_gex_bars(ps: pd.DataFrame, S: float, lv: dict, title: str) -> go.Figure:
    """Horizontal bar: strike บนแกน Y, GEX บนแกน X + เส้น Spot/Flip/Walls"""
    scale, unit = (1e9, "$Bn") if ps[["call_gex", "put_gex"]].abs().values.max() >= 1e9 else (1e6, "$M")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ps["strike"], x=ps["call_gex"] / scale, orientation="h",
        name="Call GEX", marker_color="#3498db",
        hovertemplate="K=%{y}<br>Call GEX: %{x:.2f} " + unit + "<extra></extra>"))
    fig.add_trace(go.Bar(
        y=ps["strike"], x=ps["put_gex"] / scale, orientation="h",
        name="Put GEX", marker_color="#e74c3c",
        hovertemplate="K=%{y}<br>Put GEX: %{x:.2f} " + unit + "<extra></extra>"))
    fig.add_trace(go.Scatter(
        y=ps["strike"], x=ps["net_gex"] / scale, mode="lines",
        name="Net", line=dict(color="#f1c40f", width=1.5),
        hovertemplate="K=%{y}<br>Net: %{x:.2f} " + unit + "<extra></extra>"))

    def hline(y, color, label):
        if y is None:
            return
        fig.add_hline(y=y, line_color=color, line_dash="dash", line_width=1.3,
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="top right")

    hline(S, "#ffffff", f"Spot {S:,.2f}")
    hline(lv.get("flip"), "#f39c12", f"Flip {lv['flip']:,.0f}" if lv.get("flip") else "")
    hline(lv.get("call_wall"), "#2ecc71", f"Call Wall {lv['call_wall']:,.0f}" if lv.get("call_wall") else "")
    hline(lv.get("put_wall"), "#e91e63", f"Put Wall {lv['put_wall']:,.0f}" if lv.get("put_wall") else "")

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
        barmode="overlay", title=title,
        xaxis_title=f"Gamma Exposure ({unit} ต่อ 1% move)", yaxis_title="Strike",
        font=dict(family="monospace", size=11, color="#c8d8f0"),
        height=620, legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def regime_text(net: float):
    if net > 0:
        return "🟢 Positive GEX — dealer long gamma → โน้มเอียง pinning / mean-revert"
    if net < 0:
        return "🔴 Negative GEX — dealer short gamma → โน้มเอียง trending / vol ขยาย"
    return "⚪ Neutral"


# ════════════════════════════════════════════════
# FlashAlpha client
# ════════════════════════════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)
def _fa_get_cached(path: str, params_tuple: tuple, _key: str) -> dict:
    """GET พร้อม cache 30 นาที (server free tier cache 15 นาทีอยู่แล้ว — ยิงถี่กว่านี้เปลือง quota เปล่า)
    _key ขึ้นต้น underscore = ไม่ถูก hash เข้า cache key"""
    r = requests.get(FA_BASE + path, params=dict(params_tuple),
                     headers={"X-Api-Key": _key, "Accept": "application/json"}, timeout=25)
    rl = {h: r.headers.get(h) for h in
          ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After")}
    try:
        body = r.json()
    except Exception:
        body = {"_raw_text": r.text[:800]}
    return {"status": r.status_code, "body": body, "ratelimit": rl,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def fa_fetch(path: str, key: str, params: dict | None = None) -> dict:
    res = _fa_get_cached(path, tuple(sorted((params or {}).items())), key)
    st.session_state["fa_last_ratelimit"] = res.get("ratelimit", {})
    return res


def fa_error_message(res: dict) -> str | None:
    """
    ขึ้นข้อความจาก server เป็นหลักเสมอ แล้วค่อยเติมคำแนะนำตามเนื้อหาที่ server บอก
    (403 มีได้หลายสาเหตุ — ETF/Index, 0DTE, endpoint เกิน tier — ห้ามเดาสาเหตุเดียว)
    """
    s = res["status"]
    if s == 200:
        return None
    body = res.get("body", {})
    detail = (body.get("message") or body.get("error") or body.get("detail")
              or body.get("_raw_text", "") or "").strip()
    low = detail.lower()

    if s == 401:
        return f"🔑 **API key ไม่ถูกต้อง (401)** — ตรวจ key ใน secrets/sidebar อีกครั้ง\n\n{detail}"
    if s == 403:
        if "0dte" in low or "same-day" in low:
            hint = ("👉 **สาเหตุคือ expiration เป็นวันนี้ (0DTE) ไม่ใช่ตัว symbol** — "
                    "0DTE ต้อง Growth plan ขึ้นไป\n\n"
                    "ทางแก้ที่ไม่ต้องจ่าย: เลือก expiration เป็น**วันอื่นที่ไม่ใช่วันนี้** "
                    "(เว้นว่างก็ได้) หรือใช้แหล่ง 🆓 CBOE ที่ให้ 0DTE ฟรี")
        elif "etf" in low or "index" in low:
            hint = ("👉 **ETF/Index ต้อง Basic ขึ้นไป** — ใช้หุ้นรายตัว (NVDA/AAPL/TSLA) "
                    "หรือแหล่ง 🆓 CBOE ที่ใช้ QQQ/SPX ได้ฟรี")
        else:
            hint = "👉 request นี้เกินสิทธิ์ของ tier ปัจจุบัน — ดูข้อความจาก server ด้านบน"
        return f"🔒 **FlashAlpha ปฏิเสธ (403)**\n\n> {detail}\n\n{hint}"
    if s == 429:
        ra = res.get("ratelimit", {}).get("Retry-After")
        extra = f" (~{int(ra)//3600} ชม.)" if str(ra).isdigit() else ""
        return (f"⏳ **เกิน quota แล้ว (429)** — Free tier = 5 req/วัน\n\n> {detail}\n\n"
                f"Retry-After: {ra or 'N/A'}{extra}")
    return f"⚠️ **FlashAlpha error HTTP {s}**\n\n> {detail}"


def _first_key(d: dict, names, default=None):
    for n in names:
        if isinstance(d, dict) and n in d and d[n] is not None:
            return d[n]
    return default


def parse_fa_gex(body) -> tuple[pd.DataFrame, dict]:
    """
    Defensive parser — รองรับ schema หลายทรง:
    คืน (per-strike df [strike, call_gex, put_gex, net_gex], meta {spot, net, flip, call_wall, put_wall, timestamp})
    """
    meta = {"spot": None, "net": None, "flip": None, "call_wall": None,
            "put_wall": None, "timestamp": None, "label": None, "wall_source": None}
    if body is None:
        return pd.DataFrame(), meta

    root = body if isinstance(body, dict) else {}
    data = root.get("data", root)
    if isinstance(data, dict):
        meta["spot"]      = _first_key(data, ["spot", "spot_price", "underlying_price", "price"],
                                       _first_key(root, ["spot", "spot_price"]))
        meta["net"]       = _first_key(data, ["net_gex", "netGex", "net_gamma_exposure", "total_gex"],
                                       _first_key(root, ["net_gex"]))
        meta["timestamp"] = _first_key(data, ["timestamp", "as_of", "updated_at", "time"],
                                       _first_key(root, ["timestamp"]))
        meta["call_wall"] = _first_key(data, ["call_wall", "callWall"])
        meta["put_wall"]  = _first_key(data, ["put_wall", "putWall"])
        meta["label"]     = _first_key(data, ["net_gex_label", "gex_label", "regime"])
        flip = _first_key(data, ["gamma_flip", "flip", "zero_gamma", "flip_point"])
        if isinstance(flip, (int, float)) and not isinstance(flip, bool):
            meta["flip"] = float(flip)

    # หา strike array
    rows = None
    if isinstance(body, list):
        rows = body
    elif isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for k in ("strikes", "gex", "gex_by_strike", "levels", "exposure", "by_strike", "chart"):
            v = data.get(k) or root.get(k)
            if isinstance(v, list) and v:
                rows = v
                break

    recs = []
    if rows:
        for it in rows:
            if not isinstance(it, dict):
                continue
            k = _first_key(it, ["strike", "strike_price", "k"])
            if k is None:
                continue
            cg = _first_key(it, ["call_gex", "callGex", "call_gamma_exposure", "call"], 0) or 0
            pg = _first_key(it, ["put_gex", "putGex", "put_gamma_exposure", "put"], 0) or 0
            ng = _first_key(it, ["net_gex", "netGex", "gex", "net"])
            # จุด flip แบบ boolean flag ต่อ strike
            fl = _first_key(it, ["gamma_flip", "is_flip", "flip"])
            if isinstance(fl, bool) and fl and meta["flip"] is None:
                meta["flip"] = float(k)
            try:
                cg, pg = float(cg), float(pg)
                ng = float(ng) if ng is not None else cg + pg
                recs.append({"strike": float(k), "call_gex": cg, "put_gex": pg, "net_gex": ng})
            except (TypeError, ValueError):
                continue

    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.sort_values("strike").reset_index(drop=True)
        # ถ้า put_gex เป็นบวกทั้งหมดแต่มี call แยก → แปลง sign เป็น dealer convention เพื่อ plot สอดคล้อง
        # (ยืนยัน 2026-07-30: /v1/exposure/gex ส่ง put_gex ติดลบมาแล้ว = convention เดียวกับเรา
        #  guard นี้จึงไม่ทำงาน แต่คงไว้เผื่อ endpoint อื่น/เวอร์ชันอื่น)
        if (df["put_gex"] > 0).all() and df["put_gex"].abs().sum() > 0 and (df["call_gex"] >= 0).all():
            df["put_gex"] = -df["put_gex"]
            df["net_gex"] = df["call_gex"] + df["put_gex"]
        if meta["net"] is None:
            meta["net"] = float(df["net_gex"].sum())
        # endpoint /v1/exposure/gex ไม่ส่ง call_wall/put_wall มา (ต้องเรียก /levels = อีก 1 request)
        # → derive จาก per-strike เองด้วยนิยามเดียวกับฝั่ง CBOE เพื่อเทียบกันได้ตรง ๆ ฟรี
        if meta["call_wall"] is None and (df["call_gex"] > 0).any():
            meta["call_wall"] = float(df.loc[df["call_gex"].idxmax(), "strike"])
            meta["wall_source"] = "derived"
        if meta["put_wall"] is None and (df["put_gex"] < 0).any():
            meta["put_wall"] = float(df.loc[df["put_gex"].idxmin(), "strike"])
            meta["wall_source"] = "derived"
    return df, meta


def fa_preflight(sym: str, exp: str) -> list[str]:
    """
    ตรวจก่อนยิง — คืน list เหตุผลที่ Free tier จะถูกปฏิเสธ (ว่าง = ยิงได้)
    ประหยัด quota: ไม่ต้องเสีย request ไปเรียนรู้สิ่งที่รู้อยู่แล้ว
    """
    reasons = []
    if sym in FA_BLOCKED_FREE:
        reasons.append(f"**{sym}** เป็น ETF/Index → ต้อง Basic ($63/เดือน) · "
                       "ใช้แหล่ง 🆓 CBOE แทนได้ ฟรีและได้ผลเหมือนกัน")
    if exp and exp == datetime.now().strftime("%Y-%m-%d"):
        reasons.append(f"**{exp} คือวันนี้ = 0DTE** → ต้อง Growth plan · "
                       "เลือก expiration วันอื่น หรือเว้นว่าง (หรือใช้ 🆓 CBOE ที่ให้ 0DTE ฟรี)")
    return reasons


def show_quota_badge():
    rl = st.session_state.get("fa_last_ratelimit") or {}
    rem, lim = rl.get("X-RateLimit-Remaining"), rl.get("X-RateLimit-Limit")
    if rem is not None:
        st.caption(f"📊 FlashAlpha quota เหลือ: **{rem}**" + (f" / {lim}" if lim else "") + " requests วันนี้")


# ════════════════════════════════════════════════
# TAB 2 — GEX
# ════════════════════════════════════════════════
def render_gex_tab(sym: str, name: str, cboe_fetch):
    st.subheader("🧲 Dealer Gamma Exposure (GEX)")
    st.caption("GEX = Γ × OI × 100 × S² × 0.01 (call +, put −) · ใช้เป็น **แผนที่ level** (Walls/Flip) "
               "ไม่ใช่ตัวพยากรณ์ทิศทาง — regime หลักดูจาก IV Rank")

    source = st.radio("แหล่งข้อมูล", ["🆓 CBOE (คำนวณเอง · ฟรี · ใช้กับ QQQ/SPX ได้)",
                                      "⚡ FlashAlpha API (สดกว่า · Free tier = หุ้นรายตัว 5 req/วัน)"],
                      horizontal=True, key="gex_source")

    # ── Source A: CBOE ──────────────────────────
    if source.startswith("🆓"):
        try:
            df_raw, S = cboe_fetch(sym)
        except Exception as e:
            st.error(f"CBOE fetch error: {e}")
            return
        if df_raw.empty or not S:
            st.warning("ยังไม่มีข้อมูล chain — กด 🔄 โหลดข้อมูล ใน sidebar ก่อน")
            return

        df_gex = parse_cboe_for_gex(df_raw, S)
        if df_gex.empty:
            st.warning("แปลงข้อมูล GEX ไม่ได้ (ไม่มี OI/gamma) — ดู Debug columns ใน tab IV")
            return

        exps = (df_gex.groupby("expiry")["days"].first().sort_values())
        exp_opts = ["รวมทุก expiry"] + [f"{e}  ({int(d)}d)" for e, d in exps.items()]
        chosen = st.selectbox("Expiry", exp_opts, index=min(1, len(exp_opts) - 1),
                              help="0DTE/near-term สำคัญสุดสำหรับ intraday levels")
        win = st.slider("หน้าต่าง strike รอบ spot (±%)", 3, 25, 10, key="gex_win")

        d = df_gex if chosen == "รวมทุก expiry" else df_gex[df_gex["expiry"] == chosen.split()[0]]
        d = d[np.abs(np.log(d["strike"] / S)) <= win / 100.0]
        ps = compute_gex(d, S)
        if ps.empty:
            st.warning("ไม่มีข้อมูลในหน้าต่างที่เลือก")
            return
        lv = find_levels(ps, S, df_contracts=d)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Net GEX", fmt_usd(lv["net"]))
        c2.metric("Spot", f"{S:,.2f}")
        c3.metric("Gamma Flip", f"{lv['flip']:,.1f}" if lv["flip"] else "N/A",
                  help=f"วิธี: {lv.get('flip_method') or 'N/A'}")
        c4.metric("Call Wall", f"{lv['call_wall']:,.0f}" if lv["call_wall"] else "N/A")
        c5.metric("Put Wall", f"{lv['put_wall']:,.0f}" if lv["put_wall"] else "N/A")
        st.info(regime_text(lv["net"]) + " · ⚠️ OI อัปเดตข้ามคืน — ระหว่างวันใช้ราคาเทียบ Flip เป็นตัวตัดสินจริง")
        st.caption("⚠️ **เครื่องหมาย/ขนาดของ Net GEX ขึ้นกับ convention** (สูตรนี้สมมติ dealer long call / short put "
                   "ตามมาตรฐาน SqueezeMetrics) — index chain ที่ put OI หนักจะอ่านเป็นลบเสมอ "
                   "ตัวที่ใช้ได้จริงคือ **ตำแหน่ง Walls/Flip และรูปทรง profile** ไม่ใช่เลขดิบ")

        st.plotly_chart(plot_gex_bars(
            ps, S, lv, f"GEX — {name}  [{chosen}]  ·  CBOE Delayed  ·  {datetime.now():%Y-%m-%d %H:%M}"),
            width="stretch")

        if lv.get("profile"):
            ladder, prof = lv["profile"]
            st.plotly_chart(plot_gamma_profile(ladder, prof, S, lv["flip"]),
                            width="stretch")
            if not lv["flip"]:
                st.caption(f"ℹ️ {lv['flip_method']} — chain เป็นฝั่งเดียวทั้งเส้นในช่วงนี้ "
                           "(ดู profile ว่าเข้าใกล้ศูนย์ทางไหน)")

        with st.expander("📋 GEX per strike"):
            show = ps.copy()
            for c in ("call_gex", "put_gex", "net_gex"):
                show[c] = show[c].apply(fmt_usd)
            st.dataframe(show, width="stretch", height=280)

    # ── Source B: FlashAlpha ────────────────────
    else:
        key, key_src = get_fa_key()
        if not key:
            st.warning("ยังไม่มี API key — ใส่ใน `.streamlit/secrets.toml` (FLASHALPHA_API_KEY) "
                       "หรือกรอกด้านล่าง (ไม่ถูกบันทึกถาวร)")
            st.text_input("FlashAlpha API Key", type="password", key="fa_key_input")
            return
        st.caption(f"🔑 key จาก: `{key_src}`")

        c1, c2 = st.columns([2, 2])
        fa_sym = c1.text_input("Symbol (Free tier = หุ้นรายตัว เช่น NVDA, AAPL, TSLA)",
                               value="NVDA", key="fa_gex_sym").strip().upper().lstrip("_")
        fa_exp = c2.text_input("Expiration (YYYY-MM-DD · เว้นว่าง = ทุก expiry)",
                               value="", key="fa_gex_exp",
                               help="ห้ามใส่วันนี้ — 0DTE ต้อง Growth plan. เว้นว่างปลอดภัยสุด").strip()

        reasons = fa_preflight(fa_sym, fa_exp)
        if reasons:
            st.warning("🔒 **Free tier จะปฏิเสธ request นี้** — ปุ่มถูกล็อกกัน quota เสียเปล่า\n\n"
                       + "\n\n".join(f"- {r}" for r in reasons))

        st.caption(f"⚠️ กด 1 ครั้ง = ใช้ 1 request (cache 30 นาที — กดซ้ำ symbol เดิมไม่เสีย quota เพิ่ม) · "
                   f"ข้อจำกัด: {FA_TIER_NOTE}")
        if st.button("⚡ โหลด GEX จาก FlashAlpha", type="primary", key="fa_gex_btn",
                     disabled=bool(reasons)):
            params = {"expiration": fa_exp} if fa_exp else {}
            res = fa_fetch(f"/v1/exposure/gex/{fa_sym}", key, params)
            st.session_state["fa_gex_res"] = res
            st.session_state["fa_gex_res_sym"] = fa_sym

        res = st.session_state.get("fa_gex_res")
        if not res:
            return
        show_quota_badge()
        err = fa_error_message(res)
        if err:
            st.error(err)
            return

        df_fa, meta = parse_fa_gex(res["body"])
        spot = meta.get("spot")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Net GEX", fmt_usd(meta["net"]) if meta["net"] is not None else "N/A")
        c2.metric("Spot", f"{spot:,.2f}" if spot else "N/A")
        c3.metric("Gamma Flip", f"{meta['flip']:,.1f}" if meta["flip"] else "N/A")
        c4.metric("Call Wall", f"{meta['call_wall']:,.0f}" if meta["call_wall"] else "N/A")
        c5.metric("Put Wall", f"{meta['put_wall']:,.0f}" if meta["put_wall"] else "N/A")
        if meta["net"] is not None:
            lbl = f" · vendor label: **{meta['label']}**" if meta.get("label") else ""
            st.info(regime_text(meta["net"]) + lbl)
        st.caption(f"⏱️ FlashAlpha timestamp: {meta['timestamp'] or 'N/A'} · fetched {res['fetched_at']}"
                   + (" · Walls คำนวณจาก per-strike (endpoint ไม่ส่งมา)"
                      if meta.get("wall_source") == "derived" else ""))

        if not df_fa.empty:
            lv_fa = {"net": meta["net"] or 0, "flip": meta["flip"],
                     "call_wall": meta["call_wall"], "put_wall": meta["put_wall"]}
            st.plotly_chart(plot_gex_bars(
                df_fa, spot or df_fa["strike"].median(), lv_fa,
                f"GEX — {st.session_state.get('fa_gex_res_sym','')}  ·  FlashAlpha"),
                width="stretch")
        else:
            st.warning("ไม่พบ per-strike array ใน response — ดู raw JSON ด้านล่างแล้วบอกโครงสร้างจริงได้เลย")
        with st.expander("🔍 Raw JSON (debug)"):
            st.json(res["body"])


# ════════════════════════════════════════════════
# TAB 3 — เทียบความแม่น CBOE vs FlashAlpha
# ════════════════════════════════════════════════
def render_compare_tab(cboe_fetch):
    st.subheader("⚖️ เทียบข้อมูล: CBOE Delayed (ของเดิม) vs FlashAlpha (ของใหม่)")
    st.markdown(
        "- **CBOE** = exchange โดยตรง (source of truth ของ quote) แต่ **delayed 15 นาที** · ฟรี unlimited\n"
        "- **FlashAlpha** = vendor คำนวณ Greeks/GEX ให้เสร็จ · สดกว่า (cache 15 วิ–15 นาทีตาม tier) · quota จำกัด\n"
        "- OI อัปเดตข้ามคืนเหมือนกันทั้งคู่ → GEX per strike ควรใกล้กัน ต่างหลักคือ **spot ตาม lag** และ convention หน่วย")

    key, _ = get_fa_key()
    if not key:
        st.warning("ต้องมี FlashAlpha API key ก่อน (ใส่ใน `.streamlit/secrets.toml` หรือ tab GEX ฝั่ง FlashAlpha)")
        return

    c1, c2 = st.columns([2, 2])
    sym = c1.text_input("Symbol (Free tier = หุ้นรายตัว เช่น NVDA, AAPL)",
                        value="NVDA", key="cmp_sym").strip().upper().lstrip("_")
    exp = c2.text_input("Expiration (YYYY-MM-DD · เว้นว่าง = ทุก expiry)", value="", key="cmp_exp",
                        help="ห้ามใส่วันนี้ — 0DTE ต้อง Growth plan. เว้นว่างปลอดภัยสุด").strip()

    reasons = fa_preflight(sym, exp)
    if reasons:
        st.warning("🔒 **Free tier จะปฏิเสธ request นี้** — ปุ่มถูกล็อกกัน quota เสียเปล่า\n\n"
                   + "\n\n".join(f"- {r}" for r in reasons))

    st.caption(f"⚠️ กด 1 ครั้ง = FlashAlpha 1 request + CBOE ฟรี (ผลค้างไว้ ดูซ้ำได้ไม่เสีย quota) · "
               f"ข้อจำกัด: {FA_TIER_NOTE}")
    if st.button("🔬 เทียบข้อมูลตอนนี้", type="primary", key="cmp_btn", disabled=bool(reasons)):
        with st.spinner("ดึง CBOE + FlashAlpha ..."):
            # CBOE (ฟรี)
            try:
                df_raw, S_cboe = cboe_fetch(sym)
            except Exception as e:
                st.error(f"CBOE ดึงไม่ได้: {e} — เช็คว่า symbol นี้มี options บน CBOE")
                return
            cboe_time = datetime.now()
            # FlashAlpha (1 req)
            res = fa_fetch(f"/v1/exposure/gex/{sym}", key, {"expiration": exp} if exp else {})
        st.session_state["cmp_result"] = {
            "sym": sym, "exp": exp, "df_raw": df_raw, "S_cboe": S_cboe,
            "cboe_time": cboe_time.strftime("%Y-%m-%d %H:%M:%S"), "fa_res": res,
        }

    r = st.session_state.get("cmp_result")
    if not r:
        return
    show_quota_badge()

    err = fa_error_message(r["fa_res"])
    if err:
        st.error(err)
        return

    # ── คำนวณฝั่ง CBOE ──
    S_cboe = r["S_cboe"]
    dfg = parse_cboe_for_gex(r["df_raw"], S_cboe)
    if r["exp"]:
        dfg = dfg[dfg["expiry"] == r["exp"]]
    ps_cboe = compute_gex(dfg, S_cboe)
    lv_cboe = find_levels(ps_cboe, S_cboe, df_contracts=dfg)

    # ── ฝั่ง FlashAlpha ──
    df_fa, meta = parse_fa_gex(r["fa_res"]["body"])
    S_fa = meta.get("spot")

    # ── ตารางเทียบ ──
    st.markdown(f"### 📋 ผลเทียบ — {r['sym']}" + (f"  (expiry {r['exp']})" if r["exp"] else "  (ทุก expiry)"))
    spot_diff = (f"{abs(S_fa - S_cboe):,.2f}  ({abs(S_fa - S_cboe)/S_cboe*100:.3f}%)"
                 if (S_fa and S_cboe) else "N/A")
    rows = [
        ("Spot", f"{S_cboe:,.2f}" if S_cboe else "N/A",
         f"{S_fa:,.2f}" if S_fa else "N/A", spot_diff),
        ("เวลาข้อมูล", f"fetch {r['cboe_time']} (delayed ~15 นาที)",
         str(meta.get("timestamp") or "N/A"), "FlashAlpha สดกว่าถ้า timestamp ใหม่กว่า"),
        ("Net GEX (ทั้งชุด)", fmt_usd(lv_cboe["net"]),
         fmt_usd(meta["net"]) if meta["net"] is not None else "N/A",
         "⚠️ strike coverage ไม่เท่ากัน — ดูแถวถัดไป"),
        ("จำนวน strike", f"{len(ps_cboe)}", f"{len(df_fa)}",
         "ต่างชุด → ผลรวมเทียบกันตรง ๆ ไม่ได้"),
        ("Gamma Flip", f"{lv_cboe['flip']:,.1f}" if lv_cboe["flip"] else "N/A",
         f"{meta['flip']:,.1f}" if meta["flip"] else "N/A",
         f"ห่าง {abs(lv_cboe['flip']-meta['flip']):,.1f}" if (lv_cboe["flip"] and meta["flip"]) else "—"),
        ("Call Wall", f"{lv_cboe['call_wall']:,.0f}" if lv_cboe["call_wall"] else "N/A",
         f"{meta['call_wall']:,.0f}" if meta["call_wall"] else "N/A",
         "✅ ตรงกัน" if (lv_cboe["call_wall"] and meta["call_wall"]
                        and lv_cboe["call_wall"] == meta["call_wall"]) else "—"),
        ("Put Wall", f"{lv_cboe['put_wall']:,.0f}" if lv_cboe["put_wall"] else "N/A",
         f"{meta['put_wall']:,.0f}" if meta["put_wall"] else "N/A",
         "✅ ตรงกัน" if (lv_cboe["put_wall"] and meta["put_wall"]
                        and lv_cboe["put_wall"] == meta["put_wall"]) else "—"),
    ]

    # ── Net GEX บน strike ชุดเดียวกัน = การเทียบที่ยุติธรรมจริง ──
    common = None
    if not ps_cboe.empty and not df_fa.empty:
        common = pd.merge(ps_cboe, df_fa, on="strike", suffixes=("_c", "_f"))
        if not common.empty:
            nc, nf = float(common["net_gex_c"].sum()), float(common["net_gex_f"].sum())
            same_sign = (nc >= 0) == (nf >= 0)
            rows.append((f"Net GEX ({len(common)} strike ร่วม)", fmt_usd(nc), fmt_usd(nf),
                         ("✅ เครื่องหมายเดียวกัน" if same_sign else "❌ เครื่องหมายสวนกัน")
                         + (f" · ratio {nf/nc:.2f}×" if nc else "")))
    st.table(pd.DataFrame(rows, columns=["รายการ", "CBOE (เดิม)", "FlashAlpha (ใหม่)", "ต่างกัน"]))
    if meta.get("wall_source") == "derived":
        st.caption("ℹ️ Walls ฝั่ง FlashAlpha คำนวณจาก per-strike ที่ endpoint ส่งมา "
                   "(`/v1/exposure/gex` ไม่ส่ง wall มาตรง ๆ — ถ้าจะเอาของ vendor ต้องเรียก "
                   "`/v1/exposure/levels` = เสียอีก 1 request)")

    # ── per-strike overlay + correlation ──
    verdict = []
    if common is not None and len(common) >= 3:
        m = common
        for col, label in (("call_gex", "Call"), ("put_gex", "Put"), ("net_gex", "Net")):
            a, b = m[f"{col}_c"], m[f"{col}_f"]
            corr = float(np.corrcoef(a, b)[0, 1])
            ratio = float(b.abs().sum() / a.abs().sum()) if a.abs().sum() else float("nan")
            verdict.append(f"**{label} GEX** correlation `{corr:.3f}` · ขนาด FA/CBOE `{ratio:.2f}×`")

        scale = 1e9 if m[["net_gex_c", "net_gex_f"]].abs().values.max() >= 1e9 else 1e6
        unit = "$Bn" if scale == 1e9 else "$M"
        fig = go.Figure()
        fig.add_trace(go.Bar(x=m["strike"], y=m["net_gex_c"] / scale,
                             name="CBOE (คำนวณเอง)", marker_color="#3498db"))
        fig.add_trace(go.Bar(x=m["strike"], y=m["net_gex_f"] / scale,
                             name="FlashAlpha", marker_color="#f39c12"))
        if S_cboe:
            fig.add_vline(x=S_cboe, line_color="#ffffff", line_dash="dash",
                          annotation_text=f"Spot {S_cboe:,.0f}")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080d1c", plot_bgcolor="#0d1425",
            barmode="group", title=f"Net GEX per strike — CBOE vs FlashAlpha ({r['sym']})",
            xaxis_title="Strike", yaxis_title=f"Net GEX ({unit})",
            font=dict(family="monospace", size=11, color="#c8d8f0"), height=460)
        st.plotly_chart(fig, width="stretch")
    elif common is not None:
        verdict.append(f"strikes ร่วมมีแค่ {len(common)} จุด — เทียบ per-strike ไม่ได้ (ลองไม่ระบุ expiry)")
    elif df_fa.empty:
        verdict.append("FlashAlpha ไม่ส่ง per-strike array — เทียบได้เฉพาะระดับ summary (ดู raw JSON ใน tab GEX)")

    # ── verdict ──
    if S_fa and S_cboe:
        lag_pct = abs(S_fa - S_cboe) / S_cboe * 100
        verdict.insert(0, f"Spot ห่างกัน **{lag_pct:.3f}%** — {'สอดคล้อง delayed 15 นาที' if lag_pct < 1 else 'ห่างผิดปกติ เช็ค timestamp'}")
    if lv_cboe["call_wall"] and meta["call_wall"]:
        same = lv_cboe["call_wall"] == meta["call_wall"]
        verdict.append(("Call Wall **ตรงกัน** — level เชื่อถือได้ทั้งสองแหล่ง" if same
                        else f"Call Wall ต่างกัน ({lv_cboe['call_wall']:,.0f} vs {meta['call_wall']:,.0f}) — เช็ค expiry filter ให้ตรงกัน"))
    st.markdown("### 🧾 สรุป\n" + "\n".join(f"- {v}" for v in verdict))
    st.info(
        "**อ่านผลอย่างไร** — correlation per-strike สูง (>0.95) + Walls ตรงกัน = ใช้ CBOE ฟรีแทนได้สบาย "
        "สำหรับงาน *หา level*\n\n"
        "แต่ **Net GEX รวมเป็นผลต่างของเลขใหญ่สองก้อน** (Σcall − Σ|put|) — ถ้าน้ำหนัก put ต่างกันแค่ "
        "10–15% ผลรวมพลิกเครื่องหมายได้ทั้งที่รูปทรงเหมือนกันเป๊ะ ⇒ **ห้ามใช้เครื่องหมาย Net GEX "
        "เดี่ยว ๆ เป็น Gate ตัดสิน regime** ให้ใช้ตำแหน่งราคาเทียบ Flip + IV Rank ตามที่ SKILL v2 บอก")
