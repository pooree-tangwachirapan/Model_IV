"""
snapshot.py — คำนวณ metric ทั้งชุดสำหรับ Dashboard และอีเมลรายวัน
ไม่มีการเรียก Streamlit UI ในไฟล์นี้ (นอกจาก cache) → สคริปต์ส่งเมลใช้ซ้ำได้

แหล่งข้อมูลฟรีทั้งหมด:
  CBOE delayed  → chain + greeks (delta/gamma/vega/theta/rho) + OI + IV
  yfinance      → VIX / VVIX / ราคาย้อนหลัง (realized vol)

หน่วย GEX/DEX/VEX เป็น convention ของเราเอง — เทียบข้ามเจ้าไม่ได้ ดู ratio ไม่ใช่เลขดิบ
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime
from scipy.stats import norm

import fa_gex

CBOE = "https://cdn.cboe.com/api/global/delayed_quotes/options"
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.cboe.com/", "Accept": "application/json"}
RF = 0.05


# ════════════════════════════════════════════════
# แหล่งข้อมูล
# ════════════════════════════════════════════════
def fetch_chain(sym: str):
    """คืน (raw DataFrame, spot) จาก CBOE"""
    d = requests.get(f"{CBOE}/{sym}.json", headers=HDRS, timeout=20).json()["data"]
    return pd.DataFrame(d.get("options", [])), float(d.get("current_price", 0))


def get_macro() -> dict:
    """VIX / VVIX + % เปลี่ยนแปลง (yfinance) — ล้มเหลวได้ ไม่ทำให้ทั้งระบบพัง"""
    out = {"vix": None, "vix_chg": None, "vvix": None, "vvix_chg": None, "error": None}
    try:
        import yfinance as yf
        for key, tk in (("vix", "^VIX"), ("vvix", "^VVIX")):
            h = yf.Ticker(tk).history(period="5d")
            if len(h) >= 2:
                c = h["Close"]
                out[key] = float(c.iloc[-1])
                out[f"{key}_chg"] = float((c.iloc[-1] / c.iloc[-2] - 1) * 100)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def get_realized_vol(ticker: str) -> dict:
    """HV จากราคาย้อนหลัง — close-to-close 10/20 วัน + Parkinson (ใช้ high/low แม่นกว่า)"""
    out = {"hv10": None, "hv20": None, "parkinson": None, "error": None}
    try:
        import yfinance as yf
        h = yf.download(ticker, period="3mo", interval="1d",
                        progress=False, auto_adjust=True)
        if len(h) < 21:
            out["error"] = f"ข้อมูลไม่พอ ({len(h)} แถว)"
            return out
        c = h["Close"].squeeze()
        r = np.log(c / c.shift(1)).dropna()
        out["hv10"] = float(r.tail(10).std() * np.sqrt(252) * 100)
        out["hv20"] = float(r.tail(20).std() * np.sqrt(252) * 100)
        hl = np.log(h["High"].squeeze() / h["Low"].squeeze()).tail(20)
        out["parkinson"] = float(np.sqrt((hl ** 2).mean() / (4 * np.log(2))) * np.sqrt(252) * 100)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ════════════════════════════════════════════════
# Metric
# ════════════════════════════════════════════════
def max_pain(df: pd.DataFrame) -> float | None:
    """
    strike ที่ผู้ถือ option เจ็บรวมมากสุด (= ผู้ขายได้ประโยชน์)
    payoff ที่ settlement K:  call ได้ max(K − strike, 0) · put ได้ max(strike − K, 0)
    """
    if df.empty:
        return None
    c = df[df["type"] == "call"]
    p = df[df["type"] == "put"]
    strikes = np.sort(df["strike"].unique())
    if len(strikes) < 3:
        return None
    pay = [
        (np.maximum(K - c["strike"], 0) * c["open_interest"]).sum() +
        (np.maximum(p["strike"] - K, 0) * p["open_interest"]).sum()
        for K in strikes
    ]
    return float(strikes[int(np.argmin(pay))])


def top_oi_strike(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    g = df.groupby("strike")["open_interest"].sum()
    return float(g.idxmax()) if len(g) else None


def dex_vex(raw: pd.DataFrame, S: float, expiry: str | None = None) -> dict:
    """
    DEX = Σ (delta ที่มีเครื่องหมายอยู่แล้ว) × OI × 100 × S   [net delta ของ OI]
    VEX = Σ vega × OI × 100                                    [ต่อ 1 vol point]
    ใช้ greeks ที่ CBOE ให้มาตรง ๆ
    """
    out = {"dex": None, "vex": None}
    d = raw.copy()
    for col in ("delta", "vega", "open_interest"):
        if col not in d.columns:
            return out
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if expiry and "expiry" in d.columns:
        d = d[d["expiry"] == expiry]
    d = d[d["open_interest"].fillna(0) > 0]
    if d.empty:
        return out
    dd = d[d["delta"].notna()]
    if len(dd):
        out["dex"] = float((dd["delta"] * dd["open_interest"] * 100 * S).sum())
    vv = d[d["vega"].notna()]
    if len(vv):
        out["vex"] = float((vv["vega"] * vv["open_interest"] * 100).sum())
    return out


def vanna_charm_exposure(df: pd.DataFrame, S: float) -> dict:
    """Vanna/Charm CBOE ไม่ให้ → คำนวณจาก Black-Scholes เอง (dealer sign: call +, put −)"""
    g = df[df["iv"].notna() & (df["iv"] > 0)].copy()
    if g.empty:
        return {"vanna": None, "charm": None}
    T = np.maximum(g["days"], 0.5) / 365.0
    sq = g["iv"] * np.sqrt(T)
    d1 = (np.log(S / g["strike"]) + (RF + 0.5 * g["iv"] ** 2) * T) / sq
    d2 = d1 - sq
    vanna = -norm.pdf(d1) * d2 / g["iv"]
    charm = -norm.pdf(d1) * (2 * RF * T - d2 * sq) / (2 * T * sq)
    sgn = np.where(g["type"] == "put", -1.0, 1.0)
    w = g["open_interest"] * 100 * S
    # charm จากสูตรเป็น "ต่อปี" → หาร 365 ให้อ่านเป็น delta ที่ไหลต่อวัน (ใช้จริงได้)
    return {"vanna": float((sgn * vanna * w).sum()),
            "charm": float((sgn * charm * w).sum()) / 365.0}


def skew_25d(df: pd.DataFrame, raw: pd.DataFrame, target_dte: int = 30) -> dict:
    """
    RR25 = IV(25Δ put) − IV(25Δ call)  [pts]  บวก = put แพงกว่า = ซื้อประกันหนัก
    ใช้ delta ต่อสัญญาที่ CBOE ให้มา หา contract ที่ |delta| ใกล้ 0.25 สุด
    """
    out = {"rr25": None, "iv_put25": None, "iv_call25": None, "expiry": None, "dte": None}
    if df.empty or "delta" not in raw.columns:
        return out
    # เลือก expiry ที่ DTE ใกล้เป้าสุด (ต้อง > 5 วันเพื่อเลี่ยง 0DTE ที่ delta วิ่งแรง)
    exp = df[df["days"] >= 5].groupby("expiry")["days"].first()
    if exp.empty:
        return out
    pick = (exp - target_dte).abs().idxmin()
    sub = df[df["expiry"] == pick].copy()

    key = raw.copy()
    key["delta"] = pd.to_numeric(key.get("delta"), errors="coerce")
    dmap = {}
    if "option" in key.columns:
        dmap = dict(zip(key["option"], key["delta"]))
    if "option" in sub.columns:
        sub["delta"] = sub["option"].map(dmap)
    if "delta" not in sub.columns or sub["delta"].isna().all():
        return out

    cs = sub[(sub["type"] == "call") & sub["delta"].notna() & sub["iv"].notna()]
    ps = sub[(sub["type"] == "put") & sub["delta"].notna() & sub["iv"].notna()]
    if cs.empty or ps.empty:
        return out
    c25 = cs.iloc[(cs["delta"] - 0.25).abs().argsort()[:1]]
    p25 = ps.iloc[(ps["delta"] + 0.25).abs().argsort()[:1]]
    out.update(
        iv_call25=float(c25["iv"].iloc[0] * 100),
        iv_put25=float(p25["iv"].iloc[0] * 100),
        expiry=pick, dte=int(exp[pick]))
    out["rr25"] = out["iv_put25"] - out["iv_call25"]
    return out


def atm_iv(df: pd.DataFrame, S: float, expiry: str | None = None, n: int = 8) -> float | None:
    """ATM IV = median IV ของสัญญาที่ strike ใกล้ spot สุด (median กัน outlier)"""
    d = df if expiry is None else df[df["expiry"] == expiry]
    d = d[d["iv"].notna() & (d["iv"] > 0.001)]
    if d.empty:
        return None
    near = d.iloc[(d["strike"] - S).abs().argsort()[:n]]
    return float(near["iv"].median() * 100)


def term_structure(df: pd.DataFrame, S: float) -> dict:
    """เทียบ ATM IV ใกล้ vs ไกล — contango (ไกล > ใกล้) = ปกติ"""
    out = {"near": None, "far": None, "slope_pct": None, "state": None,
           "near_exp": None, "far_exp": None}
    # ข้าม 0DTE: IV ของวันหมดอายุแกว่งแรงจน term structure เพี้ยน
    exp = df[df["days"] >= 1].groupby("expiry")["days"].first().sort_values()
    if len(exp) < 2:
        exp = df[df["days"] >= 0].groupby("expiry")["days"].first().sort_values()
    if len(exp) < 2:
        return out
    near_e = exp.index[0]
    far_pool = exp[(exp >= 20) & (exp <= 90)]
    far_e = far_pool.index[0] if len(far_pool) else exp.index[-1]
    n_iv, f_iv = atm_iv(df, S, near_e), atm_iv(df, S, far_e)
    if not n_iv or not f_iv:
        return out
    out.update(near=n_iv, far=f_iv, near_exp=near_e, far_exp=far_e,
               slope_pct=(f_iv / n_iv - 1) * 100)
    out["state"] = "contango" if out["slope_pct"] > 0 else "backwardation"
    return out


def zero_dte_share(df: pd.DataFrame, S: float) -> float | None:
    """สัดส่วน |GEX| ของ 0DTE เทียบทั้งกระดาน"""
    z = df[df["days"] == 0]
    if z.empty or df.empty:
        return None
    tot = fa_gex.compute_gex(df, S)
    zz = fa_gex.compute_gex(z, S)
    if tot.empty or zz.empty:
        return None
    denom = tot["net_gex"].abs().sum()
    return float(zz["net_gex"].abs().sum() / denom * 100) if denom else None


def expected_move(df: pd.DataFrame, S: float, expiry: str) -> dict:
    """1σ expected move = S × IV_atm × √(DTE/365)"""
    out = {"em": None, "em_pct": None, "iv": None, "dte": None}
    sub = df[df["expiry"] == expiry]
    if sub.empty:
        return out
    iv = atm_iv(df, S, expiry)
    if not iv:
        return out
    dte = max(int(sub["days"].iloc[0]), 0)
    T = max(dte, 0.5) / 365.0            # 0DTE → นับครึ่งวัน
    em = S * (iv / 100) * np.sqrt(T)
    out.update(em=float(em), em_pct=float(em / S * 100), iv=iv, dte=dte)
    return out


def dealer_shock(df: pd.DataFrame, S: float, move_pct: float = 1.0) -> dict:
    """
    dealer ต้อง hedge เท่าไหร่ถ้า spot ขยับ ±move_pct
    Δdelta รวม ≈ Σ(signed gamma × OI × 100) × ΔS  → หุ้นที่ต้องซื้อ/ขาย
    """
    out = {"shares_up": None, "shares_dn": None, "notional_up": None,
           "notional_dn": None, "gex_up": None, "gex_dn": None}
    g = df[df["gamma"].notna() & (df["gamma"] > 0)]
    if g.empty:
        return out
    ladder, prof = fa_gex.gamma_profile(df, S, span=move_pct / 100 * 2, n=41)
    if ladder is None:
        return out
    i0 = int(np.argmin(np.abs(ladder - S)))
    iu = int(np.argmin(np.abs(ladder - S * (1 + move_pct / 100))))
    idn = int(np.argmin(np.abs(ladder - S * (1 - move_pct / 100))))

    sgn = np.where(g["type"] == "put", -1.0, 1.0)
    gamma_tot = float((sgn * g["gamma"] * g["open_interest"] * 100).sum())
    dS = S * move_pct / 100
    out.update(
        shares_up=gamma_tot * dS, shares_dn=-gamma_tot * dS,
        notional_up=gamma_tot * dS * S, notional_dn=-gamma_tot * dS * S,
        gex_up=float(prof[iu] - prof[i0]), gex_dn=float(prof[idn] - prof[i0]))
    return out


def pin_score(S: float, mp: float | None, em: float | None,
              ps: pd.DataFrame, net: float) -> dict:
    """
    คะแนนโอกาสถูกตรึง 0–100 (heuristic โปร่งใส ไม่ใช่ค่าลึกลับ):
      50% : spot ใกล้ Max Pain แค่ไหน (เทียบ 1σ EM)
      30% : GEX เป็นบวก (dealer long gamma = แรงตรึง)
      20% : GEX กระจุกที่ strike เดียว (top strike / total)
    """
    out = {"score": None, "parts": {}}
    if mp is None or not em or ps.empty:
        return out
    prox = max(0.0, 1 - abs(S - mp) / em)
    pos = 1.0 if net > 0 else 0.0
    tot = ps["net_gex"].abs().sum()
    conc = float(ps["net_gex"].abs().max() / tot) if tot else 0.0
    score = 100 * (0.5 * prox + 0.3 * pos + 0.2 * min(conc * 3, 1.0))
    out["score"] = round(score, 1)
    out["parts"] = {"ใกล้ Max Pain": round(prox * 100, 1),
                    "GEX บวก": round(pos * 100, 1),
                    "GEX กระจุก": round(min(conc * 3, 1.0) * 100, 1)}
    return out


# ════════════════════════════════════════════════
# ประกอบทั้งหมด
# ════════════════════════════════════════════════
def _row(label, value, status, note):
    return {"label": label, "value": value, "status": status, "note": note}


def build_snapshot(sym: str = "QQQ", hv_ticker: str | None = None,
                   window_pct: float = 10.0) -> dict:
    """
    คืน dict ที่มีทุก metric + สถานะ พร้อมใช้ทั้งใน Streamlit และอีเมล
    hv_ticker: สัญลักษณ์สำหรับดึงราคาย้อนหลัง (CBOE ใช้ _SPX แต่ yfinance ใช้ ^SPX)
    """
    yf_map = {"_SPX": "^GSPC", "_NDX": "^NDX", "_RUT": "^RUT", "_VIX": "^VIX"}
    hv_ticker = hv_ticker or yf_map.get(sym, sym)

    raw, S = fetch_chain(sym)
    if raw.empty or not S:
        return {"error": f"ดึง chain ของ {sym} ไม่ได้", "symbol": sym}

    df = fa_gex.parse_cboe_for_gex(raw, S)
    if df.empty:
        return {"error": "แปลง chain ไม่ได้ (ไม่มี OI/gamma)", "symbol": sym}

    # หน้าต่างรอบ spot สำหรับ level/GEX
    win = df[np.abs(np.log(df["strike"] / S)) <= window_pct / 100]
    ps = fa_gex.compute_gex(win, S)
    lv = fa_gex.find_levels(ps, S, df_contracts=win)

    exp_list = df.groupby("expiry")["days"].first().sort_values()
    near_exp = exp_list.index[0]
    near = df[df["expiry"] == near_exp]
    near_win = win[win["expiry"] == near_exp]

    macro = get_macro()
    rv = get_realized_vol(hv_ticker)
    iv_atm = atm_iv(df, S, near_exp)
    mp = max_pain(near)
    toi = top_oi_strike(near)
    dv = dex_vex(raw, S)
    vc = vanna_charm_exposure(win, S)
    sk = skew_25d(df, raw)
    ts = term_structure(df, S)
    z0 = zero_dte_share(win, S)
    em = expected_move(df, S, near_exp)
    shock = dealer_shock(near_win if len(near_win) > 20 else win, S)
    pin = pin_score(S, mp, em.get("em"), ps, lv["net"])

    flip = lv.get("flip")
    flip_dist = (flip - S) / S * 100 if flip else None
    hv = rv.get("hv20")
    vrp = (iv_atm - hv) if (iv_atm and hv) else None

    rows = []
    rows.append(_row("Spot", f"${S:,.2f}", "NEUTRAL", "CBOE delayed ~15 นาที"))

    if flip is not None:
        st = "TREND RISK" if lv["net"] < 0 else "PIN ZONE"
        if abs(flip_dist) < 0.3:
            st = "KNIFE EDGE"
        rows.append(_row("Flip distance", f"{flip_dist:+.2f}%", st,
                         f"flip {flip:,.2f} · spot {'ต่ำกว่า' if flip_dist > 0 else 'สูงกว่า'} flip"
                         + (" · ห่างน้อยกว่า 0.3% ห้ามเปิดไม้รับ" if abs(flip_dist) < 0.3 else "")))

    lvl = " / ".join(x for x in [
        f"flip {flip:,.2f}" if flip else None,
        f"MP {mp:,.2f}" if mp else None,
        f"top OI {toi:,.2f}" if toi else None] if x)
    spread = [v for v in (flip, mp, toi) if v]
    disp = (max(spread) - min(spread)) / S * 100 if len(spread) > 1 else 0
    rows.append(_row("Level stack", lvl or "—",
                     "ALIGNED" if disp < 0.5 else "NEUTRAL",
                     "levels เกาะกลุ่ม = แม่เหล็กแรง" if disp < 0.5
                     else f"levels กระจาย {disp:.1f}% ไม่มีแม่เหล็กเดี่ยว"))

    rows.append(_row("Dealer cushion", fa_gex.fmt_usd(lv["net"]),
                     "TREND RISK" if lv["net"] < 0 else "PIN ZONE",
                     ("negative GEX — dealer ไล่ราคา ขยาย vol" if lv["net"] < 0
                      else "positive GEX — dealer สวนราคา กดให้นิ่ง")
                     + " · ⚠️ เครื่องหมายขึ้นกับ convention อย่าใช้เดี่ยว ๆ"))

    if lv["call_wall"] and lv["put_wall"]:
        dc = abs(lv["call_wall"] - S) / S * 100
        dp = abs(S - lv["put_wall"]) / S * 100
        same = lv["call_wall"] == lv["put_wall"]
        rows.append(_row("Walls (GEX)", f"{lv['call_wall']:,.0f}↑ / {lv['put_wall']:,.0f}↓",
                         "MAGNET" if same else "WATCH",
                         f"strike เดียวถือทั้ง call+put GEX สูงสุด ({dc:.2f}% จาก spot) = แม่เหล็กแรง"
                         if same else
                         f"call wall +{dc:.2f}% · put wall −{dp:.2f}% · "
                         + ("แนวต้านใกล้กว่า" if dc < dp else "แนวรับใกล้กว่า")))

    # wall อีกนิยาม — ถ่วงด้วย gamma vs ไม่ถ่วง ให้คนละคำตอบเสมอ ต้องเห็นทั้งคู่
    if lv.get("call_wall_oi") and lv.get("put_wall_oi"):
        agree = (lv["call_wall_oi"] == lv["call_wall"] and lv["put_wall_oi"] == lv["put_wall"])
        rows.append(_row(
            "Walls (OI ล้วน)",
            f"{lv['call_wall_oi']:,.0f}↑ / {lv['put_wall_oi']:,.0f}↓",
            "ตรงกับ GEX" if agree else "ต่างจาก GEX",
            f"call OI {lv['call_oi']:,} · put OI {lv['put_oi']:,} — "
            + ("ทั้งสองนิยามชี้จุดเดียวกัน = level แข็ง" if agree else
               "GEX ถ่วงด้วย gamma ซึ่งสูงสุดที่ ATM จึงดึง wall เข้าหาราคา · "
               "OI ล้วนคือภูเขาสัญญาคงค้างจริง มักไกลกว่าและนิ่งกว่า")))

    if macro.get("vix"):
        v = macro["vix"]
        st = "CALM" if v < 15 else ("NEUTRAL" if v < 20 else ("ELEVATED" if v < 28 else "STRESS"))
        note = f"VVIX {macro['vvix']:.0f}" if macro.get("vvix") else "VVIX N/A"
        rows.append(_row("Macro", f"VIX {v:.2f} ({macro['vix_chg']:+.1f}%)", st, note))

    if vrp is not None:
        st = "RICH" if vrp > 3 else ("NEUTRAL" if vrp > -1 else "CHEAP")
        rows.append(_row("Vol premium", f"{vrp:+.1f} pts", st,
                         f"IV {iv_atm:.1f}% vs HV20 {hv:.1f}%"
                         + (" — ขายพรีเมียมคุ้ม" if vrp > 3 else
                            (" — IV ใกล้ realized" if vrp > -1 else " — IV ถูกกว่าจริง ระวังขาย"))))
    if iv_atm and rv.get("hv20"):
        rows.append(_row("IV vs HV", f"{iv_atm:.1f}% / {rv['hv20']:.1f}%", "WATCH",
                         f"HV10 {rv['hv10']:.1f}% · Parkinson {rv['parkinson']:.1f}%"))

    if dv.get("vex") is not None:
        rows.append(_row("VEX", fa_gex.fmt_usd(dv["vex"]),
                         "WATCH" if dv["vex"] > 0 else "NEUTRAL",
                         "vega exposure ต่อ 1 vol point (หน่วยของเราเอง)"))
    if dv.get("dex") is not None:
        rows.append(_row("DEX", fa_gex.fmt_usd(dv["dex"]),
                         "WATCH" if abs(dv["dex"]) > 0 else "NEUTRAL",
                         "net delta ของ OI ทั้งกระดาน"))
    if vc.get("vanna") is not None:
        rows.append(_row("Vanna / Charm", f"{fa_gex.fmt_usd(vc['vanna'])} / {fa_gex.fmt_usd(vc['charm'])}",
                         "WATCH", "vanna ต่อ 1 vol pt · charm ต่อวัน — CBOE ไม่ให้ คำนวณ BS เอง"))

    if sk.get("rr25") is not None:
        r = sk["rr25"]
        st = "DEFENSIVE" if r > 3 else ("NEUTRAL" if r > -1 else "CALL BID")
        rows.append(_row("25Δ skew", f"{r:+.2f} pts", st,
                         f"put25 {sk['iv_put25']:.1f}% vs call25 {sk['iv_call25']:.1f}% "
                         f"({sk['dte']}d) · " + ("put แพงกว่า = ซื้อประกันหนัก" if r > 0 else "call แพงกว่า = ไล่ขึ้น")))

    if ts.get("slope_pct") is not None:
        rows.append(_row("Term struct", f"{ts['state']} ({ts['slope_pct']:+.1f}%)",
                         "NEUTRAL" if ts["slope_pct"] > 0 else "STRESS",
                         f"near {ts['near']:.1f}% ({ts['near_exp']}) → far {ts['far']:.1f}% ({ts['far_exp']})"
                         + ("" if ts["slope_pct"] > 0 else " — backwardation = ตลาดกลัวระยะสั้น")))

    if z0 is not None:
        rows.append(_row("0DTE share", f"{z0:.1f}% of |GEX|",
                         "HIGH" if z0 > 35 else "NEUTRAL",
                         "0DTE คุมเกมวันนี้" if z0 > 35 else "ผลกระทบ intraday จำกัด"))

    if em.get("em"):
        rows.append(_row("Expected move 1σ", f"±${em['em']:,.2f} (±{em['em_pct']:.2f}%)", "WATCH",
                         f"{near_exp} ({em['dte']}d) · ATM IV {em['iv']:.1f}%"))

    if shock.get("shares_up"):
        rows.append(_row("Dealer shock ±1%",
                         f"{shock['shares_up']/1e6:+.2f}M sh ขึ้น / {shock['shares_dn']/1e6:+.2f}M sh ลง",
                         "TREND RISK" if shock["shares_up"] < 0 else "PIN ZONE",
                         f"notional {fa_gex.fmt_usd(shock['notional_up'])} · "
                         + ("dealer ต้องขายตอนขึ้น/ซื้อตอนลง = ตรึง" if shock["shares_up"] > 0
                            else "dealer ต้องซื้อตามขึ้น/ขายตามลง = เร่ง")))

    if pin.get("score") is not None:
        s = pin["score"]
        rows.append(_row("Pin score", f"{s:.0f}/100",
                         "PIN ZONE" if s > 60 else ("NEUTRAL" if s > 35 else "TREND RISK"),
                         " · ".join(f"{k} {v:.0f}" for k, v in pin["parts"].items())))

    return {
        "symbol": sym, "spot": S, "asof": datetime.now(),
        "near_expiry": near_exp, "near_dte": int(exp_list.iloc[0]),
        "rows": rows, "levels": lv, "per_strike": ps,
        "macro": macro, "rv": rv, "atm_iv": iv_atm, "max_pain": mp,
        "expected_move": em, "skew": sk, "term": ts, "pin": pin,
        "n_contracts": len(df), "error": None,
    }
