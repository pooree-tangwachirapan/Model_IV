"""
cme_reports.py — รายงานฝั่ง futures/macro จากแหล่งสาธารณะทางการ
ไม่มีการเรียก Streamlit UI (นอกจาก cache) → สคริปต์อีเมลใช้ซ้ำได้

⚠️ เรื่องที่ต้องรู้ก่อน — ทำไมไม่ได้ดึงจาก cmegroup.com ตรง ๆ
    CME Group บล็อกการดึงข้อมูลอัตโนมัติ และระบุใน Data Terms of Use ว่า
    "Use of scripts, software, spiders, robots, agents, tools or other scraping
     mechanisms is strictly prohibited"
    (ยืนยันจากการยิงจริง 2026-08-06: ทุก endpoint ตอบ 403 พร้อมข้อความข้างต้น)
    ดังนั้นเครื่องมือที่เป็นของ CME โดยตรง — FedWatch Tool, Daily Volume & OI Report,
    Term SOFR, QuikStrike, Pace of Trading — จะไม่ถูกดึงในไฟล์นี้

    สิ่งที่ทำแทน: ใช้ "ข้อมูลต้นทางเดียวกัน" จากหน่วยงานที่เปิดให้ใช้อย่างเป็นทางการ
      COT           → CFTC public reporting API (ข้อมูลของ CFTC เอง เปิดสาธารณะ)
      อัตราดอกเบี้ย  → NY Fed markets API (SOFR/EFFR ทางการ ออกแบบมาให้เรียกโดยตรง)
      ราคา futures  → yfinance
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime

CFTC_TFF = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"   # Traders in Financial Futures
NYFED = "https://markets.newyorkfed.org/api/rates"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# ตลาดที่เกี่ยวกับ playbook QQQ ของ Pooree
COT_MARKETS = {
    "NASDAQ-100 (รวม)":   "NASDAQ-100 Consolidated",
    "Micro NASDAQ-100":   "MICRO E-MINI NASDAQ-100 INDEX",
    "E-mini S&P 500":     "E-MINI S&P 500 -",
    "VIX Futures":        "VIX FUTURES",
    "Russell E-mini":     "RUSSELL E-MINI",
}

# กลุ่มผู้เล่นใน TFF report
COT_GROUPS = [
    ("Dealer/Intermediary", "dealer_positions_long_all", "dealer_positions_short_all",
     "โต๊ะเทรดธนาคาร/market maker — มักสวนตลาด"),
    ("Asset Manager",       "asset_mgr_positions_long", "asset_mgr_positions_short",
     "กองทุน/สถาบันระยะยาว — เงินจริงที่ถือยาว"),
    ("Leveraged Money",     "lev_money_positions_long", "lev_money_positions_short",
     "hedge fund/CTA — เก็งกำไร ตัวที่ต้องจับตาสุด"),
]

FUTURES = {
    "ES=F":  ("E-mini S&P 500", ""),
    "NQ=F":  ("E-mini Nasdaq-100", ""),
    "ZQ=F":  ("30-Day Fed Funds", "ราคา = 100 − อัตราเฉลี่ยที่ตลาดคาด"),
    "SR3=F": ("SOFR 3M", "ราคา = 100 − SOFR ที่ตลาดคาด"),
    "ZN=F":  ("10-Year T-Note", ""),
}


# ════════════════════════════════════════════════
# COT — Commitment of Traders (CFTC)
# ════════════════════════════════════════════════
def fetch_cot(market_like: str, weeks: int = 52) -> pd.DataFrame:
    """ดึง COT ย้อนหลัง n สัปดาห์ของตลาดเดียว (CFTC Traders in Financial Futures)"""
    r = requests.get(CFTC_TFF, headers=UA, timeout=40, params={
        "$where": f"market_and_exchange_names like '%{market_like}%'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": weeks,
    })
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None)
    num = [c for c in df.columns if any(k in c for k in
           ("positions", "open_interest", "traders", "conc_"))]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def cot_summary(df: pd.DataFrame) -> dict:
    """
    สรุป net position ของแต่ละกลุ่ม + การเปลี่ยนแปลง + percentile ในกรอบที่ดึงมา
    percentile สำคัญกว่าเลขดิบ — บอกว่า "สุดโต่งแค่ไหนเทียบตัวเอง"
    """
    if df.empty:
        return {}
    last, prev = df.iloc[-1], (df.iloc[-2] if len(df) > 1 else None)
    out = {
        "market": last["market_and_exchange_names"],
        "date": last["date"],
        "weeks": len(df),
        "open_interest": int(last.get("open_interest_all") or 0),
        "groups": [],
    }
    if prev is not None:
        oi_prev = int(prev.get("open_interest_all") or 0)
        out["oi_change"] = out["open_interest"] - oi_prev

    for name, lcol, scol, note in COT_GROUPS:
        if lcol not in df.columns or scol not in df.columns:
            continue
        net_series = (df[lcol] - df[scol]).dropna()
        if net_series.empty:
            continue
        net = float(net_series.iloc[-1])
        chg = float(net - net_series.iloc[-2]) if len(net_series) > 1 else None
        pct = float((net_series < net).mean() * 100) if len(net_series) > 4 else None
        out["groups"].append({
            "name": name, "note": note,
            "long": int(last[lcol]), "short": int(last[scol]),
            "net": net, "change": chg, "percentile": pct,
            "history": net_series.tolist(), "dates": df["date"].tolist(),
        })
    return out


# ════════════════════════════════════════════════
# อัตราดอกเบี้ยอ้างอิง (NY Fed — ทางการ)
# ════════════════════════════════════════════════
def fetch_reference_rates() -> dict:
    """SOFR / EFFR / BGCR / TGCR — แทน 'CME Term SOFR' ที่เป็น licensed product"""
    out, errs = {}, []
    for key, path, label in [
            ("sofr", "secured/sofr", "SOFR (overnight, มีหลักประกัน)"),
            ("effr", "unsecured/effr", "EFFR (Fed Funds จริงที่เทรดกัน)"),
            ("bgcr", "secured/bgcr", "BGCR (broad general collateral)"),
            ("tgcr", "secured/tgcr", "TGCR (tri-party general collateral)")]:
        try:
            r = requests.get(f"{NYFED}/{path}/last/1.json", headers=UA, timeout=25)
            j = (r.json().get("refRates") or [{}])[0]
            if j.get("percentRate") is not None:
                out[key] = {
                    "label": label,
                    "rate": float(j["percentRate"]),
                    "date": j.get("effectiveDate"),
                    "p25": j.get("percentPercentile25"),
                    "p75": j.get("percentPercentile75"),
                    "volume": j.get("volumeInBillions"),
                }
        except Exception as e:
            errs.append(f"{key}: {type(e).__name__}")
    if errs:
        out["_errors"] = errs
    return out


# ════════════════════════════════════════════════
# Futures + market-implied policy rate
# ════════════════════════════════════════════════
def fetch_futures() -> dict:
    """ราคา + volume ของ futures หลัก (แทน CME Daily Volume Report ที่ดึงไม่ได้)"""
    out = {}
    try:
        import yfinance as yf
    except ImportError:
        return {"_error": "ไม่ได้ติดตั้ง yfinance"}
    for tk, (label, note) in FUTURES.items():
        try:
            h = yf.Ticker(tk).history(period="10d")
            if len(h) < 2:
                continue
            c = h["Close"]
            out[tk] = {
                "label": label, "note": note,
                "last": float(c.iloc[-1]),
                "chg_pct": float((c.iloc[-1] / c.iloc[-2] - 1) * 100),
                "volume": int(h["Volume"].iloc[-1]),
                "vol_avg5": float(h["Volume"].tail(5).mean()),
                "date": h.index[-1].date().isoformat(),
            }
        except Exception:
            continue
    return out


def implied_policy_rate(fut: dict, rates: dict) -> dict:
    """
    อัตราที่ตลาดคาด จาก Fed Funds futures: implied = 100 − price
    นี่คือ "วิธีคิด" เดียวกับที่ FedWatch ใช้ แต่ FedWatch ใช้ทั้ง curve รายเดือน
    เพื่อกระจายความน่าจะเป็นรายการประชุม — เรามีแค่ front month ต่อเนื่อง
    จึงบอกได้แค่ทิศทางรวม ไม่ใช่ % ต่อการประชุม
    """
    out = {"implied": None, "effr": None, "spread_bp": None, "note": None}
    zq = fut.get("ZQ=F")
    if not zq:
        return out
    out["implied"] = 100.0 - zq["last"]
    eff = rates.get("effr", {}).get("rate")
    if eff is None:
        return out
    out["effr"] = eff
    out["spread_bp"] = (out["implied"] - eff) * 100
    s = out["spread_bp"]
    if s < -6:
        out["note"] = "ตลาดคิดราคาว่าจะ **ลด** ดอกเบี้ย"
    elif s > 6:
        out["note"] = "ตลาดคิดราคาว่าจะ **ขึ้น** ดอกเบี้ย"
    else:
        out["note"] = "ตลาดคิดราคาว่า **คงที่** ในระยะใกล้"
    return out


# ════════════════════════════════════════════════
# ประกอบ
# ════════════════════════════════════════════════
def _fmt_signed(x, unit=""):
    return "—" if x is None else f"{x:+,.0f}{unit}"


def build_cme_snapshot(markets: list[str] | None = None, weeks: int = 52) -> dict:
    """คืน dict พร้อมใช้ทั้งใน Streamlit tab และอีเมล"""
    markets = markets or ["NASDAQ-100 (รวม)", "VIX Futures"]
    snap = {"asof": datetime.now(), "cot": [], "rates": {}, "futures": {},
            "policy": {}, "errors": []}

    for label in markets:
        pattern = COT_MARKETS.get(label, label)
        try:
            df = fetch_cot(pattern, weeks)
            s = cot_summary(df)
            if s:
                s["label"] = label
                snap["cot"].append(s)
        except Exception as e:
            snap["errors"].append(f"COT {label}: {type(e).__name__}: {e}")

    try:
        snap["rates"] = fetch_reference_rates()
    except Exception as e:
        snap["errors"].append(f"rates: {type(e).__name__}: {e}")
    try:
        snap["futures"] = fetch_futures()
    except Exception as e:
        snap["errors"].append(f"futures: {type(e).__name__}: {e}")

    snap["policy"] = implied_policy_rate(snap["futures"], snap["rates"])
    return snap


def cot_rows(summary: dict) -> list[dict]:
    """แปลง COT summary → แถวพร้อมสถานะ (รูปแบบเดียวกับ snapshot.py)"""
    rows = []
    wk = summary.get("weeks", 0)
    for g in summary.get("groups", []):
        net, chg, pct = g["net"], g["change"], g["percentile"]
        # สถานะ = ฝั่งที่ถืออยู่จริง (เครื่องหมายของ net) — ห้ามเอา percentile มาปนกัน
        status = "NET LONG" if net > 0 else ("NET SHORT" if net < 0 else "FLAT")

        note = g["note"]
        if chg is not None:
            direction = "เพิ่ม long / ลด short" if chg > 0 else "เพิ่ม short / ลด long"
            note += f" · สัปดาห์นี้ {_fmt_signed(chg)} ({direction})"
        # percentile บอก "อยู่ตรงไหนของช่วงตัวเอง" ไม่ได้บอกว่า long หรือ short
        if pct is not None:
            if pct >= 90:
                status += " ⚠"
                note += f" · **net สูงสุดในรอบ {wk} สัปดาห์** (percentile {pct:.0f})"
            elif pct <= 10:
                status += " ⚠"
                note += f" · **net ต่ำสุดในรอบ {wk} สัปดาห์** (percentile {pct:.0f})"
            else:
                note += f" · percentile {pct:.0f} ใน {wk} สัปดาห์"
        rows.append({
            "label": g["name"],
            "value": f"{net:+,.0f}",
            "status": status,
            "note": note,
        })
    return rows


def summary_lines(snap: dict) -> list[str]:
    """สรุปสั้น ๆ สำหรับอีเมล (ข้อความล้วน)"""
    L = []
    for s in snap.get("cot", []):
        L.append(f"COT {s['label']} ({s['date']:%Y-%m-%d}) · OI {s['open_interest']:,}")
        for g in s.get("groups", []):
            pct = f" · pctile {g['percentile']:.0f}" if g["percentile"] is not None else ""
            L.append(f"    {g['name']:<22} net {g['net']:+,.0f} "
                     f"(สัปดาห์นี้ {_fmt_signed(g['change'])}){pct}")
    r = snap.get("rates", {})
    if r:
        bits = [f"{k.upper()} {v['rate']:.2f}%" for k, v in r.items()
                if isinstance(v, dict) and "rate" in v]
        if bits:
            L.append("Reference rates: " + " · ".join(bits))
    p = snap.get("policy", {})
    if p.get("implied") is not None and p.get("effr") is not None:
        L.append(f"Fed Funds futures implied {p['implied']:.3f}% vs EFFR {p['effr']:.2f}% "
                 f"({p['spread_bp']:+.1f} bp)")
    return L
