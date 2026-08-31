"""
intraday.py — เครื่องมือฝั่ง "tape" จากแท่ง 5 นาที (ไม่มี UI)   [LP]

╔═══════════════════════════════════════════════════════════════════════╗
║ [LP] LONG-PREMIUM TAB — ไฟล์ใหม่ 31 ส.ค. 2026                          ║
║ ถ้าเจอบั๊กที่ไม่เคยเกิดก่อนมีแท็บนี้ → ไฟล์นี้ลบทิ้งได้ทั้งไฟล์             ║
║ ระบบเดิม (fade/breakout/email/workflow) ไม่ import อะไรจากที่นี่เลย       ║
╚═══════════════════════════════════════════════════════════════════════╝

ทำไมแยกไฟล์: กติกาเดิมของโปรเจกต์ — ตรรกะแยกจาก UI
ทุก function ที่คำนวณจริงรับ DataFrame เข้ามา ไม่ยิงเน็ตเอง เพื่อให้เทสต์ป้อนข้อมูลมือได้

═══ ขอบเขตและข้อจำกัด — อ่านก่อนเชื่อตัวเลข ═══

1. **Volume Profile จากแท่ง 5m เป็นค่าประมาณ ไม่ใช่ tick data**
   เรากระจาย volume ของแต่ละแท่งแบบสม่ำเสมอทั่วช่วง High–Low ซึ่งไม่ใช่ความจริง
   (ของจริงกระจุกตรงที่มีการเทรดเยอะ) → VAH/VAL จะคลาดจาก TradingView เล็กน้อย
   ใช้เป็น "โซนคร่าว ๆ" ได้ ใช้เป็นราคาเข้าเป๊ะ ๆ ไม่ได้

2. **yfinance ให้แท่ง 5m ย้อนหลังแค่ ~60 วัน** (วัดจริง 31 ส.ค. 2026: ได้ 4,680 แท่ง / 60 sessions)
   backtest ยาวกว่านี้ทำไม่ได้ด้วยแหล่งนี้

3. **EMA200 ใช้แท่งรายวัน ไม่ใช่ 5m** — 200 แท่ง 5m ≈ 2 วันทำการ ซึ่งซ้ำซ้อนกับ VWAP
   จนไม่ได้ข้อมูลใหม่ ส่วนรายวันคือ regime ระยะยาวจริง ๆ

4. **ไม่มีอะไรในไฟล์นี้ถูกใช้เป็นประตูตัดสิน** — เก็บเป็นข้อมูลเท่านั้น
   เหตุผล: 31 ส.ค. 2026 ทดสอบ VWAP+EMA200 เป็นตัวกรองทิศทางบน QQQ 60 sessions แล้ว
   ผลคือ **ไม่ช่วย** (MFE/MAE 0.68 ตอนกรอง vs 0.77 ตอนไม่กรอง) และ **ทำให้ไม้ช้าลง**
   (49 → 60 แท่ง) ซึ่งสำหรับ long premium คือต้นทุนล้วน ๆ
   → ตาม HANDOFF §9 "อย่าขยับ threshold จนกว่าจะมีไม้พอ" ใช้กับการ *ตั้ง* ครั้งแรกด้วย
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

# ── ค่าคงที่ ──
TICK = 0.25            # ความละเอียดของ volume profile (USD) — QQQ เคลื่อนหลักสตางค์ 0.25 พอ
VALUE_AREA_PCT = 0.70  # นิยามมาตรฐาน Market Profile: value area = 70% ของ volume
SPIKE_LOOKBACK = 20    # จำนวน session ย้อนหลังที่ใช้หา baseline ของ volume
MAX_BUCKETS = 4000     # กันหลุด: ถ้าช่วงราคากว้างจน bucket เกินนี้ ให้ขยาย tick อัตโนมัติ
MARKET_TZ = "America/New_York"


# ════════════════════════════════════════════════════════════════
# I/O — ส่วนเดียวที่ยิงเน็ต แยกออกมาเพื่อให้ที่เหลือเทสต์ได้
# ════════════════════════════════════════════════════════════════
def bars_5m(symbol: str = "QQQ", days: int = 60) -> pd.DataFrame:
    """
    แท่ง 5 นาที RTH → DataFrame index=เวลา ET, คอลัมน์ [Open High Low Close Volume day slot]

    [LP] โค้ดซ้ำกับ forward_test._bars_5m โดยตั้งใจ — ไม่ refactor ของเดิม
    เพราะข้อกำหนดของงานนี้คือห้ามกระทบระบบเดิมแม้แต่บรรทัดเดียว
    ยอมโค้ดซ้ำ ~10 บรรทัดแลกกับการที่ forward_test.py ไม่ถูกแตะเลย
    (ของเดิมคืน list[tuple] สำหรับเดินหา fill · ของเรานี้คืน DataFrame สำหรับคำนวณ profile
     คนละรูปแบบอยู่แล้ว การรวมกันจะทำให้ทั้งสองฝั่งอ่านยากขึ้น ไม่ได้ง่ายขึ้น)
    """
    import yfinance as yf

    df = yf.download(symbol, interval="5m", period=f"{int(days)}d",
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.get_level_values(0)   # yfinance คืน MultiIndex เมื่อขอหลายตัว

    idx = df.index
    idx = idx.tz_localize("UTC") if getattr(idx, "tz", None) is None else idx
    df.index = idx.tz_convert(MARKET_TZ)

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep].dropna()
    return _add_session_cols(df)


def bars_daily(symbol: str = "QQQ", period: str = "2y") -> pd.DataFrame:
    """แท่งรายวันสำหรับ EMA200 — ต้องยาวพอให้ EMA นิ่ง (2 ปี ≈ 500 แท่ง = 2.5 เท่าของ span)"""
    import yfinance as yf

    df = yf.download(symbol, interval="1d", period=period,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


def _add_session_cols(df: pd.DataFrame) -> pd.DataFrame:
    """เพิ่ม day (วันที่ของ session) + slot (นาทีจากเที่ยงคืน) — ใช้จัดกลุ่มทั้งไฟล์"""
    out = df.copy()
    out["day"] = out.index.normalize()
    out["slot"] = out.index.hour * 60 + out.index.minute
    return out


# ════════════════════════════════════════════════════════════════
# ตรรกะล้วน — เทสต์ได้ทั้งหมด ไม่ยิงเน็ต
# ════════════════════════════════════════════════════════════════
def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """
    VWAP anchored ที่ราคาเปิดของแต่ละ session (รีเซ็ตทุกวัน — นี่คือ VWAP ที่เทรดเดอร์ใช้จริง)

    typical price = (H+L+C)/3 ตามนิยามมาตรฐาน ไม่ใช่ close เปล่า ๆ
    เพราะแท่งที่แกว่งกว้างแล้วปิดที่ปลายแท่ง ใช้ close อย่างเดียวจะบิดตำแหน่ง VWAP
    """
    if bars is None or bars.empty:
        return pd.Series(dtype=float)
    b = bars if "day" in bars.columns else _add_session_cols(bars)
    tp = (b["High"] + b["Low"] + b["Close"]) / 3
    num = (tp * b["Volume"]).groupby(b["day"]).cumsum()
    den = b["Volume"].groupby(b["day"]).cumsum()
    return (num / den.replace(0, np.nan)).rename("vwap")


def value_area(bars: pd.DataFrame, tick: float = TICK,
               pct: float = VALUE_AREA_PCT) -> dict | None:
    """
    Volume Profile ของชุดแท่งที่ให้มา → {vah, poc, val, total_volume, tick}

    วิธี: สร้าง histogram volume-at-price แล้วขยายจาก POC ออกไปทีละฝั่ง
    เลือกฝั่งที่ volume มากกว่า จนสะสมครบ pct ของ volume ทั้งหมด (อัลกอริทึม Market Profile มาตรฐาน)

    การกระจาย volume ของแต่ละแท่ง: แบ่งเท่า ๆ กันทุก bucket ที่แท่งนั้นพาดผ่าน
    เป็นค่าประมาณ (ดูข้อจำกัดข้อ 1 หัวไฟล์) แต่ดีกว่าโยนทั้งก้อนไว้ที่ close
    ซึ่งจะให้ profile เป็นหนามแหลมปลอม ๆ
    """
    if bars is None or bars.empty:
        return None
    lo, hi = float(bars["Low"].min()), float(bars["High"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return None

    # ราคาไม่ขยับเลยทั้ง session (หยุดเทรด/ข้อมูลพัง) — profile ไม่มีความหมาย
    if hi == lo:
        return {"vah": hi, "poc": hi, "val": lo,
                "total_volume": float(bars["Volume"].sum()), "tick": tick}

    # กันช่วงราคากว้างผิดปกติจนสร้าง array มหึมา
    while (hi - lo) / tick > MAX_BUCKETS:
        tick *= 2

    edges = np.arange(np.floor(lo / tick) * tick, np.ceil(hi / tick) * tick + tick, tick)
    if len(edges) < 2:
        return None
    hist = np.zeros(len(edges) - 1)

    for h, l, v in zip(bars["High"], bars["Low"], bars["Volume"]):
        if not (np.isfinite(h) and np.isfinite(l) and np.isfinite(v)):
            continue
        i0 = max(int(np.searchsorted(edges, l, "right")) - 1, 0)
        i1 = min(int(np.searchsorted(edges, h, "right")) - 1, len(hist) - 1)
        if i1 < i0:
            i1 = i0
        hist[i0:i1 + 1] += v / (i1 - i0 + 1)

    total = hist.sum()
    if total <= 0:
        return None

    poc = int(hist.argmax())
    lo_i = hi_i = poc
    acc = hist[poc]
    target = pct * total
    while acc < target and (lo_i > 0 or hi_i < len(hist) - 1):
        up = hist[hi_i + 1] if hi_i < len(hist) - 1 else -1.0
        dn = hist[lo_i - 1] if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            acc += up
        else:
            lo_i -= 1
            acc += dn

    def mid(i):
        return float((edges[i] + edges[i + 1]) / 2)

    return {"vah": mid(hi_i), "poc": mid(poc), "val": mid(lo_i),
            "total_volume": float(total), "tick": tick}


def ema(series: pd.Series, span: int = 200) -> pd.Series:
    """EMA แบบ adjust=False = สูตรเดียวกับที่แพลตฟอร์มชาร์ตใช้ (recursive ไม่ใช่ weighted mean)"""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    return series.ewm(span=span, adjust=False).mean()


def volume_spike(bars: pd.DataFrame, lookback: int = SPIKE_LOOKBACK) -> pd.Series:
    """
    volume ของแท่ง ÷ median ของ "ช่วงเวลาเดียวกันของวัน" ย้อนหลัง `lookback` sessions

    ทำไมต้องเทียบตามเวลาของวัน ไม่ใช่ค่าเฉลี่ยรวม:
      แท่ง 09:30 มี volume มากกว่าแท่ง 12:00 เป็นสิบเท่าทุกวันอยู่แล้ว
      ถ้าใช้ค่าเฉลี่ยรวม แท่งเปิดตลาดจะเป็น "spike" ทุกวัน = ไร้ความหมาย

    **ต้อง .shift(1) เสมอ** — baseline ห้ามรวมวันที่กำลังวัดอยู่ ไม่งั้นเป็น lookahead
    วันที่ volume พุ่งจริงจะดันค่า median ของตัวเองขึ้น แล้ว spike ที่วัดได้จะต่ำกว่าความจริง
    """
    if bars is None or bars.empty:
        return pd.Series(dtype=float)
    b = bars if "slot" in bars.columns else _add_session_cols(bars)

    piv = b.pivot_table(index="day", columns="slot", values="Volume", aggfunc="sum")
    base = piv.rolling(lookback, min_periods=max(3, lookback // 2)).median().shift(1)

    out = []
    for vol, d, s in zip(b["Volume"], b["day"], b["slot"]):
        ref = base.at[d, s] if (d in base.index and s in base.columns) else np.nan
        out.append(float(vol) / ref if (ref == ref and ref is not None and ref > 0) else np.nan)
    return pd.Series(out, index=b.index, name="vspike")


def locate(price: float, va: dict | None) -> dict:
    """
    ราคาอยู่ตรงไหนเทียบ value area — คืนค่าเป็นข้อมูลดิบ ไม่ตัดสินอะไร

    penetration_pct: บวก = พ้น VA ออกไปแล้วกี่ % ของราคา · 0 = ยังอยู่ใน VA
    """
    if not va or price is None or not np.isfinite(price):
        return {"zone": None, "penetration_pct": None, "poc_gap_pct": None}
    vah, val, poc = va["vah"], va["val"], va["poc"]
    if price > vah:
        zone, pen = "above", (price - vah) / price * 100
    elif price < val:
        zone, pen = "below", (val - price) / price * 100
    else:
        zone, pen = "inside", 0.0
    return {"zone": zone, "penetration_pct": pen,
            "poc_gap_pct": (price - poc) / price * 100}


# ════════════════════════════════════════════════════════════════
# ประกอบร่าง — ตัวที่ recorder เรียกใช้
# ════════════════════════════════════════════════════════════════
def build_context(symbol: str = "QQQ", spot: float | None = None,
                  bars: pd.DataFrame | None = None,
                  daily: pd.DataFrame | None = None) -> dict:
    """
    รวมทุกอย่างเป็น dict เดียวสำหรับบันทึกลง log

    `bars` / `daily` ส่งเข้ามาได้เพื่อเทสต์ ถ้าไม่ส่งจะไปดึงเอง
    `spot` ถ้าไม่ส่ง ใช้ close ล่าสุดของแท่ง 5m (ซึ่งช้ากว่า CBOE spot เล็กน้อย)

    หมายเหตุ: `va_zone` เทียบกับ value area ของ **session ก่อนหน้า** ไม่ใช่วันนี้
    เพราะ VA ของวันที่ยังไม่จบจะเปลี่ยนทุกแท่ง ใช้อ้างอิงไม่ได้
    (วิธีเดียวกับที่ใช้ตอนเก็บสถิติ 31 ส.ค. 2026)
    """
    out: dict = {"symbol": symbol, "error": None,
                 "asof": datetime.now().isoformat(timespec="seconds")}
    try:
        b = bars_5m(symbol) if bars is None else bars
        if b is None or b.empty:
            out["error"] = "ไม่ได้แท่ง 5 นาที"
            return out
        b = b if "day" in b.columns else _add_session_cols(b)

        days = sorted(b["day"].unique())
        today = days[-1]
        prev = days[-2] if len(days) > 1 else None

        vw = session_vwap(b)
        sp = volume_spike(b)
        px = float(spot) if spot else float(b["Close"].iloc[-1])

        va_prev = value_area(b[b["day"] == prev]) if prev is not None else None
        va_today = value_area(b[b["day"] == today])

        d = bars_daily(symbol) if daily is None else daily
        e200 = None
        if d is not None and not d.empty:
            e = ema(d["Close"], 200)
            if len(e) and e.iloc[-1] == e.iloc[-1]:
                e200 = float(e.iloc[-1])

        today_mask = b["day"] == today
        sp_today = sp[today_mask].dropna().values
        loc = locate(px, va_prev)

        # ค่าล่าสุดของ "วันนี้" ที่ใช้ได้จริง — ไม่ใช่แค่แท่งสุดท้าย
        # เคสจริง 31 ส.ค. 2026: รันตอน 09:30 ET แท่งแรก volume ยังเป็น 0
        # → cumulative volume = 0 → VWAP หารศูนย์เป็น NaN แล้ว above_vwap หายไปทั้งแถว
        # เอาค่าที่ไม่ใช่ NaN ตัวท้ายสุดของ session แทน ถ้าไม่มีเลยค่อยเป็น None (ตลาดยังไม่เดินจริง)
        vw_today = vw[today_mask].dropna()
        sp_valid = sp.dropna()

        out.update({
            "spot_intraday": px,
            "bars_5m_n": int(len(b)), "sessions_n": int(len(days)),
            "today_bars": int(today_mask.sum()),   # < 5 = session เพิ่งเปิด ตัวเลขวันนี้ยังไม่มีความหมาย
            "session_date": str(pd.Timestamp(today).date()),
            "prev_session_date": str(pd.Timestamp(prev).date()) if prev is not None else None,

            "vwap": float(vw_today.iloc[-1]) if len(vw_today) else None,
            "ema200_daily": e200,
            "vspike_last": float(sp_valid.iloc[-1]) if len(sp_valid) else None,
            "vspike_max_today": float(np.max(sp_today)) if len(sp_today) else None,

            "prev_vah": va_prev["vah"] if va_prev else None,
            "prev_poc": va_prev["poc"] if va_prev else None,
            "prev_val": va_prev["val"] if va_prev else None,
            "today_vah": va_today["vah"] if va_today else None,
            "today_poc": va_today["poc"] if va_today else None,
            "today_val": va_today["val"] if va_today else None,

            "va_zone": loc["zone"],                 # above / inside / below (เทียบ VA ของเมื่อวาน)
            "va_penetration_pct": loc["penetration_pct"],
            "poc_gap_pct": loc["poc_gap_pct"],
        })
        # ── ตัวชี้ทิศทาง: บันทึกเป็น "ข้อมูล" เท่านั้น ไม่ใช่ประตู (ดูข้อจำกัดข้อ 4 หัวไฟล์) ──
        out["above_vwap"] = (px > out["vwap"]) if out["vwap"] else None
        out["above_ema200"] = (px > e200) if e200 else None
    except Exception as e:                          # noqa: BLE001 — recorder ห้ามล้มทั้งแท็บ
        out["error"] = f"{type(e).__name__}: {e}"
    return out
