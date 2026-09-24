"""
zero_dte.py — บันทึก option 0DTE ทุก strike ตลอดวัน แล้วทำรายงาน HTML

ทำไมต้องเก็บเอง
────────────────────────────────────────────────────────────────────────
HANDOFF §4 เขียนไว้ว่า CBOE ให้ **snapshot ปัจจุบันเท่านั้น ไม่มีย้อนหลัง**
และ §6.7 สรุปว่า "ส่วนที่น่าจะเป็น edge จริง (regime GEX + ระยะถึงกำแพง)
เทสต์ไม่ได้เลยด้วยข้อมูลที่มี" — เพราะไม่มีใครเก็บ chain ย้อนหลังไว้

ไฟล์นี้แก้ตรงนั้น: เก็บทุก snapshot ลงดิสก์ แล้วย้อนดูได้ภายหลัง
**นี่คือการสร้างชุดข้อมูล ไม่ใช่การสร้างสัญญาณ** — ตาม §9 ห้ามเอาไปตั้ง threshold
จนกว่าจะมีข้อมูลพอ ตอนนี้หน้าที่มันคือ "ทำให้คำถามเรื่อง edge ตอบได้ในอนาคต"

รูปแบบข้อมูล
────────────────────────────────────────────────────────────────────────
    zero_dte/<SYM>/<YYYY-MM-DD>.csv        วันที่ยังใช้งาน — 1 ไฟล์ = 1 วัน = 1 expiry
    zero_dte/<SYM>/<YYYY-MM>.csv.gz        คลังเดือนที่จบแล้ว (rollup_month)

**วันที่ยังเก็บอยู่เป็น CSV ธรรมดา ไม่บีบ** — pandas อ่านตรง และ Excel เปิดได้ทันที
ที่ไม่บีบเพราะไฟล์นี้ถูก "ต่อท้าย" แล้ว commit ทุกวัน: gzip เปลี่ยนทั้งก้อนทุกครั้งที่ต่อ
→ git เก็บเป็น blob ใหม่ทั้งอัน delta ไม่ได้ · CSV เป็นข้อความต่อท้าย git delta ได้จริง
(วัดแล้ว gzip แบบ multi-member ได้ 425 KB/วัน แย่กว่าที่ควรเพราะ dictionary รีเซ็ตทุก member)

**เดือนที่จบแล้วจึงบีบ** — ตอนนั้นไฟล์ไม่ถูกต่อท้ายอีก เขียนครั้งเดียวเป็น member เดียว
บีบได้ ~4 เท่า แล้ว git เก็บ blob เดียวตลอดไป · ไม่ทำแบบนี้ repo จะโตปีละ ~200 MB
รวบด้วย `python zero_dte_cli.py rollup --month 2026-09` (workflow เรียกให้อัตโนมัติวันที่ 1)

หนึ่งแถว = สัญญาหนึ่งตัว ณ เวลาหนึ่ง (long format) ไม่ใช่ตารางกว้าง
เพราะจำนวน strike เปลี่ยนได้ระหว่างวัน (CBOE เพิ่ม strike เมื่อราคาวิ่งออกนอกกรอบ)
ตารางกว้างจะต้องเติมคอลัมน์ใหม่ย้อนหลัง ซึ่งพังตอน append

volume ที่ CBOE ส่งมาเป็น **ยอดสะสมของวัน** ไม่ใช่ปริมาณในช่วง —
`session_frame()` จึงคำนวณ `vol_delta` (ส่วนต่างระหว่าง snapshot) ให้ด้วย
ถ้าเอา volume ดิบไปพลอตเป็นแท่ง จะได้กราฟขึ้นบันไดที่อ่านผิดว่า "ซื้อหนักตอนบ่าย"

ข้อจำกัดที่ต้องรู้
────────────────────────────────────────────────────────────────────────
- CBOE delayed ~15 นาที → ค่าที่บันทึกตอน 10:00 ET คือภาพราว 09:45 ET
  ห้ามใช้ข้อมูลชุดนี้วัด reaction time ระดับนาที
- ความละเอียดเท่ากับความถี่ที่เรียก `record()` (armed-alert เรียกทุก 15 นาที)
- เก็บเฉพาะ strike ที่ **มีคนเทรดวันนี้** หรืออยู่ในกรอบ ±BAND_PCT รอบ spot
  strike ไกล ๆ ที่มีแต่ OI ค้างแต่ไม่มีใครแตะ (เช่น 450 ตอน spot 740) ตัดออก
  — ไม่ได้บอกอะไรเรื่อง flow ของวันนี้ แต่กินที่ 20% ของไฟล์
  ถ้าต้องการทั้งกระดานจริง ๆ ตั้ง `ZDTE_BAND_PCT=100`
"""

from __future__ import annotations

import io
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import market_clock as mc

ROOT = os.path.dirname(os.path.abspath(__file__))
ZDTE_DIR = os.path.join(ROOT, "zero_dte")

# กรอบ strike ที่เก็บเสมอ (เผื่อ strike ที่ยังไม่มีใครเทรดแต่ใกล้ราคา)
BAND_PCT = float(os.environ.get("ZDTE_BAND_PCT", "5")) / 100

# ชื่อคอลัมน์ที่เขียนลงไฟล์ — ลำดับนี้คือ schema ห้ามสลับ (ไฟล์เก่าอ่านด้วย header อยู่แล้ว
# แต่การสลับทำให้ diff ใน git ใหญ่เกินจำเป็น)
COLUMNS = [
    "ts_utc", "sym", "expiry", "dte", "spot",
    "strike", "cp",
    "bid", "ask", "mid", "last", "volume", "open_interest", "iv",
    "delta", "gamma", "vega", "theta",
    "day_open", "day_high", "day_low", "prev_close",
]

_OCC = re.compile(r"^[A-Z]+(\d{6})([CP])(\d{8})$")

# ปัดทศนิยมก่อนเขียน — CBOE ส่ง prev_close มาเป็น 240.595001220703 ซึ่งหลักหลัง ๆ
# เป็น noise ของ float32 ไม่ใช่ข้อมูล และกินที่ ~10% ของไฟล์โดยไม่ได้ความแม่นเพิ่ม
ROUND = {"bid": 4, "ask": 4, "mid": 4, "last": 4, "spot": 4,
         "day_open": 4, "day_high": 4, "day_low": 4, "prev_close": 4,
         "iv": 5, "delta": 5, "gamma": 6, "vega": 5, "theta": 5}


# ════════════════════════════════════════════════
# แกะ OCC symbol
# ════════════════════════════════════════════════
def parse_occ(option: str) -> tuple[str | None, str | None, float | None]:
    """
    "QQQ260925C00740000" → ("2026-09-25", "C", 740.0)

    แกะเองไม่ใช้ parse_options() ของแอป เพราะตัวนั้นผูกกับ Streamlit (st.session_state)
    และไฟล์นี้ต้องรันใน CI ที่ไม่มี runtime ของ Streamlit
    """
    m = _OCC.match(str(option).strip())
    if not m:
        return None, None, None
    ymd = m.group(1)
    return f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}", m.group(2), int(m.group(3)) / 1000.0


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ════════════════════════════════════════════════
# เลือก expiry ที่นับเป็น 0DTE
# ════════════════════════════════════════════════
def pick_expiry(expiries, today: date | None = None) -> str | None:
    """
    expiry ที่ถือว่าเป็น 0DTE = ตรงกับวันนี้ (เวลา ET) · ถ้าวันนี้ไม่มี ใช้ตัวที่ใกล้ที่สุดข้างหน้า

    ห้าม clamp DTE เป็น 0 และห้ามใช้ `(expiry - now).days` — HANDOFF §6.2:
    ของเดิมทำ expiry ที่หมดอายุแล้วกลายเป็น "0DTE ปลอม" แล้ว near-term หายเงียบ
    ที่นี่จึงเทียบ `date` กับ `date` ตรง ๆ และ **ไม่เอา expiry ที่ผ่านไปแล้ว**
    """
    today = today or mc.today_et()
    valid = sorted({e for e in expiries if e})
    future = [e for e in valid if date.fromisoformat(e) >= today]
    if not future:
        return None
    return future[0]


# ════════════════════════════════════════════════
# สร้าง snapshot หนึ่งชุด
# ════════════════════════════════════════════════
def build_rows(df_raw: pd.DataFrame, spot: float, *, sym: str = "QQQ",
               ts: datetime | None = None, today: date | None = None,
               band_pct: float | None = None) -> pd.DataFrame:
    """
    raw chain จาก CBOE → DataFrame ตาม COLUMNS (เฉพาะ expiry 0DTE)

    แยกจาก record() เพื่อให้เทสต์ได้โดยไม่ต้องยิงเน็ต
    """
    band = BAND_PCT if band_pct is None else band_pct
    ts = ts or datetime.now(timezone.utc)
    today = today or mc.today_et()

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=COLUMNS)

    df = df_raw.copy()
    parsed = [parse_occ(o) for o in df.get("option", pd.Series(dtype=str))]
    df["expiry"] = [p[0] for p in parsed]
    df["cp"] = [p[1] for p in parsed]
    df["strike"] = [p[2] for p in parsed]
    df = df[df["expiry"].notna() & df["strike"].notna()]
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    exp = pick_expiry(df["expiry"].unique(), today)
    if exp is None:
        return pd.DataFrame(columns=COLUMNS)

    d = df[df["expiry"] == exp].copy()

    for c in ("bid", "ask", "iv", "open_interest", "volume", "delta", "gamma",
              "vega", "theta", "open", "high", "low", "last_trade_price",
              "prev_day_close"):
        d[c] = _num(d[c]) if c in d.columns else pd.NA

    # ── กรองก่อน แล้วค่อยคำนวณคอลัมน์ทุกตัวจากตารางที่กรองแล้ว ──
    # ห้ามคำนวณ Series ก่อนกรองแล้วเอามาประกอบ DataFrame ทีหลัง:
    # pd.DataFrame({...}) จะ **รวม index แบบ union** ไม่ใช่ตัดให้สั้นตามตัวที่สั้นสุด
    # บั๊กจริง 24 ก.ย. 2026: vol/oi คำนวณจาก d ก่อนกรอง (350 แถว) แต่ bid/ask/mid
    # จากหลังกรอง (288 แถว) → ได้ 350 แถว โดย 62 แถวเป็น NaN ทั้งบรรทัด
    # และ dedupe จับไม่ได้เลยเพราะ key เป็น NaN (NaN != NaN) → ไฟล์บวมทุกรอบที่รัน
    #
    # เกณฑ์เก็บ: มีคนเทรดวันนี้ **หรือ** อยู่ในกรอบ ±band รอบราคา
    # ไม่เอา "OI > 0" เป็นเกณฑ์เดี่ยว — 0DTE strike ไกล ๆ ที่มี OI ค้างแต่ไม่มีใครแตะ
    # ไม่ได้บอกอะไรเลย แต่กินพื้นที่ 20% ของไฟล์ (288 → 230 แถว)
    keep = ((d["volume"].fillna(0) > 0)
            | ((d["strike"] - spot).abs() <= spot * band))
    d = d[keep.fillna(False)]
    if d.empty:
        return pd.DataFrame(columns=COLUMNS)

    vol = d["volume"].fillna(0)
    oi = d["open_interest"].fillna(0)
    bid, ask = d["bid"].fillna(0), d["ask"].fillna(0)
    # mid ใช้ได้เฉพาะเมื่อมีสองฝั่งจริง — ฝั่งเดียวคือ 0.00/0.01 ที่ทำให้ mid เพี้ยน
    mid = ((bid + ask) / 2).where((bid > 0) & (ask > 0))
    mid = mid.fillna(d["last_trade_price"])

    out = pd.DataFrame({
        "ts_utc": ts.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
        "sym": sym,
        "expiry": exp,
        # DTE เทียบ date กับ date ตามบทเรียน §6.2 — ไม่ clamp
        "dte": (date.fromisoformat(exp) - today).days,
        "spot": round(float(spot), 4),
        "strike": d["strike"].astype(float),
        "cp": d["cp"],
        "bid": bid, "ask": ask, "mid": mid,
        "last": d["last_trade_price"],
        "volume": vol.astype("int64"),
        "open_interest": oi.astype("int64"),
        "iv": d["iv"],
        "delta": d["delta"], "gamma": d["gamma"],
        "vega": d["vega"], "theta": d["theta"],
        "day_open": d["open"], "day_high": d["high"], "day_low": d["low"],
        "prev_close": d["prev_day_close"],
    })

    # ด่านกันบั๊ก index-union กลับมาอีก — ถ้าจำนวนแถวไม่เท่าตารางต้นทาง
    # แปลว่ามี Series ที่ index ไม่ตรง แล้วเราจะได้แถว NaN ปนเข้าไฟล์เงียบ ๆ
    if len(out) != len(d):
        raise AssertionError(
            f"build_rows: index ไม่ตรงกัน — ได้ {len(out)} แถวจากต้นทาง {len(d)} แถว")
    bad = out["strike"].isna() | out["cp"].isna()
    if bad.any():
        raise AssertionError(f"build_rows: มี {int(bad.sum())} แถวที่ไม่มี strike/cp")

    for c, nd in ROUND.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").round(nd)

    return out.sort_values(["cp", "strike"])[COLUMNS].reset_index(drop=True)


# ════════════════════════════════════════════════
# อ่าน/เขียนไฟล์
# ════════════════════════════════════════════════
def path_for(sym: str, day: date | str) -> str:
    day = day if isinstance(day, str) else day.isoformat()
    return os.path.join(ZDTE_DIR, sym.upper(), f"{day}.csv")


_DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.csv$")
_MONTH_FILE = re.compile(r"^(\d{4}-\d{2})\.csv\.gz$")


def month_path(sym: str, month: str) -> str:
    """คลังรายเดือน — zero_dte/QQQ/2026-09.csv.gz"""
    return os.path.join(ZDTE_DIR, sym.upper(), f"{month}.csv.gz")


def _listdir(sym: str) -> list[str]:
    d = os.path.join(ZDTE_DIR, sym.upper())
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def archived_months(sym: str) -> list[str]:
    return [m.group(1) for m in (_MONTH_FILE.match(f) for f in _listdir(sym)) if m]


def available_days(sym: str) -> list[str]:
    """
    วันที่มีข้อมูลแล้ว เรียงเก่า→ใหม่ — รวมทั้งไฟล์รายวันและวันที่อยู่ในคลังรายเดือน

    อ่านคลังแค่คอลัมน์ `expiry` (usecols) จึงไม่แพงแม้คลังจะใหญ่
    """
    days = {m.group(1) for m in (_DAY_FILE.match(f) for f in _listdir(sym)) if m}
    for mon in archived_months(sym):
        for col in ("day", "expiry"):                     # คลังใหม่มี day · คลังเก่ามีแต่ expiry
            try:
                got = pd.read_csv(month_path(sym, mon), usecols=[col])
                days |= {str(v) for v in got[col].dropna().unique()}
                break
            except ValueError:
                continue                                  # ไม่มีคอลัมน์นี้ ลองตัวถัดไป
            except Exception as e:                        # noqa: BLE001
                print(f"  !! อ่านคลัง {mon} ไม่ได้ ({type(e).__name__}: {e})",
                      file=sys.stderr)
                break
    return sorted(days)


def available_symbols() -> list[str]:
    if not os.path.isdir(ZDTE_DIR):
        return []
    return sorted(s for s in os.listdir(ZDTE_DIR)
                  if os.path.isdir(os.path.join(ZDTE_DIR, s)))


_READ_ERRORS = (OSError, UnicodeDecodeError, ValueError,
                pd.errors.EmptyDataError, pd.errors.ParserError)


def _read(p: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p)
    except _READ_ERRORS as e:
        print(f"  !! อ่าน {p} ไม่ได้ ({type(e).__name__}: {e})", file=sys.stderr)
        return None


def load(sym: str, day: date | str) -> pd.DataFrame:
    """
    อ่านข้อมูลของวันนั้น · คืน DataFrame ว่างถ้าไม่มี

    **รวมไฟล์รายวันกับคลังรายเดือนเสมอ ไม่ใช่เลือกอันใดอันหนึ่ง**
    เดิมเขียนไว้ว่า "ถ้ามีไฟล์รายวันก็ใช้อันนั้น" ซึ่งผิดเงียบ ๆ:
    พอ rollup ไปแล้วแล้ววันนั้นมีข้อมูลเข้ามาอีก (cron สำรองมาช้า · กดปุ่มในแอป)
    ระบบจะสร้างไฟล์รายวันใหม่ที่มีแค่ snapshot ใหม่ แล้ว `load()` คืนแค่นั้น
    → รายงานและสรุปจะเห็นวันนั้นเหลือจุดเดียว ทั้งที่ข้อมูลเต็มยังอยู่ในคลัง
    จับได้จาก tests/test_zero_dte.py ("มีหัวข้อครบ 6 กราฟ" ล้มเพราะเหลือ snapshot เดียว)
    """
    day = str(day)
    parts = []

    p = path_for(sym, day)
    if os.path.exists(p):
        d = _read(p)
        if d is not None and not d.empty:
            parts.append(d)

    arch = month_path(sym, day[:7])
    if os.path.exists(arch):
        a = _read(arch)
        if a is not None and not a.empty:
            key = "day" if "day" in a.columns else "expiry"
            if key in a.columns:
                sl = a[a[key].astype(str) == day].drop(columns=["day"], errors="ignore")
                if not sl.empty:
                    parts.append(sl)

    if not parts:
        return pd.DataFrame(columns=COLUMNS)

    df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    if len(parts) > 1:
        sub = [c for c in ("ts_utc", "strike", "cp") if c in df.columns]
        if sub:
            df = df.drop_duplicates(subset=sub)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _append_csv(path: str, frame: pd.DataFrame, *, header: bool) -> None:
    """
    ต่อท้ายไฟล์ CSV

    lineterminator="\n" ตายตัว และเปิดไฟล์ด้วย newline="" — ถ้าปล่อยให้ Windows
    ใส่ CRLF เอง ไฟล์ที่เขียนบนเครื่องกับที่เขียนใน runner จะต่างกันทุกบรรทัด
    แล้ว git diff บวมทั้งไฟล์ทุกครั้งที่สลับเครื่องเขียน
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.StringIO()
    frame.to_csv(buf, index=False, header=header, lineterminator="\n")
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())


def write_rows(rows: pd.DataFrame, *, sym: str | None = None,
               day: date | str | None = None) -> dict:
    """
    เขียน snapshot ลงไฟล์ของวันนั้น พร้อมกัน snapshot ซ้ำ

    idempotent ด้วย key (ts_utc ปัดเป็นนาที, strike, cp) — เรียกซ้ำรอบเดิมไม่เพิ่มแถว
    จำเป็นเพราะ workflow มี cron สำรองซ้อนกัน และเรารันมือทดสอบได้ตลอด
    """
    if rows is None or rows.empty:
        return {"written": 0, "skipped": 0, "path": None, "reason": "ไม่มีแถวให้เขียน"}

    sym = sym or str(rows["sym"].iloc[0])
    # ยึดวันตาม expiry ของ 0DTE ไม่ใช่วันที่ในเครื่อง — ทำให้ไฟล์ = expiry เสมอ
    day = day or str(rows["expiry"].iloc[0])
    p = path_for(sym, day)

    existing = load(sym, day)
    fresh = not os.path.exists(p) or existing.empty

    def key(df):
        return set(zip(df["ts_utc"].astype(str).str[:16],
                       df["strike"].astype(float),
                       df["cp"].astype(str)))

    if not fresh:
        have = key(existing)
        mask = [k not in have for k in zip(rows["ts_utc"].astype(str).str[:16],
                                          rows["strike"].astype(float),
                                          rows["cp"].astype(str))]
        new = rows[mask]
    else:
        new = rows

    skipped = len(rows) - len(new)
    if new.empty:
        return {"written": 0, "skipped": skipped, "path": p,
                "reason": "snapshot นี้บันทึกไว้แล้ว"}

    _append_csv(p, new[COLUMNS], header=fresh)
    return {"written": len(new), "skipped": skipped, "path": p, "reason": ""}


# ════════════════════════════════════════════════
# บันทึกจากของจริง
# ════════════════════════════════════════════════
def rollup_month(sym: str, month: str, *, remove: bool = True) -> dict:
    """
    รวบไฟล์รายวันของเดือนนั้นเป็นคลังเดียว `YYYY-MM.csv.gz` แล้วลบไฟล์รายวัน

    ทำไมต้องมี: CSV รายวัน ~850 KB × 21 วัน = ~18 MB/เดือน ถ้าปล่อยไว้ repo public
    จะโตปีละ ~200 MB ซึ่งทำให้ clone ช้าลงเรื่อย ๆ
    คลังเขียน **ครั้งเดียวจบ** (ไม่ต่อท้ายอีก) จึงเป็น gzip member เดียว บีบได้ ~4 เท่า
    และ git เก็บเป็น blob เดียวตลอดไป ต่างจากไฟล์ที่ต่อท้ายทุกวัน

    ห้ามรวบเดือนที่ยังไม่จบ — ตัวเรียกต้องเช็คเอง (CLI เช็คให้แล้ว)
    """
    days = [m.group(1) for m in (_DAY_FILE.match(f) for f in _listdir(sym)) if m]
    days = sorted(d for d in days if d.startswith(month))
    if not days:
        return {"month": month, "days": 0, "reason": "ไม่มีไฟล์รายวันของเดือนนี้"}

    frames = []
    for d in days:
        f = _read(path_for(sym, d))
        if f is not None and not f.empty:
            # ติดป้ายว่ามาจากไฟล์วันไหน — มีแค่ในคลัง ไม่มีในไฟล์รายวัน
            # ห้ามใช้ `expiry` แทน: ปกติสองค่านี้เท่ากัน แต่ถ้าวันไหนกระดานไม่มี
            # expiry ของวันนั้น (เช่นบันทึกหลังตลาดปิด) expiry จะเป็นวันถัดไป
            # แล้วการกรองคลังด้วย expiry จะหาวันนั้นไม่เจอ = ข้อมูลหายเงียบ
            f = f.copy()
            f["day"] = d
            frames.append(f)
    if not frames:
        return {"month": month, "days": 0, "reason": "ไฟล์รายวันอ่านไม่ได้ — ไม่ลบอะไร"}

    big = pd.concat(frames, ignore_index=True)

    arch = month_path(sym, month)
    if os.path.exists(arch):                      # รวมกับคลังเดิม ไม่ทับทิ้ง
        old = _read(arch)
        if old is not None and not old.empty:
            if "day" not in old.columns:
                old = old.assign(day=old["expiry"])
            big = pd.concat([old, big], ignore_index=True)
    big = (big.drop_duplicates(subset=["day", "ts_utc", "strike", "cp"])
              .sort_values(["day", "ts_utc", "cp", "strike"]))[COLUMNS + ["day"]]

    os.makedirs(os.path.dirname(arch), exist_ok=True)
    big.to_csv(arch, index=False, compression="gzip", lineterminator="\n")

    # ลบไฟล์รายวัน **หลัง** เขียนคลังสำเร็จ และหลังอ่านกลับมายืนยันว่าครบ
    verify = _read(arch)
    ok = verify is not None and len(verify) == len(big)
    removed = 0
    if remove and ok:
        for d in days:
            try:
                os.remove(path_for(sym, d))
                removed += 1
            except OSError as e:
                print(f"  !! ลบ {d} ไม่ได้ ({e})", file=sys.stderr)

    return {"month": month, "days": len(days), "rows": len(big),
            "archive": arch, "bytes": os.path.getsize(arch),
            "removed": removed, "verified": ok, "reason": ""}


def record(sym: str = "QQQ", *, fetch=None, ts: datetime | None = None) -> dict:
    """
    ดึง chain แล้วบันทึก 0DTE ทุก strike · คืน dict สรุปผล

    ไม่โยน exception ออกมา — ตัวเรียกคือลูปใน workflow ที่ต้องไม่ตายเพราะรอบเดียวพลาด
    (บทเรียนเดียวกับ `|| true` ทุกคำสั่งใน armed-alert)
    """
    if fetch is None:
        import snapshot                                    # import ตอนใช้ ลด import cycle
        fetch = snapshot.fetch_chain
    try:
        df_raw, spot = fetch(sym)
    except Exception as e:                                  # noqa: BLE001
        return {"ok": False, "written": 0, "error": f"{type(e).__name__}: {e}"}

    rows = build_rows(df_raw, spot, sym=sym, ts=ts)
    if rows.empty:
        return {"ok": False, "written": 0,
                "error": "ไม่มี expiry 0DTE ในกระดาน (หรือกรองแล้วไม่เหลือ strike)"}

    res = write_rows(rows)
    return {"ok": True, "spot": spot, "expiry": str(rows["expiry"].iloc[0]),
            "strikes": int(rows["strike"].nunique()), "error": None, **res}


# ════════════════════════════════════════════════
# เตรียมข้อมูลให้พร้อมวิเคราะห์
# ════════════════════════════════════════════════
def session_frame(sym: str, day: date | str) -> pd.DataFrame:
    """
    ข้อมูลของวันนั้น + คอลัมน์ที่คำนวณต่อ:
        ts        datetime UTC
        et        เวลา ET อ่านง่าย (HH:MM)
        moneyness strike / spot − 1
        vol_delta ปริมาณที่เกิดในช่วงนั้น (CBOE ส่ง volume เป็นยอดสะสม)
        notional  vol_delta × mid × 100 = เงินพรีเมียมที่เปลี่ยนมือในช่วงนั้น
        oi_delta  OI ที่เปลี่ยน (OI อัปเดตวันละครั้ง — ใช้เทียบข้ามวันเท่านั้น)
    """
    df = load(sym, day)
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df[df["ts"].notna()].copy()
    if df.empty:
        return df
    df["et"] = df["ts"].dt.tz_convert(mc.ET).dt.strftime("%H:%M")
    df = df.sort_values(["strike", "cp", "ts"])

    for c in ("volume", "open_interest", "mid", "spot", "strike"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby(["strike", "cp"], sort=False)
    # ยอดสะสมอาจ "ถอย" ได้ตอน CBOE รีเซ็ตกระดานเช้า — clip ไม่ให้ติดลบ
    df["vol_delta"] = g["volume"].diff().fillna(df["volume"]).clip(lower=0)
    df["oi_delta"] = g["open_interest"].diff().fillna(0)
    df["moneyness"] = df["strike"] / df["spot"] - 1
    df["notional"] = df["vol_delta"] * df["mid"].fillna(0) * 100
    return df.reset_index(drop=True)


def day_summary(sym: str, day: date | str) -> dict:
    """สรุปหนึ่งวันให้อ่านเร็ว — ใช้ทั้งในรายงานและใน UI"""
    df = session_frame(sym, day)
    if df.empty:
        return {"day": str(day), "sym": sym, "empty": True}

    snaps = sorted(df["ts"].unique())
    calls, puts = df[df["cp"] == "C"], df[df["cp"] == "P"]
    cv, pv = float(calls["vol_delta"].sum()), float(puts["vol_delta"].sum())
    last = df[df["ts"] == snaps[-1]]

    # strike ที่มีพรีเมียมเปลี่ยนมือมากที่สุด — "เงินไปไหน" ตอบได้ตรงกว่าจำนวนสัญญา
    top = (df.groupby(["strike", "cp"])["notional"].sum()
             .sort_values(ascending=False).head(10))

    return {
        "day": str(day), "sym": sym, "empty": False,
        "expiry": str(df["expiry"].iloc[0]),
        "snapshots": len(snaps),
        "first_et": df.loc[df["ts"] == snaps[0], "et"].iloc[0],
        "last_et": df.loc[df["ts"] == snaps[-1], "et"].iloc[0],
        "strikes": int(df["strike"].nunique()),
        "spot_open": float(df.loc[df["ts"] == snaps[0], "spot"].iloc[0]),
        "spot_last": float(last["spot"].iloc[0]),
        "spot_high": float(df["spot"].max()),
        "spot_low": float(df["spot"].min()),
        "call_volume": cv, "put_volume": pv,
        "pc_ratio": (pv / cv) if cv else None,
        "notional_usd": float(df["notional"].sum()),
        "top_notional": [{"strike": float(k[0]), "cp": k[1], "notional": float(v)}
                         for k, v in top.items()],
        "rows": len(df),
    }


__all__ = [
    "COLUMNS", "ROUND", "ZDTE_DIR", "parse_occ", "pick_expiry", "build_rows",
    "path_for", "month_path", "available_days", "available_symbols",
    "archived_months", "load", "write_rows", "record", "rollup_month",
    "session_frame", "day_summary",
]
